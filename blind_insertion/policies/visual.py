"""Wrist-RGB visual policy (paper §2.3): images only, same 30-step history, compact CNN.

Each frame is embedded by a small CNN; a temporal convolution over the 30 embeddings gives
the 32-d latent, and the actor MLP maps it to joint increments. The critic is privileged
(proprioception + peg pose + contact labels), and the contact decoder supervises the latent.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from blind_insertion.policies.tcn_actor_critic import TCN, RunningNorm, mlp


class FrameCNN(nn.Module):
    def __init__(self, emb=64, res=64, ch_in=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch_in, 32, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ELU(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, ch_in, res, res)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 256), nn.ELU(), nn.Linear(256, emb),
                                  nn.LayerNorm(emb))

    def forward(self, frames_u8):
        x = frames_u8.float() / 255.0 - 0.5
        return self.head(self.net(x))


class VisualActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, critic_extra_dim, *, history=30, latent=32, emb=64,
                 res=64, n_points=64, n_classes=3, init_std=0.5):
        super().__init__()
        self.obs_dim, self.act_dim, self.history, self.emb = obs_dim, act_dim, history, emb
        self.n_points, self.n_classes = n_points, n_classes
        self.obs_norm = RunningNorm(obs_dim)
        self.crit_norm = RunningNorm(critic_extra_dim)
        self.cnn = FrameCNN(emb, res)
        self.encoder = TCN(emb, history, latent)
        self.actor = mlp(latent, (256, 256, 128), act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))
        self.decoder = mlp(latent, (128, 128), n_points * n_classes)
        self.critic = mlp(obs_dim + critic_extra_dim, (512, 256, 128), 1)
        nn.init.zeros_(self.actor[-1].bias)
        self.actor[-1].weight.data *= 0.01

    def latent_from_emb(self, emb_hist):
        return self.encoder(emb_hist)

    def act_from_emb(self, emb_hist):
        z = self.latent_from_emb(emb_hist)
        return self.actor(z), z

    def dist_from_emb(self, emb_hist):
        mu, z = self.act_from_emb(emb_hist)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), z

    def value(self, obs_now, crit):
        return self.critic(torch.cat([self.obs_norm(obs_now), self.crit_norm(crit)], 1))[:, 0]

    def contact_logits(self, z):
        return self.decoder(z).reshape(-1, self.n_points, self.n_classes)


class EmbHistory:
    """Rolling per-world history of frame embeddings for acting (no grad)."""

    def __init__(self, n, history, emb, device):
        self.h = torch.zeros(n, history, emb, device=device)
        self.count = torch.zeros(n, dtype=torch.long, device=device)   # frames in episode (<= H)

    def push(self, e, new_episode):
        self.h = torch.cat([self.h[:, 1:], e[:, None]], 1)
        if new_episode.any():
            self.h[new_episode] = e[new_episode][:, None].expand(-1, self.h.shape[1], -1)
        H = self.h.shape[1]
        self.count = torch.where(new_episode, torch.ones_like(self.count), (self.count + 1).clamp(max=H))
