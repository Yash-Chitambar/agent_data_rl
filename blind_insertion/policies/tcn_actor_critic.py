"""TCN history encoder + MLP actor, asymmetric critic, contact decoder (paper Fig. 7, App. A.1)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    """Per-feature running mean/var; frozen at evaluation (call .freeze())."""

    def __init__(self, dim, eps=1e-4, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.clip = clip
        self.frozen = False

    @torch.no_grad()
    def update(self, x):
        if self.frozen:
            return
        x = x.reshape(-1, x.shape[-1])
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)

    def freeze(self):
        self.frozen = True


def mlp(i, hidden, o, act=nn.ELU):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class TCN(nn.Module):
    """(B, H, D) -> (B, latent). Causal-free 1D convs over time, as in RMA-style adapters."""

    def __init__(self, d_in, history=30, latent=32, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, ch, 5, stride=1, padding=2), nn.ELU(),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.ELU(),
            nn.Conv1d(ch, ch, 5, stride=2, padding=2), nn.ELU(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, d_in, history)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n, 128), nn.ELU(), nn.Linear(128, latent))

    def forward(self, h):
        return self.head(self.net(h.transpose(1, 2)))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, critic_extra_dim, *, history=30, latent=32,
                 n_points=64, n_classes=3, init_std=0.5, encoder="tcn"):
        super().__init__()
        self.obs_dim, self.act_dim, self.history = obs_dim, act_dim, history
        self.n_points, self.n_classes = n_points, n_classes
        self.obs_norm = RunningNorm(obs_dim)
        self.crit_norm = RunningNorm(critic_extra_dim)
        self.encoder_kind = encoder
        if encoder == "tcn":
            self.encoder = TCN(obs_dim, history, latent)
        elif encoder == "none":     # current observation only (ablation)
            self.encoder = None
            latent = 0
        else:
            raise ValueError(encoder)
        self.latent = latent
        self.actor = mlp(latent + obs_dim, (256, 256, 128), act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), math.log(init_std)))
        self.decoder = mlp(latent, (128, 128), n_points * n_classes) if latent else None
        self.critic = mlp(obs_dim + critic_extra_dim, (512, 256, 128), 1)
        nn.init.zeros_(self.actor[-1].bias)
        self.actor[-1].weight.data *= 0.01

    def encode(self, hist):
        """hist (B, H, D) raw -> latent z (B, latent), normalized current obs (B, D)."""
        hn = self.obs_norm(hist)
        o = hn[:, -1]
        z = self.encoder(hn) if self.encoder is not None else hn.new_zeros(hn.shape[0], 0)
        return z, o

    def act_mean(self, hist, z_override=None):
        z, o = self.encode(hist)
        if z_override is not None:
            z = z_override
        return self.actor(torch.cat([z, o], 1)), z

    def dist(self, hist):
        mu, z = self.act_mean(hist)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), z

    def value(self, hist, crit):
        o = self.obs_norm(hist[:, -1])
        return self.critic(torch.cat([o, self.crit_norm(crit)], 1))[:, 0]

    def contact_logits(self, z):
        return self.decoder(z).reshape(-1, self.n_points, self.n_classes)
