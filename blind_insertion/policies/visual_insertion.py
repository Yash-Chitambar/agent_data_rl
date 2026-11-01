"""Proprio + wrist-RGB actor for Part 4 PPO (round 3: the camera question, trained directly with PPO).

Wraps the blind TCN actor-critic: the action mean is the proprio actor's mean plus a residual from a
small CNN on the latest right-wrist RGB frame (with its age). The residual head's last layer starts at
zero, so a VisualAC initialized from a blind checkpoint (v6s2) starts as exactly that policy and the
camera can only add. Critic, contact decoder, log-std and normalizers are the base model's.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from blind_insertion.policies.tcn_actor_critic import ActorCritic, mlp
from blind_insertion.policies.visual import FrameCNN


class VisualAC(nn.Module):
    def __init__(self, base: ActorCritic, res: int = 64, emb: int = 64):
        super().__init__()
        self.base, self.res = base, res
        self.cnn = FrameCNN(emb, res)
        self.head = mlp(emb + base.latent + 1, (128, 128), base.act_dim)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # shared pieces of the base model
    @property
    def obs_norm(self):
        return self.base.obs_norm

    @property
    def crit_norm(self):
        return self.base.crit_norm

    @property
    def log_std(self):
        return self.base.log_std

    @property
    def log_std_max(self):
        return self.base.log_std_max

    def contact_logits(self, z):
        return self.base.contact_logits(z)

    def value(self, hist, crit):
        return self.base.value(hist, crit)

    def act_mean(self, hist, frames, age=None):
        mu, z = self.base.act_mean(hist)
        e = self.cnn(frames.permute(0, 3, 1, 2))
        a = torch.zeros(len(e), 1, device=e.device) if age is None else (age.clamp(max=.2) * 10)[:, None]
        return mu + self.head(torch.cat([e, z.detach(), a], 1)), z

    def dist(self, hist, frames, age=None):
        mu, z = self.act_mean(hist, frames, age)
        ls = self.log_std if self.log_std_max is None else self.log_std.clamp(max=self.log_std_max)
        return torch.distributions.Normal(mu, ls.exp().expand_as(mu)), z
