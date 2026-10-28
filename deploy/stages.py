"""The three controllers of the Task H runner, one arm (N = 1), model joint frame (deploy/conventions.py).
Each mirrors its sim counterpart in blind_insertion/pipeline/pick_insert.py:

  ActStage      ActPick: TorchScript ACT, state = joints + finger (+ last sent target - joints), temporal
                ensembling (pipeline/handoff.py), done output for the handoff gate.
  InsertStage   Inserter: the PegEnv actor observation, history filled at the handoff, relative action.
  ReleaseStage  Release: hold, open the gripper, back the tcp out along its approach axis (IK).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.pipeline import handoff as H
from blind_insertion.policies.tcn_actor_critic import ActorCritic
from blind_insertion.scenes import pc_bench as S


class ActStage:
    def __init__(self, path, device, ensemble=0.01):
        self.dev = torch.device(device)
        self.m = torch.jit.load(path, map_location=self.dev).eval()
        self.cmd_err = self.m.s_mean.shape[0] == 13
        self.ens = H.TemporalEnsemble(ensemble)
        # warm-up: the first two TorchScript calls profile and compile (354 + 298 ms; 402 ms on the
        # first real tick of pick1, 2026-10-28, while the arm held at the ready pose)
        self.reset()
        img = np.zeros((320, 320, 3), np.uint8)
        for _ in range(4):                  # through step() itself: other inputs re-profile
            self.step(np.zeros(6), 0.0, img, img)
        self.reset()

    def reset(self):
        self.ens.reset()
        self.last_sent = None

    def step(self, q_model, finger_m, top, wrist):
        """top, wrist: (320, 320, 3) uint8 RGB policy images. Returns the ensembled action (8,):
        absolute joint targets (6, model frame), gripper (> 0 open), done output."""
        q = torch.as_tensor(np.asarray(q_model, np.float32), device=self.dev)
        st = torch.cat([q, torch.tensor([finger_m], dtype=torch.float32, device=self.dev)])
        if self.cmd_err:
            last = q if self.last_sent is None else torch.as_tensor(self.last_sent, dtype=torch.float32, device=self.dev)
            st = torch.cat([st, last - q])
        img = lambda x: torch.as_tensor(x, device=self.dev).permute(2, 0, 1)[None].float() / 255  # noqa: E731
        with torch.no_grad():
            c = self.m(st[None], img(top), img(wrist))
        return self.ens(c)[0].cpu().numpy().astype(float)

    def sent(self, target_model):
        """The target actually sent this step (after the step limit), for the next cmd_err state."""
        self.last_sent = np.asarray(target_model, float).copy()


class InsertStage:
    def __init__(self, path, config, robot, device="cpu"):
        """path: checkpoint; config: its run's config.json (action scale, history, clip, frames)."""
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = json.loads(Path(config).read_text())["env"]
        self.H, self.scale, self.clip = cfg["history"], cfg["action_scale"], float(cfg.get("action_clip", 3.0))
        assert cfg.get("grasp_mode") == "top" and cfg.get("gripper") == "flexpoint", "Task H insertion policy expected"
        self.ac = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["critic_dim"], history=self.H,
                              encoder=ck.get("encoder", "tcn"), n_points=ck.get("n_points", 64)).to(device)
        self.ac.load_state_dict(ck["model"])
        self.ac.eval()
        self.ac.obs_norm.freeze()
        self.robot, self.dev = robot, device
        R_nom = torch.tensor(S.R_TOP_TCP, device=device)
        self.R_obs = K.rpy_mat(torch.tensor(cfg["fix_base_rpy_deg"], dtype=torch.float32, device=device) * math.pi / 180) @ R_nom
        self.hist = None

    def obs(self, q_model, qd):
        p, R, tw = self.robot.fk(q_model, qd)
        eul = K.euler_xyz(self.R_obs.T[None] @ R[None])[0]
        return torch.cat([self.robot.t(q_model)[0], self.robot.t(qd)[0], p, eul, tw, self.prev])

    def start(self, q_model, qd):
        """Fill the history with the current observation (prev action 0), as PegEnv resets."""
        self.prev = torch.zeros(6, device=self.dev)
        self.hist = self.obs(q_model, qd)[None, None].repeat(1, self.H, 1)

    def push(self, a, q_model, qd):
        """After a step: the action it executed and the observation that followed."""
        self.prev = torch.as_tensor(np.asarray(a, np.float32), device=self.dev)
        self.hist = torch.cat([self.hist[:, 1:], self.obs(q_model, qd)[None, None]], 1)

    def act(self):
        """Clipped action (6,); the runner sends q_motor + scale * a."""
        with torch.no_grad():
            a, _ = self.ac.act_mean(self.hist)
        return a[0].clamp(-self.clip, self.clip).cpu().numpy().astype(float)


class ReleaseStage:
    def __init__(self, robot, control_dt, open_s=0.4, lift=0.08, v=0.08):
        self.robot, self.dt = robot, control_dt
        self.n_open, self.lift, self.v = int(open_s / control_dt), lift, v
        self.T = self.n_open + int(lift / v / control_dt) + 10

    def start(self, q_model):
        self.q = self.robot.t(q_model)
        self.p0, self.R0 = self.robot.chain.fk(self.q)
        self.t = 0

    def act(self):
        """Joint target (model frame) and whether the release is over."""
        h = min(max(self.t - self.n_open, 0) * self.v * self.dt, self.lift)
        if self.t >= self.n_open:
            p = self.p0 - h * self.R0[:, :, 2]            # back out along the tcp approach axis (tcp -z)
            self.q, _, _ = self.robot.chain.ik(p, self.R0, self.q, iters=6, q_rest=self.robot.home)
        self.t += 1
        return self.q[0].cpu().numpy().astype(float), self.t >= self.T
