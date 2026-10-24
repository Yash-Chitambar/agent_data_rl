"""Task H end to end in MuJoCo Warp: pick (scripted expert or ACT) -> handoff -> blind insertion
(the proprio TCN policy trained in PegEnv), in the full scene of `envs.pick_env`.

Handoff: when the pick policy raises `done` (ACT: done output > 0.5 for 5 steps; expert: its
done flag) or at the pick time limit, the insertion policy takes over with its history filled
with the current observation; the fingers keep squeezing with the pick's constant force. The
insertion policy sees exactly PegEnv's actor observation (joints, velocities, tcp pose and twist,
previous action) and commands q_target = q_meas + scale * a.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD MUJOCO_GL=egl .venv/bin/python -m blind_insertion.pipeline.pick_insert \
        --pick expert --insert runs/H-reader-proprio-top-s0/last.pt --episodes 256 --out results/H/pipe_expert.json
    ... --pick act --act runs/H-act-pick-v1/act_traced.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from blind_insertion.analysis.stats import wilson
from blind_insertion.assets import builder as B
from blind_insertion.envs import kinematics as K
from blind_insertion.envs.pick_env import Expert, PickCfg, PickEnv
from blind_insertion.policies.tcn_actor_critic import ActorCritic
from blind_insertion.scenes import pc_bench as S


class ActPick:
    """TorchScript ACT (scripts/act/export_act.py): (state, top, wrist) -> action chunk, with the
    dataset normalization inside. ensemble=k > 0: query every step and average the predictions
    for the current step from all chunks, weights exp(-k * age) (ACT temporal ensembling);
    otherwise execute `n_exec` actions of each chunk open loop."""

    def __init__(self, path, device, n_exec=10, ensemble=0.01):
        self.m = torch.jit.load(path, map_location=device).eval()
        self.n_exec, self.ens = n_exec, ensemble

    def reset(self, n, device):
        self.chunk, self.k, self.t = None, 0, 0
        self.buf = None
        self.done_run = torch.zeros(n, dtype=torch.long, device=device)

    def _query(self, env):
        fr = env.render()
        top = fr["top"].permute(0, 3, 1, 2).float() / 255
        wrist = fr["wrist_cam"].permute(0, 3, 1, 2).float() / 255
        with torch.no_grad():
            return self.m(env.state(), top, wrist)                  # (N, C, 8)

    def act(self, env: PickEnv):
        if self.ens > 0:
            c = self._query(env)
            N, C, D = c.shape
            if self.buf is None:
                self.buf = torch.zeros(N, C, C, D, device=c.device)    # [age, offset] predictions
                self.valid = torch.zeros(C, dtype=torch.bool, device=c.device)
            self.buf = torch.roll(self.buf, 1, dims=1)
            self.buf[:, 0] = c
            self.valid = torch.roll(self.valid, 1)
            self.valid[0] = True
            ages = torch.arange(C, device=c.device)
            preds = self.buf[:, ages, ages]                             # chunk of age i, entry i = now
            w = torch.exp(-self.ens * ages.float()) * self.valid
            a = (preds * w[None, :, None]).sum(1) / w.sum()
        else:
            if self.chunk is None or self.k >= self.n_exec:
                self.chunk, self.k = self._query(env), 0
            a = self.chunk[:, self.k]
            self.k += 1
        cmd = torch.cat([a[:, :6], torch.where(a[:, 6:7] > 0, 1.0, -1.0)], 1)
        self.done_run = torch.where(a[:, 7] > 0.5, self.done_run + 1, torch.zeros_like(self.done_run))
        return cmd, self.done_run >= 5


class ExpertPick:
    def __init__(self, env):
        self.ex = Expert(env, seed=env.cfg.seed)

    def reset(self, n, device):
        self.ex.reset()

    def act(self, env):
        return self.ex.act()


class Inserter:
    """PegEnv actor observation and action on the pick env's state."""

    def __init__(self, path, env: PickEnv, history=10, scale=0.01):
        ck = torch.load(path, map_location=env.device, weights_only=False)
        self.ac = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["critic_dim"], history=history,
                              encoder=ck.get("encoder", "tcn"), n_points=ck.get("n_points", 64)).to(env.device)
        self.ac.load_state_dict(ck["model"])
        self.ac.eval()
        self.ac.obs_norm.freeze()
        self.env, self.H, self.scale = env, history, scale
        R_nom = torch.tensor(S.R_TOP_TCP, device=env.device)
        self.R_obs = K.rpy_mat(torch.tensor(S.PORT_BASE_RPY_DEG, device=env.device) * math.pi / 180) @ R_nom
        self.prev = torch.zeros(env.N, 6, device=env.device)
        self.hist = None

    def obs(self):
        env = self.env
        q = env.t_qpos[:, env.arm_q]
        qd = env.t_qvel[:, torch.as_tensor(env.lay.arm_dofadr, device=env.device)]
        p, R, J = env.chain.fk(q, with_jac=True)
        tw = (J @ qd[..., None])[..., 0]
        eul = K.euler_xyz(self.R_obs.expand(env.N, 3, 3).transpose(1, 2) @ R)
        return torch.cat([q, qd, p, eul, tw, self.prev], 1)

    def start(self, ids):
        """Fill the history of worlds `ids` with their current observation (as PegEnv resets)."""
        o = self.obs()
        if self.hist is None:
            self.hist = o[:, None].repeat(1, self.H, 1)
        self.prev[ids] = 0
        o = self.obs()
        self.hist[ids] = o[ids, None].expand(-1, self.H, -1)

    def act(self):
        with torch.no_grad():
            a, _ = self.ac.act_mean(self.hist)
        a = a.clamp(-3, 3)
        q = self.env.t_qpos[:, self.env.arm_q]
        return a, q + self.scale * a

    def push(self, a, active):
        self.prev = torch.where(active[:, None], a, self.prev)
        o = self.obs()
        self.hist = torch.where(active[:, None, None], torch.cat([self.hist[:, 1:], o[:, None]], 1), self.hist)


def run(env: PickEnv, picker, inserter: Inserter, pick_s=10.0, insert_s=10.0, video=None):
    N, dev = env.N, env.device
    env.reset_all()
    picker.reset(N, dev)
    T_pick = int(pick_s / env.control_dt)
    T_ins = int(insert_s / env.control_dt)
    v = env.lay.variants[0]
    depth, tol = v.depth, 0.0002
    phase = torch.zeros(N, dtype=torch.long, device=dev)        # 0 pick, 1 insert, 2 finished
    t_hand = torch.full((N,), -1, device=dev)
    hand_err = torch.zeros(N, 3, device=dev)
    hand_tilt = torch.zeros(N, device=dev)
    success = torch.zeros(N, dtype=torch.bool, device=dev)
    t_ins = torch.zeros(N, dtype=torch.long, device=dev)
    frames = []
    for t in range(T_pick + T_ins):
        a_pick, done = picker.act(env) if (phase == 0).any() else (None, None)
        go = (phase == 0) & (done if done is not None else False)
        go |= (phase == 0) & (t >= T_pick - 1)
        if go.any():
            p, R = env.plug_in_port()
            hand_err[go] = p[go] * 1e3
            hand_tilt[go] = torch.rad2deg(torch.acos(R[go][:, 2, 2].clamp(-1, 1)))
            t_hand[go] = t
            inserter.start(go.nonzero()[:, 0])
            phase[go] = 1
        a_ins, q_ins = inserter.act() if inserter.hist is not None else (None, None)
        cmd = env.t_ctrl[:, env.arm_act].clone()
        grip = torch.where(phase == 0, a_pick[:, 6] if a_pick is not None else -1.0, torch.full((N,), -1.0, device=dev))
        if a_pick is not None:
            cmd[phase == 0] = a_pick[phase == 0, :6]
        if q_ins is not None:
            cmd[phase == 1] = q_ins[phase == 1]
        env.step(torch.cat([cmd, grip[:, None]], 1))
        if q_ins is not None:
            inserter.push(a_ins, phase == 1)
        t_ins += (phase == 1).long()
        p, R = env.plug_in_port()
        tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        seated = (p[:, 2] < -(depth - tol)) & (tilt <= 2.0)
        success |= (phase == 1) & seated
        phase = torch.where((phase == 1) & (seated | (t_ins >= T_ins)), torch.full_like(phase, 2), phase)
        if video is not None and t % 2 == 0:
            frames.append({k: x[:video].cpu().numpy() for k, x in env.render().items()})
        if (phase == 2).all():
            break
    p_r, _ = env.reader_pose()
    p_t, _ = env.tcp_pose()
    held = (p_r - p_t).norm(dim=1) < 0.06
    return dict(success=success, handoff=t_hand >= 0, hand_err_mm=hand_err, hand_tilt=hand_tilt,
                t_hand_s=t_hand * env.control_dt, t_insert_s=t_ins * env.control_dt, held=held), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", choices=["expert", "act"], default="expert")
    ap.add_argument("--act", default="")
    ap.add_argument("--ensemble", type=float, default=0.01, help="ACT temporal ensembling weight; 0 = chunks of 10")
    ap.add_argument("--insert", required=True)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=128)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--spec_parts", action="store_true", help="draw a USB-C part within tolerance per batch")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for b in range((a.episodes + a.nworld - 1) // a.nworld):
        v = B.sample_reader(1, a.seed + b)[0] if a.spec_parts else None
        env = PickEnv(PickCfg(nworld=a.nworld, seed=a.seed + b, render=a.pick == "act"), variant=v)
        picker = ActPick(a.act, env.device, ensemble=a.ensemble) if a.pick == "act" else ExpertPick(env)
        ins = Inserter(a.insert, env)
        r, _ = run(env, picker, ins)
        res.append({k: x.cpu().numpy() for k, x in r.items()})
        print(f"batch {b}: success {r['success'].float().mean():.3f}", flush=True)
        del env
    R = {k: np.concatenate([x[k] for x in res]) for k in res[0]}
    n = len(R["success"])
    k = int(R["success"].sum())
    lo, hi = wilson(k, n)
    he = R["hand_err_mm"]
    out = dict(pick=a.pick, insert=a.insert, act=a.act, n=n, success=k / n, ci=[lo, hi],
               handoff=float(R["handoff"].mean()), held=float(R["held"].mean()),
               hand_lat_mm=dict(median=float(np.median(np.linalg.norm(he[:, :2], axis=1))),
                                p90=float(np.quantile(np.linalg.norm(he[:, :2], axis=1), 0.9))),
               hand_gap_mm=dict(median=float(np.median(he[:, 2])), p10=float(np.quantile(he[:, 2], 0.1)),
                                p90=float(np.quantile(he[:, 2], 0.9))),
               hand_tilt_deg=dict(median=float(np.median(R["hand_tilt"])), p90=float(np.quantile(R["hand_tilt"], 0.9))),
               t_insert_s=dict(median=float(np.median(R["t_insert_s"][R["success"]])) if k else None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    np.savez(Path(a.out).with_suffix(".npz"), **R)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
