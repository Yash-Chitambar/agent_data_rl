"""Check EnvCfg.latency_ms (sub-step command latency) on tiny YAM envs.

(a) latency_ms = 0 keeps the integer path; latency_ms = 20 matches latency = 1 (same seed),
    within the run-to-run spread of MuJoCo Warp (A-S8) measured on the same config twice.
(b) with latency_ms = 7 a commanded joint step starts to accelerate 7 substeps (ms) late.
(c) per-world levels from noise_rand["latency_ms"] give per-world onsets.

python -m blind_insertion.checks.latency_ms_check   (GPU, ~2 min)
"""
from __future__ import annotations

import json

import numpy as np
import torch
import warp as wp

from blind_insertion.envs.peg_env import EnvCfg, PegEnv

BASE = json.load(open("blind_insertion/configs/phase_c_yam.json"))


def make(**kw):
    cfg = dict(BASE, nworld=64, n_variants=16, seed=3, episode_s=50.0)
    cfg.update(kw)
    env = PegEnv(EnvCfg(**cfg))
    env.reset_all()
    return env


def eager_trace(env):
    """Replace the captured graph by eager substeps that record joint velocities."""
    trace = []
    qd = torch.as_tensor(env.lay.arm_dofadr, device=env.device)

    def physics(launch=True):
        with wp.ScopedDevice(env.cfg.device):
            for i in range(env.cfg.substeps):
                env._substep(i)
                wp.synchronize_device(env.cfg.device)
                trace.append(env.t_qvel[:, qd].clone())
    env._physics = physics
    return trace


def onset(env, joint=0, settle=3):
    """Substep at which joint `joint` starts to accelerate after a +1 action on it."""
    A = len(env.lay.arm_act)
    zero = torch.zeros(env.N, A, device=env.device)
    for _ in range(settle + 2):                   # settle; also flushes the latency queue
        env.step(zero)
    v_pre = env.t_qvel[:, torch.as_tensor(env.lay.arm_dofadr, device=env.device)].clone()
    trace = eager_trace(env)
    a = zero.clone()
    a[:, joint] = 1.0
    env.step(a)
    for _ in range(3):                            # delayed whole steps reach the arm later
        env.step(zero)
    v = torch.stack([v_pre, *trace], 1)[:, :, joint].cpu().numpy()   # (N, 1 + 4 * substeps)
    dv = np.diff(v, axis=1)                                         # dv[:, i]: change in substep i
    thr = 0.5 * np.abs(dv).max(1, keepdims=True)
    return np.argmax(np.abs(dv) > thr, axis=1)


def first_step_q(**kw):
    env = make(**kw)
    A = len(env.lay.arm_act)
    a = 0.5 * torch.ones(env.N, A, device=env.device)
    qs = []
    for _ in range(3):
        env.step(a)
        qs.append(env.t_qpos[:, torch.as_tensor(env.lay.arm_qadr, device=env.device)].clone())
    return torch.stack(qs, 1).cpu().numpy()


def main():
    res = {}
    # (a) equivalence
    q0a, q0b = first_step_q(), first_step_q()
    qms0 = first_step_q(latency_ms=0.0)
    ql1, ql1b = first_step_q(latency=1), first_step_q(latency=1)
    qms20 = first_step_q(latency_ms=20.0)
    res["a_spread_same_cfg_rad"] = float(np.abs(q0a - q0b).max())
    res["a_ms0_vs_old_rad"] = float(np.abs(qms0 - q0a).max())
    res["a_lat1_spread_rad"] = float(np.abs(ql1 - ql1b).max())
    res["a_ms20_vs_lat1_rad"] = float(np.abs(qms20 - ql1).max())
    # (b) fixed 7 ms
    on0 = onset(make(latency_ms=0.0 + 1e-9))      # takes the sub-step path with switch 0
    on7 = onset(make(latency_ms=7.0))
    res["b_onset_0ms_median"] = float(np.median(on0))
    res["b_onset_7ms_median"] = float(np.median(on7))
    res["b_onset_7ms_minus_0ms"] = float(np.median(on7) - np.median(on0))
    # (c) per-world levels, including whole steps (27 ms = 1 step + 7 ms)
    env = make(noise_rand={"latency_ms": [0.0, 7.0, 13.0, 27.0]})
    lv = env.w_noise["latency_ms"].cpu().numpy()
    on = onset(env)
    res["c_onset_by_level"] = {f"{x:g}": float(np.median(on[lv == x])) for x in np.unique(lv)}
    res["c_worlds_by_level"] = {f"{x:g}": int((lv == x).sum()) for x in np.unique(lv)}
    print(json.dumps(res, indent=1))
    ok = (abs(res["b_onset_7ms_minus_0ms"] - 7) <= 1
          and all(abs(v - float(k)) <= 1 for k, v in res["c_onset_by_level"].items())
          and res["a_ms0_vs_old_rad"] <= 10 * max(res["a_spread_same_cfg_rad"], 1e-6))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
