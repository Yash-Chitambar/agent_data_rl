"""A-S1 CPU/GPU physics parity, A-S7 information-leak test, A-S8 determinism.

python -m blind_insertion.checks.a_s1_s7_s8 --out results/A
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.envs.peg_env import EnvCfg, PegEnv  # noqa: E402
from blind_insertion.policies.tcn_actor_critic import ActorCritic  # noqa: E402


def scripted_actions(n, T, A, seed):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.3, 1.5, (n, 1, A))
    ph = rng.uniform(0, 2 * np.pi, (n, 1, A))
    amp = rng.uniform(0.2, 1.0, (n, 1, A))
    t = np.arange(T)[None, :, None] * 0.02
    return (amp * np.sin(2 * np.pi * f * t + ph)).astype(np.float32)


def cpu_rollout(env, w, actions):
    """Replay world w's start and actions on CPU MuJoCo with the paper's control scheme."""
    m = env.mjm
    d = mujoco.MjData(m)
    lay = env.lay
    B.apply_variant_cpu(m, lay, int(env.variant[w]))
    fs, pads = lay.fixture_slots, lay.pad_geoms
    m.geom_friction[fs, 0] = env.f["geom_friction"][w, fs[0], 0].item()
    m.geom_solref[fs, 0] = env.f["geom_solref"][w, fs[0], 0].item()
    m.geom_friction[pads, 0] = env.f["geom_friction"][w, pads[0], 0].item()
    m.geom_solref[pads, 0] = env.f["geom_solref"][w, pads[0], 0].item()
    d.mocap_pos[0] = env.t_mocap_pos[w, 0].cpu().numpy()
    d.mocap_quat[0] = env.t_mocap_quat[w, 0].cpu().numpy()
    d.qpos[:] = env.t_qpos[w].cpu().numpy()
    d.ctrl[:] = env.t_ctrl[w].cpu().numpy()
    mujoco.mj_forward(m, d)
    ee = []
    for a in actions:
        q = d.qpos[lay.arm_qadr].copy()
        d.ctrl[lay.arm_act] = q + env.scale * np.clip(a, -3, 3)
        for _ in range(env.cfg.substeps):
            mujoco.mj_step(m, d)
        ee.append(d.site_xpos[lay.tcp_site].copy())
    return np.array(ee)


def a_s1(out, n=100, T=100):
    res = {}
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for row, (label, gap) in enumerate((("arm_only", 0.15), ("no_contact", 0.15), ("contact", 0.0005))):
        env = PegEnv(EnvCfg(nworld=n, n_variants=n, start_gap=gap, episode_s=100.0, far_z=1e6,
                            far_xy_margin=1e6, slip_dist=1e6, seed=1, auto_reset=False))
        if label == "arm_only":   # park the peg far away: arm dynamics alone, no contacts
            pa = env.lay.peg_qadr
            env.t_qpos[:, pa:pa + 3] = torch.tensor([0.0, 3.0, -5.0], device=env.device)
            for g in env.lay.grip_act:
                env.t_ctrl[:, int(g)] = 0.0
        acts = scripted_actions(n, T, env.n_arm, seed=row)
        cpu = {}
        for w in range(n):
            cpu[w] = cpu_rollout(env, w, acts[w])
        gpu = []
        for t in range(T):
            env.step(torch.tensor(acts[:, t], device=env.device))
            gpu.append(env.chain.fk(env._q_meas)[0].cpu().numpy())   # TCP from joint state
        gpu = np.stack(gpu, 1)
        diff = np.array([np.linalg.norm(gpu[w] - cpu[w], axis=1).max() for w in range(n)])
        d1 = np.array([np.linalg.norm(gpu[w][0] - cpu[w][0]) for w in range(n)])   # first 20 ms
        res[f"{label}_first_step_max_mm"] = float(1000 * d1.max())
        res[f"{label}_p90_mm"] = float(1000 * np.percentile(diff, 90))
        res[label] = dict(max_mm=float(1000 * diff.max()), median_mm=float(1000 * np.median(diff)),
                          frac_over_threshold=float((diff > (5e-4 if label == "contact" else 1e-4)).mean()))
        ax = axes[row, 0]
        tt = np.arange(T) * 0.02
        for w in range(3):
            for k, c in zip(range(3), ("C0", "C1", "C2")):
                ax.plot(tt, 1000 * (cpu[w][:, k] - cpu[w][0, k]), c=c, lw=1)
                ax.plot(tt, 1000 * (gpu[w][:, k] - gpu[w][0, k]), c=c, ls="--", lw=1)
        ax.set_title(f"{label}: EE xyz, CPU solid / GPU dashed (3 worlds)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mm")
        ax = axes[row, 1]
        ax.hist(1000 * diff, bins=30, color="#4477AA")
        ax.set_xlabel("max EE difference over 2 s (mm)")
        ax.set_title(f"{label}: n={n} trajectories, max {1000*diff.max():.3f} mm")
        del env
        torch.cuda.empty_cache()
    fig.suptitle("A-S1 physics parity, MuJoCo CPU vs MuJoCo Warp (same starts, same actions)")
    fig.tight_layout()
    fig.savefig(out / "A-S1_parity.png", dpi=120)
    plt.close(fig)
    ok = res["arm_only"]["max_mm"] < 0.1
    print(f"A-S1: {res} -> {'PASS' if ok else 'FAIL (see notes)'}")
    return res


def a_s7(out):
    """Privileged state never reaches the actor: change the peg/fixture state and the critic
    inputs, keep the joint state; actor observation and action must be bitwise identical."""
    env = PegEnv(EnvCfg(nworld=64, n_variants=64, seed=3))
    ac = ActorCritic(env.obs_dim, env.n_arm, env.critic_dim).to(env.device).eval()
    for _ in range(10):
        env.step(0.3 * torch.randn(64, env.n_arm, device=env.device))
    hist, crit = env.observe()
    a1, _ = ac.act_mean(hist)
    o1 = env._actor_obs_now()
    # corrupt everything privileged
    pa = env.lay.peg_qadr
    env.t_qpos[:, pa:pa + 3] += 0.05
    env.t_qvel[:, env.lay.peg_dofadr:env.lay.peg_dofadr + 6] += 1.0
    env.t_mocap_pos[:] += 0.1
    env.fix_p += 0.1
    o2 = env._actor_obs_now()
    hist2, crit2 = env.observe()
    a2, _ = ac.act_mean(hist2)
    ok = bool(torch.equal(o1, o2) and torch.equal(a1, a2) and not torch.equal(crit, crit2))
    print(f"A-S7: actor obs/action bitwise identical under privileged changes: {ok}")
    return ok


def a_s8(out, steps=1000):
    outs = []
    for _ in range(2):
        env = PegEnv(EnvCfg(nworld=256, n_variants=256, seed=7, episode_s=100.0, far_z=10.0,
                            far_xy_margin=10.0, slip_dist=10.0, auto_reset=False))
        g = torch.Generator(device=env.device)
        g.manual_seed(0)
        traj = []
        for _ in range(steps):
            env.step(0.5 * torch.randn(256, env.n_arm, generator=g, device=env.device))
            traj.append(env.t_qpos.clone())
        outs.append(torch.stack(traj))
        del env
    dd = (outs[0] - outs[1]).abs().amax(dim=(1, 2))
    first = int((dd > 0).nonzero()[0, 0]) if (dd > 0).any() else -1
    print(f"  A-S8 first differing step {first}; diff at steps 10/100/1000: "
          f"{dd[9].item():.2e} {dd[99].item():.2e} {dd[-1].item():.2e}")
    d = dd[-1].item()
    print(f"A-S8: max |qpos difference| after {steps} steps, same seed: {d:.3e} -> "
          f"{'PASS (bitwise)' if d == 0 else 'NOT bitwise'}")
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/A")
    ap.add_argument("--only", default="1,7,8")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    if "7" in args.only:
        res["A-S7_identical"] = a_s7(out)
    if "8" in args.only:
        res["A-S8_max_diff"] = a_s8(out)
    if "1" in args.only:
        res["A-S1"] = a_s1(out)
    json.dump(res, open(out / "A-S1_S7_S8.json", "w"), indent=1)
