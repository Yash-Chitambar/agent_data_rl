"""USB-C slip: replay exact GPU (MuJoCo Warp) reset states on CPU MuJoCo.

1. Build the stage-1 USB-C env on the GPU, reset all worlds, hold zero action for --steps control
   steps, and record which worlds slip (grasp-point slip > slip_dist) and how many pad-drive
   contacts each world has after the first step.
2. Copy the exact reset state (qpos, qvel, ctrl, mocap) and the per-world contact parameters
   (geom friction, solref) of slipping and non-slipping worlds into the same compiled model on CPU
   MuJoCo, run the same zero-action hold, and record slip and pad-drive contact counts there.

python -m blind_insertion.checks.e_slip_gpu_cpu --out results/E/fail/slip_gpu_cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch
import warp as wp

from blind_insertion.assets import builder as B
from blind_insertion.envs.peg_env import EnvCfg, PegEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E/fail/slip_gpu_cpu")
    ap.add_argument("--nworld", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--n_check", type=int, default=40, help="slipping and non-slipping worlds to replay")
    ap.add_argument("--env_json", default="")
    a = ap.parse_args()
    cfg = json.load(open("blind_insertion/configs/phase_e_usbc_up.json"))
    cfg.update(json.loads(a.env_json) if a.env_json else {})
    vs = B.sample_usbc(320, 1000, clearance=(3e-4, 3e-4))
    cfg.update(nworld=a.nworld, n_variants=320, variants=vs, peg_offset=6.54e-4, auto_reset=False,
               far_xy_margin=10.0, far_z=10.0, slip_dist=10.0, seed=7)
    env = PegEnv(EnvCfg(**cfg))
    lay, N = env.lay, env.N
    env.reset_all()
    wp.synchronize()
    snap = dict(qpos=env.t_qpos.clone(), qvel=env.t_qvel.clone(), ctrl=env.t_ctrl.clone(),
                mpos=env.t_mocap_pos.clone(), mquat=env.t_mocap_quat.clone(),
                fric=env.f["geom_friction"].clone(), solref=env.f["geom_solref"].clone(),
                variant=env.variant.clone())
    pads = set(int(g) for g in lay.pad_geoms)
    pegs = set(int(g) for g in lay.peg_slots)

    # GPU zero-action hold, measuring drive pose relative to the tcp
    pa = lay.peg_qadr
    gpu_ncon = None
    ovbits = np.zeros(N, dtype=np.int64)
    for k in range(a.steps):
        env.step(torch.zeros(N, env.n_arm, device=env.device))
        wp.synchronize()
        ovbits |= env.t_overflow.cpu().numpy().astype(np.int64)
        if k == 0:   # pad-drive contacts per world after the first control step
            nac = int(env.d.nacon.numpy()[0])
            geom = env.d.contact.geom.numpy()[:nac]
            wid = env.d.contact.worldid.numpy()[:nac]
            is_pd = np.array([(g0 in pads and g1 in pegs) or (g1 in pads and g0 in pegs) for g0, g1 in geom])
            gpu_ncon = np.bincount(wid[is_pd], minlength=N)
    # drive displacement relative to the tcp, before vs after the hold (kinematics on CPU)
    mjm = env.mjm
    dk = mujoco.MjData(mjm)

    def tcp_rel(qpos):
        dk.qpos[:] = qpos
        mujoco.mj_kinematics(mjm, dk)
        tp = dk.site_xpos[lay.tcp_site].copy()
        tR = dk.site_xmat[lay.tcp_site].reshape(3, 3).copy()
        return tR.T @ (qpos[pa:pa + 3] - tp)

    q_start = snap["qpos"].cpu().numpy()
    q_end = env.t_qpos.cpu().numpy()
    gpu_disp = np.array([1000 * np.linalg.norm(tcp_rel(q_end[w]) - tcp_rel(q_start[w])) for w in range(N)])
    gpu_slip = gpu_disp > 6.0
    print(f"GPU: {gpu_slip.mean():.1%} of {N} worlds move > 6 mm in the grasp in {a.steps} zero-action steps; "
          f"pad-drive contacts after step 1: median {np.median(gpu_ncon):.0f} "
          f"(slipping {np.median(gpu_ncon[gpu_slip]) if gpu_slip.any() else float('nan'):.0f}, "
          f"holding {np.median(gpu_ncon[~gpu_slip]):.0f})", flush=True)

    names = {0: "NEFC", 2: "BROADPHASE", 3: "NARROWPHASE", 4: "CCD", 8: "EPA_HORIZON", 9: "ITERATIONS",
             10: "LS_ITERATIONS"}
    ov_rows = []
    for bit, nm in names.items():
        on = (ovbits >> bit) & 1 > 0
        ov_rows.append(dict(bit=nm, worlds=int(on.sum()), slip_if_on=float(gpu_slip[on].mean()) if on.any() else None,
                            slip_if_off=float(gpu_slip[~on].mean()) if (~on).any() else None))
        print(f"  overflow {nm:13s} in {on.mean():6.1%} of worlds; slip rate with/without: "
              f"{gpu_slip[on].mean() if on.any() else float('nan'):.1%} / {gpu_slip[~on].mean():.1%}", flush=True)
    pd.DataFrame(dict(world=np.arange(N), variant=snap["variant"].cpu().numpy(), gpu_slip=gpu_slip,
                      gpu_disp_mm=gpu_disp, gpu_pad_contacts=gpu_ncon, ovbits=ovbits)).to_parquet(
        str(Path(a.out)) + "_gpu_all.parquet")

    # CPU replay of the same states
    rng = np.random.default_rng(0)
    ws = np.r_[rng.permutation(np.nonzero(gpu_slip)[0])[:a.n_check], rng.permutation(np.nonzero(~gpu_slip)[0])[:a.n_check]]
    m = env.mjm
    rows = []
    for w in ws:
        k = int(snap["variant"][w])
        B.apply_variant_cpu(m, lay, k)
        m.geom_friction[:] = snap["fric"][w].cpu().numpy()
        m.geom_solref[:] = snap["solref"][w].cpu().numpy()
        d = mujoco.MjData(m)
        d.qpos[:] = q_start[w]
        d.qvel[:] = snap["qvel"][w].cpu().numpy()
        d.ctrl[:] = snap["ctrl"][w].cpu().numpy()
        d.mocap_pos[:] = snap["mpos"][w].cpu().numpy()
        d.mocap_quat[:] = snap["mquat"][w].cpu().numpy()
        mujoco.mj_forward(m, d)
        r0 = tcp_rel(d.qpos.copy())
        cpu_ncon = None
        for s in range(a.steps):
            d.ctrl[lay.arm_act] = d.qpos[lay.arm_qadr]
            for _ in range(env.cfg.substeps):
                mujoco.mj_step(m, d)
            if s == 0:
                cpu_ncon = sum(1 for i in range(d.ncon)
                               if (d.contact[i].geom1 in pads and d.contact[i].geom2 in pegs)
                               or (d.contact[i].geom2 in pads and d.contact[i].geom1 in pegs))
        cpu_disp = 1000 * np.linalg.norm(tcp_rel(d.qpos.copy()) - r0)
        rows.append(dict(world=int(w), variant=k, gpu_slip=bool(gpu_slip[w]), gpu_disp_mm=float(gpu_disp[w]),
                         gpu_pad_contacts=int(gpu_ncon[w]), cpu_slip=bool(cpu_disp > 6.0), cpu_disp_mm=float(cpu_disp),
                         cpu_pad_contacts=int(cpu_ncon)))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out) + ".parquet")
    summ = df.groupby("gpu_slip").agg(n=("world", "count"), cpu_slip=("cpu_slip", "mean"),
                                      gpu_disp_med=("gpu_disp_mm", "median"), cpu_disp_med=("cpu_disp_mm", "median"),
                                      gpu_contacts=("gpu_pad_contacts", "median"),
                                      cpu_contacts=("cpu_pad_contacts", "median"))
    print(summ.to_string())
    json.dump(dict(gpu_slip_rate=float(gpu_slip.mean()), steps=a.steps, nworld=N,
                   gpu_contacts_all_median=float(np.median(gpu_ncon)),
                   by_gpu_slip=summ.reset_index().to_dict("records")), open(str(out) + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
