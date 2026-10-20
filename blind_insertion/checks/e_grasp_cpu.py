"""USB-C grasp-hold test on CPU MuJoCo (Phase E slip diagnosis).

Reproduces the env's USB-C reset on the YAM (drive in the pads near the top of its body,
in-hand tilt about the closing axis, plug start_gap above the housing), then holds the arm with
zero action (q_target = q_meas, as the policy's zero action) for `--T` seconds and measures how
far the drive turns and slides in the grasp. One factor changes per condition.

python -m blind_insertion.checks.e_grasp_cpu --out results/E/fail/grasp_cpu
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd
import torch

from blind_insertion.assets import builder as B
from blind_insertion.envs import kinematics as K


_CACHE = {}


def run_one(v, *, tilt_deg=0.0, gx=0.0, grip=6.0 * 2.0, pad_tc=0.002, pad_solimp=(0.99, 0.999, 0.001),
            pad_mu=1.5, cone="pyramidal", impratio=1.0, gap=0.0003, T=0.5, fix_p=(0.35, 0.0, -0.18)):
    key = id(v)
    if key not in _CACHE:
        mm, ll = B.build("yam", [v])
        B.apply_variant_cpu(mm, ll, 0)
        _CACHE[key] = (mm, ll)
    m, lay = _CACHE[key]
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if cone == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
    m.opt.impratio = impratio
    m.geom_solref[lay.pad_geoms, 0] = pad_tc
    m.geom_solimp[lay.pad_geoms, :3] = pad_solimp
    m.geom_friction[lay.pad_geoms, 0] = pad_mu
    d = mujoco.MjData(m)
    chain = K.Chain(m, lay.tcp_site, lay.arm_qadr, device="cpu")
    home = torch.tensor(lay.robot.home, dtype=torch.float32)
    R_nom = (K.rpy_mat(torch.tensor([0.0, 0.0, math.radians(lay.robot.nom_yaw_deg)]))
             @ K.rpy_mat(torch.tensor([math.pi, 0.0, 0.0]))).numpy()
    fix_p = np.asarray(fix_p, float)
    # env reset geometry (peg_env.sample_starts), fixture level, no gripper rotation
    Rpg = K.rpy_mat(torch.tensor([math.pi, math.radians(tilt_deg), 0.0])).numpy()
    gp = np.array([gx, 0.0, lay.grasp_z[0]])
    p_pt = -Rpg @ gp
    R_pf = R_nom @ Rpg
    pts = np.c_[lay.edge_points[0], np.zeros(64)] @ R_pf.T
    p_pf = np.array([0.0, 0.0, gap - pts[:, 2].min()])
    p_tf = p_pf - R_nom @ p_pt
    q, pe, _ = chain.ik(torch.tensor((fix_p + p_tf)[None], dtype=torch.float32),
                        torch.tensor(R_nom[None], dtype=torch.float32), home[None], iters=150, q_rest=home)
    assert pe.item() < 3e-4, pe.item()
    q = q[0].numpy().astype(float)
    mujoco.mj_resetData(m, d)
    d.mocap_pos[0] = fix_p
    d.mocap_quat[0] = [1, 0, 0, 0]
    d.qpos[lay.arm_qadr] = q
    mujoco.mj_kinematics(m, d)
    tp = d.site_xpos[lay.tcp_site].copy()
    tR = d.site_xmat[lay.tcp_site].reshape(3, 3).copy()
    R_peg = tR @ Rpg
    d.qpos[lay.finger_qadr] = lay.peg_half_short[0]
    pa = lay.peg_qadr
    d.qpos[pa:pa + 3] = tp - R_peg @ gp
    d.qpos[pa + 3:pa + 7] = K.mat_to_quat(torch.tensor(R_peg)).numpy()
    d.ctrl[lay.grip_act] = -grip            # YAM grip motors close with negative force
    mujoco.mj_forward(m, d)

    def rel():
        tp = d.site_xpos[lay.tcp_site]
        tR = d.site_xmat[lay.tcp_site].reshape(3, 3)
        pw = d.qpos[pa:pa + 3]
        Rw = np.zeros(9)
        mujoco.mju_quat2Mat(Rw, d.qpos[pa + 3:pa + 7])
        Rr = tR.T @ Rw.reshape(3, 3)
        return tR.T @ (pw - tp), Rr

    p0, R0 = rel()
    rot, slide, t_slip = [], [], None
    for k in range(int(T / 0.02)):
        d.ctrl[lay.arm_act] = d.qpos[lay.arm_qadr]       # zero action: q_target = q_meas
        for _ in range(20):
            mujoco.mj_step(m, d)
        p, R = rel()
        ang = math.degrees(math.acos(np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)))
        rot.append(ang)
        # displacement of the plug tip (bottom centre) relative to the TCP
        tip = p + R @ np.zeros(3)
        tip0 = p0
        slide.append(1000 * np.linalg.norm(tip - tip0))
        if t_slip is None and 1000 * np.linalg.norm((p + R @ (-gp)) - (p0 + R0 @ (-gp))) > 6.0:
            t_slip = (k + 1) * 0.02
    return dict(rot_max_deg=max(rot), rot_end_deg=rot[-1], slide_max_mm=max(slide),
                slipped=t_slip is not None, t_slip=t_slip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/E/fail/grasp_cpu")
    ap.add_argument("--n", type=int, default=24, help="drives per condition")
    ap.add_argument("--T", type=float, default=0.5)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    vs = B.sample_usbc(a.n, 1000, clearance=(3e-4, 3e-4))
    # per-drive random grasp like the env (tilt +-5 deg, along +-3 mm, grip 6 N x 1.41-2.83)
    draws = [dict(tilt_deg=rng.uniform(-5, 5), gx=rng.uniform(-0.003, 0.003),
                  grip=6.0 * math.exp(rng.uniform(math.log(1.41), math.log(2.83))),
                  pad_mu=math.exp(rng.uniform(math.log(1.41), math.log(2.0)))) for _ in vs]
    conds = {
        "env (pyramidal, impratio 1)": {},
        "in free air (gap 20 mm)": dict(gap=0.02),
        "no in-hand tilt": dict(tilt_deg=0.0),
        "elliptic cone, impratio 10": dict(cone="elliptic", impratio=10.0),
        "pyramidal, impratio 10": dict(impratio=10.0),
        "softer pads (4 ms, default solimp)": dict(pad_tc=0.004, pad_solimp=(0.9, 0.95, 0.001)),
        "grip 2x": "grip2x",
    }
    rows = []
    for name, ov in conds.items():
        for i, (v, dr) in enumerate(zip(vs, draws)):
            kw = dict(dr)
            if ov == "grip2x":
                kw["grip"] *= 2
            else:
                kw.update(ov)
            r = run_one(v, T=a.T, **kw)
            rows.append(dict(cond=name, drive=i, mass_g=1000 * v.peg_mass, body_w_mm=1000 * v.handle["width"],
                             body_t_mm=1000 * v.handle["height"], **{k: kw[k] for k in ("tilt_deg", "grip")}, **r))
        g = pd.DataFrame([x for x in rows if x["cond"] == name])
        print(f"{name:38s} slipped {g.slipped.mean():5.0%}  rot max median {g.rot_max_deg.median():5.1f} deg  "
              f"slide max median {g.slide_max_mm.median():5.2f} mm", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(str(out) + ".parquet")
    summ = df.groupby("cond").agg(slipped=("slipped", "mean"), rot_med=("rot_max_deg", "median"),
                                  slide_med=("slide_max_mm", "median"), n=("drive", "count"))
    json.dump(summ.reset_index().to_dict("records"), open(str(out) + ".json", "w"), indent=1)
    print(summ)


if __name__ == "__main__":
    main()
