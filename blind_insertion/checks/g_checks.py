"""Phase G (gear meshing) scripted checks: geometry renders, drop tests, anchors, backlash.

python -m blind_insertion.checks.g_checks [--n 256] [--out results/G]

All physics checks run in MuJoCo Warp (the training engine) with the gripper's pads and
housing made collision-free, so the gear is released at a placed pose and falls under
gravity. Outputs (results/G/):
  G-geometry.png        8 gear sets seated in mesh (robot hidden), oblique + top view
  G-starts.png          8 randomized start states with the robot holding the boss
  G-drop.csv/.json/.png drop onto the shaft at tooth-phase errors 0..0.5 pitch
  G-anchor_spin.csv/.png free anchor spin-down vs the hinge model
  G-backlash.csv/.json  angular play of a meshed anchor vs the designed backlash
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

from blind_insertion.assets import gear_builder as GB  # noqa: E402
from blind_insertion.envs import kinematics as K  # noqa: E402
from blind_insertion.envs.gear_env import GearCfg, GearEnv  # noqa: E402
from blind_insertion.viz.render import WorldRenderer, save_grid  # noqa: E402

PHASES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def rz(a):
    c, s = torch.cos(a), torch.sin(a)
    R = torch.zeros(a.shape[0], 3, 3, device=a.device)
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1
    return R


def place(env: GearEnv, e: torch.Tensor, height=0.001, yaw=None, xy=None):
    """Gear coaxial with the shaft, level, bottom `height` above the shaft top; anchors at
    tooth-phase error e (N,) or (N,2). Velocities zeroed."""
    N, dev = env.N, env.device
    k = env.variant
    yaw = torch.rand(N, device=dev) * 2 * math.pi if yaw is None else yaw
    R_pf = rz(yaw)
    p_f = torch.zeros(N, 3, device=dev)
    if xy is not None:
        p_f[:, :2] = xy
    p_f[:, 2] = env.shaft_top[k] + height
    e2 = e[:, None].expand(-1, GB.N_ANCHOR) if e.dim() == 1 else e
    qa = env.anchor_angles_for_phase(k, R_pf, e2)
    Rw = env.fix_R @ R_pf
    pw = env.fix_p + (env.fix_R @ p_f[..., None])[..., 0]
    ga = env.lay.gear_qadr
    env.t_qpos[:, ga:ga + 3] = pw
    env.t_qpos[:, ga + 3:ga + 7] = K.mat_to_quat(Rw)
    env.t_qpos[:, env.a_qadr] = qa
    env.t_qvel[:] = 0
    env.t_warm[:] = 0
    env.steps[:] = 0


def gear_anchor_penetration(env: GearEnv):
    """(N,) most negative gear-anchor contact distance (m) at the current step (0 if none)."""
    c = env.d.contact
    n = int(wp.to_torch(env.d.nacon)[0])
    out = torch.zeros(env.N, device=env.device)
    if n == 0:
        return out
    geom = wp.to_torch(c.geom)[:n].long()
    dist = wp.to_torch(c.dist)[:n]
    wid = wp.to_torch(c.worldid)[:n].long()
    is_g = torch.zeros(env.mjm.ngeom, dtype=torch.bool, device=env.device)
    is_a = torch.zeros_like(is_g)
    is_g[torch.as_tensor(env.lay.gear_slots, device=env.device)] = True
    is_a[torch.as_tensor(env.lay.anchor_slots.reshape(-1), device=env.device)] = True
    sel = (is_g[geom[:, 0]] & is_a[geom[:, 1]]) | (is_a[geom[:, 0]] & is_g[geom[:, 1]])
    out.scatter_reduce_(0, wid[sel], dist[sel], reduce="amin", include_self=True)
    return out


def drop_test(env: GearEnv, out: Path, T=75):
    rows = []
    zero = torch.zeros(env.N, env.n_arm, device=env.device)
    traces = {}
    for e in PHASES:
        env.reset_all()
        sgn = torch.where(torch.rand(env.N, device=env.device) < 0.5, -1.0, 1.0)
        place(env, sgn * e)
        env._actor_obs_now()
        pen = torch.zeros(env.N, device=env.device)
        amax = torch.zeros(env.N, device=env.device)
        z_tr = []
        seated_any = torch.zeros(env.N, dtype=torch.bool, device=env.device)
        for _t in range(T):
            _, _, _, _, info = env.step(zero)
            pen = torch.minimum(pen, gear_anchor_penetration(env))
            amax = torch.maximum(amax, env.t_qvel[:, env.a_dofadr].abs().max(1).values)
            seated_any |= info["success"]
            z_tr.append(info["peg_p"][:, 2].cpu().numpy())
        p, R, _, _ = env.peg_in_fixture()
        k = env.variant
        lab, cs = env.contact_state(p, R)
        ph = env.phase_error(R).abs()
        ph = torch.where(env.a_present[k], ph, torch.zeros_like(ph)).max(1).values
        traces[e] = np.asarray(z_tr) * 1000
        for w in range(env.N):
            rows.append(dict(phase0=e, world=w, variant=int(k[w]), n_anchor=int(env.n_anchor[k[w]]),
                             z_final_mm=1000 * float(p[w, 2]), fw_mm=1000 * float(env.fw[k[w]]),
                             seated_final=bool(info["success"][w] or (cs["inside"][w].all() and p[w, 2] < env.cfg.seat_tol)),
                             seated_any=bool(seated_any[w]), meshed=bool(cs["meshed"][w].all()),
                             phase_final=float(ph[w]), max_pen_mm=-1000 * float(pen[w]),
                             tilt_deg=float(torch.rad2deg(torch.acos(R[w, 2, 2].clamp(-1, 1)))),
                             anchor_wmax=float(amax[w])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "G-drop.csv", index=False)
    summ = df.groupby("phase0").agg(seated=("seated_any", "mean"), z_med_mm=("z_final_mm", "median"),
                                    pen_p99_mm=("max_pen_mm", lambda x: float(np.quantile(x, 0.99))),
                                    pen_max_mm=("max_pen_mm", "max"),
                                    phase_final_med=("phase_final", "median"), n=("world", "count"))
    print(summ.to_string())
    json.dump(summ.reset_index().to_dict(orient="records"), open(out / "G-drop.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].bar([str(e) for e in PHASES], 100 * summ.seated.values, color="#4477AA")
    ax[0].set_xlabel("initial tooth-phase error (pitch)")
    ax[0].set_ylabel("seated within 1.5 s (%)")
    ax[0].set_ylim(0, 105)
    t = np.arange(T) * env.control_dt
    for e, c in zip((0.0, 0.1, 0.5), ("#228833", "#CCBB44", "#EE6677")):
        tr = traces[e] - 1000 * env.fw[env.variant].cpu().numpy()[None]
        ax[1].plot(t, np.median(tr, 1), color=c, label=f"phase error {e}")
        ax[1].fill_between(t, np.quantile(tr, 0.1, 1), np.quantile(tr, 0.9, 1), color=c, alpha=0.2)
    ax[1].axhline(0, color="k", lw=0.8, ls="--")
    ax[1].text(t[-1], 0.3, "anchor top", ha="right", fontsize=7)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("gear bottom - face width (mm)")
    ax[1].legend(fontsize=7)
    fig.suptitle(f"Drop onto the shaft, coaxial and level, 1 mm above the tip (n = {env.N} per bar, "
                 f"{env.nK} gear sets; median, 10-90% band)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "G-drop.png", dpi=120)
    plt.close(fig)
    return df


def anchor_spin(env: GearEnv, out: Path, w0=3.0, T=50):
    """Gear parked far above; anchors start at w0 rad/s and spin down."""
    env.reset_all()
    place(env, torch.zeros(env.N, device=env.device), height=-1.0)   # gear falls away below
    env.t_qvel[:, env.a_dofadr] = w0
    zero = torch.zeros(env.N, env.n_arm, device=env.device)
    env.cfg.far_z = 1.0
    env.cfg.far_xy = 1.0
    hist = []
    for _ in range(T):
        env.step(zero)
        hist.append(env.t_qvel[:, env.a_dofadr[0]].cpu().numpy().copy())
    hist = np.asarray(hist)
    k = env.variant.cpu().numpy()
    I = env.lay.body["anchor_inertia"][k, 0].max(-1) + 1e-5   # polar moment + armature
    damp = env.f["dof_damping"][:, env.a_dofadr[0]].cpu().numpy()
    fl = env.f["dof_frictionloss"][:, env.a_dofadr[0]].cpu().numpy()
    t = (np.arange(T) + 1) * env.control_dt
    # analytic: I w' = -b w - f  (w > 0)
    tau = I / damp
    w_an = np.maximum((w0 + fl / damp)[None] * np.exp(-t[:, None] / tau[None]) - (fl / damp)[None], 0)
    err = np.abs(hist - w_an)
    df = pd.DataFrame(dict(world=np.arange(env.N), inertia=I, damping=damp, frictionloss=fl,
                           tau_s=tau, w_end=hist[-1], w_end_model=w_an[-1],
                           max_abs_err=err.max(0)))
    df.to_csv(out / "G-anchor_spin.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for w in range(min(6, env.N)):
        ax.plot(np.r_[0, t], np.r_[w0, hist[:, w]], "o-", ms=3, color=f"C{w}")
        ax.plot(np.r_[0, t], np.r_[w0, w_an[:, w]], color=f"C{w}", ls="--", lw=0.8)
    ax.set_xlim(0, 0.16)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("anchor speed (rad/s)")
    ax.set_title(f"Free anchor spin-down: sim (solid) vs I w' = -b w - f (dashed); 6 of n = {env.N}",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "G-anchor_spin.png", dpi=120)
    plt.close(fig)
    print(f"anchor spin: median |w - model| max over time {np.median(err.max(0)):.3f} rad/s, "
          f"p95 {np.quantile(err.max(0), 0.95):.3f}; tau median {np.median(tau):.3f} s")
    return df


def backlash_test(env: GearEnv, out: Path, taus=(0.003, 0.005, 0.01, 0.02), T_settle=40, T_push=25):
    """Seat the gear in mesh, then push anchor 0 with +tau and -tau: the change of the
    tooth-phase error between the two flank contacts is the angular play. Soft contacts add
    load-dependent flank penetration, so the play is measured at several torques and
    extrapolated linearly to zero load."""
    rows = []
    for tau in taus:
        env.reset_all()
        g = torch.Generator(device=env.device)
        g.manual_seed(0)
        place(env, torch.zeros(env.N, device=env.device),
              yaw=torch.rand(env.N, generator=g, device=env.device) * 2 * math.pi)
        zero = torch.zeros(env.N, env.n_arm, device=env.device)
        for _ in range(T_settle):
            _, _, _, _, info = env.step(zero)
        seated = info["success"].clone()
        qf = wp.to_torch(env.d.qfrc_applied)
        res = {}
        pen = torch.zeros(env.N, device=env.device)
        for sgn in (1.0, -1.0):
            qf[:, env.a_dofadr[0]] = sgn * tau
            for _ in range(T_push):
                env.step(zero)
                pen = torch.minimum(pen, gear_anchor_penetration(env))
            res[sgn] = env.phase_error()[:, 0].clone()
        qf[:] = 0
        k = env.variant.cpu().numpy()
        play = (res[1.0] - res[-1.0]).abs().cpu().numpy()                # pitches
        play = np.minimum(play, 1 - play)
        for w in range(env.N):
            v = env.lay.variants[k[w]]
            d = GB._dims(v.teeth, v.module, v.pressure_deg, v.addendum, v.tooth_width)
            da = GB._dims(v.anchors[0]["teeth"], v.module, v.pressure_deg, v.addendum,
                          v.anchors[0]["tooth_width"])
            cd = v.anchors[0]["center_distance"]
            aw = math.acos((d["base_radius"] + da["base_radius"]) / cd)
            rw = d["base_radius"] / math.cos(aw)
            p_w = 2 * math.pi * rw / v.teeth                           # operating circular pitch
            rows.append(dict(tau_mNm=1000 * tau, world=w, variant=int(k[w]), seated=bool(seated[w]),
                             n_anchor=len(v.anchors), backlash_design_mm=1000 * v.anchors[0]["backlash"],
                             backlash_sim_mm=1000 * play[w] * p_w / 2, max_pen_mm=-1000 * float(pen[w])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "G-backlash.csv", index=False)
    ok = df[df.seated]
    ok = ok.assign(ratio=ok.backlash_sim_mm / ok.backlash_design_mm)
    by = ok.groupby("tau_mNm").agg(ratio_med=("ratio", "median"), ratio_p10=("ratio", lambda x: x.quantile(0.1)),
                                   ratio_p90=("ratio", lambda x: x.quantile(0.9)),
                                   pen_med_mm=("max_pen_mm", "median"), n=("world", "count"))
    # per-world linear fit play(tau) -> zero-load intercept
    icpt = []
    for w, gdf in ok.groupby("world"):
        if gdf.tau_mNm.nunique() >= 3:
            c = np.polyfit(gdf.tau_mNm, gdf.backlash_sim_mm, 1)
            icpt.append(c[1] / gdf.backlash_design_mm.iloc[0])
    icpt = np.asarray(icpt)
    r = dict(n_worlds=env.N, by_tau=by.reset_index().to_dict(orient="records"),
             zero_load_ratio_median=float(np.median(icpt)), zero_load_ratio_p10=float(np.quantile(icpt, 0.1)),
             zero_load_ratio_p90=float(np.quantile(icpt, 0.9)), n_fit=len(icpt))
    json.dump(r, open(out / "G-backlash.json", "w"), indent=1)
    print(by.to_string())
    print("zero-load play / design: median %.3f (p10 %.3f, p90 %.3f), n=%d" % (
        r["zero_load_ratio_median"], r["zero_load_ratio_p10"], r["zero_load_ratio_p90"], len(icpt)))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for tau, c in zip(sorted(ok.tau_mNm.unique()), ("#228833", "#CCBB44", "#EE6677", "#AA3377")):
        sub = ok[ok.tau_mNm == tau]
        ax[0].scatter(sub.backlash_design_mm, sub.backlash_sim_mm, s=5, color=c, label=f"{tau:g} mN m")
    ax[0].plot([0.35, 0.65], [0.35, 0.65], "k--", lw=0.8)
    ax[0].set_xlabel("designed backlash per flank (mm)")
    ax[0].set_ylabel("simulated play / 2 on the pitch circle (mm)")
    ax[0].legend(fontsize=7)
    ax[1].errorbar(by.index, by.ratio_med, yerr=[by.ratio_med - by.ratio_p10, by.ratio_p90 - by.ratio_med],
                   fmt="o-", color="#4477AA", capsize=3)
    ax[1].axhline(1, color="k", lw=0.8, ls="--")
    ax[1].plot([0], [np.median(icpt)], "s", color="#EE6677", label="zero-load fit (median)")
    ax[1].set_xlabel("torque on anchor 0 (mN m)")
    ax[1].set_ylabel("simulated / designed (median, p10-p90)")
    ax[1].legend(fontsize=7)
    fig.suptitle(f"Backlash of anchor 0 with the gear seated in mesh (n = {len(ok) // len(taus)} seated "
                 f"worlds per torque)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "G-backlash.png", dpi=120)
    plt.close(fig)
    return df


def hide_robot(m, lay):
    part = {lay.fixture_body, lay.gear_body, *lay.anchor_body.tolist()}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in part and m.geom_bodyid[g] != 0:
            m.geom_rgba[g, 3] = 0.0


def geometry_sheet(out: Path, n=8):
    env = GearEnv(GearCfg(nworld=n, n_variants=n, pads_collide=False))
    env.reset_all()
    place(env, torch.zeros(n, device=env.device))
    ga = env.lay.gear_qadr
    # seat: gear bottom on the collar
    k = env.variant
    p_f = torch.zeros(n, 3, device=env.device)
    pw = env.fix_p + (env.fix_R @ p_f[..., None])[..., 0]
    env.t_qpos[:, ga:ga + 3] = pw
    wr = WorldRenderer(env, 480, 360)
    hide_robot(wr.m, env.lay)
    ims = []
    for w in range(n):
        wr.load(w)
        fp = env.t_mocap_pos[w, 0].cpu().numpy()
        r = float(env.outer_r[k[w]])
        wr.look_at(fp + [0, 0, 0.005], distance=6 * r + 0.08, azimuth=120, elevation=-35)
        ims.append(wr.render().copy())
        wr.look_at(fp, distance=6 * r + 0.08, azimuth=90, elevation=-90)
        ims.append(wr.render().copy())
    wr.close()
    save_grid(ims, str(out / "G-geometry.png"), ncol=4)
    print("wrote", out / "G-geometry.png")


def starts_sheet(out: Path, n=8):
    env = GearEnv(GearCfg(nworld=n, n_variants=n))
    wr = WorldRenderer(env, 480, 360)
    ims = []
    for w in range(n):
        wr.load(w)
        fp = env.t_mocap_pos[w, 0].cpu().numpy()
        wr.look_at(fp + [0, 0, 0.03], distance=0.30, azimuth=200, elevation=-12)
        ims.append(wr.render(contacts=True).copy())
    wr.close()
    save_grid(ims, str(out / "G-starts.png"), ncol=4)
    print("wrote", out / "G-starts.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default="results/G")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    if only is None or "render" in only:
        geometry_sheet(out)
        starts_sheet(out)
    if only is None or only & {"drop", "spin", "backlash"}:
        env = GearEnv(GearCfg(nworld=args.n, n_variants=args.n, pads_collide=False, auto_reset=False,
                              far_xy=1.0, far_z=1.0, slip_dist=10.0))
        if only is None or "drop" in only:
            drop_test(env, out)
        if only is None or "spin" in only:
            anchor_spin(env, out)
        if only is None or "backlash" in only:
            backlash_test(env, out)
    if only is None or "scripted" in only:
        scripted_test(out)



class ScriptedGear:
    """Privileged scripted insertion (feasibility check, not a baseline the paper reports):
    align the gear's axis with the shaft, turn it to zero tooth-phase error with anchor 0,
    descend; once on the anchors' tops, turn to fix the anchor with the largest phase error
    while pressing lightly (the meshed anchor turns along, the other slips under the gear)."""

    def __init__(self, env: GearEnv, gain=5.0, press=1.0, ki=0.5, lift=0.0015, tol=0.0002):
        self.env, self.gain, self.press, self.ki, self.lift, self.tol = env, gain, press, ki, lift, tol
        self.integ = torch.zeros(env.N, 3, device=env.device)
        self.last_step = torch.zeros(env.N, dtype=torch.long, device=env.device)
        self.kp = torch.as_tensor(env.lay.robot.kp, device=env.device, dtype=torch.float32)
        self.mass = torch.as_tensor(env.lay.body["gear_mass"], device=env.device, dtype=torch.float32)

    def __call__(self, hist):
        env = self.env
        k = env.variant
        p, R, _, _ = env.peg_in_fixture()
        e = env.phase_error(R)
        pres = env.a_present[k]
        on_top = p[:, 2] < env.fw[k] + 0.0015
        big = torch.where(pres, e.abs(), torch.full_like(e, -1)).argmax(1)
        e_sel = torch.where(on_top, e.gather(1, big[:, None])[:, 0], e[:, 0])
        pg = 2 * math.pi / env.teeth[k]
        dyaw = (e_sel * pg).clamp(-0.05, 0.05)
        xy_err = p[:, :2].norm(dim=1)
        new_ep = env.steps < self.last_step
        self.last_step = env.steps.clone()
        self.integ[new_ep | (env.steps == 0)] = 0
        go = (xy_err < self.tol) | (p[:, 2] < env.shaft_top[k] - 0.001)    # aligned or already in the bore
        z_des = torch.where(go, torch.full_like(xy_err, -0.003), env.shaft_top[k] + self.lift)
        p_des = torch.cat([torch.zeros_like(p[:, :2]), z_des[:, None]], 1)
        self.integ[:, :2] = (self.integ[:, :2] + (p_des - p)[:, :2] * env.control_dt).clamp(-0.002, 0.002)
        p_des = p_des + self.ki * self.integ
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]) + dyaw
        R_des_f = rz(yaw)
        q = env._q_meas
        tp, tR, J = env.chain.fk(q, with_jac=True)
        Rf, pf = env.fix_R, env.fix_p
        R_g_w = Rf @ R
        p_g_w = pf + (Rf @ p[..., None])[..., 0]
        R_rel = tR.transpose(1, 2) @ R_g_w
        p_rel = (tR.transpose(1, 2) @ (p_g_w - tp)[..., None])[..., 0]
        R_g_des_w = Rf @ R_des_f
        p_g_des_w = pf + (Rf @ p_des[..., None])[..., 0]
        R_tcp_des = R_g_des_w @ R_rel.transpose(1, 2)
        p_tcp_des = p_g_des_w - (R_tcp_des @ p_rel[..., None])[..., 0]
        err = torch.cat([(p_tcp_des - tp).clamp(-0.01, 0.01), K.rot_log(R_tcp_des @ tR.transpose(1, 2))], 1)
        JJt = J @ J.transpose(1, 2) + 1e-4 * torch.eye(6, device=q.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, err[..., None]))[..., 0]
        F = torch.zeros(env.N, 6, device=q.device)
        F[:, 2] = 9.81 * self.mass[k]
        F[:, :3] -= self.press * Rf[:, :, 2] * go.float()[:, None]
        ff = (J.transpose(1, 2) @ F[..., None])[..., 0] / self.kp
        return ((self.gain * dq + ff) / env.scale).clamp(-3, 3)


def scripted_test(out: Path, n=1024):
    from blind_insertion.eval_gear import evaluate_gear  # noqa: PLC0415
    env = GearEnv(GearCfg(nworld=n, n_variants=min(n, 1024), variant_seed=1000))
    df, _ = evaluate_gear(env, ScriptedGear(env))
    df.to_parquet(out / "G-scripted_privileged.parquet")
    r = dict(n=len(df), success=float(df.success.mean()),
             success_1anchor=float(df[df.n_anchor == 1].success.mean()),
             success_2anchor=float(df[df.n_anchor == 2].success.mean()),
             median_time_s=float(df[df.success > 0].time.median()),
             failures=df.failure.value_counts().to_dict())
    json.dump(r, open(out / "G-scripted_privileged.json", "w"), indent=1)
    print("scripted privileged:", r)


if __name__ == "__main__":
    main()
