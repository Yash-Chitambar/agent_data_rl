"""Phase A signal checks on CPU MuJoCo (ground truth): A-S3, A-S4, A-S5, A-S6.

python -m blind_insertion.checks.signal_checks --robot flexiv --out results/A
Each check writes a CSV and a PNG and prints its pass/fail number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.checks.cpu_scene import CpuScene  # noqa: E402

NOISE_LEVELS = (1e-4, 5e-4)


def a_s3_step_response(robot, out, variant=None, gains=None):
    s = CpuScene(robot, variant, gains=gains)
    rows, fig, axes = [], *plt.subplots(2, 4, figsize=(14, 5.5))
    axes = axes.ravel()
    n = len(s.lay.arm_qadr)
    for j in range(n):
        q0 = s.place(xy=(0, 0), height=0.06)      # held 6 cm above the plate, free space
        for _ in range(500):                        # settle under the peg's weight
            s.step_ctrl(q0, 1)
        qs0 = s.q
        tgt = q0.copy()
        tgt[j] += 0.05
        tr = []
        for k in range(2000):
            s.step_ctrl(tgt, 1)
            tr.append(s.q[j])
        tr = np.array(tr) - qs0[j]
        t = np.arange(len(tr)) * 1e-3
        final = tr[-200:].mean()
        try:
            t10 = t[np.argmax(tr >= 0.1 * final)]
            t90 = t[np.argmax(tr >= 0.9 * final)]
        except ValueError:
            t10 = t90 = np.nan
        over = (tr.max() - final) / max(abs(final), 1e-9)
        # count oscillation cycles: zero crossings of (tr - final) after first reaching final
        e = tr - final
        cross = np.sum(np.diff(np.sign(e[np.argmax(tr >= final):])) != 0) / 2
        rows.append(dict(joint=j + 1, rise_s=t90 - t10, overshoot=over, ss_err_rad=0.05 - final,
                         cycles=cross, final=final))
        ax = axes[j]
        ax.plot(t, np.full_like(t, 0.05), "k--", lw=1, label="target step")
        ax.plot(t, tr, lw=1.5, label="measured")
        ax.set_title(f"J{j+1}: rise {1000*(t90-t10):.0f} ms, OS {100*over:.0f}%")
        ax.set_xlabel("time (s)")
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].set_ylabel("q - q0 (rad)")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"A-S3 step response, {robot} (0.05 rad step, free space, peg held)")
    fig.tight_layout()
    fig.savefig(out / f"A-S3_step_{robot}.png", dpi=120)
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"A-S3_step_{robot}.csv", index=False)
    ok = bool((df.cycles <= 3).all())
    print(f"A-S3 {robot}: max cycles {df.cycles.max():.1f}, overshoot max {df.overshoot.max():.2f} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def a_s4_gravity(robot, out, variant=None, gains=None, n_poses=20, seed=0):
    rng = np.random.default_rng(seed)
    s = CpuScene(robot, variant, gains=gains)
    errs = []
    for i in range(n_poses):
        if robot == "flexiv":
            s.fix_p = np.array([rng.uniform(0.5, 0.75), rng.uniform(-0.25, 0.15), rng.uniform(-0.1, 0.25)])
        else:
            s.fix_p = np.array([rng.uniform(0.25, 0.45), rng.uniform(-0.2, 0.2), rng.uniform(-0.3, -0.1)])
        s.d.mocap_pos[0] = s.fix_p
        try:
            q0 = s.place(xy=(0, 0), height=0.08)
        except AssertionError:
            continue
        for _ in range(2000):
            s.step_ctrl(q0, 1)
        errs.append(q0 - s.q)
    errs = np.array(errs)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    n = errs.shape[1]
    ax.bar(np.arange(1, n + 1), np.abs(errs).mean(0), yerr=np.abs(errs).std(0), capsize=3, color="#4477AA")
    ax.set_xlabel("joint")
    ax.set_ylabel("|steady-state error| (rad)")
    ax.set_title(f"A-S4 tracking under the peg's weight, {robot} (n={len(errs)} poses, 2 s hold)")
    fig.tight_layout()
    fig.savefig(out / f"A-S4_gravity_{robot}.png", dpi=120)
    plt.close(fig)
    pd.DataFrame(errs, columns=[f"j{j+1}" for j in range(n)]).to_csv(out / f"A-S4_gravity_{robot}.csv", index=False)
    cv = np.abs(errs).std(0) / (np.abs(errs).mean(0) + 1e-12)
    print(f"A-S4 {robot}: mean |err| per joint {np.abs(errs).mean(0).round(4)}; recorded")
    return True


def _rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (0.5 if ang < 1e-6 else ang / (2 * np.sin(ang)))


def _in_poly(xy, poly):
    a, b = poly, np.roll(poly, -1, 0)
    e = (b - a)[None]
    r = xy[:, None, :] - a[None]
    return ((e[..., 0] * r[..., 1] - e[..., 1] * r[..., 0]) >= -1e-9).all(1)


def _settle(s, tgt, steps=1500):
    for _ in range(steps):
        s.step_ctrl(tgt, 1)


def a_s5_contact_signal(robot, out, variant=None, gains=None):
    """Static form: fixed joint targets pressing the peg into the plate / a hole wall."""
    s = CpuScene(robot, variant, gains=gains)
    v = s.v
    rows = []
    # vertical press on the top face, beside the hole. The arm sags under the peg's weight
    # (kp=80), so sweep the commanded height from well above contact down into the plate;
    # the deflection baseline is the free-space offset just before contact.
    away = (v.hole_w / 2 + 0.6 * (v.plate_size / 2 - v.hole_w / 2), 0.0)
    # free-space sag of the TCP under the peg's weight with targets at the placed pose
    q0 = s.place(xy=away, height=0.06)
    tp0, _ = s.tcp()
    _settle(s, q0)
    tp1, _ = s.tcp()
    sag = float((s.fix_R.T @ (tp0 - tp1))[2])
    print(f"  free-space sag under the peg's weight: {1000*sag:.1f} mm")
    fine = np.geomspace(1e-5, 0.006, 22)
    for kind, axis, offsets in (
            ("vertical", 2, np.r_[sag + np.geomspace(0.01, 1e-5, 20), sag - fine]),
            ("lateral", 0, np.r_[-np.linspace(0.6 * v.clearance, 0.1 * v.clearance, 4), fine * 3])):
        base = None
        for off in offsets:
            if kind == "vertical":
                q0 = s.place(xy=away, height=0.0)
                d_ref = np.array([0, 0, off])
            else:
                q0 = s.place(xy=(0, 0), height=-v.depth / 2)
                d_ref = np.array([v.clearance + off, 0, sag])   # hang at half depth
            J = s.jac()
            dx = np.zeros(6)
            dx[:3] = s.fix_R @ d_ref
            tgt = q0 + np.linalg.pinv(J) @ dx
            _settle(s, tgt)
            F = s.fix_R.T @ s.peg_plate_force()
            raw = tgt - s.q
            if abs(F[axis]) < 0.01:
                base = raw            # last free-space sample
                continue
            if base is None:
                continue
            defl = raw - base
            rows.append(dict(kind=kind, offset=float(off), F=float(abs(F[axis])),
                             **{f"d{j+1}": float(defl[j]) for j in range(len(defl))},
                             ee=float(np.linalg.norm(J[:3] @ defl))))
    df = pd.DataFrame(rows)
    df.to_csv(out / f"A-S5_contact_signal_{robot}.csv", index=False)
    n = len(s.lay.arm_qadr)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    res = {"sag_mm": 1000 * sag}
    for ax, kind in zip(axes[:2], ("vertical", "lateral")):
        d = df[(df.kind == kind) & (df.F > 1e-3)].sort_values("F")
        for j in range(n):
            ax.loglog(d.F, d[f"d{j+1}"].abs() + 1e-9, marker=".", label=f"J{j+1}")
        for sig, ls in zip(NOISE_LEVELS, (":", "--")):
            ax.axhline(3 * sig, color="k", ls=ls, lw=1, label=f"3 sigma_q = {3*sig:.0e}")
        mx = d[[f"d{j+1}" for j in range(n)]].abs().max(1).values
        for sig in NOISE_LEVELS:
            above = d.F.values[mx > 3 * sig]
            res[f"Fmin_{kind}_{sig:.0e}"] = float(above.min()) if len(above) else float("nan")
        # monotonic up to solver noise (5% of the value or 2e-6 rad)
        mono = bool(np.all(np.diff(mx) > -np.maximum(0.05 * mx[1:], 2e-6)))
        res[f"monotonic_{kind}"] = mono
        ax.set_xlabel("contact force (N)")
        ax.set_ylabel("|q_cmd - q| (rad)")
        ax.set_title(f"{kind} press: F_min(1e-4)={res[f'Fmin_{kind}_1e-04']:.2g} N")
        ax.legend(fontsize=6, ncol=2)
    ax = axes[2]
    for kind, c in (("vertical", "#4477AA"), ("lateral", "#EE6677")):
        d = df[(df.kind == kind) & (df.F > 1e-3)].sort_values("F")
        ax.loglog(d.F, 1000 * d.ee, marker="o", color=c, label=kind)
    ax.set_xlabel("contact force (N)")
    ax.set_ylabel("EE deflection (mm)")
    ax.set_title("EE-space deflection vs force")
    ax.legend()
    fig.suptitle(f"A-S5 contact signal, {robot} (static joint targets, n={len(df)} presses)")
    fig.tight_layout()
    fig.savefig(out / f"A-S5_contact_signal_{robot}.png", dpi=120)
    plt.close(fig)
    json.dump(res, open(out / f"A-S5_contact_signal_{robot}.json", "w"), indent=1)
    ok = res["monotonic_vertical"] and res["monotonic_lateral"]
    print(f"A-S5 {robot}: {res} -> {'PASS' if ok else 'FAIL'}")
    return ok, res


def a_s6_dropin(robot, out, variant=None, gains=None, speed=0.01, press=0.5, duration=5.0, noise=1e-4, seed=0,
                scene_kw=None, tag=""):
    """Slide the aligned peg across the top face into the hole with the paper's control scheme
    (q_target = q_meas + scale*a at 50 Hz); a scripted Cartesian action pushes down and sideways."""
    s = CpuScene(robot, variant, gains=gains, **(scene_kw or {}))
    v = s.v
    rng = np.random.default_rng(seed)
    robot_tag = robot + tag
    import torch  # noqa: PLC0415
    x0 = -(v.clearance + 0.004)        # bottom face overlaps the rim by ~4 mm; slides into the hole
    s.place(xy=(x0, 0.0), height=0.0003)
    rec = []
    t = 0.0
    capture_t = None
    tp0, tR0 = s.tcp()
    kp = s.lay.robot.kp
    g = 9.81 * v.peg_mass
    q_prev = torch.tensor(s.q[None], dtype=torch.float32)
    for k in range(int(duration / 0.02)):
        # absolute TCP path: straight line along +x (fixture frame) at `speed`
        p_des = tp0 + s.fix_R @ np.array([min(speed * (t + 0.02), -3 * x0), 0, 0])
        qd_, pe, _ = s.chain.ik(torch.tensor(p_des[None], dtype=torch.float32),
                                torch.tensor(tR0[None], dtype=torch.float32), q_prev, iters=20)
        q_prev = qd_
        J = s.jac()
        w = np.zeros(6)
        w[:3] = s.fix_R @ np.array([0, 0, g - press])     # cancel weight, press with `press` N
        q_t = qd_[0].numpy().astype(float) + (J.T @ w) / kp
        s.step_ctrl(q_t, 20)
        t += 0.02
        p, R = s.peg_fixture()
        pts = np.c_[s.edge, np.zeros(64)] @ R.T + p
        # capture = the peg drops into the opening (bottom centre > 1 mm below the top face)
        inside = p[2] < -0.001 and _in_poly(pts[:, :2], s.lay.hole_polys[0]).all()
        if inside and capture_t is None:
            capture_t = t
        tp, _ = s.tcp()
        rec.append(dict(t=t, peg_x=p[0], peg_z=p[2], ee_z=(s.fix_R.T @ (tp - s.fix_p))[2],
                        **{f"q{j+1}": s.q[j] for j in range(len(s.q))},
                        **{f"qd{j+1}": s.qd[j] for j in range(len(s.q))}))
    df = pd.DataFrame(rec)
    df.to_csv(out / f"A-S6_dropin_{robot_tag}.csv", index=False)
    n = len(s.lay.arm_qadr)
    ez = df.ee_z.values
    vz = np.gradient(ez, 0.02)
    # observed with encoder noise: EE z from noisy q is ~ J * noise; estimate its std
    noise_ee = noise * np.linalg.norm(s.jac()[2])
    pnr = pnr_qd = float("nan")
    if capture_t is not None:
        pre = (df.t < capture_t - 0.1) & (df.t > capture_t - 0.6)
        win = (df.t > capture_t - 0.1) & (df.t < capture_t + 0.3)
        sig = np.abs(vz[win] - np.median(vz[pre])).max()
        nz = np.sqrt(np.std(vz[pre]) ** 2 + (np.sqrt(2) * noise_ee / 0.02) ** 2)
        pnr = float(sig / nz)
        # joint-velocity signature: change of each qdot around capture vs its jitter + the
        # velocity noise a 50 Hz finite difference of sigma_q encoder noise would add
        best = 0.0
        for j in range(n):
            qd_ = df[f"qd{j+1}"].values
            jit = np.std(np.diff(qd_[pre.values]))
            sig_j = np.abs(qd_[win.values] - qd_[pre.values][-1]).max()
            best = max(best, sig_j / np.sqrt(jit ** 2 + (np.sqrt(2) * noise / 0.02) ** 2))
        pnr_qd = float(best)
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for j in range(n):
        axes[0].plot(df.t, df[f"q{j+1}"] - df[f"q{j+1}"].iloc[0], label=f"J{j+1}")
        axes[1].plot(df.t, df[f"qd{j+1}"], label=f"J{j+1}")
    axes[2].plot(df.t, 1000 * df.ee_z, label="EE z (from q)")
    axes[2].plot(df.t, 1000 * df.peg_z, label="peg bottom z (true)")
    axes[3].plot(df.t, 1000 * df.peg_x, label="peg x (true)")
    axes[3].axhspan(-1000 * v.clearance, 1000 * v.clearance, color="gray", alpha=0.3, label="capture band")
    for ax in axes:
        if capture_t is not None:
            ax.axvline(capture_t, color="r", ls="--", lw=1)
        ax.legend(fontsize=6, ncol=4)
    axes[0].set_ylabel("q - q(0) (rad)")
    axes[1].set_ylabel("qdot (rad/s)")
    axes[2].set_ylabel("z (mm)")
    axes[3].set_ylabel("x (mm)")
    axes[3].set_xlabel("time (s)")
    fig.suptitle(f"A-S6 drop-in signature, {robot}: capture at {(capture_t or float('nan')):.2f} s (red); "
                 f"peak-to-noise EE-z {pnr:.1f}, joint-velocity {pnr_qd:.1f} (sigma_q={noise:g})")
    fig.tight_layout()
    fig.savefig(out / f"A-S6_dropin_{robot_tag}.png", dpi=120)
    plt.close(fig)
    print(f"A-S6 {robot_tag}: capture {capture_t}, PNR EE-z {pnr:.1f}, joint-vel {pnr_qd:.1f}")
    json.dump(dict(capture_t=capture_t, pnr_eez=pnr, pnr_qd=pnr_qd, sigma_q=noise),
              open(out / f"A-S6_dropin_{robot_tag}.json", "w"), indent=1)
    return capture_t is not None, pnr_qd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="flexiv")
    ap.add_argument("--out", default="results/A")
    ap.add_argument("--only", default="3,4,5,6")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    v = B.sample_study_pegs(1, 0)[0]
    only = args.only.split(",")
    if "3" in only:
        a_s3_step_response(args.robot, out, v)
    if "4" in only:
        a_s4_gravity(args.robot, out, v)
    if "5" in only:
        a_s5_contact_signal(args.robot, out, v)
    if "6" in only:
        a_s6_dropin(args.robot, out, v)


if __name__ == "__main__":
    main()
