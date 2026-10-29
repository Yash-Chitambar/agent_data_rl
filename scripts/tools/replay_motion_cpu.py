"""Replay a real approach through the current CPU MuJoCo scene, offline.

This is a baseline comparison, not a fitted controller or validated physics profile.
The model uses its existing gravity compensation and friction; contacts are disabled.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from blind_insertion.scenes import pc_bench as S


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--side", choices=["right", "left"], default="right")
    p.add_argument("--joint", type=int, choices=range(1, 7), default=3)
    p.add_argument("--source", choices=["approach", "steps"], default="approach", help="command log to replay")
    p.add_argument("--joint3-friction", type=float, help="development-only MuJoCo frictionloss candidate")
    p.add_argument("--stiction", nargs=2, type=float, metavar=("STATIC", "SLIDING"),
                   help="experimental stateful joint-3 torque limits in N m")
    p.add_argument("--stiction-mode", choices=["local", "coupled"], default="local")
    p.add_argument("--timestep", type=float, default=0.001)
    p.add_argument("--driver-feedforward", action="store_true", help="approximate the logged driver model offline")
    p.add_argument("--recorded-state-feedforward", action="store_true", help="reconstruct feedforward from logged encoder states")
    p.add_argument("--logged-feedforward", action="store_true", help="replay logged last-computed driver torque at observation times; not measured torque")
    p.add_argument("--out-dir", type=Path, help="write outside the immutable capture bundle")
    p.add_argument("--output-stem", default="cpu_baseline_replay", help="new output name within run directory")
    a = p.parse_args()
    if not a.output_stem or Path(a.output_stem).name != a.output_stem:
        raise SystemExit("output-stem must be a single path component")
    if a.joint3_friction is not None and (not np.isfinite(a.joint3_friction) or a.joint3_friction < 0):
        raise SystemExit("joint3-friction must be finite and nonnegative")
    if a.stiction and (a.joint3_friction is not None or not np.isfinite(a.stiction).all() or
                       a.stiction[1] < 0 or a.stiction[0] < a.stiction[1]):
        raise SystemExit("stiction requires finite STATIC >= SLIDING >= 0 and no joint3-friction")
    if not np.isfinite(a.timestep) or not 0.00025 <= a.timestep <= 0.002:
        raise SystemExit("timestep must be between 0.00025 and 0.002 s")
    if sum((a.driver_feedforward, a.recorded_state_feedforward, a.logged_feedforward)) > 1:
        raise SystemExit("choose only one driver feedforward reconstruction")
    output = a.out_dir or a.run
    output.mkdir(parents=True, exist_ok=True)
    stem = output / a.output_stem
    if any(stem.with_suffix(ext).exists() for ext in ('.json', '.csv', '.png')):
        raise SystemExit(f"refusing to overwrite existing replay outputs: {stem}")
    meta = json.loads((a.run / "run.json").read_text())
    info = meta["arm"][a.side]
    offsets = np.asarray(info["joint_offset_rad"], dtype=float)
    with (a.run / f"{a.source}.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["side"] == a.side]
    if not rows:
        raise SystemExit(f"no {a.source} commands with send timestamps")
    if any(not r["q_cmd"] or not r["t_send"] or not r["t_obs"] for r in rows):
        raise SystemExit("command or timing data is incomplete; preserve and review the partial run")
    obs0 = float(rows[0]["t_obs"])
    send_t = np.asarray([float(r["t_send"]) - obs0 for r in rows])
    if not np.all(np.diff(send_t) >= 0):
        raise SystemExit("CAN send timestamps are not ordered")
    commands = np.asarray([json.loads(r["q_cmd"]) for r in rows]) + offsets
    obs_t = np.asarray([float(r["t_obs"]) - obs0 for r in rows])
    if (a.recorded_state_feedforward or a.logged_feedforward) and not np.all(np.diff(obs_t) >= 0):
        raise SystemExit("observation timestamps are not ordered")
    logged_ff = None
    if a.logged_feedforward:
        try:
            logged_ff = np.asarray([json.loads(r["computed_motor_feedforward_driver"])[:6] for r in rows], float)
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("missing logged driver feedforward; cannot substitute zero") from exc
        if logged_ff.shape != (len(rows), 6) or not np.isfinite(logged_ff).all():
            raise SystemExit("invalid logged driver feedforward")
    with (a.run / f"encoder_{a.side}.csv").open(newline="") as f:
        enc = list(csv.DictReader(f))
    native_t = np.asarray([float(r["host_arrival_s"]) - obs0 for r in enc])
    native_q = np.asarray([[float(r[f"q_{i}_rad"]) for i in range(1, 7)] for r in enc])
    mask = (native_t >= 0) & (native_t <= send_t[-1] + 1.0)
    native_t, native_q = native_t[mask], native_q[mask]
    if not len(native_t):
        raise SystemExit("no native feedback overlaps the command interval")

    m, lay = S.build_scene(bimanual=a.side == "left")
    prefix = "left_" if a.side == "left" else ""
    arm_qadr = np.array([int(m.joint(prefix+f"joint{i}").qposadr[0]) for i in range(1,7)])
    arm_dofadr = np.array([int(m.joint(prefix+f"joint{i}").dofadr[0]) for i in range(1,7)])
    arm_act = np.array([m.actuator(prefix+f"joint{i}").id for i in range(1,7)])
    m.opt.timestep = a.timestep
    joint3 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "joint3")
    dof3 = int(m.jnt_dofadr[joint3])
    if a.stiction:
        m.dof_frictionloss[dof3] = 0.0
    if a.joint3_friction is not None:
        joint3 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "joint3")
        m.dof_frictionloss[m.jnt_dofadr[joint3]] = a.joint3_friction
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    gain = info["gains"]
    driver_kdl = None
    if a.driver_feedforward or a.recorded_state_feedforward:
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        from i2rt.utils.mujoco_utils import MuJoCoKDL
        driver_xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310, ee_mass=0.5)
        driver_kdl = MuJoCoKDL(driver_xml)
        gravity_factors = np.asarray(gain["gravity_comp_factor"], float)[:6]
        friction_factors = np.asarray(gain["coulomb_friction"], float)[:6]
    for idx, kp, kd in zip(arm_act, gain["kp"], gain["kd"]):
        m.actuator_gainprm[idx, 0] = kp
        m.actuator_biasprm[idx, 1] = -kp
        m.actuator_biasprm[idx, 2] = -kd
    d = mujoco.MjData(m)
    d.qpos[arm_qadr] = np.asarray(json.loads(rows[0]["q"])) + offsets
    d.qvel[[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + f"joint{i}")] for i in range(1, 7)]] = json.loads(rows[0]["qd"])
    d.ctrl[arm_act] = d.qpos[arm_qadr]
    mujoco.mj_forward(m, d)
    sticking = bool(a.stiction and abs(d.qvel[dof3]) < 0.002)
    unit = np.zeros((1, m.nv)); unit[0, dof3] = 1.0
    inv_column = np.empty_like(unit)
    state_counts = {"sticking": 0, "sliding": 0}
    obs_idx, recorded_ff = 0, None
    sim_t, sim_q, k = [0.0], [d.qpos[arm_qadr].copy() - offsets], 0
    while d.time < native_t[-1]:
        while k < len(send_t) and send_t[k] <= d.time:
            d.ctrl[arm_act] = commands[k]
            k += 1
        if logged_ff is not None:
            while obs_idx + 1 < len(obs_t) and obs_t[obs_idx + 1] <= d.time:
                obs_idx += 1
            d.qfrc_applied[arm_dofadr] = 0.0
            mujoco.mj_forward(m, d)
            d.qfrc_applied[arm_dofadr] = logged_ff[obs_idx] - d.qfrc_gravcomp[arm_dofadr]
        if driver_kdl is not None:
            # Replace scene-perfect gravity compensation with the driver's computed gravity and
            # velocity-sign feedforward. Simulated velocity is a proxy for the driver's sampled
            # real encoder velocity; these commands are not measured output torque.
            d.qfrc_applied[arm_dofadr] = 0.0
            mujoco.mj_forward(m, d)
            if a.recorded_state_feedforward:
                while obs_idx < len(obs_t) and obs_t[obs_idx] <= d.time:
                    q_encoder = np.asarray(json.loads(rows[obs_idx]["q"]), float)
                    q_velocity = np.asarray(json.loads(rows[obs_idx]["qd"]), float)
                    gravity = driver_kdl.compute_inverse_dynamics(q_encoder, np.zeros(6), np.zeros(6))
                    recorded_ff = gravity * gravity_factors + friction_factors * np.sign(q_velocity)
                    obs_idx += 1
                if recorded_ff is None:
                    raise RuntimeError("no recorded state available for feedforward")
                driver_ff = recorded_ff
            else:
                q_encoder = d.qpos[arm_qadr] - offsets
                q_velocity = d.qvel[arm_dofadr]
                gravity = driver_kdl.compute_inverse_dynamics(q_encoder, np.zeros(6), np.zeros(6))
                driver_ff = gravity * gravity_factors + friction_factors * np.sign(q_velocity)
            d.qfrc_applied[arm_dofadr] = driver_ff - d.qfrc_gravcomp[arm_dofadr]
        if driver_kdl is None and logged_ff is None:
            d.qfrc_applied[dof3] = 0.0
        base_force3 = float(d.qfrc_applied[dof3])
        if a.stiction:
            # Experimental torque cancellation with stick/slip hysteresis. Coupled dynamics,
            # constraints and the driver's feedforward are not fully reproduced.
            mujoco.mj_forward(m, d)
            drive = float(d.qfrc_actuator[dof3] + d.qfrc_passive[dof3] + base_force3 - d.qfrc_bias[dof3])
            velocity = float(d.qvel[dof3])
            static, sliding = a.stiction
            required = drive
            if a.stiction_mode == "coupled":
                mujoco.mj_solveM(m, d, inv_column, unit)
                required = float((d.qacc[dof3] + velocity / m.opt.timestep) / inv_column[0, dof3])
            if sticking and abs(required) > static:
                sticking = False
            elif not sticking and abs(velocity) < 0.002 and abs(required) < sliding:
                sticking = True
            if sticking:
                d.qfrc_applied[dof3] = base_force3 - np.clip(required, -static, static)
                state_counts["sticking"] += 1
            else:
                direction = np.sign(velocity) if abs(velocity) >= 0.002 else np.sign(drive)
                d.qfrc_applied[dof3] = base_force3 - sliding * direction
                state_counts["sliding"] += 1
        mujoco.mj_step(m, d)
        sim_t.append(float(d.time)); sim_q.append(d.qpos[arm_qadr].copy() - offsets)
    sim_t, sim_q = np.asarray(sim_t), np.asarray(sim_q)
    interp = np.column_stack([np.interp(native_t, sim_t, sim_q[:, j]) for j in range(6)])
    residual = interp - native_q
    summary = {"side": a.side, "run": str(a.run), "source": a.source, "initial_state": f"first {a.source} observation; encoder offsets applied once",
               "command_time": "first matching CAN send on host monotonic clock; zero-order held",
               "controller": "logged kp/kd; current MuJoCo scene gravity compensation; optional joint-3 friction candidate",
               "logged_feedforward": "last-computed driver command held at observation timestamps; command execution timestamp unknown; not measured torque" if a.logged_feedforward else None,
               "joint3_frictionloss_nm": float(m.dof_frictionloss[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "joint3")]]),
               "experimental_stiction_nm": a.stiction, "experimental_stiction_mode": a.stiction_mode if a.stiction else None,
               "experimental_state_steps": state_counts if a.stiction else None,
               "driver_feedforward_reconstruction": ("approximate KDL gravity and velocity sign from logged states; held between observations; not actual recorded command" if a.recorded_state_feedforward else "approximate KDL gravity with logged factors and simulated velocity sign; not actual recorded command" if a.driver_feedforward else None),
               "limitations": ["contacts disabled", "feedforward is a logged computed command, reconstruction, or omitted; no measured torque",
                               "stick/slip force uses inverse mass, not a full constrained contact solve",
                               "not a fitted or held-out validated model", "no external wrist ground truth"],
               "native_feedback_samples": len(native_t),
               "input_sha256": {name: hashlib.sha256((a.run / name).read_bytes()).hexdigest()
                                for name in ("run.json", f"{a.source}.csv", f"encoder_{a.side}.csv")},
               "joint_rmse_rad": np.sqrt(np.mean(residual ** 2, axis=0)).tolist(),
               "joint_peak_abs_error_rad": np.max(np.abs(residual), axis=0).tolist(),
               "model_timestep_s": m.opt.timestep}
    stem.with_suffix('.json').write_text(json.dumps(summary, indent=2) + '\n')
    with stem.with_suffix('.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['time_from_first_observation_s', *[f'real_q{i}_rad' for i in range(1, 7)],
                                      *[f'sim_q{i}_rad' for i in range(1, 7)]])
        for t, real, sim in zip(native_t, native_q, interp): w.writerow([t, *real, *sim])
    j = a.joint - 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(native_t, native_q[:, j], label='Real encoder')
    ax.plot(native_t, interp[:, j], label='MuJoCo baseline')
    ax.step(send_t, commands[:, j]-offsets[j], where='post', label='Executed target', alpha=.7)
    ax.set(xlabel='Seconds from first observation', ylabel='Joint position (rad)', title=f'Joint {a.joint}: real vs CPU baseline')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(stem.with_suffix('.png'), dpi=160); plt.close(fig)
    print(json.dumps({'samples':len(native_t),'joint3_rmse_rad':summary['joint_rmse_rad'][2], 'output':str(stem)}, indent=2))


if __name__ == '__main__':
    main()
