"""Guarded YAM friction, loaded-hold and coordinated bimanual measurement runner.

This is a lab tool: `plan` is offline; `single` and `bimanual` require --run and
move real arms. Run rigctl.py in another terminal, clamp both bases and review
the complete start/return path before using it. One process owns both arms for
bimanual timing. The grippers are only calibrated on connect, never commanded here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calib  # noqa: E402
import deploy  # noqa: E402
import estop  # noqa: E402
import rig  # noqa: E402
import safety  # noqa: E402

DT = deploy.CONTROL_DT
START_POSE_TOL = 0.02  # rad, maximum encoder error after a requested move to the measurement pose
FIELDS = ("k", "side", "phase", "t_tick", "t_obs", "obs_exact", "t_read_return", "t_action_ready",
          "t_command_return", "t_send", "q", "qd", "q_target", "q_target_velocity_rad_s",
          "q_cmd", "safety_step_clipped", "motor_current_a", "motor_torque_nm",
          "joint_eff_driver", "computed_motor_feedforward_driver", "temp_mos_driver", "temp_rotor_driver", "loop_overrun_s")


class Trajectory:
    """Bounded absolute joint target offset. All time values are in seconds."""

    def __init__(self, mode: str, amp: float, rate: float, dwell: float, reps: int):
        if mode not in ("hold", "ramp", "triangle", "dwell"):
            raise ValueError(mode)
        if not math.isfinite(amp) or abs(amp) > 0.08:
            raise ValueError("amplitude must be finite and <= 0.08 rad")
        if not math.isfinite(rate) or not 0.002 <= rate <= 0.08:
            raise ValueError("rate must be 0.002..0.08 rad/s")
        if not math.isfinite(dwell) or not 0.2 <= dwell <= 5.0:
            raise ValueError("dwell must be 0.2..5 s")
        if not 1 <= reps <= 10:
            raise ValueError("reps must be 1..10")
        self.mode, self.amp, self.rate, self.dwell, self.reps = mode, amp, rate, dwell, reps
        self.travel = abs(amp) / rate if amp else 0.0
        self.initial_hold = 1.0
        self.cycle = {
            "hold": 1.0,
            "ramp": 2 * self.travel + 2 * dwell,
            "triangle": 4 * self.travel + dwell,
            "dwell": 4 * dwell,
        }[mode]
        self.seconds = (self.initial_hold + reps * self.cycle) if mode != "hold" else max(1.0, reps * dwell)

    def offset(self, t: float) -> float:
        if self.mode == "hold" or t < self.initial_hold or t >= self.seconds:
            return 0.0
        u = (t - self.initial_hold) % self.cycle
        a, d, r = self.amp, self.dwell, self.travel
        if self.mode == "ramp":
            if u < r:
                return a * u / r
            if u < r + d:
                return a
            if u < 2 * r + d:
                return a * (1 - (u - r - d) / r)
            return 0.0
        if self.mode == "triangle":
            if u < r:
                return a * u / r
            if u < 3 * r:
                return a * (1 - (u - r) / r)
            if u < 4 * r:
                return -a * (1 - (u - 3 * r) / r)
            return 0.0
        # dwell: positive step, center pause, reverse step, center pause.
        if u < d:
            return a
        if u < 2 * d:
            return 0.0
        if u < 3 * d:
            return -a
        return 0.0

    def extremes(self) -> tuple[float, ...]:
        return (0.0,) if self.mode == "hold" else ((0.0, self.amp, -self.amp)
                                                       if self.mode in ("triangle", "dwell") else (0.0, self.amp))

    def nsteps(self) -> int:
        return math.ceil(self.seconds / DT)


def validate_run_name(session: Path, name: str) -> Path:
    if not (session / "session.json").is_file():
        raise SystemExit("initialize the session with measure_manual.py init first")
    if not name or "/" in name or name in (".", ".."):
        raise SystemExit("name must be one path component")
    out = session / "motion" / name
    out.mkdir(parents=True, exist_ok=False)
    return out


def path_checker(side: str):
    """Reuse the calibrated local table, reach and inward limits without connecting."""
    m = calib.Mover.__new__(calib.Mover)
    m.side = side
    m.check = calib.calib_checker(calib.PATH_CLEARANCE)
    return m


def checked_path(side: str, q0, q1, *, recovery=False) -> None:
    m = path_checker(side)
    check = calib.calib_checker(calib.RECOVER_CLEARANCE) if recovery else m.check
    reason = m.path_ok(np.asarray(q0), np.asarray(q1), n=41, check=check)
    if reason:
        raise safety.SafetyStop(f"{side} path refused: {reason}")


def connect(side: str) -> tuple[deploy.RealArm, np.ndarray]:
    kp = deploy.SIM_KP
    kd = np.minimum(deploy.SIM_KD, deploy.DM_KD_MAX)
    arm = deploy.RealArm(side, kp, kd)
    arm.safe.check = calib.calib_checker(calib.MONITOR_CLEARANCE)
    q, _, _, _ = arm.read()
    return arm, q[:6].copy()


def snapshot(arm: deploy.RealArm, k: int, side: str, phase: str, target: np.ndarray,
             target_velocity: np.ndarray, t_tick: float, t_ready: float, overrun: float) -> dict:
    q, qd, t_obs, exact = arm.read()
    t_read = time.monotonic()
    r = arm.robot
    with r._state_lock:
        state = r._joint_state
        eff = np.asarray(state.eff, float).copy() if getattr(state, "eff", None) is not None else None
        ff = getattr(r, "_last_motor_torques", None)
        ff = np.asarray(ff, float).copy() if ff is not None else None
        mos = np.asarray(state.temp_mos, float).copy() if getattr(state, "temp_mos", None) is not None else None
        rotor = np.asarray(state.temp_rotor, float).copy() if getattr(state, "temp_rotor", None) is not None else None
    sent = arm.send(np.r_[target, q[6]], k)
    return dict(k=k, side=side, phase=phase, t_tick=t_tick, t_obs=t_obs, obs_exact=exact,
                t_read_return=t_read, t_action_ready=t_ready, t_command_return=time.monotonic(),
                q=q[:6].tolist(), qd=qd[:6].tolist(), q_target=target.tolist(),
                q_target_velocity_rad_s=target_velocity.tolist(), q_cmd=np.asarray(sent)[:6].tolist(),
                safety_step_clipped=bool(np.max(np.abs(np.asarray(sent)[:6] - target)) > 1e-9),
                motor_current_a=None, motor_torque_nm=None,
                joint_eff_driver=None if eff is None else eff.tolist(),
                computed_motor_feedforward_driver=None if ff is None else ff.tolist(),
                temp_mos_driver=None if mos is None else mos.tolist(),
                temp_rotor_driver=None if rotor is None else rotor.tolist(), loop_overrun_s=overrun)


def run_ticks(arms: dict, centers: dict, trajectories: dict, joints: dict, seconds: float, rows: list[dict],
              *, infer=False) -> list[dict]:
    import torch
    actors = {side: deploy.dummy_actor() for side in arms} if infer else {}
    obs = {side: deploy.ObsBuilder() for side in arms} if infer else {}
    previous_target = {side: centers[side].copy() for side in arms}
    next_tick = time.monotonic()
    for k in range(math.ceil(seconds / DT)):
        if estop.stopped():
            raise safety.SafetyStop("software stop")
        t_tick = time.monotonic()
        overrun = max(0.0, t_tick - next_tick)
        targets = {}
        for side, arm in arms.items():
            if infer:
                q, qd, _, _ = arm.read()
                with torch.no_grad():
                    actors[side].act_mean(obs[side].push(q[:6] + rig.joint_offsets(side), qd[:6]))
            target = centers[side].copy()
            traj = trajectories.get(side)
            if traj is not None and traj.mode != "hold":
                target[joints[side] - 1] += traj.offset(k * DT)
            targets[side] = target
        t_ready = time.monotonic()
        # Alternate send order so one side is not always penalized by sequential CAN writes.
        order = tuple(arms) if k % 2 == 0 else tuple(reversed(tuple(arms)))
        for side in order:
            target_velocity = (targets[side] - previous_target[side]) / DT
            rows.append(snapshot(arms[side], k, side, trajectories.get(side).mode if side in trajectories else "hold",
                                 targets[side], target_velocity, t_tick, t_ready, overrun))
            previous_target[side] = targets[side]
        next_tick += DT
        next(iter(arms.values())).wait(next_tick)
    return rows


def logged_approach(arm: deploy.RealArm, side: str, target: np.ndarray, rows: list[dict]) -> None:
    """Record the existing guarded move without changing its command or safety behavior."""
    original = arm.safe.command

    def command(q7):
        label = f"approach:{side}:{len(rows)}"
        q, qd, t_obs, exact = arm.read()
        with arm.robot._state_lock:
            ff = getattr(arm.robot, "_last_motor_torques", None)
            ff = None if ff is None else np.asarray(ff, float).copy().tolist()
        record = {"label": label, "side": side, "t_obs": t_obs, "obs_exact": exact,
                  "t_request": time.monotonic(), "q": q[:6].tolist(), "qd": qd[:6].tolist(),
                  "q_target": np.asarray(q7, float)[:6].tolist(), "q_cmd": None,
                  "computed_motor_feedforward_driver": ff, "t_command_return": None, "error": None}
        arm.timing.label = label
        try:
            sent = original(q7)
            record["q_cmd"] = np.asarray(sent, float)[:6].tolist()
            record["t_command_return"] = time.monotonic()
            return sent
        except BaseException as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            rows.append(record)

    arm.safe.command = command
    try:
        arm.move_to(target, max_vel=0.2)
    finally:
        arm.safe.command = original
        arm.timing.label = None


def save(out: Path, arms: dict, rows: list[dict], meta: dict, approach_rows: list[dict] | None = None) -> None:
    for row in rows:
        row["t_send"] = arms[row["side"]].timing.sent.get(row["k"])
    with (out / "steps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key)
                        for key in FIELDS})
    for side, arm in arms.items():
        with (out / f"encoder_{side}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("host_arrival_s", "motor_iteration", *(f"q_{i}_rad" for i in range(1, 8))))
            for t, it, q in list(arm.timing.enc):
                w.writerow((t, it, *q))
    approach_rows = approach_rows or []
    fields = ("label", "side", "t_obs", "obs_exact", "t_request", "t_command_return", "t_send",
              "q", "qd", "q_target", "q_cmd", "computed_motor_feedforward_driver", "error")
    with (out / "approach.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in approach_rows:
            row["t_send"] = arms[row["side"]].timing.sent.get(row["label"])
            w.writerow({k: json.dumps(row[k]) if isinstance(row.get(k), list) else row.get(k) for k in fields})
    meta["steps"] = len(rows)
    meta["approach_steps"] = len(approach_rows)
    meta["approach_unsent_steps"] = sum(row["t_send"] is None for row in approach_rows if row["q_cmd"] is not None)
    meta["unsent_steps"] = sum(r["t_send"] is None for r in rows)
    if len(arms) == 2:
        by_k = {}
        for row in rows:
            by_k.setdefault(row["k"], {})[row["side"]] = row
        skew = [abs(p["right"]["t_send"] - p["left"]["t_send"])
                for p in by_k.values() if len(p) == 2 and p["right"]["t_send"] is not None
                and p["left"]["t_send"] is not None]
        meta["can_send_skew_ms"] = {"mean": float(np.mean(skew) * 1e3),
                                    "p99": float(np.percentile(skew, 99) * 1e3)} if skew else None
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")


def finish(arms: dict, rest: dict, *, interrupted=False) -> None:
    if interrupted:
        estop.STOP_FILE.write_text("stop")
        print("STOP: both arms holding. Clear the stop in rigctl/estop to return to rest.", flush=True)
    while estop.stopped():
        for arm in arms.values():
            arm.hold()
        time.sleep(0.2)
    for side in reversed(tuple(arms)):
        arm = arms[side]
        while True:
            try:
                arm.safe.check = calib.calib_checker(calib.RECOVER_CLEARANCE)
                q, _, _, _ = arm.read()
                checked_path(side, q[:6], rest[side], recovery=True)
                arm.move_to(rest[side], max_vel=0.2)
                q, _, _, _ = arm.read()
                if np.abs(q[1:3]).max() > 0.05:
                    raise safety.SafetyStop(f"{side} not at folded rest; still holding")
                arm.close()
                print(f"{side}: at rest, torque off", flush=True)
                break
            except safety.SafetyStop as e:
                arm.hold()
                estop.STOP_FILE.write_text("stop")
                print(f"{side}: {e}. Holding; clear stop to retry return.", flush=True)
                while estop.stopped():
                    time.sleep(0.2)


def run(a):
    if not a.run:
        raise SystemExit("this command moves the real arm(s); pass --run after reviewing the path")
    if estop.stopped():
        raise SystemExit("software stop is set")
    traj = Trajectory(a.mode, a.amp, a.rate, a.dwell, a.reps)
    if not 1 <= a.joint <= 6:
        raise SystemExit("joint must be 1..6")
    if a.mode != "hold" and a.amp == 0:
        raise SystemExit("motion mode requires nonzero --amp")
    if a.cmd == "bimanual" and a.mode != "hold" and a.moving_side is None:
        raise SystemExit("bimanual motion mode needs --moving-side")
    if a.cmd == "bimanual" and a.mode != "hold" and not a.motion_reviewed:
        raise SystemExit("bimanual motion requires --motion-reviewed after checking other-arm clearance")
    if a.cmd == "bimanual" and a.camera_save_every < 1:
        raise SystemExit("camera-save-every must be >= 1")
    if a.seconds is None:
        a.seconds = traj.seconds
    if not math.isfinite(a.seconds) or not 1 <= a.seconds <= 300:
        raise SystemExit("duration must be 1..300 s")
    if a.mode != "hold" and a.seconds < traj.seconds:
        raise SystemExit(f"duration must include the complete trajectory ({traj.seconds:.2f} s)")
    out = validate_run_name(Path(a.session).resolve(), a.name)
    sides = (a.side,) if a.cmd == "single" else ("right", "left")
    moving = a.side if a.cmd == "single" else a.moving_side
    arms, rest, centers, initial, rows, approach_rows = {}, {}, {}, {}, [], []
    camera_proc = None
    stopped = False
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(), "args": vars(a),
            "calibration_file": str(rig.SPEC), "control_dt_s": DT,
            "feedback_note": "q and qd are raw encoder-frame state; q_target_velocity_rad_s "
                             "is the finite difference of planned targets, not a motor velocity command. "
                             "joint_eff_driver and temperatures are uncalibrated driver reports. "
                             "computed_motor_feedforward_driver is the last computed command (gravity, "
                             "friction and any explicit feedforward), not measured output torque; "
                             "it is sampled asynchronously and may predate the target. "
                             "Motor current and calibrated output torque are unavailable. "
                             "encoder_<side>.csv contains native-arrival positions only. "
                             "Missing telemetry is null. t_send is first matching CAN burst."}
    try:
        if getattr(a, "cameras", False):
            ready_file = out / "cameras_ready.json"
            camera_proc = subprocess.Popen([sys.executable, str(HERE / "measure_cameras.py"),
                "--session", a.session, "--name", a.name, "--seconds", "300",
                "--save-every", str(a.camera_save_every), "--ready-file", str(ready_file)])
            deadline = time.monotonic() + 30
            while not ready_file.exists():
                if estop.stopped() or camera_proc.poll() is not None or time.monotonic() >= deadline:
                    raise safety.SafetyStop("camera startup failed/stopped before arm connection")
                time.sleep(.05)
        for side in sides:
            arm, q_initial = connect(side)
            arms[side], rest[side] = arm, np.zeros(6)
            initial[side] = q_initial
            if np.abs(q_initial[1:3]).max() > 0.05:
                raise safety.SafetyStop(f"{side} did not start at folded rest; no measurement started")
            centers[side] = np.array(getattr(a, f"start_{side}", None) or
                                     (a.start if a.cmd == "single" and a.start else q_initial), float)
            checked_path(side, q_initial, centers[side])
            for extreme in (traj.extremes() if side == moving else (0.0,)):
                q = centers[side].copy()
                q[a.joint - 1] += extreme
                checked_path(side, centers[side], q)
            meta.setdefault("arm", {})[side] = {"initial_q_encoder": q_initial.tolist(),
                                                   "rest_target_q_encoder": rest[side].tolist(),
                                                   "start_q_encoder": centers[side].tolist(),
                                                   "interface": rig.can_channel(side), "gains": arm.info(),
                                                   "joint_offset_rad": rig.joint_offsets(side).tolist()}
        # Only one arm moves at a time during the start path; both are held by one process.
        for side in sides:
            if not np.allclose(initial[side], centers[side]):
                logged_approach(arms[side], side, centers[side], approach_rows)
            q_reached, _, _, _ = arms[side].read()
            start_error = q_reached[:6] - centers[side]
            meta.setdefault("arm", {})[side]["measurement_pose_q_encoder"] = centers[side].tolist()
            meta["arm"][side]["measurement_pose_reached_q_encoder"] = q_reached[:6].tolist()
            meta["arm"][side]["measurement_pose_error_rad"] = start_error.tolist()
            if np.max(np.abs(start_error)) > START_POSE_TOL:
                raise safety.SafetyStop(
                    f"{side} did not reach the requested measurement pose: max error "
                    f"{np.max(np.abs(start_error)):.3f} rad > {START_POSE_TOL:.3f} rad; holding")
        time.sleep(1.0)
        if camera_proc is not None and camera_proc.poll() is not None:
            raise safety.SafetyStop("camera recording ended before motion trial")
        trajectories = {moving: traj} if moving is not None else {}
        joints = {side: a.joint for side in sides}
        run_ticks(arms, centers, trajectories, joints, a.seconds, rows, infer=a.infer)
    except (safety.SafetyStop, KeyboardInterrupt) as e:
        stopped = True
        meta["stop_reason"] = str(e) or "interrupted"
    except BaseException as e:
        stopped = True
        meta["stop_reason"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        try:
            for arm in arms.values():
                arm.hold()
            if camera_proc is not None:
                try:
                    meta["camera_capture_exit"] = camera_proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    camera_proc.terminate()
                    meta["camera_capture_exit"] = camera_proc.wait(timeout=5)
            if arms:
                # The approach can fail before the first 50 Hz trial row. Preserve native
                # encoder arrivals in that case as well as the stop reason and empty CSV.
                save(out, arms, rows, meta, approach_rows)
            else:
                (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        finally:
            if arms:
                finish(arms, rest, interrupted=stopped)
    if stopped:
        raise SystemExit(f"measurement stopped: {meta['stop_reason']}")
    print(f"wrote {out}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan", help="offline trajectory preview; no robot connection")
    s.add_argument("--mode", choices=["hold", "ramp", "triangle", "dwell"], required=True)
    s.add_argument("--amp", type=float, default=0.02)
    s.add_argument("--rate", type=float, default=0.02)
    s.add_argument("--dwell", type=float, default=1.0)
    s.add_argument("--reps", type=int, default=3)
    for cmd in ("single", "bimanual"):
        s = sub.add_parser(cmd, help="MOTION: connect arm(s), log and return to folded rest")
        s.add_argument("--session", required=True)
        s.add_argument("--name", required=True)
        s.add_argument("--run", action="store_true", help="required acknowledgement that hardware moves")
        s.add_argument("--mode", choices=["hold", "ramp", "triangle", "dwell"], default="hold")
        s.add_argument("--joint", type=int, default=3, help="1-based arm joint for motion")
        s.add_argument("--amp", type=float, default=0.02, help="signed rad, <= 0.08")
        s.add_argument("--rate", type=float, default=0.02, help="rad/s for ramp/triangle")
        s.add_argument("--dwell", type=float, default=1.0)
        s.add_argument("--reps", type=int, default=3)
        s.add_argument("--seconds", type=float, help="default: planned trajectory length")
        s.add_argument("--infer", action="store_true", help="dummy network forward pass; commands remain fixed-target")
        if cmd == "single":
            s.add_argument("--cameras", action="store_true", help="initialize all three cameras before motors")
            s.add_argument("--camera-save-every", type=int, default=5)
            s.add_argument("--side", choices=["right", "left"], required=True)
            s.add_argument("--start", type=float, nargs=6, help="encoder-frame start joints; checked slow move from folded rest")
        else:
            s.add_argument("--moving-side", choices=["right", "left"], help="omit for both-arm hold")
            s.add_argument("--start-right", type=float, nargs=6)
            s.add_argument("--start-left", type=float, nargs=6)
            s.add_argument("--motion-reviewed", action="store_true", help="required for moving-arm test: operator checked other-arm clearance")
            s.add_argument("--no-cameras", dest="cameras", action="store_false", help="skip concurrent three-camera capture")
            s.add_argument("--camera-save-every", type=int, default=5,
                           help="save every Nth raw image; all frame times still logged")
            s.set_defaults(cameras=True)
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    if a.cmd == "plan":
        t = Trajectory(a.mode, a.amp, a.rate, a.dwell, a.reps)
        print(json.dumps({"seconds": t.seconds, "steps_50_hz": t.nsteps(), "extremes_rad": t.extremes(),
                          "samples": [[round(k * DT, 3), round(t.offset(k * DT), 5)]
                                      for k in range(0, t.nsteps(), max(1, t.nsteps() // 20))]}, indent=2))
    else:
        run(a)
