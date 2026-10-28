"""Measure a normal reader grasp without dropping it on disconnect.

Connect and auto-calibrate while the jaws are empty. The script then waits for the operator to
place the reader, closes in small normalized steps, records readback for a stationary hold, and
waits for the operator to retrieve the reader before opening and disabling torque.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import estop  # noqa: E402
import measure_motion  # noqa: E402
import rig  # noqa: E402
import safety  # noqa: E402


def telemetry(robot):
    with robot._state_lock:
        s = robot._joint_state
        q = np.asarray(s.pos, float).copy()
        qd = np.asarray(s.vel, float).copy()
        eff = getattr(s, "eff", None)
        temps = getattr(s, "temp_mos", None)
        eff = None if eff is None else np.asarray(eff, float).tolist()
        temps = None if temps is None else np.asarray(temps, float).tolist()
    return q, qd, eff, temps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, default=Path("data/real_yam/usbc_lab_20261027"))
    p.add_argument("--name", default="reader_grasp_002")
    p.add_argument("--hold-seconds", type=float, default=20.0)
    a = p.parse_args()
    out = a.session / "motion" / f"{a.name}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if estop.stopped():
        raise SystemExit("software stop is set")

    robot = safety.connect("right")  # known gripper calibration cycle happens with empty jaws
    arm = safety.SafeArm(robot)
    keep = False
    try:
        q, _, _, _ = telemetry(robot)
        if np.abs(q[1:3]).max() > 0.05:
            raise safety.SafetyStop("right arm is not at folded rest; no grasp trial")
        print(f"Calibrated with empty jaws; gripper readback={q[6]:.3f} (open is near 1).", flush=True)
        input("Place the reader in the open jaws and press Enter to close gradually: ")

        steps = []
        contact_q = None
        prev_readback = float(q[6])
        keep = True  # once the reader may be between the jaws, never torque-off automatically
        for target_g in (0.80, 0.60, 0.45, 0.35, 0.25, 0.18, 0.14, 0.10, 0.06, 0.02):
            q, _, _, _ = telemetry(robot)
            cmd = q.copy()
            cmd[6] = target_g
            arm.command(cmd)
            time.sleep(0.4)
            q, _, eff, temps = telemetry(robot)
            steps.append({"host_monotonic_ns": time.monotonic_ns(), "target_gripper_normalized": target_g,
                          "joint_readback": q.tolist(), "driver_effort_report": eff,
                          "motor_temperature_report": temps})
            print(f"target={target_g:.2f}, readback={q[6]:.3f}; driver effort is uncalibrated", flush=True)
            progress = prev_readback - float(q[6])
            if target_g <= 0.35 and progress < 0.02:
                contact_q = q.copy()
                print("Gripper readback plateau detected; holding this position for operator inspection.", flush=True)
                break
            prev_readback = float(q[6])

        if contact_q is None:
            contact_q = q.copy()
            print("No readback plateau detected in the bounded close. Holding current position; do not disconnect.", flush=True)
        arm._hold(contact_q)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({"created_utc": datetime.now(timezone.utc).isoformat(), "side": "right",
                       "start_arm_q_encoder": q[:6].tolist(), "gripper_steps": steps,
                       "contact_readback": contact_q.tolist(), "hold_seconds": a.hold_seconds,
                       "hold_samples_csv": str(out.with_suffix(".csv")),
                       "note": "Normalized gripper readback only. Driver effort is uncalibrated, not force."}, f, indent=2)
            f.write("\n")
        hold_path = out.with_suffix(".csv")
        start = time.monotonic()
        with hold_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("host_monotonic_ns", *(f"q{i}_encoder" for i in range(1, 7)),
                        "gripper_normalized", "driver_effort_report", "motor_temperature_report"))
            while time.monotonic() - start < a.hold_seconds:
                if estop.stopped():
                    print("Software stop set; gripper remains at current command. Do not disconnect until reader is secured.", flush=True)
                    while estop.stopped():
                        time.sleep(0.1)
                q, _, eff, temps = telemetry(robot)
                w.writerow((time.monotonic_ns(), *q[:6], q[6], json.dumps(eff), json.dumps(temps)))
                arm._hold(q)
                time.sleep(0.05)
                f.flush()

        answer = input("Confirm the reader is held and not slipping; press Enter for loaded motion, or type n to skip: ").strip().lower()
        if answer == "n":
            arm._hold(contact_q)
            print("Loaded motion skipped; gripper remains powered. Keep the process running.", flush=True)
            while True:
                if estop.stopped():
                    time.sleep(0.1)
                else:
                    q, _, _, _ = telemetry(robot)
                    arm._hold(q)
                    time.sleep(0.1)

        # Loaded response check at folded rest. The 5 mrad amplitude stays within the reviewed
        # clearance envelope while exercising both directions on joint 3.
        q_start, _, _, _ = telemetry(robot)
        trajectory = measure_motion.Trajectory("triangle", amp=0.005, rate=0.005, dwell=1.0, reps=5)
        for offset in trajectory.extremes():
            q_check = q_start[:6].copy()
            q_check[2] += offset
            measure_motion.checked_path("right", q_start[:6], q_check)
        motion_path = out.with_name(out.stem + "_arm_motion.csv")
        with motion_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(("k", "host_monotonic_ns", "target_q_encoder", "q_encoder", "qd_encoder",
                        "gripper_normalized", "driver_effort_report", "motor_temperature_report"))
            start = time.monotonic()
            dt = measure_motion.DT
            next_tick = start
            for k in range(trajectory.nsteps()):
                if estop.stopped():
                    raise safety.SafetyStop("software stop during loaded motion; keep reader supported")
                q, qd, eff, temps = telemetry(robot)
                target = q_start.copy()
                target[2] += trajectory.offset(k * dt)
                arm.command(target)
                q_after, qd_after, eff_after, temps_after = telemetry(robot)
                w.writerow((k, time.monotonic_ns(), json.dumps(target[:6].tolist()),
                            json.dumps(q_after[:6].tolist()), json.dumps(qd_after[:6].tolist()), q_after[6],
                            json.dumps(eff_after), json.dumps(temps_after)))
                f.flush()
                next_tick += dt
                time.sleep(max(0.0, next_tick - time.monotonic()))
        q_end, _, _, _ = telemetry(robot)
        if np.max(np.abs(q_end[:6] - q_start[:6])) > measure_motion.START_POSE_TOL:
            raise safety.SafetyStop("loaded motion did not return to its start pose; keep gripper powered")
        arm._hold(q_end)
        print(f"Loaded joint-3 response recorded to {motion_path}; gripper remains powered.", flush=True)
        input("Hold recorded. Secure the reader by hand, then press Enter to open and disable torque: ")
        for target_g in (0.35, 0.55, 0.75, 0.95):
            q, _, _, _ = telemetry(robot)
            cmd = q.copy()
            cmd[6] = target_g
            arm.command(cmd)
            time.sleep(0.3)
        q, _, _, _ = telemetry(robot)
        arm._hold(q)
        keep = False
        print(f"Jaws open; final normalized readback={q[6]:.3f}.", flush=True)
    finally:
        if not keep:
            safety.close_robot(robot)
        else:
            # Keep the CAN/controller connection alive and position held until the operator
            # explicitly confirms retrieval and opening above.
            print("Gripper remains powered and holding. Keep this process running.", flush=True)


if __name__ == "__main__":
    main()
