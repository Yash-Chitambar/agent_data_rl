"""Real rig: move from the folded rest to ACT's ready pose with the settle trim, check, return.

Jaws empty, PC and reader out of the arm's way, estop.py running. Steps (all logged):
  1. slow move to act.start_q_model, settle with the bounded integral trim (deploy/approach_trim.py)
  2. hold the trimmed command 2 s: the pose must stay within SETTLE_TOL
  3. command the measured pose (trim dropped, as ACT's first tick does) for 1 s: logs the drift
  4. slow move back to the recorded rest
Pass: every joint within 0.02 rad of the ready pose after 2 and 3 (the runner's gate), and
reported against the sim's start envelope (ready +- PickCfg.start_q_noise). On a failure the arm
holds powered until the operator types SUPPORTED.

    python -m deploy.checks.ready_settle --name ready_settle_001
"""
import argparse
import json
import time

import numpy as np

from deploy import conventions as C
from deploy.runner import load_config

OUT = C.REPO / "data/task_h"
GATE = 0.02


def hold_for(be, q7, s, goal):
    """Command q7 for s seconds; returns the logged (t, measured) rows and the max |goal - q|."""
    import estop
    rows, worst = [], 0.0
    t_next, end = time.monotonic(), time.monotonic() + s
    while time.monotonic() < end:
        if estop.stopped():
            be.hold()
            raise RuntimeError("software stop during hold")
        be.arm.safe.command(q7)
        q = np.asarray(be.arm.robot.get_joint_pos(), float)
        rows.append(dict(host_monotonic=time.monotonic(), target=np.asarray(q7).tolist(), measured=q.tolist()))
        worst = max(worst, float(np.max(np.abs(goal - q[:6]))))
        t_next += be.control_dt
        be.wait(t_next)
    return rows, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    import estop
    from deploy.backends import RealBackend
    from deploy.recovery import supported_shutdown
    if estop.stopped():
        raise SystemExit(f"{estop.STOP_FILE} is set: clear it in estop.py first")
    out = OUT / f"{a.name}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    cfg = load_config()
    off = C.joint_offsets(cfg["side"])
    goal = np.asarray(cfg["act"]["start_q_model"], float) - off
    be = RealBackend(cfg, off)
    q0 = be.read()[0].copy()
    rep = dict(goal_motor=goal.tolist(), start=q0.tolist(), controller=be.arm.info())
    ok = False
    try:
        if np.max(np.abs(q0[1:3])) > 0.05:
            raise RuntimeError(f"not at the folded rest: joints 2-3 {q0[1:3].round(3).tolist()}")
        t0 = time.monotonic()
        be.move_to(goal, grip01=cfg["gripper"]["open"], settle_trim=True)
        rep["t_move_settle_s"] = time.monotonic() - t0
        settle = [x for x in be.arm.motion_trace if x["mode"] == "settle"]
        rep["settle_ticks"] = len(settle)
        rep["bias_end"] = settle[-1]["bias"] if settle else None
        q7 = np.asarray(be.arm.safe.last_safe, float).copy()        # the trimmed command
        reached = be.read()[0].copy()
        rep["err_settled"] = (reached - goal).tolist()
        print(f"settled in {len(settle) * be.control_dt:.1f} s, error mrad {np.round((reached - goal) * 1e3, 1)}, "
              f"trim mrad {np.round(np.asarray(rep['bias_end']) * 1e3, 1)}", flush=True)
        rows, rep["max_err_trim_hold"] = hold_for(be, q7, 2.0, goal)
        rep["trim_hold"] = rows
        q7m = np.asarray(be.arm.robot.get_joint_pos(), float)
        q7m[6] = q7[6]
        rows, rep["max_err_untrimmed_hold"] = hold_for(be, q7m, 1.0, goal)
        rep["untrimmed_hold"] = rows
        end = be.read()[0].copy()
        rep["err_untrimmed_end"] = (end - goal).tolist()
        noise = 0.08    # PickCfg.start_q_noise
        rep["in_sim_start_envelope"] = bool(np.all(np.abs(end - goal) <= noise))
        rep["pass"] = bool(rep["max_err_trim_hold"] < GATE and rep["max_err_untrimmed_hold"] < GATE)
        print(f"hold with trim: max error {rep['max_err_trim_hold'] * 1e3:.1f} mrad; trim dropped: max "
              f"{rep['max_err_untrimmed_hold'] * 1e3:.1f} mrad, end mrad {np.round((end - goal) * 1e3, 1)}; "
              f"{'PASS' if rep['pass'] else 'FAIL'} (gate {GATE * 1e3:.0f} mrad)", flush=True)
        print("returning to the recorded rest", flush=True)
        be.move_to(q0, grip01=cfg["gripper"]["open"])
        rep["result"] = "completed"
        ok = True
    except BaseException as exc:      # noqa: BLE001  (SafetyStop, KeyboardInterrupt: the arm holds)
        rep["result"] = repr(exc)
        estop.STOP_FILE.write_text("stop: ready_settle failed\n")
    finally:
        rep["motion_trace"] = be.arm.motion_trace
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=1) + "\n")
        print(rep["result"], "->", out, flush=True)
        if ok:
            be.close()
        else:
            print("Arm holds powered. Do not terminate this process.", flush=True)
            supported_shutdown(be)


if __name__ == "__main__":
    main()
