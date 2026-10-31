"""Real rig: validate the encoder grasp gate (deploy/grasp.reader_present) on the right gripper.

The arm holds its folded-rest pose; only the gripper moves. `--reps` times: close empty (the gate
must say no), then you hold the SD reader between the open fingers (top-down, across its 36 mm
body, as ACT grasps it) and it closes on it (the gate must say yes over a 2 s squeeze). Readings
over each squeeze are logged (a falling reading on the reader = slipping). Writes
data/task_h/<name>.json; pass = every verdict right. Then set gripper.verified_reader_gate.

    python -m deploy.checks.grasp_gate --name grasp_gate_001      (scripts/real/estop.py running)
"""
import argparse
import json
import time

import numpy as np

from deploy import conventions as C
from deploy.checks.gripper_cal import settle, wait_input
from deploy.grasp import reader_present
from deploy.runner import load_config


def squeeze(arm, g, s=2.0):
    """Close to g and log the reading every 20 ms for s seconds after a 1 s close."""
    settle(arm, g, 1.0)
    rows, end = [], time.monotonic() + s
    while time.monotonic() < end:
        rows.append(float(arm.read()[6]))
        time.sleep(0.02)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    import estop
    import safety
    out = C.REPO / "data/task_h" / f"{a.name}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if estop.stopped():
        raise SystemExit("software stop is set")
    cfg = load_config()
    gc = cfg["gripper"]
    robot = safety.connect(cfg["side"])
    arm = safety.SafeArm(robot)
    rep = dict(g_empty=gc["g_empty"], g_reader=gc["g_reader"], close=gc["close"], trials=[])
    reader_may_be_held = False
    try:
        for i in range(a.reps):
            settle(arm, gc["open"])
            rows = squeeze(arm, gc["close"])
            rep["trials"].append(dict(kind="empty", rep=i, readings=rows,
                                      verdicts=[reader_present(g, gc["g_empty"], gc["g_reader"]) for g in rows]))
            print(f"empty {i}: reading {np.median(rows):.3f}, gate {'yes' if any(rep['trials'][-1]['verdicts']) else 'no'}", flush=True)
            settle(arm, gc["open"])
            reader_may_be_held = True
            wait_input(f"[{i + 1}/{a.reps}] Hold the SD reader between the open fingers, then press Enter: ")
            rows = squeeze(arm, gc["close"])
            rep["trials"].append(dict(kind="reader", rep=i, readings=rows,
                                      verdicts=[reader_present(g, gc["g_empty"], gc["g_reader"]) for g in rows]))
            print(f"reader {i}: reading {np.median(rows):.3f} (min {min(rows):.3f}, max {max(rows):.3f}), "
                  f"gate {'yes' if all(rep['trials'][-1]['verdicts']) else 'NO'}", flush=True)
            wait_input("Hold the reader securely; press Enter to reopen the jaws: ")
            settle(arm, gc["open"])
            reader_may_be_held = False
    except Exception as exc:
        rep["result"] = repr(exc)
        out.write_text(json.dumps(rep, indent=1) + "\n")
        if reader_may_be_held:
            print("Reader may be held: maintaining power. Secure the reader, then stop this process.", flush=True)
            while True:
                time.sleep(0.1)
        raise
    finally:
        if not reader_may_be_held:
            safety.close_robot(robot)
    emp = [t for t in rep["trials"] if t["kind"] == "empty"]
    rd = [t for t in rep["trials"] if t["kind"] == "reader"]
    rep["pass"] = bool(emp and rd and not any(any(t["verdicts"]) for t in emp) and all(all(t["verdicts"]) for t in rd))
    rep["empty_readings"] = [float(np.median(t["readings"])) for t in emp]
    rep["reader_readings"] = [float(np.median(t["readings"])) for t in rd]
    rep["result"] = "completed"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"empty {np.round(rep['empty_readings'], 3).tolist()}, reader {np.round(rep['reader_readings'], 3).tolist()} "
          f"-> {'PASS' if rep['pass'] else 'FAIL'} ({out})")


if __name__ == "__main__":
    main()
