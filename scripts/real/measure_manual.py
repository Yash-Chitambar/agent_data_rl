"""Create a real-rig measurement session and record manually measured quantities.

This tool never opens a robot or camera. Values are SI units; unknown values are blank.
See docs/real2sim/BIMANUAL_MEASUREMENT_PLAN.md for the measurement protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import rig


MEASUREMENT_FIELDS = ("item", "side", "quantity", "value", "unit", "uncertainty", "method", "trial_id", "notes")
POSE_FIELDS = ("item", "trial_id", "parent_frame", "child_frame", "x_m", "y_m", "z_m",
               "qw", "qx", "qy", "qz", "method", "uncertainty_m", "support", "notes")
FORCE_FIELDS = ("trial_id", "kind", "item", "t_s", "depth_m", "force_n", "speed_m_s",
                "normal_load_n", "direction", "pc_motion_m", "powered", "notes")
EVENT_FIELDS = ("event", "trial_id", "host_monotonic_ns", "utc", "clock_domain", "notes")


def append_row(path: Path, fields: tuple[str, ...], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


def finite(value: float | None, name: str, *, nonnegative=False) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return value


def session_dir(arg: str) -> Path:
    out = Path(arg).expanduser().resolve()
    if not (out / "session.json").is_file():
        raise SystemExit(f"not an initialized measurement session: {out}")
    return out


def cmd_init(a):
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec = rig.SPEC
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    meta = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "local_timezone": datetime.now().astimezone().tzname(),
        "git_commit": commit,
        "calibration_file": str(spec),
        "calibration_sha256": hashlib.sha256(spec.read_bytes()).hexdigest(),
        "operator": a.operator,
        "session_id": a.session_id,
        "world_frame": "right_arm_base; metres; quaternion w,x,y,z",
        "joint_frame": "raw motor encoder; q_model = q_encoder + calibrated_offset",
        "clock_domains": {"host_monotonic_ns": "CLOCK_MONOTONIC; not directly UTC or camera hardware time"},
        "mounts_and_payload": a.mounts,
        "surface_condition": a.surface,
        "notes": a.notes,
    }
    (out / "session.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, fields in (("measurements.csv", MEASUREMENT_FIELDS), ("poses.csv", POSE_FIELDS),
                         ("force_traces.csv", FORCE_FIELDS), ("events.csv", EVENT_FIELDS)):
        with (out / name).open("w", newline="") as f:
            csv.writer(f).writerow(fields)
    (out / "README.md").write_text(
        f"# {a.session_id}\n\nOperator: {a.operator}\n\nRecord what was performed, omitted, or changed since calibration.\n"
        "Choose train and held-out trial IDs before fitting. Keep failed trials with reasons.\n")
    print(out)


def cmd_measure(a):
    out = session_dir(a.session)
    val = finite(a.value, "value")
    unc = finite(a.uncertainty, "uncertainty", nonnegative=True)
    if val is not None and not a.unit:
        raise SystemExit("--unit is required with --value; use SI units")
    append_row(out / "measurements.csv", MEASUREMENT_FIELDS,
               dict(item=a.item, side=a.side, quantity=a.quantity, value=val, unit=a.unit,
                    uncertainty=unc, method=a.method, trial_id=a.trial_id, notes=a.notes))


def cmd_pose(a):
    out = session_dir(a.session)
    q = [finite(v, "quaternion component") for v in a.quat]
    if abs(math.sqrt(sum(v * v for v in q)) - 1) > 0.01:
        raise SystemExit("quaternion must be unit length, ordered w x y z")
    p = [finite(v, "position") for v in a.pos]
    append_row(out / "poses.csv", POSE_FIELDS,
               dict(item=a.item, trial_id=a.trial_id, parent_frame=a.parent, child_frame=a.child,
                    x_m=p[0], y_m=p[1], z_m=p[2], qw=q[0], qx=q[1], qy=q[2], qz=q[3],
                    method=a.method, uncertainty_m=finite(a.uncertainty, "uncertainty", nonnegative=True),
                    support=a.support, notes=a.notes))


def cmd_force(a):
    out = session_dir(a.session)
    append_row(out / "force_traces.csv", FORCE_FIELDS,
               dict(trial_id=a.trial_id, kind=a.kind, item=a.item,
                    t_s=finite(a.time, "time", nonnegative=True),
                    depth_m=finite(a.depth, "depth"), force_n=finite(a.force, "force"),
                    speed_m_s=finite(a.speed, "speed"), normal_load_n=finite(a.normal_load, "normal load", nonnegative=True),
                    direction=a.direction, pc_motion_m=finite(a.pc_motion, "PC motion", nonnegative=True),
                    powered=a.powered, notes=a.notes))


def cmd_event(a):
    out = session_dir(a.session)
    stamp = time.monotonic_ns() if a.now else a.host_monotonic_ns
    if stamp is None:
        raise SystemExit("provide --now or --host-monotonic-ns")
    append_row(out / "events.csv", EVENT_FIELDS,
               dict(event=a.event, trial_id=a.trial_id, host_monotonic_ns=stamp,
                    utc=datetime.now(timezone.utc).isoformat(), clock_domain=a.clock_domain, notes=a.notes))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="create a new session folder and CSV schemas; no hardware")
    s.add_argument("--out", required=True)
    s.add_argument("--session-id", required=True)
    s.add_argument("--operator", required=True)
    s.add_argument("--mounts", default="")
    s.add_argument("--surface", default="")
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("measure", help="append a mass, dimension, offset or other scalar")
    s.add_argument("--session", required=True)
    s.add_argument("--item", required=True)
    s.add_argument("--side", choices=["right", "left", "both", "none"], default="none")
    s.add_argument("--quantity", required=True)
    s.add_argument("--value", type=float)
    s.add_argument("--unit", default="")
    s.add_argument("--uncertainty", type=float)
    s.add_argument("--method", required=True)
    s.add_argument("--trial-id", required=True)
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_measure)

    s = sub.add_parser("pose", help="append a measured reader, PC, port or tag pose")
    s.add_argument("--session", required=True)
    s.add_argument("--item", required=True)
    s.add_argument("--trial-id", required=True)
    s.add_argument("--parent", default="right_arm_base")
    s.add_argument("--child", required=True)
    s.add_argument("--pos", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    s.add_argument("--quat", type=float, nargs=4, required=True, metavar=("W", "X", "Y", "Z"))
    s.add_argument("--method", required=True)
    s.add_argument("--uncertainty", type=float)
    s.add_argument("--support", default="")
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_pose)

    s = sub.add_parser("force", help="append a manual force/depth or sliding-resistance sample")
    s.add_argument("--session", required=True)
    s.add_argument("--trial-id", required=True)
    s.add_argument("--kind", choices=["insert", "extract", "support_start", "support_slide"], required=True)
    s.add_argument("--item", required=True)
    s.add_argument("--time", type=float, required=True, help="seconds from start of this trial")
    s.add_argument("--depth", type=float, help="metres; leave blank when unavailable")
    s.add_argument("--force", type=float, help="newtons; leave blank without a force gauge")
    s.add_argument("--speed", type=float, help="metres/second")
    s.add_argument("--normal-load", type=float, help="newtons")
    s.add_argument("--pc-motion", type=float, help="metres")
    s.add_argument("--direction", default="")
    s.add_argument("--powered", choices=["yes", "no", "unknown"], default="unknown")
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_force)

    s = sub.add_parser("event", help="record a separately timed LED/trigger event")
    s.add_argument("--session", required=True)
    s.add_argument("--event", required=True)
    s.add_argument("--trial-id", required=True)
    stamp = s.add_mutually_exclusive_group(required=True)
    stamp.add_argument("--host-monotonic-ns", type=int)
    stamp.add_argument("--now", action="store_true", help="timestamp the moment this command runs")
    s.add_argument("--clock-domain", default="host CLOCK_MONOTONIC")
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_event)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    args.func(args)
