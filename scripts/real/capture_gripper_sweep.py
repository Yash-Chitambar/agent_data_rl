"""Gripper opening sweep seen by each arm's own wrist camera, for the sim fingertip fit.

Moves ONLY the gripper jaws; the arm must be at folded rest and stays held there. One arm is
connected at a time. At each normalized jaw target the jaws settle, then `--frames` raw wrist
frames (1280x1024) are saved with the gripper encoder read just before and after each frame.

    .venv/bin/python scripts/real/capture_gripper_sweep.py --side both --out data/gripper_sweep/<name>

Then, on any machine with the repo:

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/tools/fit_gripper_tips.py --sweep data/gripper_sweep/<name>

Run the rig console (rigctl.py) or estop.py first. Keep the jaws clear: nothing between them.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import calib  # noqa: E402
import estop  # noqa: E402
import rig  # noqa: E402
import safety  # noqa: E402

LEVELS = (1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.12, 0.06, 0.02, 1.0)   # 0.02 = closed empty; ends open


def sweep_side(side, out, levels, frames, settle_s):
    if estop.stopped():
        raise SystemExit(f'{estop.STOP_FILE} is set: clear it first')
    r = safety.connect(side)
    cam, rows = None, []
    try:
        arm = safety.SafeArm(r)
        q = np.asarray(r.get_joint_pos(), float)
        if np.max(np.abs(q[1:3])) > .05:
            raise SystemExit(f'{side}: arm not at folded rest (joints 2-3 {q[1:3].round(3)}); nothing moved')
        print(f'{side}: jaws will open and close {len(levels)} times. Keep them clear. Enter to start.', flush=True)
        input()
        cam = calib.WristCam(side)
        for i, g in enumerate(levels):
            if estop.stopped():
                raise RuntimeError('software stop')
            target = np.asarray(r.get_joint_pos(), float)
            target[6] = g
            arm.command(target)
            t_end = time.monotonic() + settle_s
            while time.monotonic() < t_end:
                if estop.stopped():
                    raise RuntimeError('software stop')
                time.sleep(.02)
            for k in range(frames):
                before = np.asarray(r.get_joint_pos(), float)
                f = cam.grab()
                after = np.asarray(r.get_joint_pos(), float)
                name = f'{side}_{i:02d}_{k}.png'
                cv2.imwrite(str(out / name), f)
                rows.append({'side': side, 'level_index': i, 'target_gripper': g, 'frame': name,
                             'gripper_encoder_before': float(before[6]), 'gripper_encoder_after': float(after[6]),
                             'arm_q_encoder': after[:6].tolist(), 'host_monotonic_ns': time.monotonic_ns()})
            print(f'{side}: target {g:.2f} -> encoder {rows[-1]["gripper_encoder_after"]:.3f}', flush=True)
    finally:
        if cam is not None:
            cam.close()
        safety.close_robot(r)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--side', choices=['left', 'right', 'both'], default='both')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--frames', type=int, default=3)
    p.add_argument('--settle-s', type=float, default=1.5)
    a = p.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out} is not empty: choose a new name')
    a.out.mkdir(parents=True, exist_ok=True)
    rows, result = [], 'incomplete'
    try:
        for side in (('right', 'left') if a.side == 'both' else (a.side,)):
            rows += sweep_side(side, a.out, LEVELS, a.frames, a.settle_s)
        result = 'completed'
    except BaseException as exc:
        result = repr(exc)
        raise
    finally:
        (a.out / 'sweep.json').write_text(json.dumps({
            'result': result, 'levels': LEVELS, 'frames_per_level': a.frames, 'settle_s': a.settle_s,
            'camera_spec': str(rig.SPEC), 'camera_spec_sha256': hashlib.sha256(Path(rig.SPEC).read_bytes()).hexdigest(),
            'note': 'gripper encoder is the driver-normalized jaw reading (1 open); frames are raw wrist images',
            'rows': rows}, indent=1) + '\n')
        print(f'{result}: {len(rows)} frames in {a.out}', flush=True)


if __name__ == '__main__':
    main()
