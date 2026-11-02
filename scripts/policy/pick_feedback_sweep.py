"""Sim sweep for the real-pick feedback of 2026-11-01 (duchamp3, data/real_runs/2026-11-01/SESSION_NOTES.md):
grasp tilt (operator: "the angle the gripper comes down at does not make full surface contact; slightly more angle")
x approach finger opening (real: 24 mm caught with press relief, 25 / 26 mm closed empty).

Runs scripts/policy/pick_expert_eval.py once per (tilt, opening) in a subprocess with reader_grasp.TILT_DEG
patched (the eval itself is unchanged); planning and servo on encoder FK (--believed_frame). Summary table ->
<out>/sweep.json and printed.

    PYTHONPATH=. .venv/bin/python scripts/policy/pick_feedback_sweep.py --tilts 50 55 60 65 --open_mm 23 24 25 26 --episodes 512 --extra --servo oracle
    # the real pick's nets (trained at 55 deg):
    ... --extra --servo runs/perception/reader_servo_hover.pt --servo_near runs/perception/reader_servo_near.pt \
        --servo_near_rot runs/perception/reader_servo_near6.pt --servo_fracs 0.5 0.8 0.92 --rehover_mm 8
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNNER = r'''
import runpy, sys
from blind_insertion.scenes import reader_grasp as G
G.TILT_DEG = float(sys.argv[1])
sys.argv = ['pick_expert_eval.py'] + sys.argv[2:]
runpy.run_path('scripts/policy/pick_expert_eval.py', run_name='__main__')
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tilts', type=float, nargs='+', default=[50., 55., 60., 65.])
    ap.add_argument('--open_mm', type=float, nargs='+', default=[23., 24., 25., 26.])
    ap.add_argument('--episodes', type=int, default=512)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--out', default='results/usbc/policy/pick/feedback_sweep_20261101')
    ap.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='more pick_expert_eval.py arguments')
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tilt in a.tilts:
        for mm in a.open_mm:
            tag = f'tilt{tilt:g}_open{mm:g}'
            sub = out / tag
            cmd = [sys.executable, '-c', RUNNER, str(tilt), '--episodes', str(a.episodes), '--nworld', str(a.nworld),
                   '--seed', str(a.seed), '--finger_open', str(mm * 1e-3), '--believed_frame', '--out', str(sub), *a.extra]
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            (out / f'{tag}.log').write_text(r.stdout + r.stderr)
            summ = sorted(out.glob(f'{tag}_*s{a.seed}.json'))
            row = dict(tilt_deg=tilt, open_mm=mm, returncode=r.returncode, seconds=round(time.time() - t0, 1))
            if summ:
                s = json.loads(summ[-1].read_text())
                row.update({k: s[k] for k in s if not isinstance(s[k], (list, dict))})
            rows.append(row)
            print(json.dumps(row), flush=True)
    (out / 'sweep.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
