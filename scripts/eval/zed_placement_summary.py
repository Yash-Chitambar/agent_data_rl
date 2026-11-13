"""Summary of zed_placement.py --pc-poses sweeps (docs/real/ZED_V5_20261114.md): per estimator, per pose class
(step-2 box |dx|, |dy| <= 25 mm and |dyaw| <= 15 deg; v5 box <= 40 mm, <= 20 deg; outside), the gate pass rate, the
estimate error p50 / p90 over all worlds and over the accepted ones, and the accepted worlds with an error > 5 / 10 mm.

    python scripts/eval/zed_placement_summary.py v1=runs/placement_zed_v5/sweep_v1.json v5=... --out results/placement/zed_v5/sweep_summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

REAL = (0.231, 0.394, 4.0)


def cls(r):
    dx, dy, dw = (r['pc'][0] - REAL[0]) * 1e3, (r['pc'][1] - REAL[1]) * 1e3, r['pc'][2] - REAL[2]
    m, w = max(abs(dx), abs(dy)), abs(dw)
    return 'step2 box (+/-25 mm, +/-15 deg)' if m <= 25.5 and w <= 15.5 else 'v5 box rim (<= 40 mm, <= 20 deg)' if m <= 40.5 and w <= 20.5 else 'outside v5 box'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+'); ap.add_argument('--out')
    a = ap.parse_args()
    out = {}
    for spec in a.runs:
        name, f = spec.split('=', 1)
        rows = json.loads(Path(f).read_text())
        res = {}
        for c in ('step2 box (+/-25 mm, +/-15 deg)', 'v5 box rim (<= 40 mm, <= 20 deg)', 'outside v5 box'):
            rs = [r for r in rows if cls(r) == c]
            n = sum(r['n'] for r in rs)
            ok = sum(r['pc_ok'] for r in rs)
            res[c] = dict(poses=len(rs), worlds=n, gate_ok=ok, gate_rate=ok / n if n else None,
                          err_p50_of_pose_p50=float(np.median([r['pc_err_p50'] for r in rs])) if rs else None,
                          worst_pose_err_p50=max((r['pc_err_p50'], r['tag']) for r in rs) if rs else None,
                          poses_gate_passes_err_p50_over_10mm=[r['tag'] for r in rs if r['pc_ok'] > 0 and r['pc_err_p50'] > 10],
                          poses_gate_passes_yaw_err_p50_over_3deg=[r['tag'] for r in rs if r['pc_ok'] > 0 and r['pc_yaw_err_p50'] > 3],
                          worst_pose_yaw_err_p50=max((r['pc_yaw_err_p50'], r['tag']) for r in rs) if rs else None)
        out[name] = res
        print(name)
        for c, v in res.items():
            print(f"  {c:36s} poses {v['poses']:2d} worlds {v['worlds']:3d} gate {v['gate_ok']}/{v['worlds']} "
                  f"pose err p50 median {v['err_p50_of_pose_p50']:.1f} mm, worst {v['worst_pose_err_p50'][0]:.1f} ({v['worst_pose_err_p50'][1]}), "
                  f"worst yaw {v['worst_pose_yaw_err_p50'][0]:.1f} deg ({v['worst_pose_yaw_err_p50'][1]})\n"
                  f"    gate passes with err p50 > 10 mm: {v['poses_gate_passes_err_p50_over_10mm']}; yaw > 3 deg: {v['poses_gate_passes_yaw_err_p50_over_3deg']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
