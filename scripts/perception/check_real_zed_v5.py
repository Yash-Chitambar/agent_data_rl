"""Real-frame check of a ZED PC estimator against the rig-proven one (no labels): every logged ZED capture frame
(`frames/top/*_zed_pc.png`, the 320-px policy frame the rig fed to the net) of the real runs (meta dry_run false)
in data/real_runs/<date>/*. Per frame: the reference estimator (default zed + zed_v1_gate_arm, the rig's) and the new
one (default the zed_v5_gate estimator key), with the believed arm angles of the logged estimate for the arm gate.
Checks: (1) the reference reproduces the logged estimate (same preprocessing); (2) new minus reference in xy (mm)
and yaw (deg); (3) the new gate passes the frames.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/check_real_zed_v5.py --out results/placement/zed_v5/real_frames.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')


def frames(dates):
    import cv2
    out = []
    for d in dates:
        for run in sorted((RUNS / d).glob('*')):
            meta = run / 'meta.json'
            if not run.is_dir() or not meta.exists() or json.loads(meta.read_text()).get('dry_run', True):
                continue
            ev = {}
            ef = run / 'events.jsonl'
            if ef.exists():
                for line in ef.read_text().splitlines():
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc':
                        ev[e.get('frame_tag')] = e
            for k, f in enumerate(sorted((run / 'frames' / 'top').glob('*_zed_pc.png'))):
                logged = ev.get(f'zed_pc@{k}') or (list(ev.values())[k] if k < len(ev) else None)
                img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)      # runlog writes RGB policy frames as BGR
                out.append(dict(date=d, run=run.name, file=str(f), img=img, logged=logged))
    return out


GT = Path('/home/yashchitambar/peg_test/data/zed_gt')


def gt_check(ref, new, dev):
    """data/zed_gt (2026-11-01): raw rectified ZED-left frames at the capture pose, PC at measured spots (spots.json:
    tape + case-edge fit, +/-2-2.5 mm). Arm gate not applied (joints stored without the offsets); presence + OOD only."""
    import cv2
    from blind_insertion.scenes.physical_rig import real_top_policy_image
    spots = json.loads((GT / 'spots.json').read_text())
    out = []
    for name, sp in spots.items():
        for f in sorted((GT / name).glob('capture_*_zed_left.png')):
            x = torch.tensor(real_top_policy_image(cv2.imread(str(f)), 320)[None], device=dev)
            row = dict(spot=name, file=str(f), truth_xy_mm=[v * 1e3 for v in sp['port_xy']], truth_yaw_deg=sp['yaw_deg'])
            for nm, est in (('ref', ref), ('new', new)):
                p, y, ok, info = est(x)
                xy = p[0].cpu().numpy() * 1e3
                row[nm] = dict(xy_mm=xy.tolist(), yaw_deg=math.degrees(float(y[0])), ok=bool(ok[0]), mahalanobis=float(info['mahalanobis'][0]),
                               err_xy_mm=float(np.linalg.norm(xy - np.array(row['truth_xy_mm']))), err_yaw_deg=math.degrees(float(y[0])) - sp['yaw_deg'])
            out.append(row)
            print(f"gt {name:14s} {f.name:28s} ref err {row['ref']['err_xy_mm']:.1f} mm {row['ref']['err_yaw_deg']:+.1f} deg ok {row['ref']['ok']:d} md {row['ref']['mahalanobis']:.1f} | "
                  f"new err {row['new']['err_xy_mm']:.1f} mm {row['new']['err_yaw_deg']:+.1f} deg ok {row['new']['ok']:d} md {row['new']['mahalanobis']:.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', nargs='+', default=['2026-11-02', '2026-11-03', '2026-11-06', '2026-11-07', '2026-11-10', '2026-11-11'])
    ap.add_argument('--ref-gate', default='zed_v1_gate_arm'); ap.add_argument('--ref-pose', default='zed')
    ap.add_argument('--new-gate', default='zed_v5_gate')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gt', action='store_true', help='also the data/zed_gt spots (truth from spots.json)')
    a = ap.parse_args()
    from blind_insertion.perception.zed_estimator import ZedEstimator
    dev = 'cuda:0'
    P = ROOT / 'runs/perception'
    ref = ZedEstimator(dev, pose=str(P / f'{a.ref_pose}.pt'), presence=str(P / f'{a.ref_gate}.pt'))
    new = ZedEstimator(dev, presence=str(P / f'{a.new_gate}.pt'))
    fr = frames(a.dates)
    rows = []
    for r in fr:
        x = torch.tensor(r['img'][None], device=dev)
        arms = None
        lg = r['logged']
        if lg and lg.get('arms_b'):
            ql, qr, fo = lg['arms_b']
            arms = (torch.tensor([ql], device=dev), torch.tensor([qr], device=dev), torch.tensor([float(fo)], device=dev))
        row = dict(date=r['date'], run=r['run'], file=r['file'])
        for name, est in (('ref', ref), ('new', new)):
            p, y, ok, info = est(x, arms=arms)
            row[name] = dict(xy_mm=(p[0].cpu().numpy() * 1e3).tolist(), yaw_deg=math.degrees(float(y[0])), ok=bool(ok[0]),
                             **{k: float(v[0]) for k, v in info.items()})
        if lg:
            row['logged'] = dict(xy_mm=[v * 1e3 for v in lg['port_xy']], yaw_deg=lg['yaw_deg'], ok=lg['ok'], mahalanobis=lg.get('mahalanobis'))
            row['ref_minus_logged_mm'] = float(np.linalg.norm(np.subtract(row['ref']['xy_mm'], row['logged']['xy_mm'])))
        d = np.subtract(row['new']['xy_mm'], row['ref']['xy_mm'])
        row['new_minus_ref'] = dict(dx_mm=float(d[0]), dy_mm=float(d[1]), xy_mm=float(np.linalg.norm(d)),
                                    yaw_deg=row['new']['yaw_deg'] - row['ref']['yaw_deg'])
        rows.append(row)
        print(f"{r['date']} {r['run'][:34]:34s} ref ({row['ref']['xy_mm'][0]:.1f}, {row['ref']['xy_mm'][1]:.1f}; {row['ref']['yaw_deg']:.1f}) ok {row['ref']['ok']:d} "
              f"md {row['ref']['mahalanobis']:.1f} | new ({row['new']['xy_mm'][0]:.1f}, {row['new']['xy_mm'][1]:.1f}; {row['new']['yaw_deg']:.1f}) ok {row['new']['ok']:d} "
              f"md {row['new']['mahalanobis']:.1f} dis {row['new']['disagree_xy_mm']:.1f} | d {row['new_minus_ref']['xy_mm']:.1f} mm {row['new_minus_ref']['yaw_deg']:+.1f} deg"
              + (f" | ref-logged {row['ref_minus_logged_mm']:.2f}" if 'ref_minus_logged_mm' in row else ''), flush=True)
    q = lambda v: {'p50': float(np.percentile(v, 50)), 'p90': float(np.percentile(v, 90)), 'max': float(np.max(v))} if len(v) else None  # noqa: E731
    both = [r for r in rows if r['ref']['ok']]
    new_gate = ZedEstimator(dev, presence=str(P / f'{a.new_gate}.pt'))
    summ = dict(n_frames=len(rows), n_runs=len({(r['date'], r['run']) for r in rows}), dates=a.dates,
                ref=f'{a.ref_pose} + {a.ref_gate}', new=f'{a.new_gate} ({", ".join(Path(f).stem for f in new.pose_files if f)})',
                new_ood_limit=new_gate.ood_thresh * new_gate.ood_scale, ref_ood_limit=ref.ood_thresh * ref.ood_scale,
                ref_reproduces_logged_mm=q([r['ref_minus_logged_mm'] for r in rows if 'ref_minus_logged_mm' in r]),
                ref_gate_ok=sum(r['ref']['ok'] for r in rows), new_gate_ok=sum(r['new']['ok'] for r in rows),
                new_gate_ok_where_ref_ok=f"{sum(r['new']['ok'] for r in both)}/{len(both)}",
                new_minus_ref_xy_mm=q([r['new_minus_ref']['xy_mm'] for r in both]),
                new_minus_ref_dx_mm_median=float(np.median([r['new_minus_ref']['dx_mm'] for r in both])) if both else None,
                new_minus_ref_dy_mm_median=float(np.median([r['new_minus_ref']['dy_mm'] for r in both])) if both else None,
                new_minus_ref_yaw_deg_abs=q([abs(r['new_minus_ref']['yaw_deg']) for r in both]),
                new_minus_ref_yaw_deg_median=float(np.median([r['new_minus_ref']['yaw_deg'] for r in both])) if both else None,
                ref_mahalanobis=q([r['ref']['mahalanobis'] for r in rows]), new_mahalanobis=q([r['new']['mahalanobis'] for r in rows]),
                new_presence_min=float(min(r['new']['presence'] for r in rows)) if rows else None,
                new_disagree_xy_mm=q([r['new']['disagree_xy_mm'] for r in rows]),
                ref_xy_mm_range=[float(np.min([r['ref']['xy_mm'] for r in rows], 0)[i]) for i in (0, 1)] + [float(np.max([r['ref']['xy_mm'] for r in rows], 0)[i]) for i in (0, 1)] if rows else None)
    gt = gt_check(ref, new, dev) if a.gt else []
    if gt:
        summ['gt_spots'] = {k: dict(n=len(g), truth=[g[0]['truth_xy_mm'], g[0]['truth_yaw_deg']],
                                    **{nm: dict(ok=sum(r[nm]['ok'] for r in g), err_xy_mm=q([r[nm]['err_xy_mm'] for r in g]),
                                                err_yaw_deg_median=float(np.median([r[nm]['err_yaw_deg'] for r in g])),
                                                mahalanobis_max=max(r[nm]['mahalanobis'] for r in g)) for nm in ('ref', 'new')})
                            for k in sorted({r['spot'] for r in gt}) for g in [[r for r in gt if r['spot'] == k]]}
    print(json.dumps(summ, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(method=__doc__.strip(), summary=summ, frames=rows, gt_frames=gt), indent=1))


if __name__ == '__main__':
    main()
