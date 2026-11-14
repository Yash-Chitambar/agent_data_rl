"""ZED PC nets against the case-edge fit (and the arm) on real frames: docs/real/ZED_MAP_20261114.md.

Frames: every real ZED capture frame with a raw HD1080 image (frames/top/*_zed_pc.jpg; the 320-px policy png next to it
is what the rig fed the net), the data/zed_gt spots (2026-11-01), and the E1 mapping captures of 2026-11-13.
Per frame: the edge fit (fit_pc_edges.py, geometry 'real' = physical port 2, and 'sim' = the sim label convention),
v1 (zed + zed_v1_gate_arm, the rig's) and zed_v5 (zed_v5_gate key: zed_v5s0/s1), with the logged arm angles for the
arm gate. Overlays of the mapping frames: results/placement/zed_map/*.jpg (small).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/zed_map.py --out results/placement/zed_map/frames.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/perception'))
import fit_pc_edges as F  # noqa: E402

RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
GT = Path('/home/yashchitambar/peg_test/data/zed_gt')
MAPS = ['capture_173123_map_real', 'capture_173419_map_x_plus', 'capture_173700_map_x_minus',
        'capture_174030_map_y_plus', 'capture_174453_map_y_minus', 'capture_174911_map_yaw']
DATES = ['2026-11-01', '2026-11-02', '2026-11-03', '2026-11-06', '2026-11-07', '2026-11-10', '2026-11-11', '2026-11-13']


def run_frames(dates):
    out = []
    for d in dates:
        for run in sorted((RUNS / d).glob('*')):
            meta = run / 'meta.json'
            if not run.is_dir() or not meta.exists() or json.loads(meta.read_text()).get('dry_run', True):
                continue
            ev = []
            if (run / 'events.jsonl').exists():
                for line in (run / 'events.jsonl').read_text().splitlines():
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc':
                        ev.append(e)
            pngs = sorted((run / 'frames' / 'top').glob('*_zed_pc.png'))
            for k, png in enumerate(pngs):
                jpg = png.with_name(f'{int(png.name[:6]) + 1:06d}_zed_pc.jpg')
                if not jpg.exists():
                    continue
                lg = next((e for e in ev if e.get('frame_tag') == f'zed_pc@{k}'), ev[k] if k < len(ev) else None)
                out.append(dict(src='run', date=d, run=run.name, png=str(png), jpg=str(jpg), logged=lg))
    return out


def gt_frames():
    spots = json.loads((GT / 'spots.json').read_text())
    out = []
    for name, sp in spots.items():
        for f in sorted((GT / name).glob('*_zed_left.png')):
            out.append(dict(src='zed_gt', date='2026-11-01', run=f'zed_gt/{name}', tag=f.name[:-len('_zed_left.png')], jpg=str(f), png=None,
                            logged=None, gt=dict(xy_mm=[v * 1e3 for v in sp['port_xy']], yaw_deg=sp['yaw_deg'])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', nargs='+', default=DATES)
    ap.add_argument('--out', required=True)
    ap.add_argument('--overlays', default=str(ROOT / 'results/placement/zed_map'))
    a = ap.parse_args()
    from blind_insertion.perception.zed_estimator import ZedEstimator
    from blind_insertion.scenes.physical_rig import real_top_policy_image
    dev = 'cuda:0'
    P = ROOT / 'runs/perception'
    nets = {'v1': ZedEstimator(dev, pose=str(P / 'zed.pt'), presence=str(P / 'zed_v1_gate_arm.pt')),
            'v5': ZedEstimator(dev, presence=str(P / 'zed_v5_gate.pt'))}
    cam = F.camera()
    rows = []
    for fr in gt_frames() + run_frames(a.dates):
        raw = cv2.imread(fr['jpg'])
        if fr['png']:
            img = cv2.cvtColor(cv2.imread(fr['png']), cv2.COLOR_BGR2RGB)     # runlog writes RGB policy frames as BGR
        else:
            img = real_top_policy_image(raw, 320)
        x = torch.tensor(img[None], device=dev)
        lg = fr['logged']
        arms = None
        if lg and lg.get('arms_b'):
            ql, qr, fo = lg['arms_b']
            arms = (torch.tensor([ql], device=dev), torch.tensor([qr], device=dev), torch.tensor([float(fo)], device=dev))
        row = {k: v for k, v in fr.items() if k not in ('logged',)}
        if lg:
            row['logged'] = dict(xy_mm=[v * 1e3 for v in lg['port_xy']], yaw_deg=lg['yaw_deg'], ok=lg['ok'], mahalanobis=lg.get('mahalanobis'))
        for name, est in nets.items():
            p, y, ok, info = est(x, arms=arms)
            row[name] = dict(xy_mm=(p[0].cpu().numpy() * 1e3).tolist(), yaw_deg=math.degrees(float(y[0])), ok=bool(ok[0]),
                             **{k: float(v[0]) for k, v in info.items()})
        init = row['v1']['xy_mm'] + [row['v1']['yaw_deg']]
        init = (init[0] / 1e3, init[1] / 1e3, init[2])
        for g in ('real', 'sim'):
            r = F.fit(raw, init, cam=cam, geom=F.GEOMS[g])
            row['edge_' + g] = dict(xy_mm=[r['pose'][0] * 1e3, r['pose'][1] * 1e3], yaw_deg=r['pose'][2], sd_stat=[r['sd_stat'][0] * 1e3, r['sd_stat'][1] * 1e3, r['sd_stat'][2]],
                                    inlier_rms_px=r['inlier_rms_px'], edges=r['edges'])
            if g == 'real' and fr['run'] in MAPS:
                ov = F.overlay(raw, r, cam=cam, geom=F.GEOMS[g], label=f"{fr['run'].split('_', 2)[-1]}: ({r['pose'][0]*1e3:.1f}, {r['pose'][1]*1e3:.1f}) mm {r['pose'][2]:.1f} deg")
                Path(a.overlays).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(a.overlays) / f"{fr['run'].split('_', 2)[-1]}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
        rows.append(row)
        e = row['edge_real']
        print(f"{row['date']} {row['run'][:30]:30s} edge ({e['xy_mm'][0]:6.1f},{e['xy_mm'][1]:6.1f},{e['yaw_deg']:5.2f}) rms {e['inlier_rms_px']:.2f} | "
              f"v1 ({row['v1']['xy_mm'][0]:6.1f},{row['v1']['xy_mm'][1]:6.1f},{row['v1']['yaw_deg']:5.2f}) {row['v1']['ok']:d} | "
              f"v5 ({row['v5']['xy_mm'][0]:6.1f},{row['v5']['xy_mm'][1]:6.1f},{row['v5']['yaw_deg']:5.2f}) {row['v5']['ok']:d}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(method=__doc__.strip(), geoms=F.GEOMS, frames=rows), indent=1))


if __name__ == '__main__':
    main()
