"""Offline replay of the capture-stage edge refine (deploy/bimanual/zed_edgefit.py) on real ZED capture frames.

docs/real/ZED_EDGEFIT_20261115.md. Per frame (every real run with a raw HD1080 capture frame, zed_map.run_frames):
the gate net (v1: zed + zed_v1_gate_arm, today's rig estimator) and the start net (zed_v5_gate) on the logged policy
image with the logged arm angles, then zed_edgefit.refine on the raw JPEG with a config block (default: the block of
deploy/config/bimanual_align10_all_edgefit.json). Truth: the arm-derived port 2 per run
(results/placement/zed_map/arm_port.json, scripts/perception/zed_arm_port.py; physical port), where it exists.
Also replays the start from v1 (start_gate '') to show the convergence does not depend on the start net.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/zed_edgefit_replay.py --out results/placement/zed_edgefit/replay.json
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
import zed_map as ZM  # noqa: E402
from deploy.bimanual import zed_edgefit as ZE  # noqa: E402

DATES = ['2026-11-01', '2026-11-02', '2026-11-03', '2026-11-06', '2026-11-10', '2026-11-11', '2026-11-13']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align10_all_edgefit.json'))
    ap.add_argument('--dates', nargs='+', default=DATES)
    ap.add_argument('--arm', default=str(ROOT / 'results/placement/zed_map/arm_port.json'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from blind_insertion.perception.zed_estimator import ZedEstimator
    cfg = json.loads(Path(a.config).read_text())
    c = ZE.parse(cfg)
    assert c is not None, 'config has no enabled zed_edge_refine block'
    dev = 'cuda:0'
    P = ROOT / 'runs/perception'
    gate = ZedEstimator(dev, pose=str(P / 'zed.pt'), presence=str(P / f"{cfg['chain_args']['zed_gate']}.pt"))
    start = ZedEstimator(dev, pose=str(P / 'zed.pt'), presence=str(P / f"{c['start_gate']}.pt"))
    arm = {r['run']: r for r in json.loads(Path(a.arm).read_text())['runs']}
    rows = []
    for fr in ZM.run_frames(a.dates):
        raw = cv2.imread(fr['jpg'])
        img = cv2.cvtColor(cv2.imread(fr['png']), cv2.COLOR_BGR2RGB)
        x = torch.tensor(img[None], device=dev)
        lg = fr['logged']
        arms = None
        if lg and lg.get('arms_b'):
            ql, qr, fo = lg['arms_b']
            arms = (torch.tensor([ql], device=dev), torch.tensor([qr], device=dev), torch.tensor([float(fo)], device=dev))
        nets = {}
        for name, est in (('gate', gate), ('start', start)):
            p, y, ok, info = est(x, arms=arms)
            nets[name] = dict(xy=p[0].cpu().numpy().astype(float), yaw=math.degrees(float(y[0])), ok=bool(ok[0]),
                              info={k: float(v[0]) for k, v in info.items()})
        g, s = nets['gate'], nets['start']
        xy, yaw, ok_fit, rec = ZE.refine(raw, 'raw_jpeg_replay', s['xy'], s['yaw'], g['xy'], g['yaw'], c, c['geom'])
        _, _, ok_v1s, rec_v1s = ZE.refine(raw, 'raw_jpeg_replay', g['xy'], g['yaw'], g['xy'], g['yaw'], c, c['geom'])
        row = dict(date=fr['date'], run=fr['run'], jpg=fr['jpg'],
                   gate=dict(xy_mm=(g['xy'] * 1e3).tolist(), yaw_deg=g['yaw'], ok=g['ok'], **g['info']),
                   start=dict(xy_mm=(s['xy'] * 1e3).tolist(), yaw_deg=s['yaw'], ok=s['ok'], **s['info']),
                   fit=rec, fit_from_v1=dict(ok=ok_v1s, fit_xy=rec_v1s.get('fit_xy'), fit_yaw_deg=rec_v1s.get('fit_yaw_deg'),
                                             why=rec_v1s['why']),
                   used=dict(xy_mm=(np.asarray(xy) * 1e3).tolist(), yaw_deg=yaw),
                   ok=bool(g['ok'] and ok_fit))
        if fr['run'] in arm:
            t = arm[fr['run']]
            row['arm'] = dict(xy_mm=t['port_mm'][:2], yaw_deg=t['yaw_deg'])
        rows.append(row)
        f = rec
        print(f"{fr['date']} {fr['run'][:28]:28s} v1 ({row['gate']['xy_mm'][0]:6.1f},{row['gate']['xy_mm'][1]:6.1f}) {g['ok']:d} "
              f"v5 ({row['start']['xy_mm'][0]:6.1f},{row['start']['xy_mm'][1]:6.1f}) {s['ok']:d} | "
              f"fit ({f['fit_xy'][0] * 1e3:6.1f},{f['fit_xy'][1] * 1e3:6.1f},{f['fit_yaw_deg']:5.2f}) {f['edges_used']} rms {f['inlier_rms_px']:.2f} "
              f"ok {ok_fit:d} {f['seconds']:.2f}s" + (f" | arm ({row['arm']['xy_mm'][0]:6.1f},{row['arm']['xy_mm'][1]:6.1f})" if 'arm' in row else '')
              + ('' if ok_fit else f"  why {f['why']}"), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(method=__doc__.strip(), config=a.config, block=c, frames=rows), indent=1, default=float))


if __name__ == '__main__':
    main()
