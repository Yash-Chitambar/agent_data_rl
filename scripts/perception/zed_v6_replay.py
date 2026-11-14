"""zed_v6 on the real ZED capture frames (docs/real/PERCEPTION_BIAS_20261115.md, Part A).

Frames: results/placement/zed_map/frames.json (89 run frames 2026-11-01..13 incl. the 6 mapping captures, and the
18 zed_gt frames). Truth: the case-edge fit 'real' (physical port 2) per frame, and the arm-derived port 2
(results/placement/zed_map/arm_port.json, wrist-camera PnP + FK, n = 41 runs). Pose only: zed_v6 has no gate yet (the
zed_v5 gate statistics do not apply to new features). Mean of the given seeds (as the v5 key does with s0, s1).
Also zed_v5 (the stored estimates) for comparison, and the pose-change error of the six 2026-11-13 mapping poses.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/zed_v6_replay.py --nets zed_v6s0 zed_v6s1 --out results/placement/perception_bias/zed_v6_replay.json
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
MAPS = ['capture_173123_map_real', 'capture_173419_map_x_plus', 'capture_173700_map_x_minus',
        'capture_174030_map_y_plus', 'capture_174453_map_y_minus', 'capture_174911_map_yaw']


def load_net(name, dev):
    from blind_insertion.perception.nets import PoseNet
    ck = torch.load(ROOT / 'runs/perception' / f'{name}.pt', map_location=dev, weights_only=False)
    net = PoseNet(ck.get('n_out', len(ck['scale'])), ck['size'], width=ck.get('width', 32)).to(dev).eval()
    net.load_state_dict(ck['net'])
    return net, ck['scale'].to(dev), tuple(ck['nominal_port_xy']), tuple(ck['zed_crop'])


@torch.no_grad()
def estimate(nets, imgs, dev):
    """imgs (B, 320, 320, 3) uint8 RGB -> (B, 3) x mm, y mm, yaw deg (mean over nets) and the per-net spread."""
    from blind_insertion.perception.zed_estimator import zed_crop
    E = []
    for net, scale, nominal, crop in nets:
        x = zed_crop(torch.as_tensor(imgs, device=dev), crop)
        e = net(x)[:, :3] * scale
        E.append(torch.cat([e[:, :2] + torch.tensor(nominal, device=dev) * 1e3, e[:, 2:]], 1).cpu().numpy())
    E = np.stack(E)
    return E.mean(0), np.linalg.norm(E[..., :2] - E.mean(0)[..., :2], axis=-1).max(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nets', nargs='+', default=['zed_v6s0', 'zed_v6s1'])
    ap.add_argument('--frames', default=str(ROOT / 'results/placement/zed_map/frames.json'))
    ap.add_argument('--arm', default=str(ROOT / 'results/placement/zed_map/arm_port.json'))
    ap.add_argument('--twins', nargs='*', default=[], help='twin json + npy dir pairs: name=json:dir (sim check)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda:0'
    nets = [load_net(n, dev) for n in a.nets]
    from blind_insertion.scenes.physical_rig import real_top_policy_image
    F = json.loads(Path(a.frames).read_text())['frames']
    imgs = []
    for f in F:
        if f['png']:
            imgs.append(cv2.cvtColor(cv2.imread(f['png']), cv2.COLOR_BGR2RGB))
        else:
            imgs.append(real_top_policy_image(cv2.imread(f['jpg']), 320))
    E, spread = estimate(nets, np.stack(imgs), dev)
    arm = {(r['date'], r['run']): r for r in json.loads(Path(a.arm).read_text())['runs']}
    rows = []
    for f, e, s in zip(F, E, spread):
        er = f['edge_real']
        row = dict(src=f['src'], date=f['date'], run=f['run'], v6=e.round(3).tolist(), v6_seed_spread_mm=round(float(s), 3),
                   edge=er['xy_mm'] + [er['yaw_deg']], v5=f['v5']['xy_mm'] + [f['v5']['yaw_deg']], v1=f['v1']['xy_mm'] + [f['v1']['yaw_deg']])
        ar = arm.get((f['date'], f['run']))
        if ar:
            row['arm'] = ar['port_mm'][:2] + [ar['yaw_deg']]
        rows.append(row)

    def stats(sel, ref, key):
        R = [r for r in rows if sel(r) and ref in r]
        if not R:
            return None
        d = np.array([np.array(r[key]) - np.array(r[ref]) for r in R])
        return dict(n=len(R), dx_med=round(float(np.median(d[:, 0])), 2), dy_med=round(float(np.median(d[:, 1])), 2),
                    dyaw_med=round(float(np.median(d[:, 2])), 2), dx_range=[round(float(d[:, 0].min()), 2), round(float(d[:, 0].max()), 2)],
                    dy_range=[round(float(d[:, 1].min()), 2), round(float(d[:, 1].max()), 2)],
                    dxy_p50=round(float(np.median(np.linalg.norm(d[:, :2], axis=1))), 2), dxy_max=round(float(np.linalg.norm(d[:, :2], axis=1).max()), 2))
    groups = {'all_run_frames': lambda r: r['src'] == 'run', 'zed_gt': lambda r: r['src'] == 'zed_gt'}
    for d in sorted({r['date'] for r in rows if r['src'] == 'run'}):
        groups[d] = (lambda dd: (lambda r: r['src'] == 'run' and r['date'] == dd and r['run'] not in MAPS))(d)
    groups['map_poses'] = lambda r: r['run'] in MAPS
    summ = {}
    for g, sel in groups.items():
        summ[g] = {k: {ref: stats(sel, ref, k) for ref in ('edge', 'arm')} for k in ('v6', 'v5', 'v1')}
    # pose-change error of the mapping poses (estimate change minus edge change, from map_real)
    M = {r['run']: r for r in rows if r['run'] in MAPS}
    if MAPS[0] in M:
        b = M[MAPS[0]]
        summ['map_change_error'] = {k: {m.split('_', 2)[-1]: (np.array(M[m][k]) - np.array(b[k]) - (np.array(M[m]['edge']) - np.array(b['edge']))).round(2).tolist()
                                        for m in MAPS[1:] if m in M} for k in ('v6', 'v5', 'v1')}
    # slope against the edge truth (all run frames)
    R = [r for r in rows if r['src'] == 'run']
    T = np.array([r['edge'] for r in R])
    for k in ('v6', 'v5', 'v1'):
        X = np.array([r[k] for r in R])
        summ.setdefault('slope', {})[k] = [round(float(np.polyfit(T[:, j], X[:, j], 1)[0]), 3) for j in range(3)]
    summ['seed_spread_mm_p50_max'] = [round(float(np.median(spread)), 2), round(float(spread.max()), 2)]
    sims = {}
    for spec in a.twins:
        name, rest = spec.split('=')
        js, dr = rest.split(':')
        tw = np.load(Path(dr) / 'twin_rgb.npy')
        tr = np.array([r['truth'] for r in json.loads(Path(js).read_text())['rows']])[:len(tw)]
        Es, _ = estimate(nets, tw, dev)
        d = Es - tr
        sims[name] = dict(n=len(tw), dx_med=round(float(np.median(d[:, 0])), 2), dy_med=round(float(np.median(d[:, 1])), 2), dyaw_med=round(float(np.median(d[:, 2])), 2),
                          dxy_p50=round(float(np.median(np.linalg.norm(d[:, :2], axis=1))), 2))
    summ['sim_twins'] = sims
    print(json.dumps(summ, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(method=__doc__.strip(), nets=a.nets, summary=summ, rows=rows), indent=0))


if __name__ == '__main__':
    main()
