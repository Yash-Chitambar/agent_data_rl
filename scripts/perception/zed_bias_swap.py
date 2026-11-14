"""Where does the zed_v5 real-frame offset come from? Image swaps between each real capture frame and its sim twin
(scripts/perception/zed_bias_render.py, same PC pose and arm angles). docs/real/PERCEPTION_BIAS_20261115.md, Part A.

Per frame (n = 89 run frames):
  * PC swap: the PC region (case top + panel + sides, projected through the deploy ZED calibration from the edge fit,
    dilated) is copied from the real frame into the twin, and the other way round.
  * Patch map: the v5 crop is cut into 16 px patches; each patch of the real frame is replaced by the twin's patch, one
    at a time; the change of the v5 estimate is recorded (which image parts carry the real-sim offset).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/zed_bias_swap.py --twin-dir <dir with twin_rgb.npy> --twin-json <twin json> --out results/placement/perception_bias/swap.json
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

TABLE_Z = -0.033


def to_policy(uv, cam_spec=None):
    s = json.loads(F.SPEC.read_text())['top']['intrinsics_rectified']['HD1080']
    cx, cy = round(s['cx']), round(s['cy'])
    half = min(cx, s['resolution'][0] - cx, cy, s['resolution'][1] - cy)
    return (uv - [cx - half, cy - half]) * 320 / (2 * half)


def pc_mask(pose_real, pose_sim, cam, dil=3):
    m = np.zeros((320, 320), np.uint8)
    for pose, g in ((pose_real, F.GEOMS['real']), (pose_sim, F.GEOMS['sim'])):
        p = (pose[0] / 1e3, pose[1] / 1e3, math.radians(pose[2]))
        top = F.corners(p, **g)
        bot = top.copy(); bot[:, 2] = TABLE_Z
        uv = to_policy(F.project(np.r_[top, bot], cam))
        cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
    return cv2.dilate(m, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--twin-dir', required=True)
    ap.add_argument('--twin-json', required=True)
    ap.add_argument('--frames', default=str(ROOT / 'results/placement/zed_map/frames.json'))
    ap.add_argument('--patch', type=int, default=16)
    ap.add_argument('--pose-nets', nargs='*', default=None, help='pose nets by name (e.g. zed_v6s0 zed_v6s1) instead of zed_v5')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from blind_insertion.perception.zed_estimator import ZedEstimator
    dev = 'cuda:0'
    v5 = ZedEstimator(dev, presence=str(ROOT / 'runs/perception/zed_v5_gate.pt'))
    tw = np.load(Path(a.twin_dir) / 'twin_rgb.npy')
    rows_t = json.loads(Path(a.twin_json).read_text())['rows']
    fr = {(f['run'], f['date']): f for f in json.loads(Path(a.frames).read_text())['frames'] if f['src'] == 'run'}
    cam = F.camera()

    if a.pose_nets:
        from zed_v6_replay import load_net, estimate
        pn = [load_net(nm, dev) for nm in a.pose_nets]
        x0_, y0_, s_ = pn[0][3]

        def est(imgs):
            return np.concatenate([estimate(pn, imgs[i:i + 512], dev)[0] for i in range(0, len(imgs), 512)])
        v5.crop = (x0_, y0_, s_)
    else:
        est = None

    def est5(imgs):
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), 512):
                p, y, ok, _ = v5(torch.tensor(np.ascontiguousarray(imgs[i:i + 512]), device=dev))
                out.append(np.c_[p.cpu().numpy() * 1e3, np.degrees(y.cpu().numpy())])
        return np.concatenate(out)
    est = est or est5

    x0, y0, s = v5.crop
    P = a.patch
    grid = [(u, v) for v in range(y0, y0 + s, P) for u in range(x0, x0 + s, P)]
    res, pmap = [], np.zeros((len(grid), 3))
    for i, rt in enumerate(rows_t):
        f = fr[(rt['run'], rt['date'])]
        real = cv2.cvtColor(cv2.imread(f['png']), cv2.COLOR_BGR2RGB)   # the RGB policy image the net saw (zed_map.py)
        twin = tw[i]
        m = pc_mask(f['edge_real']['xy_mm'] + [f['edge_real']['yaw_deg']], f['edge_sim']['xy_mm'] + [f['edge_sim']['yaw_deg']], cam)[..., None]
        imgs = [real, twin, twin * (1 - m) + real * m, real * (1 - m) + twin * m]
        for (u, v) in grid:
            r2 = real.copy(); r2[v:v + P, u:u + P] = twin[v:v + P, u:u + P]; imgs.append(r2)
        E = est(np.stack(imgs).astype(np.uint8))
        res.append(dict(run=rt['run'], date=rt['date'], truth=rt['truth'], real=E[0].tolist(), twin=E[1].tolist(),
                        twin_with_real_pc=E[2].tolist(), real_with_twin_pc=E[3].tolist(), mask_px=int(m.sum())))
        pmap += E[4:] - E[0]
    pmap /= len(rows_t)
    T = np.array([r['truth'] for r in res])
    summ = {k: np.median(np.array([r[k] for r in res]) - T, 0).round(2).tolist() for k in ('real', 'twin', 'twin_with_real_pc', 'real_with_twin_pc')}
    summ['n'] = len(res)
    print(json.dumps(summ))
    n = s // P
    print('patch map: mean change of y (mm) when one real patch is replaced by the twin patch (rows = image v)')
    for j in range(n):
        print(' '.join(f'{pmap[j * n + k, 1]:+5.2f}' for k in range(n)))
    print('same for x')
    for j in range(n):
        print(' '.join(f'{pmap[j * n + k, 0]:+5.2f}' for k in range(n)))
    Path(a.out).write_text(json.dumps(dict(summary=summ, crop=[x0, y0, s], patch=P, patch_map_dxdydyaw=pmap.round(3).tolist(), rows=res), indent=0))
    # figure: patch map over a real frame
    base = cv2.imread(fr[(res[-1]['run'], res[-1]['date'])]['png'])[y0:y0 + s, x0:x0 + s]
    big = cv2.resize(base, (s * 3, s * 3), interpolation=cv2.INTER_NEAREST)
    for k, (u, v) in enumerate(grid):
        val = pmap[k, 1]
        if abs(val) >= .15:
            col = (0, 0, 255) if val > 0 else (255, 0, 0)
            cv2.putText(big, f'{val:+.1f}', ((u - x0) * 3 + 4, (v - y0) * 3 + 28), cv2.FONT_HERSHEY_SIMPLEX, .5, col, 1)
        cv2.rectangle(big, ((u - x0) * 3, (v - y0) * 3), ((u - x0 + P) * 3, (v - y0 + P) * 3), (80, 80, 80), 1)
    cv2.imwrite(str(Path(a.out).with_suffix('.png')), big)


if __name__ == '__main__':
    main()
