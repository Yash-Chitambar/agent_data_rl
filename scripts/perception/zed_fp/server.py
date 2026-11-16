"""FoundationPose server / batch for the zed_fp capture block (fp env; docs/real/FOUNDATIONPOSE_20261116.md).

Finds <root>/**/zed_fp/<tag>/request.json without result.json, runs the pipeline on left.png / right.png with the
request's prior, writes result.json (+ overlay.jpg, mask.png). Models load once.

    # after the session, on mason (batch: one pass, then a summary table)
    CUDA_VISIBLE_DEVICES=1 /home/yashchitambar/fp_work/fp_python scripts/perception/zed_fp/server.py \
        --root /home/yashchitambar/peg_test/data/real_runs/2026-11-17 --once --summary results/perception/fp/rig_20261117.json
    # live (the zed_fp block with server 'wait'): keep running next to the rig process
    .../fp_python scripts/perception/zed_fp/server.py --root data/real_runs
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from zed_fp.pipeline import Pipeline, overlay  # noqa: E402

ROOT = HERE.parents[2]
SIM_MESH = ROOT / 'results/perception/fp/mesh_pc_sim.obj'


def make_pipelines(geoms, yaw_step):
    P = {}
    if 'real' in geoms:
        P['real'] = Pipeline(mesh='real', yaw_step=yaw_step)
    if 'sim' in geoms:
        meta = json.loads(SIM_MESH.with_suffix('.json').read_text())
        P['sim'] = Pipeline(mesh=str(SIM_MESH), R_up=np.array(meta['R_up']), z_top=meta['z_origin'], yaw_step=yaw_step)
    return P


def _is_pc(p):
    try:
        return json.loads(p.read_text()).get('object', 'pc') == 'pc'
    except (OSError, json.JSONDecodeError):
        return False


def pending(root):
    return sorted(p.parent for p in Path(root).glob('**/zed_fp/*/request.json')
                  if not (p.parent / 'result.json').exists() and _is_pc(p))


def process(P, d, all_pairs=False):
    req = json.loads((d / 'request.json').read_text())
    if req.get('object', 'pc') != 'pc':          # reader frames: saved for a later reader fit (no reader mesh yet)
        return None
    geom = req.get('geom', 'real')
    if geom not in P:
        return None
    L, R = cv2.imread(str(d / 'left.png')), cv2.imread(str(d / 'right.png'))
    t0 = time.time()
    try:
        r = P[geom].run(L, R, prior=req.get('prior'), keep=True)
    except Exception as e:  # noqa: BLE001
        r = dict(ok=False, reason=f'exception: {e!r}', trace=traceback.format_exc())
    if r.get('ok'):
        cv2.imwrite(str(d / 'overlay.jpg'), overlay(L, r, P[geom].mesh, P[geom].K, P[geom].T_cam))
        cv2.imwrite(str(d / 'mask.png'), r['_mask'] * 255)
    out = {k: v for k, v in r.items() if not k.startswith('_')}
    if all_pairs:                       # extra pairs of the same capture (zed_fp n_pairs): per-pair poses and the median
        poses = [out['pose']] if out.get('ok') else []
        for i in range(1, 50):
            if not (d / f'left_{i}.png').exists():
                break
            ri = P[geom].run(cv2.imread(str(d / f'left_{i}.png')), cv2.imread(str(d / f'right_{i}.png')), prior=req.get('prior'))
            if ri.get('ok'):
                poses.append(ri['pose'])
        if poses:
            out['pairs'] = poses
            out['pose_median'] = np.median(np.array(poses), 0).tolist()
    out.update(geom=geom, wall_s=time.time() - t0, processed=time.strftime('%Y-%m-%d %H:%M:%S'))
    tmp = d / 'result.json.tmp'
    tmp.write_text(json.dumps(out, indent=1, default=float))
    tmp.rename(d / 'result.json')
    return req, out


def row(d, req, out):
    r = dict(dir=str(d), tag=req.get('tag'), geom=req.get('geom'), ok=out.get('ok'), reason=out.get('reason'))
    if out.get('ok'):
        x, y, yaw = out['pose'][:3]
        if out.get('pose_median'):
            x, y, yaw = out['pose_median'][:3]
            r['n_pairs_ok'] = len(out['pairs'])
        r.update(fp=[x, y, yaw], score=out['score'], iou=out['iou_fp_sam'], dz_raw_mm=out.get('dz_mm'),
                 k=out.get('quarter_turns'), time_s=out.get('time_s'), gpu_gb=out.get('gpu_peak_gb'))
        n = req.get('net') or {}
        if n.get('port_xy'):
            r['fp_minus_net_mm'] = ((np.array([x, y]) - n['port_xy']) * 1e3).tolist() + [yaw - n['yaw_deg']]
        e = req.get('edge_fit') or {}
        if e.get('fit_xy'):
            r['fp_minus_edge_mm'] = ((np.array([x, y]) - e['fit_xy']) * 1e3).tolist() + [yaw - e['fit_yaw_deg']]
        if req.get('true_port_xy'):
            r['fp_minus_truth_mm'] = ((np.array([x, y]) - req['true_port_xy']) * 1e3).tolist() + [yaw - req['true_yaw_deg']]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--geoms', default='real,sim')
    ap.add_argument('--yaw-step', type=float, default=30.0)
    ap.add_argument('--summary', default='')
    ap.add_argument('--all-pairs', action='store_true', help='batch: also run left_<i>/right_<i> (n_pairs) and log the median')
    ap.add_argument('--redo', action='store_true', help='also redo folders that have a result.json (batch only)')
    a = ap.parse_args()
    P = make_pipelines(a.geoms.split(','), a.yaw_step)
    print(f'zed_fp server: models loaded in {sum(p.load_s for p in P.values()):.1f} s; watching {a.root}', flush=True)
    rows = []
    while True:
        todo = pending(a.root)
        if a.redo and a.once:
            todo = sorted(p.parent for p in Path(a.root).glob('**/zed_fp/*/request.json'))
        for d in todo:
            res = process(P, d, a.all_pairs)
            if res is None:
                continue
            rows.append(row(d, *res))
            r = rows[-1]
            print(d, 'ok' if r['ok'] else f"FAILED {r['reason']}", r.get('fp'), 'edge', r.get('fp_minus_edge_mm'),
                  'truth', r.get('fp_minus_truth_mm'), flush=True)
        if a.once:
            break
        time.sleep(0.1)
    if a.summary:
        Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary).write_text(json.dumps(dict(root=a.root, n=len(rows), rows=rows), indent=1, default=float))


if __name__ == '__main__':
    main()
