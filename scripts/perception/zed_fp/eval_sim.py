"""Sim truth check of the ZED FoundationPose pipeline on fp_sim_render.py cases (fp env).

Per case, three variants (same FP model, mesh = the sim PC geoms, mesh_pc_sim.obj):
  oracle: render depth + truth mask;  stereo: SGBM depth + truth mask;  full: SGBM depth + SAM 3 mask (the rig pipeline).
Prior (selects the SAM 3 detection and resolves the 4-fold near-symmetry): truth + N(0, prior_sd_mm) in x, y and
N(0, prior_sd_deg) in yaw (zed_v5-level error). The case-edge fit (fit_pc_edges, geometry 'sim') runs from the same
prior for comparison; IoU of the SAM 3 mask against the truth mask and against the edge-fit pose render.

    CUDA_VISIBLE_DEVICES=1 /home/yashchitambar/fp_work/fp_python scripts/perception/zed_fp/eval_sim.py \
        --cases /home/yashchitambar/fp_work/sim --out results/perception/fp/sim_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from zed_fp import geometry as G            # noqa: E402
from zed_fp.pipeline import Pipeline, overlay  # noqa: E402
import fit_pc_edges as EF                     # noqa: E402


def err(T_est, T_true):
    dR = T_est[:3, :3] @ T_true[:3, :3].T
    d = (T_est[:3, 3] - T_true[:3, 3]) * 1e3
    return dict(dx_mm=float(d[0]), dy_mm=float(d[1]), dz_mm=float(d[2]), dxy_mm=float(np.hypot(d[0], d[1])),
                dyaw_deg=float(math.degrees(math.atan2(dR[1, 0], dR[0, 0]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default='/home/yashchitambar/fp_work/sim')
    ap.add_argument('--out', default='results/perception/fp/sim_eval.json')
    ap.add_argument('--prior-sd-mm', type=float, default=5.0)
    ap.add_argument('--prior-sd-deg', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--yaw-step', type=float, default=30.0)
    ap.add_argument('--iterations', type=int, default=5)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--vis', default='')
    a = ap.parse_args()
    root = Path(a.cases)
    cases = sorted(p for p in root.glob('case_*') if (p / 'truth.json').exists())
    if a.limit:
        cases = cases[:a.limit]
    t0 = json.loads((cases[0] / 'truth.json').read_text())
    T0 = np.array(t0['T_pc_in_base'])
    R_up = G.pose_T(0, 0, -t0['pc_port_xy_yaw'][2])[:3, :3] @ T0[:3, :3]
    P = Pipeline(mesh=str(root / 'mesh_pc_sim.obj'), R_up=R_up, z_top=float(T0[2, 3]), yaw_step=a.yaw_step,
                 iterations=a.iterations)
    rng = np.random.default_rng(a.seed)
    cam = EF.camera()
    rows = []
    if a.vis:
        Path(a.vis).mkdir(parents=True, exist_ok=True)
    for c in cases:
        t = json.loads((c / 'truth.json').read_text())
        Tt = np.array(t['T_pc_in_base'])
        x, y, yaw = t['pc_port_xy_yaw']
        prior = (x + rng.normal(0, a.prior_sd_mm * 1e-3), y + rng.normal(0, a.prior_sd_mm * 1e-3), yaw + rng.normal(0, a.prior_sd_deg))
        L, R = cv2.imread(str(c / 'left.png')), cv2.imread(str(c / 'right.png'))
        dep = np.load(c / 'depth_left.npy')
        mt = G.largest_cc(cv2.imread(str(c / 'mask_pc.png'), 0))
        row = dict(case=c.name, truth=[x, y, yaw], prior=list(prior))
        sam_mask = None
        for name, kw in (('oracle', dict(depth=dep, mask=mt)), ('stereo', dict(mask=mt)), ('full', dict(keep=True))):
            r = P.run(L, R, prior=prior, **kw)
            if r.get('ok'):
                T = np.array(r['T_obj_in_base'])
                Traw = None
                r_out = {k: r[k] for k in ('score', 'score_2nd_other_yaw', 'quarter_turns', 'iou_fp_sam', 'depth_res_mm',
                                           'dz_mm', 'time_s', 'fp_s', 'gpu_peak_gb', 'depth_valid_frac', 'pose', 'pose_raw')
                         if k in r}
                r_out.update(err(T, Tt))
                r_out['stereo_s'] = r.get('stereo_s')
            else:
                r_out = dict(ok=False, reason=r.get('reason'))
            if 'sam' in r and isinstance(r['sam'], dict) and r['sam'].get('mode') != 'given':
                s = r['sam']
                r_out['sam'] = {k: s.get(k) for k in ('prompt', 'score', 'prior_box_iou', 'time_s', 'n_det')}
                if '_mask' in r:
                    r_out['sam']['iou_truth'] = G.iou(r['_mask'], mt)
                    sam_mask = r['_mask']
            if name == 'full' and a.vis and r.get('ok'):
                cv2.imwrite(str(Path(a.vis) / f'{c.name}.jpg'), overlay(L, r, P.mesh, P.K, P.T_cam))
            row[name] = r_out
        # case-edge fit from the prior (sim geometry), and the SAM mask against its pose render
        try:
            f = EF.fit(L, prior, cam=cam, geom=EF.GEOMS['sim'])
            fx, fy, fyaw = f['pose']
            row['edge_fit'] = dict(dx_mm=(fx - x) * 1e3, dy_mm=(fy - y) * 1e3, dyaw_deg=fyaw - yaw,
                                   inlier_rms_px=f['inlier_rms_px'], inl={k: v['inl'] for k, v in f['edges'].items()})
            Tf = G.pose_T(fx, fy, fyaw, z=float(Tt[2, 3])); Tf[:3, :3] = Tf[:3, :3] @ R_up
            mf = G.mesh_mask(P.mesh, np.linalg.inv(P.T_cam) @ Tf, P.K, L.shape[:2])
            mtr = G.mesh_mask(P.mesh, np.linalg.inv(P.T_cam) @ Tt, P.K, L.shape[:2])
            row['edge_fit']['iou_render_vs_truth_render'] = G.iou(mf, mtr)
            if sam_mask is not None:
                row['full']['sam']['iou_edgefit_render'] = G.iou(sam_mask, mf)
            row['iou_truthmask_vs_truth_render'] = G.iou(mt, mtr)
        except Exception as e:  # noqa: BLE001
            row['edge_fit'] = dict(error=repr(e))
        rows.append(row)
        fu = row['full']
        print(c.name, 'oracle %.2f mm %.2f deg' % (row['oracle'].get('dxy_mm', np.nan), row['oracle'].get('dyaw_deg', np.nan)),
              '| stereo %.2f %.2f' % (row['stereo'].get('dxy_mm', np.nan), row['stereo'].get('dyaw_deg', np.nan)),
              '| full %.2f %.2f k%s score %.1f iouSAM %.3f' % (fu.get('dxy_mm', np.nan), fu.get('dyaw_deg', np.nan), fu.get('quarter_turns'),
                                                              fu.get('score', np.nan), (fu.get('sam') or {}).get('iou_truth') or np.nan), flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(a), n=len(rows), load_s=P.load_s, rows=rows), indent=1, default=float))


if __name__ == '__main__':
    main()
