"""Real ZED stereo frames (data/zed_gt, the only real frames with the right image): FoundationPose vs the case-edge fit.

Mesh = geometry.pc_real_mesh (real case 103.5 x 103 x 44.5 mm, port 2 31.95 mm from the left edge). Prior = the v1 net
reading logged in data/zed_gt/score_spot*.json (else the spots.json placement): it only picks the SAM 3 detection and
resolves the 4-fold near-symmetry. The edge fit (fit_pc_edges, geometry 'real') starts from the same prior. Per frame:
FP pose, FP - edge fit, checks, SAM 3 mask IoU against the edge-fit pose render. Also a run without prior
(SAM 3 text only, raw FP yaw) to show the raw symmetry behaviour.

    CUDA_VISIBLE_DEVICES=1 /home/yashchitambar/fp_work/fp_python scripts/perception/zed_fp/eval_real.py \
        --out results/perception/fp/real_zed_gt.json --vis results/perception/fp/real_vis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from zed_fp import geometry as G            # noqa: E402
from zed_fp.pipeline import Pipeline, overlay  # noqa: E402
import fit_pc_edges as EF                     # noqa: E402

ZG = Path('/home/yashchitambar/peg_test/data/zed_gt')


def frames(root=ZG, init=None):
    """Stereo pairs <dir>/<tag>_zed_left.png / _zed_right.png under root (data/zed_gt layout, also scripts/real/snapshot.py
    output). Prior: score_spot*.json (v1 net), else spots.json, else `init` (x m, y m, yaw deg, e.g. the tape pose)."""
    root = Path(root)
    spots = json.loads((root / 'spots.json').read_text()) if (root / 'spots.json').exists() else {}
    net = {}
    for f in root.glob('score_spot*.json'):
        for e in json.loads(f.read_text()):
            net[(Path(e['dir']).name, e['tag'])] = (e['port_xy'][0], e['port_xy'][1], e['yaw_deg'])
    out = []
    for d in [root] + sorted(p for p in root.iterdir() if p.is_dir()):
        for L in sorted(d.glob('*_zed_left.png')):
            tag = L.name[:-len('_zed_left.png')]
            R = d / f'{tag}_zed_right.png'
            if not R.exists():
                continue
            s = spots.get(d.name)
            prior = net.get((d.name, tag)) or ((s['port_xy'][0], s['port_xy'][1], s['yaw_deg']) if s else init)
            if prior is None:
                continue
            out.append(dict(dir=d.name, tag=tag, left=L, right=R, prior=prior,
                            prior_src='v1 net (score file)' if (d.name, tag) in net else ('spots.json' if s else 'init')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/perception/fp/real_zed_gt.json')
    ap.add_argument('--root', default=str(ZG))
    ap.add_argument('--init', nargs=3, type=float, default=None, help='prior x m, y m, yaw deg when the folder has none')
    ap.add_argument('--vis', default='')
    ap.add_argument('--yaw-step', type=float, default=30.0)
    ap.add_argument('--iterations', type=int, default=5)
    ap.add_argument('--hyp', default='upright', choices=('upright', 'prior', 'full'))
    ap.add_argument('--no-plane-fix', action='store_true')
    ap.add_argument('--no-prior-run', action='store_true')
    a = ap.parse_args()
    P = Pipeline(mesh='real', hyp=a.hyp, yaw_step=a.yaw_step, iterations=a.iterations, plane_fix=not a.no_plane_fix)
    cam = EF.camera()
    rows = []
    if a.vis:
        Path(a.vis).mkdir(parents=True, exist_ok=True)
    for fr in frames(a.root, a.init):
        L, R = cv2.imread(str(fr['left'])), cv2.imread(str(fr['right']))
        prior = fr['prior']
        row = dict(dir=fr['dir'], tag=fr['tag'], prior=list(prior), prior_src=fr['prior_src'])
        f = EF.fit(L, prior, cam=cam, geom=EF.GEOMS['real'])
        ex, ey, eyaw = f['pose']
        row['edge_fit'] = dict(pose=[ex, ey, eyaw], inlier_rms_px=f['inlier_rms_px'],
                               inl={k: v['inl'] for k, v in f['edges'].items()})
        Te = G.pose_T(ex, ey, eyaw)
        me = G.mesh_mask(P.mesh, np.linalg.inv(P.T_cam) @ Te, P.K, L.shape[:2])
        r = P.run(L, R, prior=prior, keep=True)
        if r.get('ok'):
            x, y, yaw, z, tilt = r['pose']
            row['fp'] = {k: r[k] for k in ('pose', 'pose_raw', 'quarter_turns', 'score', 'score_2nd_other_yaw', 'iou_fp_sam',
                                           'depth_res_mm', 'dz_mm', 'depth_valid_frac', 'time_s', 'fp_s', 'stereo_s', 'gpu_peak_gb')}
            row['fp']['sam'] = {k: r['sam'].get(k) for k in ('prompt', 'score', 'prior_box_iou', 'time_s')}
            row['fp']['sam']['iou_edgefit_render'] = G.iou(r['_mask'], me)
            row['fp']['iou_fp_render_vs_edgefit_render'] = G.iou(r['_sil'], me)
            row['fp_minus_edge'] = dict(dx_mm=(x - ex) * 1e3, dy_mm=(y - ey) * 1e3, dyaw_deg=yaw - eyaw)
            if a.vis:
                im = overlay(L, r, P.mesh, P.K, P.T_cam)
                cv2.imwrite(str(Path(a.vis) / f"{fr['dir']}_{fr['tag']}.jpg"), im)
        else:
            row['fp'] = dict(ok=False, reason=r.get('reason'), sam=r.get('sam'))
        if not a.no_prior_run:
            r0 = P.run(L, R, prior=None)
            row['fp_noprior'] = dict(ok=r0.get('ok'), pose_raw=r0.get('pose_raw'), score=r0.get('score'),
                                     sam_prompt=(r0.get('sam') or {}).get('prompt'), sam_score=(r0.get('sam') or {}).get('score'))
        rows.append(row)
        d = row.get('fp_minus_edge', {})
        print(fr['dir'], fr['tag'], 'fp-edge dx %.2f dy %.2f dyaw %.2f' % (d.get('dx_mm', np.nan), d.get('dy_mm', np.nan), d.get('dyaw_deg', np.nan)),
              'k', row['fp'].get('quarter_turns'), 'score %.1f' % row['fp'].get('score', np.nan),
              'dz %.1f tilt %.2f' % (row['fp'].get('dz_mm', np.nan), (row['fp'].get('pose') or [0] * 5)[4]),
              'iouSAM-edge %.3f' % (row['fp'].get('sam') or {}).get('iou_edgefit_render', np.nan) if row['fp'].get('sam') else '',
              flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(args=vars(a), n=len(rows), load_s=P.load_s, rows=rows), indent=1, default=float))


if __name__ == '__main__':
    main()
