"""Sim PC panel colour vs the case-edge fit and the ZED nets (docs/real/SIM_RELEASE_PANEL_20261116.md). Sim only.

One batch of ZED top frames (ApproachRenderer: the dry run's scene, visual DR on, PC poses from the pick layout range),
rendered once per panel RGB (material pc_rear_photo; deploy.bimanual.io_sim.panel_rgb; geometry unchanged). Per frame:
  * the edge fit (deploy/bimanual/zed_edgefit.run_fit, geometry 'sim', block of bimanual_align10_all_edgefit.json) on the
    frame upsampled into its HD1080 crop, started from zed_v5 on the DEFAULT frame (as the dry run: the nets see the
    default look, the fit sees the dark one); error against the sim truth; the fit's quality gate (inliers, rms);
  * v1 (zed_v1_gate_arm) and zed_v5 (zed_v5_gate) on the frame: error against the sim truth, gate.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=. python scripts/perception/sim_panel_edgefit.py --n 64 \
        --rgbs 1 0.5 0.35 0.25 --out results/placement/sim_panel/sweep.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=64)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--rgbs', type=float, nargs='+', default=[1.0, 0.5, 0.35, 0.25])
    ap.add_argument('--vary', type=float, default=0.0, help='ApproachRenderer.sample_top vary (arm poses); 0 = reset pose')
    ap.add_argument('--layout', default='results/usbc/policy/pick/task_layout.json')
    ap.add_argument('--config', default='deploy/config/bimanual_align10_all_edgefit.json')
    ap.add_argument('--top-cam', choices=('dr', 'nominal'), default='dr', help='top camera of every frame: the DR draw or the calibrated pose (nominal_camera)')
    ap.add_argument('--save-images', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import warp as wp
    import mujoco_warp as mjw
    from blind_insertion.perception.render_data import ApproachRenderer
    from deploy.bimanual import netcache as NC
    from deploy.bimanual import zed_edgefit as ZE
    import contextlib
    from deploy.bimanual.io_sim import nominal_camera, panel_rgb
    cfg = json.loads((ROOT / a.config).read_text())
    c = ZE.parse(cfg)
    r = ApproachRenderer(nworld=a.n, seed=a.seed, task_layout=a.layout)
    rt, h, dev = r.rt, r.h, r.rt.device
    h.reset()                                   # arms at BimanualRuntime.initial_q = the ZED capture pose
    if a.vary:
        r.vary_arms(a.vary)
    else:                                       # capture: both grippers fully open (render only, as SimIO.move_to)
        rt.q[:, rt.finger_qadr_right] = .0475
        rt.q[:, r.left_fingers] = .0475
    with wp.ScopedDevice(rt.cfg.device):
        mjw.forward(rt.m, rt.d)
    port_p, port_R = h.pc_pose()
    truth = torch.cat([port_p[:, :2] * 1e3, torch.rad2deg(torch.atan2(-port_R[:, 1, 2], -port_R[:, 0, 2]))[:, None]], 1).cpu().numpy()
    gk = list(rt.visual.groups)
    rc = {g: rt.visual.params['random_colour'][:, gk.index(g)].cpu().numpy() > 0 for g in ('pc_case', 'pc_dark', 'table') if g in gk}
    from blind_insertion.envs.visual_dr import TEXTURED
    rc['table_textured'] = rt.visual.params['textured'][:, TEXTURED.index('table')].cpu().numpy() > 0
    nets = {'v1': NC.zed(dev, 'zed_v1_gate_arm'), 'v5': NC.zed(dev, 'zed_v5_gate')}
    imgs, est = {}, {}
    rt.render()                                 # one photometric draw per world, kept for every panel variant (paired frames)
    rt.visual.cfg.image_per_frame = False
    for rgb in a.rgbs:
        cam = nominal_camera(rt, 'top') if a.top_cam == 'nominal' else contextlib.nullcontext()
        with panel_rgb(rt, [rgb] * 3), cam:
            top = rt.render()['top'][..., :3].contiguous()
        imgs[rgb] = top
        with torch.no_grad():
            est[rgb] = {k: e(top) for k, e in nets.items()}
    base = a.rgbs[0]
    v5p, v5y = est[base]['v5'][0].cpu().numpy(), np.degrees(est[base]['v5'][1].cpu().numpy())
    rows, summ = [], {}
    for rgb in a.rgbs:
        top = imgs[rgb].cpu().numpy().astype(np.uint8)
        R = []
        for i in range(a.n):
            fit, used = ZE.run_fit(ZE.canvas_from_policy(top[i]), (float(v5p[i, 0]), float(v5p[i, 1]), float(v5y[i])), c, c['geom_sim'])
            fx, fy, fyaw = fit['pose']
            q_ok = all(fit['edges'][e]['inl'] >= c['min_inliers'] for e in ZE.REQUIRED_EDGES) and \
                fit['inlier_rms_px'] is not None and fit['inlier_rms_px'] <= c['max_rms_px']
            row = dict(i=i, rgb=rgb, truth=truth[i].round(3).tolist(), fit=[fx * 1e3, fy * 1e3, fyaw], edges_used=used,
                       edges={k: v['inl'] for k, v in fit['edges'].items()}, rms_px=fit['inlier_rms_px'], fit_quality_ok=bool(q_ok),
                       fit_err=[fx * 1e3 - truth[i, 0], fy * 1e3 - truth[i, 1], fyaw - truth[i, 2]],
                       random_colour=[g for g, v in rc.items() if v[i]])
            for k, (p, yw, ok, _) in est[rgb].items():
                row[k] = [*(p[i].cpu().numpy() * 1e3 - truth[i, :2]).tolist(), math.degrees(float(yw[i])) - truth[i, 2], bool(ok[i])]
            R.append(row)
        rows += R
        fe = np.array([x['fit_err'] for x in R])
        dxy = np.hypot(fe[:, 0], fe[:, 1])
        s = dict(n=a.n, fit_quality_ok=int(sum(x['fit_quality_ok'] for x in R)),
                 fit_err_xy_mm_p50_p90_max=np.percentile(dxy, [50, 90, 100]).round(2).tolist(),
                 fit_within_2mm=int((dxy <= 2).sum()),
                 fit_err_med_x_y_yaw=np.median(fe, 0).round(2).tolist(),
                 fit_err_xy_quality_ok_p50_max=(np.percentile(dxy[[x['fit_quality_ok'] for x in R]], [50, 100]).round(2).tolist()
                                                if any(x['fit_quality_ok'] for x in R) else None),
                 fit_quality_ok_pc_nominal_colour=[int(sum(x['fit_quality_ok'] for x in R if not x['random_colour'])),
                                                   int(sum(not x['random_colour'] for x in R))],
                 fit_quality_ok_scene_nominal=[int(sum(x['fit_quality_ok'] for x in R if not set(x['random_colour']) & {'pc_case', 'table', 'table_textured'})),
                                               int(sum(not set(x['random_colour']) & {'pc_case', 'table', 'table_textured'} for x in R))],
                 edges_inl_med={e: float(np.median([x['edges'][e] for x in R])) for e in R[0]['edges']})
        for k in nets:
            E = np.array([x[k][:3] for x in R])
            s[k] = dict(err_xy_mm_p50_p90=np.percentile(np.hypot(E[:, 0], E[:, 1]), [50, 90]).round(2).tolist(),
                        err_med_x_y_yaw=np.median(E, 0).round(2).tolist(), gate_ok=int(sum(x[k][3] for x in R)))
        summ[str(rgb)] = s
        print(rgb, json.dumps(s), flush=True)
    if a.save_images:
        import cv2
        sd = Path(a.save_images); sd.mkdir(parents=True, exist_ok=True)
        for i in range(min(4, a.n)):
            cv2.imwrite(str(sd / f'{i}.png'), cv2.cvtColor(np.hstack([imgs[g][i].cpu().numpy().astype(np.uint8) for g in a.rgbs]), cv2.COLOR_RGB2BGR))
    if a.save_images:                           # every frame (all variants side by side) as one npz
        np.savez_compressed(Path(a.save_images) / 'frames.npz', **{f'rgb_{g}': imgs[g].cpu().numpy().astype(np.uint8) for g in a.rgbs})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(args=vars(a), summary=summ, rows=rows), default=float))


if __name__ == '__main__':
    main()
