"""Sim against real at the same joint angles: wrist view and a side view, reader in the preset's mean in-hand pose.

The same detector (inhand_measure.detect) reads the sim and the real frame on the same undistorted canvas, so its bias
cancels; output: pixel error per feature and side-by-side images in results/real/inhand_real/.

    PYTHONPATH=. MUJOCO_GL=egl python scripts/real2sim/inhand_compare.py --preset v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inhand_measure import CANVAS, KM, OFF, PLUG, all_runs, detect, fit_pose, held_frames, model_points, undist  # noqa: E402
from blind_insertion.envs import inhand_real as IR  # noqa: E402
from blind_insertion.scenes import pc_bench as S  # noqa: E402

OUT = ROOT / 'results/real/inhand_real'
FEATS = ['silver front L', 'silver front R', 'silver back L', 'silver back R', 'plug tip L', 'plug tip R', 'plug base L', 'plug base R']


def build(preset):
    from deploy.bimanual.scene import PROFILE, CONTROLLER_PROFILE
    look = IR.load(preset)['look'] if preset else None
    m, lay = S.build_scene(detailed=True, bimanual=True, measured_support=True, support_mode='free',
                           profile=json.loads(PROFILE.read_text()), controller_profile=CONTROLLER_PROFILE, reader_look=look)
    fit = json.loads((ROOT / 'results/usbc/gripper_sweep/sweep_20261030/fit.json').read_text())
    off = np.array(list(fit['arms']['right']['fitted_offset_mm'].values())) * 1e-3     # as BimanualRuntime, tip_offset_target='camera'
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.body('gripper').quat)
    cam = m.camera('wrist_cam').id
    m.cam_pos[cam] -= rot.reshape(3, 3) @ off
    return m, lay


def set_canvas(m):
    """The wrist camera renders the undistorted canvas of inhand_measure: same K."""
    cam = m.camera('wrist_cam').id
    w, h = CANVAS
    m.cam_sensorsize[cam] = [w, h]
    m.cam_resolution[cam] = [w, h]
    m.cam_intrinsic[cam] = [KM[0, 0], KM[1, 1], -(KM[0, 2] - (w / 2 - .5)), -(KM[1, 2] - (h / 2 - .5))]
    m.vis.global_.offwidth, m.vis.global_.offheight = max(w, 1600), max(h, 1600)


def place(m, d, lay, q_b, delta, grip_finger_m, pc=None, lateral=0.):
    for j in range(6):
        d.qpos[m.joint(f'joint{j + 1}').qposadr] = q_b[j]
    for f in ('left_finger', 'right_finger'):
        d.qpos[m.joint(f).qposadr] = grip_finger_m
    if pc is not None:
        S.place_pc(m, d, port_xy=pc[0], yaw_deg=pc[1])
    else:
        fa = int(m.joint('pc_free_joint').qposadr[0])
        d.qpos[fa:fa + 3] = [-.3, 1., .05]
    mujoco.mj_kinematics(m, d)
    s = m.site('grasp_site').id
    Rs, ps = d.site_xmat[s].reshape(3, 3), d.site_xpos[s]
    R_rs, p_rs = IR.reader_in_site(torch.tensor(delta, dtype=torch.float32)[None], lateral_mm=lateral)
    Rr = Rs @ R_rs[0].numpy().astype(float)
    pr = ps + Rs @ p_rs[0].numpy().astype(float)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rr.flatten())
    d.qpos[lay.peg_qadr:lay.peg_qadr + 3], d.qpos[lay.peg_qadr + 3:lay.peg_qadr + 7] = pr, q
    mujoco.mj_forward(m, d)


def project_reader(m, d, lay, silver):
    """Pixels of the reader's model corners in the sim's wrist camera (the canvas K), in inhand_measure's order."""
    b = lay.peg_qadr
    Rr = np.zeros(9)
    mujoco.mju_quat2Mat(Rr, d.qpos[b + 3:b + 7])
    P = (Rr.reshape(3, 3) @ (model_points(silver, PLUG) * 1e-3).T).T + d.qpos[b:b + 3]
    c = m.camera('wrist_cam').id
    Pc = (P - d.cam_xpos[c]) @ d.cam_xmat[c].reshape(3, 3)      # MuJoCo camera: -z forward, +y up
    return np.c_[KM[0, 2] + KM[0, 0] * Pc[:, 0] / -Pc[:, 2], KM[1, 2] - KM[1, 1] * Pc[:, 1] / -Pc[:, 2]]


def yellow(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return (hsv[..., 1] > 110) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='v1')
    ap.add_argument('--mode', default='loo', choices=('loo', 'own', 'mean'), help="with a rows preset: the mean of the other grasps, the grasp's own row, or the mean")
    ap.add_argument('--dy', type=float, default=None, help='override the closing-axis delta (mm)')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = IR.load(a.preset)['inhand']
    rows_ = np.array(c['rows'], float) if 'rows' in c else None
    delta = rows_.mean(0) if rows_ is not None else np.array(c['mean_mm'] + c['mean_rotvec_deg'], float)
    if a.dy is not None:
        delta[1] = a.dy
    fit = json.loads((ROOT / 'results/usbc/gripper_sweep/sweep_20261030/fit.json').read_text())['arms']['right']['gripper_map']
    res = {}
    # the sim's silver plate per look: old = caliper numbers on the swapped axes, placed as bench_details places it
    silver = {'old': dict(w=34.6, l=29.6, z0=69.83 - 1. - 14.8, y=4.4), 'new': dict(w=29.6, l=34.6, z0=49.2, y=4.4)}
    for name, preset, dl in (('old', None, None), ('new', a.preset, delta)):
        m, lay = build(preset)
        set_canvas(m)
        d = mujoco.MjData(m)
        rd = mujoco.Renderer(m, CANVAS[1], CANVAS[0])
        if dl is None:
            from blind_insertion.envs.inhand import measured_mean, nominal_reader_in_site
            s = json.loads((ROOT / 'results/usbc/policy/pick/inhand_spread.json').read_text())
            dl = np.array([s['about_nominal'][k]['mean'] for k in ('inhand_dx_mm', 'inhand_dy_mm', 'inhand_dz_mm', 'inhand_rx_deg', 'inhand_ry_deg', 'inhand_rz_deg')])
        rows = []
        for run in all_runs():
            fr = held_frames(run)
            best = None
            for f in fr:
                r, _ = detect(undist(cv2.imread(str(run / f['file']))))
                if r is None:
                    continue
                best = best or (f, r)
                if fit_pose(r['pts'], np.r_[0, 0, 0, 0, 60, 140])[2] < 5:
                    best = (f, r)
                    break
            if best is None:
                continue
            f, r = best
            z = np.load(run / 'ticks.npz')
            i = np.searchsorted(z['t'], f['t_capture'])
            q = z['q'][i, 0] + OFF
            grip = float(z['grip'][i, 0]) if 'grip' in z else .31
            ev = next((json.loads(x) for x in open(run / 'events.jsonl') if '"est": "zed_pc"' in x), None)
            pc = (ev['port_xy'], ev['yaw_deg']) if ev else None
            if name == 'new' and rows_ is not None and a.mode != 'mean':
                i_ = c['runs'].index(run.name)
                dl = rows_[i_] if a.mode == 'own' else np.delete(rows_, i_, 0).mean(0)     # loo: the mean of the other 12 grasps
            place(m, d, lay, q, dl, fit['slope'] * grip + fit['intercept'], pc, c.get('render_lateral_mm', 0.) if name == 'new' else 0.)
            rd.update_scene(d, camera='wrist_cam')
            sim = rd.render()[:, :, ::-1].copy()
            real = undist(cv2.imread(str(run / f['file'])))
            row = dict(run=run.name)
            e = project_reader(m, d, lay, silver[name]) - r['pts']
            for k, n_ in enumerate(FEATS):
                row[n_ + ' dx'], row[n_ + ' dy'] = e[k, 0], e[k, 1]
            row['err_px'] = float(np.sqrt(np.nanmean((e ** 2).sum(1))))
            row['plug_err_px'] = float(np.sqrt(np.nanmean((e[4:] ** 2).sum(1))))
            row['silver_err_px'] = float(np.sqrt(np.nanmean((e[:4] ** 2).sum(1))))
            ym, ys = yellow(real), yellow(sim)
            row['tip_iou'] = float((ym & ys).sum() / max(1, (ym | ys).sum()))
            rows.append(row)
            if name == 'new' and len(rows) <= 2 or name == 'old' and len(rows) == 1:
                vis = np.hstack([real, sim, cv2.addWeighted(real, .5, sim, .5, 0)])
                for lab, x in (('real', 10), (f'sim {name}', 1290), ('blend', 2570)):
                    cv2.putText(vis, lab, (x, 40), 0, 1.2, (0, 255, 0), 2)
                cv2.imwrite(str(OUT / f'wrist_{name}_{run.name}.jpg'), cv2.resize(vis, None, fx=.42, fy=.42, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
        import pandas as pd
        df = pd.DataFrame(rows)
        res[name] = df
        pd.set_option('display.width', 200)
        print(name, 'n', len(df), 'fails', df['fail'].notna().sum() if 'fail' in df else 0)
        cols = [c_ for c_ in df.columns if c_ not in ('run', 'fail')]
        print(df[cols].agg(['mean', 'std']).T.round(2).to_string())
        df.to_csv(OUT / f'pixel_error_{name}{"_" + a.preset + "_" + a.mode if name == "new" else ""}.csv', index=False)
        rd.close()


if __name__ == '__main__':
    main()
