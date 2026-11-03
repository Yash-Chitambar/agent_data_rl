"""Source of the servo nets' roll bias on real frames (real -3.21 deg at the last waypoint, sim +1.74).

Tests on the logged frames of the 13 grasps of 2026-11-02, one frame per grasp (the last servo waypoint):
  base         reproduce the logged target roll from the saved policy frame
  rotate       turn the image in the plane by theta: output per deg, and the theta that cancels the bias
  crop         principal point and resize variants of the real pipeline
  cam_fingers  the camera's own roll against the sim, from the yellow tips (rigid to the camera)
  look         net on sim renders at the same joint angles: old look, new look, reader yaw, reader tilt
  paste        real frame with the tips replaced by sim tips, and the reverse

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl python scripts/real2sim/servo_roll_source.py
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
from inhand_measure import DIST, K0, KM, OFF, W, all_runs  # noqa: E402
from blind_insertion.envs import kinematics as K  # noqa: E402
from blind_insertion.pipeline.pick_servo import load_servo_net  # noqa: E402

BUNDLE = ROOT / 'deploy/bundle/models'
OUT = ROOT / 'results/real/inhand_real'


def policy_from_raw(raw, cx=None, cy=None, K_=None, D=None, interp=cv2.INTER_AREA, res=320):
    Km = KM if K_ is None else K_
    und = cv2.undistort(raw, Km, DIST if D is None else D)
    x, y = int(round(K0['cx'])), int(round(K0['cy']))
    if cx is not None or cy is not None:   # a principal point error: shift the frame, keep the crop
        und = cv2.warpAffine(und, np.float32([[1, 0, x - round(K0['cx'] if cx is None else cx)], [0, 1, y - round(K0['cy'] if cy is None else cy)]]), und.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)
    h = W['policy_crop_px'] // 2
    return cv2.cvtColor(cv2.resize(und[y - h:y + h, x - h:x + h], (res, res), interpolation=interp), cv2.COLOR_BGR2RGB)


class Rollnet:
    def __init__(self, dev):
        self.dev = dev
        self.nets = {n: load_servo_net(BUNDLE / f'{n}.pt', dev) for n in ('reader_servo_near6', 'reader_servo_near', 'reader_servo_hover')}

    @torch.no_grad()
    def rv(self, ims, name='reader_servo_near6'):
        """ims (N, 320, 320, 3) uint8 RGB -> rotation vectors (N, 3), rad."""
        net, sc, n = self.nets[name]
        x = torch.as_tensor(np.stack(ims), device=self.dev)
        out = net(x) * sc
        return out[:, -3:]


def target_roll(q_b, rv, chain):
    """Roll of the commanded hand: asin(R[2, 1]) of FK(q) exp(rv), deg (the closing axis from the horizontal)."""
    q = torch.as_tensor(np.asarray(q_b), dtype=torch.float32).reshape(-1, 6)
    rv = torch.as_tensor(rv, dtype=torch.float32).cpu().reshape(-1, 3)
    _, Rh = chain.fk(q)
    a = rv.norm(dim=1, keepdim=True).clamp(min=1e-9)
    Rt = Rh @ K.axis_angle_mat(rv / a, a[:, 0])
    return np.degrees(np.arcsin(Rt[:, 2, 1].numpy())), np.degrees(np.arcsin(Rh[:, 2, 1].numpy()))


def waypoints():
    """One row per grasp: the last waypoint's raw frame, policy frame, joints, logged delta and logged roll."""
    out = []
    for run in all_runs():
        ev = [json.loads(x) for x in open(run / 'events.jsonl')]
        sf = [e for e in ev if e['kind'] == 'servo_frame' and e['frame_kind'] == 'waypoint']
        if not sf:
            continue
        e = sf[-1]
        k = int(e['frame_tag'].split('@')[1])
        fr = [json.loads(x) for x in open(run / 'frames.jsonl')]
        fr = [f for f in fr if f['cam'] == 'wrist_right' and f['tag'] == '1_pick' and f.get('tick') == k]
        raw = next(f for f in fr if f['kind'] == 'raw')
        pol = next(f for f in fr if f['kind'] == 'policy')
        z = np.load(run / 'ticks.npz')
        i = np.searchsorted(z['k'], k)
        rf = [x for x in ev if x.get('event') == 'roll_fix' and x['tick'] == k + 1]
        out.append(dict(run=run, raw=cv2.imread(str(run / raw['file'])), pol=cv2.imread(str(run / pol['file']))[:, :, ::-1].copy(), q=z['q'][i, 0] + OFF, q_all=z['q'],
                        delta=np.array(e['delta']), roll_log=rf[0]['roll_servo_deg'] if rf else float('nan'), grip=float(z['grip'][i, 0]) if 'grip' in z else None, tick=k))
    return out


def rot_img(im, theta):
    h, w = im.shape[:2]
    return cv2.warpAffine(im, cv2.getRotationMatrix2D((w / 2 - .5, h / 2 - .5), theta, 1.), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def yellow(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return ((hsv[..., 1] > 100) & (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 2] > 70)).astype(np.uint8)


def best_align(real, sim, angs=np.arange(-5, 5.01, .25), shifts=range(-14, 15, 2)):
    """Rotation (deg, image plane) and shift of the sim mask that best match the real mask: IoU search."""
    best = (0, 0, 0, -1)
    h, w = real.shape
    for a in angs:
        Rm = cv2.getRotationMatrix2D((w / 2 - .5, h / 2 - .5), a, 1.)
        for dx in shifts:
            for dy in shifts:
                Rm2 = Rm + np.array([[0, 0, dx], [0, 0, dy]], float)
                s = cv2.warpAffine(sim, Rm2, (w, h), flags=cv2.INTER_NEAREST)
                iou = (s & real).sum() / max(1, (s | real).sum())
                if iou > best[3]:
                    best = (float(a), dx, dy, float(iou))
    return best


def reader_on_table(w, hand, poses, shift_mm, table_z=-.033):
    """World pose of the real reader on the table before the close, from this grasp's measured in-hand pose and the FK at
    the end of the squeeze: flat, at the table. (R, p m) or None."""
    from scipy.spatial.transform import Rotation as Rot
    g = poses[(poses.run == w['run'].name) & (poses.rms_px < 5)]
    if not len(g):
        return None
    med = g[['dx', 'dy', 'dz', 'rx', 'ry', 'rz']].median().to_numpy()
    z = np.load(w['run'] / 'ticks.npz')
    ev = [json.loads(x) for x in open(w['run'] / 'events.jsonl')]
    se = next(e for e in ev if e.get('event') == 'squeeze_end')
    hand.pose(z['q'][np.searchsorted(z['k'], se['tick']), 0] + OFF)
    Rs, ps = hand.site()
    R_rs = Rot.from_rotvec(np.radians(med[3:])).as_matrix() @ hand.R_nom
    p_rs = hand.p_nom + (med[:3] + shift_mm) * 1e-3
    Rw, pw = Rs @ R_rs, ps + Rs @ p_rs
    a_ = Rw[:, 2].copy()
    a_[2] = 0
    a_ /= np.linalg.norm(a_)
    y_ = np.array([0, 0, 1.])
    return np.c_[np.cross(y_, a_), y_, a_], np.array([pw[0], pw[1], table_z + .004415 + 1e-4])


def place_at_frame(m, d, lay, w, finger_m, pose, pc=None):
    """Joints of the frame, PC at the run's ZED pose, the reader at its real table pose."""
    from blind_insertion.scenes import pc_bench as S
    for j in range(6):
        d.qpos[m.joint(f'joint{j + 1}').qposadr] = w['q'][j]
    for fn in ('left_finger', 'right_finger'):
        d.qpos[m.joint(fn).qposadr] = finger_m
    if pc:
        S.place_pc(m, d, port_xy=pc['port_xy'], yaw_deg=pc['yaw_deg'])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, pose[0].flatten())
    d.qpos[lay.peg_qadr:lay.peg_qadr + 3], d.qpos[lay.peg_qadr + 3:lay.peg_qadr + 7] = pose[1], q
    mujoco.mj_forward(m, d)


def scene_for(preset):
    from inhand_compare import build
    from blind_insertion.envs import inhand_real as IR
    m, lay = build(preset)
    return m, lay, IR


def main():
    # noqa
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', default='base,rotate,crop,cam_fingers,look,paste')
    ap.add_argument('--name', default='')
    a = ap.parse_args()
    tests = a.tests.split(',')
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from deploy.bimanual.scene import SceneModel
    sc = SceneModel('cpu')
    chain = sc.chain['right']
    rn = Rollnet(dev)
    W_ = waypoints()
    Q = np.array([w['q'] for w in W_])
    rep = {'n': len(W_), 'runs': [w['run'].name for w in W_]}
    print('n grasps', len(W_), flush=True)

    def roll_of(ims, name='reader_servo_near6'):
        return target_roll(Q, rn.rv(ims, name), chain)[0]

    base = roll_of([w['pol'] for w in W_])
    rep['base'] = dict(mean=float(base.mean()), sd=float(base.std(ddof=1)), lo=float(base.min()), hi=float(base.max()),
                       logged=[w['roll_log'] for w in W_], reproduced=base.round(2).tolist(),
                       hand_roll=[round(float(x), 3) for x in target_roll(Q, np.zeros((len(Q), 3)), chain)[1]])
    print('base', {k: v for k, v in rep['base'].items() if k in ('mean', 'sd', 'lo', 'hi')}, 'logged mean', np.nanmean(rep['base']['logged']), flush=True)
    print(' hand roll at frame', np.mean(rep['base']['hand_roll']), flush=True)

    if 'rotate' in tests:
        rows = {}
        for th in (-6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6):
            r = roll_of([rot_img(w['pol'], th) for w in W_])
            rows[th] = float(r.mean())
        ths, vals = np.array(list(rows)), np.array(list(rows.values()))
        m, c = np.polyfit(ths, vals, 1)
        rep['rotate'] = dict(mean_roll_by_theta=rows, slope=float(m), intercept=float(c), theta_to_zero=float(-c / m), theta_to_sim=float((1.74 - c) / m))
        print('rotate', rep['rotate'], flush=True)

    if 'crop' in tests:
        rep['crop'] = {}
        for nm, kw in (('recompute', {}), ('bilinear', dict(interp=cv2.INTER_LINEAR)), ('cx+8', dict(cx=K0['cx'] + 8)), ('cx-8', dict(cx=K0['cx'] - 8)), ('cy+8', dict(cy=K0['cy'] + 8)), ('cy-8', dict(cy=K0['cy'] - 8)),
                       ('no_distortion_k2', dict(D=DIST * np.array([1, 0, 1, 1, 0]))), ('distortion_x0.8', dict(D=DIST * .8))):
            r = roll_of([policy_from_raw(w['raw'], **kw) for w in W_])
            rep['crop'][nm] = float(r.mean())
        print('crop', rep['crop'], flush=True)

    if 'cam_fingers' in tests or 'look' in tests or 'paste' in tests:
        from inhand_compare import place
        fit = json.loads((ROOT / 'results/usbc/gripper_sweep/sweep_20261030/fit.json').read_text())['arms']['right']['gripper_map']

    if 'cam_fingers' in tests:
        m, lay, IR = scene_for(None)
        d = mujoco.MjData(m)
        cam = m.camera('wrist_cam').id
        rd = mujoco.Renderer(m, 320, 320)
        rows = []
        for w in W_:
            grip = w['grip'] if w['grip'] is not None else .8
            place(m, d, lay, w['q'], np.zeros(6), fit['slope'] * grip + fit['intercept'])
            # reader away
            d.qpos[lay.peg_qadr:lay.peg_qadr + 3] = [-.3, -.4, 0]
            mujoco.mj_forward(m, d)
            rd.update_scene(d, camera='wrist_cam')
            sim = rd.render()
            rows.append(best_align(yellow(w['pol']), yellow(sim)))
        rows = np.array(rows)
        rep['cam_fingers'] = dict(rot_mean=float(rows[:, 0].mean()), rot_sd=float(rows[:, 0].std(ddof=1)), dx=float(rows[:, 1].mean()), dy=float(rows[:, 2].mean()), iou=float(rows[:, 3].mean()), per_frame=rows.round(2).tolist())
        print('cam_fingers', {k: v for k, v in rep['cam_fingers'].items() if k != 'per_frame'}, flush=True)
        rd.close()

    if 'look' in tests or 'paste' in tests:
        from inhand_measure import Hand
        hand = Hand()
        import pandas as pd
        poses = pd.read_csv(OUT / 'poses.csv')
        shift = hand.camera_shift_site() * 1e3
        Fits = [reader_on_table(w, hand, poses, shift) for w in W_]
        keep = [i for i, f_ in enumerate(Fits) if f_ is not None]
        print('kept', len(keep), flush=True)
        WK, FK_ = [W_[i] for i in keep], [Fits[i] for i in keep]
        Qk = Q[keep]
        base_k = base[keep]
        rep['n_waypoint_fit'] = len(keep)

        def roll_k(ims):
            return target_roll(Qk, rn.rv(ims), chain)[0]

        def pc_of(w):
            ev = [json.loads(x) for x in open(w['run'] / 'events.jsonl')]
            return next((e for e in ev if e.get('est') == 'zed_pc'), None)

        def render_set(m, lay, rd, seg=False, yaw=0., tilt=0.):
            out = []
            d = mujoco.MjData(m)
            for w, fr_ in zip(WK, FK_):
                fr2 = fr_
                if yaw or tilt:     # small extra rotation of the reader about its own axes, through the body centre
                    from scipy.spatial.transform import Rotation as Rot
                    dR = Rot.from_rotvec(np.radians([0., yaw, tilt])).as_matrix()
                    c_ = np.array([0, 0, .070])
                    R2 = fr_[0] @ dR
                    fr2 = (R2, fr_[1] + fr_[0] @ c_ - R2 @ c_)
                place_at_frame(m, d, lay, w, fit['slope'] * w['grip'] + fit['intercept'], fr2, pc_of(w))
                rd.update_scene(d, camera='wrist_cam')
                if seg:
                    rd.enable_segmentation_rendering()
                    rd.update_scene(d, camera='wrist_cam')
                    sg = rd.render()
                    rd.disable_segmentation_rendering()
                    out.append(np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), m.geom_bodyid[np.clip(sg[..., 0], 0, m.ngeom - 1)], -1))
                else:
                    out.append(rd.render().copy())
            return out

    if 'look' in tests:
        rep['look'] = {'real_same_frames': dict(mean=float(base_k.mean()), sd=float(base_k.std(ddof=1)))}
        print('real, same frames', rep['look']['real_same_frames'], flush=True)
        for nm, preset in (('old_look', None), ('new_look', 'v1')):
            m, lay, IR = scene_for(preset)
            rd = mujoco.Renderer(m, 320, 320)
            for label, kw in (('real pose', {}), ('yaw +3', dict(yaw=3.)), ('yaw -3', dict(yaw=-3.)), ('roll about the length +3', dict(tilt=3.)), ('roll about the length -3', dict(tilt=-3.))):
                ims = render_set(m, lay, rd, **kw)
                r = roll_k(ims)
                rep['look'][f'{nm}: {label}'] = dict(mean=float(r.mean()), sd=float(r.std(ddof=1)))
                print(nm, label, rep['look'][f'{nm}: {label}'], flush=True)
                if label == 'real pose':
                    cv2.imwrite(str(OUT / f'servo_waypoint_real_sim_{nm}.jpg'), cv2.cvtColor(np.hstack([WK[0]['pol'], ims[0]]), cv2.COLOR_RGB2BGR))
            rd.close()
    if 'paste' in tests:
        m, lay, IR = scene_for(None)
        rd = mujoco.Renderer(m, 320, 320)
        sims = render_set(m, lay, rd)
        sg = render_set(m, lay, rd, seg=True)
        rd.close()
        fingers = [m.body('left_finger').id, m.body('right_finger').id, m.body('gripper').id]
        Fm = [np.isin(g_, fingers) for g_ in sg]
        Rm_ = [g_ == m.body('peg').id for g_ in sg]
        dil = lambda x, n=7: cv2.dilate(x.astype(np.uint8), np.ones((n, n), np.uint8)) > 0   # noqa: E731
        comp = lambda a, b, ma: np.where(ma[..., None], a, b)                                # noqa: E731
        reals = [w['pol'] for w in WK]
        base = base_k
        roll_of = roll_k
        rep['paste'] = {}
        for nm, ims in (('real', reals), ('sim', sims),
                        ('real + sim tips', [comp(s_, r_, dil(f_)) for s_, r_, f_ in zip(sims, reals, Fm)]),
                        ('sim + real tips', [comp(r_, s_, dil(f_)) for s_, r_, f_ in zip(sims, reals, Fm)]),
                        ('real + sim reader', [comp(s_, r_, dil(g_)) for s_, r_, g_ in zip(sims, reals, Rm_)]),
                        ('sim + real reader', [comp(r_, s_, dil(g_)) for s_, r_, g_ in zip(sims, reals, Rm_)]),
                        ('real tips and reader + sim scene', [comp(r_, s_, dil(f_ | g_)) for s_, r_, f_, g_ in zip(sims, reals, Fm, Rm_)]),
                        ('sim tips and reader + real scene', [comp(s_, r_, dil(f_ | g_)) for s_, r_, f_, g_ in zip(sims, reals, Fm, Rm_)]),
                        ('real, tips grey', [comp(np.full_like(r_, 128), r_, dil(f_)) for r_, f_ in zip(reals, Fm)]),
                        ('sim, tips grey', [comp(np.full_like(s_, 128), s_, dil(f_)) for s_, f_ in zip(sims, Fm)])):
            r = roll_of(ims)
            rep['paste'][nm] = dict(mean=float(r.mean()), sd=float(r.std(ddof=1)))
            print(nm, rep['paste'][nm], flush=True)
        # which property of the real reader: edits inside the sim reader's mask (dilated), the rest of the real frame kept
        def edit(fn, w_=15):
            out = []
            for r_, g_ in zip(reals, Rm_):
                ma = dil(g_, w_)
                out.append(fn(r_, ma))
            return out

        def dim(k):
            return lambda r_, ma: np.where(ma[..., None], (r_ * k).astype(np.uint8), r_)

        def mirror(r_, ma):
            xs_ = np.where(ma.any(0))[0]
            c_ = (xs_[0] + xs_[-1]) // 2
            f_ = r_[:, ::-1]
            f_ = np.roll(f_, 2 * c_ - r_.shape[1] + 1, axis=1)
            return np.where(ma[..., None], f_, r_)

        def shift(dx):
            def fn(r_, ma):
                return np.where(ma[..., None], np.roll(r_, dx, axis=1), r_)
            return fn

        def rot_patch(th):
            def fn(r_, ma):
                ys_, xs_ = np.where(ma)
                c_ = (float(xs_.mean()), float(ys_.mean()))
                Rm2 = cv2.getRotationMatrix2D(c_, th, 1.)
                return np.where(ma[..., None], cv2.warpAffine(r_, Rm2, r_.shape[1::-1], borderMode=cv2.BORDER_REPLICATE), r_)
            return fn

        def mirrored(r_, ma):
            xs_ = np.where(ma.any(0))[0]
            return np.roll(r_[:, ::-1], (xs_[0] + xs_[-1]) - r_.shape[1] + 1, axis=1)

        def symm(r_, ma):
            return np.where(ma[..., None], ((r_.astype(np.float32) + mirrored(r_, ma)) / 2).astype(np.uint8), r_)

        def part(fn, which):     # apply only to the body or only to the neck and plug: split at the sim's body front row
            def f2(r_, ma):
                ys_ = np.where(ma.any(1))[0]
                wide = ma.sum(1) > .6 * ma.sum(1).max()
                y_split = int(np.where(wide)[0][0])
                keep = np.zeros_like(ma)
                keep[y_split:] = True
                m2 = ma & (keep if which == 'body' else ~keep)
                return fn(r_, m2 if m2.any() else ma)
            return f2

        rep['reader_edits'] = {}
        for nm, fn in (('dim x0.8', dim(.8)), ('dim x0.6', dim(.6)), ('mirror L-R', mirror), ('shift -6 px', shift(-6)), ('shift +6 px', shift(6)),
                       ('symmetrized', symm), ('mirror body only', part(mirror, 'body')), ('mirror neck and plug only', part(mirror, 'neck')), ('symmetrized body only', part(symm, 'body')), ('symmetrized neck and plug only', part(symm, 'neck')), ('turn +3 deg', rot_patch(3)), ('turn -3 deg', rot_patch(-3)), ('turn +6 deg', rot_patch(6)), ('turn -6 deg', rot_patch(-6))):
            r = roll_of(edit(fn))
            rep['reader_edits'][nm] = dict(mean=float(r.mean()), sd=float(r.std(ddof=1)))
            print(nm, rep['reader_edits'][nm], flush=True)
        # occlusion map: mean change of the target roll when a 48 px grey patch covers this part of the real frame
        G, st = [], 40
        for y in range(0, 281, st):
            row = []
            for x in range(0, 281, st):
                ims = []
                for r_ in reals:
                    i_ = r_.copy()
                    i_[y:y + 48, x:x + 48] = 128
                    ims.append(i_)
                row.append(float(roll_of(ims).mean() - base.mean()))
            G.append(row)
        rep['occlusion_grid_40px'] = np.round(G, 2).tolist()
        print('occlusion', np.round(G, 1), flush=True)
    (OUT / f"roll_source{'_' + a.name if a.name else ''}.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == '__main__':
    main()
