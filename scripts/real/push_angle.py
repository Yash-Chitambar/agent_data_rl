"""Angle of the final push against the port axis, from the real logs (docs/real/PUSH_ANGLE_20261119.md). No motion.

    PYTHONPATH=. python scripts/real/push_angle.py [--no-vision] [--json results/real_align/push_angle/push_angle.json]

Per real insertion of 2026-11-10/11/13 (n = 16; 12 with a final push):
- Port axis "arm": port-vision panel pose (logged panel_rv_deg of every align frame before the hand-over) through the
  hand-eye calibration and the believed FK (encoder + rig joint offsets) into the right base. Yaw and pitch of the
  panel normal. Also the v1 ZED yaw (the axis the push uses) and the edge-fit yaw (replay.json; x, y truth; its yaw
  reads about 1.9 deg below the arm, absolute yaw open).
- Hand rotation (FK) from the hand-over (insert start) to the seat, to the push start and to the push end, as a
  rotation vector in the arm port axes: rx pitch (about across), ry yaw (about up), rz roll (about the axis).
- Plug against the panel: the align stage's last vision reading (rv, rigid mean grasp) composed with the FK hand
  rotation since that reading.
- Push motion: plug-tip FK displacement from the push start to the push end (across, up, in) and its angles; the
  commanded (target) displacement and its angles.
- Vision during the insertion and the push (stream frames, 2 Hz): panel pose in the right wrist camera by
  port_vision.panel_pose. Camera against panel = hand against panel, with no FK. The FK camera pose then gives the
  panel normal in the base: a change means the PC turned or the arm bent where the encoders do not see it.
Axes: x across, y up, z out of the port (port_R convention); angles in deg, lengths in mm.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts/real')); sys.path.insert(0, str(ROOT / 'scripts/perception'))
import insertion_log_analysis as L  # noqa: E402

RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
FOLDERS = ['2026-11-10/full_145824_kjrc_1', '2026-11-10/full_150628_kjrc_2', '2026-11-10/full_152149_kjrc_3_push', '2026-11-10/full_152717_k4_1',
           '2026-11-10/full_155541_k4_3', '2026-11-10/full_161011_k4_4', '2026-11-10/full_163515_k4_5', '2026-11-10/full_170521_k4_6',
           '2026-11-11/full_155351_demo_1', '2026-11-11/full_160310_demo_2', '2026-11-11/full_161223_demo_3', '2026-11-11/full_161705_demo_4',
           '2026-11-11/full_163144_speed075_1', '2026-11-11/full_171905_fastmove_1', '2026-11-13/full_165400_all9_full_1',
           '2026-11-13/full_171043_all9b_full_1']
# operator labels (docs/real/DEPTH_SPRINGBACK_20261114.md section 4); others: "seated", no depth word
LABEL = {'demo_4': 'full', 'k4_5': 'full', 'k4_3': 'full', 'fastmove_1': 'full',
         'all9b_full_1': 'short', 'demo_1': 'short', 'all9_full_1': 'short', 'k4_1': 'short', 'kjrc_1': 'short',
         'kjrc_3_push': 'short', 'kjrc_2': 'short'}
OUT = ROOT / 'results/real_align/push_angle'


def short(name):
    return name.split('_', 2)[2]


def axis_angles(z_out):
    """Yaw (deg, ZED convention: insertion direction -z_out) and pitch (deg, + = the port axis points up out of the port) of an out-of-port axis."""
    z = np.asarray(z_out, float)
    return math.degrees(math.atan2(-z[1], -z[0])), math.degrees(math.asin(np.clip(z[2], -1, 1)))


def frame_from_z(z_out):
    """Port frame (x across, y up-ish, z out) from an out-of-port axis, x horizontal."""
    z = np.asarray(z_out, float) / np.linalg.norm(z_out)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def rv_in(R_rel_base, E):
    """Rotation (base frame, R_after @ R_before^T) as a rotation vector in the axes E, deg."""
    return Rot.from_matrix(E.T @ R_rel_base @ E).as_rotvec(degrees=True)


class Cam:
    def __init__(self):
        from zed_arm_port import ArmCam
        self.ac = ArmCam()

    def pose(self, q_b):
        return self.ac.cam_pose(np.asarray(q_b, float))


def ticks_at(t_tick, k_tick, t):
    return int(round(float(np.interp(t, t_tick, k_tick))))


def edge_fit_yaw():
    p = ROOT / 'results/placement/zed_edgefit/replay.json'
    out = {}
    if p.exists():
        for f in json.loads(p.read_text())['frames']:
            fit = f.get('fit')
            if isinstance(fit, dict) and fit.get('ok'):
                out[f['run']] = float(fit['fit_yaw_deg'])
    return out


def vision_track(run, cam, t, i_from, i_to):
    """Panel pose in the right wrist camera on the stream frames between two tick indices. Returns rows."""
    import cv2
    from deploy.bimanual import port_vision as PV
    fr = [json.loads(x) for x in open(os.path.join(run, 'frames.jsonl')) if x.strip()]
    fr = [f for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw']
    tt, kk = t['t'], t['k'].astype(int)
    rows = []
    for f in fr:
        k = ticks_at(tt, kk, f['t_capture'])
        if not (kk[i_from] - 25 <= k <= kk[i_to] + 25):
            continue
        img = cv2.imread(os.path.join(run, f['file']))
        if img is None or img.shape[:2] != (1024, 1280):
            continue
        und, K = PV.undistort(img, 'right')
        pose, why = PV.panel_pose(und, K, None)
        if pose is None:
            rows.append(dict(k=k, ok=False, why=why, tag=f['tag']))
            continue
        i = int(np.clip(np.searchsorted(kk, k), 0, len(kk) - 1))
        qb = t['q'][i, 0] + L.OFF
        pc, Rc = cam.pose(qb)
        zo = Rc @ pose['R'][:, 2]
        yaw, pitch = axis_angles(zo)
        rows.append(dict(k=k, ok=True, tag=f['tag'], R_cam=pose['R'].tolist(), emax=float(pose['emax']), lines=sorted(pose['lines']),
                         yaw_base=yaw, pitch_base=pitch, Rc=Rc.tolist()))
    return rows


def one(folder, cam, efit, vision=True):
    run = str(RUNS / folder)
    name = os.path.basename(run)
    ev = L.events(run)
    st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
    t = dict(np.load(os.path.join(run, 'ticks.npz'), allow_pickle=True))
    kk = t['k'].astype(int)
    idx = lambda k: int(np.clip(np.searchsorted(kk, k), 0, len(kk) - 1))  # noqa: E731
    pc = [e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc'][-1]
    yaw_v1 = float(pc['yaw_deg'])
    E_v1 = L.port_R(yaw_v1)
    # --- port axis from port vision at the align stage (before the hand-over)
    ho = next((e for e in ev if e['kind'] == 'align_handover'), None)
    t_ho = ho['t'] if ho else float('inf')
    meas = [e for e in ev if e.get('kind') == 'align_measure' and e.get('ok') and e['t'] <= t_ho + 1e-6 and e.get('q_b')]
    Z = []
    for e in meas:
        _, Rc = cam.pose(e['q_b'])
        for f in e.get('frames') or []:
            if f.get('panel_rv_deg') is not None:
                Z.append(Rc @ Rot.from_rotvec(f['panel_rv_deg'], degrees=True).as_matrix()[:, 2])
    Z = np.array(Z)
    z_arm = np.median(Z, 0)
    z_arm /= np.linalg.norm(z_arm)
    yp = np.array([axis_angles(z) for z in Z])
    yaw_arm, pitch_arm = axis_angles(z_arm)
    E = frame_from_z(z_arm)                 # arm port axes (pitched)
    # last 8 measurements: the panel just before the hand-over (PC pressed by the left hand)
    yp_last = yp[-8 * 5:] if len(yp) else yp
    # --- ticks
    k_ins = st['5 insert']
    seat_e = next((e for e in ev if e['kind'] == 'seated_detected'), None)
    k_seat = k_ins + int(seat_e['step']) + 1 if seat_e else None
    push = next((e for e in ev if e['kind'] == 'insert_final_push'), None)
    skip = next((e for e in ev if e['kind'] == 'insert_final_push_skipped'), None)
    k_rel = st.get('6 release')
    out = dict(run=short(name), folder=folder, date=folder[:10], label=LABEL.get(short(name), 'seated (no depth word)'),
               push=None if push is None else push.get('mode', 'position'), push_mm=None if push is None else push.get('mm'),
               push_force_n=None if push is None else push.get('force_n'), push_skipped=None if skip is None else skip.get('reason'),
               yaw_v1=yaw_v1, yaw_arm=yaw_arm, pitch_arm=pitch_arm, yaw_edgefit=efit.get(name),
               arm_n_frames=int(len(Z)), arm_yaw_sd=float(yp[:, 0].std()), arm_pitch_sd=float(yp[:, 1].std()),
               arm_yaw_last=float(np.median(yp_last[:, 0])), arm_pitch_last=float(np.median(yp_last[:, 1])))
    out['push_dir_err_v1'] = dict(yaw=yaw_v1 - yaw_arm, pitch=-pitch_arm)      # the push vector is horizontal at the v1 yaw
    # --- hand rotation and tip motion (FK)
    marks = dict(handover=k_ins, seat=k_seat)
    if push is not None:
        k_p0 = k_seat + 10
        un = push.get('unload')
        k_p1 = int(un['tick_start']) if un else int(push['tick']) - 17      # end of the hold, before the 0.3 s drop
        marks.update(push_start=k_p0, push_end=k_p1)
    elif k_seat is not None:
        marks.update(push_start=k_seat + 10, push_end=k_seat + 10)
    marks['release'] = k_rel
    qb = {m: t['q'][idx(k) - (1 if m == 'handover' else 0), 0] + L.OFF for m, k in marks.items() if k is not None}
    tb = {m: t['target_b'][idx(k), 0] for m, k in marks.items() if k is not None}
    tip, tcp, Rh = L.fk_tip(np.array(list(qb.values())))
    tip = dict(zip(qb, tip)); Rh = dict(zip(qb, Rh))
    tipt, _, Rt = L.fk_tip(np.array(list(tb.values())))           # target_b: believed frame already (no offset)
    tipt = dict(zip(tb, tipt)); Rt = dict(zip(tb, Rt))
    out['hand_rot_from_handover'] = {m: rv_in(Rh[m] @ Rh['handover'].T, E).round(2).tolist() for m in Rh}
    if 'push_end' in Rh:
        out['hand_rot_push'] = rv_in(Rh['push_end'] @ Rh['push_start'].T, E).round(2).tolist()
        out['target_rot_push'] = rv_in(Rt['push_end'] @ Rt['push_start'].T, E).round(2).tolist()
        d = (tip['push_end'] - tip['push_start']) @ E * 1e3
        dt_ = (tipt['push_end'] - tipt['push_start']) @ E * 1e3
        out['tip_push_mm'] = d.round(2).tolist()          # across, up, out (negative = in)
        out['target_push_mm'] = dt_.round(2).tolist()
        if push is not None and -d[2] > 0.2:
            out['tip_push_dir_deg'] = dict(yaw=math.degrees(math.atan2(d[0], -d[2])), pitch=math.degrees(math.atan2(d[1], -d[2])))
        if push is not None and -dt_[2] > 0.2:
            out['target_push_dir_deg'] = dict(yaw=math.degrees(math.atan2(dt_[0], -dt_[2])), pitch=math.degrees(math.atan2(dt_[1], -dt_[2])))
        out['tip_seat_to_end_mm'] = ((tip['push_end'] - tip['seat']) @ E * 1e3).round(2).tolist()
    out['tip_insert_mm'] = ((tip['seat'] - tip['handover']) @ E * 1e3).round(2).tolist() if 'seat' in tip else None
    # --- plug against the panel: last align vision reading + FK hand rotation since
    lm = [e for e in ev if e.get('kind') == 'align_measure' and e.get('rv_deg') is not None and e['t'] <= t_ho + 1e-6]
    if lm:
        e = lm[-1]
        rvf = e.get('rv_fused') or e['rv_deg']
        Rv = Rot.from_rotvec(rvf, degrees=True).as_matrix()     # reader in the panel model frame
        P = cam.pose(e['q_b'])[1] @ Rot.from_rotvec(np.median([f['panel_rv_deg'] for f in e['frames'] if f.get('panel_rv_deg')], 0), degrees=True).as_matrix()
        q_e = np.asarray(e['q_b'], float)
        _, _, R_e = L.fk_tip(q_e[None])
        out['plug_vs_panel_handover'] = np.asarray(rvf, float).round(2).tolist()
        out['panel_axes_in_E'] = (E.T @ P).round(3).tolist()
        out['rot_applied_align'] = e.get('rot_applied')
        out['plug_vs_panel'] = {}
        for m in Rh:
            Rm = P.T @ (Rh[m] @ R_e[0].T) @ P @ Rv
            out['plug_vs_panel'][m] = Rot.from_matrix(Rm).as_rotvec(degrees=True).round(2).tolist()
    ih = next((e for e in ev if e['kind'] == 'insert_start'), None)
    out['inhand6'] = None if ih is None else np.round(ih['inhand6'], 2).tolist()
    sq = [e for e in ev if e['kind'] == 'grasp_v3' and e.get('event') == 'squeeze_end']
    out['grasp_rot_deg'] = None if not sq else [round(sq[-1].get('rot_deg', float('nan')), 2), round(sq[-1].get('roll_deg', float('nan')), 2)]
    # --- vision during insertion and push
    if vision:
        i0 = idx(k_ins) - 100
        i1 = idx(marks.get('release') or marks.get('push_end') or k_seat) + 40
        rows = vision_track(run, cam, t, max(i0, 0), min(i1, len(kk) - 1))
        ok = [r for r in rows if r['ok'] and r['emax'] <= 3.5]
        out['vision_n'] = dict(frames=len(rows), ok=len(ok))
        # reference: the vision frames before the insert (align, still)
        ref = [r for r in ok if r['k'] < k_ins]
        seg = {}
        for name_, lo, hi in (('pre', -10 ** 9, k_ins), ('insert', k_ins, (k_seat or k_ins) + 10),
                              ('push', (k_seat or k_ins) + 10, marks.get('push_end') or (k_seat or k_ins) + 10),
                              ('after', marks.get('push_end') or (k_seat or k_ins) + 10, (k_rel or 10 ** 9))):
            rs = [r for r in ok if lo <= r['k'] < hi]
            if not rs:
                continue
            s = dict(n=len(rs), yaw_base=float(np.median([r['yaw_base'] for r in rs])), pitch_base=float(np.median([r['pitch_base'] for r in rs])))
            if ref:
                # camera rotation against the panel, since the pre-insert reference: rotation of the hand against the panel (no FK)
                Rref = np.mean([np.asarray(r['R_cam']) for r in ref[-6:]], 0)
                U, _, Vt = np.linalg.svd(Rref)
                Rref = U @ Vt
                Pref = np.asarray(ref[-1]['Rc']) @ np.asarray(ref[-1]['R_cam'])     # panel in the base at the reference (FK camera)
                rv = []
                for r in rs:
                    Rc_ = np.asarray(r['R_cam'])
                    # hand turn in panel axes: panel seen from the camera, R_cam(t) = Rhc^T ... ; hand rotation in panel axes = R_ref R_t^T (panel frame)
                    Tp = Rc_.T @ Rref                                   # hand turn in panel axes
                    rv.append(rv_in(Pref @ Tp @ Pref.T, E))              # same, in the arm port axes (as the FK rows)
                s['hand_vs_panel_turn'] = np.median(rv, 0).round(2).tolist()
            seg[name_] = s
        out['vision'] = seg
        # per frame: hand turn against the panel by vision and by FK, both since the last pre-insert vision frame, arm port axes
        if ref:
            r0 = ref[-1]
            R0c, Rc0 = np.asarray(r0['R_cam']), np.asarray(r0['Rc'])
            P0 = Rc0 @ R0c
            for r in ok:
                if r['k'] < r0['k']:
                    continue
                Tp = np.asarray(r['R_cam']).T @ R0c
                r['turn_vision'] = rv_in(P0 @ Tp @ P0.T, E).round(2).tolist()
                r['turn_fk'] = rv_in(np.asarray(r['Rc']) @ Rc0.T, E).round(2).tolist()
        out['vision_rows'] = [{k: v for k, v in r.items() if k not in ('R_cam', 'Rc')} for r in rows]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-vision', action='store_true')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--json', default=str(OUT / 'push_angle.json'))
    a = ap.parse_args()
    cam = Cam()
    efit = edge_fit_yaw()
    res = []
    for f in FOLDERS:
        if a.only and short(os.path.basename(f)) not in a.only:
            continue
        r = one(f, cam, efit, vision=not a.no_vision)
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != 'vision_rows'}), flush=True)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(dict(method=__doc__.strip(), runs=res), indent=1))


if __name__ == '__main__':
    main()
