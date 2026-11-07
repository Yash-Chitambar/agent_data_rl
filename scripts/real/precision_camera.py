"""Camera check of the arm's landing accuracy: a checkerboard flat on the table, seen by the moving arm's wrist camera.

Used by precision_check.py --camera. The board is calib.py's (8 x 7 inner corners, 15 mm squares); intrinsics,
distortion and the camera's pose in link 6 are the rig's calibration (calib.cam_prior). Encoder FK alone misses link
flex, gear backlash and joint-zero errors; the camera sees where the hand really is.

  * board_from_zed(side): the board's pose in the arm's base frame from the fixed ZED (no motion).
  * plan_view(side, T_base_board, seeds): joints putting the camera `dist` m from the board centre, tilted `tilt` deg
    toward the arm base (as calib.py's search poses), by IK on Menagerie link 6. The caller checks it for safety.
  * BoardCam(side).measure(n): median camera pose against the board over n frames (the arm must be still).
  * camera_metrics(records, side): per step, where vision puts the camera against the FK of the commanded target and
    of the measured joints, each relative to the last hold at the test pose before it (hand-eye and model errors
    cancel for small steps); and how far vision sees the camera return to the test pose (model-free repeatability).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SPEC = REPO / 'blind_insertion/configs/real_cameras_duchamp3.json'


def _calib():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import calib
    return calib


def _T(pos, quat_wxyz):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat_wxyz, float))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.reshape(3, 3), pos
    return T


def offsets(side):
    return np.asarray(json.loads(SPEC.read_text())['arms'][side]['joint_offsets_rad'], float)


class CamModel:
    """FK of the wrist camera (OpenCV optical frame) in the arm's base frame from MOTOR-frame joints."""

    def __init__(self, side):
        C = _calib()
        self.kin = C.Kin()
        self.T6c, self.K, self.dist = C.cam_prior(self.kin, side)
        self.off = offsets(side)

    def T_base_cam(self, q_motor):
        return self.kin.fk(np.asarray(q_motor, float)[:6] + self.off) @ self.T6c


def board_from_zed(side, n=10):
    """T_base_board (4x4, the side's arm base frame) and the mean reprojection error (px), from the ZED."""
    import cv2
    C = _calib()
    top = json.loads(SPEC.read_text())['top'][f'extrinsics_cam_in_{side}_base']
    T_base_zed = _T(top['pos'], top['quat'])
    zed = C.Zed()
    try:
        poses = []
        for _ in range(n):
            c = C.detect(cv2.cvtColor(zed.grab(), cv2.COLOR_BGR2GRAY))
            if c is not None:
                poses.append(C.board_pose(c, zed.K, None))
    finally:
        zed.close()
    if len(poses) < n // 2:
        raise SystemExit(f'the ZED sees the board in {len(poses)} of {n} frames: clear its view')
    t = np.median([p[0][:3, 3] for p in poses], axis=0)
    T = poses[int(np.argmin([np.linalg.norm(p[0][:3, 3] - t) for p in poses]))][0]
    return T_base_zed @ T, float(np.mean([p[1] for p in poses]))


def plan_view(side, T_base_board, seeds, dist=0.26, tilt=40.0, roll_ref_q=None):
    """Joints (motor frame) with the camera `dist` from the board centre, tilted `tilt` deg toward the base."""
    C = _calib()
    cm = CamModel(side)
    c = T_base_board[:3, :3] @ C.BOARD_CENTER + T_base_board[:3, 3]
    toward_base = -np.r_[c[:2], 0.0] / np.linalg.norm(c[:2])
    p = c + dist * (np.cos(np.radians(tilt)) * np.array([0, 0, 1.0]) + np.sin(np.radians(tilt)) * toward_base)
    R_ref = cm.T_base_cam(roll_ref_q if roll_ref_q is not None else seeds[0])[:3, :3]
    T_cam = C.look_at(p, c, R_ref)
    T6 = T_cam @ np.linalg.inv(cm.T6c)
    q = cm.kin.ik(T6, [np.asarray(s, float)[:6] + cm.off for s in seeds])
    return None if q is None else q - cm.off


class BoardCam:
    def __init__(self, side):
        C = _calib()
        self.C = C
        self.cm = CamModel(side)
        self.cam = C.WristCam(side)

    def measure(self, n=10, between=None):
        """dict(T_cam_board (4x4) median over the frames, n_ok, reproj px) or None if the board is not seen.
        between(): called before every frame (the caller's guard keeps running while frames are taken)."""
        import cv2
        Ts, rp = [], []
        for _ in range(n):
            if between is not None:
                between()
            g = cv2.cvtColor(self.cam.grab(), cv2.COLOR_BGR2GRAY)
            c = self.C.detect(g)
            if c is None:
                continue
            T, e = self.C.board_pose(c, self.cm.K, self.cm.dist)
            Ts.append(T)
            rp.append(e)
        if len(Ts) < max(3, n // 2):
            return None
        t = np.median([T[:3, 3] for T in Ts], axis=0)
        k = int(np.argmin([np.linalg.norm(T[:3, 3] - t) for T in Ts]))
        T = Ts[k].copy()
        T[:3, 3] = t
        return dict(T_cam_board=T, n_ok=len(Ts), reproj_px=float(np.mean(rp)),
                    spread_mm=float(np.max([np.linalg.norm(T_[:3, 3] - t) for T_ in Ts]) * 1e3))

    def close(self):
        self.cam.close()


def camera_metrics(records, side):
    """records: dicts (mode, joint, delta, q, target, T_cam_board) in order, one per hold with vision. joint 0 /
    delta 0 marks a hold at the test pose."""
    cm = CamModel(side)
    out, base, first = [], {}, {}
    for r in records:
        Tcb = np.asarray(r['T_cam_board'], float)
        p_board = np.linalg.inv(Tcb)[:3, 3]                    # camera in the board frame: model-free
        if r['delta'] == 0.0:
            base[r['mode']] = r
            if r['mode'] not in first:
                first[r['mode']] = p_board
            out.append(dict(mode=r['mode'], joint=r['joint'], delta=0.0,
                            return_mm=float(np.linalg.norm(p_board - first[r['mode']]) * 1e3)))
            continue
        b = base.get(r['mode'])
        if b is None:
            continue
        T_base_board = cm.T_base_cam(b['q']) @ np.asarray(b['T_cam_board'], float)     # anchored at the last base hold
        p_vis = (T_base_board @ np.linalg.inv(Tcb))[:3, 3]
        p_tgt, p_fk = cm.T_base_cam(r['target'])[:3, 3], cm.T_base_cam(r['q'])[:3, 3]
        p0 = cm.T_base_cam(b['target'])[:3, 3]
        out.append(dict(mode=r['mode'], joint=r['joint'], delta=r['delta'],
                        step_mm=float(np.linalg.norm(p_tgt - p0) * 1e3),
                        vis_vs_target_mm=float(np.linalg.norm(p_vis - p_tgt) * 1e3),
                        vis_vs_encoders_mm=float(np.linalg.norm(p_vis - p_fk) * 1e3),
                        encoders_vs_target_mm=float(np.linalg.norm(p_fk - p_tgt) * 1e3)))
    return out
