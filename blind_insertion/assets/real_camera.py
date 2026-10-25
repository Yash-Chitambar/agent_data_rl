"""Real wrist camera spec -> sim camera, and the matching real image pipeline.

A spec file (e.g. configs/real_cameras_duchamp3.json) describes the real rig's cameras. Its
`wrist.sim` section holds what the renderer needs:

  modality       "rgb" | "depth" | "rgbd": what the real camera gives (EnvCfg.cam_modality must match)
  fovy_deg       vertical field of view of the square image the policy sees (deg)
  pose_in_link6  {"pos": [x, y, z], "quat": [w, x, y, z]}: camera optical frame in the YAM link_6
                 frame, OpenCV convention (+z along the optical axis, +y down), i.e. what a
                 hand-eye calibration gives

A null field keeps the model's own camera (Menagerie: a D405 on I2RT's bracket) and is reported
as a placeholder. The real side must feed the policy the same image: undistort, crop the
policy_crop_px square centred on the principal point (the largest that fits), resize to cam_res
(`real_policy_image`). With that crop the sim needs no principal-point offset, and
fovy = 2 atan(s / 2 / fy) for crop side s.
"""
import json
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
# OpenCV optical frame -> MuJoCo camera frame (looks along -z, +y up): 180 deg about x
_CV_TO_MJ = np.array([0.0, 1.0, 0.0, 0.0])


def load_spec(path: str) -> dict:
    p = Path(path)
    return json.loads((p if p.is_absolute() or p.exists() else REPO / p).read_text())


def apply_wrist_spec(mjm: mujoco.MjModel, path: str, modality: str) -> dict:
    """Set the wrist camera of a compiled model from the spec (in place). Returns
    {"applied": [...], "placeholder": [...]}. Raises if `modality` is not what the camera gives."""
    sim = load_spec(path)["wrist"]["sim"]
    allowed = sim["modality"]
    if modality not in (allowed if isinstance(allowed, list) else [allowed]):
        raise ValueError(f"cam_modality '{modality}' but the real wrist camera gives {allowed} ({path})")
    cid = max(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("wrist", "wrist_cam"))
    assert cid >= 0, "no wrist camera in the model"
    out = {"applied": ["modality"], "placeholder": []}

    if sim.get("fovy_deg") is not None:
        mjm.cam_sensorsize[cid] = 0.0   # sensorsize > 0 makes MuJoCo use focal/sensorsize, not fovy
        mjm.cam_intrinsic[cid] = 0.0
        mjm.cam_fovy[cid] = float(sim["fovy_deg"])
        out["applied"].append("fovy_deg")
    else:
        out["placeholder"].append("fovy_deg")

    pose = sim.get("pose_in_link6")
    if pose is not None:
        body = mjm.cam_bodyid[cid]
        parent = mjm.body_parentid[body]
        link6 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        if body == link6:
            bp, bq = np.zeros(3), np.array([1.0, 0, 0, 0])
        else:
            assert parent == link6, "wrist camera body must be link_6 or a child of it"
            bp, bq = mjm.body_pos[body].copy(), mjm.body_quat[body].copy()
        q_cv = np.asarray(pose["quat"], float)
        q_cv /= np.linalg.norm(q_cv)
        q_mj = np.zeros(4)
        mujoco.mju_mulQuat(q_mj, q_cv, _CV_TO_MJ)              # camera in link_6, MuJoCo convention
        bq_inv = np.zeros(4)
        mujoco.mju_negQuat(bq_inv, bq)
        pos = np.zeros(3)
        mujoco.mju_rotVecQuat(pos, np.asarray(pose["pos"], float) - bp, bq_inv)
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, bq_inv, q_mj)                  # camera in its parent body
        mjm.cam_pos[cid], mjm.cam_quat[cid] = pos, quat
        out["applied"].append("pose_in_link6")
    else:
        out["placeholder"].append("pose_in_link6")
    return out


def wrist_pose_in_link6(mjm: mujoco.MjModel) -> dict:
    """The model's wrist camera pose in link_6, OpenCV convention (for the spec's reference)."""
    cid = max(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("wrist", "wrist_cam"))
    d = mujoco.MjData(mjm)
    mujoco.mj_kinematics(mjm, d)
    mujoco.mj_camlight(mjm, d)
    link6 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "link_6")
    R6, p6 = d.xmat[link6].reshape(3, 3), d.xpos[link6]
    Rc = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # MuJoCo -> OpenCV axes
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R6.T @ Rc).flatten())
    return {"pos": (R6.T @ (d.cam_xpos[cid] - p6)).round(6).tolist(), "quat": q.round(6).tolist()}


def real_policy_image(frame: np.ndarray, spec: dict, res: int, side: str = "right") -> np.ndarray:
    """Real wrist frame of `side` (H, W, 3, BGR from OpenCV) -> the (res, res, 3) RGB image the sim
    renders: undistort, crop policy_crop_px square centred on the principal point, resize."""
    import cv2
    w = spec["wrist"].get(side)
    if not w or w.get("intrinsics") is None or w.get("distortion") is None:
        raise ValueError(f"{side} wrist camera not calibrated: the spec has nulls")
    K = w["intrinsics"]
    Km = np.array([[K["fx"], 0, K["cx"]], [0, K["fy"], K["cy"]], [0, 0, 1]], float)
    und = cv2.undistort(frame, Km, np.asarray(w["distortion"], float))
    cx, cy = int(round(K["cx"])), int(round(K["cy"]))
    half = w["policy_crop_px"] // 2
    crop = und[cy - half:cy + half, cx - half:cx + half]
    return cv2.cvtColor(cv2.resize(crop, (res, res), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
