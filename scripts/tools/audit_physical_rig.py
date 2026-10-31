"""Audit: the training sim's cameras and arms against the real rig calibration (no hardware access).

Checks the model the training env uses (PickEnv, MuJoCo Warp, randomization off) against
configs/real_cameras_duchamp3.json and the recorded snapshot data/sim_ref/2026-10-25_scene/pointed.json:

1. camera intrinsics and sensor window = calibrated K and policy crop;
2. camera poses: top (ZED extrinsics), wrists (pose in link_6, and in the right base at the recorded
   joints + offsets vs the poses the real-side calibration recorded);
3. reprojection: table/object points through the real camera model (OpenCV pinhole, policy crop and
   resize) vs the sim camera model (Warp's ray formula), in policy pixels;
4. end to end: the reader's centre in the GPU-rendered policy image vs its projection;
5. arm bases (right = world, left = measured transform) and joint offsets.

    CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/tools/audit_physical_rig.py
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from blind_insertion.envs.pick_env import PickCfg, PickEnv
from blind_insertion.scenes import pc_bench as S, physical_rig as R

RES = 320
spec = R.load()
snap = json.loads(Path("data/sim_ref/2026-10-25_scene/pointed.json").read_text())
env = PickEnv(PickCfg(nworld=1, seed=0, rand={}, phys={}, cams=("top", "wrist_cam", "left_wrist_cam")))
m = env.mjm
d = mujoco.MjData(m)
R.reference_pose(m, d, env.lay)
S.place_pc(m, d)
S.place_reader(m, d, env.lay)
mujoco.mj_forward(m, d)
out, ok = {}, True


def quat_mat(q):
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, np.asarray(q, float) / np.linalg.norm(q))
    return M.reshape(3, 3)


def ang_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def real_K(name):
    if name == "top":
        K = spec["top"]["intrinsics_rectified"]["HD1080"]
        w, h = K["resolution"]
        crop = 2 * min(round(K["cx"]), w - round(K["cx"]), round(K["cy"]), h - round(K["cy"]))
    else:
        side = "right" if name == "wrist_cam" else "left"
        K = spec["wrist"][side]["intrinsics"]
        crop = spec["wrist"][side]["policy_crop_px"] // 2 * 2
    return K, crop


def real_pixel(P, pos, R_cv, name):
    """Real policy pixel of world point P: OpenCV pinhole (undistorted / rectified image), square crop
    centred on the rounded principal point, resized to RES."""
    K, crop = real_K(name)
    pc = R_cv.T @ (P - pos)
    u = K["fx"] * pc[0] / pc[2] + K["cx"]                       # OpenCV: pixel centres at integers
    v = K["fy"] * pc[1] / pc[2] + K["cy"]
    s = crop / RES                                               # area resize: output pixel i spans [i s, (i+1) s)
    return np.array([(u - (round(K["cx"]) - crop // 2) + 0.5) / s - 0.5, (v - (round(K["cy"]) - crop // 2) + 0.5) / s - 0.5])


def sim_pixel(P, c):
    """Policy pixel of P in sim camera c, as MuJoCo Warp's compute_ray maps pixels (intrinsics branch)."""
    fx, fy, cx, cy = m.cam_intrinsic[c]
    sw, sh = m.cam_sensorsize[c]
    pc = d.cam_xmat[c].reshape(3, 3).T @ (P - d.cam_xpos[c])     # MuJoCo camera: -z forward, y up
    x, y = pc[0] / -pc[2], pc[1] / -pc[2]                         # on the plane at znear = 1
    left, right = -(sw / 2 - cx) / fx, (sw / 2 + cx) / fx
    top, bottom = (sh / 2 - cy) / fy, -(sh / 2 + cy) / fy
    return np.array([(x - left) / (right - left) * RES - 0.5, (y - top) / (bottom - top) * RES - 0.5])


# --- 1-3: per camera
pts = np.array([[x, y, z] for x in np.linspace(0.12, 0.45, 8) for y in np.linspace(0.08, 0.40, 8) for z in (0.0, 0.04)])
for name in ("top", "wrist_cam", "left_wrist_cam"):
    c = m.camera(name).id
    K, crop = real_K(name)
    r = dict(sensorsize_px=m.cam_sensorsize[c].tolist(), intrinsic=np.round(m.cam_intrinsic[c], 3).tolist(),
             real_fx_fy=[K["fx"], K["fy"]], real_crop_px=crop)
    good = np.allclose(m.cam_sensorsize[c], crop) and np.allclose(m.cam_intrinsic[c][:2], [K["fx"], K["fy"]]) \
        and np.allclose(m.cam_intrinsic[c][2:], R._pp_offset(K["cx"], K["cy"]), atol=1e-5)
    if name == "top":
        e = spec["top"]["extrinsics_cam_in_right_base"]
        pos, R_cv = np.array(e["pos"]), quat_mat(e["quat"])
        r["body"] = m.body(m.cam_bodyid[c]).name
    else:
        side = "right" if name == "wrist_cam" else "left"
        pl = spec["wrist"][side]["pose_in_link6"]
        r["body"] = m.body(m.cam_bodyid[c]).name
        good &= r["body"] == ("link_6" if side == "right" else "left_link_6")
        # pose in link_6 (MuJoCo camera = OpenCV camera rotated 180 deg about x)
        R_loc = quat_mat(m.cam_quat[c]) @ np.diag([1, -1, -1])
        r["pose_in_link6_err_mm_deg"] = [float(np.linalg.norm(m.cam_pos[c] - pl["pos"]) * 1e3), ang_deg(R_loc, quat_mat(pl["quat"]))]
        # real-side recorded pose at the snapshot (real FK + offsets + hand-eye) vs sim FK at the same joints
        sw = snap["wrist_cam_in_right_base"][side]
        pos, R_cv = np.array(sw["pos"]), quat_mat(sw["quat"])
    R_sim = d.cam_xmat[c].reshape(3, 3) @ np.diag([1, -1, -1])
    r["world_pose_err_mm_deg"] = [float(np.linalg.norm(d.cam_xpos[c] - pos) * 1e3), ang_deg(R_sim, R_cv)]
    errs = []
    for P in pts:
        if (R_cv.T @ (P - pos))[2] <= 0.05:
            continue
        a, b = real_pixel(P, pos, R_cv, name), sim_pixel(P, c)
        if (0 <= a).all() and (a < RES).all():
            errs.append(np.linalg.norm(a - b))
    r["reprojection_px"] = dict(n=len(errs), max=float(np.max(errs)), mean=float(np.mean(errs)))
    good &= r["world_pose_err_mm_deg"][0] < 0.1 and r["world_pose_err_mm_deg"][1] < 0.01 and r["reprojection_px"]["max"] < 0.05
    # Warp's batched camera (what the GPU renders) equals the model's
    wq, wp_ = env.f["cam_quat"][0, c].cpu().numpy(), env.f["cam_pos"][0, c].cpu().numpy()
    good &= np.allclose(wp_, m.cam_pos[c], atol=1e-6) and np.allclose(wq, m.cam_quat[c], atol=1e-6)
    r["pass"] = bool(good)
    ok &= good
    out[name] = r

# --- 4: end to end through the GPU renderer: reader top centre in the policy image
env.t_qpos[0] = torch.as_tensor(d.qpos, dtype=torch.float32, device=env.device)
env.t_qvel[0] = 0
env.t_mocap_pos[0] = torch.as_tensor(d.mocap_pos, dtype=torch.float32, device=env.device)
env.t_mocap_quat[0] = torch.as_tensor(d.mocap_quat, dtype=torch.float32, device=env.device)
fr = env.render()
pr = d.xpos[env.lay.peg_body] + d.xmat[env.lay.peg_body].reshape(3, 3) @ np.array(env.lay.variants[0].peg_boxes[0][0])
P = np.array([pr[0], pr[1], 2 * env.body_half_t])                    # body top centre
for name in ("top", "wrist_cam", "left_wrist_cam"):
    img = fr[name][0].cpu().numpy().astype(float)
    c = m.camera(name).id
    guess = sim_pixel(P, c)
    win = 6 if name == "top" else 30
    x0, y0 = (np.round(guess) - win).astype(int)
    sub = img[max(y0, 0):y0 + 2 * win, max(x0, 0):x0 + 2 * win].mean(2)
    ys, xs = np.nonzero(sub > 110)                                      # silver top on the dark table
    cen = np.array([xs.mean() + max(x0, 0), ys.mean() + max(y0, 0)]) if len(xs) else np.array([np.nan] * 2)
    out[name]["gpu_reader_centre_vs_projection_px"] = float(np.linalg.norm(cen - guess))

# --- 5: arms
L = spec["arms"]["left_base_in_right_base"]
bl = m.body("left_arm").id
out["arms"] = dict(
    right_base_pos=m.body_pos[m.body("arm").id].tolist(), right_base_quat=m.body_quat[m.body("arm").id].tolist(),
    left_base_pos_err_mm=float(np.linalg.norm(m.body_pos[bl] - L["pos"]) * 1e3),
    left_base_rot_err_deg=ang_deg(quat_mat(m.body_quat[bl]), quat_mat(L["quat"])),
    joint_offsets_rad=dict(right=spec["arms"]["right"]["joint_offsets_rad"], left=spec["arms"]["left"]["joint_offsets_rad"]),
    left_arm_joints_in_training=env.t_qpos[0, [m.jnt_qposadr[m.joint(f"left_joint{i}").id] for i in range(1, 7)]].tolist())
out["objects"] = dict(port_xy=S.PORT_XY, reader_xy=S.READER_XY, port_height=S.PORT_HEIGHT, pc_size=S.PC_SIZE)
out["all_pass"] = bool(ok)
print(json.dumps(out, indent=1))
Path("results/H/physical_rig").mkdir(parents=True, exist_ok=True)
Path("results/H/physical_rig/audit.json").write_text(json.dumps(out, indent=1) + "\n")
