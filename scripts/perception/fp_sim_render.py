"""Sim truth set for the ZED FoundationPose check: rectified HD1080 stereo pairs of the calibrated ZED top camera.

Per case: left.png / right.png (BGR, as the rig saves them), depth_left.npy (render z-depth, m, float32), mask_pc.png and
mask_reader.png (render segmentation), truth.json (fixture = port 2 frame pose and reader pose in the right base, the
same frame as the rig logs). Once per output folder: mesh_pc_sim.obj / mesh_reader_sim.obj (the sim geoms of the PC
fixture body and the reader body, in their body frames, with vertex colours from the geom / material rgba).

Camera: blind_insertion/configs/real_cameras_duchamp3.json top (HD1080 rectified intrinsics, extrinsics_cam_in_right_base);
the right camera is the left one moved by baseline_mm along the optical +x axis (rectified pair). No camera DR.
Arms: the joints of a real capture pose (data/zed_gt/spot1/capture_noreader.json + spec joint offsets).

    MUJOCO_GL=egl PYTHONPATH=. python scripts/perception/fp_sim_render.py --out results/perception/fp/sim --n 30 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from blind_insertion.scenes import pc_bench as S
from blind_insertion.scenes import physical_rig as R

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_JSON = Path('/home/yashchitambar/peg_test/data/zed_gt/spot1/capture_noreader.json')
PC0 = (0.231, 0.398, 4.0)          # K7 box centre (edge-fit frame of the 2026-11-10 spot), deg
READER0 = (0.131, 0.375, 90.0)      # place_reader body xy (visible next to the PC, as on zed_gt spot1)
PX_SIGN, PY_SIGN = -1.0, -1.0       # MuJoCo principal offset signs (checked: edge fit on the render 0.6 mm, rms 0.14 px)


def build(look=True):
    prof = json.loads((ROOT / 'data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
    extra = None
    if look:
        from blind_insertion.envs.bimanual_runtime import _add_pc_real_look
        lookdir = ROOT / 'blind_insertion/assets/pc_real_look'
        if lookdir.exists():
            extra = lambda sp: _add_pc_real_look(sp, lookdir)     # noqa: E731
    m, lay = S.build_scene(detailed=True, bimanual=True, measured_support=True, support_mode='free', profile=prof,
                           extra_spec=extra)
    return m, lay


def set_camera(m, spec, cam='top', right=False):
    top = spec['top']
    k = top['intrinsics_rectified']['HD1080']
    pos, quat = R.camera_pose(top['extrinsics_cam_in_right_base'])
    if right:
        q = np.asarray(top['extrinsics_cam_in_right_base']['quat'], float); q /= np.linalg.norm(q)
        Rm = np.zeros(9); mujoco.mju_quat2Mat(Rm, q)
        pos = pos + Rm.reshape(3, 3)[:, 0] * top['baseline_mm'] * 1e-3
    cid = m.camera(cam).id
    m.cam_mode[cid] = 0
    m.cam_bodyid[cid] = 0
    m.cam_pos[cid] = pos
    m.cam_quat[cid] = quat
    W, H = k['resolution']
    m.cam_resolution[cid] = [W, H]
    m.cam_sensorsize[cid] = [W, H]                         # 1 length unit = 1 pixel
    m.cam_intrinsic[cid] = [k['fx'], k['fy'], PX_SIGN * (k['cx'] - W / 2), PY_SIGN * (k['cy'] - H / 2)]
    return k


def body_geoms(m, body):
    """Geoms of `body` and its descendants."""
    ids = set()
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        while b > 0:
            if b == body:
                ids.add(g)
                break
            b = m.body_parentid[b]
    return sorted(ids)


def export_mesh(m, d, body, path):
    """Visible geoms of a body subtree as one OBJ in the body frame (vertex colours)."""
    import trimesh
    mujoco.mj_forward(m, d)
    Rb = d.xmat[body].reshape(3, 3); pb = d.xpos[body]
    parts = []
    for g in body_geoms(m, body):
        rgba = m.geom_rgba[g].copy()
        if m.geom_matid[g] >= 0:
            rgba = m.mat_rgba[m.geom_matid[g]].copy()
        if rgba[3] < .1 or m.geom_group[g] > 2:
            continue
        t = m.geom_type[g]; s = m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            tm = trimesh.creation.box(extents=2 * s)
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
            tm = trimesh.Trimesh(m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy(), process=False)
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            tm = trimesh.creation.cylinder(radius=s[0], height=2 * s[1])
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            tm = trimesh.creation.icosphere(radius=s[0], subdivisions=2)
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            tm = trimesh.creation.capsule(radius=s[0], height=2 * s[1])
        else:
            continue
        Rg = d.geom_xmat[g].reshape(3, 3); pg = d.geom_xpos[g]
        T = np.eye(4); T[:3, :3] = Rb.T @ Rg; T[:3, 3] = Rb.T @ (pg - pb)
        tm.apply_transform(T)
        tm.visual.vertex_colors = np.tile((np.clip(rgba, 0, 1) * 255).astype(np.uint8), (len(tm.vertices), 1))
        parts.append(tm)
    mesh = trimesh.util.concatenate(parts)
    mesh.export(path)
    return mesh


def pose_of(d, body):
    T = np.eye(4); T[:3, :3] = d.xmat[body].reshape(3, 3); T[:3, 3] = d.xpos[body]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/perception/fp/sim')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--box', nargs=6, type=float, default=[-5, 25, -12, 25, -5, 15],
                    help='dx lo hi, dy lo hi (mm), dyaw lo hi (deg) around PC0')
    ap.add_argument('--no-look', action='store_true')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spec = R.load()
    m, lay = build(look=not a.no_look)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
    d = mujoco.MjData(m)
    cap = json.loads(CAPTURE_JSON.read_text())
    for side, prefix in (('right', ''), ('left', 'left_')):
        q = np.asarray(cap['joints_measured'][side][:6]) + np.asarray(spec['arms'][side]['joint_offsets_rad'])
        for j, v in enumerate(q, 1):
            d.qpos[m.joint(prefix + f'joint{j}').qposadr] = v
        for f in ('left', 'right'):
            jn = prefix + f + '_finger'
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) >= 0:
                d.qpos[m.joint(jn).qposadr] = .0475
    fix = m.body('fixture').id
    reader = m.body(int(m.jnt_bodyid[[j for j in range(m.njnt) if m.jnt_qposadr[j] == lay.peg_qadr][0]])).id
    pc_g, rd_g = set(body_geoms(m, fix)), set(body_geoms(m, reader))
    S.place_pc(m, d, port_xy=PC0[:2], yaw_deg=PC0[2]); S.place_reader(m, d, lay, xy=READER0[:2], yaw_deg=READER0[2])
    export_mesh(m, d, fix, out / 'mesh_pc_sim.obj')
    export_mesh(m, d, reader, out / 'mesh_reader_sim.obj')
    rng = np.random.default_rng(a.seed)
    rL = mujoco.Renderer(m, height=1080, width=1920)
    cases = []
    try:
        for i in range(a.n):
            if i == 0:
                dx, dy, dyaw = 0., 0., 0.
            else:
                dx, dy, dyaw = rng.uniform(a.box[0], a.box[1]), rng.uniform(a.box[2], a.box[3]), rng.uniform(a.box[4], a.box[5])
            rdx, rdy, rdyaw = (0., 0., 0.) if i == 0 else (rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-10, 10))
            pc = (PC0[0] + dx * 1e-3, PC0[1] + dy * 1e-3, PC0[2] + dyaw)
            rd = (READER0[0] + rdx * 1e-3, READER0[1] + rdy * 1e-3, READER0[2] + rdyaw)
            S.place_pc(m, d, port_xy=pc[:2], yaw_deg=pc[2]); S.place_reader(m, d, lay, xy=rd[:2], yaw_deg=rd[2])
            mujoco.mj_forward(m, d)
            cd = out / f'case_{i:03d}'; cd.mkdir(exist_ok=True)
            for right in (False, True):
                set_camera(m, spec, right=right)
                mujoco.mj_forward(m, d)
                rL.update_scene(d, camera='top')
                img = rL.render().copy()
                cv2.imwrite(str(cd / ('right.png' if right else 'left.png')), img[:, :, ::-1])
                if not right:
                    rL.enable_depth_rendering(); rL.update_scene(d, camera='top')
                    np.save(cd / 'depth_left.npy', rL.render().astype(np.float32)); rL.disable_depth_rendering()
                    rL.enable_segmentation_rendering(); rL.update_scene(d, camera='top')
                    seg = rL.render().copy(); rL.disable_segmentation_rendering()
                    gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
                    cv2.imwrite(str(cd / 'mask_pc.png'), np.isin(gid, list(pc_g)).astype(np.uint8) * 255)
                    cv2.imwrite(str(cd / 'mask_reader.png'), np.isin(gid, list(rd_g)).astype(np.uint8) * 255)
            t = dict(pc_port_xy_yaw=[pc[0], pc[1], pc[2]], reader_xy_yaw=[rd[0], rd[1], rd[2]],
                     T_pc_in_base=pose_of(d, fix).tolist(), T_reader_in_base=pose_of(d, reader).tolist(),
                     frame='right base; fixture = PC port 2 frame (sim label convention); reader = reader body frame')
            (cd / 'truth.json').write_text(json.dumps(t, indent=1))
            cases.append(dict(case=cd.name, **t))
            print(cd.name, [round(v, 4) for v in pc], flush=True)
    finally:
        rL.close()
    (out / 'cases.json').write_text(json.dumps(dict(seed=a.seed, box=a.box, pc0=PC0, reader0=READER0, cases=cases), indent=1))


if __name__ == '__main__':
    main()
