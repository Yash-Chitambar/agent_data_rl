"""Task H scene: the GMKtec G10 mini PC and the SanDisk USB-C SD reader on the YAM bench.

All poses are in the right ("task") arm's base frame, as everywhere in this package: the bench
top is z = 0 (`builder.YAM_TABLE`, RoboSimStudio `yam_i2rt` on `bench_075`). The PC stands at the
back right of the table with its rear panel facing the arms; the reader lies flat in front of it
with its plug pointing at the PC (+x). The target is the second USB-C port from the left.

The port is the USB-C receptacle of `builder.usbc_variant` (the fixture plate); the PC case is
plain boxes around it (`pc_case_boxes`), on the same mocap fixture body, so the batched env
(`PegEnv`) and the full pick scene share one geometry. Dimensions are photo estimates until
measured (FINDINGS, task H).

Fixture frame with `PORT_BASE_RPY_DEG`: z out of the rear panel (world -x), x along the
opening's long side (world -y), y up.

    .venv/bin/python -m blind_insertion.scenes.pc_bench --write_config
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

PC_SIZE = (0.103, 0.042, 0.098)   # width (panel), height, depth (m)
PORT_FROM_LEFT = 0.030            # port centre from the panel's left edge, seen from the arms (m)
PORT_HEIGHT = 0.025               # port centre above the table (m)
PLATE = 0.030                     # square receptacle plate around the port (Variant.plate_size)
PLATE_DEPTH = 0.0075              # USB-C cavity 6.50 mm + 1 mm floor (builder.usbc_spec_variant)
PORT_BASE_RPY_DEG = (90.0, 0.0, -90.0)   # fixture z -> world -x, x -> world -y, y -> world z
PORT_XY = (0.321370, 0.236917)             # nominal port centre on the table (arm base frame)

PAD_SPHERES = 10                  # contact spheres per pad side (builder.sphere_pads); 10: a tilted grasp keeps 4+ contacts per pad (4: 1-2, the reader pivots)
FINGER_DAMPING = 200.0            # N s/m per finger in the pick scene (see build_scene)
LEFT_BASE_Y = 0.62                # idle left arm base (bimanual mount, arms at y = +-0.31)
LEFT_HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # folded rest (out of the top camera)
READER_XY = (0.203937, 0.220908)           # nominal reader centre (body) on the table
# yaw about the vertical from the calibrated ZED (pointed_zed_left.png, rays onto the table): the PC's
# rear-panel bottom edge and the reader's body-to-plug-tip direction (2026-10-26)
PC_YAW_DEG = 2.3
READER_YAW_DEG = -0.7
# top grasp height: tcp above the reader body centre (body 8.83 mm tall, centre 4.4 mm up), per gripper.
# linear: fingertips 8.6 mm below the tcp clear the table, pads cover 2.3-8.8 mm of the body.
# flexpoint: the tcp IS the tip ends; tips 1.5-3.0 mm above the table, the rubber pads (from 1.4 mm
# above the tips, assets/flexpoint.py) cover 2.9-8.8 mm of the body.
GRASP_Y = {"linear": (0.0048, 0.0060), "flexpoint": (-0.0029, -0.0014)}
# grasp tilt about the closing axis (deg, uniform per episode; PegEnv inhand_tilt sign): the wrist leans
# back toward the robot so the wrist camera, on the PC side of the gripper, looks toward the PC
# (untilted it looks back at the tips). GRASP_Y holds at the low tip corner (Expert raises the tcp).
GRASP_TILT_DEG = (25.0, 45.0)
# fixed top camera: on a mount midway between the two arm bases (y = LEFT_BASE_Y / 2), above the
# bench behind the arm bases, looking down the whole bench (as the lab photo); image up = back wall. Pose is a placeholder
# until measured (PickCfg.rand jitters it).
TOP_CAM = dict(pos=(-0.10, LEFT_BASE_Y / 2, 0.95), look_at=(0.40, LEFT_BASE_Y / 2, 0.0), fovy=62.0, up=(1.0, 0.0, 0.0))
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "task_h_reader.json"


def pc_case_boxes(port_from_left=PORT_FROM_LEFT, port_h=PORT_HEIGHT, plate=PLATE,
                  plate_depth=PLATE_DEPTH, size=PC_SIZE):
    """The PC case as boxes [(centre, half)] in the fixture frame, leaving the plate free.
    Viewed from the arms the panel's left edge is world +y = fixture -x."""
    w, h, d = size
    x0, x1 = -port_from_left, w - port_from_left
    y0, y1 = -port_h, h - port_h
    p = plate / 2

    def box(lo, hi):
        return ([(a + b) / 2 for a, b in zip(lo, hi)], [(b - a) / 2 for a, b in zip(lo, hi)])

    out = [box((x0, y0, -d), (x1, y1, -plate_depth)),           # case behind the panel
           box((x0, y0, -plate_depth), (-p, y1, 0.0)),           # panel left of the plate
           box((p, y0, -plate_depth), (x1, y1, 0.0)),            # panel right of the plate
           box((-p, y0, -plate_depth), (p, -p, 0.0))]            # panel below the plate
    if y1 > p:
        out.append(box((-p, p, -plate_depth), (p, y1, 0.0)))     # panel above the plate
    return out


def _look_xyaxes(pos, look_at, up=(0, 0, 1)):
    z = np.asarray(pos, float) - np.asarray(look_at, float)
    z /= np.linalg.norm(z)                          # MuJoCo cameras look along -z
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return [*x, *np.cross(z, x)]


def add_scene(spec: mujoco.MjSpec, top_cam=TOP_CAM, physical=True):
    """`builder.build(extra=...)` hook: the idle left arm (visual), the fixed top camera, the
    PC's other ports and colours. Physics of the task arm, port and reader are untouched."""
    from blind_insertion.assets.builder import MENAGERIE   # noqa: PLC0415
    left = mujoco.MjSpec.from_file(str(MENAGERIE / "i2rt_yam" / "yam_linear_4310.xml"))
    if physical:
        from blind_insertion.assets.flexpoint import install
        install(left)
    for k in list(left.keys) + list(left.lights) + list(left.actuators):
        left.delete(k)
    for g in left.geoms:
        g.contype = g.conaffinity = 0
    for b in left.bodies:          # parked: no actuators, so float it against gravity and damp it
        if b.name:
            b.gravcomp = 1
    for j in left.joints:
        j.damping = [50.0, 0.0, 0.0]
    frame = spec.worldbody.add_frame(pos=[0.0, LEFT_BASE_Y, 0.0])
    frame.attach_body(left.body("arm"), "left_", "")
    spec.worldbody.add_camera(name="top", pos=list(top_cam["pos"]),
                              xyaxes=_look_xyaxes(top_cam["pos"], top_cam["look_at"], top_cam.get("up", (0, 0, 1))),
                              fovy=top_cam["fovy"])
    fix = spec.body("fixture")
    # the panel's other ports (visual), relative to the target port, from the photo:
    # [USB-C 1][USB-C 2 = target][USB-A][RJ45][HDMI over DP]
    for name, x, y, w, h in [("usbc_1", -0.015, 0.0, 0.0084, 0.0026), ("usb_a", 0.013, 0.0, 0.0045, 0.012),
                             ("rj45", 0.028, -0.002, 0.016, 0.013), ("hdmi", 0.050, 0.006, 0.015, 0.0045),
                             ("dp", 0.050, -0.006, 0.016, 0.0045)]:
        fix.add_geom(name=f"vis_{name}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[w / 2, h / 2, 0.0003],
                     pos=[x, y, 0.0003], rgba=[0.05, 0.05, 0.06, 1], contype=0, conaffinity=0)


def build_scene(variant=None, top_cam=TOP_CAM, *, detailed=True, physical=True):
    """CPU model of the full task H scene (pick + insert), from the same builder as PegEnv."""
    from blind_insertion.assets import builder as B   # noqa: PLC0415
    variant = variant or B.reader_variant()
    def decorate(sp):
        add_scene(sp, top_cam, physical=physical)
        if detailed:
            from blind_insertion.scenes.bench_details import add_details
            add_details(sp, variant)
        if physical:
            from blind_insertion.scenes.physical_rig import decorate as decorate_rig
            decorate_rig(sp)

    m, lay = B.build("yam", [variant], pad_spheres=PAD_SPHERES, table_top=0.0, fixture_boxes=pc_case_boxes(),
                     extra=decorate, gripper="flexpoint" if physical else "linear")
    B.apply_variant_cpu(m, lay, 0)
    cfg = json.loads(CONFIG.read_text())        # the insertion env's pad contact (PegEnv._apply_starts)
    m.geom_solref[lay.pad_geoms, 0] = cfg["pad_timeconst"]
    m.geom_solimp[lay.pad_geoms, :3] = cfg["pad_solimp"]
    m.geom_friction[lay.pad_geoms, 0] = 1.0 * 1.68   # mid of finger_friction_scale (1.41, 2.0)
    # The fingers are constant-force motors; undamped they slam shut and throw a 27 g part. The real
    # gripper is speed limited: damping gives ~6 cm/s closing at 12 N (static grip force unchanged).
    for j in lay.robot.finger_joints:
        m.dof_damping[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = FINGER_DAMPING
    for j, slot in enumerate(lay.peg_slots):      # plug metal, neck black, body light grey
        z = m.geom_aabb[slot].reshape(2, 3)[0, 2] + m.geom_pos[slot, 2]
        m.geom_rgba[slot] = ([0.45, 0.45, 0.47, 1] if z < B.USBC_PLUG_LEN else   # plug metal: real ~150/255
                             [0.08, 0.08, 0.09, 1] if z < B.USBC_PLUG_LEN + variant.handle_height else [0.82, 0.83, 0.85, 1])
    if detailed:
        # Keep the original convex collision body, render its rounded visual shell.
        for slot in lay.peg_slots:
            z = m.geom_aabb[slot].reshape(2, 3)[0, 2] + m.geom_pos[slot, 2]
            if z > B.USBC_PLUG_LEN + variant.handle_height:
                m.geom_rgba[slot, 3] = 0
    for j in range(len(lay.fixture_slots)):
        m.geom_rgba[lay.fixture_slots[j]] = [0.50, 0.505, 0.51, 1]      # bench_details.SILVER
    # the arm bases' visual meshes sit exactly on the table top; MuJoCo Warp's renderer z-fights them
    # (speckles in the top camera). Lift them 0.5 mm (visual only).
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name in ("arm", "left_arm") and m.geom_contype[g] == m.geom_conaffinity[g] == 0:
            m.geom_pos[g, 2] += 5e-4
    if physical:
        from blind_insertion.scenes.physical_rig import apply
        apply(m)
    # MuJoCo Warp's renderer ignores alpha and draws every geom in groups 0-2 (its default
    # enabled_geom_groups): move invisible geoms (the reader's square collision body under its rounded
    # shell, hidden case boxes, the replaced D405 mesh) to group 3, which neither renderer draws
    m.geom_group[m.geom_rgba[:, 3] == 0] = 3
    return m, lay


def place_reader(m, d, lay, xy=READER_XY, yaw_deg=READER_YAW_DEG):
    """Reader flat on the table, plug pointing +x (rotated by yaw), body on the table."""
    from blind_insertion.envs import kinematics as K   # noqa: PLC0415
    import torch                                        # noqa: PLC0415
    R = K.rpy_mat(torch.tensor([0.0, 0.0, math.radians(yaw_deg)])) @ K.rpy_mat(
        torch.tensor(PORT_BASE_RPY_DEG) * math.pi / 180)
    v = lay.variants[0]
    bz = v.peg_boxes[0][0][2]                            # body centre along the part axis
    bt = v.peg_boxes[0][1][1]                            # body half thickness
    c = R.numpy() @ np.array([0.0, 0.0, bz])
    pa = lay.peg_qadr
    d.qpos[pa:pa + 3] = [xy[0] - c[0], xy[1] - c[1], bt + 1e-4]
    d.qpos[pa + 3:pa + 7] = K.mat_to_quat(R).numpy()


def place_pc(m, d, port_xy=PORT_XY, yaw_deg=PC_YAW_DEG):
    from blind_insertion.envs import kinematics as K   # noqa: PLC0415
    import torch                                        # noqa: PLC0415
    R = K.rpy_mat(torch.tensor([0.0, 0.0, math.radians(yaw_deg)])) @ K.rpy_mat(
        torch.tensor(PORT_BASE_RPY_DEG) * math.pi / 180)
    d.mocap_pos[0] = [port_xy[0], port_xy[1], PORT_HEIGHT]
    d.mocap_quat[0] = K.mat_to_quat(R).numpy()


R_TOP_TCP = ((0., -1., 0.), (0., 0., -1.), (1., 0., 0.))   # tcp in the port/part frame (PegEnv "top")


def grasp_R_world(yaw_deg=0.0):
    """World tcp orientation of the top-down grasp of a part whose frame is the port's base
    frame turned by yaw about the vertical (part axis along -x when yaw = 0, plug at +x)."""
    from blind_insertion.envs import kinematics as K   # noqa: PLC0415
    import torch                                        # noqa: PLC0415
    R = K.rpy_mat(torch.tensor([0.0, 0.0, math.radians(yaw_deg)])) @ K.rpy_mat(
        torch.tensor(PORT_BASE_RPY_DEG) * math.pi / 180)
    return R @ torch.tensor(R_TOP_TCP)


def task_config():
    """EnvCfg overrides for the proprio insertion policy (USB-C E settings + this scene)."""
    usbc = json.loads((CONFIG.parent / "phase_e_usbc_spheres.json").read_text())
    usbc.update(
        robot="yam", table_top=0.0, grasp_mode="top",
        fix_base_rpy_deg=list(PORT_BASE_RPY_DEG),
        fix_x=[PORT_XY[0] - 0.02, PORT_XY[0] + 0.02], fix_y=[PORT_XY[1] - 0.04, PORT_XY[1] + 0.04],
        fix_z=[PORT_HEIGHT - PLATE / 2] * 2,
        fix_rpy_range_deg=[0.5, 10.0, 0.5],         # PC yaw +-10 deg about the vertical (fixture y)
        fixture_boxes=pc_case_boxes(),
        grip_rot_deg=3.0, inhand_tilt_deg=5.0, inhand_yaw_deg=0.0, grasp_along=0.005,
        grasp_y=list(GRASP_Y["linear"]),            # legacy config; task_h_reader_real.json uses "flexpoint"
        start_gap=[0.0003, 0.010],                  # handoff: plug tip 0-10 mm before the panel
        episode_s=10.0, far_z=0.03)
    return json.loads(json.dumps(usbc), parse_float=lambda x: round(float(x), 6))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write_config", action="store_true")
    a = ap.parse_args()
    if a.write_config:
        CONFIG.write_text(json.dumps(task_config(), indent=1) + "\n")
        print("wrote", CONFIG)
