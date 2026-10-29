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

FINGER_DAMPING = 200.0            # N s/m per finger in the pick scene (see build_scene)
LEFT_BASE_Y = 0.62                # idle left arm base (bimanual mount, arms at y = +-0.31)
LEFT_HOME = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # folded rest (out of the top camera)
READER_XY = (0.203937, 0.220908)           # nominal reader centre (body) on the table
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


def add_scene(spec: mujoco.MjSpec, top_cam=TOP_CAM, physical=True, bimanual=False):
    """`builder.build(extra=...)` hook: the idle left arm (visual), the fixed top camera, the
    PC's other ports and colours. Physics of the task arm, port and reader are untouched."""
    from blind_insertion.assets.builder import MENAGERIE   # noqa: PLC0415
    left = mujoco.MjSpec.from_file(str(MENAGERIE / "i2rt_yam" / "yam_linear_4310.xml"))
    if physical:
        from blind_insertion.assets.flexpoint import install
        install(left)
    if bimanual:
        from blind_insertion.assets.builder import load_robot
        left, _ = load_robot("yam", pad_spheres=4,
                             gripper="flexpoint" if physical else "linear")
    else:
        for k in list(left.keys) + list(left.lights) + list(left.actuators):
            left.delete(k)
        for g in left.geoms:
            g.contype = g.conaffinity = 0
        for b in left.bodies:
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


def _free_pc_spec(spec, profile):
    """Turn the fixture into an experimental free PC with four approximate feet.

    Uniform-box inertia and foot inset/friction are assumptions, not measurements. This is
    a CPU scene for physics evaluation; existing Warp environments still use fixed fixtures.
    """
    vals = profile['pc']['measurements']
    w, h, depth = (float(vals[k]['value']) for k in ('width', 'body_height', 'depth'))
    mass = float(vals['mass']['value'])
    foot_h = float(profile['pc']['pad_height_derived']['value'])
    radius = float(vals['pad_diameter']['value']) / 2
    if (min(w, h, depth, mass, foot_h, radius) <= 0 or radius * 2 >= min(w, depth)
            or int(vals['pad_count']['value']) != 4 or PORT_HEIGHT - foot_h <= PLATE / 2):
        raise ValueError('invalid PC dimensions, mass or feet')
    fix = next(body for body in spec.bodies if body.name == 'fixture')
    fix.mocap = False
    fix.add_freejoint(name='pc_free_joint')
    fix.explicitinertial = True
    fix.mass = mass
    # Box center in fixture coordinates: x across panel, y vertical, z into PC.
    fix.ipos = [w / 2 - PORT_FROM_LEFT, -PORT_HEIGHT + foot_h + h / 2, -depth / 2]
    fix.iquat = [1, 0, 0, 0]
    fix.inertia = [mass * (h*h + depth*depth) / 12,
                   mass * (w*w + depth*depth) / 12,
                   mass * (w*w + h*h) / 12]
    # Fixture/table previously shared a non-colliding bit. Add a dedicated support bit.
    support_bit = 16
    table = next(geom for geom in spec.geoms if geom.name == 'table')
    table.conaffinity |= support_bit
    for geom in fix.geoms:
        if geom.name.startswith(('fix_', 'fixbox_')):
            geom.contype |= support_bit
    for x in (-PORT_FROM_LEFT + radius, w - PORT_FROM_LEFT - radius):
        for z in (-radius, -depth + radius):
            fix.add_geom(name=f'pc_pad_{len(fix.geoms)}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         pos=[x, -PORT_HEIGHT + foot_h / 2, z],
                         quat=[math.sqrt(.5), math.sqrt(.5), 0, 0], size=[radius, foot_h / 2, 0],
                         rgba=[.1, .1, .1, 1], mass=0, contype=support_bit, conaffinity=1,
                         priority=2, friction=[.6, .01, .001], condim=4)


def build_scene(variant=None, top_cam=TOP_CAM, *, detailed=True, physical=True,
                support_mode='fixed', profile=None, bimanual=False, table_top=0.0, measured_support=False):
    """CPU scene; ``free`` is an experimental measured-profile PC, not a training mode."""
    from blind_insertion.assets import builder as B   # noqa: PLC0415
    if support_mode not in ('fixed', 'free'):
        raise ValueError(f'unsupported PC support mode: {support_mode}')
    if support_mode == 'free':
        if profile is None or profile.get('schema') != 'usbc-rig-profile-v0':
            raise ValueError('free PC scene requires an explicit provisional measured profile')
        if detailed:
            raise ValueError('legacy decorative PC geometry is not compatible with the measured free scene; use detailed=False')
        r = profile['reader']['measurements']
        pc = profile['pc']['measurements']
        feet_h = float(profile['pc']['pad_height_derived']['value'])
        size = (float(pc['width']['value']), float(pc['body_height']['value']), float(pc['depth']['value']))
        boxes = pc_case_boxes(port_h=PORT_HEIGHT - feet_h, size=size)
        variant = variant or B.reader_variant(body=tuple(float(r[k]['value']) for k in ('body_width','body_thickness','body_length')),
                                               neck=tuple(float(r[k]['value']) for k in ('neck_width','neck_thickness','neck_length')),
                                               mass=float(r['mass']['value']))
    else:
        boxes = pc_case_boxes()
        variant = variant or B.reader_variant()
    def decorate(sp):
        add_scene(sp, top_cam, physical=physical, bimanual=bimanual)
        if detailed:
            from blind_insertion.scenes.bench_details import add_details
            add_details(sp, variant)
        if physical:
            from blind_insertion.scenes.physical_rig import decorate as decorate_rig
            decorate_rig(sp)
        if support_mode == 'free':
            _free_pc_spec(sp, profile)
        if measured_support:
            from blind_insertion.scenes.measured_support import apply as apply_support
            apply_support(sp)

    m, lay = B.build("yam", [variant], pad_spheres=4, table_top=table_top, fixture_boxes=boxes,
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
    if bimanual:
        for side in ('left', 'right'):
            m.dof_damping[m.joint('left_' + side + '_finger').dofadr] = FINGER_DAMPING
        for gid in range(m.ngeom):
            if m.geom(gid).name.startswith(('left_pad_s', 'right_pad_s', 'left_left_pad_s', 'left_right_pad_s')):
                m.geom_contype[gid], m.geom_conaffinity[gid] = B.GRIP_BITS
                m.geom_solref[gid, 0] = cfg['pad_timeconst']
                m.geom_solimp[gid, :3] = cfg['pad_solimp']
                m.geom_friction[gid, 0] = 1.68
        # Measured hardware derivative-gain cap, only in the new bimanual mode.
        for prefix in ('', 'left_'):
            for i in range(1, 4):
                m.actuator_biasprm[m.actuator(prefix + f'joint{i}').id, 2] = -5.
    for j, slot in enumerate(lay.peg_slots):      # plug metal, neck black, body light grey
        z = m.geom_aabb[slot].reshape(2, 3)[0, 2] + m.geom_pos[slot, 2]
        m.geom_rgba[slot] = ([0.75, 0.75, 0.78, 1] if z < B.USBC_PLUG_LEN else
                             [0.08, 0.08, 0.09, 1] if z < B.USBC_PLUG_LEN + variant.handle_height else [0.82, 0.83, 0.85, 1])
    if detailed:
        # Keep the original convex collision body, render its rounded visual shell.
        for slot in lay.peg_slots:
            z = m.geom_aabb[slot].reshape(2, 3)[0, 2] + m.geom_pos[slot, 2]
            if z > B.USBC_PLUG_LEN + variant.handle_height:
                m.geom_rgba[slot, 3] = 0
    for j in range(len(lay.fixture_slots)):
        m.geom_rgba[lay.fixture_slots[j]] = [0.64, 0.65, 0.66, 1]
    if physical:
        from blind_insertion.scenes.physical_rig import apply
        apply(m)
    return m, lay


def place_reader(m, d, lay, xy=READER_XY, yaw_deg=0.0, table_top=None):
    """Reader flat on the table, plug pointing +x (rotated by yaw), body on the table."""
    from blind_insertion.envs import kinematics as K   # noqa: PLC0415
    import torch                                        # noqa: PLC0415
    R = K.rpy_mat(torch.tensor([0.0, 0.0, math.radians(yaw_deg)])) @ K.rpy_mat(
        torch.tensor(PORT_BASE_RPY_DEG) * math.pi / 180)
    if table_top is None:
        table_top = float(m.geom("table").pos[2] + m.geom("table").size[2])
    v = lay.variants[0]
    bz = v.peg_boxes[0][0][2]                            # body centre along the part axis
    bt = v.peg_boxes[0][1][1]                            # body half thickness
    c = R.numpy() @ np.array([0.0, 0.0, bz])
    pa = lay.peg_qadr
    d.qpos[pa:pa + 3] = [xy[0] - c[0], xy[1] - c[1], table_top + bt + 1e-4]
    d.qpos[pa + 3:pa + 7] = K.mat_to_quat(R).numpy()


def place_pc(m, d, port_xy=PORT_XY, yaw_deg=0.0, table_top=None):
    from blind_insertion.envs import kinematics as K   # noqa: PLC0415
    import torch                                        # noqa: PLC0415
    R = K.rpy_mat(torch.tensor([0.0, 0.0, math.radians(yaw_deg)])) @ K.rpy_mat(
        torch.tensor(PORT_BASE_RPY_DEG) * math.pi / 180)
    if table_top is None:
        table_top = float(m.geom("table").pos[2] + m.geom("table").size[2])
    position = [port_xy[0], port_xy[1], table_top + PORT_HEIGHT]
    quat = K.mat_to_quat(R).numpy()
    body = m.body('fixture').id
    mid = int(m.body_mocapid[body])
    if mid >= 0:
        d.mocap_pos[mid] = position
        d.mocap_quat[mid] = quat
    else:
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'pc_free_joint')
        if joint < 0:
            raise ValueError('fixture has neither mocap nor PC free joint')
        qadr, dadr = m.jnt_qposadr[joint], m.jnt_dofadr[joint]
        d.qpos[qadr:qadr + 7] = [*position, *quat]
        d.qvel[dadr:dadr + 6] = 0


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
        grasp_y=[0.0048, 0.0060],                   # tcp above the body centre: fingertips (8.6 mm below the
                                                    # tcp) clear the table, pads cover 2.3-8.8 mm of the 8.83 mm body
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
