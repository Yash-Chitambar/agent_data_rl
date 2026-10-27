"""Safety layer for the real YAM arms on duchamp3: hold-mode connect, joint box, table clearance.

Why: in i2rt's default zero-gravity (floating) mode the right arm's shoulder drifts outward under
its own gravity compensation (i2rt applies 1.1x the model torque on joints 2-3) and drove the
gripper into the table on connect (2026-10-25; CAN log: kp = 0, shoulder 0 -> 0.79 rad in 1.3 s,
stopped with the finger tip at the table). i2rt does not enforce joint limits in torque mode.

So:
  * `connect(side)` uses zero_gravity_mode=False: the arm holds its current pose (PD) from the
    first instant. Never float an arm near the table.
  * `SafeArm.command(q)` refuses a target outside the joint box or whose pose (forward
    kinematics) puts any arm mesh vertex less than CLEARANCE above the table plane (finger tips:
    TIP_CLEARANCE): nothing is sent, the arm holds its measured pose and SafetyStop is raised.
    A safe target is capped to MAX_STEP rad per joint from the measured pose.
  * `SafeArm.read()` checks the measured pose the same way (catches drift and contact).

The table plane is z = 0 in the arm's base frame: the base stands on the table (at the stop pose
above, the finger tip computed to -7 mm with foam under it).

    .venv/bin/python scripts/real/safety.py      # offline self-test, no hardware
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rig  # noqa: E402

TABLE_Z = 0.0           # table plane in the arm base frame (m)
CLEARANCE = 0.04        # min height of any arm vertex above the table, except the finger tips (m)
TIP_CLEARANCE = 0.02    # min height of the finger tips (m); insertion fixtures sit on the table
MAX_STEP = 0.10         # max joint target change from the previous command, per command (rad)
# max distance of a command from the measured pose (rad). The step cap works on the previous
# command, not the measured pose: capping on the measured pose made a joint that trails its
# command (i2rt holds the wrist at kp 10) fall ever further behind, until the tracking check
# stopped the arm (2026-10-25). This bound still refuses commands far from where the arm is.
MAX_DEVIATION = 0.40
# Joint box = model range widened by this (rad), as i2rt's own limits (get_yam_robot). The sim
# YAM policies (results/C rollouts) use joint 4 down to -1.701 (range -1.571); joints 2-3 rest on
# their stops at 0. The table check below is the real protection.
LIMIT_BUFFER = 0.15
TIP_BODIES = ("tip_left", "tip_right")
EE_MASS = 0.5           # kg, as in yam-control's rig_config (close to the model's 0.553 kg)


class SafetyStop(RuntimeError):
    pass


def _model() -> mujoco.MjModel:
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    path = combine_arm_and_gripper_xml(ArmType.from_string_name("yam"),
                                       GripperType.from_string_name("linear_4310"), ee_mass=EE_MASS)
    return mujoco.MjModel.from_xml_path(path)


class Checker:
    """Forward-kinematics table-clearance and joint-box checks on a 6-joint arm pose."""

    def __init__(self, clearance: float = CLEARANCE, tip_clearance: float = TIP_CLEARANCE,
                 body_clearance: dict | None = None):
        """clearance / tip_clearance (m) above the table; body_clearance {body name: m} overrides
        single bodies, e.g. more margin on the wrist and gripper for parts the model lacks (the
        wrist webcam and its cable)."""
        body_clearance = body_clearance or {}
        self.m = _model()
        self.d = mujoco.MjData(self.m)
        rng = self.m.jnt_range[:6].copy()
        self.lo, self.hi = rng[:, 0] - LIMIT_BUFFER, rng[:, 1] + LIMIT_BUFFER
        tip_ids = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TIP_BODIES}
        self.geoms = []  # (geom id, local vertices or None, min z)
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            if b == 0 or self.m.body_parentid[b] == 0:  # world (the base mesh) and link1, which
                continue                                # only yaws: neither can move toward the table
            verts = None
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                verts = self.m.mesh_vert[a:a + n].copy()
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
            zmin = TABLE_Z + body_clearance.get(name, tip_clearance if b in tip_ids else clearance)
            self.geoms.append((g, verts, zmin))

    def clip(self, q6: np.ndarray) -> np.ndarray:
        return np.clip(q6, self.lo, self.hi)

    def in_box(self, q6: np.ndarray) -> bool:
        return bool(np.all(q6 >= self.lo - 1e-9) and np.all(q6 <= self.hi + 1e-9))

    def violation(self, q6: np.ndarray) -> str | None:
        """None if the pose is safe, else a short reason."""
        if not self.in_box(q6):
            j = int(np.argmax(np.maximum(self.lo - q6, q6 - self.hi)))
            return f"joint {j + 1} = {q6[j]:+.3f} outside [{self.lo[j]:+.3f}, {self.hi[j]:+.3f}]"
        self.d.qpos[:] = 0
        self.d.qpos[:6] = q6
        mujoco.mj_kinematics(self.m, self.d)
        for g, verts, zmin in self.geoms:
            if verts is None:
                z = self.d.geom_xpos[g][2] - self.m.geom_rbound[g]
            else:
                z = (verts @ self.d.geom_xmat[g].reshape(3, 3).T[:, 2] + self.d.geom_xpos[g][2]).min()
            if z < zmin:
                body = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])
                return f"{body} at z = {z * 1000:+.0f} mm < {zmin * 1000:.0f} mm above the table"
        return None


def connect(side: str, gripper_kp: float = 2.0, gripper_kd: float = 0.5):
    """i2rt robot for one arm, holding its current pose (PD) from the start. Gripper calibrates."""
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import ArmType, GripperType
    return get_yam_robot(channel=rig.can_channel(side), arm_type=ArmType.from_string_name("yam"),
                         gripper_type=GripperType.from_string_name("linear_4310"), ee_mass=EE_MASS,
                         gripper_kp=gripper_kp, gripper_kd=gripper_kd, zero_gravity_mode=False)


class SafeArm:
    """Wraps an i2rt robot (7 values: 6 joints + gripper 0..1). All commands go through checks."""

    def __init__(self, robot, checker: Checker | None = None):
        """`checker` guards every command and measured pose. Planners that use a stricter
        Checker for their targets should give SafeArm a looser one here (a margin), or the
        arm settling a millimetre lower than planned stops it and blocks the way home."""
        self.robot = robot
        self.check = checker or Checker()
        q = self.read()  # raises if the arm already starts in an unsafe pose
        self.last_safe = q

    def _hold(self, q7: np.ndarray):
        self.robot.command_joint_pos(np.asarray(q7, float))
        self.last_safe = np.asarray(q7, float).copy()

    def read(self) -> np.ndarray:
        q = np.asarray(self.robot.get_joint_pos(), float)
        why = self.check.violation(q[:6])
        if why is not None:
            self._hold(q)
            raise SafetyStop(f"measured pose unsafe: {why}; holding")
        return q

    def command(self, q7_target, fric_comp=None) -> np.ndarray:
        """Send the target, capped to MAX_STEP per joint from the previous command and refused
        beyond MAX_DEVIATION from the measured pose; returns what was sent.

        fric_comp = (tau, ramp): Coulomb friction feed-forward on the arm joints from the sent
        target and the measured pose (blind_insertion/envs/friction_comp.py), in the command's
        `torques` field (i2rt adds it to gravity compensation; command_joint_pos zeroes it, so it is
        set right after, under the same lock: at worst one motor-thread cycle goes without it).

        An unsafe request (outside the joint box or too near the table) is refused as a whole:
        nothing is sent, the arm holds its measured pose and SafetyStop is raised. Clipping it
        instead would creep toward the unsafe pose one step per call."""
        q = self.read()
        req = np.asarray(q7_target, float)
        why = self.check.violation(req[:6])
        if why is not None:
            self._hold(q)
            raise SafetyStop(f"target unsafe: {why}; holding")
        tgt = req.copy()
        base = self.last_safe[:6]
        tgt[:6] = base + np.clip(req[:6] - base, -MAX_STEP, MAX_STEP)
        tgt[6] = float(np.clip(req[6], 0.0, 1.0))
        dev = np.abs(tgt[:6] - q[:6])
        if dev.max() > MAX_DEVIATION:
            self._hold(q)
            j = int(np.argmax(dev))
            raise SafetyStop(f"command {dev[j]:.2f} rad from the measured joint {j + 1}; holding")
        why = self.check.violation(tgt[:6])  # the capped step lies between two poses; check it too
        if why is not None:
            self._hold(q)
            raise SafetyStop(f"step unsafe: {why}; holding")
        self.robot.command_joint_pos(tgt)
        if fric_comp is not None:
            from blind_insertion.envs import friction_comp as FC   # repo root on sys.path (deploy.py)
            tau = FC.feedforward(tgt[:6] - q[:6], *fric_comp)
            with self.robot._command_lock:
                self.robot._commands.torques[:6] = tau
            self.last_ff = tau
        self.last_safe = tgt
        return tgt

    def close(self):
        self.robot.close()


if __name__ == "__main__":
    c = Checker()
    np.set_printoptions(precision=3, suppress=True)
    print("joint box lo:", c.lo, "\njoint box hi:", c.hi)
    cases = {  # poses logged on the real right arm, 2026-10-25
        "rest (folded)": [0.089, 0.001, 0.001, -0.106, -0.010, -0.027],
        "stopped on table": [0.085, 0.792, 0.007, 0.051, -0.018, -0.015],
        "halfway out": [0.09, 0.40, 0.006, -0.096, -0.01, -0.03],
    }
    for name, q in cases.items():
        print(f"{name:18s} -> {c.violation(np.array(q)) or 'OK'}")
    # how far the shoulder may swing out from rest with the elbow folded
    for q2 in np.arange(0.0, 0.85, 0.05):
        if c.violation(np.array([0.09, q2, 0.006, -0.1, -0.01, -0.03])):
            print(f"elbow folded: shoulder blocked from q2 = {q2:.2f} rad")
            break
