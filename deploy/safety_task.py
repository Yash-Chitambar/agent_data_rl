"""Task H safety checker: scripts/real/safety.Checker's joint box and table clearances on the FlexPoint
YAM (the gripper on the real arms, with the wrist camera body), in the model frame.

safety.Checker loads i2rt's linear_4310 gripper and checks the motor reading directly. For the top-down
reader grasp that is wrong twice: the FlexPoint fingers are shaped differently, and the joint offsets
(up to 1.3 deg on joints 2-5) move the fingertips ~10 mm at the grasp. FlexChecker adds the offsets
before forward kinematics and takes per-body clearances (config safety.*; deploy/checks/clearance.py
and the dry run's true tip heights give the numbers). SafeArm calls `violation(q_motor)` unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "scripts/real"))   # after REPO: its deploy.py must not shadow this package
import safety  # noqa: E402
from blind_insertion.assets import builder as B  # noqa: E402

TIP_BODIES = ("left_finger", "right_finger")


class FlexChecker(safety.Checker):
    def __init__(self, offsets, clearance=safety.CLEARANCE, tip_clearance=safety.TIP_CLEARANCE, body_clearance=None):
        body_clearance = body_clearance or {}
        spec, _ = B.load_robot("yam", gripper="flexpoint")
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.off = np.asarray(offsets, float)
        rng = self.m.jnt_range[:6].copy()
        self.lo, self.hi = rng[:, 0] - safety.LIMIT_BUFFER - self.off, rng[:, 1] + safety.LIMIT_BUFFER - self.off   # motor frame
        tip_ids = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b) for b in TIP_BODIES}
        self.geoms = []
        for g in range(self.m.ngeom):
            b = self.m.geom_bodyid[g]
            if b == 0 or self.m.body_parentid[b] == 0:   # world and the arm base: fixed
                continue
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b)
            if name == "link_1":                          # only yaws: cannot move toward the table
                continue
            verts = None
            if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[g]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                verts = self.m.mesh_vert[a:a + n].copy()
            elif self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                verts = np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T * self.m.geom_size[g]
            zmin = safety.TABLE_Z + body_clearance.get(name, tip_clearance if b in tip_ids else clearance)
            self.geoms.append((g, verts, zmin))
        self.qadr = np.array([self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
                              for i in range(1, 7)])

    def heights(self, q6_model):
        """{body: lowest point above the table (m)} at a model-frame pose."""
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = q6_model
        mujoco.mj_kinematics(self.m, self.d)
        out = {}
        for g, verts, _ in self.geoms:
            z = (self.d.geom_xpos[g][2] - self.m.geom_rbound[g] if verts is None
                 else (verts @ self.d.geom_xmat[g].reshape(3, 3).T[:, 2] + self.d.geom_xpos[g][2]).min())
            b = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])
            out[b] = min(out.get(b, np.inf), float(z))
        return out

    def violation(self, q6):
        """q6: motor frame (what SafeArm reads and sends)."""
        q6 = np.asarray(q6, float)
        if not self.in_box(q6):
            j = int(np.argmax(np.maximum(self.lo - q6, q6 - self.hi)))
            return f"joint {j + 1} = {q6[j]:+.3f} outside [{self.lo[j]:+.3f}, {self.hi[j]:+.3f}] (motor frame)"
        self.d.qpos[:] = 0
        self.d.qpos[self.qadr] = q6 + self.off
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
