"""How close the ACT demos bring each arm body to the table, for the safety layer's clearances.

scripts/real/safety.py refuses targets with any finger tip < 20 mm (other bodies < 40 mm) above the
table, on i2rt's linear_4310 gripper model. The top-down grasp of the reader brings the FlexPoint
fingers near the table, so the runner needs task clearances (config safety.*). This replays the
joint states of ACT demo episodes through the FlexPoint YAM (the gripper on the real arms) and
prints the minimum height of each body's geometry above the table, with the fingers as recorded.

    .venv/bin/python deploy/checks/clearance.py [--dataset data/act/reader_pick_v5] [--files 4]
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from blind_insertion.assets import builder as B

BODIES = ("link_4", "link_5", "link_6", "wrist_camera", "gripper", "left_finger", "right_finger")


def body_heights(m, d, bodies):
    out = {}
    for g in range(m.ngeom):
        b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
        if b not in bodies:
            continue
        kind = "collision" if m.geom_contype[g] or m.geom_conaffinity[g] else "visual"
        R = d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            v = np.array(np.meshgrid(*[[-1, 1]] * 3)).reshape(3, -1).T * m.geom_size[g]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            v = np.array([[0, 0, 0]]) - R.T @ np.array([0, 0, m.geom_size[g][0]])
        else:
            v = None
        z = (v @ R.T[:, 2] + d.geom_xpos[g][2]).min() if v is not None else d.geom_xpos[g][2] - m.geom_rbound[g]
        key = f"{b} ({kind})"
        out[key] = min(out.get(key, np.inf), z)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/home/yashchitambar/peg_test/data/act/reader_pick_v5")
    ap.add_argument("--files", type=int, default=1)
    ap.add_argument("--stride", type=int, default=50)
    a = ap.parse_args()
    spec, info = B.load_robot("yam", gripper="flexpoint")
    m = spec.compile()
    d = mujoco.MjData(m)
    J = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]  # noqa: E731
    qa = np.array([J(j) for j in info.arm_joints])
    fa = [J("left_finger"), J("right_finger")]
    files = sorted(Path(a.dataset, "data").glob("chunk-*/file-*.parquet"))[:a.files]
    rows = []
    for f in files:
        st = np.stack(pd.read_parquet(f, columns=["observation.state"])["observation.state"].to_numpy())[::a.stride]
        for s in st:
            d.qpos[:] = 0
            d.qpos[qa] = s[:6]
            d.qpos[fa] = s[6]
            mujoco.mj_kinematics(m, d)
            rows.append(body_heights(m, d, BODIES))
    df = pd.DataFrame(rows) * 1e3
    print(f"{len(df)} demo states ({len(files)} files, every {a.stride}th frame): height above the table (mm)")
    print(df.describe(percentiles=[0.001, 0.01, 0.05]).T[["min", "0.1%", "1%", "5%"]].round(1).to_string())


if __name__ == "__main__":
    main()
