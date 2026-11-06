"""Distance between the two YAM arms on duchamp3, on the bimanual scene model (deploy/bimanual/scene.SceneModel:
measured mounts, wrist cameras, fingers). Joint angles in the MOTOR (encoder) frame; the rig's joint offsets are added.

Why: 2026-11-06 gravity_id collect, pose 20: the right hand pressed on the limp left arm at rest (joint 1 blocked by
37-52 mrad at 3-4 N m). The table check (safety.Checker) has no other arm in it, and the left arm's limp pose was not
known. Use the left arm's MEASURED joints (connect it and hold it), never an assumed rest.

Model check on that run: with the left arm assumed at motor zeros the model gave 32.6 mm at the pose that touched and
1.9 mm at a pose that did not, so the model is only as good as the left pose it gets: keep a large margin (100 mm).

    d = TwoArm(); d.distance(q_right, q_left) -> (metres, (right body, left body))
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HAND = ('link_6', 'gripper', 'left_finger', 'right_finger', 'wrist_camera', 'calibrated_webcam_visual')


from contextlib import contextmanager


@contextmanager
def deploy_package():
    """Inside: `deploy` is the repo's package, even when scripts/real/deploy.py holds the name (scripts in scripts/real
    put their own directory first on sys.path). The caller's modules come back afterwards."""
    saved = {k: v for k, v in sys.modules.items() if k == 'deploy' or k.startswith('deploy.')}
    for k in saved:
        del sys.modules[k]
    path0 = list(sys.path)
    sys.path[:] = [str(REPO)] + [x for x in path0 if Path(x).resolve() != (REPO / 'scripts/real').resolve()]
    try:
        yield
    finally:
        sys.path[:] = path0
        for k in [k for k in sys.modules if k == 'deploy' or k.startswith('deploy.')]:
            del sys.modules[k]
        sys.modules.update(saved)


class TwoArm:
    def __init__(self):
        import importlib
        import mujoco
        self.mj = mujoco
        with deploy_package():
            C = importlib.import_module('deploy.conventions')
            sc = importlib.import_module('deploy.bimanual.scene').SceneModel('cpu')
            self.off = {s: C.joint_offsets(s) for s in ('right', 'left')}
        m = self.m = sc.mjm
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = sc.initial_q.numpy()
        self.d.qpos[sc.pc_qadr:sc.pc_qadr + 3] = [5.0, 5.0, -5.0]     # the PC is off the table for these runs
        self.qa = {s: sc.indices[s][0] for s in ('right', 'left')}

        def subtree(root):
            out, grow = {root}, True
            while grow:
                grow = False
                for b in range(m.nbody):
                    if m.body_parentid[b] in out and b not in out:
                        out.add(b)
                        grow = True
            return out
        root = {s: m.jnt_bodyid[m.joint(('' if s == 'right' else 'left_') + 'joint1').id] for s in ('right', 'left')}
        R = subtree(root['right'])
        L = subtree(root['left']) | {m.body_parentid[root['left']]}       # with the left base
        self.gR = [g for g in range(m.ngeom) if m.geom_bodyid[g] in R]
        self.gL = [g for g in range(m.ngeom) if m.geom_bodyid[g] in L]
        self._ft = np.zeros(6)

    def body(self, g):
        return self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])

    def distance(self, q_right, q_left, cap=0.3):
        """Smallest distance (m, negative = overlap) between any right-arm geom and any left-arm or left-base geom."""
        m, d = self.m, self.d
        d.qpos[self.qa['right']] = np.asarray(q_right, float)[:6] + self.off['right']
        d.qpos[self.qa['left']] = np.asarray(q_left, float)[:6] + self.off['left']
        self.mj.mj_kinematics(m, d)
        best, pair = cap, None
        for a in self.gR:
            for b in self.gL:
                x = self.mj.mj_geomDistance(m, d, a, b, cap, self._ft)
                if x < best:
                    best, pair = x, (a, b)
        return best, (None if pair is None else (self.body(pair[0]), self.body(pair[1])))

    def right_extent(self, q_right):
        """(max y over all right-arm geoms, min x over the hand's geoms) in the scene (m; centre +- bounding radius)."""
        m, d = self.m, self.d
        d.qpos[self.qa['right']] = np.asarray(q_right, float)[:6] + self.off['right']
        self.mj.mj_kinematics(m, d)
        y = max(d.geom_xpos[g][1] + m.geom_rbound[g] for g in self.gR)
        x = min(d.geom_xpos[g][0] - m.geom_rbound[g] for g in self.gR if self.body(g) in HAND)
        return float(y), float(x)

    def hand_box(self, q_right):
        """(min, max) corners (3,) over the hand's geom centres (world, m)."""
        self.right_extent(q_right)
        P = np.array([self.d.geom_xpos[g] for g in self.gR if self.body(g) in HAND])
        return P.min(0), P.max(0)

    def path_min(self, qa, qb, q_left, n=31):
        """Smallest distance along the straight joint path qa -> qb."""
        return min(self.distance((1 - s) * np.asarray(qa, float) + s * np.asarray(qb, float), q_left)[0]
                   for s in np.linspace(0, 1, n))


class ContactGuard:
    """Stops on signs of contact while a script moves or holds one arm (2026-11-06: joint 1 blocked 37-52 mrad,
    3-4 N m, pressed on the other arm; normal holds on the same run: joint 1 within ~5.5 mrad and 0.3 N m).

    check(q, qdes, eff, moving) -> None or a reason. Joint 1 carries almost no gravity, so its effort and tracking
    error are the clean contact signal; the other joints only get a wide tracking bound (their gravity sag is large)."""

    def __init__(self, j1_err_hold=0.012, j1_eff_hold=1.2, j1_err_move=0.030, j1_eff_move=2.0,
                 any_err=(0.080, 0.080, 0.080, 0.080, 0.150, 0.150)):
        """any_err: per joint (rad). Wrist joints 5-6 run on kp 10 and trail a moving target: 2026-11-06 real moves at
        0.15 rad/s lagged up to 43 and 82 mrad there (82 tripped the old single 80 mrad bound with no contact)."""
        self.j1_err_hold, self.j1_eff_hold = j1_err_hold, j1_eff_hold
        self.j1_err_move, self.j1_eff_move = j1_err_move, j1_eff_move
        self.any_err = np.broadcast_to(np.asarray(any_err, float), (6,))

    def check(self, q, qdes, eff, moving):
        e = np.asarray(qdes, float)[:6] - np.asarray(q, float)[:6]
        eff = np.asarray(eff, float)[:6]
        lim_e, lim_f = (self.j1_err_move, self.j1_eff_move) if moving else (self.j1_err_hold, self.j1_eff_hold)
        if abs(e[0]) > lim_e:
            return f'joint 1 off its target by {e[0] * 1e3:+.1f} mrad (limit {lim_e * 1e3:.0f}): contact?'
        if np.isfinite(eff[0]) and abs(eff[0]) > lim_f:
            return f'joint 1 effort {eff[0]:+.2f} N m (limit {lim_f}): contact?'
        over = np.abs(e) / self.any_err
        j = int(np.argmax(over))
        if over[j] > 1.0:
            return f'joint {j + 1} off its target by {e[j] * 1e3:+.1f} mrad (limit {self.any_err[j] * 1e3:.0f}): contact?'
        return None
