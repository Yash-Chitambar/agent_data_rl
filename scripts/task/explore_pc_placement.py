"""Where should the PC (and the SD reader) sit so that the bimanual task is most reachable?

Task: the right arm inserts the reader into the PC's second USB-C port (`usbc_2_target`,
port panel facing the arm bases, insertion along +x); the left arm holds the PC, either
bracing its rear face (wall side) or pressing its top. For every PC pose on a grid
(port x, y, yaw) this solves, on the CPU model with the measured mounts and joint limits:
  right: IK along the insertion line (seated, 6, 30, 60 mm out), reader in hand at the
         geometric grasp offset, plus in-hand tilts of +/-8 deg at the seated pose;
  left:  IK for brace/press candidates (rear face 3 mm off, mid height; top rear half 5 mm
         above), several gripper yaws; the best one is kept.
A pose counts as feasible when every IK converges (1 mm, 1 deg), joints stay >= 0.05 rad
inside their limits and, at the final poses, no arm link / gripper touches the other arm,
the table or the PC (reader-in-port and in-hand contacts excepted; 5 mm margin). Score:
min(right, left) dexterity (smallest singular value of the 6x6 Jacobian, rotation rows x 0.1 m)
x joint-margin factor. The recommended nominal maximises the WORST score over a box of
+/-`--box-cm` and +/-`--box-deg` around it (all of it feasible), so the per-episode
randomization stays inside the good region.

Then, for that PC box, it maps the (cell, yaw) pairs where a top-down reader pick (approach 80 mm
above; yaw -90..90 deg in 20 deg bins) is reachable, the reader clears every PC pose of the
training box and does not touch an arm at the reset pose. Cells with >= 3 feasible yaw bins form
the reader region; the runtime samples uniformly over the feasible (cell, yaw bin) pairs.

    PYTHONPATH=. .venv/bin/python scripts/task/explore_pc_placement.py
Writes results/usbc/bimanual_20261029/placement/{report.json, maps.png}.
"""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from blind_insertion.scenes import pc_bench as S, physical_rig as R

PROFILE = Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json')
INSERT_S = (.06, .03, .006, 0.)          # TCP offsets out of the port along the insertion line
TILTS_DEG = ((8, 0), (-8, 0), (0, 8), (0, -8))
MARGIN_RAD, CONTACT_MARGIN = .05, .005
PAD_BELOW_TCP = -.0096                   # pad centroid along grasp_site +z (+z points out of the hand)
PAD_TIP = -.0032                         # pad point furthest along +z
READER_BACK, READER_FRONT, READER_HALF_W = .02, .07, .018   # reader extent about its body centre (plug +x)
READER_MARGIN = .015
YAW_BIN, MIN_YAW_BINS = 20., 3            # reader yaw bins (deg); a cell needs >= 3 feasible bins
_W = {}


def _rot(q):
    r = np.zeros(9); mujoco.mju_quat2Mat(r, np.asarray(q, float)); return r.reshape(3, 3)


class World:
    def __init__(self):
        prof = json.loads(PROFILE.read_text())
        self.m, self.lay = S.build_scene(detailed=False, bimanual=True, support_mode='free', profile=prof, measured_support=True)
        m = self.m
        self.d = mujoco.MjData(m)
        R.reference_pose(m, self.d, self.lay); S.place_pc(m, self.d); S.place_reader(m, self.d, self.lay)
        mujoco.mj_forward(m, self.d)
        self.offset = S.reader_grasp_offset(m, self.d, self.lay)
        for f in ('left', 'right'):              # right: closed on the reader; left: closed to brace/press
            self.d.qpos[m.joint(f + '_finger').qposadr] = .005
            self.d.qpos[m.joint('left_' + f + '_finger').qposadr] = 0.
        self.rest = self.d.qpos.copy()
        self.pc = prof['pc']['measurements']
        from blind_insertion.scenes.pc_rear import layout
        self.tfl = layout(prof)['target_from_left']
        self.table_top = float(m.geom('table').pos[2] + m.geom('table').size[2])
        self.arms = {}
        for side, pre in (('right', ''), ('left', 'left_')):
            j = [m.joint(pre + f'joint{i}').id for i in range(1, 7)]
            self.arms[side] = dict(qa=m.jnt_qposadr[j], da=m.jnt_dofadr[j], lim=m.jnt_range[j].copy(),
                                   site=m.site(pre + 'grasp_site').id, seeds=[self.rest[m.jnt_qposadr[j]].copy()])
        for a in self.arms.values():
            a['seeds'] += [np.clip(s, a['lim'][:, 0] + .1, a['lim'][:, 1] - .1) for s in
                           ([0, 1.0, 1.0, -.5, 0, 0], [0, 1.6, 1.3, -.9, 0, 0], [0, 2.2, 1.8, -1.2, 0, 0], [0, .6, .5, -.3, 0, 0],
                             [0, 2.6, 2.2, -1.2, 0, 0], [0, 2.9, 2.7, -1.5, 0, 0])]
        self.robot = {side: set() for side in self.arms}
        for g in range(m.ngeom):
            b = m.body(int(m.geom_bodyid[g])).name
            if b.startswith(('left_link_', 'left_left_finger', 'left_right_finger', 'left_gripper')):
                self.robot['left'].add(g)
            elif b.startswith(('link_', 'left_finger', 'right_finger', 'gripper')):
                self.robot['right'].add(g)
        self.pc_body, self.reader_body = m.body('fixture').id, m.body('peg').id
        self.m.geom_margin[list(self.robot['left'] | self.robot['right'])] = CONTACT_MARGIN
        self.pc_qadr = int(m.joint('pc_free_joint').qposadr[0])
        self.left_fingers = {m.body('left_left_finger').id, m.body('left_right_finger').id}

    def finger_gap(self, q_left):
        """Smallest left-finger to PC distance (m), None if beyond the contact margin."""
        self.d.qpos[self.arms['left']['qa']] = q_left
        mujoco.mj_forward(self.m, self.d)
        gaps = [c.dist for c in self.d.contact[:self.d.ncon]
                if {int(self.m.geom_bodyid[int(c.geom1)]), int(self.m.geom_bodyid[int(c.geom2)])} & self.left_fingers
                and self.pc_body in (int(self.m.geom_bodyid[int(c.geom1)]), int(self.m.geom_bodyid[int(c.geom2)]))]
        return min(gaps) if gaps else None

    # --- geometry -----------------------------------------------------------------
    def place_pc(self, port_xy, yaw_deg):
        S.place_pc(self.m, self.d, port_xy=port_xy, yaw_deg=yaw_deg)
        p = self.d.qpos[self.pc_qadr:self.pc_qadr + 7]
        return p[:3].copy(), _rot(p[3:])

    def pc_centre(self, port_xy, yaw):
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        off = np.array([self.pc['depth']['value'] / 2, -(self.pc['width']['value'] / 2 - self.tfl)])
        return np.asarray(port_xy) + np.array([[c, -s], [s, c]]) @ off

    def left_targets(self, port_xy, yaw):
        """Pad-centroid targets: brace (closed fingers 12 mm behind the rear face, mid height) and
        press (pad tips 7 mm above the top, rear half)."""
        cx, cy = self.pc_centre(port_xy, yaw)
        dep = self.pc['depth']['value']
        feet = self.pc['height_with_feet']['value'] - self.pc['body_height']['value']
        top = self.table_top + self.pc['height_with_feet']['value']
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out = []
        for dy in (-.025, .025):
            out.append(('brace', np.array([cx, cy, 0]) + Rz @ [dep / 2 + .012, dy, 0]
                        + [0, 0, self.table_top + feet + .5 * self.pc['body_height']['value']]))
            out.append(('press', np.array([cx, cy, 0]) + Rz @ [.3 * dep, dy, 0] + [0, 0, top + .007 + PAD_TIP - PAD_BELOW_TCP]))
        return out

    def ik_pad(self, side, point, max_tilt_deg=45, iters=100):
        """Pad centroid at `point`, tool axis as close to vertical as the reach allows (free yaw)."""
        a, m, d = self.arms[side], self.m, self.d
        body = int(m.site_bodyid[a['site']])
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        base = m.body('left_arm' if side == 'left' else 'arm').pos
        aim = np.arctan2(point[1] - base[1], point[0] - base[0])
        for q0 in [np.r_[aim, s[1:]] for s in a['seeds'][1:]] + a['seeds'][:1]:
            q = np.array(q0, float)
            for _ in range(iters):
                d.qpos[a['qa']] = q
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                Rc = d.site_xmat[a['site']].reshape(3, 3)
                pad = d.site_xpos[a['site']] + Rc[:, 2] * PAD_BELOW_TCP
                e_p = point - pad
                e_r = np.cross(Rc[:, 2], [0, 0, -1.])
                if np.linalg.norm(e_p) < 3e-4 and np.linalg.norm(e_r) < 1e-3:
                    break
                mujoco.mj_jac(m, d, jp, jr, pad, body)
                J = np.vstack([jp[:, a['da']], .05 * jr[:, a['da']]])
                e = np.r_[e_p, .05 * e_r]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                q = np.clip(q + np.clip(dq, -.3, .3), a['lim'][:, 0], a['lim'][:, 1])
            tilt = np.degrees(np.arccos(np.clip(-Rc[2, 2], -1, 1)))
            if np.linalg.norm(e_p) < 1e-3 and tilt <= max_tilt_deg:
                return q, float(tilt)
        return None, None

    # --- kinematics ---------------------------------------------------------------
    def ik(self, side, p_t, R_t, seeds, iters=80):
        a, m, d = self.arms[side], self.m, self.d
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        best = None
        base = self.m.body('left_arm' if side == 'left' else 'arm').pos
        aim = np.arctan2(p_t[1] - base[1], p_t[0] - base[0])           # joint 1 facing the target
        seeds = list(seeds) + [np.r_[aim, s[1:]] for s in self.arms[side]['seeds'][1:]]
        for q0 in seeds:
            q = np.array(q0, float)
            for _ in range(iters):
                d.qpos[a['qa']] = q
                mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
                e_p = p_t - d.site_xpos[a['site']]
                Rc = d.site_xmat[a['site']].reshape(3, 3)
                e_r = .5 * (np.cross(Rc[:, 0], R_t[:, 0]) + np.cross(Rc[:, 1], R_t[:, 1]) + np.cross(Rc[:, 2], R_t[:, 2]))
                if np.linalg.norm(e_p) < 2e-4 and np.linalg.norm(e_r) < 2e-3:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, a['site'])
                J = np.vstack([jp[:, a['da']], .1 * jr[:, a['da']]])
                e = np.r_[e_p, .1 * e_r]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-5 * np.eye(6), e)
                q = np.clip(q + np.clip(dq, -.3, .3), a['lim'][:, 0], a['lim'][:, 1])
            err = (np.linalg.norm(e_p), np.linalg.norm(e_r))
            if err[0] < 1e-3 and err[1] < np.radians(1):
                return q
            if best is None or err[0] < best[0]:
                best = (err[0], q)
        return None

    def dexterity(self, side, q):
        a, m, d = self.arms[side], self.m, self.d
        d.qpos[a['qa']] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, a['site'])
        sv = np.linalg.svd(np.vstack([jp[:, a['da']], .1 * jr[:, a['da']]]), compute_uv=False)
        margin = float(np.min(np.minimum(q - a['lim'][:, 0], a['lim'][:, 1] - q)))
        return float(sv[-1]), margin

    def collides(self, q_right, q_left, reader_pose, left_mode):
        m, d = self.m, self.d
        d.qpos[self.arms['right']['qa']] = q_right; d.qpos[self.arms['left']['qa']] = q_left
        pa = self.lay.peg_qadr
        d.qpos[pa:pa + 3] = reader_pose[0]
        q = np.zeros(4); mujoco.mju_mat2Quat(q, reader_pose[1].ravel()); d.qpos[pa + 3:pa + 7] = q
        mujoco.mj_forward(m, d)
        for c in d.contact[:d.ncon]:
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            r1 = 'right' if g1 in self.robot['right'] else 'left' if g1 in self.robot['left'] else None
            r2 = 'right' if g2 in self.robot['right'] else 'left' if g2 in self.robot['left'] else None
            if r1 is None and r2 is None:
                continue
            if r1 == r2:
                continue                           # same arm (self-collision disabled in the scene)
            other = b2 if r1 else b1
            side = r1 or r2
            if other == self.reader_body and side == 'right':
                continue                           # reader in the right hand
            if side == 'left' and other == self.pc_body and min(b1, b2) in self.left_fingers or \
               side == 'left' and other == self.pc_body and max(b1, b2) in self.left_fingers:
                continue                           # the intended hold (gap set by finger_gap)
            if c.dist > CONTACT_MARGIN - 1e-4:
                continue
            return True
        return False

    # --- evaluation -----------------------------------------------------------------
    def evaluate(self, port_xy, yaw):
        port_p, port_R = self.place_pc(port_xy, yaw)
        R_tcp = port_R @ np.asarray(S.R_TOP_TCP)
        seeds, qs = list(self.arms['right']['seeds']), []
        for s in INSERT_S:
            q = self.ik('right', port_p + port_R @ (self.offset + [0, 0, s]), R_tcp, seeds[:1] + seeds if qs else seeds)
            if q is None:
                return {'ok': False, 'why': f'right IK at {s * 1e3:.0f} mm'}
            qs.append(q); seeds = [q]
        for tx, ty in TILTS_DEG:
            Rt = port_R @ Rotation.from_euler('xy', [tx, ty], degrees=True).as_matrix()
            if self.ik('right', port_p + Rt @ self.offset, Rt @ np.asarray(S.R_TOP_TCP), [qs[-1]]) is None:
                return {'ok': False, 'why': 'right IK with in-hand tilt'}
        r_dex, r_margin = min((self.dexterity('right', q) for q in qs), key=lambda x: x[0])
        if r_margin < MARGIN_RAD:
            return {'ok': False, 'why': 'right joint margin'}
        reader = (port_p, port_R)
        best = None
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        for mode, point in self.left_targets(port_xy, yaw):
            out = np.array([c, s_, 0.]) if mode == 'brace' else np.array([0, 0, 1.])
            for _ in range(4):                     # settle a 3 mm finger-to-PC gap
                ql, tilt = self.ik_pad('left', point)
                if ql is None:
                    break
                gap = self.finger_gap(ql)
                gap = CONTACT_MARGIN + .002 if gap is None else gap
                if abs(gap - .003) < .001:
                    break
                point = point + out * (.003 - gap)
            if ql is None or abs(gap - .003) > .002:
                continue
            dex, margin = self.dexterity('left', ql)
            if margin < MARGIN_RAD or self.collides(qs[-1], ql, reader, mode) or self.collides(qs[0], ql, (port_p + port_R @ [0, 0, INSERT_S[0]], port_R), mode):
                continue
            score = dex * min(1, margin / .3)
            if best is None or score > best[0]:
                best = (score, mode, dex, margin, ql, tilt)
        if best is None:
            return {'ok': False, 'why': 'no left hold'}
        r_score = r_dex * min(1, r_margin / .3)
        return {'ok': True, 'score': float(min(r_score, best[0])), 'right_dex': r_dex, 'right_margin': r_margin,
                'left_mode': best[1], 'left_dex': best[2], 'left_margin': best[3], 'left_tilt_deg': best[5],
                'q_right': qs[-1].tolist(), 'q_left': best[4].tolist()}

    def reader_pick(self, xy, yaw):
        S.place_reader(self.m, self.d, self.lay, xy=xy, yaw_deg=yaw)
        pa = self.lay.peg_qadr
        self.d.qpos[:pa] = self.rest[:pa]                 # arms at the reset (reference) pose
        self.d.qpos[pa + 7:] = self.rest[pa + 7:]
        self.place_pc((5., 5.), 0.)                       # PC clearance is checked separately
        mujoco.mj_forward(self.m, self.d)
        for c in self.d.contact[:self.d.ncon]:            # reader must not touch an arm or the mounting plate
            b = {int(self.m.geom_bodyid[int(c.geom1)]), int(self.m.geom_bodyid[int(c.geom2)])}
            if self.reader_body in b and c.dist < .003 and (int(c.geom1) in self.robot['right'] | self.robot['left'] or int(c.geom2) in self.robot['right'] | self.robot['left']
                                                             or self.m.geom(int(c.geom1)).name == 'shared_mounting_plate' or self.m.geom(int(c.geom2)).name == 'shared_mounting_plate'):
                return 0.
        rp, rR = self.d.qpos[pa:pa + 3].copy(), _rot(self.d.qpos[pa + 3:pa + 7])
        g = rp + rR @ self.offset
        R_t = rR @ np.asarray(S.R_TOP_TCP)
        q1 = self.ik('right', g + [0, 0, .08], R_t, self.arms['right']['seeds'])
        if q1 is None:
            return 0.
        q2 = self.ik('right', g, R_t, [q1])
        if q2 is None:
            return 0.
        dex, margin = self.dexterity('right', q2)
        return dex * min(1, margin / .3) if margin >= MARGIN_RAD else 0.


def _init():
    _W['w'] = World()


def _pc(args):
    return _W['w'].evaluate(args[:2], args[2])


def _reader(args):
    return _W['w'].reader_pick(args[:2], args[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('results/usbc/bimanual_20261029/placement'))
    ap.add_argument('--step-cm', type=float, default=1.0)
    ap.add_argument('--yaw-step', type=float, default=5.0)
    ap.add_argument('--box-cm', type=float, default=3.0)
    ap.add_argument('--box-deg', type=float, default=10.0)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--nominal-yaw', type=float, default=0.,
                    help='fix the nominal PC yaw (deg; easy to reproduce by hand); nan: free')
    ap.add_argument('--reuse', action='store_true', help='reuse maps.npz from a previous run (skip the PC grid)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    w = World()
    from blind_insertion.scenes.measured_support import dimensions
    wall = dimensions()['wall_edge_x_m']
    xs = np.round(np.arange(.15, wall - w.pc['depth']['value'] - .01 + 1e-9, a.step_cm / 100), 4)
    ys = np.round(np.arange(-.15, .60 + 1e-9, a.step_cm / 100), 4)
    yaws = np.arange(-30, 30 + 1e-9, a.yaw_step)
    grid = [(x, y, t) for x in xs for y in ys for t in yaws]
    if a.reuse:
        old = json.loads((a.out / 'report.json').read_text())
        score, why = np.load(a.out / 'maps.npz')['score'], old['failure_counts']
    else:
        print(f'{len(grid)} PC poses', flush=True)
        with mp.Pool(a.workers, initializer=_init) as pool:
            res = pool.map(_pc, grid, chunksize=64)
        score = np.array([r['score'] if r['ok'] else -1. for r in res]).reshape(len(xs), len(ys), len(yaws))
        why = {}
        for r in res:
            if not r['ok']:
                why[r['why']] = why.get(r['why'], 0) + 1
    # robust score: the worst score in the +/- box around each nominal
    bx, bt = int(round(a.box_cm / a.step_cm)), int(round(a.box_deg / a.yaw_step))
    robust = np.full_like(score, -1.)
    for i in range(bx, len(xs) - bx):
        for j in range(bx, len(ys) - bx):
            for k in range(bt, len(yaws) - bt):
                robust[i, j, k] = score[i - bx:i + bx + 1, j - bx:j + bx + 1, k - bt:k + bt + 1].min()
    free_best = np.unravel_index(np.argmax(robust), robust.shape)
    if np.isnan(a.nominal_yaw):
        i, j, k = free_best
    else:
        k = int(np.argmin(abs(yaws - a.nominal_yaw)))
        i, j = np.unravel_index(np.argmax(robust[:, :, k]), robust[:, :, k].shape)
    best = w.evaluate((xs[i], ys[j]), yaws[k])
    nominal = {'port_xy_m': [float(xs[i]), float(ys[j])], 'yaw_deg': float(yaws[k]), 'robust_score': float(robust[i, j, k]),
               'score': best['score'], 'left_mode': best['left_mode'], 'pc_centre_xy_m': w.pc_centre((xs[i], ys[j]), yaws[k]).tolist(),
               'right_dex': best['right_dex'], 'left_dex': best['left_dex'], 'right_margin_rad': best['right_margin'],
               'left_margin_rad': best['left_margin'], 'q_right': best['q_right'], 'q_left': best['q_left'],
               'best_any_yaw': {'port_xy_m': [float(xs[free_best[0]]), float(ys[free_best[1]])], 'yaw_deg': float(yaws[free_best[2]]),
                                'robust_score': float(robust[free_best])},
               'robust_score_by_yaw': {f'{t:.0f}': float(robust[:, :, kk].max()) for kk, t in enumerate(yaws)}}
    print('nominal', json.dumps(nominal), flush=True)

    # reader pick map: (cell, yaw bin) pairs where a top-down pick works and the reader is clear of the PC
    rx = np.round(np.arange(.08, wall - .03 + 1e-9, .02), 3)
    ry = np.round(np.arange(-.25, .60 + 1e-9, .02), 3)
    ryaw = np.arange(-90, 90 + 1e-9, 20)
    with mp.Pool(a.workers, initializer=_init) as pool:
        rr = np.array(pool.map(_reader, [(x, y, t) for x in rx for y in ry for t in ryaw], chunksize=32)).reshape(len(rx), len(ry), len(ryaw))
    pick = rr.max(-1)
    # Everywhere the PC can be during training: convex hull of its footprint over the box extremes.
    dep, wid = w.pc['depth']['value'], w.pc['width']['value']
    hull = []
    for dx in (-a.box_cm, a.box_cm):
        for dy in (-a.box_cm, a.box_cm):
            for dt in (-a.box_deg, 0, a.box_deg):
                c, t = w.pc_centre((xs[i] + dx / 100, ys[j] + dy / 100), yaws[k] + dt), np.radians(yaws[k] + dt)
                Rz = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
                hull += [c + Rz @ [sx * dep / 2, sy * wid / 2] for sx in (-1, 1) for sy in (-1, 1)]
    hull = cv2.convexHull(np.array(hull, np.float32))
    # Reader footprint (body centre at xy, plug along +x rotated by yaw) + READER_MARGIN must stay
    # outside the hull, checked at each yaw bin's centre and edges (+/-YAW_BIN/2).
    u, v = np.meshgrid(np.linspace(-READER_BACK, READER_FRONT, 9), np.linspace(-READER_HALF_W, READER_HALF_W, 5))
    foot = np.stack([u.ravel(), v.ravel()], -1)
    ok = rr > 0
    for ii, x in enumerate(rx):
        for jj, y in enumerate(ry):
            for kk, t0 in enumerate(ryaw):
                if not ok[ii, jj, kk]:
                    continue
                for t in np.radians([t0 - YAW_BIN / 2, t0, t0 + YAW_BIN / 2]):
                    pts = np.array([x, y]) + foot @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
                    if pts[:, 0].max() > wall - .01 or \
                       max(cv2.pointPolygonTest(hull, (float(px), float(py)), True) for px, py in pts) > -READER_MARGIN:
                        ok[ii, jj, kk] = False
                        break
    free = ok.sum(-1) >= MIN_YAW_BINS
    samples = [[float(rx[ii]), float(ry[jj]), float(ryaw[kk])] for ii, jj, kk in zip(*np.nonzero(ok)) if free[ii, jj]]
    reader = None
    if samples:
        cells = np.array([s_[:2] for s_ in samples])
        centre = cells[np.argmin(np.linalg.norm(cells - cells.mean(0), axis=1))]
        reader = {'samples_xy_yaw': samples, 'cell_m': float(rx[1] - rx[0]), 'yaw_bin_deg': YAW_BIN,
                  'x_range_m': [float(cells[:, 0].min()), float(cells[:, 0].max())],
                  'y_range_m': [float(cells[:, 1].min()), float(cells[:, 1].max())],
                  'yaw_range_deg': [float(ryaw[0] - YAW_BIN / 2), float(ryaw[-1] + YAW_BIN / 2)],
                  'centre_xy_m': centre.tolist(), 'area_m2': round(int(free.sum()) * float(rx[1] - rx[0]) ** 2, 4),
                  'n_cells': int(free.sum()), 'mean_yaw_bins_per_cell': round(len(samples) / int(free.sum()), 2),
                  'clearance': f'reader footprint (+{READER_MARGIN * 100:.1f} cm) outside the PC hull for the whole yaw bin, '
                               '>= 1 cm inside the table back edge; no contact with the arms at the reset pose'}
    print('reader', json.dumps(reader), flush=True)

    report = {'method': __doc__.strip(), 'grid': {'port_x_m': xs.tolist(), 'port_y_m': ys.tolist(), 'yaw_deg': yaws.tolist()},
              'feasible_fraction': float((score > 0).mean()), 'failure_counts': why,
              'box': {'xy_cm': a.box_cm, 'yaw_deg': a.box_deg}, 'nominal': nominal, 'reader_region': reader,
              'legacy_runtime_port_xy_m': list(S.PORT_XY),
              'legacy_runtime_pose_ok': w.evaluate(S.PORT_XY, 0.)}
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    np.savez_compressed(a.out / 'maps.npz', xs=xs, ys=ys, yaws=yaws, score=score, robust=robust, rx=rx, ry=ry, ryaw=ryaw, pick=rr,
                        reader_free=free, pc_hull=hull[:, 0])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    ext = [ys[0] * 100, ys[-1] * 100, xs[0] * 100, xs[-1] * 100]
    for axi, (img, title) in zip(ax, ((score[:, :, k], f'score at yaw {yaws[k]:.0f} deg (port position)'),
                                      (robust[:, :, k], f'worst score in +/-{a.box_cm:.0f} cm, +/-{a.box_deg:.0f} deg box'))):
        im = axi.imshow(np.where(img > 0, img, np.nan), origin='lower', extent=ext, cmap='viridis', aspect='equal')
        axi.plot(ys[j] * 100, xs[i] * 100, 'r*', ms=14, label='recommended nominal')
        axi.plot(S.PORT_XY[1] * 100, S.PORT_XY[0] * 100, 'rx', ms=10, mew=2, label='legacy runtime nominal')
        axi.plot([0, 60.8], [0, -1], 'ks', ms=8, label='arm bases')
        axi.set(title=title, xlabel='y (cm, + toward left arm)', ylabel='x (cm, + toward wall)'); fig.colorbar(im, ax=axi)
        axi.invert_xaxis()
    ax[0].legend(fontsize=7, loc='lower left')
    ext2 = [ry[0] * 100, ry[-1] * 100, rx[0] * 100, rx[-1] * 100]
    im = ax[2].imshow(np.where(pick > 0, pick, np.nan), origin='lower', extent=ext2, cmap='magma', aspect='equal')
    if reader:
        ax[2].contourf(ry * 100, rx * 100, free.astype(float), levels=[.5, 1.5], colors='c', alpha=.35)
        ax[2].contour(ry * 100, rx * 100, free.astype(float), levels=[.5], colors='c', linewidths=2)
        ax[2].fill([], [], 'c', alpha=.35, label='reader region (>= 3 yaw bins)')
    h = np.vstack([hull[:, 0], hull[:1, 0]])
    ax[2].plot(h[:, 1] * 100, h[:, 0] * 100, 'r--', label='PC (whole training box)')
    ax[2].set(title='reader top-down pick, best over yaw -90..90 deg', xlabel='y (cm)', ylabel='x (cm)'); ax[2].invert_xaxis()
    ax[2].legend(fontsize=7, loc='lower left'); fig.colorbar(im, ax=ax[2])
    fig.tight_layout(); fig.savefig(a.out / 'maps.png', dpi=110)
    print(json.dumps({k: report[k] for k in ('feasible_fraction', 'failure_counts', 'legacy_runtime_pose_ok')}, indent=1))


if __name__ == '__main__':
    main()
