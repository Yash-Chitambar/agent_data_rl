"""Pose-dependent hold and right-arm motions for the wide PC placement box (docs/real/WIDE_MOTION_20261116.md).
Off by default: an absent config block, or "enabled": false, keeps today's code path unchanged.

Config block (all keys optional except enabled; defaults in DEFAULTS):

    "wide_motion": {
      "enabled": true,
      "right_clear": true,                     # after the pick: right hand to a clear pose before the left swings in
      "right_clear_dp_m": [-0.06, -0.10, 0.04],  # clear pose = lift TCP + this (right-base frame), lift orientation
      "right_clear_abs_m": [0.079, 0.297, 0.094],  # or a fixed TCP point (overrides dp; align12 configs use this)
      "right_clear_s": 1.5,                    # that move (smoothstep in joint space; / --speed)
      "transport_mode": "joint_via",           # 'cartesian' (eval_chain) | 'joint_via': joint smoothstep to the via
                                               # pose, then the Cartesian via -> look leg
      "via_back_mm": 0.0, "via_up_mm": 30.0,   # eval_chain via pose (defaults 40 / 30): straight above the look pose
      "via_high_mm": 60.0,                     # joint_via: the joint move ends this much above the via pose, then a
                                               # Cartesian descent to the via
      "hold_select": true,                     # left press point (along) and tool yaw chosen per PC estimate
      "hold_along_m": [-0.02, -0.0275, -0.015, -0.035],     # in preference order; K7 / K4 training -0.04..-0.015
      "hold_yaw_offsets_deg": [0, 15, -15],    # about the policy hold yaw (= the yaw set of every real run); K7 / K4 training +/-30
      "on_fail": "stop",                       # no candidate passes the floors: 'stop' (GuardStop before the hold) or 'best'
      "min_arm_arm_mm": 10, "min_joint_margin_rad": 0.30, "min_cam_table_mm": 3, ...   # selection floors
    }

What it changes (everything else of the chain is unchanged):
  * right clear pose (stage_pick end): the right hand leaves the lift pose for right_clear_abs_m (or lift +
    right_clear_dp_m) before the left swing. Today the lifted right fingers and the left press fingers come within 1 mm (model) at the nominal
    spot and overlap by 24 mm with the PC 25 mm nearer the bases. The transport starts from the clear pose.
  * transport: joint-space smoothstep from the clear pose to the via pose, then the Cartesian via -> look leg. The
    via pose is the look pose raised via_up_mm (via_back_mm 0): right joint 6 loses margin as the plug backs away
    from the port along a turned (negative-yaw) axis, so the via stays at the look gap.
  * left hold: LeftHold.solve over the along x yaw candidates, scored on the bimanual scene model (ClearModel)
    against the planned right poses of this run (clear pose, transport, approach, align, seat, retreat), plus the
    left swing, the SafeArm table check of the press TARGET (contact + force offset; SafeArm checks targets) and
    the left lift. The first candidate in preference order that passes every floor wins. K7 and K4 mask every
    left-arm observation (env mask_left_obs 'all') and were trained with along -0.04..-0.015 and yaw +/-30 deg;
    parse() refuses candidates outside that range.
No safety limit changes. SafeArm, the guards and the FlexChecker limits stay those of the config.
"""
from __future__ import annotations

import math

import numpy as np
import torch

SIDES = ('right', 'left')
DEFAULTS = dict(
    enabled=False,
    right_clear=False, right_clear_dp_m=[-0.06, -0.10, 0.04], right_clear_abs_m=None, right_clear_s=1.5,
    transport_mode='cartesian', via_back_mm=None, via_up_mm=None, via_high_mm=0.0, transport_max_qd=0.8,
    hold_select=False,
    hold_along_m=[-0.02, -0.0275, -0.015, -0.035],
    hold_yaw_offsets_deg=[0., 15., -15.],
    hold_across_m=[-0.0208],
    on_fail='stop',
    min_arm_arm_mm=10.0, min_joint_margin_rad=0.30, min_cam_table_mm=3.0, min_table_mm=3.0,
    min_left_pc_mm=-3.0, min_reader_table_mm=15.0)
TRAINED_ALONG = (-0.04, -0.015)
TRAINED_YAW_DEG = 30.0
TRAINED_ACROSS = -0.0208


def parse(cfg):
    """The config block with defaults, or None (off)."""
    c = cfg.get('wide_motion')
    if not c or not c.get('enabled', False):
        return None
    out = dict(DEFAULTS)
    for k, v in c.items():
        if k.startswith('_'):
            continue
        if k not in DEFAULTS:
            raise ValueError(f'wide_motion: unknown key {k!r}')
        out[k] = v
    al = [float(x) for x in out['hold_along_m']]
    if min(al) < TRAINED_ALONG[0] - 1e-9 or max(al) > TRAINED_ALONG[1] + 1e-9:
        raise ValueError(f'wide_motion.hold_along_m {al} outside the insertion policies\' training range {TRAINED_ALONG}')
    if max(abs(float(y)) for y in out['hold_yaw_offsets_deg']) > TRAINED_YAW_DEG + 1e-9:
        raise ValueError(f'wide_motion.hold_yaw_offsets_deg outside +/-{TRAINED_YAW_DEG} deg (training range)')
    if any(abs(float(x) - TRAINED_ACROSS) > 1e-9 for x in out['hold_across_m']):
        raise ValueError(f'wide_motion.hold_across_m: the insertion policies were trained at {TRAINED_ACROSS} only')
    if out['transport_mode'] not in ('cartesian', 'joint_via'):
        raise ValueError(f"wide_motion.transport_mode must be 'cartesian' or 'joint_via', not {out['transport_mode']!r}")
    if out['on_fail'] not in ('stop', 'best'):
        raise ValueError(f"wide_motion.on_fail must be 'stop' or 'best', not {out['on_fail']!r}")
    return out


def pc_of(ep, eR):
    """((x, y) m, yaw deg in the ZED convention) of a port pose."""
    p, R = ep.reshape(-1, 3)[0].cpu().numpy(), eR.reshape(-1, 3, 3)[0].cpu().numpy()
    return (float(p[0]), float(p[1])), math.degrees(math.atan2(-R[1, 2], -R[0, 2]))


# ------------------------------------------------------------------ geometry
class ClearModel:
    """Distances on the bimanual scene model at believed joint angles (model frame): every geom of each arm (links,
    gripper, D405 wrist-camera body and bracket, calibrated webcam, fingers; the visual meshes included: they are
    the real-like shapes, and the sim does not collide them), the reader rigidly in the right hand (measured mean
    in-hand pose), the PC case (visual case meshes and collision boxes) at the estimated port pose, the table top
    (reader), and the table through FlexChecker (SafeArm's own check, motor frame, per-body limits of the config).
    Sphere bounds prune the exact mj_geomDistance calls."""

    LEFT_PRESS = ('left_left_finger', 'left_right_finger')            # touch the PC on purpose (the press)
    RIGHT_GRIP = ('left_finger', 'right_finger')                       # hold the reader

    def __init__(self, scene, offsets, safety_cfg, R_sr0=None, p_sr0=None):
        import mujoco
        from blind_insertion.scenes import pc_bench as S
        from blind_insertion.envs import kinematics as K
        from deploy.safety_task import FlexChecker
        self.mj, self.S, self.K = mujoco, S, K
        m = self.m = scene.mjm
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = scene.initial_q.cpu().numpy()
        self.qa = {s: list(scene.indices[s][0]) for s in SIDES}
        self.off = np.asarray(offsets, float).reshape(2, 6)
        if R_sr0 is None:
            from blind_insertion.envs.inhand import measured_mean
            R_sr0, p_sr0 = measured_mean(device='cpu')
        self.R_sr0 = np.asarray(R_sr0.cpu() if torch.is_tensor(R_sr0) else R_sr0, float)
        self.p_sr0 = np.asarray(p_sr0.cpu() if torch.is_tensor(p_sr0) else p_sr0, float)
        self.bname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])  # noqa: E731

        def subtree(root):
            out, grow = {root}, True
            while grow:
                grow = False
                for b in range(m.nbody):
                    if m.body_parentid[b] in out and b not in out:
                        out.add(b)
                        grow = True
            return out
        g = {}
        for s, pre in (('right', ''), ('left', 'left_')):
            root = m.jnt_bodyid[m.joint(pre + 'joint1').id]
            B = subtree(root) | {m.body_parentid[root]}
            g[s] = np.array([k for k in range(m.ngeom) if m.geom_bodyid[k] in B])
        self.g = g
        self.g_pc = np.array([k for k in range(m.ngeom) if self.bname(k) == 'fixture'])
        self.g_rd = np.array([k for k in range(m.ngeom) if self.bname(k) == 'peg'])
        self.g_left_body = np.array([k for k in g['left'] if self.bname(k) not in self.LEFT_PRESS])
        self.g_right_body = np.array([k for k in g['right'] if self.bname(k) not in self.RIGHT_GRIP])
        self.g_right_all = np.concatenate([g['right'], self.g_rd])
        self.g_table = np.array([m.geom('table').id])
        self.site = m.site('grasp_site').id
        self.rd_adr = int(m.joint('peg_joint').qposadr[0])
        self._ft = np.zeros(6)
        sc = safety_cfg
        self.chk = {s: FlexChecker(np.zeros(6), clearance=sc['clearance'], tip_clearance=sc['tip_clearance'],
                                   body_clearance=sc.get('body_clearance'), table_z=sc['table_z']) for s in SIDES}
        self.S.place_pc(m, self.d, port_xy=(0.2, 0.4), yaw_deg=0.)
        mujoco.mj_kinematics(m, self.d)
        R = self.d.xmat[m.body('fixture').id].reshape(3, 3)
        self.yaw0_pc = math.degrees(math.atan2(-R[1, 2], -R[0, 2]))
        self.set_pc(None)

    def set_pc(self, pc):
        """pc: ((x, y) m, yaw deg ZED convention) of port 2, or None (PC far away)."""
        if pc is None:
            self.S.place_pc(self.m, self.d, port_xy=(5., 5.), yaw_deg=0.)
        else:
            self.S.place_pc(self.m, self.d, port_xy=tuple(pc[0]), yaw_deg=float(pc[1]) - self.yaw0_pc)

    def pose(self, qr_b, ql_b, reader=True):
        """believed (model-frame) angles of both arms; reader=True puts the reader in the right hand."""
        m, d = self.m, self.d
        d.qpos[self.qa['right']] = np.asarray(qr_b, float).reshape(6)
        d.qpos[self.qa['left']] = np.asarray(ql_b, float).reshape(6)
        self.mj.mj_kinematics(m, d)
        if reader:
            sp, sR = d.site_xpos[self.site].copy(), d.site_xmat[self.site].reshape(3, 3).copy()
            rR = sR @ self.R_sr0.T
            d.qpos[self.rd_adr:self.rd_adr + 3] = sp - rR @ self.p_sr0
            d.qpos[self.rd_adr + 3:self.rd_adr + 7] = self.K.mat_to_quat(torch.tensor(rR, dtype=torch.float32)[None])[0].numpy()
        else:
            d.qpos[self.rd_adr:self.rd_adr + 3] = [-5., 5., -5.]
        self.mj.mj_kinematics(m, d)

    def dmin(self, gA, gB, cap=0.3):
        m, d = self.m, self.d
        lb = (np.linalg.norm(d.geom_xpos[gA][:, None] - d.geom_xpos[gB][None], axis=2)
              - m.geom_rbound[gA][:, None] - m.geom_rbound[gB][None])
        best, pair = cap, None
        for f in np.argsort(lb, axis=None):
            i, j = divmod(int(f), len(gB))
            if lb[i, j] >= best:
                break
            x = self.mj.mj_geomDistance(m, d, int(gA[i]), int(gB[j]), best, self._ft)
            if x < best:
                best, pair = x, (int(gA[i]), int(gB[j]))
        return float(best), (None if pair is None else (self.bname(pair[0]), self.bname(pair[1])))

    def table_margin(self, side, q_b):
        """SafeArm table check (FlexChecker, motor frame): smallest (body bottom - its limit) in m, and the body."""
        chk = self.chk[side]
        chk.d.qpos[:] = 0
        chk.d.qpos[chk.qadr] = np.asarray(q_b, float).reshape(6) - self.off[SIDES.index(side)] + chk.off
        self.mj.mj_kinematics(chk.m, chk.d)
        best, who = 9., None
        for g, verts, zmin in chk.geoms:
            z = chk._bottom(g, verts) - zmin
            if z < best:
                best, who = z, self.mj.mj_id2name(chk.m, self.mj.mjtObj.mjOBJ_BODY, chk.m.geom_bodyid[g])
        return float(best), who

    def joint_margin(self, side, q_b):
        """distance (rad) to the SafeArm joint box (Menagerie range + LIMIT_BUFFER, motor frame) and the joint."""
        chk = self.chk[side]
        q = np.asarray(q_b, float).reshape(6) - self.off[SIDES.index(side)]
        mm = np.minimum(q - chk.lo, chk.hi - q)
        j = int(np.argmin(mm))
        return float(mm[j]), j + 1

    def right_only(self, qr_b, ql_far):
        """Right-arm quantities that do not depend on the left: right arm (no fingers) - PC, reader - table, SafeArm
        table and joint margins."""
        self.pose(qr_b, ql_far)
        return dict(right_pc=self.dmin(self.g_right_body, self.g_pc), reader_table=self.dmin(self.g_rd, self.g_table),
                    table_right=self.table_margin('right', qr_b), joint_right=self.joint_margin('right', qr_b))

    def arm_arm(self, qr_b, ql_b, reader=True):
        self.pose(qr_b, ql_b, reader)
        return self.dmin(self.g_right_all if reader else self.g['right'], self.g['left'])


# ------------------------------------------------------------------ kinematic plan of the chain's motions
class Planner:
    """The chain's scripted motions as kinematic paths (believed frame, CPU), with the chain's own geometry:
    LeftHold.solve for the hold, eval_chain's transport geometry (site_target, via pose, smoothstep legs), the
    port-axis targets of the approach / align / insertion / retreat, the left lift and the home moves."""

    def __init__(self, scene, a, consts, cfg_insert, wm, R_sr0=None, p_sr0=None):
        from types import SimpleNamespace
        from blind_insertion.envs.left_hold import Harness
        from deploy.bimanual.fastchain import FastChain
        m = scene.mjm
        self.a, self.consts, self.ic, self.wm = a, consts, cfg_insert, wm
        if R_sr0 is None:
            from blind_insertion.envs.inhand import measured_mean
            R_sr0, p_sr0 = measured_mean(device='cpu')
        self.R_sr0, self.p_sr0 = R_sr0.cpu().float(), p_sr0.cpu().float()
        self.ch = {s: FastChain(m, m.site(pre + 'grasp_site').id, np.array(scene.indices[s][0]))
                   for s, pre in (('right', ''), ('left', 'left_'))}
        self.h = SimpleNamespace(chain=self.ch, kp=scene.kp.cpu(), dev=torch.device('cpu'), n=1, rt=SimpleNamespace(mjm=m))
        self.h.jac_point = lambda side, q, point: Harness.jac_point(self.h, side, q, point)
        self.via_back = float(wm['via_back_mm'] if wm.get('via_back_mm') is not None else a.via_back)
        self.via_up = float(wm['via_up_mm'] if wm.get('via_up_mm') is not None else a.via_up)

    def site_target(self, ep, eR, gap_m, extra=None):
        look = torch.tensor([0., 0., gap_m])
        p = ep + (eR @ look[:, None])[..., 0] + (eR @ self.p_sr0[:, None])[..., 0]
        if extra is not None:
            p = p + extra
        return p, eR @ self.R_sr0

    def right_clear(self, q_lift):
        """Clear pose: lift TCP + right_clear_dp_m (right-base frame), lift orientation. Returns (q, pos err m)."""
        ch = self.ch['right']
        q = torch.as_tensor(q_lift, dtype=torch.float32).reshape(1, 6).cpu()
        p, R = ch.fk(q)
        if self.wm.get('right_clear_abs_m') is not None:      # fixed TCP position (any reader pose), lift orientation
            pt = torch.tensor(self.wm['right_clear_abs_m'], dtype=torch.float32)[None]
        else:
            pt = p + torch.tensor(self.wm['right_clear_dp_m'], dtype=torch.float32)[None]
        qc, pe, re = ch.ik(pt, R, q, iters=150)
        return qc, float(pe[0])

    def via_look(self, ep, eR, q_seed):
        """(q_via, q_look, p/R of both) with the via IK seeded from the look-pose solution (the one branch at the port)."""
        ch = self.ch['right']
        p_look, R_look = self.site_target(ep, eR, self.a.look_gap * 1e-3)
        p_via, R_via = self.site_target(ep, eR, (self.a.look_gap + self.via_back) * 1e-3, torch.tensor([0., 0., self.via_up * 1e-3]))
        q_lk, pe1, _ = ch.ik(p_look, R_look, q_seed, iters=150)
        margin = float(torch.minimum(q_lk - ch.lo, ch.hi - q_lk).min())
        if float(pe1[0]) > 1e-3 or margin < 0.05:      # seed from far away: retry from the nominal pre-insert pose
            q2, pe2_, _ = ch.ik(p_look, R_look, torch.tensor(self.consts['q_right_nominal'], dtype=torch.float32)[None], iters=150)
            m2 = float(torch.minimum(q2 - ch.lo, ch.hi - q2).min())
            if float(pe2_[0]) + 0.01 * (m2 < 0.05) < float(pe1[0]) + 0.01 * (margin < 0.05):
                q_lk, pe1 = q2, pe2_
        q_via, pe2, _ = ch.ik(p_via, R_via, q_lk, iters=150)
        return q_via, q_lk, (p_via, R_via, p_look, R_look), max(float(pe1[0]), float(pe2[0]))

    def joint_via_targets(self, ep, eR, q0):
        """joint_via: the joint-move target (the via pose, or via_high_mm above it) and the Cartesian legs after it.
        Returns (q_target, q_via, (p_high or None, p_via, R_via, p_look, R_look), IK error m)."""
        q_via, _, (p_via, R_via, p_look, R_look), err = self.via_look(ep, eR, q0)
        hi = float(self.wm.get('via_high_mm') or 0.)
        if hi <= 0:
            return q_via, q_via, (None, p_via, R_via, p_look, R_look), err
        p_hi = p_via + torch.tensor([[0., 0., hi * 1e-3]])
        q_hi, pe, _ = self.ch['right'].ik(p_hi, R_via, q_via, iters=150)
        return q_hi, q_via, (p_hi, p_via, R_via, p_look, R_look), max(err, float(pe[0]))

    def transport(self, q0, ep, eR, n_per=8):
        """Right transport as joint samples: 'cartesian' = eval_chain.Chain.stage_transport's legs (start -> above the
        via -> via -> look); 'joint_via' = joint smoothstep start -> via, then the Cartesian via -> look leg."""
        a, ch = self.a, self.ch['right']
        q = torch.as_tensor(q0, dtype=torch.float32).reshape(1, 6).cpu()
        if self.wm.get('transport_mode') == 'joint_via':
            q_hi, q_via, (p_hi, p_via, R_via, p_look, R_look), err = self.joint_via_targets(ep, eR, q)
            out = [q + (q_hi - q) * (u * u * (3 - 2 * u)) for u in np.linspace(0, 1, 2 * n_per + 1)[1:]]
            legs = ((p_hi, R_via, p_via, R_via), (p_via, R_via, p_look, R_look)) if p_hi is not None else ((p_via, R_via, p_look, R_look),)
            rest, e2 = self._legs(q_hi, legs, n_per)
            return out + rest, max(err, e2)
        up = torch.tensor([0., 0., self.via_up * 1e-3])
        p_look, R_look = self.site_target(ep, eR, a.look_gap * 1e-3)
        p_via, R_via = self.site_target(ep, eR, (a.look_gap + self.via_back) * 1e-3, up)
        p0, R0 = ch.fk(q)
        p_hi = p_via.clone()
        p_hi[:, 2] = torch.maximum(p0[:, 2], p_via[:, 2])
        return self._legs(q, ((p0, R0, p_hi, R_via), (p_hi, R_via, p_via, R_via), (p_via, R_via, p_look, R_look)), n_per)

    def _legs(self, q, segs, n_per):
        from blind_insertion.envs import kinematics as K
        ch = self.ch['right']
        out, err = [], 0.
        for (pa, Ra, pb, Rb) in segs:
            w = K.rot_log(Rb @ Ra.transpose(1, 2))
            for k in range(n_per):
                u = (k + 1) / n_per
                u = u * u * (3 - 2 * u)
                Rt = K.axis_angle_mat(w / w.norm(dim=1, keepdim=True).clamp(min=1e-9), w.norm(dim=1) * u) @ Ra
                q, pe, _ = ch.ik(pa + (pb - pa) * u, Rt, q, iters=40 if k + 1 < n_per else 80)
                err = max(err, float(pe[0]))
                out.append(q.clone())
        return out, err

    def port_path(self, q_look, ep, eR, gaps_mm):
        """Right poses with the plug tip on the port axis at the given gaps (mm; negative = inside), IK seeded along."""
        ch = self.ch['right']
        q = torch.as_tensor(q_look, dtype=torch.float32).reshape(1, 6).cpu()
        out = []
        for g in gaps_mm:
            p, R = self.site_target(ep, eR, g * 1e-3)
            q, _, _ = ch.ik(p, R, q, iters=80)
            out.append(q.clone())
        return out

    def hold_plan(self, ep, eR, along, yaw_deg, across=TRAINED_ACROSS, yaws=None):
        """LeftHold.solve with one press point and one tool yaw (deg, absolute LeftHold yaw); yaws: a yaw set (LeftHold
        picks by joint margin, today's behaviour)."""
        from blind_insertion.envs.left_hold import HoldCfg, LeftHold
        hc = HoldCfg(mode='press', force_n=self.a.hold_force, across_m=float(across), along_m=float(along),
                     yaws_deg=tuple(yaws) if yaws is not None else (float(yaw_deg),))
        lh = LeftHold(self.h, hc)
        return lh.solve(ep, eR)

    @staticmethod
    def smooth(q0, q1, n):
        q0, q1 = torch.as_tensor(q0, dtype=torch.float32).reshape(6), torch.as_tensor(q1, dtype=torch.float32).reshape(6)
        return [q0 + (q1 - q0) * float(u * u * (3 - 2 * u)) for u in np.linspace(0, 1, n + 1)[1:]]

    def left_swing(self, q_park, plan, n=10):
        """park -> stand-off (joint smoothstep, LeftHold move_s), then the guarded descent to the contact."""
        qs = plan['q_stand'].reshape(6)
        return self.smooth(q_park, qs, n) + self.smooth(qs, plan['q_contact'].reshape(6), 4)

    def left_lift(self, q_press, lift_m):
        ch = self.ch['left']
        q = torch.as_tensor(q_press, dtype=torch.float32).reshape(1, 6)
        p0, R0 = ch.fk(q)
        out = []
        for k in range(4):
            q, _, _ = ch.ik(p0 + torch.tensor([[0., 0., lift_m * (k + 1) / 4]]), R0, q, iters=20)
            out.append(q.clone())
        return out

    def right_plan(self, ep, eR, q_lift):
        """Right-arm paths of a run (believed): clear move, transport, approach (look -> stop), align and insertion
        (stop -> seat), retreat. Returns {group: [q (1, 6)]}, transport IK error (m), clear-pose IK error (m)."""
        a = self.a
        q_lift = torch.as_tensor(q_lift, dtype=torch.float32).reshape(1, 6).cpu()
        paths = {'clear': []}
        q_wait, e_clear = q_lift, 0.
        if self.wm.get('right_clear'):
            q_wait, e_clear = self.right_clear(q_lift)
            paths['clear'] = [q[None] for q in self.smooth(q_lift, q_wait, 8)]
        tr, e_tr = self.transport(q_wait, ep, eR)
        paths['transport'] = tr
        stop = float(a.stop)
        appr = self.port_path(tr[-1], ep, eR, [a.look_gap, (a.look_gap + stop) / 2, stop])
        seat_mm = 1e3 * float(self.consts['seat_depth_m'])
        port = self.port_path(appr[-1], ep, eR, [12., 6., 2., 0., -3., -seat_mm])
        paths['port'] = appr + port
        r = 1e3 * float(self.ic['retreat_m'])
        paths['release'] = self.port_path(port[-1], ep, eR, [-seat_mm + r * u for u in (0.25, 0.5, 0.75, 1.0)])
        paths['wait'] = [q_wait]
        return paths, e_tr, e_clear


# ------------------------------------------------------------------ hold selection and the whole-chain check
def right_mins(M, paths, q_far):
    """Per group minima of the right-only quantities."""
    mins = {}
    for grp in ('wait', 'transport', 'port', 'release'):
        for q in paths[grp]:
            for k, v in M.right_only(q.reshape(6), q_far).items():
                if k not in mins.setdefault(grp, {}) or v[0] < mins[grp][k][0]:
                    mins[grp][k] = v
    return mins


def candidates(wm, consts):
    """(along, yaw deg, across) in preference order: along-major, the default (along, offset 0) first."""
    ys = consts['hold_yaws_deg']
    base = float(ys[len(ys) // 2])
    return [(float(al), base + float(dy), float(ac)) for ac in wm['hold_across_m'] for al in wm['hold_along_m']
            for dy in wm['hold_yaw_offsets_deg']]


def score(P, M, plan, key, paths, rmins, q_park_left, quick=True, q_home=None):
    """Clearance minima of one hold candidate over the chain (m / rad). Groups: clear (right to the clear pose, left
    parked), swing (left park -> stand-off -> contact, right waiting), press (left press: contact pose distances,
    SafeArm table / joint checks of the contact pose AND the press target), wait / transport / port / release
    (every right pose against the latched left contact pose), release also has the left lift, home (optional:
    left home, then right home, as RealChain.home)."""
    wm = P.wm
    mins = {g: dict(v) for g, v in rmins.items()}

    def upd(grp, k, v):
        if k not in mins.setdefault(grp, {}) or v[0] < mins[grp][k][0]:
            mins[grp][k] = v

    def done():
        return quick and bool(floors(mins, wm))

    q_c = plan['q_contact'].reshape(6)
    q_f = (plan['q_contact'] + plan['dq_force']).reshape(6)
    for q in (q_c, q_f):
        upd('press', 'table_left', M.table_margin('left', q))
        upd('press', 'joint_left', M.joint_margin('left', q))
    q_wait = paths['wait'][0]
    order = [('press', None)] + [('clear', q) for q in paths['clear']] + [('swing', q) for q in P.left_swing(q_park_left, plan)] + \
        [(g, q) for g in ('wait', 'port', 'transport', 'release') for q in paths[g]]
    for grp, q in order:
        if done():
            break
        if grp == 'press':
            M.pose(q_wait.reshape(6), q_c)
            upd('press', 'left_body_pc', M.dmin(M.g_left_body, M.g_pc))
        elif grp == 'clear':
            upd('clear', 'arm_arm', M.arm_arm(q.reshape(6), q_park_left))
        elif grp == 'swing':
            upd('swing', 'arm_arm', M.arm_arm(q_wait.reshape(6), q))
            upd('swing', 'left_body_pc', M.dmin(M.g_left_body, M.g_pc))
            upd('swing', 'table_left', M.table_margin('left', q))
            upd('swing', 'joint_left', M.joint_margin('left', q))
        else:
            upd(grp, 'arm_arm', M.arm_arm(q.reshape(6), q_c))
    if not done():
        q_end = paths['release'][-1].reshape(6)
        ql = q_c
        for ql in P.left_lift(q_c, float(P.ic['left_lift_m'])):
            upd('release', 'arm_arm', M.arm_arm(q_end, ql.reshape(6), reader=False))
        if q_home is not None:
            for q in P.smooth(ql.reshape(6), q_home['left'], 12):
                upd('home', 'arm_arm', M.arm_arm(q_end, q, reader=False))
            for q in P.smooth(q_end, q_home['right'], 12):
                upd('home', 'arm_arm', M.arm_arm(q, q_home['left'], reader=False))
    f = floors(mins, wm)
    if not bool(plan['ok'][0]):
        f.append('hold IK')
    return dict(key=key, ok=not f, fails=f, mins=mins, slack=slack(mins, wm), yaw_deg=float(plan['yaw_deg'][0]))


def floors(mins, wm):
    f = []
    for g, v in mins.items():
        if 'arm_arm' in v and v['arm_arm'][0] * 1e3 < wm['min_arm_arm_mm']:
            f.append(f"arm-arm {g} {v['arm_arm'][0] * 1e3:.0f} mm {v['arm_arm'][1]}")
        for s in SIDES:
            k = 'table_' + s
            if k in v:
                lim = wm['min_cam_table_mm'] if v[k][1] == 'wrist_camera' else wm['min_table_mm']
                if s == 'right' and g == 'wait':
                    pass
                if v[k][0] * 1e3 < lim:
                    f.append(f"table {s} {g} {v[k][1]} {v[k][0] * 1e3:.1f} mm")
            k = 'joint_' + s
            if k in v and v[k][0] < wm['min_joint_margin_rad']:
                f.append(f"joint {s} {g} j{v[k][1]} {v[k][0]:.2f} rad")
        if 'reader_table' in v and v['reader_table'][0] * 1e3 < wm['min_reader_table_mm']:
            f.append(f"reader-table {g} {v['reader_table'][0] * 1e3:.0f} mm")
        if 'left_body_pc' in v and v['left_body_pc'][0] * 1e3 < wm['min_left_pc_mm']:
            f.append(f"left body-PC {g} {v['left_body_pc'][0] * 1e3:.1f} mm {v['left_body_pc'][1]}")
    return f


def slack(mins, wm):
    """Smallest margin over the floors (mm; joints x 100 mm/rad)."""
    s = []
    for g, v in mins.items():
        if 'arm_arm' in v:
            s.append(v['arm_arm'][0] * 1e3 - wm['min_arm_arm_mm'])
        for sd in SIDES:
            if 'table_' + sd in v:
                lim = wm['min_cam_table_mm'] if v['table_' + sd][1] == 'wrist_camera' else wm['min_table_mm']
                s.append(v['table_' + sd][0] * 1e3 - lim)
            if 'joint_' + sd in v:
                s.append((v['joint_' + sd][0] - wm['min_joint_margin_rad']) * 100.)
        if 'left_body_pc' in v:
            s.append(v['left_body_pc'][0] * 1e3 - wm['min_left_pc_mm'])
        if 'reader_table' in v:
            s.append(v['reader_table'][0] * 1e3 - wm['min_reader_table_mm'])
    return float(min(s)) if s else 0.


def select(P, M, ep, eR, q_lift, q_park_left, full=False, q_home=None):
    """Hold choice for one PC estimate. Returns dict(choice=(along, yaw, across) or None, ok, scored=[...], paths,
    ...). full=True scores every candidate completely (the offline check); otherwise candidates are scored in
    preference order with early exit and the first that passes wins."""
    pc = pc_of(ep, eR)
    ep, eR = ep.reshape(1, 3).cpu().float(), eR.reshape(1, 3, 3).cpu().float()
    M.set_pc(pc)
    paths, e_tr, e_clear = P.right_plan(ep, eR, q_lift)
    rm = right_mins(M, paths, q_park_left)
    out = dict(pc=pc, transport_ik_err_mm=e_tr * 1e3, clear_ik_err_mm=e_clear * 1e3, paths=paths, scored=[])
    if not P.wm.get('hold_select'):
        plan = P.hold_plan(ep, eR, P.a.hold_along, None, yaws=P.consts['hold_yaws_deg'])
        r = score(P, M, plan, (P.a.hold_along, float(plan['yaw_deg'][0]), TRAINED_ACROSS), paths, rm, q_park_left, quick=False, q_home=q_home)
        out['scored'].append(r)
        out.update(choice=r['key'], ok=r['ok'], best=r)
        return out
    for c in candidates(P.wm, P.consts):
        plan = P.hold_plan(ep, eR, c[0], c[1], c[2])
        r = score(P, M, plan, c, paths, rm, q_park_left, quick=not full, q_home=q_home if full else None)
        out['scored'].append(r)
        if r['ok'] and not full:
            break
    ok = [r for r in out['scored'] if r['ok']]
    best = ok[0] if ok else max(out['scored'], key=lambda r: r['slack'])
    if full and ok and q_home is not None:
        pass
    out.update(choice=best['key'] if (ok or P.wm['on_fail'] == 'best') else None, ok=bool(ok), best=best)
    return out


def summary(r):
    """Loggable summary of one scored candidate (mm / rad)."""
    return dict(key=list(r['key']), ok=r['ok'], fails=r['fails'][:6], slack=round(r['slack'], 1),
                mins={g: {k: [round(v[0] * (1. if k.startswith('joint') else 1e3), 3 if k.startswith('joint') else 1), v[1]]
                          for k, v in vv.items()} for g, vv in r['mins'].items()})
