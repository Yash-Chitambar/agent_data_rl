"""End-to-end sim evaluation of the bimanual USB-C chain: pick -> hold -> approach -> insertion.

One continuous MuJoCo Warp simulation per batch of worlds (no teleports between stages). Each world
draws one layout (PC pose box, reader on the Part 1 strip) and full DR at reset: arm dynamics
(Coulomb x U(0.5, 1.3), gravity jitter), control (latency 0-1 step, encoder-offset error +/-1.5 deg on
joints 2-5, one draw per world kept through all stages), contact (PC feet mu U(0.8, 1), reader-table
mu U(0.4, 1), pad mu 1.2, plug-port mu U(0.1, 0.2), port solref 3-4 ms) and policy/v1 visual DR
(moderate defaults, camera pose/intrinsics error up to 2x real). RGB only.

Stages (all worlds in lockstep):
  0. ZED frame -> PC port pose (CNN + presence/validity gate; perception/zed_estimator.py).
     --zed-at start: frame at the reset pose (arms at rest, before the left arm parks; the condition the
     ZED net was trained and gated on); parked: after the left arm parks; after_hold: the hold is planned
     from the start frame, the approach from a frame taken after the hold engages.
  1. pick: pluggable (--pick expert = Part 1 privileged scripted expert, PickTask/PickExpert).
  2. hold: Part 3 guarded 25 N press (left_hold.LeftHold) planned from the ZED estimate; the right arm
     holds the reader at the lift pose.
  3. transport: joint-space move from the lift pose to a via pose (look pose + via_back along the port
     axis + via_up) and on to the look pose (reader --look-gap mm in front of the estimated port, believed
     in-hand pose = Part 1 measured mean). Not in any part's own eval (Part 2 teleports to the look pose).
  4. approach: Part 2 wrist-RGB servo (scripts/task/eval_approach.py logic, round-4 settings: wrist_v10w,
     6 waypoints to the 6 mm stop gap + 6 iterations there, roll correction, Coulomb feed-forward).
  5. insertion: Part 4 blind PPO policy (InsertionTask observation/controller/success definition: seated =
     0.2 s geometric hold with a 25 um aperture tolerance; failures: slip, PC pushed, far, non-finite),
     10 s, started from the live state (no reset).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/eval/eval_chain.py \
        --nworld 512 --batches 2 --seed 7000 --name chain_v1
Writes results/usbc/policy/chain/<name>.json (+ per-episode .csv). --demo N renders MP4s instead.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'perception'))
import warp as wp  # noqa: E402
wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from blind_insertion.envs import friction_comp as FC, kinematics as K  # noqa: E402
from blind_insertion.envs.bimanual_pick import PickExpert, PickTask, PickTaskCfg  # noqa: E402
from blind_insertion.envs.bimanual_runtime import BimanualCfg  # noqa: E402
from blind_insertion.envs.insertion_task import InsertionCfg, InsertionTask, failure_class  # noqa: E402
from blind_insertion.envs.left_hold import Harness, HarnessCfg, HoldCfg, LeftHold  # noqa: E402
from blind_insertion.eval_insertion import load_policy, wilson  # noqa: E402
from blind_insertion.perception.nets import PoseNet, TwoViewPoseNet  # noqa: E402
from blind_insertion.perception.zed_estimator import ZedEstimator  # noqa: E402
from blind_insertion.scenes import reader_grasp as G  # noqa: E402

PICK_LAYOUT = 'results/usbc/policy/pick/task_layout.json'
import os  # noqa: E402
# Part 4 best (blind PPO v6 seed 2; copy of policy-insertion runs/P4-insert-best/best.pt, outside git). Override: P4_INSERT_BEST.
INSERT_CKPT = os.environ.get('P4_INSERT_BEST', 'runs/chain/P4-insert-best_v10dr.pt')   # v10dr (hold DR); earlier: P4-insert-best_v6s2.pt
INSERT_V5 = 'runs/chain/P4-insert-proprio-v5-s0_last.pt'    # earlier: policy-insertion runs/P4-insert-proprio-v5-s0/last.pt
FRIC_COMP = {'tau': [0.0275, 0.546, 0.819, 0.0463, 0.0433, 0.0467], 'ramp': .005}   # eval_approach.py
OUT = ROOT / 'results/usbc/policy/chain'


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--batches', type=int, default=1)
    ap.add_argument('--seed', type=int, default=7000, help='held-out (Part 1 gate 20000, Part 4 1000, Part 2 1-2)')
    ap.add_argument('--name', default='chain')
    ap.add_argument('--pick', default='expert', help="pick stage: 'expert' (Part 1 scripted, privileged); ACT plugs in via PICK_STAGES")
    ap.add_argument('--zed-at', choices=('start', 'parked', 'after_hold'), default='start')
    ap.add_argument('--hold-along', type=float, default=-.02, help='press point from the case centre (m). -0.02: the insertion policy training value (chain 99.80 %%, n = 1024); -0.035: Part 3 round 4 recommendation (chain 96-98 %%: the blind policy reads the left encoders)')
    ap.add_argument('--hold-force', type=float, default=25.)
    ap.add_argument('--hold-yaws', choices=('policy', 'all'), default='policy',
                    help="left tool-yaw candidates: 'policy' = the insertion policy's training set (best nominal yaw +/-15 deg), 'all' = LeftHold default")
    ap.add_argument('--look-gap', type=float, default=30.)
    ap.add_argument('--via-back', type=float, default=40., help='transport via pose: further out along the port axis (mm)')
    ap.add_argument('--via-up', type=float, default=30., help='transport via pose: higher (mm)')
    ap.add_argument('--transport-s', type=float, nargs=2, default=(2.5, 1.0), help='turn/translate segment, then each of the two descent segments (s)')
    ap.add_argument('--stop', type=float, default=6.)
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--final-iters', type=int, default=6)
    ap.add_argument('--final-pos-only', type=int, default=2)
    ap.add_argument('--settle-s', type=float, default=.3)
    ap.add_argument('--ramp-s', type=float, default=.5)
    ap.add_argument('--grasp-settle-s', type=float, default=1.)
    ap.add_argument('--wrist-net', default='wrist_v10w')
    ap.add_argument('--fine-net', default='wrist_v10ws', help='stop-gap specialist used at waypoints <= --fine-below-mm (round 4 final)')
    ap.add_argument('--fine-below-mm', type=float, default=7., help='0: coarse net only')
    ap.add_argument('--servo', choices=('wrist', 'oracle'), default='wrist', help='oracle: privileged relative pose (diagnostic)')
    ap.add_argument('--handover', choices=('cmd', 'measured'), default='cmd',
                    help='insertion start target: the approach command (holds the pose) or the measured angles (as at an InsertionTask reset)')
    ap.add_argument('--insert-reader-mu', type=float, default=-1.,
                    help='reader geom friction from the insertion handover on (Part 4 trained with 0.5; Part 1 samples U(0.4, 1) for '
                         'reader-table). <= 0: keep the per-world pick value')
    ap.add_argument('--policy', default=INSERT_CKPT)
    ap.add_argument('--insert-s', type=float, default=10.)
    ap.add_argument('--visual-dr', type=int, default=1)
    ap.add_argument('--demo', type=int, default=0, help='render this many demo videos (success + failures) instead of the stats run')
    ap.add_argument('--demo-worlds', type=int, default=24)
    return ap.parse_args(argv)


def wrist_loader(name, dev):
    ck = torch.load(ROOT / f'runs/perception/{name}.pt', map_location=dev)
    net = (TwoViewPoseNet if ck['which'] == 'wrist2' else PoseNet)(len(ck['scale']), ck['size'], width=ck.get('width', 32)).to(dev)
    net.load_state_dict(ck['net']); net.eval()
    return net, ck['scale'].to(dev), ck.get('wrist_mode', 'resize')


def stats(x, pct=(50, 90, 99, 100)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {f'p{p}': float(np.percentile(x, p)) for p in pct} | {'mean': float(x.mean()), 'n': int(len(x))} if len(x) else {'n': 0}


class Chain:
    def __init__(self, a):
        self.a = a
        dev = 'cuda:0'
        ck = torch.load(ROOT / a.policy, map_location=dev, weights_only=False)
        over = json.loads(ck.get('env_json') or '{}')
        icfg = InsertionCfg()
        for k, v in over.items():
            target = icfg if hasattr(icfg, k) else icfg.task if hasattr(icfg.task, k) else icfg.task.physics
            setattr(target, k, tuple(v) if isinstance(v, list) else v)
        icfg.task.episode_s = a.insert_s
        ph = icfg.task.physics
        ph.nworld, ph.device = a.nworld, dev
        ph.render_rgb, ph.visual_dr, ph.camera_resolution, ph.measured_camera_cadence = True, bool(a.visual_dr) or None, 320, False
        ph.task_layout = PICK_LAYOUT
        self.env = env = InsertionTask(icfg, seed=a.seed)
        self.rt = rt = env.runtime
        rt.cfg.render_rgb = False                 # task observations must not render every step
        env.reset = lambda *args, **kw: None      # no auto-reset: one continuous episode per world
        self.dev, self.n, self.dt = rt.device, a.nworld, env.dt
        pcfg = PickTaskCfg(physics=BimanualCfg(nworld=a.nworld, device=dev, task_layout=PICK_LAYOUT), episode_s=13.)
        self.pick = PickTask(pcfg, runtime=rt)
        self.h = Harness(rt, HarnessCfg())
        self.zed = ZedEstimator(dev, pose=str(ROOT / 'runs/perception/zed.pt'), presence=str(ROOT / 'runs/perception/zed_presence.pt'))
        self.wrist = wrist_loader(a.wrist_net, dev)
        self.fine = wrist_loader(a.fine_net, dev) if a.fine_below_mm > 0 else None
        self.ac, self.ck = load_policy(ROOT / a.policy, dev)
        self.fric = FC.parse(FRIC_COMP, side='right')
        from blind_insertion.envs.inhand import measured_mean
        self.R_sr0, self.p_sr0 = measured_mean(device=dev)      # believed in-hand pose (site in reader)
        self.rec = None
        self.left_ref = rt.initial_q[rt.indices['left'][0]].clone()

    # ------------------------------------------------------------------ helpers
    def render(self):
        self.rt.cfg.render_rgb = True
        try:
            return self.rt.render()
        finally:
            self.rt.cfg.render_rgb = False

    def forward(self):
        with wp.ScopedDevice(self.rt.cfg.device):
            mjw.forward(self.rt.m, self.rt.d)

    def tick(self, label):
        if self.rec is not None:
            self.rec['q'].append(self.rt.q[self.rec['ids']].cpu().numpy().copy())
            self.rec['label'].append(label)

    def rel_pose(self):
        pp, pR = self.h.pc_pose()
        tip, rR = self.h.reader_pose()
        return (pR.transpose(1, 2) @ (tip - pp)[..., None])[..., 0], K.rot_log(pR.transpose(1, 2) @ rR)

    def inhand(self):
        """Reader in the right grasp_site frame (p, R) from the true state."""
        return self.pick.inhand()

    def hstep(self, qr, ql, label, grip_r=-1., grip_l=-1.):
        self.h.step(qr, grip_r, ql, grip_l)
        self.tick(label)

    # ------------------------------------------------------------------ stages
    def zed_estimate(self):
        fr = self.render()['top']
        return self.zed(fr)

    def stage_pick(self, log):
        task = self.pick
        ex = PickExpert(task, seed=self.seed)
        ex.reset()
        N, dev = self.n, self.dev
        start_hold = {'p': torch.zeros(N, 3, device=dev), 'R': torch.eye(3, device=dev).repeat(N, 1, 1), 'tilt': torch.zeros(N, device=dev)}
        for k in range(task.max_steps - 1):
            prev = ex.phase.clone()
            task.step(ex.act())
            self.tick('1 pick')
            entered = (ex.phase == 5) & (prev == 4)
            if entered.any():
                m = task.pick_metrics()
                start_hold['p'][entered] = m['inhand_p'][entered]; start_hold['R'][entered] = m['inhand_R'][entered]
                start_hold['tilt'][entered] = m['tilt_deg'][entered]
            if ex.done.all():
                break
        m = task.pick_metrics()
        dp = (m['inhand_p'] - start_hold['p']).norm(dim=1)
        dR = torch.rad2deg(K.rot_log(m['inhand_R'] @ start_hold['R'].transpose(1, 2)).norm(dim=1))
        ok = ex.done & ~task.failed & m['finite'] & (m['lift_m'] > .06) & (m['tilt_deg'] < 3) & (start_hold['tilt'] < 3) & (dp < 1e-3) & (dR < 1.)
        off = task.offsets[(G.TILT_DEG, False)]
        np_, nR = G.nominal_inhand(off, G.TILT_DEG, False)
        rel_p = (m['inhand_p'] - torch.tensor(np_, dtype=torch.float32, device=dev)) * 1e3
        rel_r = torch.rad2deg(K.rot_log(m['inhand_R'] @ torch.tensor(nR, dtype=torch.float32, device=dev).T))
        log.update(pick_ok=ok, pick_done=ex.done, pick_failed=task.failed, pick_ik_ok=ex.ik_ok, pick_lift_mm=m['lift_m'] * 1e3,
                   pick_tilt_deg=m['tilt_deg'], pick_steps=torch.full((N,), k + 1, device=dev),
                   **{f'pick_inhand_d{c}_mm': rel_p[:, i] for i, c in enumerate('xyz')},
                   **{f'pick_inhand_r{c}_deg': rel_r[:, i] for i, c in enumerate('xyz')})
        # hand over to the absolute-target harness: same control DR draw, same latency buffer
        h = self.h
        h.enc_error.copy_(task.enc_error); h.latency.copy_(task.latency); h.hist.copy_(task.sent_history)
        h.fresh[:] = False; h.xfrc.zero_()
        return task.joint_target[:, 0].clone()

    def stage_hold(self, qr, ep, eR, log):
        a, h = self.a, self.h
        # Tool yaw: the insertion policy observes the left encoders and was trained with the hold yaw
        # searched within +/-15 deg of the nominal best (InsertionTask._hold_yaws). LeftHold's default
        # search (-90..90 deg) picks a left wrist (joint 6) 0.29 rad away: the blind policy then fails
        # (5/128 seated vs 128/128 from teleported starts in the same runtime).
        yaws = tuple(self.env.hold.cfg.yaws_deg) if a.hold_yaws == 'policy' else tuple(range(-90, 91, 15))
        hold = LeftHold(h, HoldCfg(mode='press', force_n=a.hold_force, across_m=-.0208, along_m=a.hold_along, yaws_deg=yaws))
        plan = hold.solve(ep, eR)
        ql0 = h.believed('left').clone()
        pc0, pcR0 = h.pc_pose()
        arm_arm = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        for k in range(int(round(hold.duration / self.dt))):
            self.hstep(qr, hold.target(k * self.dt, ql0), '2 hold')
            if k % 10 == 0:
                arm_arm |= h.contacts()['arm_arm']
        pc1, pcR1 = h.pc_pose()
        log.update(hold_plan_ok=plan['ok'], hold_latched=hold.latched, hold_arm_arm=arm_arm,
                   hold_pc_moved_mm=(pc1 - pc0).norm(dim=1) * 1e3,
                   hold_pc_rot_deg=torch.rad2deg(K.rot_log(pcR1 @ pcR0.transpose(1, 2)).norm(dim=1)))
        return hold.hold_target()

    def site_target(self, ep, eR, gap_m, extra_world=None):
        n = self.n
        look = torch.tensor([0., 0., gap_m], device=self.dev)
        p = ep + (eR @ look[:, None].expand(n, 3, 1))[..., 0] + (eR @ self.p_sr0[:, None].expand(n, 3, 1))[..., 0]
        if extra_world is not None:
            p = p + extra_world
        return p, eR @ self.R_sr0

    def ik_best(self, p, R, q_now):
        ch = self.h.chain['right']
        seeds = [q_now, torch.tensor([0, 1.6, 1.3, -.9, 0, 0], device=self.dev).expand(self.n, 6).clone(),
                 torch.tensor([1.1772, 2.1079, 1.4923, -.9551, 0., 1.1772], device=self.dev).expand(self.n, 6).clone()]
        seeds[1][:, 0] = torch.atan2(p[:, 1], p[:, 0])
        best = None
        for s in seeds:
            q, pe, re = ch.ik(p, R, s, iters=150)
            margin = torch.minimum(q - ch.lo, ch.hi - q).min(1).values
            score = pe + .05 * re + .002 * (q - q_now).norm(dim=1) + (margin < .03).float()
            if best is None:
                best = [q, pe, re, score]
            else:
                b = score < best[3]
                best = [torch.where(b[:, None], q, best[0]), torch.where(b, pe, best[1]), torch.where(b, re, best[2]), torch.where(b, score, best[3])]
        return best[0], best[1], best[2]

    def stage_transport(self, qr, q_hold, ep, eR, log):
        """Cartesian path of the believed hand pose (IK every step, seeded by the previous solution):
        lift pose -> above the via pose (turn the reader to the port heading at lift height) -> via pose
        (look pose + via_back out along the port axis + via_up) -> look pose. A joint-space move between
        the lift and look configurations swung the plug into the table (tip 3 cm below the lift path,
        in-hand rotation p50 16 deg, n = 16)."""
        a, h, n, dev = self.a, self.h, self.n, self.dev
        ch = h.chain['right']
        up = torch.tensor([0., 0., a.via_up * 1e-3], device=dev)
        p_look, R_look = self.site_target(ep, eR, a.look_gap * 1e-3)
        p_via, R_via = self.site_target(ep, eR, (a.look_gap + a.via_back) * 1e-3, up)
        p0, R0 = ch.fk(qr)
        p_hi = p_via.clone(); p_hi[:, 2] = torch.maximum(p0[:, 2], p_via[:, 2])
        segs = ((p0, R0, p_hi, R_via, a.transport_s[0]), (p_hi, R_via, p_via, R_via, a.transport_s[1]),
                (p_via, R_via, p_look, R_look, a.transport_s[1]))
        right_pc = torch.zeros(n, dtype=torch.bool, device=dev)
        ip0, iR0 = self.inhand()
        q = qr.clone()
        ik_err = torch.zeros(n, device=dev)
        ik_rot = torch.zeros(n, device=dev)
        for (pa, Ra, pb, Rb, T) in segs:
            w = K.rot_log(Rb @ Ra.transpose(1, 2))
            nr = int(round(T / self.dt))
            for k in range(nr):
                u = (k + 1) / nr; u = u * u * (3 - 2 * u)
                pt = pa + (pb - pa) * u
                Rt = K.axis_angle_mat(w / w.norm(dim=1, keepdim=True).clamp(min=1e-9), w.norm(dim=1) * u) @ Ra
                q, pe, re = ch.ik(pt, Rt, q, iters=8 if k + 1 < nr else 60)
                self.hstep(q, q_hold, '3 transport')
                if k % 5 == 0:
                    right_pc |= h.contacts()['right_pc']
            ik_err = torch.maximum(ik_err, pe); ik_rot = torch.maximum(ik_rot, re)
        ip1, iR1 = self.inhand()
        log.update(transport_ik_err_mm=ik_err * 1e3, transport_ik_rot_deg=torch.rad2deg(ik_rot), transport_right_pc_contact=right_pc,
                   transport_inhand_dp_mm=(ip1 - ip0).norm(dim=1) * 1e3,
                   transport_inhand_dr_deg=torch.rad2deg(K.rot_log(iR1 @ iR0.transpose(1, 2)).norm(dim=1)))
        return q

    def stage_approach(self, cmd, q_hold, ep, eR, log):
        a, h, dev, n = self.a, self.h, self.dev, self.n
        ch = h.chain['right']
        for k in range(int(a.grasp_settle_s / self.dt)):
            self.hstep(cmd, q_hold, '4 approach')
        tp, _ = self.rel_pose()
        log['look_lateral_mm'] = tp[:, :2].norm(dim=1) * 1e3
        stop = a.stop * 1e-3
        sched = np.geomspace(a.look_gap * 1e-3, stop, max(a.iters - 1, 1)).tolist() + [stop] * a.final_iters
        ip0, iR0 = self.inhand()
        for it, g in enumerate(sched):
            if a.servo == 'wrist':
                net, scale, mode = self.fine if (self.fine is not None and g * 1e3 <= a.fine_below_mm) else self.wrist
                fr = self.render()['wrist_cam']
                if mode == 'crop':
                    img = fr[:, 40:40 + 192, 64:64 + 192].contiguous()
                else:
                    img = F.interpolate(fr.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
                with torch.no_grad():
                    e = net(img) * scale
                rel_p_est, rel_rv_est = e[:, :3] * 1e-3, torch.deg2rad(e[:, 3:])
            else:
                rel_p_est, rel_rv_est = self.rel_pose()
            tgt_p = torch.tensor([0., 0., g], device=dev).expand(n, 3)
            rv = -rel_rv_est
            final = it >= len(sched) - a.final_iters
            if final and it >= len(sched) - a.final_pos_only:
                rv = rv * 0
            elif final:
                rv = rv * torch.tensor([0., 0., 1.], device=dev)
            dR_port = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-9), rv.norm(dim=1))
            sp_b, sR_b = ch.fk(cmd)
            tip_b = sp_b - (sR_b @ (self.R_sr0.transpose(0, 1) @ self.p_sr0)[:, None])[..., 0]
            dR_w = eR @ dR_port @ eR.transpose(1, 2)
            dt_w = (eR @ (tgt_p - rel_p_est)[..., None])[..., 0]
            new_R = dR_w @ sR_b
            new_p = tip_b + dt_w + (dR_w @ (sp_b - tip_b)[..., None])[..., 0]
            prev = cmd
            cmd, _, _ = ch.ik(new_p, new_R, cmd, iters=60)
            nr = int(a.ramp_s / self.dt)
            for k in range(nr):
                u = (k + 1) / nr; u = u * u * (3 - 2 * u)
                self.hstep(prev + (cmd - prev) * u, q_hold, '4 approach')
            for k in range(int(a.settle_s / self.dt)):
                self.hstep(cmd, q_hold, '4 approach')
        tp, trv = self.rel_pose()
        ip1, iR1 = self.inhand()
        log.update(pre_x_mm=tp[:, 0] * 1e3, pre_y_mm=tp[:, 1] * 1e3, pre_gap_mm=tp[:, 2] * 1e3, pre_lateral_mm=tp[:, :2].norm(dim=1) * 1e3,
                   pre_rx_deg=torch.rad2deg(trv[:, 0]), pre_ry_deg=torch.rad2deg(trv[:, 1]), pre_roll_deg=torch.rad2deg(trv[:, 2]),
                   pre_tilt_deg=torch.rad2deg(trv[:, :2].norm(dim=1)),
                   approach_inhand_dp_mm=(ip1 - ip0).norm(dim=1) * 1e3,
                   approach_inhand_dr_deg=torch.rad2deg(K.rot_log(iR1 @ iR0.transpose(1, 2)).norm(dim=1)))
        return cmd

    def stage_insert(self, cmd, q_hold, log):
        env, rt, h, n, dev = self.env, self.rt, self.h, self.n, self.dev
        rt.comp['right'] = None                       # the insertion policy was trained without the feed-forward
        if self.a.insert_reader_mu > 0:
            rt.geom_friction[:, self.pick.reader_geoms, 0] = self.a.insert_reader_mu
        env.enc_error.copy_(h.enc_error); env.latency.copy_(h.latency); env.sent_history.copy_(h.hist)
        meas = env._measured_model_angles()
        env.joint_target.copy_(meas)
        if self.a.handover == 'cmd':
            env.joint_target[:, 0] = cmd
        env.left_press_q.copy_(q_hold)
        env.steps.zero_(); env.seated_steps.zero_(); env.done.zero_(); env.previous_action.zero_()
        env.pc_start.copy_(rt.q[:, rt.pc_qadr:rt.pc_qadr + 7])
        env.captured.zero_(); env.strict_run.zero_(); env.strict_best.zero_()
        env.ref_rel.copy_(env._reader_in_tcp())
        tp, trv = self.rel_pose()
        env.start['err_p'] = tp.clone(); env.start['err_ang'] = trv.clone()
        env.phi.copy_(env._potential())
        o = env._actor_obs()
        env.hist = o[:, None].expand(-1, env.icfg.history, -1).clone()
        outcome = [None] * n
        t_seat = torch.full((n,), float('nan'), device=dev)
        final_rec = {}
        pc_max = torch.zeros(n, device=dev)
        for t in range(env.max_steps):
            with torch.no_grad():
                act, _ = self.ac.act_mean(env.hist)
            _, _, _, done, info = env.step(act)
            self.tick('5 insert')
            pc_max = torch.maximum(pc_max, info['pc_d'])
            if t + 1 == env.icfg.warmup_steps:
                tp, trv = self.rel_pose()
                log.update(warm_lateral_mm=tp[:, :2].norm(dim=1) * 1e3, warm_gap_mm=tp[:, 2] * 1e3, warm_tilt_deg=torch.rad2deg(trv[:, :2].norm(dim=1)))
            if 'final' in info:
                ids, f = info['final_ids'], info['final']
                cls = failure_class(f)
                for j, w in enumerate(ids.tolist()):
                    if outcome[w] is None:
                        outcome[w] = cls[j]
                        t_seat[w] = float(info['steps'][w]) * self.dt
                        final_rec[w] = (float(f['p'][j, 0]), float(f['p'][j, 1]), float(f['p'][j, 2]), float(f['tilt'][j]))
            if all(o_ is not None for o_ in outcome):
                break
        outcome = [o_ or 'timeout' for o_ in outcome]
        log.update(insert_success=torch.tensor([o_ == 'seated' for o_ in outcome], device=dev), insert_t_s=t_seat, insert_pc_max_mm=pc_max * 1e3,
                   insert_end_z_mm=torch.tensor([final_rec.get(w, (np.nan,) * 4)[2] * 1e3 for w in range(n)], device=dev))
        log['insert_outcome'] = outcome

    # ------------------------------------------------------------------ episode
    def run_batch(self, seed, record_ids=None):
        a, rt, h, dev, n = self.a, self.rt, self.h, self.dev, self.n
        self.seed = seed
        torch.manual_seed(seed)
        if hasattr(rt.visual, 'gen'):
            rt.visual.gen.manual_seed(seed)
        self.rec = None if record_ids is None else {'ids': torch.as_tensor(record_ids, device=dev), 'q': [], 'label': []}
        rt.comp['right'] = None
        self.pick.reset()                                  # layout, physics/contact/visual DR, control DR, left arm parked
        log = {}
        p_true, R_true = h.pc_pose()
        yaw_true = torch.atan2(-R_true[:, 1, 2], -R_true[:, 0, 2])
        log['pc_x'], log['pc_y'], log['pc_yaw_deg'] = p_true[:, 0], p_true[:, 1], torch.rad2deg(yaw_true)
        r0 = self.pick.reader_pose_q()
        log['reader_x'], log['reader_y'] = r0[:, 0], r0[:, 1]
        # ZED frames: at the reset pose (left arm at rest) and parked
        # 'start' = the rest pose the ZED net was trained on: left arm at the reference pose and the right
        # gripper fully open. With the pick's 24 mm approach opening in view the ZED estimate is biased
        # ~3 mm (p50 3.0 vs 0.8 mm, n = 256): the frame is taken before the fingers close to it.
        la, fq = rt.indices['left'][0], self.pick.finger_qadr
        q_park, q_f = rt.q[:, la].clone(), rt.q[:, fq].clone()
        rt.q[:, la] = self.left_ref
        rt.q[:, fq] = .0475
        self.forward()
        est = {'start': self.zed_estimate()}
        rt.q[:, la] = q_park
        rt.q[:, fq] = q_f
        self.forward()
        est['parked'] = self.zed_estimate()
        self.tick('0 ZED')

        def zerr(e):
            ex, eyaw, ok, info = e
            dyaw = torch.atan2(torch.sin(eyaw - yaw_true), torch.cos(eyaw - yaw_true))
            return (ex - p_true[:, :2]).norm(dim=1) * 1e3, torch.rad2deg(dyaw).abs(), ok

        for k_, e in est.items():
            xy, yw, ok = zerr(e)
            log[f'zed_{k_}_xy_mm'], log[f'zed_{k_}_yaw_deg'], log[f'zed_{k_}_ok'] = xy, yw, ok

        def frame(e):
            ex, eyaw = e[0], e[1]
            Rz = K.axis_angle_mat(torch.tensor([0., 0., 1.], device=dev).expand(n, 3), eyaw - yaw_true)
            return torch.cat([ex, p_true[:, 2:]], 1), Rz @ R_true   # height known from the table; PC flat

        use0 = est['start' if a.zed_at in ('start', 'after_hold') else 'parked']
        ep, eR = frame(use0)
        log['zed_ok'] = use0[2]
        qr = self.stage_pick(log)
        q_hold = self.stage_hold(qr, ep, eR, log)
        if a.zed_at == 'after_hold':
            e2 = self.zed_estimate()
            xy, yw, ok = zerr(e2)
            log['zed_after_hold_xy_mm'], log['zed_after_hold_yaw_deg'], log['zed_after_hold_ok'] = xy, yw, ok
            ep, eR = frame(e2)
            log['zed_ok'] = log['zed_ok'] & ok
        rt.comp['right'] = self.fric
        q_look = self.stage_transport(qr, q_hold, ep, eR, log)
        cmd = self.stage_approach(q_look, q_hold, ep, eR, log)
        self.stage_insert(cmd, q_hold, log)
        log['finite_end'] = torch.isfinite(rt.q).all(1)
        out = {k: (v.detach().float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in log.items()}
        out['seed'] = np.full(n, seed)
        return out


def classify(df):
    """First failed stage per episode; e2e success = insertion seated with an accepted ZED estimate."""
    cls = []
    for r in df.itertuples():
        if not r.zed_ok:
            cls.append('zed_rejected')
        elif not r.pick_ok:
            cls.append('pick_failed')
        elif not (r.hold_plan_ok and r.hold_latched):
            cls.append('hold_failed')
        elif r.transport_inhand_dp_mm > 10 or r.transport_ik_err_mm > 2:
            cls.append('transport_failed')
        elif not (r.pre_lateral_mm <= 8 and .5 <= r.pre_gap_mm <= 15):
            cls.append('approach_far')
        elif r.insert_outcome != 'seated':
            cls.append('insert_' + r.insert_outcome)
        else:
            cls.append('success')
    return cls


def summarize(df, a, wall):
    import pandas as pd  # noqa: F401
    n = len(df)
    k = int((df.e2e_class == 'success').sum())
    lo, hi = wilson(k, n)
    reach = lambda m: int(m.sum())   # noqa: E731
    pick = df.pick_ok.astype(bool) & df.zed_ok.astype(bool)
    hold = pick & df.hold_plan_ok.astype(bool) & df.hold_latched.astype(bool)
    trans = hold & (df.transport_inhand_dp_mm <= 10) & (df.transport_ik_err_mm <= 2)
    appr = trans & (df.pre_lateral_mm <= 8) & df.pre_gap_mm.between(.5, 15)
    ins = appr & (df.insert_outcome == 'seated')
    stages = []
    prev = n
    for name, m in (('zed gate', df.zed_ok.astype(bool)), ('pick', pick), ('hold', hold), ('transport', trans),
                    ('approach (lateral <= 8 mm, gap 0.5-15 mm)', appr), ('insertion (seated)', ins)):
        kk = reach(m)
        stages.append({'stage': name, 'entered': prev, 'passed': kk, 'rate': kk / max(prev, 1), 'wilson95': list(wilson(kk, prev))})
        prev = kk
    ok = df.e2e_class == 'success'
    a_ok = appr
    pre = {c: stats(df[c][a_ok]) for c in ('pre_lateral_mm', 'pre_tilt_deg', 'pre_gap_mm', 'warm_lateral_mm', 'warm_tilt_deg', 'warm_gap_mm')}
    pre['pre_roll_abs_deg'] = stats(df.pre_roll_deg[a_ok].abs())
    pre['pre_x_mm'] = {'mean': float(df.pre_x_mm[a_ok].mean()), 'std': float(df.pre_x_mm[a_ok].std())}
    pre['pre_y_mm'] = {'mean': float(df.pre_y_mm[a_ok].mean()), 'std': float(df.pre_y_mm[a_ok].std())}
    lat = df.pre_lateral_mm[a_ok]
    bins = [0, .5, 1, 1.5, 2, 3, 8]
    by_lat = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = a_ok & (df.pre_lateral_mm > b0) & (df.pre_lateral_mm <= b1)
        by_lat.append({'lateral_mm': [b0, b1], 'n': int(m.sum()), 'seated': float((df.insert_outcome[m] == 'seated').mean()) if m.any() else None})
    pk = df.pick_ok.astype(bool)
    inhand = {c: stats(df[c][pk], (1, 50, 99)) for c in df.columns if c.startswith('pick_inhand_')}
    zed = {w: {'xy_mm': stats(df[f'zed_{w}_xy_mm']), 'yaw_deg': stats(df[f'zed_{w}_yaw_deg']), 'gate_accept': float(df[f'zed_{w}_ok'].mean())}
           for w in ('start', 'parked', 'after_hold') if f'zed_{w}_xy_mm' in df}
    return {'name': a.name, 'n': n, 'success': k, 'rate': k / n, 'wilson95': [lo, hi], 'gate_98': bool(k / n >= .98 and n >= 500),
            'args': vars(a), 'stages': stages, 'failure_classes': df.e2e_class.value_counts().to_dict(),
            'insert_outcomes_all': df.insert_outcome.value_counts().to_dict(),
            'zed': zed,
            'pick': {'success': float(df.pick_ok.mean()), 'tilt_deg': stats(df.pick_tilt_deg[pk]), 'inhand_about_nominal': inhand,
                     'inhand_pos_mm_about_mean': stats(np.sqrt(sum((df[f'pick_inhand_d{c}_mm'][pk] - df[f'pick_inhand_d{c}_mm'][pk].mean()) ** 2 for c in 'xyz'))),
                     'inhand_rot_deg_about_mean': stats(np.sqrt(sum((df[f'pick_inhand_r{c}_deg'][pk] - df[f'pick_inhand_r{c}_deg'][pk].mean()) ** 2 for c in 'xyz')))},
            'hold': {'pc_moved_mm': stats(df.hold_pc_moved_mm), 'arm_arm_contact': float(df.hold_arm_arm.mean()), 'latched': float(df.hold_latched.mean())},
            'transport': {'inhand_dp_mm': stats(df.transport_inhand_dp_mm[pk]), 'inhand_dr_deg': stats(df.transport_inhand_dr_deg[pk]),
                          'right_pc_contact': float(df.transport_right_pc_contact[pk].mean()), 'ik_err_mm': stats(df.transport_ik_err_mm)},
            'approach': {'look_lateral_mm': stats(df.look_lateral_mm[trans]), 'preinsert': pre,
                         'within_1mm_2deg': float(((df.pre_lateral_mm <= 1) & (df.pre_tilt_deg <= 2))[a_ok].mean()),
                         'inhand_dp_mm': stats(df.approach_inhand_dp_mm[trans]), 'inhand_dr_deg': stats(df.approach_inhand_dr_deg[trans])},
            'insertion': {'seated_given_approach': float((df.insert_outcome[a_ok] == 'seated').mean()), 'by_preinsert_lateral': by_lat,
                          't_seat_s': stats(df.insert_t_s[ok]), 'pc_max_mm': stats(df.insert_pc_max_mm[a_ok])},
            'finite_end': float(df.finite_end.mean()), 'wall_s': wall}


def main(argv=None):
    a = parse(argv)
    import pandas as pd
    t0 = time.time()
    ch = Chain(a)
    print(f'built in {time.time() - t0:.0f} s', flush=True)
    if a.demo:
        from demo_chain import render_demos   # scripts/eval/demo_chain.py
        return render_demos(ch, a)
    rows = []
    for b in range(a.batches):
        tb = time.time()
        out = ch.run_batch(a.seed + b)
        df = pd.DataFrame(out)
        df['e2e_class'] = classify(df)
        rows.append(df)
        print(f'batch {b}: e2e {int((df.e2e_class == "success").sum())}/{len(df)} {df.e2e_class.value_counts().to_dict()} ({time.time() - tb:.0f} s)', flush=True)
    df = pd.concat(rows, ignore_index=True)
    rep = summarize(df, a, time.time() - t0)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f'{a.name}.csv', index=False)
    (OUT / f'{a.name}.json').write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps({k: rep[k] for k in ('name', 'n', 'success', 'rate', 'wilson95', 'stages', 'failure_classes')}, indent=1, default=float))
    return rep


if __name__ == '__main__':
    main()
