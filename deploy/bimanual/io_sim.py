"""Dry-run backend: one world of scripts/eval/eval_chain.py's Chain (the 99.56-99.66 % sim: same scene, physics,
contact / control / visual DR, encoder-offset error and latency), behind RealIO's interface. The runner code above
it is the real-rig code path unchanged.

Conventions (as RealIO): read() returns the MOTOR frame, i.e. the reading r such that r + rig offsets = the believed
(model) angles the sim's controllers see (= sim encoder - encoder-offset error); send() takes motor targets and
i2rt gripper values. move_to() teleports (as eval_chain teleports to the capture pose); a gripper opening wider than
the sim's finger limit (0.024 m on the right: PickTaskCfg.finger_open) is rendered and then restored before the next
physics step, exactly as eval_chain does for its capture frames.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SIDES = ('right', 'left')
CAM = {'top': 'top', 'wrist_right': 'wrist_cam', 'wrist_left': 'left_wrist_cam'}


def eval_chain():
    """scripts/eval/eval_chain.py as a module (shared with the real runner: stage code is borrowed from it)."""
    if 'eval_chain' not in sys.modules:
        sys.path.insert(0, str(ROOT / 'scripts' / 'eval'))
        spec = importlib.util.spec_from_file_location('eval_chain', ROOT / 'scripts/eval/eval_chain.py')
        mod = importlib.util.module_from_spec(spec)
        sys.modules['eval_chain'] = mod
        spec.loader.exec_module(mod)
    return sys.modules['eval_chain']


def chain_argv(cfg, policy, nworld=1, extra=()):
    """eval_chain arguments of the real chain (deploy/config/bimanual.json chain_args)."""
    c = cfg['chain_args']
    argv = ['--nworld', str(nworld), '--policy', str(policy), '--hold-force', str(c['hold_force']),
            '--hold-along', str(c['hold_along']), '--look-gap', str(c['look_gap']), '--stop', str(c['stop']),
            '--wrist-net', c['wrist_net'], '--fine-net', c['fine_net'], '--inhand-net', c['inhand_net'],
            '--zed-gate', c['zed_gate'], '--handover', c['handover'],
            '--stop-offset-mm', *[str(x) for x in c['stop_offset_mm']]]
    if c.get('believed_frame'):
        argv.append('--believed-frame')
    return argv + list(extra)


class SimIO:
    is_sim = True

    def __init__(self, cfg, gmaps, policy, device='cuda:0', extra_args=()):
        EC = eval_chain()
        self.EC, self.cfg, self.gmaps = EC, cfg, gmaps
        self.a = EC.parse(chain_argv(cfg, policy, 1, extra_args))
        self.ch = EC.Chain(self.a)
        self.rt, self.h = self.ch.rt, self.ch.h
        self.dev = self.rt.device
        self.off = self.rt.encoder_offsets.cpu().numpy().astype(float)        # (2, 6) rig calibration
        m = self.rt.mjm
        self.fq = {'right': list(self.ch.pick.finger_qadr),
                   'left': [int(m.joint('left_' + f + '_finger').qposadr[0]) for f in ('left', 'right')]}
        self.limit = {'right': float(cfg['gripper']['right_approach_open_m']), 'left': float(cfg['gripper']['full_open_m'])}
        self.pending = {}
        self.t, self.k = 0.0, 0
        self.dt = self.ch.dt

    def reset(self, seed):
        """eval_chain.Chain.run_batch's episode start for one seed (layout, DR draws, left arm parked)."""
        ch, rt, a = self.ch, self.rt, self.a
        ch.seed = seed
        torch.manual_seed(seed)
        if hasattr(rt.visual, 'gen'):
            rt.visual.gen.manual_seed(seed)
        ch.rec = None
        rt.comp['right'] = None
        ch.pick.reset()
        if a.believed_frame:
            lq = rt.indices['left'][0]
            rt.q[:, lq] = rt.q[:, lq] + ch.pick.enc_error[:, 1]
            ch.forward()
            ch.pick.joint_target.copy_(ch.pick._measured_model_angles())
        h = self.h
        h.enc_error.copy_(ch.pick.enc_error)
        h.latency.copy_(ch.pick.latency)
        h.hist.zero_()
        h.fresh[:] = True
        h.xfrc.zero_()
        self.pending = {}
        self.t, self.k = 0.0, 0
        p, R = h.pc_pose()
        self.truth0 = {'pc_p': p[0].cpu().numpy(), 'pc_R': R[0].cpu().numpy(),
                       'reader': ch.pick.reader_pose_q()[0].cpu().numpy(),
                       'enc_error': h.enc_error[0].cpu().numpy(), 'latency': int(h.latency[0])}
        return self.truth0

    # ------------------------------------------------------------------ state and commands
    def _finger_m(self):
        return np.array([float(self.rt.q[0, self.fq[s]].mean()) for s in SIDES])

    def read(self):
        pr = self.rt.proprioception()
        enc = pr['encoder_rad'].reshape(-1, 2, 6)[0]
        q = (enc - self.h.enc_error[0]).cpu().numpy().astype(float)
        qd = pr['encoder_velocity_rad_s'].reshape(-1, 2, 6)[0].cpu().numpy().astype(float)
        fm = self._finger_m()
        g = np.array([self.gmaps[s].reading(fm[i]) for i, s in enumerate(SIDES)])
        return dict(q=q, qd=qd, grip=g, eff=np.full((2, 6), np.nan), t_obs=np.full(2, self.t), t=self.t)

    def _restore_fingers(self):
        if self.pending:
            for s, w in self.pending.items():
                self.rt.q[:, self.fq[s]] = w
            self.pending = {}
            self.ch.forward()

    def send(self, q_motor, grip, ff_right=None):
        from blind_insertion.envs import friction_comp as FC
        self._restore_fingers()
        b = torch.as_tensor(np.asarray(q_motor, float) + self.off, dtype=torch.float32, device=self.dev)
        sign = [1. if grip[i] > self.gmaps[s].close + 0.05 else -1. for i, s in enumerate(SIDES)]
        self.rt.comp['right'] = FC.parse(ff_right, side='right') if isinstance(ff_right, dict) else ff_right
        ff = np.zeros((2, 6))
        if self.rt.comp['right'] is not None:
            tau, ramp = self.rt.comp['right']
            bq = np.asarray(q_motor[0], float) - self.read()['q'][0]
            ff[0] = FC.feedforward(bq, tau, ramp)
        self.h.step(b[0][None], sign[0], b[1][None], sign[1])
        self.t += self.dt
        self.k += 1
        sent = np.concatenate([np.asarray(q_motor, float), np.asarray(grip, float)[:, None]], 1)
        return sent, ff

    def start_clock(self):
        pass

    def wait_tick(self):
        return 0.0

    def move_to(self, q_motor, grip=None, settle_trim=True, max_vel=0.15):
        """Teleport: true angles = motor + offsets + encoder-offset error (the commanded believed pose)."""
        rt, h = self.rt, self.h
        for s, q in q_motor.items():
            i = SIDES.index(s)
            qa = rt.indices[s][0]
            rt.q[:, qa] = torch.as_tensor(np.asarray(q, float) + self.off[i], dtype=torch.float32, device=self.dev) + h.enc_error[:, i]
            rt.v[:, rt.indices[s][1]] = 0
        for s, g in (grip or {}).items():
            w = float(self.gmaps[s].finger_m(g))
            w = min(max(w, 0.), self.cfg['gripper']['full_open_m'])
            prev = float(rt.q[0, self.fq[s]].mean())
            rt.q[:, self.fq[s]] = w
            if w > self.limit[s] + 1e-6:             # render only; restored before the next physics step
                self.pending[s] = min(prev, self.limit[s])
        self.ch.forward()
        h.fresh[:] = True

    def set_grip(self, side, value, wait_s=1.0):
        q = self.read()['q']
        g = self.read()['grip'].copy()
        g[SIDES.index(side)] = value
        for _ in range(int(round(wait_s / self.dt))):
            self.send(q, g)

    def hold(self):
        pass

    def place_reader_in_hand(self):
        """Dry run of the hand-placed reader (stage approach_insert): teleport the reader to the Part 1 measured mean
        in-hand pose in the right grasp_site (as envs/left_hold.RightInserter places it)."""
        from blind_insertion.envs import kinematics as K
        from blind_insertion.envs.inhand import measured_mean
        rt, ch = self.rt, self.ch
        R_sr, p_sr = measured_mean(device=self.dev)            # grasp_site in the reader frame
        p_t, R_t = ch.pick.chain.fk(rt.q[:, rt.indices['right'][0]])
        R_r = R_t @ R_sr.T
        p_r = p_t - (R_r @ p_sr[:, None])[..., 0]
        pa = rt.lay.peg_qadr
        rt.q[:, pa:pa + 3] = p_r
        rt.q[:, pa + 3:pa + 7] = K.mat_to_quat(R_r)
        from blind_insertion.envs.left_hold import RightInserter
        ins = RightInserter(self.h)
        rt.q[:, ins.fingers] = ins._finger_q(rt.q[0])          # pads just touching: the next close squeezes
        rt.v[:] = 0
        self.pending.pop('right', None)
        self.ch.forward()

    def stopped(self):
        return False

    def drain_can(self):
        return {}

    def frames(self, names=('top', 'wrist_right')):
        fr = self.ch.render()
        out = {}
        for n in names:
            img = fr[CAM[n]][0].cpu().numpy()
            out[n] = (img, None, self.t, self.k)
        return out

    fresh_frames = frames

    def truth(self):
        """Privileged sim state for the dry run's logs and comparisons (never used for control)."""
        ch, h = self.ch, self.h
        tp, trv = ch.rel_pose()
        p, R = h.pc_pose()
        ip, iR = ch.inhand()
        rt = self.rt
        met = rt.insertion_metrics(.00025, 2., 25e-6)
        return {'plug_in_port_mm': (tp[0] * 1e3).cpu().numpy(), 'plug_rot_deg': torch.rad2deg(trv[0]).cpu().numpy(),
                'pc_p': p[0].cpu().numpy(), 'pc_R': R[0].cpu().numpy(), 'inhand_p': ip[0].cpu().numpy(),
                'inhand_R': iR[0].cpu().numpy(), 'seated': bool(met['geometric_seated'][0]),
                'tip_depth_mm': float(met['tip_depth_m'][0]) * 1e3,
                'q_true': torch.stack([rt.q[0, rt.indices[s][0]] for s in SIDES]).cpu().numpy(),
                'finite': bool(torch.isfinite(rt.q).all())}

    def close(self, read_line=None):
        pass
