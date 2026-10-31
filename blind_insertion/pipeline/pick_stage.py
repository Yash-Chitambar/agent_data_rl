"""Pluggable pick stage for the end-to-end chain (scripts/eval/eval_chain.py --pick ...).

A pick stage drives the right arm (and keeps the left arm parked) through the shared runtime until the
reader is lifted, then returns. The chain then hands over to the absolute-target harness.

Interface (duck-typed):

    class MyPick:
        name = 'my_pick'
        def run(self, chain) -> dict:
            '''Step the simulation with chain.pick.step(action14) (PickTask: normalized accumulate actions,
            control DR applied) and call chain.tick('1 pick') after every step (demo recording).
            Return per-world tensors: {'done': (N,) bool, 'failed': (N,) bool} plus any extra diagnostics
            (logged with a 'pick_' prefix). The chain measures lift, tilt and the in-hand pose itself.'''

`chain` gives: chain.pick (PickTask on the shared runtime), chain.rt (BimanualRuntime), chain.render()
(RGB frames dict: 'top', 'wrist_cam', 'left_wrist_cam'; 320 px; visual DR), chain.n, chain.dev, chain.seed.

Stages here:
  expert_privileged  Part 1 scripted expert (PickExpert) planned from the true reader pose.
  expert_estimated   the same expert planned from an estimated reader pose: an estimator object
                     `est(chain) -> (p (N,3) m, R (N,3,3), ok (N,) bool)` in world coordinates (reader body frame
                     as PickTask.reader_pose). Loaded from --reader-estimator 'module:factory' (factory(chain) ->
                     estimator); 'noisy_oracle:<pos_mm>,<yaw_deg>' is a built-in test estimator (true pose +
                     uniform noise) for the interface and for error-budget sweeps. Worlds with ok False are
                     reported as pick failures (no grasp attempt). 'zed:<ckpt>': the Part 1 ZED reader estimator
                     (perception/reader_zed.py) + presence/OOD gate on the chain's single start ZED frame.
  act                ACT pick policy (wrist + ZED RGB + encoders). Not wired yet: the policy runs in the separate
                     LeRobot venv (scripts/act/); plug a TorchScript export in ActPick.run.
"""
from __future__ import annotations

import importlib
import math

import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickExpert


class ExpertPick:
    """Part 1 scripted expert. With `estimator`, the reader pose it plans with is the estimate (PickExpert
    reader_pose_fn); with `wrist` (a callable (chain, expert, at_hover) -> corrected (p, R)), the pose is refined from
    the right-wrist frame at the pre-grasp hover (PickExpert.replan). Execution feedback stays the expert's own."""

    def __init__(self, estimator=None, wrist=None, hover_relative=True, hover_wait_s=1.5, hover_tol_m=2e-3):
        self.estimator, self.wrist = estimator, wrist
        self.hover_relative, self.hover_wait, self.hover_tol = hover_relative, int(round(hover_wait_s / .02)), hover_tol_m
        self.name = 'expert_privileged' if estimator is None else 'expert_estimated'

    def run(self, chain):
        task, n, dev = chain.pick, chain.n, chain.dev
        extra = {}
        ok = torch.ones(n, dtype=torch.bool, device=dev)
        fn = None
        if self.estimator is not None:
            p_est, R_est, ok = self.estimator(chain)
            p_t, R_t = task.reader_pose()
            extra['est_pos_err_mm'] = (p_est - p_t).norm(dim=1) * 1e3
            extra['est_rot_err_deg'] = torch.rad2deg(K.rot_log(R_est @ R_t.transpose(1, 2)).norm(dim=1))
            extra['est_ok'] = ok
            for k_, v_ in getattr(chain, 'reader_est_info', {}).items():
                extra['est_' + k_] = v_
            fn = lambda: (p_est.clone(), R_est.clone())       # noqa: E731
        ex = PickExpert(task, seed=chain.seed, reader_pose_fn=fn)
        ex.reset()
        ex.ik_ok &= ok                    # rejected estimate: hold (counted as a pick failure)
        self.expert = ex
        self.start_hold = {'p': torch.zeros(n, 3, device=dev), 'R': torch.eye(3, device=dev).repeat(n, 1, 1),
                           'tilt': torch.zeros(n, device=dev)}
        hover_lag = torch.full((n,), float('nan'), device=dev)
        ref_err, ref_rot = hover_lag.clone(), hover_lag.clone()
        for k in range(task.max_steps - 1):
            prev = ex.phase.clone()
            act = ex.act()
            if self.wrist is not None:
                at_hover = (ex.phase == 1) & (prev == 0) & ~ex.refined
                if at_hover.any():
                    # The joint move ends (phase 0 -> 1) with the hand still ~80-100 mm short of the hover (PD lag,
                    # n = 16): the wrist frame would be taken far from the pose the net was trained at. Hold the
                    # expert's joint command (q_cmd + integral) until the hand is within 2 mm of the hover (<= 1.5 s).
                    for _ in range(self.hover_wait):
                        pa, _ = task.tcp_pose()
                        lag = (pa - ex.p_pre).norm(dim=1)
                        if (lag[at_hover] < self.hover_tol).all():
                            break
                        # keep commanding the expert's current joint command (accumulate increments toward it)
                        a0 = torch.zeros(n, 14, device=dev); a0[:, 6] = ex.grip; a0[:, 13] = 1.
                        tgt = ex.q_cmd + ex.integ - task.enc_error[:, 0]
                        a0[:, :6] = ((tgt - task.joint_target[:, 0]) / task.cfg.action_scale_rad).clamp(-1, 1)
                        task.step(a0)
                        chain.tick('1 pick')
                    hover_lag[at_hover] = lag[at_hover]
                    p_new, R_new = self.wrist(chain, ex)
                    ex.replan(at_hover, p_new, R_new, hover_relative=self.hover_relative)
                    p_t, R_t = task.reader_pose()      # reader still on the table: refined-estimate error
                    ref_err[at_hover] = (ex.p_est - p_t).norm(dim=1)[at_hover]
                    ref_rot[at_hover] = torch.rad2deg(K.rot_log(ex.R_est @ R_t.transpose(1, 2)).norm(dim=1))[at_hover]
            task.step(act)
            chain.tick('1 pick')
            entered = (ex.phase == 5) & (prev == 4)
            if entered.any():
                m = task.pick_metrics()
                for key, mk in (('p', 'inhand_p'), ('R', 'inhand_R'), ('tilt', 'tilt_deg')):
                    self.start_hold[key][entered] = m[mk][entered]
            if ex.done.all():
                break
        if self.wrist is not None:
            extra['refined'] = ex.refined
            extra['hover_lag_mm'] = hover_lag * 1e3
            extra['refined_pos_err_mm'] = ref_err * 1e3
            extra['refined_rot_err_deg'] = ref_rot
        extra['steps'] = torch.full((n,), k + 1, device=dev)
        extra['ik_ok'] = ex.ik_ok
        return dict(done=ex.done, failed=task.failed | ~ok, start_hold=self.start_hold, **extra)


class WristRefine:
    """Part 1 wrist refinement (scripts/perception/train_reader_wrist.py): body-centre xy + yaw correction of the
    planned reader pose from the right-wrist frame at the pre-grasp hover."""

    def __init__(self, ckpt, dev):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'perception'))
        import train_reader_wrist as RW
        from blind_insertion.perception.nets import PoseNet
        ck = torch.load(ckpt, map_location=dev)
        self.RW = RW
        self.net = PoseNet(3, ck['size'], width=ck['width']).to(dev).eval()
        self.net.load_state_dict(ck['net'])
        self.scale = ck['scale'].to(dev)

    @torch.no_grad()
    def __call__(self, chain, ex):
        fr = chain.render()['wrist_cam']
        corr = torch.cat([self.net(fr[i:i + 256]) for i in range(0, len(fr), 256)]) * self.scale
        return self.RW.apply_correction(chain.pick, ex.p_est, ex.R_est, corr)


class ActPick:
    name = 'act'

    def run(self, chain):
        raise NotImplementedError(
            'ACT pick is not wired into the chain yet: export the trained policy (scripts/act/export_act.py, '
            'TorchScript) and implement ActPick.run: per control step build the observation (chain.render() wrist '
            '+ top RGB, encoders = chain.pick.observation()), call the policy, chain.pick.step(action14), '
            "chain.tick('1 pick'); return {'done', 'failed'} when the policy signals the lift is complete.")


def noisy_oracle(spec):
    """'pos_mm,yaw_deg': true reader pose + uniform position noise (xy, +/- pos_mm) and yaw noise (+/- yaw_deg)."""
    pos_mm, yaw_deg = (float(x) for x in spec.split(','))

    def est(chain):
        p, R = chain.pick.reader_pose()
        n, dev = len(p), p.device
        g = torch.Generator(device=dev); g.manual_seed(int(chain.seed) + 77)
        u = lambda *s: torch.rand(*s, device=dev, generator=g) * 2 - 1   # noqa: E731
        p = p.clone(); p[:, :2] += u(n, 2) * pos_mm * 1e-3
        Rz = K.axis_angle_mat(torch.tensor([0., 0., 1.], device=dev).expand(n, 3), u(n) * math.radians(yaw_deg))
        return p, Rz @ R, torch.ones(n, dtype=torch.bool, device=dev)
    return est


class ZedReader:
    """Part 1 ZED reader estimator (perception/reader_zed.py: estimate(top, q_left, q_right, right_finger_m) -> xy, yaw,
    ok; gate = presence + arm-pose envelope) on the reader capture frame (chain.reader_frame(): left arm parked, right
    arm at the capture pose, right gripper open). Rejected worlds re-take the frame `retries` times."""

    def __init__(self, ckpt, dev, retries=1):
        from blind_insertion.perception.reader_zed import ReaderZedEstimator
        self.est, self.retries = ReaderZedEstimator(ckpt, dev), retries

    @torch.no_grad()
    def __call__(self, chain):
        from blind_insertion.perception.reader_zed import reader_pose
        top, arms = chain.reader_frame()
        xy, yaw, ok = self.est.estimate(top, *arms)
        tries = torch.ones(len(xy), device=xy.device)
        for _ in range(self.retries):
            if ok.all():
                break
            top, arms = chain.reader_frame()           # re-take (fresh camera noise)
            xy2, yaw2, ok2 = self.est.estimate(top, *arms)
            redo = ~ok
            xy[redo], yaw[redo], ok[redo] = xy2[redo], yaw2[redo], ok2[redo]
            tries[redo] += 1
        p, R = reader_pose(chain.pick, xy, yaw)
        chain.reader_est_info = {'tries': tries}
        return p, R, ok


def make_pick_stage(kind, reader_estimator='', chain=None):
    if kind in ('expert', 'expert_privileged'):
        return ExpertPick()
    if kind == 'expert_estimated':
        if not reader_estimator:
            raise ValueError("--pick expert_estimated needs --reader-estimator 'module:factory' or 'noisy_oracle:<mm>,<deg>'")
        if reader_estimator.startswith('zed:'):
            wrist = WristRefine(chain.a.reader_wrist, chain.dev) if getattr(chain.a, 'reader_wrist', '') else None
            return ExpertPick(ZedReader(reader_estimator.split(':', 1)[1], chain.dev, retries=chain.a.reader_retries), wrist,
                              hover_relative=bool(chain.a.wrist_hover_relative), hover_wait_s=chain.a.wrist_hover_wait_s)
        if reader_estimator.startswith('noisy_oracle:'):
            return ExpertPick(noisy_oracle(reader_estimator.split(':', 1)[1]))
        mod, fn = reader_estimator.split(':')
        return ExpertPick(getattr(importlib.import_module(mod), fn)(chain))
    if kind == 'act':
        return ActPick()
    raise ValueError(f'unknown pick stage {kind}')
