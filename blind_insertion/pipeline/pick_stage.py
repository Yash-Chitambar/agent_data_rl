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
                     reported as pick failures (no grasp attempt).
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
    """Part 1 scripted expert. With `estimator`, the reader pose used for planning (grasp poses, flip) is the
    estimate; execution feedback stays the expert's own (arm joint angles)."""

    def __init__(self, estimator=None):
        self.estimator = estimator
        self.name = 'expert_privileged' if estimator is None else 'expert_estimated'

    def run(self, chain):
        task, n, dev = chain.pick, chain.n, chain.dev
        extra = {}
        ok = torch.ones(n, dtype=torch.bool, device=dev)
        true_pose = task.reader_pose
        if self.estimator is not None:
            p_est, R_est, ok = self.estimator(chain)
            p_t, R_t = true_pose()
            extra['est_pos_err_mm'] = (p_est - p_t).norm(dim=1) * 1e3
            extra['est_rot_err_deg'] = torch.rad2deg(K.rot_log(R_est @ R_t.transpose(1, 2)).norm(dim=1))
            extra['est_ok'] = ok
            task.reader_pose = lambda: (p_est.clone(), R_est.clone())       # planning only (PickExpert.reset)
        try:
            ex = PickExpert(task, seed=chain.seed)
            ex.reset()
        finally:
            task.reader_pose = true_pose
        ex.ik_ok &= ok                    # rejected estimate: hold (counted as a pick failure)
        self.expert = ex
        self.start_hold = {'p': torch.zeros(n, 3, device=dev), 'R': torch.eye(3, device=dev).repeat(n, 1, 1),
                           'tilt': torch.zeros(n, device=dev)}
        for k in range(task.max_steps - 1):
            prev = ex.phase.clone()
            task.step(ex.act())
            chain.tick('1 pick')
            entered = (ex.phase == 5) & (prev == 4)
            if entered.any():
                m = task.pick_metrics()
                for key, mk in (('p', 'inhand_p'), ('R', 'inhand_R'), ('tilt', 'tilt_deg')):
                    self.start_hold[key][entered] = m[mk][entered]
            if ex.done.all():
                break
        extra['steps'] = torch.full((n,), k + 1, device=dev)
        extra['ik_ok'] = ex.ik_ok
        return dict(done=ex.done, failed=task.failed | ~ok, start_hold=self.start_hold, **extra)


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


def make_pick_stage(kind, reader_estimator='', chain=None):
    if kind in ('expert', 'expert_privileged'):
        return ExpertPick()
    if kind == 'expert_estimated':
        if not reader_estimator:
            raise ValueError("--pick expert_estimated needs --reader-estimator 'module:factory' or 'noisy_oracle:<mm>,<deg>'")
        if reader_estimator.startswith('noisy_oracle:'):
            return ExpertPick(noisy_oracle(reader_estimator.split(':', 1)[1]))
        mod, fn = reader_estimator.split(':')
        return ExpertPick(getattr(importlib.import_module(mod), fn)(chain))
    if kind == 'act':
        return ActPick()
    raise ValueError(f'unknown pick stage {kind}')
