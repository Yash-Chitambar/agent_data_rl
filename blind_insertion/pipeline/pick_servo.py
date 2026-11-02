"""Servo-descent pick stage (Part 1 final): the scripted PickExpert planned from an estimated reader pose and
corrected from the right-wrist camera at the hover and at fractions of the descent.

This is the `if a.servo:` block of scripts/policy/pick_expert_eval.py as a reusable pick stage (the gate run:
2040/2048 = 99.61 % at +/-1.5 deg encoder-offset DR, believed frame, n = 2048, seed 60000; results/usbc/policy/pick/
expert_zedcap_combo_servo_believed_enc1.5_s60000.json). Same expert settings as that run: believed frame (plan and
servo on encoder FK), hover tolerance 4 mm, settle at most 3 s, servo frames at the hover and at 0.5 / 0.8 / 0.92 of
the descent, re-hover when the hover correction moves the grasp by more than 8 mm (at most twice).

Nets (scripts/perception/train_reader_servo.py; input = the 320 px right-wrist policy frame, uint8 RGB; output =
grasp pose relative to the hand: t (m, grasp_site frame) and rotation vector (rad)):
  hover  reader_servo_hover.pt (3 outputs: position only) on the hover frames
  near   reader_servo_near.pt  (3 outputs) on the waypoint frames: position
  rot    reader_servo_near6.pt (6 outputs) on the waypoint frames: rotation only (optional)

Used by the end-to-end sim chain and by the real runner (deploy/bimanual/chain.py), on the same duck-typed `chain`
(blind_insertion/pipeline/pick_stage.py interface): chain.pick (PickTask, or deploy's encoder-only adapter),
chain.render() -> {'wrist_cam': (N, 320, 320, 3) uint8}, chain.tick(label), chain.seed, chain.n, chain.dev.
Every step goes through chain.pick.step(action14). `on_frame(kind, frame, delta)` (optional) sees each servo frame.
"""
from __future__ import annotations

import torch

from blind_insertion.envs.bimanual_pick import PickExpert
from blind_insertion.perception.nets import PoseNet


def load_servo_net(path, dev):
    """(net, scale, n_out) of a train_reader_servo.py checkpoint."""
    ck = torch.load(path, map_location=dev)
    net = PoseNet(ck.get('n_out', 6), ck['size'], width=ck['width']).to(dev).eval()
    net.load_state_dict(ck['net'])
    return net, ck['scale'].to(dev), ck.get('n_out', 6)


@torch.no_grad()
def run_net(net, scale, fr):
    """Batched forward on 320 px frames; a 3-output (position-only) net gets NaN rotations."""
    d = torch.cat([net(fr[i:i + 256]) for i in range(0, len(fr), 256)]) * scale
    return torch.cat([d, torch.full_like(d, float('nan'))], 1) if d.shape[1] == 3 else d


class ServoPick:
    name = 'expert_servo'

    def __init__(self, estimator, hover, near, rot=None, fracs=(.5, .8, .92), rehover_mm=8., rehover_max=2,
                 settle_max_s=3., hold_s=1., v_max=.10, joint_speed=1.2, integral_gain=.03, hover_tol_m=.004,
                 believed_frame=True, on_frame=None, make_expert=None):
        """estimator(chain) -> (p (N,3), R (N,3,3), ok (N,)) reader pose (pick_stage.ZedReader); hover / near / rot:
        (net, scale, n_out) from load_servo_net (rot may be None)."""
        self.estimator, self.hover, self.near, self.rot = estimator, hover, near, rot
        self.fracs, self.rehover_mm, self.rehover_max = tuple(fracs), rehover_mm, rehover_max
        self.kw = dict(hold_s=hold_s, v_max=v_max, joint_speed=joint_speed, integral_gain=integral_gain,
                       settle_max_s=settle_max_s, hover_tol_m=hover_tol_m)
        self.believed_frame, self.on_frame = believed_frame, on_frame
        self.make_expert = make_expert or PickExpert      # grasp_v3.GraspV3Expert: guarded grasp with retries

    def frame_delta(self, fr, need_frame):
        """Servo estimate per world: hover net, or (at a waypoint) near-net position + rot-net rotation."""
        delta = run_net(self.hover[0], self.hover[1], fr)
        if need_frame.any():
            dn = run_net(self.near[0], self.near[1], fr)
            if self.rot is not None:
                r = torch.cat([self.rot[0](fr[i:i + 256]) for i in range(0, len(fr), 256)]) * self.rot[1]
                dn = torch.cat([dn[:, :3], r if r.shape[1] == 3 else r[:, 3:]], 1)
            delta = torch.where(need_frame[:, None], dn, delta)
        return delta

    def run(self, chain):
        task, n, dev = chain.pick, chain.n, chain.dev
        p_est, R_est, ok = self.estimator(chain)
        extra = {'est_ok': ok}
        for k_, v_ in getattr(chain, 'reader_est_info', {}).items():
            extra['est_' + k_] = v_
        fn = lambda: (p_est.clone(), R_est.clone())       # noqa: E731
        ex = self.make_expert(task, seed=chain.seed, reader_pose_fn=fn, **self.kw)
        ex.believed_frame = self.believed_frame
        ex.servo_fracs, ex.servo_rot = self.fracs, True
        ex.reset()
        ex.ik_ok &= ok                    # rejected estimate: hold (counted as a pick failure)
        self.expert = ex
        n_frames = torch.zeros(n, dtype=torch.long, device=dev)
        k = -1
        for k in range(task.max_steps - 1):
            prev = ex.phase.clone()
            act = ex.act()
            want = ((ex.phase == 1) & (prev == 0) & ~ex.refined) | ex.need_frame
            if want.any():
                fr = chain.render()['wrist_cam']
                with torch.no_grad():
                    delta = self.frame_delta(fr, ex.need_frame)
                if self.on_frame is not None:
                    self.on_frame('waypoint' if bool(ex.need_frame[want].any()) else 'hover', fr, delta, ex)
                hov = want & ~ex.need_frame
                pg_old = ex.p_g.clone()
                ex.servo_update(want, delta)
                n_frames += want.long()
                if self.rehover_mm > 0 and hov.any():
                    big = hov & ((ex.p_g - pg_old).norm(dim=1) > self.rehover_mm * 1e-3) & (ex.hover_iters < self.rehover_max)
                    if big.any():
                        ex.rehover(big)
            task.step(act)
            chain.tick('1 pick')
            if ex.done.all():
                break
        extra.update(steps=torch.full((n,), k + 1, device=dev), ik_ok=ex.ik_ok, servo_frames=n_frames,
                     hover_iters=ex.hover_iters)
        failed = getattr(task, 'failed', torch.zeros(n, dtype=torch.bool, device=dev))
        gave_up = getattr(ex, 'gave_up', torch.zeros(n, dtype=torch.bool, device=dev))
        return dict(done=ex.done, failed=failed | ~ok | gave_up, **extra)
