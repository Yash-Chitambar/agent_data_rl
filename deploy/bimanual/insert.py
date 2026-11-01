"""Part 4 insertion on encoders: InsertionTask's actor observation, controller and warm-up, the K-J policy, and a
seated detector that stops the push.

Observation (InsertionTask._actor_obs, obs_dim 68), all from the encoders + the in-hand estimate:
  [0:12]  encoder angles, right then left, MOTOR frame (sim: quantized encoder - offset error)
  [12:24] joint velocities
  [24:38] previous normalized action (14: right 6, right grip -1, left 6 = 0, left grip -1; zero in the warm-up)
  [38:50] joint target - believed angles (left target = the latched press target)
  [50:53] (right TCP - nominal pre-insert TCP) x 10, believed FK (m)
  [53:56] XYZ Euler angles of the TCP rotation relative to the nominal TCP rotation
  [56:62] TCP twist J(q_believed) qd
  [62:68] in-hand estimate (inhand_ih5 at the 6 mm stop): mm on the TCP axes, rotation vector deg
Left joint-6 entries (11, 23, 36, 49) are zeroed (the policy's env_json mask_left_obs 'j6'; the loader masks too).
Controller (InsertionTask.step/_update_joint_targets): right target accumulates 0.04 a (a in [-1, 1]), within
target_cap (policy meta, 0.05 rad) of the believed angles and inside the joint limits; actions are zeroed for the
first warmup_steps (10); the left arm holds its latched press target (absolute, no cap). Grippers closed.

Seated detector (no sim truth on the robot): plug advance along the port axis = TCP displacement since the
hand-over projected on -port_z (believed FK; offset errors cancel in a displacement). depth = advance - stop gap
(the approach servo's 6 mm). Seated when depth >= seat_depth - seat_margin for seat_hold_s with the axial speed
below seat_speed; the push then stops (target = measured pose) and the release follows. The policy time limit
(insert_s) is the fallback release (the older policies kept pushing after seating).
"""
from __future__ import annotations

import numpy as np
import torch

from blind_insertion.envs import friction_comp as FC, kinematics as K


class InsertObs:
    def __init__(self, scene, consts, device):
        self.ch = scene.tcp_chain
        self.dev = device
        self.R_obs = torch.tensor(consts['R_obs'], dtype=torch.float32, device=device)
        self.p_obs = torch.tensor(consts['p_obs'], dtype=torch.float32, device=device)
        self.mask = list(consts['mask_idx'])
        self.off = scene.encoder_offsets.to(device)

    def __call__(self, q_motor, qd, prev14, jt, inhand6):
        """q_motor, qd (2, 6) numpy/tensor; prev14 (14,); jt (2, 6) believed targets; inhand6 (6,) -> (1, 68)."""
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32) if not torch.is_tensor(x) else x, dtype=torch.float32, device=self.dev)  # noqa: E731
        q, v = t(q_motor).reshape(2, 6), t(qd).reshape(2, 6)
        qb = q + self.off
        p, R, J = self.ch.fk(qb[0:1], with_jac=True)
        tw = (J @ v[0][:, None])[None, :, 0] if J.dim() == 2 else (J @ v[0:1][..., None])[..., 0]
        eul = K.euler_xyz(self.R_obs.T @ R)
        base = torch.cat([q.reshape(1, 12), v.reshape(1, 12), t(prev14).reshape(1, 14), (t(jt).reshape(2, 6) - qb).reshape(1, 12)], 1)
        if self.mask:
            base[:, self.mask] = 0
        out = torch.cat([base, (p - self.p_obs) * 10, eul, tw.reshape(1, 6), t(inhand6).reshape(1, 6)], 1)
        return torch.nan_to_num(out, 0., 0., 0.)


class SeatDetector:
    def __init__(self, port_R, gap0_m, seat_depth_m, margin_m, hold_s, speed_m_s, dt):
        self.z = torch.as_tensor(port_R, dtype=torch.float32).reshape(3, 3)[:, 2].cpu().numpy()
        self.gap0, self.need = gap0_m, seat_depth_m - margin_m
        self.n_hold, self.vmax, self.dt = max(1, int(round(hold_s / dt))), speed_m_s, dt
        self.p0, self.prev, self.run, self.seated = None, None, 0, False
        self.win = []                      # axial speed over 0.1 s (one-tick differences are encoder-noise dominated)

    def update(self, p_tcp):
        p = np.asarray(p_tcp, float).reshape(3)
        if self.p0 is None:
            self.p0 = p
        depth = float(-(p - self.p0) @ self.z) - self.gap0
        self.win = (self.win + [depth])[-6:]
        v = 0.0 if len(self.win) < 2 else (self.win[-1] - self.win[0]) / ((len(self.win) - 1) * self.dt)
        self.prev = depth
        deep = depth >= self.need and abs(v) <= self.vmax
        self.run = self.run + 1 if deep else 0
        self.seated = self.seated or self.run >= self.n_hold
        return depth, v, self.seated


def contact_summary(pol, hist):
    """Contact-head prediction of the policy's encoder (first ensemble member): class shares over its labels."""
    ac = pol.members[0][0]
    h = hist[..., :pol.obs_dim].clone()
    if pol.mask_idx:
        h[..., list(pol.mask_idx)] = 0
    with torch.no_grad():
        z, _ = ac.encode(h)
        p = torch.softmax(ac.contact_logits(z), -1)[0]          # (n_points, classes)
    return p.mean(0).cpu().numpy(), torch.bincount(p.argmax(-1), minlength=p.shape[-1]).cpu().numpy()


def run_insertion(chain, cmd, q_hold, log, sim_obs=None):
    """Hand-over and insertion from the approach command `cmd` (1, 6 believed) with the left at q_hold (1, 6).
    Returns a dict (outcome, steps, seated tick, depth). sim_obs(jt, prev, inhand) -> the sim env's actor obs for
    the same controller state (dry-run equivalence check)."""
    cfg, ic, dev = chain.cfg, chain.cfg['insert'], chain.dev
    consts, pol = chain.consts, chain.pol
    scale, cap = 0.04, float(pol.meta['target_cap_rad'])
    assert pol.meta['action_mode'] == 'accumulate' and pol.obs_dim == consts['obs_dim'] == 68, pol.meta
    H, warm = int(pol.meta['history']), int(consts['warmup_steps'])
    T = int(round(ic['insert_s'] / chain.dt))
    lim = chain.scene.indices['right'][4].to(dev)
    obs_fn = InsertObs(chain.scene, consts, dev)
    # torque hand-over (eval_chain --handover torque): fold the feed-forward torque into the PD target, then FF off
    ff_was_on = chain.rt.comp['right'] is not None
    chain.rt.comp['right'] = None
    qb = chain.q_b.clone()
    jt = torch.stack([cmd.reshape(6).to(dev), q_hold.reshape(6).to(dev)]).float()
    if ff_was_on and chain.fric is not None:
        tau_ff = FC.feedforward(jt[0] - qb[0], *chain.fric)
        jt[0] = jt[0] + tau_ff / chain.scene.kp[0].to(dev)
        log['handover_ff_dq_mrad'] = float((tau_ff / chain.scene.kp[0].to(dev)).abs().max()) * 1e3
    inhand6 = chain.inhand_est6.reshape(6).to(dev).float()
    prev = torch.zeros(14, device=dev)
    o = obs_fn(chain.state['q'], chain.state['qd'], prev, jt, inhand6)
    hist = o[:, None].expand(-1, H, -1).clone()
    pR = chain.port[1]
    det = SeatDetector(pR[0].cpu(), chain.a.stop * 1e-3 + chain.a.stop_offset_mm[2] * 1e-3, consts['seat_depth_m'],
                       ic['seat_margin_m'], ic['seat_hold_s'], ic['seat_speed_m_s'], chain.dt)
    p0, _ = chain.scene.tcp_chain.fk(qb[0:1])
    det.update(p0[0].cpu().numpy())
    chain.log_event('insert_start', jt=jt.tolist(), inhand6=inhand6.tolist(), cap=cap, warm=warm, T=T)
    outcome, t_seat, obs_err, truth_seat = 'timeout', None, [], None
    for t in range(T):
        with torch.no_grad():
            a6 = pol(hist)[0].clamp(-1, 1)
        if t < warm:
            a6 = torch.zeros_like(a6)
        if consts.get('max_tcp_speed_m_s', 0) > 0:
            _, _, J = chain.scene.tcp_chain.fk(chain.q_b[0:1], with_jac=True)
            v = (J[:, :3] @ (a6 * scale)[None, :, None])[..., 0].norm() / chain.dt
            a6 = a6 * min(1., consts['max_tcp_speed_m_s'] / max(float(v), 1e-9))
        qb = chain.q_b.clone()
        jr = torch.maximum(torch.minimum(jt[0] + scale * a6, qb[0] + cap), qb[0] - cap)
        jt = torch.stack([jr.clamp(lim[:, 0], lim[:, 1]), q_hold.reshape(6).to(dev).float()])
        chain.hstep(jt[0], jt[1], '5 insert')
        prev = torch.cat([a6, torch.tensor([-1.], device=dev), torch.zeros(6, device=dev), torch.tensor([-1.], device=dev)])
        o = obs_fn(chain.state['q'], chain.state['qd'], prev, jt, inhand6)
        hist = torch.cat([hist[:, 1:], o[:, None]], 1)
        if sim_obs is not None:
            obs_err.append(float((sim_obs(jt, prev, inhand6) - o).abs().max()))
        p, _ = chain.scene.tcp_chain.fk(chain.q_b[0:1])
        depth, vel, seated = det.update(p[0].cpu().numpy())
        share, counts = contact_summary(pol, hist)
        tr = chain.truth()
        if tr is not None and truth_seat is None and tr['seated']:
            truth_seat = t
        if chain.logger is not None:
            chain.logger.row('policy', k=chain.k, t=t, obs=o[0], action=a6, joint_target=jt, contact_share=share,
                             contact_counts=counts, seat_depth_mm=depth * 1e3, seat_speed=vel, seated=seated,
                             true_seated=np.nan if tr is None else float(tr['seated']),
                             true_depth_mm=np.nan if tr is None else tr['tip_depth_mm'])
        if seated:
            outcome, t_seat = 'seated_detected', t
            chain.log_event('seated_detected', step=t, depth_mm=depth * 1e3)
            break
    # stop the push: hold the measured right pose for 0.2 s (the left keeps its press)
    for _ in range(int(round(0.2 / chain.dt))):
        chain.hstep(chain.q_b[0].clone(), q_hold.reshape(6), '5 insert')
    res = dict(outcome=outcome, steps=t + 1, t_seat_s=None if t_seat is None else t_seat * chain.dt,
               depth_mm=det.prev * 1e3, true_seat_tick=truth_seat)
    if obs_err:
        res['obs_max_abs_err'] = float(max(obs_err))
    tr = chain.truth()
    if tr is not None:
        res['true_seated_end'] = bool(tr['seated'])
        res['true_depth_mm_end'] = tr['tip_depth_mm']
    chain.log_event('insert_result', **res)
    log.update({f'insert_{k}': v for k, v in res.items()})
    return res
