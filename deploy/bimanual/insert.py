"""Part 4 insertion on encoders: InsertionTask's actor observation, controller and warm-up, the K-J policy, and a
seated detector that stops the push. Config insert.seat2 / insert.guard add deploy/bimanual/insert_guard.py (off in the older configs).

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


def make_guards(chain, qb, pR, consts):
    """insert.guard and insert.seat2 of the config (deploy/bimanual/insert_guard.py), CPU, one world. Seat 2 needs the align stage's measured gap."""
    from deploy.bimanual.insert_guard import InsertGuard, SeatDetector2
    ic = chain.cfg['insert']
    dev = torch.device('cpu')
    s = (chain.R_sr0.T @ chain.p_sr0).cpu().float()[None]
    gc = {k: v for k, v in (ic.get('guard') or {}).items() if not k.startswith('_')}
    gd = InsertGuard(chain.scene.chain['right'], s, pR.cpu(), 1, dev, dt=chain.dt, **gc)
    gd.reset(torch.tensor([0]), qb[0:1].cpu(), pR.cpu())
    det2 = None
    if ic.get('seat2'):
        ar = getattr(chain, 'align_result', None)
        if not ar or ar.get('gap_mm') is None:
            raise RuntimeError('insert.seat2 needs the measured gap of the align stage (config align.enabled)')
        sc = {k: v for k, v in ic['seat2'].items() if not k.startswith('_')}
        det2 = SeatDetector2(1, dev, ar['gap_mm'], ar['lat_mm'], seat_depth_mm=float(consts['seat_depth_m']) * 1e3, dt=chain.dt, **sc)
    return gd, det2


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


def final_push(chain, pR, q_left, fp):
    """insert.final_push: after a detected seat (and the 0.2 s hold that drops the policy's target lead), move the right-arm target
    straight into the port along its axis by fp['mm'] at fp['v_mm_s'] (scaled by --speed, smoothstep), orientation kept, jaws closed,
    hold fp['hold_s'], then hold the measured pose for 0.3 s so the release starts with no lead. Position-controlled: the push force is
    the PD stiffness times the part of fp['mm'] the plug cannot travel (rig, 2026-11-10: K-J left the plug 1.5-1.8 mm short with a
    ~10 mm deep, 22-28 mm high target). No upward component, unlike K-J's late target."""
    R = torch.as_tensor(pR, dtype=torch.float32).reshape(3, 3)
    tc = chain.scene.chain['right']
    dev = chain.scene.device
    q0 = chain.q_b[0:1].to(dev).float()
    p0, R0 = tc.fk(q0)
    d = (-R[:, 2] * float(fp['mm']) * 1e-3).to(p0.device)          # -z of the port frame = into the port
    v = float(fp['v_mm_s']) * max(float(chain.speed), 1e-3)
    n = max(1, int(round(float(fp['mm']) / v / chain.dt)))
    zp = R[:, 2].to(p0.device)
    lc = chain.scene.chain['left']
    pl0, _ = lc.fk(chain.q_b[1:2].to(dev).float())
    g0 = [float(x) for x in chain.state['grip']]
    max_adv, max_left, max_grip = float(fp.get('max_advance_mm', 4.0)), float(fp.get('max_left_mm', 2.0)), float(fp.get('max_grip_change', 0.02))
    stop, worst = None, dict(advance_mm=0., left_mm=0., grip_change=[0., 0.])

    def watch():
        p, _ = tc.fk(chain.q_b[0:1].to(dev).float())
        pl, _ = lc.fk(chain.q_b[1:2].to(dev).float())
        a_ = float(-((p[0] - p0[0]) @ zp)) * 1e3
        l_ = float((pl[0] - pl0[0]).norm()) * 1e3
        g_ = [abs(float(chain.state['grip'][i]) - g0[i]) for i in range(2)]
        worst['advance_mm'], worst['left_mm'] = max(worst['advance_mm'], a_), max(worst['left_mm'], l_)
        worst['grip_change'] = [max(worst['grip_change'][i], g_[i]) for i in range(2)]
        if a_ > max_adv:
            return f'hand advanced {a_:.2f} mm > {max_adv} (plug moved more than a seated plug can, or slipped in the jaws)'
        if l_ > max_left:
            return f'left hand moved {l_:.2f} mm > {max_left} (the PC press gave way)'
        if max(g_) > max_grip:
            return f'gripper reading changed {[round(x, 3) for x in g_]} > {max_grip} (right, left)'
        return None

    q = q0
    for k in range(n):
        u = (k + 1) / n
        u = u * u * (3 - 2 * u)
        q, _, _ = tc.ik(p0 + d[None] * u, R0, q, iters=10)
        chain.hstep(q[0].to(chain.dev), q_left, '5 insert')
        stop = watch()
        if stop:
            break
    if stop is None:
        for _ in range(int(round(float(fp['hold_s']) / chain.dt))):
            chain.hstep(q[0].to(chain.dev), q_left, '5 insert')
            stop = watch()
            if stop:
                break
    p1, _ = tc.fk(chain.q_b[0:1].to(dev).float())
    adv = float(-((p1[0] - p0[0]) @ zp)) * 1e3
    for _ in range(int(round(0.3 / chain.dt))):          # drop the push: measured right pose, the left keeps its press target
        chain.hstep(chain.q_b[0].clone(), q_left, '5 insert')
    tr = chain.truth()
    chain.log_event('insert_final_push', mm=float(fp['mm']), v_mm_s=float(fp['v_mm_s']), hold_s=float(fp['hold_s']), advance_mm=adv,
                    stopped=stop, worst=worst, true_depth_mm=None if tr is None else tr['tip_depth_mm'], tick=chain.k)
    print(f'final push: advance {adv:.2f} mm, left hand moved {worst["left_mm"]:.2f} mm, grip change {[round(x, 3) for x in worst["grip_change"]]}'
          + (f'; STOPPED: {stop}' if stop else ''), flush=True)
    return adv


def final_push_force(chain, pR, q_left, fp, lat_mm=None, depth0_mm=None):
    """insert.final_push with mode "force" (docs/real/FULL_DEPTH_20261113.md). The position push (final_push) gives a force
    that depends on the arm stiffness, and on the rig the hand moved 5-6 mm down while the plug advanced 2 mm. Here the
    right-arm target is the MEASURED pose plus the joint step whose PD torque equals a chosen Cartesian load:
    kp dq = J^T [F, M], F = force_n along -port_z (ramp ramp_n_s, hold hold_s) + a lateral spring lat_k_n_mm that keeps the
    hand on a reference line, M = a rotation spring rot_k_nm_rad on the start orientation. So the axial PD force is at most
    force_n (joint friction only lowers it; the joint torques are scaled together to stay under tau_max_nm), and the hand stays on the line. With centre_max_mm > 0 the reference line first
    moves toward the port axis by the seat reading lat_mm (clipped), at centre_v_mm_s: after a K4 seat the hand can sit
    2-4 mm off the axis with the plug in the port (2026-11-11 demo_1, speed075_1: up -3.5 / -3.3 mm, reader hanging).
    Stops as final_push (advance, left hand, gripper readings) plus the hand leaving the line by max_lat_err_mm.
    Ends early when the hand has not advanced by still_mm in still_s at full force (the plug is home or stuck).
    fp['unload'] (optional, off by default; docs/real/DEPTH_SPRINGBACK_20261114.md) replaces the 0.3 s drop: see unload_force.
    fp['lat_bias_n'] (optional, absent = [0, 0]; docs/real/FORCE_BUDGET_20261115.md): a lateral load [across, up] (N, port axes) added to
    the lateral spring, scaled with the axial load (f / force_n), so it ramps up and down with it; the sum stays under lat_f_max_n.
    depth0_mm: depth (seat-2 units) at the push start, for fp['depth_report'] only."""
    dev = chain.scene.device
    tc, lc = chain.scene.chain['right'], chain.scene.chain['left']
    R = torch.as_tensor(pR, dtype=torch.float32).reshape(3, 3).to(dev)
    kp = chain.scene.kp[0].to(dev).float()
    q0 = chain.q_b[0:1].to(dev).float()
    p0, R0 = tc.fk(q0)
    pl0, _ = lc.fk(chain.q_b[1:2].to(dev).float())
    g0 = [float(x) for x in chain.state['grip']]
    f_max, ramp = float(fp['force_n']), float(fp.get('ramp_n_s', 4.0)) * max(float(chain.speed), 1e-3)
    hold_s, k_lat, f_lat_max = float(fp.get('hold_s', 1.0)), float(fp.get('lat_k_n_mm', 2.0)), float(fp.get('lat_f_max_n', 6.0))
    k_rot, m_max = float(fp.get('rot_k_nm_rad', 6.0)), float(fp.get('rot_m_max_nm', 0.5))
    tau_max = torch.as_tensor(fp.get('tau_max_nm', [6.0, 6.0, 6.0, 2.5, 2.0, 2.0]), dtype=torch.float32, device=dev)
    max_adv, max_left, max_grip = float(fp.get('max_advance_mm', 4.0)), float(fp.get('max_left_mm', 2.0)), float(fp.get('max_grip_change', 0.02))
    max_lat_err = float(fp.get('max_lat_err_mm', 3.0))
    lat_bias = torch.as_tensor(fp.get('lat_bias_n', [0.0, 0.0]), dtype=torch.float32, device=dev)
    still_mm, still_s = float(fp.get('still_mm', 0.15)), float(fp.get('still_s', 0.6))
    c_max, c_v = float(fp.get('centre_max_mm', 0.0)), float(fp.get('centre_v_mm_s', 2.0)) * max(float(chain.speed), 1e-3)
    shift = torch.zeros(2, device=dev)                      # reference line against the start, port x (across), y (up), mm
    goal = torch.zeros(2, device=dev)
    if c_max > 0 and lat_mm is not None:
        goal = (-torch.as_tensor(lat_mm, dtype=torch.float32, device=dev)).clamp(-c_max, c_max)
    stop, worst = None, dict(advance_mm=0., left_mm=0., grip_change=[0., 0.], lat_err_mm=0.)
    f, t_full, hist, k, done = 0., None, [], 0, None
    n_max = int(round((float(goal.abs().max()) / max(c_v, 1e-3) + f_max / max(ramp, 1e-3) + hold_s + 1.0) / chain.dt))
    def load_step(f_ax, shift_, grip_r=-1.):
        """One tick of the load law: axial f_ax (N) + lateral spring to the line shift_ + rotation spring. Returns (adv, e_lat)."""
        q = chain.q_b[0:1].to(dev).float()
        p, Rm, J = tc.fk(q, with_jac=True)
        rel = (p[0] - p0[0]) @ R * 1e3                     # hand against the push start, port axes, mm
        e_lat = shift_ - rel[:2]                           # mm
        F_lat = (k_lat * e_lat + lat_bias * (f_ax / max(f_max, 1e-6))).clamp(-f_lat_max, f_lat_max)
        F = R[:, 0] * F_lat[0] + R[:, 1] * F_lat[1] - R[:, 2] * f_ax
        Re = R0[0] @ Rm[0].T
        w = torch.stack([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) / 2      # small-angle rotation vector
        M = (k_rot * w).clamp(-m_max, m_max)
        tau = J[0].T.float() @ torch.cat([F, M])
        tau = tau * float((tau_max / tau.abs().clamp(min=1e-9)).clamp(max=1.).min())      # one scale for all joints: the load direction is kept
        dq = tau / kp
        chain.hstep((q[0] + dq).to(chain.dev), q_left, '5 insert', grip_r=grip_r)
        return float(-rel[2]), e_lat

    while k < n_max:
        step = (goal - shift).clamp(-c_v * chain.dt, c_v * chain.dt)
        shift = shift + step
        centring = bool((goal - shift).abs().max() > 1e-6)
        if not centring:
            f = min(f_max, f + ramp * chain.dt)
        adv, e_lat = load_step(f, shift)
        k += 1
        pl, _ = lc.fk(chain.q_b[1:2].to(dev).float())
        l_ = float((pl[0] - pl0[0]).norm()) * 1e3
        g_ = [abs(float(chain.state['grip'][i]) - g0[i]) for i in range(2)]
        le = float(e_lat.norm())
        worst['advance_mm'], worst['left_mm'], worst['lat_err_mm'] = max(worst['advance_mm'], adv), max(worst['left_mm'], l_), max(worst['lat_err_mm'], le)
        worst['grip_change'] = [max(worst['grip_change'][i], g_[i]) for i in range(2)]
        if adv > max_adv:
            stop = f'hand advanced {adv:.2f} mm > {max_adv} (plug moved more than a seated plug can, or slipped in the jaws)'
        elif l_ > max_left:
            stop = f'left hand moved {l_:.2f} mm > {max_left} (the PC press gave way)'
        elif max(g_) > max_grip:
            stop = f'gripper reading changed {[round(x, 3) for x in g_]} > {max_grip} (right, left)'
        elif le > max_lat_err:
            stop = f'hand {le:.2f} mm off the push line > {max_lat_err}'
        if stop:
            break
        if f >= f_max:
            t_full = k if t_full is None else t_full
            hist.append(adv)
            n_still = int(round(still_s / chain.dt))
            if len(hist) > n_still and abs(hist[-1] - hist[-1 - n_still]) < still_mm:
                done = 'still'
                break
            if (k - t_full) * chain.dt >= hold_s:
                done = 'hold'
                break
    p1, _ = tc.fk(chain.q_b[0:1].to(dev).float())
    adv = float(-((p1[0] - p0[0]) @ R[:, 2])) * 1e3
    tr = chain.truth()
    true_loaded = None if tr is None else tr['tip_depth_mm']
    uc = fp.get('unload')
    un = None
    if uc and stop is None:
        un = unload_force(chain, fp, uc, f, shift, load_step, dict(p0=p0, R=R, pl0=pl0, g0=g0, max_adv=max_adv, max_left=max_left,
                                                                    max_grip=max_grip, max_lat_err=max_lat_err, q_left=q_left), worst)
    else:
        for _ in range(int(round(0.3 / chain.dt))):          # drop the push: measured right pose, the left keeps its press target
            chain.hstep(chain.q_b[0].clone(), q_left, '5 insert')
    p2, _ = tc.fk(chain.q_b[0:1].to(dev).float())
    rest = float(-((p2[0] - p0[0]) @ R[:, 2])) * 1e3
    tr = chain.truth()
    chain.log_event('insert_final_push', mode='force', force_n=f_max, force_reached_n=f, centre_mm=shift.tolist(), advance_mm=adv, rest_advance_mm=rest,
                    stopped=stop, end=done, worst=worst, true_depth_mm=None if tr is None else tr['tip_depth_mm'], true_depth_loaded_mm=true_loaded,
                    unload=un, ticks=k, tick=chain.k)
    print(f'final push (force {f:.1f} of {f_max} N): advance {adv:.2f} mm, at rest {rest:.2f} mm' + (' (jaws open)' if un else '')
          + f', centred {[round(float(x), 2) for x in shift]} mm, '
          f'left hand {worst["left_mm"]:.2f} mm, off line {worst["lat_err_mm"]:.2f} mm, grip change {[round(x, 3) for x in worst["grip_change"]]}'
          + (f'; STOPPED: {stop}' if stop else ''), flush=True)
    if un:
        print(f"unload: load {un['f_start_n']:.1f} -> {un['f_open_n']:.1f} N over {un['ramp_s']:.2f} s (jaws closed): hand back {un['back_mm']:.2f} mm "
              f"({un['slope_mm_per_n']:.3f} mm/N); jaws opened at {un['f_open_n']:.1f} N: hand moved {un['open_move_mm']:+.2f} mm; "
              f"after the relax {un['relax_adv_mm']:.2f} mm" + (f"; STOPPED: {un['stopped']}" if un['stopped'] else ''), flush=True)
    if fp.get('depth_report') and depth0_mm is not None:
        rep = dict(push_start_mm=depth0_mm, loaded_mm=depth0_mm + adv,
                   closed_low_load_mm=None if not un else depth0_mm + un['adv_low_mm'], rest_mm=depth0_mm + rest, jaws_open_at_rest=bool(un),
                   label='not verified', note=str(fp['depth_report']))
        chain.log_event('insert_depth_report', **rep)
        print(f"depth (seat-2 units, FK of the hand, not the plug): push start {depth0_mm:.2f}, loaded {depth0_mm + adv:.2f}"
              + (f", jaws closed at {un['f_open_n']:.1f} N {depth0_mm + un['adv_low_mm']:.2f}" if un else f', at rest {depth0_mm + rest:.2f}')
              + ' mm. FULL/SHORT: NOT VERIFIED by FK (2026-11-14: no FK threshold separates the operator labels); the operator checks the silver band.', flush=True)
    return adv


def unload_force(chain, fp, uc, f_start, shift, load_step, ref, worst):
    """fp['unload'] (docs/real/DEPTH_SPRINGBACK_20261114.md): after the force push, instead of dropping the load in one tick (the hand then
    springs back 1.9-3.4 mm in 40-200 ms with the jaws closed, rig 2026-11-13):
      1. ramp the axial load from the reached force down to open_force_n at ramp_n_s (x --speed), jaws closed, the same load law;
      2. open the right jaws with open_force_n still on, for open_s (open_force_n 0: the right target is held fixed at the pose
         reached at 0 N, a stiff position hold, while the jaws open);
      3. ramp the remaining load to zero over relax_s (jaws open; skipped at 0 N), then hold the measured pose for hold_s.
    Dry run 2026-11-14 (seed 7043): with 3 N on and the jaws open the hand jumped 2.47 mm (the 1.0 mm watch fired at 1.64 mm, one
    tick late): a residual load at the jaw opening is a fast force source; the config uses open_force_n 0.
    The left arm keeps its press target throughout (the PC stays clamped while the jaws open). Watches every tick: hand advance since the
    push start > max_advance_mm, left hand > max_left_mm, hand off the line > max_lat_err_mm (as the push); the gripper readings must not
    change by more than max_grip_change while the jaws are closed (step 1) and the left one in all steps (the right jaws open on purpose in
    steps 2-3); and in steps 2-3 the hand must not advance by more than open_max_adv_mm beyond its pose at the start of step 2 (a force
    source with nothing to push moves). A stop holds the measured pose (no load) and the release follows as before."""
    dev = chain.scene.device
    tc, lc = chain.scene.chain['right'], chain.scene.chain['left']
    p0, R, pl0, g0 = ref['p0'], ref['R'], ref['pl0'], ref['g0']
    spd = max(float(chain.speed), 1e-3)
    f_open = float(uc.get('open_force_n', 3.0))
    rate = float(uc.get('ramp_n_s', 4.0)) * spd
    open_s, relax_s, hold_s = float(uc.get('open_s', 0.6)) / spd, float(uc.get('relax_s', 0.4)) / spd, float(uc.get('hold_s', 0.2))
    open_max = float(uc.get('open_max_adv_mm', 1.0))
    f_open = min(f_open, f_start)
    out = dict(f_start_n=f_start, f_open_n=f_open, stopped=None, step=None, tick_start=chain.k)

    def adv_now():
        p, _ = tc.fk(chain.q_b[0:1].to(dev).float())
        return float(-((p[0] - p0[0]) @ R[:, 2])) * 1e3

    def watch(adv, e_lat, right_grip, adv_open=None):
        pl, _ = lc.fk(chain.q_b[1:2].to(dev).float())
        l_ = float((pl[0] - pl0[0]).norm()) * 1e3
        g_ = [abs(float(chain.state['grip'][i]) - g0[i]) for i in range(2)]
        le = float(e_lat.norm())
        worst['advance_mm'], worst['left_mm'], worst['lat_err_mm'] = max(worst['advance_mm'], adv), max(worst['left_mm'], l_), max(worst['lat_err_mm'], le)
        if right_grip:
            worst['grip_change'][0] = max(worst['grip_change'][0], g_[0])
        worst['grip_change'][1] = max(worst['grip_change'][1], g_[1])
        if adv > ref['max_adv']:
            return f'hand advanced {adv:.2f} mm > {ref["max_adv"]} (unload)'
        if l_ > ref['max_left']:
            return f'left hand moved {l_:.2f} mm > {ref["max_left"]} (the PC press gave way; unload)'
        if (right_grip and g_[0] > ref['max_grip']) or g_[1] > ref['max_grip']:
            return f'gripper reading changed {[round(x, 3) for x in g_]} > {ref["max_grip"]} (right, left; unload)'
        if le > ref['max_lat_err']:
            return f'hand {le:.2f} mm off the push line > {ref["max_lat_err"]} (unload)'
        if adv_open is not None and adv - adv_open > open_max:
            return f'hand advanced {adv - adv_open:.2f} mm while the jaws opened > {open_max} (unload)'
        return None

    a_start = adv_now()
    f, stop, trace = f_start, None, []
    while f > f_open and stop is None:                                    # 1. ramp down, jaws closed
        f = max(f_open, f - rate * chain.dt)
        adv, e_lat = load_step(f, shift)
        trace.append((f, adv))
        stop = watch(adv, e_lat, True)
        out['step'] = 'ramp'
    a_low = adv_now()
    tr = chain.truth()
    out['true_depth_low_mm'] = None if tr is None else tr['tip_depth_mm']
    a_open0 = a_low
    out['tick_open'] = chain.k
    q_fix = chain.q_b[0].clone()                                          # open_force_n 0: a fixed position hold while the jaws open
    if stop is None:                                                      # 2. open the jaws with f_open on (or the fixed hold)
        out['step'] = 'open'
        for _ in range(int(round(open_s / chain.dt))):
            if f_open > 0:
                adv, e_lat = load_step(f, shift, grip_r=1.)
            else:
                chain.hstep(q_fix, ref['q_left'], '5 insert', grip_r=1.)
                p_, _ = tc.fk(chain.q_b[0:1].to(dev).float())
                rel_ = (p_[0] - p0[0]) @ R * 1e3
                adv, e_lat = float(-rel_[2]), shift - rel_[:2]
            stop = watch(adv, e_lat, False, a_open0)
            if stop:
                break
    a_opened = adv_now()
    out['tick_relax'] = chain.k
    if stop is None and f_open > 0:                                       # 3. relax the rest of the load, jaws open
        out['step'] = 'relax'
        n = max(1, int(round(relax_s / chain.dt)))
        for i in range(n):
            adv, e_lat = load_step(f * (1 - (i + 1) / n), shift, grip_r=1.)
            stop = watch(adv, e_lat, False, a_open0)
            if stop:
                break
    for _ in range(int(round(hold_s / chain.dt))):                        # hold the measured pose (no load), jaws open unless a closed-jaw stop
        chain.hstep(chain.q_b[0].clone(), ref['q_left'], '5 insert', grip_r=1. if out['step'] != 'ramp' else -1.)
    out['stopped'] = stop
    if stop is None:
        out['step'] = 'done'
    tr = chain.truth()
    fs = np.array([x[0] for x in trace]) if trace else np.zeros(0)
    ad = np.array([x[1] for x in trace]) if trace else np.zeros(0)
    slope = float(np.polyfit(fs, ad, 1)[0]) if len(fs) > 5 and np.ptp(fs) > 0.5 else float('nan')
    out.update(adv_start_mm=a_start, adv_low_mm=a_low, back_mm=a_start - a_low, slope_mm_per_n=slope, ramp_s=len(trace) * chain.dt,
               open_move_mm=a_opened - a_open0, relax_adv_mm=adv_now(), true_depth_end_mm=None if tr is None else tr['tip_depth_mm'])
    chain.log_event('insert_unload', **out)
    return out



def run_insertion(chain, cmd, q_hold, log, sim_obs=None):
    """Hand-over and insertion from the approach command `cmd` (1, 6 believed) with the left at q_hold (1, 6).
    Returns a dict (outcome, steps, seated tick, depth). sim_obs(jt, prev, inhand) -> the sim env's actor obs for
    the same controller state (dry-run equivalence check)."""
    cfg, ic, dev = chain.cfg, chain.cfg['insert'], chain.dev
    consts, pol = chain.consts, chain.pol
    scale, cap = 0.04, float(pol.meta['target_cap_rad'])
    ej_pol = pol.meta.get('env_json') or {}
    n_prior = 3 if ej_pol.get('vision_prior_obs') else 0
    assert pol.meta['action_mode'] == 'accumulate' and consts['obs_dim'] == 68 and pol.obs_dim == 68 + n_prior, pol.meta
    H, warm = int(pol.meta['history']), int(consts['warmup_steps'])
    T = int(round(ic['insert_s'] / chain.dt))
    lim = chain.scene.indices['right'][4].to(dev)
    obs_fn = InsertObs(chain.scene, consts, dev)
    if n_prior:
        # K3: port-vision plug position at the hand-over (mm / 10: across, up, gap), held (InsertionCfg.vision_prior_obs)
        ar = getattr(chain, 'align_result', None)
        if not ar or ar.get('gap_mm') is None or ar.get('lat_mm') is None:
            raise RuntimeError('this policy needs the align stage hand-over measurement (config align.enabled, mode full)')
        prior = torch.tensor([float(ar['lat_mm'][0]), float(ar['lat_mm'][1]), float(ar['gap_mm'])], device=dev) / 10
        _base_fn = obs_fn
        obs_fn = lambda *a_, _f=_base_fn, _p=prior: torch.cat([_f(*a_), _p[None]], 1)  # noqa: E731
        obs_fn.ch, obs_fn.base = _base_fn.ch, _base_fn
        chain.log_event('insert_vision_prior', prior_mm=(prior * 10).tolist())
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
    if (pol.meta.get('env_json') or {}).get('rel_pose_obs'):
        # K2: obs[50:56] relative to the believed TCP pose at the hand-over (InsertionCfg.rel_pose_obs)
        if ic.get('obs_recenter'):
            raise RuntimeError('insert.obs_recenter is for policies with absolute pose features; this policy uses rel_pose_obs')
        p_ho, R_ho = chain.scene.tcp_chain.fk(qb[0:1].to(dev).float())
        tgt = getattr(obs_fn, 'base', obs_fn)
        tgt.p_obs, tgt.R_obs = p_ho[0], R_ho[0]
        chain.log_event('insert_rel_pose_obs', p_ho=p_ho[0].tolist())
    o = obs_fn(chain.state['q'], chain.state['qd'], prev, jt, inhand6)
    # insert.obs_recenter (docs/real/CONTACT_ALIGN_PLAN_20261107.md): the TCP position and Euler features [50:56] are FK relative to a
    # fixed nominal pose; on the rig they read ~3 sd outside the sim's hand-over spread (PC pose, encoder offsets). Shift them, once at the
    # hand-over, to the sim's hand-over mean; the shift is held, so the policy sees motion relative to the hand-over only.
    rc = ic.get('obs_recenter')
    if rc:
        shift = torch.zeros_like(o)
        shift[:, 50:53] = torch.tensor(rc['tcp_obs'], device=dev) - o[:, 50:53]
        shift[:, 53:56] = torch.tensor(rc['eul'], device=dev) - o[:, 53:56]
        if rc.get('inhand') is not None:
            inhand6 = torch.tensor(rc['inhand'], device=dev, dtype=torch.float32)
            o = obs_fn(chain.state['q'], chain.state['qd'], prev, jt, inhand6)
        _obs_fn0 = obs_fn
        obs_fn = lambda *a_, _f=_obs_fn0, _s=shift: _f(*a_) + _s  # noqa: E731
        o = o + shift
        chain.log_event('insert_obs_recenter', shift=shift[0, 50:56].tolist(), inhand=inhand6.tolist())
    hist = o[:, None].expand(-1, H, -1).clone()
    pR = chain.port[1]
    det = SeatDetector(pR[0].cpu(), chain.a.stop * 1e-3 + chain.a.stop_offset_mm[2] * 1e-3, consts['seat_depth_m'],
                       ic['seat_margin_m'], ic['seat_hold_s'], ic['seat_speed_m_s'], chain.dt)
    p0, _ = chain.scene.tcp_chain.fk(qb[0:1])
    det.update(p0[0].cpu().numpy())
    gd, det2 = make_guards(chain, qb, pR, consts) if (ic.get('guard') or ic.get('seat2')) else (None, None)
    chain.log_event('insert_start', jt=jt.tolist(), inhand6=inhand6.tolist(), cap=cap, warm=warm, T=T)
    outcome, t_seat, obs_err, truth_seat = 'timeout', None, [], None
    s2 = None
    obs_seg = []
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
        jr = jr.clamp(lim[:, 0], lim[:, 1])
        if gd is not None and ic.get('guard'):
            # no press abort or bound stop with the plug at seat depth on the port axis: it is in, not on the case (dry run: lifted out at true depth 5.4 mm)
            in_port = None if det2 is None else det2.seated | getattr(det2, 'deep_ok', torch.zeros_like(det2.seated))
            jr2, _ = gd.limit(chain.q_b[0:1].to(gd.dev), jr[None].to(gd.dev), in_port)
            jr = jr2[0].to(dev)
        jt = torch.stack([jr, q_hold.reshape(6).to(dev).float()])
        chain.hstep(jt[0], jt[1], '5 insert')
        prev = torch.cat([a6, torch.tensor([-1.], device=dev), torch.zeros(6, device=dev), torch.tensor([-1.], device=dev)])
        o = obs_fn(chain.state['q'], chain.state['qd'], prev, jt, inhand6)
        hist = torch.cat([hist[:, 1:], o[:, None]], 1)
        if sim_obs is not None:
            d_ = (sim_obs(jt, prev, inhand6) - o).abs()[0]
            obs_err.append(float(d_[:50].max()))      # joints, velocities, actions, target errors (frame-independent)
            obs_seg.append([float(d_[50:56].max()), float(d_[56:68].max()), float(d_[68:].max()) if d_.numel() > 68 else 0.])
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
        if gd is not None:
            rel = gd.rel(chain.q_b[0:1].to(gd.dev))
            s2 = det2.update(rel) if det2 is not None else None
            if chain.logger is not None:
                chain.logger.row('guard', k=chain.k, t=t, rel_m=rel[0], rel_t=gd.rel(jt[0:1].to(gd.dev))[0], clamped=int(gd.clamped[0]), stop=int(gd.stop[0]),
                                 depth2_mm=float('nan') if s2 is None else float(s2['depth'][0]), lat2_mm=np.full(2, np.nan) if s2 is None else s2['lat'][0],
                                 gate_ok=False if s2 is None else bool(s2['gate_ok'][0]), seated2=False if s2 is None else bool(s2['seated'][0]))
            if s2 is not None:
                seated, depth = bool(s2['seated'][0]), float(s2['depth'][0]) * 1e-3
            if ic.get('guard') and int(gd.stop[0]) and not seated:
                outcome = {1: 'guard_bound', 2: 'press_abort'}[int(gd.stop[0])]
                _, _, Jf = chain.scene.tcp_chain.fk(chain.q_b[0:1], with_jac=True)
                tau = chain.scene.kp[0].to(dev) * (jt[0] - chain.q_b[0])
                F = torch.linalg.lstsq(Jf[0].T.to(dev), tau[:, None]).solution[:3, 0].cpu().numpy()
                chain.log_event('insert_stop', outcome=outcome, step=t, rel_mm=rel[0].tolist(), F_pd_port_N=(pR[0].cpu().numpy().T @ F).tolist(), clamped=int(gd.clamped[0]))
                break
        if seated:
            outcome, t_seat = 'seated_detected', t
            chain.log_event('seated_detected', step=t, depth_mm=depth * 1e3)
            break
        if det2 is not None and t == T - 1 and bool(det2.deep_ok[0]):      # time limit with the plug at depth on the axis, never at rest for 0.5 s
            outcome, t_seat = 'seated_detected', t
            chain.log_event('seated_detected', step=t, depth_mm=depth * 1e3, at_time_limit=True)
    # stop the push: hold the measured right pose for 0.2 s (the left keeps its press)
    for _ in range(int(round(0.2 / chain.dt))):
        chain.hstep(chain.q_b[0].clone(), q_hold.reshape(6), '5 insert')
    fp = ic.get('final_push')
    if fp and outcome == 'seated_detected':
        # push only into a plug that already reads as in the port: both seat readings at depth, close to the axis
        d2 = None if s2 is None else float(s2['depth'][0])
        l2 = None if s2 is None else [float(x) for x in s2['lat'][0]]
        why = None
        if d2 is None:
            why = 'no seat-2 reading (insert.seat2 off)'
        elif d2 < float(fp.get('min_depth_mm', 5.0)):
            why = f'depth {d2:.2f} mm < {fp.get("min_depth_mm", 5.0)}'
        elif max(abs(x) for x in l2) > float(fp.get('max_lat_mm', 2.5)):
            why = f'lateral {l2} mm beyond {fp.get("max_lat_mm", 2.5)}'
        if why is None:
            if fp.get('mode', 'position') == 'force':
                d0 = None
                if fp.get('depth_report'):
                    p_now, _ = chain.scene.tcp_chain.fk(chain.q_b[0:1])
                    zp_ = torch.as_tensor(pR[0], dtype=torch.float32).reshape(3, 3)[:, 2].to(p_now.device)
                    d0 = float(-((p_now[0] - p0[0].to(p_now.device)) @ zp_)) * 1e3 - float(det2.gap0[0])
                final_push_force(chain, pR[0], q_hold.reshape(6), fp, lat_mm=l2, depth0_mm=d0)
            else:
                final_push(chain, pR[0], q_hold.reshape(6), fp)
        else:
            chain.log_event('insert_final_push_skipped', reason=why, depth2_mm=d2, lat2_mm=l2, tick=chain.k)
            print(f'final push skipped: {why}', flush=True)
    p_end, _ = chain.scene.tcp_chain.fk(chain.q_b[0:1])
    z_port = torch.as_tensor(pR[0], dtype=torch.float32).reshape(3, 3)[:, 2].to(p_end.device)
    advance_mm = float(-((p_end[0] - p0[0].to(p_end.device)) @ z_port)) * 1e3     # TCP advance into the port since the hand-over (believed FK displacement)
    res = dict(outcome=outcome, steps=t + 1, t_seat_s=None if t_seat is None else t_seat * chain.dt,
               depth_mm=det.prev * 1e3, true_seat_tick=truth_seat, advance_mm=advance_mm)
    if fp and fp.get('full_depth_mm') is not None and outcome == 'seated_detected':
        # depth at rest (no push force) in the seat reading's units: advance since the hand-over minus the measured gap
        gap0 = float(det2.gap0[0]) if det2 is not None else det.gap0 * 1e3
        res['rest_depth_mm'] = advance_mm - gap0
        res['full_depth'] = bool(res['rest_depth_mm'] >= float(fp['full_depth_mm']))
        chain.log_event('insert_depth', rest_depth_mm=res['rest_depth_mm'], full_depth_mm=float(fp['full_depth_mm']), full_depth=res['full_depth'], gap_mm=gap0)
        print(f"depth at rest {res['rest_depth_mm']:.2f} mm (fully in from {fp['full_depth_mm']} mm): {'FULL' if res['full_depth'] else 'SHORT'}", flush=True)
    if obs_err:
        # obs_max_abs_err: dims 0:50 only. The pose features [50:56] (nominal-relative in the sim env, hand-over-relative
        # for rel_pose_obs policies), twist/in-hand [56:68] and the K3 vision prior [68:71] (the sim env's own reset draw,
        # not the align stage's measurement) are reported apart, as maxima per segment.
        res['obs_max_abs_err'] = float(max(obs_err))
        res['obs_seg_max_abs_err'] = dict(zip(('pose_50_56', 'twist_inhand_56_68', 'prior_68_71'), np.asarray(obs_seg).max(0).round(4).tolist()))
    tr = chain.truth()
    if tr is not None:
        res['true_seated_end'] = bool(tr['seated'])
        res['true_depth_mm_end'] = tr['tip_depth_mm']
        res['true_rot_deg_end'] = np.asarray(tr['plug_rot_deg'], float).tolist()
        res['true_p_mm_end'] = np.asarray(tr['plug_in_port_mm'], float).tolist()
    chain.log_event('insert_result', **res)
    log.update({f'insert_{k}': v for k, v in res.items()})
    return res
