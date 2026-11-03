"""Real training data for the wrist and in-hand nets: wrist frames of a really picked reader at known poses against the port.

Why: real run 2026-11-02 (full_A, full_A2): same grasp, hand 6.3 mm higher and 4.9 mm further sideways between the runs,
while wrist_v18ws moved about 1 mm. Labels need a ground truth the nets do not provide, so this stage takes one from
contact: the operator jogs the plug into the port. The port then fixes the plug pose to within the port clearance, and
encoder FK displacements from that anchor label every frame of the sweep that follows.

Sequence (run_bimanual --stage collect): capture, pick, left guarded press, press removed (the left hand stays on the PC, so
the view matches a real run), transport, approach, ANCHOR (operator jog), SWEEP, RECHECK (operator re-enters; FK against the
anchor shows grasp slip or a moved PC), retreat, reader back to the operator's hand.

No left press after the latch: hold_max_s (60 s) is much shorter than a sweep. The operator braces the PC by hand while the
plug is in contact. Every target goes through chain.hstep: guards, SafeArm, fingertip floor.

Labels, as the nets' training (scripts/perception/train_approach_estimators.py): plug tip in the port frame (x across, y up,
z gap; mm) and rotation vector (deg). With the anchor (TCP R_a, tip t_a, known rel_a, rv_a):
    p = rel_a + eR^T (tip - t_a) * 1e3        R_rel = eR^T R R_a^T eR Rot(rv_a)
eR = ZED port axes. The plug's offset in the hand only sets the rotation pivot (nominal in-hand pose). Offsets of the FK
cancel in a displacement. Dry runs compare these labels with the sim truth (collect_label_check event).
"""
from __future__ import annotations

import numpy as np
import torch

from blind_insertion.envs import kinematics as K
from deploy.bimanual.guards import GuardStop

LABEL = 'collect'
JOG = {'r': (0, 1.), 'l': (0, -1.), 'u': (1, 1.), 'd': (1, -1.), 'o': (2, 1.), 'i': (2, -1.)}
ROT = {'rx': 0, 'ry': 1, 'rz': 2}
STEP_MM, STEP_DEG = 3.0, 2.0        # largest single jog
JOG_MAX_MM, JOG_MAX_DEG = 25.0, 8.0  # largest total jog from the approach stop
PUSH_MAX_MM = 30.0                  # jog refuses to go further in while the target is this far ahead of the hand
JOG_HELP = """Jog (port frame, as seen from behind the plug in the wrist camera; one command, then Enter):
  r / l  right / left    u / d  up / down    i / o  in / out, followed by mm (largest 3), e.g.  i 0.5
  rx / ry / rz  rotation about the plug tip, deg (largest 2), e.g.  ry -0.5
  anchor full    plug fully seated        anchor 2.5    plug entered about 2.5 mm by eye
  skip           (recheck only) no re-entry         abort   stop, go to retreat"""


def rot(rv_deg):
    """Rotation matrix (3, 3) of a rotation vector in degrees."""
    v = torch.as_tensor(rv_deg, dtype=torch.float32).reshape(3)
    a = torch.deg2rad(v.norm())
    if float(a) < 1e-9:
        return torch.eye(3)
    return K.axis_angle_mat(v / v.norm(), a)


def sample_plan(n, seed=0, y_lo=-6.):
    """(n, 6) x, y, gap mm, rx, ry, rz deg, in visiting order. Ranges follow the two wrist nets' training sets, wider near
    the stop because the real errors are wider (real hand spread above: 6 mm). Gap >= 3 mm: no contact on the way."""
    g = np.random.default_rng(seed)
    n_stop, n_fine = int(n * .2), int(n * .4)
    n_far = n - n_stop - n_fine
    def draw(m, xy, gap, r, rr):
        return np.c_[g.uniform(-xy, xy, m), g.uniform(max(-xy, y_lo), xy, m), g.uniform(*gap, m),
                     g.uniform(-r, r, m), g.uniform(-r, r, m), g.uniform(-rr, rr, m)]
    pts = np.r_[draw(n_stop, 3., (5., 7.), 1.5, 1.5), draw(n_fine, 12., (3., 10.), 3., 3.), draw(n_far, 12., (10., 45.), 6., 3.)]
    order, cur, left = [], np.array([0., 0., 3., 0., 0., 0.]), list(range(len(pts)))
    while left:                         # shortest next move (mm, with 1 deg counted as 1 mm)
        d = np.linalg.norm(pts[left] - cur, axis=1)
        i = left.pop(int(d.argmin()))
        order.append(i)
        cur = pts[i]
    return pts[order]


class Collector:
    def __init__(self, ch, q_left, eR, log, speed):
        self.ch, self.ql, self.log, self.speed = ch, q_left.reshape(6), log, speed
        self.fk = ch.scene.chain['right']
        self.eR = eR.reshape(3, 3).cpu().float()
        self.s = (ch.R_sr0.T @ ch.p_sr0).cpu().float()    # chain convention: tip = p - R s
        self.qt = ch.q_b[0].clone()                       # last commanded right target (believed)
        self.anchor = None

    def pose(self, q):
        p, R = self.fk.fk(torch.as_tensor(q, dtype=torch.float32).reshape(1, 6).to(self.ch.scene.device))
        return p.reshape(3).cpu(), R.reshape(3, 3).cpu()

    def tip(self, p, R):
        return p - R @ self.s

    def label(self, q):
        """Plug in the port frame from FK and the anchor: (p mm, rv deg)."""
        p, R = self.pose(q)
        a = self.anchor
        lp = a['rel'] + self.eR.T @ (self.tip(p, R) - a['tip']) * 1e3
        Rr = self.eR.T @ R @ a['R'].T @ self.eR @ rot(a['rv'])
        return lp, torch.rad2deg(K.rot_log(Rr))

    def go_anchor_frame(self, rel_mm, rv_deg, **kw):
        """Measured plug to rel_mm, rv_deg in the port frame (free space, gap >= 3 mm)."""
        a = self.anchor
        R = self.eR @ rot(rv_deg) @ rot(a['rv']).T @ self.eR.T @ a['R']
        tip = a['tip'] + self.eR @ (torch.as_tensor(rel_mm, dtype=torch.float32) - a['rel']) * 1e-3
        return self.servo_to(tip + R @ self.s, R, **kw)

    def solve(self, p, R):
        dev = self.ch.scene.device
        q, _, _ = self.fk.ik(p.to(dev)[None], R.to(dev)[None], self.qt.to(dev).float()[None], iters=60)
        pm, Rm = self.pose(q)
        if float((pm - p).norm()) > 5e-4 or float(K.rot_log(Rm @ R.T).norm()) > 5e-3:
            return None
        return q[0].to(self.ch.dev)

    def move(self, q_goal, v_mm_s=None, w_deg_s=5., settle_s=.5):
        """Smooth joint-space move of the right target; the left stays where it is."""
        v = v_mm_s if v_mm_s is not None else 25. * self.speed
        p0, R0 = self.pose(self.qt)
        p1, R1 = self.pose(q_goal)
        T = max(.3, float((p1 - p0).norm()) * 1e3 / v, float(K.rot_log(R1 @ R0.T).norm()) * 57.3 / w_deg_s)
        n = max(1, int(round(T / self.ch.dt)))
        q0 = self.qt.clone().to(self.ch.dev)
        for k in range(n):
            u = (k + 1) / n
            u = u * u * (3 - 2 * u)
            self.ch.hstep(q0 + (q_goal - q0) * u, self.ql, LABEL)
        self.qt = q_goal.clone()
        for _ in range(int(round(settle_s / self.ch.dt))):
            self.ch.hstep(self.qt, self.ql, LABEL)

    def servo_to(self, p_des, R_des, iters=3, tol_mm=.3, **mv):
        """Bring the MEASURED pose to (p_des, R_des): the arm sags and sticks by millimetres against its target (dry run: a
        2 mm jog moved the plug 0.3 mm), so the target gets the measured error, as the approach servo does. Free space only: in
        contact this would ramp the force. Returns False if a target is refused."""
        pt, Rt = self.pose(self.qt)
        for _ in range(max(iters, 1)):
            pm, Rm = self.pose(self.ch.q_b[0])
            err = p_des - pm
            Re = R_des @ Rm.T
            if float(err.norm()) * 1e3 < tol_mm and float(K.rot_log(Re).norm()) * 57.3 < .2:
                break
            Rt = Re @ Rt
            pt = pt + err
            q = self.solve(pt, Rt)
            if q is None or self.ch.floor_chk.violation(q.cpu().numpy().astype(float)):
                return False
            self.move(q, **mv)
        return True

    # ------------------------------------------------------------------ anchor
    def jog(self, title, allow_skip=False):
        """Operator loop on the current target. Returns the answer: ('anchor', depth mm), 'skip' or 'abort'."""
        ch = self.ch
        print('\n' + title + '\n' + JOG_HELP, flush=True)
        off, rv = torch.zeros(3), torch.zeros(3)
        pm0, Rm0 = self.pose(ch.q_b[0])                   # jog offsets count from the measured pose here
        tip0 = self.tip(pm0, Rm0)
        while True:
            self.snapshot('jog_latest')
            ans = ch.ask('jog >', 'skip' if allow_skip else 'anchor auto').strip().lower()
            ch.log_event('collect_jog', cmd=ans, offset_mm=off.tolist(), rv_deg=rv.tolist(), tick=ch.k)
            w = ans.split()
            if not w:
                continue
            if w[0] == 'abort' or (w[0] == 'skip' and allow_skip):
                return w[0]
            if w[0] == 'anchor' and len(w) == 2:
                return ('anchor', w[1])
            try:
                val = float(w[1]) if len(w) == 2 else None
            except ValueError:
                val = None
            if val is None or (w[0] not in JOG and w[0] not in ROT):
                print(JOG_HELP, flush=True)
                continue
            n_off, n_rv = off.clone(), rv.clone()
            if w[0] in JOG:
                if abs(val) > STEP_MM:
                    print(f'largest step is {STEP_MM} mm', flush=True)
                    continue
                ax, sg = JOG[w[0]]
                n_off[ax] += sg * val
            else:
                if abs(val) > STEP_DEG:
                    print(f'largest step is {STEP_DEG} deg', flush=True)
                    continue
                n_rv[ROT[w[0]]] += val
            if float(n_off.norm()) > JOG_MAX_MM or float(n_rv.norm()) > JOG_MAX_DEG:
                print(f'jog limit: {JOG_MAX_MM} mm, {JOG_MAX_DEG} deg from the start of this jog', flush=True)
                continue
            if w[0] == 'i':                               # in: open loop on the target, so the push does not ramp by itself
                pt, Rt = self.pose(self.qt)
                pm, _ = self.pose(ch.q_b[0])
                ahead = float((pm - pt) @ self.eR[:, 2]) * 1e3
                if ahead > PUSH_MAX_MM:
                    print(f'target is {ahead:.1f} mm ahead of the hand (limit {PUSH_MAX_MM}): stop pushing, check the plug', flush=True)
                    continue
                q = self.solve(pt - self.eR[:, 2] * val * 1e-3, Rt)
                if q is None:
                    print('no IK solution for that pose', flush=True)
                    continue
                self.move(q, v_mm_s=3., w_deg_s=2., settle_s=.3)
            else:
                R = self.eR @ rot(n_rv) @ self.eR.T @ Rm0
                tip = tip0 + self.eR @ n_off * 1e-3
                if not self.servo_to(tip + R @ self.s, R, v_mm_s=3., w_deg_s=2., settle_s=.3):
                    print('that pose is refused (IK or table clearance)', flush=True)
                    continue
            off, rv = n_off, n_rv
            pt, _ = self.pose(self.qt)
            pm, Rm = self.pose(ch.q_b[0])
            d = (self.eR.T @ (self.tip(pm, Rm) - tip0)) * 1e3
            ahead = float((pm - pt) @ self.eR[:, 2]) * 1e3
            print(f'asked x {off[0]:+.1f} y {off[1]:+.1f} z {off[2]:+.1f} mm | plug moved x {d[0]:+.1f} y {d[1]:+.1f} z {d[2]:+.1f} mm | '
                  f'target ahead of hand {ahead:+.1f} mm', flush=True)

    def set_anchor(self, depth, seat_depth_mm):
        """The plug is in the port: this measured pose is the reference."""
        ch = self.ch
        for _ in range(int(round(.3 / ch.dt))):           # stop pushing
            self.qt = ch.q_b[0].clone()
            ch.hstep(self.qt, self.ql, LABEL)
        p, R = self.pose(ch.q_b[0])
        tr = ch.truth()
        if tr is not None:                                # dry run: the reference comes from the sim truth
            rel, rv = torch.tensor(tr['plug_in_port_mm'], dtype=torch.float32), torch.tensor(tr['plug_rot_deg'], dtype=torch.float32)
            kind = 'sim_truth'
        else:
            d = seat_depth_mm if depth == 'full' else float(depth)
            rel, rv = torch.tensor([0., 0., -d]), torch.zeros(3)
            kind = 'full' if depth == 'full' else 'partial'
        self.anchor = dict(rel=rel, rv=rv, R=R, tip=self.tip(p, R))
        ch.log_event('collect_anchor', anchor_kind=kind, rel_mm=rel.tolist(), rv_deg=rv.tolist(), q_b=ch.q_b[0].tolist(), tcp_p=p.tolist(), tcp_R=R.tolist(),
                     tip=self.anchor['tip'].tolist(), port_R=self.eR.tolist(), tip_in_hand_s=self.s.tolist(), inhand6=ch.inhand_est6.reshape(-1).tolist(), tick=ch.k)
        print(f'anchor set ({kind}): plug at {np.round(rel.numpy(), 2)} mm in the port frame', flush=True)

    # ------------------------------------------------------------------ sweep
    def snapshot(self, tag):
        """Wrist frame now: policy PNG and raw JPEG in the run's frames/, as every other net input."""
        return self.ch.render(names=('wrist_right',), tag=tag)

    def sample(self, i, plan):
        ch = self.ch
        self.snapshot(f'collect_{i:04d}')
        q = ch.q_b[0].clone()
        lp, lr = self.label(q)
        ev = dict(i=i, plan=[float(x) for x in plan], label_p_mm=lp.tolist(), label_rv_deg=lr.tolist(), q_b=q.tolist(), qd_max=float(np.abs(ch.state['qd'][0]).max()),
                  frame_tag=ch.last_frame_tag, grip=float(ch.state['grip'][0]), tick=ch.k)
        tr = ch.truth()
        if tr is not None:
            ev['truth_p_mm'], ev['truth_rv_deg'] = np.asarray(tr['plug_in_port_mm'], float).tolist(), np.asarray(tr['plug_rot_deg'], float).tolist()
        ch.log_event('collect_sample', **ev)
        return ev

    def sweep(self, n, seed):
        ch = self.ch
        plan = sample_plan(n, seed)
        done, skipped, rows = 0, 0, []
        for i, s in enumerate(plan):
            if not self.go_anchor_frame(s[:3], s[3:], iters=2, tol_mm=.5):
                skipped += 1
                ch.log_event('collect_skip', i=i, plan=s.tolist())
                continue
            rows.append(self.sample(i, s))
            done += 1
            if done % 10 == 0:
                print(f'sweep: {done} frames, {skipped} skipped of {len(plan)}; grip {float(ch.state["grip"][0]):.3f}', flush=True)
        ch.log_event('collect_sweep_done', frames=done, skipped=skipped)
        if rows and 'truth_p_mm' in rows[0]:
            ep = np.array([np.array(r['label_p_mm']) - r['truth_p_mm'] for r in rows])
            er = np.array([np.array(r['label_rv_deg']) - r['truth_rv_deg'] for r in rows])
            res = dict(n=len(rows), p_err_mm_p95=np.percentile(np.abs(ep), 95, axis=0).tolist(), rv_err_deg_p95=np.percentile(np.abs(er), 95, axis=0).tolist())
            ch.log_event('collect_label_check', **res)
            print(f'label check against the sim truth: {res}', flush=True)
        return done, skipped


def run_collect(runner, ch, qr, ep, eR, log):
    """From the picked reader (qr, 1 x 6) to the reader back in the operator's hand."""
    a, dry = runner.a, runner.a.dry_run
    if not runner.confirm(ch, 'hold, press removed: left guarded press, then zero force with the hand left on the PC'):
        return dict(aborted=True)
    q_press = ch.stage_hold(qr, ep, eR, log)
    if not bool(ch.hold_obj.latched[0]):
        raise GuardStop('left hold did not latch (no contact force within the search depth)')
    for _ in range(50):                                   # zero PD error: no press force, no hold-time count
        q_left = ch.q_b[1].clone()
        ch.hstep(qr.reshape(6), q_left, '2 hold')
    ch.pressing = False
    ch.log_event('collect_press_removed', q_left=q_left.tolist(), q_press=q_press.reshape(-1).tolist())
    ch.rt.comp['right'] = ch.fric
    q_left1 = q_left[None]
    q_look = ch.stage_transport(qr, q_left1, ep, eR, log)
    cmd = ch.stage_approach(q_look, q_left1, ep, eR, log)
    print(f'in-hand estimate (mm, deg): {np.round(ch.inhand_est6[0].cpu().numpy(), 2)}', flush=True)
    co = Collector(ch, q_left, eR, log, ch.speed)
    co.qt = cmd.reshape(6).to(ch.dev).float()
    seat_mm = float(ch.consts['seat_depth_m']) * 1e3
    if dry:
        co.set_anchor('sim', seat_mm)
    else:
        ans = co.jog('ANCHOR: brace the PC from the far side with one hand, keep the other on the stop. Jog the plug into the port.')
        if ans == 'abort':
            return back_out(runner, ch, co, ep, eR, log, frames=0)
        co.set_anchor(ans[1], seat_mm)
    rel3 = co.anchor['rel'].clone()
    rel3[2] = 3.
    pm, Rm = co.pose(ch.q_b[0])                           # straight out of the port, open loop first: the plug is still guided
    q_out = co.solve(pm + co.eR[:, 2] * float(rel3[2] - co.anchor['rel'][2]) * 1e-3, Rm)
    if q_out is None:
        raise GuardStop('no IK solution for the anchor retreat')
    co.move(q_out, v_mm_s=3., settle_s=.5)                # the operator still braces
    if not co.go_anchor_frame(rel3, np.zeros(3), v_mm_s=3.):
        raise GuardStop('anchor retreat refused')
    if not runner.confirm(ch, f'sweep: {a.collect_n} poses, about {a.collect_n * 3 / 60:.0f} min. Hands off the PC and clear of the robot'):
        return back_out(runner, ch, co, ep, eR, log, frames=0)
    frames, skipped = co.sweep(a.collect_n, a.collect_seed)
    co.go_anchor_frame(rel3, np.zeros(3))
    if not dry:
        ans = co.jog('RECHECK: brace the PC again. The plug is back 3 mm in front of the anchor: jog it in, then "anchor full" (or the depth). "skip" = no check.', allow_skip=True)
        if isinstance(ans, tuple):
            for _ in range(int(round(.3 / ch.dt))):
                co.qt = ch.q_b[0].clone()
                ch.hstep(co.qt, co.ql, LABEL)
            lp, lr = co.label(ch.q_b[0])
            d = seat_mm if ans[1] == 'full' else float(ans[1])
            ch.log_event('collect_recheck', label_p_mm=lp.tolist(), label_rv_deg=lr.tolist(), depth_mm=d)
            print(f'recheck: back in the port the labels read {np.round(lp.numpy(), 2)} mm, {np.round(lr.numpy(), 2)} deg; expected x, y about 0 and z about {-d:.1f}. '
                  'More than about 1 mm across means the grasp or the PC moved: do not trust this run.', flush=True)
            pm, Rm = co.pose(ch.q_b[0])
            q_out = co.solve(pm + co.eR[:, 2] * (d + 3.) * 1e-3, Rm)
            if q_out is not None:
                co.move(q_out, v_mm_s=3., settle_s=.3)
    return back_out(runner, ch, co, ep, eR, log, frames=frames, skipped=skipped)


def back_out(runner, ch, co, ep, eR, log, **res):
    """Out along the port axis, left hand up, reader into the operator's hand."""
    p, R = co.pose(co.qt)
    q = co.solve(p + co.eR[:, 2] * .04, R)
    if q is not None:
        co.move(q, settle_s=.2)
    ch.unload_left()
    if not runner.a.dry_run:
        runner.release_by_hand(ch)
    return res
