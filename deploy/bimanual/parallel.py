"""Parallel two-arm schedule for the slow checked moves (capture and home). Off by default.

Config block (absent, or enabled false = today's serial behaviour, the same calls as before):

    "parallel_schedule": {
      "enabled": true,
      "capture": true,              # right and left to the ZED capture pose at the same time
      "home": true,                 # left and right home at the same time (after the left lead)
      "home_left_lead_s": 2.0,      # the right home move starts this long after the left one (s)
      "capture_left_lead_s": 0.0,   # capture: the left start after the right start (s); negative = left first
      "clearance_min_mm": 50.0,     # planned paths: smallest arm-arm distance allowed, else the serial move
      "watch_min_mm": 30.0,         # measured poses during the combined move: under it, both arms hold
      "on_plan_fail": "serial",     # 'serial' (today's order) or 'stop'
      "rule": "absolute"            # 'absolute': planned min >= clearance_min_mm; 'start_relative' (needs the operator's
                                    # approval): planned min >= min(clearance_min_mm, start distance - start_tol_mm) and
                                    # >= floor_mm, i.e. never closer than the arms already are at the start
    }

What a combined move does (docs/real/PARALLEL_SCHEDULE_20261115.md):
  * the same per-arm path, speed and checks as RealIO.move_to -> RealArm.move_to (smoothstep in joint space at
    safety.move_max_vel_rad_s, the straight path checked by SafeArm, every command through SafeArm.command, the
    settle and its 0.02 rad goal check). Each arm runs its own RealArm.move_to in a thread; only the start times
    change;
  * before the go: the planned smoothstep paths of both arms from the MEASURED start poses, on the bimanual scene
    model (links, grippers, wrist cameras, fingers), at 50 Hz with the lead: smallest distance >= clearance_min_mm,
    else the serial move (or a stop);
  * during the move: a watch on the measured joints of both arms (the same model): under watch_min_mm, both stop;
  * a stop of either arm (SafetyStop, the software e-stop, the watch, Ctrl-C) stops both: the other arm's
    RealArm.move_to sees the shared stop at its next 20 ms tick and holds; then RealIO.hold() holds both.
The dry run (SimIO) teleports as in the serial path; the plan check and its log run the same way.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

SIDES = ('right', 'left')
DEFAULTS = dict(enabled=False, capture=False, home=False, home_left_lead_s=0.0, capture_left_lead_s=0.0,
                clearance_min_mm=50.0, watch_min_mm=30.0, on_plan_fail='serial', plan_stride=1,
                rule='absolute', start_tol_mm=2.0, floor_mm=30.0)
CONTROL_DT = 0.02          # scripts/real/deploy.CONTROL_DT (RealArm.move_to's tick)


def parse(cfg):
    """The config block with defaults, or None (off)."""
    c = cfg.get('parallel_schedule')
    if not c or not c.get('enabled', False):
        return None
    out = dict(DEFAULTS)
    out.update({k: v for k, v in c.items() if not k.startswith('_')})
    if out['on_plan_fail'] not in ('serial', 'stop'):
        raise ValueError(f"parallel_schedule.on_plan_fail must be 'serial' or 'stop', not {out['on_plan_fail']!r}")
    if out['rule'] not in ('absolute', 'start_relative'):
        raise ValueError(f"parallel_schedule.rule must be 'absolute' or 'start_relative', not {out['rule']!r}")
    if float(out['floor_mm']) < float(out['watch_min_mm']):
        raise ValueError('parallel_schedule: floor_mm must not be under watch_min_mm')
    if float(out['watch_min_mm']) > float(out['clearance_min_mm']):
        raise ValueError('parallel_schedule: watch_min_mm must not exceed clearance_min_mm')
    return out


def smoothstep_path(q0, q1, max_vel, dt=CONTROL_DT):
    """RealArm.move_to's commanded path (motor or believed frame alike): n ticks, smoothstep, endpoints included."""
    q0, q1 = np.asarray(q0, float)[:6], np.asarray(q1, float)[:6]
    n = max(1, int(np.ceil(np.abs(q1 - q0).max() / (max_vel * dt) * 1.5)))
    s = np.arange(0, n + 1) / n
    s = s * s * (3 - 2 * s)
    return q0[None] + s[:, None] * (q1 - q0)[None]


class ArmArmModel:
    """Smallest distance between the two arms (every geom of each arm and its base) on the bimanual scene model, at
    believed joint angles. Sphere bounds prune the exact mj_geomDistance calls (same sets as
    scripts/real/arm_clearance.TwoArm)."""

    def __init__(self, scene):
        import mujoco
        self.mj, m = mujoco, scene.mjm
        self.m, self.d = m, mujoco.MjData(m)
        self.d.qpos[:] = scene.initial_q.cpu().numpy()
        self.qa = {s: list(scene.indices[s][0]) for s in SIDES}

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
        self._ft = np.zeros(6)

    def distance(self, q_right_b, q_left_b, cap=0.3):
        m, d = self.m, self.d
        d.qpos[self.qa['right']] = np.asarray(q_right_b, float)[:6]
        d.qpos[self.qa['left']] = np.asarray(q_left_b, float)[:6]
        self.mj.mj_kinematics(m, d)
        gA, gB = self.g['right'], self.g['left']
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
        names = None if pair is None else tuple(self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_BODY, m.geom_bodyid[k]) for k in pair)
        return float(best), names


def plan(model, start_b, goal_b, max_vel, lead_s, first, stride=1, settle_s=1.0):
    """Planned combined move. start_b / goal_b {side: believed q6}; `first` starts at 0, the other at lead_s.
    Returns dict(min_mm, pair, t_min_s, serial_s, parallel_s): durations from the smoothstep tick counts plus
    settle_s per arm (RealArm.move_to without trim holds 1 s)."""
    other = 'left' if first == 'right' else 'right'
    P = {s: smoothstep_path(start_b[s], goal_b[s], max_vel) for s in SIDES}
    k_lead = int(round(lead_s / CONTROL_DT))
    n = max(len(P[first]), k_lead + len(P[other]))
    best, pair, k_min, start = 1.0, None, 0, None
    for k in range(0, n, max(1, int(stride))):
        q = {first: P[first][min(k, len(P[first]) - 1)],
             other: P[other][min(max(k - k_lead, 0), len(P[other]) - 1)]}
        x, p = model.distance(q['right'], q['left'])
        if start is None:
            start = x
        if x < best:
            best, pair, k_min = x, p, k
    T = {s: (len(P[s]) - 1) * CONTROL_DT + settle_s for s in SIDES}
    return dict(min_mm=round(best * 1e3, 1), start_mm=round(start * 1e3, 1), pair=pair, t_min_s=round(k_min * CONTROL_DT, 2), first=first, lead_s=float(lead_s),
                move_s={s: round(T[s], 2) for s in SIDES}, serial_s=round(T['right'] + T['left'], 2),
                parallel_s=round(max(T[first], lead_s + T[other]), 2))


class _SharedStop:
    """Stands in for scripts/real/estop inside RealArm.move_to during a combined move: stopped() is the operator's
    software stop OR the shared stop of this move."""

    def __init__(self, estop):
        self.estop, self.event, self.why = estop, threading.Event(), None

    def stopped(self):
        return self.event.is_set() or self.estop.stopped()

    def set(self, why):
        if self.why is None:
            self.why = why
        self.event.set()


def move_real(io, q_motor, grip, max_vel, settle_trim, lead_s, first, model, off, watch_min_mm, log=None):
    """Both arms of a RealIO at the same time (the `first` arm at 0, the other after lead_s). Per-arm preparation as
    RealIO.move_to; each arm's own RealArm.move_to in a thread; one shared stop. Raises the first arm's SafetyStop
    (or the watch stop) after both arms hold."""
    D = io.D                     # scripts/real/deploy.py (RealArm, estop)
    sides = [first, 'left' if first == 'right' else 'right']
    for s in sides:              # RealIO.move_to's preparation, per arm
        arm = io.arms[s]
        if io._pending_off.get(s):
            io._pending_off[s] = False
            io._precision_active(arm, s, False)
        if np.any(io.bias.get(s, 0)):
            if log is not None:
                log.event('precision_bias_cleared', side=s, bias=io.bias[s].tolist(), k=getattr(io, 'k', 0), at='move_parallel')
            io.bias[s] = np.zeros(6)
    if grip:                     # both grippers at once, then wait for both (RealIO.move_to: 1 s each, one arm after the other)
        for s in sides:
            if s in grip:
                arm = io.arms[s]
                q7 = np.asarray(arm.robot.get_joint_pos(), float)
                q7[6] = grip[s]
                arm.safe.command(q7)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.0 and any(abs(io.arms[s].robot.get_joint_pos()[6] - grip[s]) > 0.03 for s in grip):
            time.sleep(0.02)
    stop = _SharedStop(D.estop)
    errs, n0 = {}, {s: len(io.arms[s].motion_trace) for s in sides}

    def run(s, delay):
        try:
            t0 = time.monotonic()
            while time.monotonic() - t0 < delay:          # the lead: this arm holds its pose (its last command)
                if stop.stopped():
                    raise io.safety.SafetyStop('combined move stopped during the lead')
                time.sleep(0.005)
            if log is not None:
                log.event('move_to', side=s, goal_motor=np.asarray(q_motor[s], float).tolist(), parallel=True, delay_s=float(delay))
            io.arms[s].move_to(np.asarray(q_motor[s], float), max_vel=max_vel, settle_trim=settle_trim)
        except BaseException as e:  # noqa: BLE001  (SafetyStop, a bug): the other arm stops too
            errs[s] = e
            stop.set(f'{s}: {type(e).__name__}: {e}')

    old_estop = D.estop
    D.estop = stop               # RealArm.move_to reads the module global `estop` every tick
    th = [threading.Thread(target=run, args=(s, 0.0 if s == first else float(lead_s)), name=f'move_{s}', daemon=True) for s in sides]
    watch = dict(min_mm=1e9, pair=None, n=0)
    try:
        for t in th:
            t.start()
        while any(t.is_alive() for t in th):
            q = {s: np.asarray(io.arms[s].robot.get_joint_pos(), float)[:6] + off[SIDES.index(s)] for s in SIDES}
            x, pair = model.distance(q['right'], q['left'])
            watch['n'] += 1
            if x * 1e3 < watch['min_mm']:
                watch.update(min_mm=x * 1e3, pair=pair)
            if x * 1e3 < watch_min_mm and not stop.event.is_set():
                stop.set(f'arm-arm watch: {x * 1e3:.1f} mm < {watch_min_mm:.0f} mm ({pair})')
            time.sleep(CONTROL_DT)
    except BaseException as e:  # noqa: BLE001  (Ctrl-C in the main thread)
        stop.set(f'main: {type(e).__name__}: {e}')
        for t in th:
            t.join(timeout=20.0)
        D.estop = old_estop
        io.hold()
        raise
    finally:
        for t in th:
            t.join(timeout=20.0)
        D.estop = old_estop
        if log is not None:
            for s in sides:
                for r in io.arms[s].motion_trace[n0[s]:]:
                    log.row('moves_' + s, t=r['host_monotonic'], target=np.asarray(r['target'], float),
                            measured=np.asarray(r['measured'], float), settle=float(r['mode'] == 'settle'))
            log.event('move_parallel_watch', min_mm=round(watch['min_mm'], 1), pair=watch['pair'], samples=watch['n'],
                      stop=stop.why)
    io.t_next = None
    if stop.why is not None:
        io.hold()
        first_err = errs.get(first) or errs.get(sides[1])
        if isinstance(first_err, io.safety.SafetyStop) or first_err is None:
            raise io.safety.SafetyStop(f'combined move stopped, both arms hold: {stop.why}')
        raise first_err


def required_mm(rec, c):
    """The planned arm-arm distance a combined move needs (mm)."""
    lim = float(c['clearance_min_mm'])
    if c['rule'] == 'start_relative':
        lim = max(min(lim, rec['start_mm'] - float(c['start_tol_mm'])), float(c['floor_mm']))
    return round(lim, 1)


def passes(rec, c):
    return rec['min_mm'] >= required_mm(rec, c)


def plan_for(ch, goals_b, max_vel, lead_s, first):
    """The plan check of a combined move from the arms' measured poses now (logged as parallel_plan). Cached: the
    move after the operator's go reuses it when every measured joint is still within 0.01 rad of the planned start
    (else it plans again)."""
    c = ch.par
    if ch.par_model is None:
        ch.par_model = ArmArmModel(ch.scene)
    ch.read_state()
    start_b = {s: ch.state['q'][SIDES.index(s)] + ch.off[SIDES.index(s)] for s in SIDES}
    key = (tuple(np.round(np.r_[goals_b['right'], goals_b['left']], 6)), float(lead_s), first, float(max_vel))
    hit = getattr(ch, '_par_cache', None)
    if hit is not None and hit[0] == key and all(np.abs(hit[1][s] - start_b[s]).max() < 0.01 for s in SIDES):
        return hit[2]
    t0 = time.monotonic()
    rec = plan(ch.par_model, start_b, goals_b, max_vel, lead_s, first, stride=c['plan_stride'])
    rec.update(ok=passes(rec, c), clearance_min_mm=required_mm(rec, c), rule=c['rule'], label=ch.label,
               plan_s=round(time.monotonic() - t0, 3))
    ch.log_event('parallel_plan', **rec)
    ch._par_cache = (key, start_b, rec)
    return rec


def move(ch, goals_b, grip, max_vel, lead_s, first):
    """RealChain entry: plan check from the measured start poses (plan_for), then the combined move (or the serial
    move when the plan fails and on_plan_fail is 'serial'). Returns the plan record."""
    c = ch.par
    rec = plan_for(ch, goals_b, max_vel, lead_s, first)
    ok = rec['ok']
    print(f"parallel plan ({ch.label}): arms {rec['min_mm']:.1f} mm (min {rec['clearance_min_mm']:.0f}), "
          f"{rec['parallel_s']:.1f} s instead of {rec['serial_s']:.1f} s{'' if ok else ' -> ' + c['on_plan_fail']}", flush=True)
    order = [first, 'left' if first == 'right' else 'right']
    q_motor = {s: ch.motor(s, goals_b[s]) for s in order}
    if not ok:
        if c['on_plan_fail'] == 'stop':
            from deploy.bimanual.guards import GuardStop
            raise GuardStop(f"parallel plan: arms {rec['min_mm']:.1f} mm < {rec['clearance_min_mm']:.0f} mm")
        ch.io.move_to(q_motor, grip, max_vel=max_vel)          # today's serial move, `first` first
        ch.log_event('parallel_mode', mode='serial (plan check)', label=ch.label)
        return rec
    if getattr(ch.io, 'is_sim', False):
        ch.io.move_to(q_motor, grip, max_vel=max_vel)          # SimIO teleports: order and timing do not matter
        ch.log_event('parallel_mode', mode='parallel (sim teleport)', label=ch.label)
        return rec
    ch.log_event('parallel_mode', mode='parallel', label=ch.label)
    move_real(ch.io, q_motor, grip, max_vel, True, lead_s, first, ch.par_model, ch.off, float(c['watch_min_mm']), log=ch.logger)
    return rec
