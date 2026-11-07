"""Gravity identification of one YAM arm on duchamp3: payload mass, centre of mass and per-joint factors for i2rt's
gravity model, plus a static friction estimate per joint. Our own tool (the athena3 cell's yam_gravity_id.py is not
public); its output is the model safety.connect(gravity=...) and precision_arm.gravity_fit read.

    .venv/bin/python scripts/real/gravity_id.py plan --side right                 # offline: choose and check poses
    .venv/bin/python scripts/real/gravity_id.py collect --side right              # MOVES THE ARM
    .venv/bin/python scripts/real/gravity_id.py fit data/real_yam/gravity_id/<run>  # offline: fit and check

plan: poses are picked from the right arm's configurations in the logged real runs (data/real_runs/2026-11-0[23]),
spread by farthest-point sampling on joints 2-6 and moved by up to `jitter` rad, so they stay where the arm has been
(no self-collision the table check cannot see). Every pose, both approach points and every straight joint path between
consecutive poses must keep `margin` above the table (safety.Checker, all bodies). Saved to
deploy/config/gravity_poses_<side>.json; review it before collect.

collect: plain i2rt PD (no precision controller), the gripper as connect left it. For every pose, twice: move to the
pose + delta on all joints (from above), then to the pose, hold `hold_s`; the same from pose - delta (from below).
The last `avg_s` of each hold is averaged: measured joints, velocity, the driver effort, and the commanded total torque
(i2rt's last motor torque = gravity model + its Coulomb term + command torques, plus kp (target - q) - kd qd: the
motor's MIT PD). At rest both agree to 0.06-0.3 N m (2026-11-06 precision check). Raw samples and per-hold means go
to data/real_yam/gravity_id/<time>_<side>/. Stops (estop, Enter in the console, a safety check) hold the arm and ask
for SUPPORTED before letting go; the end returns to the start pose.

fit: the gravity torque at a pose is the mean of the two approaches (static friction holds the joint on opposite
sides, so it cancels to first order); half their difference estimates the static friction. Models, on i2rt's own yam
+ linear_4310 description (the payload is the "gripper" body):
  current  i2rt as we run it: ee_mass 0.5 kg, model centre of mass, factors 1, 1.1, 1.1, 1.2, 1, 1
  payload  mass and centre of mass fitted, factors 1
  full     mass, centre of mass and the six factors fitted (as athena3's "full")
Every 5th pose is held out and only checked. Output fit.json: ee_mass, ee_ipos, gravity_comp_factor (the chosen model,
`--model`, default full), friction_Nm_median, per-model check rms per joint, and the predicted holding error at the
work pose (residual / kp).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPO))

KP = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0])
KD = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5])
DEFAULT_FACTOR = np.array([1.0, 1.1, 1.1, 1.2, 1.0, 1.0])
EE_MASS = 0.5
DT = 0.02


# ---------------------------------------------------------------- model
class Grav:
    """i2rt's yam + linear_4310 MuJoCo model; gravity torque at q for a payload (mass, centre of mass)."""

    def __init__(self):
        import mujoco
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(
            ArmType.from_string_name('yam'), GripperType.from_string_name('linear_4310'), ee_mass=EE_MASS))
        self.d = mujoco.MjData(self.m)
        self.b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'gripper')
        self.mass0, self.ipos0 = float(self.m.body_mass[self.b]), self.m.body_ipos[self.b].copy()

    def tau(self, q, mass=None, ipos=None):
        m, d = self.m, self.d
        m.body_mass[self.b] = self.mass0 if mass is None else mass
        m.body_ipos[self.b] = self.ipos0 if ipos is None else ipos
        d.qpos[:] = 0
        d.qpos[:6] = q
        d.qvel[:] = 0
        d.qacc[:] = 0
        self.mj.mj_inverse(m, d)
        out = d.qfrc_inverse[:6].copy()
        m.body_mass[self.b], m.body_ipos[self.b] = self.mass0, self.ipos0
        return out


# ---------------------------------------------------------------- plan
def logged_configs(side):
    i = 0 if side == 'right' else 1
    Q = []
    for f in sorted(glob.glob(str(REPO / 'data/real_runs/2026-11-0[23]/*/ticks.npz'))):
        if 'dry' in f:
            continue
        try:
            Q.append(np.load(f, allow_pickle=True)['q'][::25, i, :])
        except Exception:  # noqa: BLE001
            pass
    if not Q:
        raise SystemExit('no logged real runs to pick poses from')
    return np.concatenate(Q)


def path_ok(chk, a, b, n=41):
    return all(chk.violation((1 - s) * a + s * b) is None for s in np.linspace(0, 1, n))


OTHER_REST = [np.zeros(6)] + [np.r_[j1, 0.0, 0.0, j4, j5, j6] for j1 in (-0.25, 0.25) for j4 in (-0.5, 0.5)
                               for j5 in (-0.5, 0.5) for j6 in (0.0,)]     # the other arm limp at rest: where it may lie
LEFT_LIMP = OTHER_REST                                                       # name used before the left arm
# per side: hand box (scene, m), own-side bound for every geom of the arm, joint 1 search range. The right arm's numbers
# are the ones its 2026-11-06 plan used; the left arm's box comes from its logged hand positions (auto) and its own-side
# bound mirrors the right one about the midline between the bases (y 0 and 0.608).
SIDE = {'right': dict(box=((0.0, 0.05, -1.0), (0.22, 0.46, 0.40)), own=('y_max', 0.46), j1=(-0.8, 1.8)),
        'left': dict(box='auto', own=('y_min', 0.608 - 0.46), j1=(-1.8, 0.8))}


def visited_box(two, side, L, pad=0.02, z_up=0.15, z_cap=0.40):
    """Hand box from the arm's logged configurations: min/max of the hand's geom centres, +-pad in x and y, the top
    raised by z_up (open space above the table) up to z_cap."""
    from arm_clearance import HANDS
    P = []
    for q in L[::5]:
        two._pose(side, q)
        P += [two.d.geom_xpos[g].copy() for g in two.g[side] if two.body(g) in HANDS[side]]
    P = np.array(P)
    lo, hi = P.min(0) - [pad, pad, 0.0], P.max(0) + [pad, pad, 0.0]
    return (float(lo[0]), float(lo[1]), -1.0), (float(hi[0]), float(hi[1]), float(min(z_cap, hi[2] + z_up)))


def side_ok(two, q, args, left_poses=OTHER_REST, side='right', box=None):
    """The arm on its own side, its hand inside `box`, and `arm_margin` from the other arm in each of left_poses
    (the other arm's joints). Returns (ok, smallest distance m)."""
    box = box if box is not None else SIDE['right']['box']
    own, bound = SIDE[side]['own']
    y_lo, y_hi, x = two.extent(side, q)
    if (own == 'y_max' and y_hi > bound) or (own == 'y_min' and y_lo < bound) or x < box[0][0]:
        return False, None
    lo, hi = two.hand_box_of(side, q)
    if np.any(lo < np.asarray(box[0])) or np.any(hi > np.asarray(box[1])):
        return False, None
    dmin = min(two.dist(side, q, qo)[0] for qo in left_poses)
    return dmin >= args.arm_margin, dmin


def plan(args):
    """Joints 2-6 from logged configurations (they carry all the gravity); joint 1 (no gravity torque) solved so the
    whole arm stays on its own side, away from the other arm, with the hand inside the area it already used."""
    import safety
    from arm_clearance import TwoArm
    rng = np.random.default_rng(args.seed)
    chk = safety.Checker(clearance=args.margin, tip_clearance=args.margin)
    two = TwoArm()
    side = args.side
    L = logged_configs(side)
    box = SIDE[side]['box']
    if box == 'auto':
        box = visited_box(two, side, L)
    j1_lo, j1_hi = SIDE[side]['j1'] if args.j1_min is None else (args.j1_min, args.j1_max)
    lo, hi = L.min(0), L.max(0)
    B = rng.uniform(lo, hi, size=(20000, 6))
    d23 = np.min(np.linalg.norm(B[:, None, 1:3] - L[None, ::4, 1:3], axis=2), axis=1)
    C = np.concatenate([L, B[d23 <= args.near]])
    C = C[[chk.violation(q) is None for q in C]]
    w = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    j1_grid = np.arange(j1_lo, j1_hi + 1e-9, 0.02)
    own, bound = SIDE[side]['own']

    def solve_j1(q):
        """Joint 1 for this q[1:]: the middle of the longest run of feasible values (most room both ways)."""
        ok = []
        for j1 in j1_grid:
            qq = q.copy()
            qq[0] = j1
            y_lo, y_hi, x = two.extent(side, qq)
            lo_, hi_ = two.hand_box_of(side, qq)
            own_ok = (y_hi <= bound - 0.02) if own == 'y_max' else (y_lo >= bound + 0.02)
            ok.append(own_ok and x >= box[0][0] + 0.02 and np.all(lo_[:2] >= np.asarray(box[0][:2]) + 0.02)
                      and np.all(hi_[:2] <= np.asarray(box[1][:2]) - 0.02) and hi_[2] <= box[1][2])
        ok = np.array(ok)
        if not ok.any():
            return None
        runs, k = [], 0
        while k < len(ok):
            if ok[k]:
                e = k
                while e + 1 < len(ok) and ok[e + 1]:
                    e += 1
                runs.append((k, e))
                k = e + 1
            else:
                k += 1
        a, b = max(runs, key=lambda r: r[1] - r[0])
        return float(j1_grid[(a + b) // 2])

    sel = [int(np.argmin(np.linalg.norm((C - np.median(C, 0)) * w, axis=1)))]
    dmin = np.linalg.norm((C - C[sel[0]]) * w, axis=1)
    poses, dists = [], []
    while len(poses) < args.n and len(sel) < len(C):
        k = int(np.argmax(dmin))
        sel.append(k)
        dmin = np.minimum(dmin, np.linalg.norm((C - C[k]) * w, axis=1))
        for _ in range(10):
            q = np.clip(C[k] + rng.uniform(-args.jitter, args.jitter, 6) * (w > 0), lo, hi)
            j1 = solve_j1(q)
            if j1 is None:
                continue
            q[0] = j1
            pts = [q, q + args.delta, q - args.delta]
            if not all(chk.violation(p) is None for p in pts):
                continue
            oks = [side_ok(two, p, args, side=side, box=box) for p in pts]
            if all(o for o, _ in oks) and all(path_ok(chk, q, p, 11) for p in pts[1:]):
                poses.append(q)
                dists.append(min(dd for _, dd in oks))
                break
    poses, dists = np.array(poses), np.array(dists)

    def hop_ok(a_, b_):
        if not path_ok(chk, a_, b_):
            return False
        return all(side_ok(two, (1 - s) * a_ + s * b_, args, side=side, box=box)[0] for s in np.linspace(0, 1, 21))
    order, left = [0], set(range(1, len(poses)))
    while left:
        cur = poses[order[-1]]
        cand = sorted(left, key=lambda i: np.linalg.norm(poses[i] - cur))
        nxt = next((i for i in cand if hop_ok(cur, poses[i])), None)
        if nxt is None:
            print(f'dropped {len(left)} poses: no safe straight hop from pose {len(order) - 1}')
            break
        order.append(nxt)
        left.discard(nxt)
    poses, dists = poses[order], dists[order]
    other = 'left' if side == 'right' else 'right'
    out = dict(_about=f'gravity_id.py plan: {len(poses)} poses for the {side} arm (motor frame, rad). Joints 2-6 from '
               f'logged real configurations and nearby samples ({args.near} rad on joints 2-3), jitter {args.jitter} rad; '
               f'joint 1 (no gravity torque) solved so every geom of the arm keeps {own} {bound:.3f} m (own side; bases at '
               f'y 0 and 0.608 in the scene) and the hand stays inside the box below (where it has been in the logged runs), '
               f'{args.arm_margin * 1e3:.0f} mm from the {other} arm at rest and at {len(OTHER_REST) - 1} limp variations of '
               f'it. Every pose, both approach points (delta {args.delta} rad, all joints) and every hop path also keep '
               f'{args.margin * 1e3:.0f} mm above the table. collect re-checks all of it against the {other} arm\'s '
               f'measured joints before moving.',
               side=side, delta=args.delta, margin=args.margin, arm_margin=args.arm_margin, own_side=[own, bound],
               hand_box=[list(box[0]), list(box[1])], j1_range=[j1_lo, j1_hi], seed=args.seed,
               poses=[[round(float(x), 5) for x in q] for q in poses],
               other_distance_mm=[round(float(x) * 1e3, 1) for x in dists])
    path = Path(args.out) if args.out else REPO / f'deploy/config/gravity_poses_{args.side}.json'
    path.write_text(json.dumps(out, indent=1))
    G = Grav()
    T = np.array([G.tau(q) for q in poses])
    print(f'{len(poses)} poses -> {path}  (asked {args.n})')
    print('joint ranges covered:', [f'{a:+.2f}..{b:+.2f}' for a, b in zip(poses.min(0), poses.max(0))])
    print('model gravity torque range per joint (N m):', [f'{a:+.1f}..{b:+.1f}' for a, b in zip(T.min(0), T.max(0))])
    print(f'hand box {np.round(box[0], 3).tolist()} .. {np.round(box[1], 3).tolist()}; own side {own} {bound:.3f}')
    print(f'smallest distance to the {other} arm (rest and limp variations): {dists.min() * 1e3:.0f} mm; '
          f'smallest table clearance above the margin: {min(min_clear(chk, q) for q in poses) * 1e3:.1f} mm')
    return out


def min_clear(chk, q):
    """Smallest height of any checked body above its own limit (m), as safety.Checker measures it."""
    import mujoco
    chk.d.qpos[:] = 0
    chk.d.qpos[:6] = q
    mujoco.mj_kinematics(chk.m, chk.d)
    best = np.inf
    for g, verts, zmin in chk.geoms:
        if verts is None:
            z = chk.d.geom_xpos[g][2] - chk.m.geom_rbound[g]
        else:
            z = (verts @ chk.d.geom_xmat[g].reshape(3, 3).T[:, 2] + chk.d.geom_xpos[g][2]).min()
        best = min(best, z - zmin)
    return best


# ---------------------------------------------------------------- collect
def collect(args):
    import safety
    D = __import__('deploy')                              # scripts/real/deploy.py (HERE first on sys.path)
    import estop
    from arm_clearance import ContactGuard, TwoArm
    cfg = json.loads((REPO / f'deploy/config/gravity_poses_{args.side}.json').read_text())
    poses, delta = np.asarray(cfg['poses'], float), float(cfg['delta'])
    other = 'left' if args.side == 'right' else 'right'
    chk = safety.Checker(clearance=cfg['margin'], tip_clearance=cfg['margin'])
    for q in poses:
        for p in (q, q + delta, q - delta):
            if chk.violation(p):
                raise SystemExit(f'pose file has an unsafe point: {chk.violation(p)}; rerun plan')
    two, guard = TwoArm(), ContactGuard()
    sys.setswitchinterval(0.0005)
    out_dir = REPO / 'data/real_yam/gravity_id' / f'{time.strftime("%Y%m%d_%H%M%S")}_{args.side}'
    out_dir.mkdir(parents=True, exist_ok=True)
    # the other arm: connected and held where it lies, so its real joints are known and it cannot be pushed away
    arm_o = D.RealArm(other, KP, KD)
    q_other = np.asarray(arm_o.robot.get_joint_pos(), float)[:6]
    arm = D.RealArm(args.side, KP, KD)
    r = arm.robot
    q_start = np.asarray(r.get_joint_pos(), float)
    rows, holds, at_start, skipped = [], [], True, []
    margin = args.arm_margin_live
    meta = dict(args=vars(args), poses_file=f'deploy/config/gravity_poses_{args.side}.json', delta=delta, q_start=q_start.tolist(),
                q_other=q_other.tolist(), info=arm.info(), kp=KP.tolist(), kd=KD.tolist(),
                started=time.strftime('%Y-%m-%d %H:%M:%S'))

    def clear(p):
        return chk.violation(p) is None and two.dist(args.side, p, q_other)[0] >= margin

    def hop_ok(a_, b_):
        return path_ok(chk, a_, b_) and two.path_min_of(args.side, a_, b_, q_other, n=21) >= margin

    d_start = two.dist(args.side, q_start[:6], q_other)[0]
    print(f'{other} arm measured at {np.round(q_other, 3).tolist()}; {args.side} arm starts {d_start * 1e3:.0f} mm from it')
    usable = [i for i, q in enumerate(poses) if i >= args.start and all(clear(p) for p in (q, q + delta, q - delta))]
    skipped += [(i, 'pose or approach point within %.0f mm of the measured %s arm or the table' % (margin * 1e3, other))
                for i in range(args.start, len(poses)) if i not in usable]
    meta['usable'] = usable

    def sample(tag):
        with r._state_lock:
            q, qd = r._joint_state.pos.copy(), r._joint_state.vel.copy()
            eff = np.asarray(getattr(r._joint_state, 'eff', np.full(7, np.nan)), float).copy()
        lm = getattr(r, '_last_motor_torques', None)
        lm = np.full(7, np.nan) if lm is None else np.asarray(lm, float).copy()
        qdes = np.asarray(arm.safe.last_safe, float)
        total = lm[:6] + KP * (qdes[:6] - q[:6]) - KD * qd[:6]
        rows.append(dict(t=time.monotonic(), q=q[:6], qd=qd[:6], eff=eff[:6], motor_ff=lm[:6], qdes=qdes[:6], total=total,
                         grip=float(q[6]) if len(q) > 6 else np.nan, tag=tag))
        why = guard.check(q, qdes, eff, moving=tag.startswith('move'))
        if why is not None:
            arm.safe._hold(np.asarray(r.get_joint_pos(), float))     # stop pushing: hold where it is now
            raise safety.SafetyStop(f'contact guard ({tag}): {why}')

    def move(goal, tag):
        q0 = np.asarray(arm.safe.last_safe, float).copy()
        q1 = q0.copy()
        q1[:6] = goal
        if not hop_ok(q0[:6], q1[:6]):
            raise safety.SafetyStop(f'path to {np.round(goal, 3).tolist()} not clear of the table / the {other} arm')
        n = max(1, int(np.ceil(max(0.6, 1.5 * float(np.abs(q1[:6] - q0[:6]).max()) / args.max_vel) / DT)))
        t_next = time.monotonic()
        for i in range(1, n + 1):
            if estop.stopped():
                arm.hold()
                raise safety.SafetyStop('software stop (estop)')
            s = i / n
            arm.safe.command(q0 + (q1 - q0) * (s * s * (3 - 2 * s)))
            sample('move:' + tag)
            t_next += DT
            arm.wait(t_next)

    def hold(tag):
        t_end = time.monotonic() + args.hold_s
        t_next = time.monotonic()
        k0 = len(rows)
        while time.monotonic() < t_end:
            if estop.stopped():
                arm.hold()
                raise safety.SafetyStop('software stop (estop)')
            sample(tag)
            t_next += 0.01
            arm.wait(t_next)
        R = [x for x in rows[k0:] if x['t'] >= rows[-1]['t'] - args.avg_s]
        m = {k: np.mean([x[k] for x in R], axis=0) for k in ('q', 'qd', 'eff', 'motor_ff', 'qdes', 'total')}
        m['q_ptp'] = np.ptp([x['q'] for x in R], axis=0)
        holds.append(dict(tag=tag, **{k: np.asarray(v).tolist() for k, v in m.items()}))

    try:
        print(f'{len(usable)} of {len(poses)} poses clear of the measured {other} arm by {margin * 1e3:.0f} mm; skipped {[i for i, _ in skipped]}')
        print(f'about {len(usable) * 2 * (args.hold_s + 3) / 60:.0f} min. Table clear (no PC, no reader in the gripper).')
        if input(f'Start moving the {args.side} arm? [y/N] ').strip() != 'y':
            raise SystemExit('not moved')
        at_start = False
        for i in usable:
            q = poses[i]
            cur = np.asarray(arm.safe.last_safe, float)[:6]
            if not hop_ok(cur, q + delta):
                skipped.append((i, 'no clear straight path from the current pose'))
                print(f'pose {i}: skipped, no clear straight path from here', flush=True)
                continue
            for s_, name in ((1, 'above'), (-1, 'below')):
                move(q + s_ * delta, f'{i}:{name}')
                move(q, f'{i}:{name}')
                hold(f'{i}:{name}')
            h = holds[-2:]
            print(f'pose {i:2d}  total N m {np.round((np.array(h[0]["total"]) + h[1]["total"]) / 2, 2).tolist()}'
                  f'  half-diff {np.round((np.array(h[0]["total"]) - h[1]["total"]) / 2, 2).tolist()}', flush=True)
        move(q_start[:6], 'return')
        at_start = True
    except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
        meta['stopped'] = f'{type(e).__name__}: {e}'
        print(f'\nSTOPPED: {meta["stopped"]}', flush=True)
        try:
            arm.hold()
        except Exception:  # noqa: BLE001
            pass
    finally:
        meta['ended'] = time.strftime('%Y-%m-%d %H:%M:%S')
        meta['n_holds'] = len(holds)
        meta['skipped'] = skipped
        (out_dir / 'holds.json').write_text(json.dumps(dict(meta=meta, holds=holds), indent=1, default=str))
        if rows:
            np.savez_compressed(out_dir / 'samples.npz', **{k: np.array([x[k] for x in rows]) for k in
                                                            ('t', 'q', 'qd', 'eff', 'motor_ff', 'qdes', 'total', 'grip')},
                                tag=np.array([x['tag'] for x in rows]))
        print(f'saved {out_dir}  ({len(holds)} holds)')
        if not at_start:
            print(f'The {args.side} arm is NOT at its start pose and holds powered. Support it, then type SUPPORTED to let it go.', flush=True)
            while True:
                try:
                    if input('support confirmation > ').strip() == 'SUPPORTED':
                        break
                except (EOFError, KeyboardInterrupt):
                    time.sleep(0.5)
        arm.close()
        arm_o.close()                                          # never moved: limp again where it lay


# ---------------------------------------------------------------- fit
def fit(args):
    from scipy.optimize import least_squares
    from arm_clearance import ContactGuard
    guard = ContactGuard()
    runs = [Path(x).resolve() for x in args.run]
    run = runs[-1]                                          # fit.json goes into the last run
    by, meta, dropped = {}, None, []
    for rr in runs:
        H = json.loads((rr / 'holds.json').read_text())
        meta = H['meta'] if meta is None else meta
        for h in H['holds']:
            i, name = h['tag'].split(':')
            if guard.check(h['q'], h['qdes'], h['eff'], moving=False) is not None:     # contact during the hold
                dropped.append(f'{rr.name}:{h["tag"]}')
                continue
            by.setdefault((rr.name, int(i)), {})[name] = h
    idx = sorted(k for k, v in by.items() if 'above' in v and 'below' in v)
    if dropped:
        print('dropped holds with a contact signature:', dropped)
    sig = args.signal
    Q = np.array([(np.array(by[i]['above']['q']) + by[i]['below']['q']) / 2 for i in idx])
    T = np.array([(np.array(by[i]['above'][sig]) + by[i]['below'][sig]) / 2 for i in idx])
    F = np.array([(np.array(by[i]['above'][sig]) - by[i]['below'][sig]) / 2 for i in idx])
    check = np.array([k % 5 == 4 for k in range(len(idx))])
    G = Grav()
    print(f'{", ".join(r_.name for r_ in runs)}: {len(idx)} poses with both approaches ({check.sum()} held out), signal {sig}')

    # a joint whose model gravity torque hardly changes over the poses (joint 1, joint 6) has no identifiable factor:
    # keep it at 1
    G0 = np.array([G.tau(q) for q in Q])
    free = np.ptp(G0, axis=0) > args.min_range

    def pred(x, Qs):
        mass, ipos = x[0], x[1:4]
        fac = np.ones(6)
        if len(x) > 4:
            fac[free] = x[4:4 + free.sum()]
        return np.array([fac * G.tau(q, mass, ipos) for q in Qs])

    def rms(P, Tm):
        return np.sqrt(np.mean((P - Tm) ** 2, axis=0))
    models = {}
    cur = np.array([DEFAULT_FACTOR * G.tau(q) for q in Q])
    models['current'] = dict(ee_mass=EE_MASS, ee_ipos=G.ipos0.tolist(), gravity_comp_factor=DEFAULT_FACTOR.tolist(),
                             fit_rms_Nm=rms(cur[~check], T[~check]).round(3).tolist(), check_rms_Nm=rms(cur[check], T[check]).round(3).tolist())
    x0 = np.r_[EE_MASS, G.ipos0]
    for name, xinit, lo, hi in (('payload', x0, np.r_[0.05, -.1, -.1, -.1], np.r_[2.5, .1, .1, .1]),
                                ('full', np.r_[x0, np.ones(free.sum())], np.r_[0.05, -.1, -.1, -.1, [0.5] * free.sum()],
                                 np.r_[2.5, .1, .1, .1, [2.0] * free.sum()])):
        res = least_squares(lambda x: (pred(x, Q[~check]) - T[~check]).ravel(), xinit, bounds=(lo, hi))
        x = res.x
        fac = np.ones(6)
        if len(x) > 4:
            fac[free] = x[4:]
        models[name] = dict(ee_mass=float(x[0]), ee_ipos=x[1:4].tolist(), gravity_comp_factor=fac.tolist(),
                            fit_rms_Nm=rms(pred(x, Q[~check]), T[~check]).round(3).tolist(),
                            check_rms_Nm=rms(pred(x, Q[check]), T[check]).round(3).tolist(), cost=float(res.cost))
    # holding error the gravity residual leaves at the work pose on the PD springs alone
    work = json.loads((REPO / 'deploy/config/precision_poses.json').read_text()).get(f'{meta["args"]["side"]}_work')
    for name, mdl in models.items():
        if work is None:                                   # no work pose for this arm yet
            break
        work = np.asarray(work, float)
        x = np.r_[mdl['ee_mass'], mdl['ee_ipos'], np.asarray(mdl['gravity_comp_factor'])[free]]
        k = int(np.argmin(np.linalg.norm(Q - work, axis=1)))
        resid = pred(x, Q[k:k + 1])[0] - T[k]
        mdl['nearest_pose_to_work_resid_Nm'] = resid.round(3).tolist()
        mdl['nearest_pose_to_work_sag_mrad'] = (resid / KP * 1e3).round(2).tolist()
    fr = np.median(np.abs(F), axis=0)
    chosen = models[args.model]
    out = dict(source=[str(r_.relative_to(REPO)) for r_ in runs], dropped_holds=dropped, side=meta['args']['side'], signal=sig, model=args.model,
               ee_mass=round(chosen['ee_mass'], 4), ee_ipos=[round(v, 5) for v in chosen['ee_ipos']],
               gravity_comp_factor=[round(v, 4) for v in chosen['gravity_comp_factor']],
               friction_Nm_median=fr.round(3).tolist(), friction_Nm_p90=np.percentile(np.abs(F), 90, axis=0).round(3).tolist(),
               n_poses=len(idx), n_check=int(check.sum()), models=models)
    path = run / 'fit.json'
    path.write_text(json.dumps(out, indent=1))
    print(f'{"model":8s} {"mass kg":>7s} {"com mm":>20s}  check rms per joint N m           fit rms all')
    for name, mdl in models.items():
        com = np.round(np.array(mdl['ee_ipos']) * 1e3, 1).tolist()
        print(f'{name:8s} {mdl["ee_mass"]:7.3f} {str(com):>20s}  {mdl["check_rms_Nm"]}  {np.sqrt(np.mean(np.square(mdl["fit_rms_Nm"]))):.3f}')
    print('full factors', np.round(models['full']['gravity_comp_factor'], 3).tolist(), ' fitted on joints', (np.where(free)[0] + 1).tolist())
    print('static friction median / p90 N m', out['friction_Nm_median'], out['friction_Nm_p90'])
    print(f'-> {path.relative_to(REPO)}')
    from precision_arm import gravity_fit
    gravity_fit(str(path))                                 # the deploy side reads it
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('plan')
    p.add_argument('--side', default='right', choices=('right', 'left'))
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--jitter', type=float, default=0.08)
    p.add_argument('--delta', type=float, default=0.03)
    p.add_argument('--margin', type=float, default=0.045)
    p.add_argument('--near', type=float, default=0.3)
    p.add_argument('--arm_margin', type=float, default=0.10)
    p.add_argument('--j1_min', type=float, default=None, help='joint 1 search range (default per side, SIDE)')
    p.add_argument('--out', default=None, help='write the plan here instead of deploy/config/gravity_poses_<side>.json')
    p.add_argument('--j1_max', type=float, default=None)
    p.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('collect')
    c.add_argument('--side', default='right', choices=('right', 'left'))
    c.add_argument('--hold_s', type=float, default=2.0)
    c.add_argument('--avg_s', type=float, default=0.8)
    c.add_argument('--max_vel', type=float, default=0.15, help='rad/s; the contact guard limits were set from real moves at 0.15')
    c.add_argument('--arm_margin_live', type=float, default=0.08, help='m from the measured other arm, poses and paths')
    c.add_argument('--start', type=int, default=0, help='first pose index of the plan (resume a stopped run)')
    f = sub.add_parser('fit')
    f.add_argument('run', nargs='+', help='one or more collect runs, fitted together')
    f.add_argument('--signal', default='total', choices=('total', 'eff'))
    f.add_argument('--model', default='full', choices=('current', 'payload', 'full'))
    f.add_argument('--min_range', type=float, default=0.3, help='N m: fit a joint factor only if its gravity torque varies more')
    a = ap.parse_args()
    {'plan': plan, 'collect': collect, 'fit': fit}[a.cmd](a)


if __name__ == '__main__':
    main()
