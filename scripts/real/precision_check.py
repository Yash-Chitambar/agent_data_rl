"""Precision controller check on one real arm: step accuracy and settling, plain i2rt PD against the controller.

MOVES THE ARM. The other arm is connected and held where it lies (its gripper calibrates: it opens and closes), and
every pose, step and path must keep --arm_margin from it; a contact guard (joint 1 error / effort) stops the run. Clear the table under the arm (no PC, no reader in
the gripper: connecting calibrates the gripper, it opens and closes).

    .venv/bin/python scripts/real/precision_check.py --plan                       # offline: poses, paths, no arm
    .venv/bin/python scripts/real/precision_check.py --side right --joints 1,2,3 --deltas 0.03,0.003

Sequence: connect (holds where the console parked it), smooth move to the test pose (deploy/config/precision_poses.json,
the seated run's hand-over pose raised 40 mm), then per mode (pd, precision) and per joint and step size: step +d,
hold, back, hold, step -d, hold, back, hold. Holds are recorded at 100 Hz: measured joints, target, velocity, the
command's torque field, the driver effort, the controller's integral and lock. Then back to the start pose and close
(limp where the console parks it). On a stop (Enter in the console, estop.py, a safety check) the arm holds and the
script asks for SUPPORTED before it lets go.

Per step it reports the moved joint's error at the end of the hold, the time until it stays within 1 mrad, the
peak-to-peak over the last second, the largest error of the other joints, and the hand's position error (FK of the
measured against the target pose, mm). Output: data/real_yam/precision/<time>_<side>/{samples.npz, summary.json}.

Controller features: --features integral,settle_lock (default). friction_ff needs this rig's friction values
(--friction j1..j6 N m): leave it off until they are measured.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(REPO))

KP = [80.0, 80.0, 80.0, 40.0, 10.0, 10.0]     # i2rt yam.yml = the bimanual chain's profile
KD = [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
DT = 0.02                                       # command rate, as the chain (50 Hz)
REC_DT = 0.01


def fk_hand(m, d, q6):
    import mujoco
    d.qpos[:] = 0
    d.qpos[:6] = q6
    mujoco.mj_kinematics(m, d)
    return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'gripper')].copy()


def plan(args, q_start=None, q_pose=None):
    import safety
    if q_pose is None:
        poses = json.loads((REPO / 'deploy/config/precision_poses.json').read_text())
        q_pose = np.asarray(poses[args.pose], float)
    c = safety.Checker()
    print(f'test pose {"camera view" if getattr(args, "camera_plan", False) else args.pose}: {np.round(q_pose, 4).tolist()}  checker: {c.violation(q_pose) or "OK"}')
    worst = []
    for j in args.joints:
        for dlt in args.deltas:
            for s in (1, -1):
                q = q_pose.copy()
                q[j - 1] += s * dlt
                why = c.violation(q)
                if why:
                    worst.append((j, s * dlt, why))
    print('step targets:', 'all OK' if not worst else worst)
    if q_start is not None:
        bad = [a for a in np.linspace(0, 1, 81) if c.violation((1 - a) * q_start[:6] + a * q_pose)]
        print(f'path from {np.round(q_start[:6], 3).tolist()}: {"OK" if not bad else f"UNSAFE at {bad[:3]}"}')
        if bad:
            raise SystemExit('unsafe path to the test pose')
    if worst:
        raise SystemExit('a step target is unsafe')
    return q_pose


class Recorder:
    def __init__(self, arm):
        self.arm, self.rows = arm, []

    def sample(self, phase, mode, joint, delta):
        r = self.arm.robot
        with r._state_lock:
            q, qd = r._joint_state.pos.copy(), r._joint_state.vel.copy()
            eff = np.asarray(getattr(r._joint_state, 'eff', np.full(7, np.nan)), float).copy()
        with r._command_lock:
            tq = np.asarray(r._commands.torques, float).copy()
        c = self.arm.ctrl
        st = c.status() if c is not None and c.active else None
        self.rows.append(dict(t=time.monotonic(), q=q[:6], qd=qd[:6], eff=eff[:6], tau_cmd=tq[:6],
                              target=np.asarray(self.arm.safe.last_safe, float)[:6],
                              I=np.zeros(6) if st is None else st['I'], latched=0.0 if st is None else float(st['latched']),
                              phase=phase, mode=mode, joint=joint, delta=delta))


def stopped():
    import estop
    return estop.stopped()


GUARD = None
VISION_FRAMES = 10


def guard_check(arm, rec, moving):
    """Contact guard (arm_clearance.ContactGuard) on the newest sample: hold where the arm is and stop."""
    import safety
    x = rec.rows[-1]
    why = GUARD.check(x['q'], x['target'], x['eff'], moving)
    if why is not None:
        arm.safe._hold(np.asarray(arm.robot.get_joint_pos(), float))
        raise safety.SafetyStop(f'contact guard: {why}')


def other_clear(two, q_other, q_pose, args, margin):
    """Smallest distance (m) from the moving arm to the measured other arm over the test pose, every step target and
    the straight paths pose <-> target."""
    worst = two.dist(args.side, q_pose, q_other)[0]
    for j in args.joints:
        for dlt in args.deltas:
            for s in (1, -1):
                q = q_pose.copy()
                q[j - 1] += s * dlt
                worst = min(worst, two.path_min_of(args.side, q_pose, q, q_other, n=7))
    return worst


def look_point(side, q):
    """Where the wrist camera's optical axis meets the table (base frame, m)."""
    import precision_camera as PC
    T = PC.CamModel(side).T_base_cam(q)
    o, z = T[:3, 3], T[:3, 2]
    return o + (-o[2] / z[2]) * z if z[2] < -1e-3 else o


def place_board(arm, rec, mode):
    """Arm holding at the test pose: wait until the board is seen steadily, then until the operator's hands are clear."""
    import safety
    while True:
        hold(arm, rec, 0.5, mode, 0, 0.0)               # guarded; with VISION set it also tries a measurement
        v = VISION_RECORDS.pop() if VISION_RECORDS else None
        if v and v['T_cam_board'] is not None and v['n_ok'] >= 0.8 * VISION_FRAMES and v['spread_mm'] < 0.5:
            print(f'board seen in {v["n_ok"]} of {VISION_FRAMES} frames, spread {v["spread_mm"]:.2f} mm, '
                  f'reprojection {v["reproj_px"]:.2f} px', flush=True)
            ans = input('Hands clear of the arm and the board stays where it is? [y = go, r = check again, q = stop] ').strip()
            if ans == 'y':
                return
            if ans == 'q':
                raise safety.SafetyStop('operator stop while placing the board')
            continue
        seen = 0 if not v else v['n_ok']
        ans = input(f'board seen in {seen} of {VISION_FRAMES} frames: place it flat under the wrist camera (watch camview), '
                    f'then Enter to look again, q to stop > ').strip()
        if ans == 'q':
            raise safety.SafetyStop('operator stop while placing the board')


def camera_view(args):
    """Board pose (ZED, or --board_xyz) and a safe view pose for the wrist camera; refuses an unsafe one."""
    import safety
    from arm_clearance import TwoArm
    import precision_camera as PC
    import gravity_id as GI
    C = PC._calib()
    if args.board_xyz:
        T_base_board = np.eye(4)
        T_base_board[:3, 3] = np.asarray([float(x) for x in args.board_xyz.split(',')]) - C.BOARD_CENTER
        src = 'given'
    else:
        T_base_board, rp = PC.board_from_zed(args.side)
        src = f'ZED, reprojection {rp:.2f} px'
    c = T_base_board[:3, :3] @ C.BOARD_CENTER + T_base_board[:3, 3]
    poses = json.loads((REPO / 'deploy/config/precision_poses.json').read_text())
    gp = json.loads((REPO / f'deploy/config/gravity_poses_{args.side}.json').read_text())['poses']
    seeds = [poses.get(f'{args.side}_test', gp[0])] + gp
    q = PC.plan_view(args.side, T_base_board, seeds, dist=args.view_dist)
    print(f'board centre {np.round(c, 3).tolist()} m in the {args.side} base ({src})')
    if q is None:
        raise SystemExit('no IK solution for a view of the board: move the board closer to where the hand works')
    two, chk = TwoArm(), safety.Checker()
    side_cfg = GI.SIDE[args.side]
    box = side_cfg['box'] if side_cfg['box'] != 'auto' else GI.visited_box(two, args.side, GI.logged_configs(args.side))
    ok, dmin = GI.side_ok(two, q, types.SimpleNamespace(arm_margin=0.10), GI.OTHER_REST, side=args.side, box=box)
    why = chk.violation(q)
    print(f'view pose {np.round(q, 4).tolist()}: table {why or "OK"}, other arm at rest / limp >= '
          f'{"-" if dmin is None else f"{dmin * 1e3:.0f}"} mm, own side and visited hand box {"OK" if ok else "NOT OK"}')
    if why or not ok:
        raise SystemExit('the view pose fails a safety check: move the board (toward where the hand works, away from the other arm)')
    return q, dict(T_base_board=T_base_board.tolist(), board_centre=c.tolist(), source=src, q_view=q.tolist())


def move(arm, rec, q_goal, mode, joint, delta, max_vel):
    """Smoothstep at 50 Hz through SafeArm, recording."""
    import safety
    q0 = np.asarray(arm.safe.last_safe, float).copy()
    q1 = q0.copy()
    q1[:6] = q_goal
    T = max(0.5, 1.5 * float(np.max(np.abs(q1[:6] - q0[:6]))) / max_vel)
    n = max(1, int(np.ceil(T / DT)))
    t_next = time.monotonic()
    for i in range(1, n + 1):
        if stopped():
            arm.hold()
            raise safety.SafetyStop('software stop (estop)')
        s = i / n
        s = s * s * (3 - 2 * s)
        arm.safe.command(q0 + (q1 - q0) * s)
        rec.sample('move', mode, joint, delta)
        guard_check(arm, rec, moving=True)
        t_next += DT
        arm.wait(t_next)


VISION = None          # precision_camera.BoardCam with --camera
VISION_ON = True       # off for the move back to the start pose (the board is not in view there)
VISION_RECORDS = []


def hold(arm, rec, seconds, mode, joint, delta):
    import safety

    def tick():
        if stopped():
            arm.hold()
            raise safety.SafetyStop('software stop (estop)')
        rec.sample('hold', mode, joint, delta)
        guard_check(arm, rec, moving=False)
    t_end = time.monotonic() + seconds
    t_next = time.monotonic()
    while time.monotonic() < t_end:
        tick()
        t_next += REC_DT
        arm.wait(t_next)
    if VISION is not None and VISION_ON:                # still holding, still guarded between frames
        v = VISION.measure(VISION_FRAMES, between=tick)
        x = rec.rows[-1]
        VISION_RECORDS.append(dict(mode=mode, joint=joint, delta=delta, q=np.asarray(x['q'], float).tolist(),
                                   target=np.asarray(x['target'], float).tolist(),
                                   T_cam_board=None if v is None else v['T_cam_board'].tolist(),
                                   n_ok=0 if v is None else v['n_ok'], spread_mm=None if v is None else v['spread_mm'],
                                   reproj_px=None if v is None else v['reproj_px']))


def metrics(rows, m, d):
    """Per hold after a step: moved joint error at the end, settle time to stay within 1 mrad, last-second p-p,
    other joints' largest end error, hand position error (mm)."""
    out = []
    i = 0
    while i < len(rows):
        if rows[i]['phase'] != 'hold':
            i += 1
            continue
        k = i
        while k < len(rows) and rows[k]['phase'] == 'hold':
            k += 1
        seg = rows[i:k]
        i = k
        j = seg[0]['joint'] - 1
        t = np.array([r['t'] for r in seg])
        t -= t[0]
        E = np.array([r['target'] - r['q'] for r in seg])
        tgt = seg[-1]['target']
        last = t >= t[-1] - 1.0
        ok = np.abs(E[:, j]) < 0.001
        settle = next((t[a] for a in range(len(t)) if ok[a:].all()), None)
        other = np.delete(np.abs(E[-10:]).mean(0), j)
        out.append(dict(mode=seg[0]['mode'], joint=j + 1, delta=seg[0]['delta'], target_j=float(tgt[j]),
                        end_err_mrad=float(E[-10:, j].mean() * 1e3), settle_1mrad_s=None if settle is None else float(settle),
                        pp_last1s_mrad=float(np.ptp(E[last, j]) * 1e3), other_end_err_max_mrad=float(other.max() * 1e3),
                        hand_err_mm=float(np.linalg.norm(fk_hand(m, d, seg[-1]['q']) - fk_hand(m, d, tgt)) * 1e3),
                        I_end=[round(float(x), 3) for x in seg[-1]['I']], latched_end=bool(seg[-1]['latched'])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--side', default='right', choices=('right', 'left'))
    ap.add_argument('--pose', default=None, help='key in deploy/config/precision_poses.json (default <side>_test)')
    ap.add_argument('--joints', default='1,2,3')
    ap.add_argument('--deltas', default='0.03,0.003', help='step sizes, rad (each at most 0.06)')
    ap.add_argument('--modes', default='pd,precision')
    ap.add_argument('--features', default='integral,settle_lock')
    ap.add_argument('--friction', default=None, help='6 static friction values, N m (only with friction_ff)')
    ap.add_argument('--hold_s', type=float, default=3.0)
    ap.add_argument('--max_vel', type=float, default=0.15)
    ap.add_argument('--plan', action='store_true', help='offline: check the pose and step targets, no arm')
    ap.add_argument('--arm_margin', type=float, default=0.08, help='m from the measured other arm')
    ap.add_argument('--gravity', default=None, help='fitted gravity model (gravity_id.py fit.json) for the moving arm; both modes use it')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=JSON', help='precision controller value, e.g. deadband=0.0004')
    ap.add_argument('--camera', action='store_true', help='camera check: a checkerboard flat on the table, seen by the wrist camera')
    ap.add_argument('--camera_plan', action='store_true', help='plan a view pose of the board (ZED or --board_xyz) instead of using the test pose')
    ap.add_argument('--board_xyz', default=None, help='with --camera_plan: board centre x,y,z in the arm base frame (m); default: the ZED measures it')
    ap.add_argument('--cam_frames', type=int, default=10)
    ap.add_argument('--view_dist', type=float, default=0.26)
    args = ap.parse_args()
    args.pose = args.pose or f'{args.side}_test'
    args.joints = [int(x) for x in args.joints.split(',')]
    args.deltas = [float(x) for x in args.deltas.split(',')]
    args.modes = args.modes.split(',')
    feats = [f for f in args.features.split(',') if f]
    assert all(1 <= j <= 6 for j in args.joints) and all(0 < x <= 0.06 for x in args.deltas), 'joints 1-6, deltas 0-0.06 rad'
    assert set(args.modes) <= {'pd', 'precision'}
    if 'friction_ff' in feats and args.friction is None:
        raise SystemExit('friction_ff needs --friction (this rig\'s measured values)')
    if args.camera and args.camera_plan:
        q_pose, view = camera_view(args)
        q_pose = plan(args, q_pose=q_pose)
    else:
        q_pose, view = plan(args), None
        if args.camera:
            view = dict(look_at_table=look_point(args.side, q_pose).tolist())
            print(f'the wrist camera at the test pose looks at the table near {np.round(view["look_at_table"], 3).tolist()} m '
                  f'(x, y in the {args.side} base frame): the board goes there')
    if args.plan:
        return

    import mujoco
    import safety
    from precision_arm import ArmController
    D = __import__('deploy')            # scripts/real/deploy.py (HERE is first on sys.path)
    sys.setswitchinterval(0.0005)
    out_dir = REPO / 'data/real_yam/precision' / f'{time.strftime("%Y%m%d_%H%M%S")}_{args.side}'
    m = safety._model()
    d = mujoco.MjData(m)
    params = {f: True for f in feats}
    if args.friction:
        params['friction'] = [float(x) for x in args.friction.split(',')]
    for kv in args.set:
        k, v = kv.split('=', 1)
        params[k] = json.loads(v)
    from blind_insertion.envs.precision_ctrl import params as check_params
    check_params(params)                               # unknown keys stop here, before anything moves
    global GUARD
    from arm_clearance import ContactGuard, TwoArm
    GUARD, two = ContactGuard(), TwoArm()
    other = 'left' if args.side == 'right' else 'right'
    # the other arm: connected and held where it lies, so its real joints are known (2026-11-06: a planned pose
    # pressed on the limp left arm; this test pose overlaps it in some limp poses of the model)
    arm_o = D.RealArm(other, np.asarray(KP), np.asarray(KD))
    q_other = np.asarray(arm_o.robot.get_joint_pos(), float)[:6]
    arm = D.RealArm(args.side, np.asarray(KP), np.asarray(KD), gravity=args.gravity)
    rec = Recorder(arm)
    q_start = np.asarray(arm.robot.get_joint_pos(), float)
    at_start = True
    summary = dict(args=vars(args), params=params, q_start=q_start.tolist(), q_pose=q_pose.tolist(), q_other=q_other.tolist(),
                   info=arm.info())
    try:
        plan(args, q_start)
        d_pose = other_clear(two, q_other, q_pose, args, args.arm_margin)
        d_path = two.path_min_of(args.side, q_start[:6], q_pose, q_other, n=41)
        summary['other_arm_clear_mm'] = dict(pose_and_steps=round(d_pose * 1e3, 1), path_from_start=round(d_path * 1e3, 1))
        print(f'{other} arm measured at {np.round(q_other, 3).tolist()}: test pose and steps {d_pose * 1e3:.0f} mm from it, '
              f'path from the start {d_path * 1e3:.0f} mm (need {args.arm_margin * 1e3:.0f})')
        if min(d_pose, d_path) < args.arm_margin:
            raise SystemExit(f'too close to the {other} arm: not moving')
        if args.camera:
            global VISION, VISION_FRAMES
            import precision_camera as PC
            VISION, VISION_FRAMES = PC.BoardCam(args.side), args.cam_frames
            summary['camera_view'] = view
        if input(f'\nMove the {args.side} arm to the test pose at {args.max_vel} rad/s? Table clear, no reader. [y/N] ').strip() != 'y':
            raise SystemExit('not moved')
        at_start = False
        move(arm, rec, q_pose, 'pd', 0, 0.0, args.max_vel)
        if args.camera and not args.camera_plan:
            place_board(arm, rec, 'pd')
        hold(arm, rec, args.hold_s, 'pd', 0, 0.0)
        for mode in args.modes:
            if mode == 'precision':
                arm.ctrl = ArmController(arm.robot, lambda: arm.safe.last_safe, params, name=args.side)
                arm.ctrl.start()
                arm.safe.ctrl = arm.ctrl
                hold(arm, rec, args.hold_s, mode, 0, 0.0)
            for j in args.joints:
                for dlt in args.deltas:
                    for s in (1, -1):
                        q = q_pose.copy()
                        q[j - 1] += s * dlt
                        print(f'{mode}: joint {j} {s * dlt * 1e3:+.0f} mrad', flush=True)
                        move(arm, rec, q, mode, j, s * dlt, args.max_vel)
                        hold(arm, rec, args.hold_s, mode, j, s * dlt)
                        move(arm, rec, q_pose, mode, j, 0.0, args.max_vel)
                        hold(arm, rec, args.hold_s, mode, j, 0.0)
            if mode == 'precision':
                summary['ctrl_status_end'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in arm.ctrl.status().items()}
                summary['ctrl_events'] = [(round(t, 3), e, x) for t, e, x in arm.ctrl.events]
                arm.ctrl.stop()
                arm.safe.ctrl, arm.ctrl = None, None
                hold(arm, rec, 1.0, 'pd', 0, 0.0)
        global VISION_ON
        VISION_ON = False
        move(arm, rec, q_start[:6], 'pd', 0, 0.0, args.max_vel)
        hold(arm, rec, 1.0, 'pd', 0, 0.0)
        at_start = True
    except (Exception, KeyboardInterrupt) as e:  # noqa: BLE001
        summary['stopped'] = f'{type(e).__name__}: {e}'
        print(f'\nSTOPPED: {summary["stopped"]}', flush=True)
        try:
            arm.hold()
        except Exception:  # noqa: BLE001
            pass
    finally:
        rows = rec.rows
        if rows:
            out_dir.mkdir(parents=True, exist_ok=True)
            keys = ('t', 'q', 'qd', 'eff', 'tau_cmd', 'target', 'I', 'latched', 'joint', 'delta')
            np.savez_compressed(out_dir / 'samples.npz', **{k: np.array([r[k] for r in rows]) for k in keys},
                                phase=np.array([r['phase'] for r in rows]), mode=np.array([r['mode'] for r in rows]))
            res = [x for x in metrics(rows, m, d) if x['joint'] >= 1 and x['delta'] != 0.0]
            summary['steps'] = res
            if VISION_RECORDS:
                import precision_camera as PC
                summary['vision_records'] = VISION_RECORDS
                ok_recs = [r for r in VISION_RECORDS if r['T_cam_board'] is not None]
                cam = PC.camera_metrics(ok_recs, args.side)
                summary['camera'] = cam
                print(f'\ncamera ({len(ok_recs)} of {len(VISION_RECORDS)} holds saw the board): mm')
                print(f'{"mode":10s} {"j":>2s} {"step mrad":>9s} {"step mm":>7s} {"vis-target":>10s} {"vis-encoder":>11s} {"enc-target":>10s}')
                for x in cam:
                    if x['delta'] != 0.0:
                        print(f'{x["mode"]:10s} {x["joint"]:2d} {x["delta"] * 1e3:+9.0f} {x["step_mm"]:7.2f} {x["vis_vs_target_mm"]:10.2f} '
                              f'{x["vis_vs_encoders_mm"]:11.2f} {x["encoders_vs_target_mm"]:10.2f}')
                for mode in ('pd', 'precision'):
                    rr = [x['return_mm'] for x in cam if x['delta'] == 0.0 and x['mode'] == mode]
                    if rr:
                        print(f'{mode}: back at the test pose, vision {np.median(rr):.2f} mm median, {max(rr):.2f} max from the first hold (n={len(rr)})')
            (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
            print(f'\n{"mode":10s} {"j":>2s} {"step mrad":>9s} {"end err":>8s} {"settle s":>8s} {"p-p 1s":>7s} {"others":>7s} {"hand mm":>8s}')
            for x in res:
                st = '-' if x['settle_1mrad_s'] is None else f'{x["settle_1mrad_s"]:.2f}'
                print(f'{x["mode"]:10s} {x["joint"]:2d} {x["delta"] * 1e3:+9.0f} {x["end_err_mrad"]:+8.2f} {st:>8s} '
                      f'{x["pp_last1s_mrad"]:7.2f} {x["other_end_err_max_mrad"]:7.2f} {x["hand_err_mm"]:8.2f}')
            print(f'\nsaved {out_dir}')
        if arm.ctrl is not None:
            arm.ctrl.stop()
        if VISION is not None:
            VISION.close()
        if not at_start:
            print('The arm is NOT at its start pose and holds powered. Support it, then type SUPPORTED to let it go.', flush=True)
            while True:
                try:
                    if input('support confirmation > ').strip() == 'SUPPORTED':
                        break
                except (EOFError, KeyboardInterrupt):
                    time.sleep(0.5)
        arm.close()
        arm_o.close()                                      # never moved: limp again where it lay


if __name__ == '__main__':
    main()
