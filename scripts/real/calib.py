"""Wrist camera intrinsics + hand-eye, and ZED-to-base, for the real YAM rig (duchamp3).

Setup (as BerkeleyAutomation/mild-trackdlo yam_bimanual): one checkerboard flat on the table,
static for the whole session, seen by the ZED (fixed) and by each wrist camera as its arm moves.
  * wrist intrinsics: cv2.calibrateCamera over all views of that camera
  * T_link6_cam (camera optical frame in Menagerie link_6, the sim's frame): AX = XB from
    Park-Martin, refined by least squares on corner reprojection (OpenCV 5 has no
    calibrateHandEye)
  * T_base_zed = T_base_board @ inv(T_zed_board)
Arms move only through safety.SafeArm (hold-mode connect, table clearance, joint box); each
interpolated path is checked first, plus a lateral limit that keeps the arm off the other arm.
Run scripts/real/estop.py in your own terminal first: Enter there (or any safety stop or a tracking
error) holds the arm where it is until you clear it; then it returns slowly to rest.

    .venv/bin/python scripts/real/calib.py zed                       # ZED board pose (no motion)
    .venv/bin/python scripts/real/calib.py find --side right         # MOVES: search for the board
    .venv/bin/python scripts/real/calib.py collect --side right --n 50   # MOVES: diverse views
    .venv/bin/python scripts/real/calib.py intrinsics --side right --run data/calib/run_NNN
                                       # no motion: hold the board in front of the camera, edges too
    .venv/bin/python scripts/real/calib.py solve --side right        # offline
Captures go to data/calib/<run>/ (gitignored).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import camshare  # noqa: E402
import estop  # noqa: E402
import rig  # noqa: E402
import safety  # noqa: E402

BOARD = (8, 7)              # inner corners (cols, rows): 9 x 8 squares
SQUARE = 0.015              # m (ZED stereo measured 14.94 mm; mild-trackdlo config: 15 mm)
OBJ = np.zeros((BOARD[0] * BOARD[1], 3))
OBJ[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE
BOARD_CENTER = np.array([(BOARD[0] - 1) * SQUARE / 2, (BOARD[1] - 1) * SQUARE / 2, 0.0])
WRIST_RES = (1280, 1024)
MENAGERIE = REPO / "third_party/mujoco_menagerie/i2rt_yam/yam_linear_4310.xml"
INWARD = {"right": +1.0, "left": -1.0}   # base +y is to the arm's left: the board is inward
MAX_INWARD = 0.40           # m: no arm body further than this toward the other arm
# 2026-10-25: a run reaching ~0.58 m with unsecured bases nearly tipped the robots over. The bases
# must be clamped; reaches stay short: the board goes ~0.30 m straight in front of the arm.
MAX_REACH = 0.42            # m: horizontal distance of link_6 from the base axis
# Table clearance during calibration (m), in four levels with a real gap between each, so no
# check can be tripped by the one above it (2026-10-25: a path checked at the same level as the
# live stop tripped it, and a stopped arm sat under the level it needed to go home):
#   PLAN     choosing a pose
#   PATH     the travelled path from the measured pose to a planned pose
#   MONITOR  live stop on the measured pose and on every command
#   RECOVER  the way home after a stop (the operator is watching; the path rises to rest)
# The wrist webcam and its cable hang around the wrist and gripper and are not in the model, so
# those bodies get the larger value.
WRIST_BODIES = ("link4", "link5", "link6", "gripper", "tip_left", "tip_right")
PLAN_CLEARANCE = {"arm": 0.040, "wrist": 0.090}
PATH_CLEARANCE = {"arm": 0.035, "wrist": 0.075}
MONITOR_CLEARANCE = {"arm": 0.025, "wrist": 0.050}
RECOVER_CLEARANCE = {"arm": 0.015, "wrist": 0.020}


def calib_checker(c):
    return safety.Checker(c["arm"], c["wrist"], {b: c["wrist"] for b in WRIST_BODIES})
PATH_SLACK = 0.02           # m: reach/lateral margin for paths from the measured pose
# the ZED sees the board (or a base) moved -> stop and hold: two readings in a row over
# BOARD_SHIFT_STOP, or one over BOARD_SHIFT_JUMP. Single readings scatter 0.2-1 mm while the arm
# shadows the board and once read 2.07 mm with the board still in place (2026-10-25, run_004).
BOARD_SHIFT_STOP = 0.002
BOARD_SHIFT_JUMP = 0.005
MAX_VEL = 0.3               # rad/s joint speed cap for calibration moves
# rad: measured vs commanded beyond this -> blocked or collision -> hold and stop. i2rt holds the
# wrist far softer (kp 80, 80, 80, 40, 10, 10), so joints 5-6 trail their command more in a move
# (cable drag of ~1.5 N m alone is 0.15 rad at kp 10); a soft joint also cannot push hard.
TRACK_ABORT = np.array([0.15, 0.15, 0.15, 0.20, 0.30, 0.30])
DATA = REPO / "data/calib"


# ----------------------------------------------------------------------------- geometry
def T_from(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.ravel(t)
    return T


def T_inv(T):
    return T_from(T[:3, :3].T, -T[:3, :3].T @ T[:3, 3])


def T_from_rt(rvec, tvec):
    return T_from(cv2.Rodrigues(np.asarray(rvec, float))[0], tvec)


def rt_from_T(T):
    return cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3].copy()


def rot_between(a, b):
    """Smallest rotation taking unit vector a to b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else cv2.Rodrigues(np.pi * np.cross(a, [1, 0, 0]) / max(np.linalg.norm(np.cross(a, [1, 0, 0])), 1e-9))[0]
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


class Kin:
    """FK/IK of Menagerie link_6 (the sim's frame) on the YAM arm joints."""

    def __init__(self, path=MENAGERIE, body="link_6"):
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.b = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body)
        self.lo, self.hi = self.m.jnt_range[:6, 0], self.m.jnt_range[:6, 1]

    def fk(self, q6):
        self.d.qpos[:] = 0
        self.d.qpos[:6] = q6
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)   # mj_jacBody needs cdof from here
        return T_from(self.d.xmat[self.b].reshape(3, 3), self.d.xpos[self.b])

    def ik(self, T_goal, seeds, iters=300, tol_p=1e-3, tol_r=np.radians(0.5)):
        best = None
        for q0 in seeds:
            q = np.clip(np.array(q0, float), self.lo, self.hi)
            for _ in range(iters):
                T = self.fk(q)
                ep = T_goal[:3, 3] - T[:3, 3]
                er = cv2.Rodrigues(T_goal[:3, :3] @ T[:3, :3].T)[0].ravel()
                if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                    break
                jp, jr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
                mujoco.mj_jacBody(self.m, self.d, jp, jr, self.b)
                J = np.vstack([jp[:, :6], jr[:, :6]])
                e = np.r_[ep, er]
                q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), self.lo, self.hi)
            T = self.fk(q)
            err = (np.linalg.norm(T_goal[:3, 3] - T[:3, 3]),
                   np.linalg.norm(cv2.Rodrigues(T_goal[:3, :3] @ T[:3, :3].T)[0]))
            if err[0] < tol_p and err[1] < tol_r:
                cost = np.abs(q - np.asarray(seeds[0])).sum()
                if best is None or cost < best[1]:
                    best = (q, cost)
        return None if best is None else best[0]


# ----------------------------------------------------------------------------- cameras
def detect(gray):
    ok, c = cv2.findChessboardCornersSB(gray, BOARD, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    return c.reshape(-1, 2) if ok else None


class WristCam:
    def __init__(self, side):
        self.share = camshare.claim(f"wrist_{side}")   # the live viewer lets go of the camera
        self.cap = cv2.VideoCapture(rig.WRIST_CAM[side], cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WRIST_RES[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WRIST_RES[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        assert self.cap.isOpened(), f"{side} wrist camera did not open"

    def grab(self):
        for _ in range(6):   # drop buffered frames: the image must be from after the arm settled
            ok, f = self.cap.read()
        assert ok and f.shape[1::-1] == WRIST_RES, f"bad wrist frame {None if f is None else f.shape}"
        self.share.publish(f)
        return f

    def close(self):
        self.cap.release()
        self.share.release()


class Zed:
    """Rectified left image + its pinhole K (distortion already removed)."""

    def __init__(self):
        import pyzed.sl as sl
        self.sl = sl
        self.share = camshare.claim("zed")
        self.cam = sl.Camera()
        p = sl.InitParameters()
        p.camera_resolution = sl.RESOLUTION.HD1080
        p.depth_mode = sl.DEPTH_MODE.NONE
        assert self.cam.open(p) == sl.ERROR_CODE.SUCCESS, "ZED open failed (USB 3? another process?)"
        c = self.cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
        self.K = np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1.0]])
        self.mat = sl.Mat()
        self.ref = None   # T_zed_board of the run (board-shift watchdog)
        self.last_over = False

    def use_reference(self, run):
        """Load the run's T_zed_board and check the board is still there before any motion."""
        self.ref = np.array(json.loads((run / "zed.json").read_text())["T_zed_board"])
        for _ in range(5):
            c = detect(cv2.cvtColor(self.grab(), cv2.COLOR_BGR2GRAY))
            if c is not None:
                shift = np.linalg.norm(board_pose(c, self.K, None)[0][:3, 3] - self.ref[:3, 3])
                if shift > BOARD_SHIFT_STOP:
                    raise SystemExit(f"board is {shift * 1000:.1f} mm from this run's reference: "
                                     f"start a new run (calib.py zed)")
                return
        raise SystemExit("the ZED does not see the board: clear its view before starting")

    def grab(self):
        for _ in range(2):
            self.cam.grab()
        self.cam.retrieve_image(self.mat, self.sl.VIEW.LEFT)
        f = self.mat.get_data()[:, :, :3].copy()
        self.share.publish(f)
        return f

    def close(self):
        self.cam.close()
        self.share.release()


def board_pose(corners, K, dist):
    ok, rv, tv = cv2.solvePnP(OBJ, corners, K, dist, flags=cv2.SOLVEPNP_IPPE)
    pr, _ = cv2.projectPoints(OBJ, rv, tv, K, dist)
    return T_from_rt(rv, tv), float(np.linalg.norm(pr.reshape(-1, 2) - corners, axis=1).mean())


# ----------------------------------------------------------------------------- motion
class Mover:
    def __init__(self, side, kin, checker):
        self.side, self.kin, self.check = side, kin, checker   # plans: PLAN_CLEARANCE
        self.monitor = calib_checker(MONITOR_CLEARANCE)
        self.path_check = calib_checker(PATH_CLEARANCE)
        self.recover = calib_checker(RECOVER_CLEARANCE)
        self.robot = safety.connect(side)
        self.arm = safety.SafeArm(self.robot, self.monitor)    # commands + measured poses
        self.max_lag = np.zeros(6)

    def lateral_ok(self, q6, slack=0.0):
        """Toward-the-other-arm and reach limits; `slack` (m) loosens both for paths travelled
        from the measured pose (planning uses slack 0)."""
        c = self.check
        c.d.qpos[:] = 0
        c.d.qpos[:6] = q6
        mujoco.mj_kinematics(c.m, c.d)
        l6 = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, "link6")
        reach = float(np.linalg.norm(c.d.xpos[l6, :2]))
        return (float((c.d.xpos[2:, 1] * INWARD[self.side]).max()) <= MAX_INWARD + slack
                and reach <= MAX_REACH + slack)

    def path_ok(self, q_from, q_to, n=25, check=None):
        check = check or self.check
        for a in np.linspace(0, 1, n):
            q = (1 - a) * q_from + a * q_to
            why = check.violation(q)
            if why:
                return f"path at {a:.2f}: {why}"
            if not self.lateral_ok(q, slack=PATH_SLACK):
                return (f"path at {a:.2f}: arm more than {MAX_INWARD} m toward the other arm or "
                        f"link_6 beyond {MAX_REACH} m reach")
        return None

    def move_to(self, q6_goal, max_vel=MAX_VEL, dt=0.02, home=False):
        """The goal must clear the planning clearance (except home); the path from where the arm
        really is (a few mm under its planned pose) is checked at PATH_CLEARANCE, home at RECOVER."""
        q0 = self.arm.read()
        if not home and self.check.violation(q6_goal):
            raise safety.SafetyStop(f"goal unsafe: {self.check.violation(q6_goal)}")
        why = self.path_ok(q0[:6], q6_goal, check=self.recover if home else self.path_check)
        if why:
            raise safety.SafetyStop(f"path unsafe: {why}")
        steps = max(1, int(np.ceil(np.abs(q6_goal - q0[:6]).max() / (max_vel * dt) * 1.5)))  # smoothstep peak 1.5x
        self.max_lag = np.zeros(6)
        for i in range(1, steps + 1):
            if estop.stopped():
                self.hold()
                raise safety.SafetyStop("software stop (estop.py)")
            a = i / steps
            a = a * a * (3 - 2 * a)
            tgt = q0.copy()
            tgt[:6] = (1 - a) * q0[:6] + a * q6_goal
            self.arm.command(tgt)
            time.sleep(dt)
            lag = np.abs(self.arm.read()[:6] - tgt[:6])
            self.max_lag = np.maximum(self.max_lag, lag)
            if (lag > TRACK_ABORT).any():
                self.hold()
                j = int(np.argmax(lag / TRACK_ABORT))
                raise safety.SafetyStop(f"tracking error on joint {j + 1}: {lag[j]:.2f} rad (limit "
                                        f"{TRACK_ABORT[j]:.2f}), all {np.round(lag, 3).tolist()}: "
                                        f"blocked or collision; holding")
        time.sleep(0.6)   # settle

    def joints(self, n=10):
        return np.mean([self.arm.read() for _ in range(n) if not time.sleep(0.01)], axis=0)

    def hold(self):
        """PD hold at the measured pose (bypasses the checks: holding never moves the arm)."""
        self.arm._hold(np.asarray(self.robot.get_joint_pos(), float))   # also resets the step base

    def wait_clear(self, why):
        """Hold until the operator clears the stop in estop.py."""
        self.hold()
        estop.STOP_FILE.write_text("stop")
        print(f"\n!! STOPPED: {why}\n!! The arm holds its pose. Clear in estop.py (c + Enter) to send it "
              f"slowly back to rest.", flush=True)
        while estop.stopped():
            time.sleep(0.1)

    def go_home(self, q_rest6):
        """Back to rest; any stop on the way -> hold and wait again. Only then may close() run."""
        self.arm.check = self.recover          # live checks at the recovery level while homing
        try:
            while True:
                try:
                    self.move_to(q_rest6, max_vel=0.2, home=True)
                    return
                except safety.SafetyStop as e:
                    self.wait_clear(e)
        finally:
            self.arm.check = self.monitor

    def close(self):
        safety.close_robot(self.robot)


def supervised(side, kin, checker, body):
    """Connect, run body(mover, q_rest), and always end at rest before closing. A stop anywhere
    (estop.py, a safety check, a tracking error) holds the arm until the operator clears it."""
    if estop.stopped():
        raise SystemExit(f"{estop.STOP_FILE} is set: clear it in estop.py first")
    mover = Mover(side, kin, checker)
    q_rest = mover.joints()
    try:
        body(mover, q_rest)
    except (safety.SafetyStop, KeyboardInterrupt) as e:
        mover.wait_clear(e if str(e) else "interrupted")
    finally:
        print("returning to rest", flush=True)
        mover.go_home(q_rest[:6])
        mover.close()


# ----------------------------------------------------------------------------- planning
def cam_mount_guess(kin):
    """Placeholder T_link6_cam (OpenCV) from the sim model's D405 (used until solved)."""
    from blind_insertion.assets.real_camera import wrist_pose_in_link6
    p = wrist_pose_in_link6(kin.m)
    q = np.array(p["quat"])
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return T_from(R.reshape(3, 3), p["pos"])


def cam_prior(kin, side):
    """(T_link6_cam, K, dist) to bootstrap with: this side's calibration if the spec has it, else
    the other side's (same webcam model and bracket), else the model's D405 and a 90 deg guess."""
    spec = json.loads((REPO / "blind_insertion/configs/real_cameras_duchamp3.json").read_text())["wrist"]
    for s_ in (side, "left" if side == "right" else "right"):
        w = spec.get(s_)
        if w and w.get("pose_in_link6") and w.get("intrinsics"):
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, np.array(w["pose_in_link6"]["quat"], float))
            i = w["intrinsics"]
            K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
            return T_from(R.reshape(3, 3), w["pose_in_link6"]["pos"]), K, np.array(w["distortion"], float)
    return cam_mount_guess(kin), K_guess_from_fov(90.0), None


def on_table(T_base_board):
    """The board lies flat on the table: keep the estimate's horizontal position and yaw only."""
    c = T_base_board[:3, :3] @ BOARD_CENTER + T_base_board[:3, 3]
    x = T_base_board[:3, 0].copy()
    x[2] = 0
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross([0, 0, 1.0], x), [0, 0, 1.0]])
    return T_from(R, np.r_[c[:2], 0.0] - R @ BOARD_CENTER)


def look_at(p_cam, p_target, R_ref, roll=0.0):
    """Camera (OpenCV) at p_cam with +z toward p_target; roll kept closest to R_ref, plus roll."""
    z = p_target - p_cam
    R = rot_between(R_ref[:, 2], z) @ R_ref
    return T_from(R @ cv2.Rodrigues(np.array([0, 0, roll]))[0], p_cam)


def plan_views(T_base_board, R_ref, side, n, rng, d_range=(0.22, 0.32), max_tilt=40):
    """n camera poses around the board: varied distance, tilt from the board normal, azimuth on
    the arm's side, aim point across the board, and roll."""
    c = T_base_board[:3, :3] @ BOARD_CENTER + T_base_board[:3, 3]
    nrm = T_base_board[:3, :3][:, 2]
    if nrm[2] < 0:
        nrm = -nrm
    toward_base = -c.copy()
    toward_base[2] = 0
    toward_base /= np.linalg.norm(toward_base)
    out = []
    for _ in range(n):
        tilt = np.radians(rng.uniform(10, max_tilt))
        az = np.radians(rng.uniform(-70, 70))
        h = cv2.Rodrigues(nrm * az)[0] @ toward_base
        dirv = np.cos(tilt) * nrm + np.sin(tilt) * h
        d = rng.uniform(*d_range)
        aim = c + T_base_board[:3, :3] @ np.r_[rng.uniform(-0.04, 0.04, 2), 0]
        out.append(look_at(aim + d * dirv, aim, R_ref, roll=np.radians(rng.uniform(-30, 30))))
    return out


# ----------------------------------------------------------------------------- solving
def park_martin(Fs, Cs):
    """X = T_link6_cam from Y = F_i X C_i for all i (F: T_base_link6, C: T_cam_board)."""
    M = np.zeros((3, 3))
    rows, rhs = [], []
    for i in range(len(Fs)):
        for j in range(i + 1, len(Fs)):
            A = T_inv(Fs[j]) @ Fs[i]
            B = Cs[j] @ T_inv(Cs[i])
            a, b = cv2.Rodrigues(A[:3, :3])[0].ravel(), cv2.Rodrigues(B[:3, :3])[0].ravel()
            M += np.outer(b, a)
            rows.append((A, B))
    w, V = np.linalg.eigh(M.T @ M)
    Rx = V @ np.diag(w ** -0.5) @ V.T @ M.T
    for A, B in rows:
        rhs.append((Rx @ B[:3, 3] - A[:3, 3], A[:3, :3] - np.eye(3)))
    tx = np.linalg.lstsq(np.vstack([r[1] for r in rhs]), np.hstack([r[0] for r in rhs]), rcond=None)[0]
    return T_from(Rx, tx)


def reproj_residual(X, Y, Fs, corners, K, dist):
    out = []
    for F, c in zip(Fs, corners):
        C = T_inv(X) @ T_inv(F) @ Y
        rv, tv = rt_from_T(C)
        pr, _ = cv2.projectPoints(OBJ, rv, tv, K, dist)
        out.append(pr.reshape(-1, 2) - c)
    return np.concatenate(out)


def refine(X, Y, Fs, corners, K, dist):
    from scipy.optimize import least_squares

    def unpack(x):
        return T_from_rt(x[:3], x[3:6]), T_from_rt(x[6:9], x[9:12])

    x0 = np.r_[np.hstack(rt_from_T(X)), np.hstack(rt_from_T(Y))]
    r = least_squares(lambda x: reproj_residual(*unpack(x), Fs, corners, K, dist).ravel(), x0,
                      loss="huber", f_scale=2.0)
    return unpack(r.x)


OFFSET_JOINTS = [1, 2, 3, 4]   # joints 2-5; a joint 1 (base yaw) offset is the same as turning the
                               # board, a joint 6 offset the same as turning the camera mount


def fit_offsets(kin, Q, corners, K, dist, X, Y, idx):
    """Refine X, Y and encoder zero offsets of OFFSET_JOINTS on views idx. Returns X, Y, dq (6)."""
    from scipy.optimize import least_squares

    def unpack(x):
        dq = np.zeros(6)
        dq[OFFSET_JOINTS] = x[12:]
        return T_from_rt(x[:3], x[3:6]), T_from_rt(x[6:9], x[9:12]), dq

    def resid(x):
        Xx, Yy, dq = unpack(x)
        return reproj_residual(Xx, Yy, [kin.fk(Q[i] + dq) for i in idx], [corners[i] for i in idx], K, dist).ravel()

    x0 = np.r_[np.hstack(rt_from_T(X)), np.hstack(rt_from_T(Y)), np.zeros(len(OFFSET_JOINTS))]
    return unpack(least_squares(resid, x0, loss="huber", f_scale=2.0).x)


def solve_side(run, side, kin_paths):
    S = [json.loads(p.read_text()) | {"_name": p.stem} for p in sorted((run / side).glob("[0-9]*.json"))]
    S = [s for s in S if s["wrist_corners"] is not None]
    corners = [np.array(s["wrist_corners"], np.float32) for s in S]
    n = len(S)
    # hand-held frames (calib.py intrinsics) cover the image edges; the arm's views only its middle
    intr = [np.array(json.loads(p.read_text())["corners"], np.float32)
            for p in sorted((run / f"{side}_intr").glob("[0-9]*.json"))]
    allc = corners + intr
    # standard 5-coefficient lens model: the rational model fits the right lens 0.15 px better
    # but its large, cancelling terms can extrapolate badly where no corners were seen (the
    # bottom of the image, behind the finger pads, which the policy crop partly includes)
    rms, K, dist, _, _ = cv2.calibrateCamera([OBJ.astype(np.float32)] * len(allc), allc, WRIST_RES, None, None)
    Cs = [board_pose(c, K, dist)[0] for c in corners]
    pts = np.concatenate(allc)
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[8, 8], range=[[0, WRIST_RES[0]], [0, WRIST_RES[1]]])
    res = {"n_views": n, "n_intrinsic_frames": len(intr), "image_cells_covered": int((H > 0).sum()),
           "intrinsics_rms_px": float(rms), "K": K.tolist(), "dist": dist.ravel().tolist()}
    for name, path, body in kin_paths:
        kin = Kin(path, body)
        Fs = [kin.fk(np.array(s["q"][:6])) for s in S]
        X0 = park_martin(Fs, Cs)
        Y0 = np.mean([F @ X0 @ C for F, C in zip(Fs, Cs)], axis=0)
        U, _, Vt = np.linalg.svd(Y0[:3, :3])
        Y0[:3, :3] = U @ Vt
        X, Y = refine(X0, Y0, Fs, corners, K, dist)
        # drop views far worse than the rest (image and joints not simultaneous, arm settling),
        # then refit on the kept views
        per_view = np.linalg.norm(reproj_residual(X, Y, Fs, corners, K, dist), axis=1).reshape(n, -1).mean(1)
        keep = per_view <= max(3 * np.median(per_view), 15.0)
        Fs_k, c_k = [F for F, k in zip(Fs, keep) if k], [c for c, k in zip(corners, keep) if k]
        X, Y = refine(X, Y, Fs_k, c_k, K, dist)
        dropped = [S[i]["_name"] for i in np.flatnonzero(~keep)]
        Fs, corners_k, nk = Fs_k, c_k, int(keep.sum())
        e = np.linalg.norm(reproj_residual(X, Y, Fs, corners_k, K, dist), axis=1)
        # k-fold hold-out: refit X, Y without the fold, measure on the fold
        idx = np.random.default_rng(0).permutation(nk)
        ho = []
        for f in range(5):
            te = idx[f::5]
            tr = np.setdiff1d(idx, te)
            Xf, Yf = refine(X, Y, [Fs[i] for i in tr], [corners_k[i] for i in tr], K, dist)
            ho.append(np.linalg.norm(reproj_residual(Xf, Yf, [Fs[i] for i in te], [corners_k[i] for i in te], K, dist), axis=1))
        res[name] = {"T_link6_cam": X.tolist(), "T_base_board": Y.tolist(), "n_used": nk, "dropped": dropped,
                     "reproj_mean_px": float(e.mean()), "reproj_p95_px": float(np.percentile(e, 95)),
                     "holdout_mean_px": float(np.concatenate(ho).mean())}
        # encoder zero offsets of joints 2-5, fitted with X and Y on all views (the offsets explain
        # what looked like outliers on the right arm: 10.3 -> 2.7 px hold-out, 2026-10-25)
        Q = np.array([s["q"][:6] for s in S])
        Xo, Yo, dq = fit_offsets(kin, Q, corners, K, dist, X, Y, np.arange(n))
        idx = np.random.default_rng(0).permutation(n)
        ho = []
        for f in range(5):
            te = idx[f::5]
            Xf, Yf, dqf = fit_offsets(kin, Q, corners, K, dist, Xo, Yo, np.setdiff1d(idx, te))
            ho.append(np.linalg.norm(reproj_residual(Xf, Yf, [kin.fk(Q[i] + dqf) for i in te],
                                                     [corners[i] for i in te], K, dist), axis=1))
        res[name]["with_joint_offsets"] = {
            "joint_offsets_rad": dq.tolist(), "T_link6_cam": Xo.tolist(), "T_base_board": Yo.tolist(),
            "holdout_mean_px": float(np.concatenate(ho).mean()), "n_used": n}
    return res


# ----------------------------------------------------------------------------- commands
def new_run():
    DATA.mkdir(parents=True, exist_ok=True)
    k = 1 + max([int(p.name.split("_")[1]) for p in DATA.glob("run_*")] or [0])
    return DATA / f"run_{k:03d}"


def capture(run, side, idx, mover, wcam, zed, K_guess):
    q = mover.joints()
    img = wcam.grab()
    c = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    rec = {"q": q.tolist(), "t": time.time(), "wrist_corners": None if c is None else c.tolist(),
           "zed_corners": None}
    shift = None
    if zed is not None:
        zi = zed.grab()
        zc = detect(cv2.cvtColor(zi, cv2.COLOR_BGR2GRAY))
        rec["zed_corners"] = None if zc is None else zc.tolist()
        if zc is not None and zed.ref is not None:   # the arm often hides the board from the ZED
            shift = float(np.linalg.norm(board_pose(zc, zed.K, None)[0][:3, 3] - zed.ref[:3, 3]))
            rec["zed_board_shift_m"] = shift
    d = run / side
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"{idx:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (d / f"{idx:03d}.json").write_text(json.dumps(rec))
    if shift is not None and zed is not None:
        over = shift > BOARD_SHIFT_STOP
        if shift > BOARD_SHIFT_JUMP or (over and zed.last_over):
            raise safety.SafetyStop(f"the ZED sees the board {shift * 1000:.1f} mm from its reference: "
                                    f"the board or a base moved")
        zed.last_over = over
    return c


def K_guess_from_fov(hfov_deg=70.0):
    f = WRIST_RES[0] / 2 / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, WRIST_RES[0] / 2], [0, f, WRIST_RES[1] / 2], [0, 0, 1.0]])


def cmd_intrinsics(a):
    """Hand-held intrinsic capture, no arm motion: hold the board 20-40 cm in front of the wrist
    camera and move it into every corner and along every edge, tilted. Frames are kept when the
    board is found and has moved since the last kept frame; stops at full coverage or --seconds."""
    run = Path(a.run)
    out = run / f"{a.side}_intr"
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(rig.WRIST_CAM[a.side], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WRIST_RES[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WRIST_RES[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cover = np.zeros((8, 8), int)
    last, kept, t0 = None, len(list(out.glob("[0-9]*.json"))), time.time()
    print(f"{a.side} wrist camera: move the board into every corner and edge of the view, tilted.\n"
          f"Grid = the camera image (8 x 8): '#' has board corners, '.' still empty. Ctrl+C to stop.", flush=True)
    try:
        while time.time() - t0 < a.seconds:
            ok, f = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(g, None, fx=0.5, fy=0.5)
            ok, _ = cv2.findChessboardCornersSB(small, BOARD)        # fast check, then full resolution
            c = detect(g) if ok else None
            if c is None:
                continue
            if last is not None and np.linalg.norm(c - last, axis=1).mean() < 40:
                continue                                              # board has not moved enough
            cells = np.clip((c / np.array(WRIST_RES) * 8).astype(int), 0, 7)
            new = sum(cover[x, y] == 0 for x, y in set(map(tuple, cells)))
            for x, y in set(map(tuple, cells)):
                cover[x, y] += 1
            cv2.imwrite(str(out / f"{kept:03d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / f"{kept:03d}.json").write_text(json.dumps({"corners": c.tolist(), "t": time.time()}))
            kept, last = kept + 1, c
            grid = "\n".join("   " + " ".join("#" if cover[x, y] else "." for x in range(8)) for y in range(8))
            print(f"\nframe {kept}: +{new} new cells, {int((cover > 0).sum())}/64 covered\n{grid}", flush=True)
            if (cover > 0).sum() >= a.cells and kept >= a.min_frames:
                print("\ncoverage reached: done", flush=True)
                break
    except KeyboardInterrupt:
        pass
    cap.release()
    print(f"kept {kept} frames in {out}", flush=True)


def cmd_zed(a):
    zed = Zed()
    poses = []
    for _ in range(10):
        c = detect(cv2.cvtColor(zed.grab(), cv2.COLOR_BGR2GRAY))
        if c is not None:
            poses.append(board_pose(c, zed.K, None))
    zed.close()
    T = poses[len(poses) // 2][0]
    print(f"T_zed_board t = {T[:3, 3].round(4)} m, reproj {np.mean([p[1] for p in poses]):.2f} px, n={len(poses)}")
    out = Path(a.run) if a.run else new_run()
    out.mkdir(parents=True, exist_ok=True)
    (out / "zed.json").write_text(json.dumps({"T_zed_board": T.tolist(), "K": zed.K.tolist()}))
    print("saved", out / "zed.json")


def search_poses(side, kin, R_ref, hint=None):
    """Camera poses looking down at candidate board centres ~0.3 m in front of the arm, most
    likely first (closest to `hint` = (x, y) in the base frame, when given)."""
    s = INWARD[side]
    cands = [(x, s * y) for y in (0.0, 0.05, -0.05, 0.10, 0.15, 0.20, 0.25) for x in (0.30, 0.25, 0.35)]
    if hint is not None:
        cands = [tuple(hint)] + sorted(cands, key=lambda c: np.hypot(c[0] - hint[0], c[1] - hint[1]))
    out = []
    for x, y in cands:
        c = np.array([x, y, 0.0])
        toward_base = -c / np.linalg.norm(c)
        p = c + 0.26 * (np.cos(np.radians(40)) * np.array([0, 0, 1.0]) + np.sin(np.radians(40)) * toward_base)
        out.append(((x, y), look_at(p, c, R_ref)))
    return out


def cmd_find(a):
    run = Path(a.run) if a.run else new_run()
    kin, checker = Kin(), calib_checker(PLAN_CLEARANCE)
    X = cam_mount_guess(kin)
    zed = Zed() if a.zed else None      # open the ZED before the arm (its open stalls Python ~2 s)
    if zed:
        zed.use_reference(run)
    wcam = WristCam(a.side)

    def body(mover, q_rest):
        R_ref = (kin.fk(q_rest[:6]) @ X)[:3, :3]
        for k, ((x, y), T_cam) in enumerate(search_poses(a.side, kin, R_ref, a.hint)[:a.max_tries]):
            q = kin.ik(T_cam @ T_inv(X), seeds=[q_rest[:6], [0, 1.0, 1.0, -0.5, 0, 0], [0, 1.6, 1.3, -0.9, 0, 0]])
            if q is None or checker.violation(q) or not mover.lateral_ok(q):
                print(f"candidate {k} ({x:.2f}, {y:+.2f}): no safe IK, skipped")
                continue
            if mover.path_ok(mover.arm.read()[:6], q, check=mover.path_check):
                print(f"candidate {k} ({x:.2f}, {y:+.2f}): path not clear, skipped")
                continue
            print(f"candidate {k}: looking at ({x:.2f}, {y:+.2f}) -> q {q.round(2)}", flush=True)
            mover.move_to(q)
            c = capture(run, a.side, k, mover, wcam, zed, None)
            print(f"  board {'FOUND' if c is not None else 'not seen'}", flush=True)
            if c is not None:
                (run / a.side / "found.json").write_text(json.dumps({"q": mover.joints().tolist(), "idx": k}))
                break

    try:
        supervised(a.side, kin, checker, body)
    finally:
        wcam.close()
        if zed:
            zed.close()
    print("run dir:", run)


def cmd_collect(a):
    run = Path(a.run)
    kin, checker = Kin(), calib_checker(PLAN_CLEARANCE)
    X, K, dist = cam_prior(kin, a.side)
    found = json.loads((run / a.side / "found.json").read_text())
    rec = json.loads((run / a.side / f"{found['idx']:03d}.json").read_text())
    T_cam_board, _ = board_pose(np.array(rec["wrist_corners"]), K, dist)
    T_base_board = on_table(kin.fk(np.array(rec["q"][:6])) @ X @ T_cam_board)
    print("board centre (base frame, rough):", (T_base_board[:3, :3] @ BOARD_CENTER + T_base_board[:3, 3]).round(3))
    rng = np.random.default_rng(a.seed)
    zed = Zed() if a.zed else None
    if zed:
        zed.use_reference(run)
    wcam = WristCam(a.side)

    def body(mover, q_rest):
        R_ref = (kin.fk(np.array(found["q"][:6])) @ X)[:3, :3]
        views = plan_views(T_base_board, R_ref, a.side, a.n * 4, rng)
        idx0 = 1 + max(int(p.stem) for p in (run / a.side).glob("[0-9]*.json"))
        got = tried = 0
        q_prev = np.array(found["q"][:6])
        for T_cam in views:
            if got >= a.n:
                break
            q = kin.ik(T_cam @ T_inv(X), seeds=[q_prev, found["q"][:6]])
            if q is None or checker.violation(q) or not mover.lateral_ok(q):
                continue
            if mover.path_ok(mover.arm.read()[:6], q, check=mover.path_check):
                continue   # skip the view rather than stop the run
            tried += 1
            mover.move_to(q)
            c = capture(run, a.side, idx0 + tried, mover, wcam, zed, K)
            got += c is not None
            q_prev = q
            print(f"view {tried}: board {'seen' if c is not None else 'NOT seen'} ({got}/{a.n}); "
                  f"max lag per joint {np.round(mover.max_lag, 3).tolist()}", flush=True)

    try:
        supervised(a.side, kin, checker, body)
    finally:
        wcam.close()
        if zed:
            zed.close()


def cmd_solve(a):
    run = Path(a.run)
    i2rt = safety.Checker()  # noqa: F841  (builds i2rt's xml; path below)
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
    i2rt_xml = combine_arm_and_gripper_xml(ArmType.from_string_name("yam"),
                                           GripperType.from_string_name("linear_4310"), ee_mass=safety.EE_MASS)
    res = solve_side(run, a.side, [("menagerie", MENAGERIE, "link_6"), ("i2rt", i2rt_xml, "link6")])
    zed_p = run / "zed.json"
    if zed_p.exists():
        Tzb = np.array(json.loads(zed_p.read_text())["T_zed_board"])
        for d in (res["menagerie"], res["menagerie"]["with_joint_offsets"]):
            d["T_base_zed"] = (np.array(d["T_base_board"]) @ T_inv(Tzb)).tolist()
    (run / f"solve_{a.side}.json").write_text(json.dumps(res, indent=1))
    for k in ("menagerie", "i2rt"):
        w = res[k]["with_joint_offsets"]
        print(f"{k}: with joint offsets (deg) {np.degrees(w['joint_offsets_rad']).round(2).tolist()} "
              f"-> hold-out {w['holdout_mean_px']:.1f} px")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if "px" in kk or kk in ("n_used", "dropped")})
                      for k, v in res.items() if k not in ("K", "dist")}, indent=1))
    print("K", np.array(res["K"]).round(1).tolist(), "dist", np.round(res["dist"], 4).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("zed", "find", "collect", "solve", "intrinsics"):
        p = sub.add_parser(name)
        p.add_argument("--run", default="")
        if name != "zed":
            p.add_argument("--side", required=True, choices=("left", "right"))
        if name == "intrinsics":
            p.add_argument("--seconds", type=float, default=240)
            p.add_argument("--cells", type=int, default=52, help="image cells (of 64) to cover")
            p.add_argument("--min_frames", type=int, default=30)
        if name in ("find", "collect"):
            p.add_argument("--zed", action="store_true", help="also grab the ZED at every capture")
        if name == "find":
            p.add_argument("--max_tries", type=int, default=20, help="search poses to visit at most")
            p.add_argument("--hint", type=float, nargs=2, default=None, help="rough board centre x y (m, base frame)")
        if name == "collect":
            p.add_argument("--n", type=int, default=40)
            p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"zed": cmd_zed, "find": cmd_find, "collect": cmd_collect, "solve": cmd_solve,
     "intrinsics": cmd_intrinsics}[a.cmd](a)
