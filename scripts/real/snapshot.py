"""Reference images of the real scene for the sim: the ZED and both wrist cameras.

    .venv/bin/python scripts/real/snapshot.py capture --out data/sim_ref/<name> --tag before
    .venv/bin/python scripts/real/snapshot.py point --out data/sim_ref/<name> --target 0.30 0.30 [--tag pointed]

`capture` takes the images without moving anything. `point` turns both wrist cameras toward one
table point (x, y in the right arm's base frame, on the table plane), takes the images, then sends
both arms back to rest. Run scripts/real/estop.py first: Enter holds both arms.

Per tag: zed_left.png / zed_right.png (rectified HD1080, a stereo pair), wrist_<side>.png (raw 1280x1024) and wrist_<side>_policy.png (undistorted policy
crop), plus <tag>.json with the joint readings, camera poses and the camera spec used.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import calib  # noqa: E402
import camshare  # noqa: E402
import estop  # noqa: E402
import rig  # noqa: E402
import safety  # noqa: E402
from blind_insertion.assets.real_camera import load_spec, real_policy_image  # noqa: E402

SIDES = ("right", "left")


def T_pose(p):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(p["quat"], float))
    return calib.T_from(R.reshape(3, 3), np.asarray(p["pos"], float))


def grab_zed(out, tag):
    import pyzed.sl as sl
    share = camshare.claim("zed")
    cam = sl.Camera()
    p = sl.InitParameters()
    p.camera_resolution = sl.RESOLUTION.HD1080
    # no depth: the SDK's neural depth model is not installed on duchamp3 ("CORRUPTED SDK
    # INSTALLATION"), and reopening after a failed open segfaults. The stereo pair is saved.
    p.depth_mode = sl.DEPTH_MODE.NONE
    assert cam.open(p) == sl.ERROR_CODE.SUCCESS, "ZED open failed (USB 3? another process?)"
    try:
        img = sl.Mat()
        for _ in range(15):   # let auto exposure and the depth network settle
            cam.grab()
        cam.retrieve_image(img, sl.VIEW.LEFT)
        share.publish(img.get_data()[:, :, :3])
        cv2.imwrite(str(out / f"{tag}_zed_left.png"), img.get_data()[:, :, :3])
        cam.retrieve_image(img, sl.VIEW.RIGHT)
        cv2.imwrite(str(out / f"{tag}_zed_right.png"), img.get_data()[:, :, :3])
        print("ZED: saved", flush=True)
    finally:
        cam.close()
        share.release()


def grab_wrists(out, tag, spec):
    for side in SIDES:
        w = calib.WristCam(side)
        try:
            f = w.grab()
        finally:
            w.close()
        cv2.imwrite(str(out / f"{tag}_wrist_{side}.png"), f)
        cv2.imwrite(str(out / f"{tag}_wrist_{side}_policy.png"),
                    cv2.cvtColor(real_policy_image(f, spec, 480, side), cv2.COLOR_RGB2BGR))
        print(f"{side} wrist: saved", flush=True)


def write_meta(out, tag, spec, q=None, extra=None):
    meta = {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "camera_spec": str(rig.SPEC.relative_to(HERE.parents[1])),
            "frames": "right base frame = the right arm's base, table plane z = 0; poses OpenCV camera "
                      "frames {pos m, quat wxyz}; joints as measured (add spec arms offsets for the model)",
            "zed_in_right_base": spec["top"]["extrinsics_cam_in_right_base"],
            "left_base_in_right_base": spec["arms"]["left_base_in_right_base"]}
    if q is not None:
        meta["joints_measured"] = {s: v.tolist() for s, v in q.items()}
    meta.update(extra or {})
    (out / f"{tag}.json").write_text(json.dumps(meta, indent=1))


def cmd_capture(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(str(rig.SPEC))
    grab_zed(out, a.tag)
    grab_wrists(out, a.tag, spec)
    write_meta(out, a.tag, spec)


def cmd_point(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = load_spec(str(rig.SPEC))
    kin, checker = calib.Kin(), calib.calib_checker(calib.PLAN_CLEARANCE)
    T_r_l = T_pose(spec["arms"]["left_base_in_right_base"])
    tgt_r = np.array([a.target[0], a.target[1], 0.0])
    goals = {}
    for side in SIDES:   # plan both before anything moves
        tgt = tgt_r if side == "right" else (calib.T_inv(T_r_l) @ np.append(tgt_r, 1))[:3]
        X = T_pose(spec["wrist"][side]["pose_in_link6"])
        goals[side] = (tgt, X)
    if estop.stopped():
        raise SystemExit(f"{estop.STOP_FILE} is set: clear it in estop.py first")

    movers, rest = {}, {}
    try:
        for side in SIDES:
            movers[side] = calib.Mover(side, kin, checker)
            rest[side] = movers[side].joints()
        plans = {}
        for side in SIDES:
            tgt, X = goals[side]
            off = rig.joint_offsets(side)
            R_ref = (kin.fk(rest[side][:6] + off) @ X)[:3, :3]
            q = None
            for dist in (0.32, 0.36, 0.28):
                for elev in (45, 55, 35):
                    toward = -tgt[:2] / np.linalg.norm(tgt[:2])
                    e = np.radians(elev)
                    p = tgt + dist * np.array([np.cos(e) * toward[0], np.cos(e) * toward[1], np.sin(e)])
                    T_cam = calib.look_at(p, tgt, R_ref)
                    qm = kin.ik(T_cam @ calib.T_inv(X), seeds=[rest[side][:6] + off, [0, 1.0, 1.0, -0.5, 0, 0],
                                                            [0, 1.6, 1.3, -0.9, 0, 0]])
                    if qm is None:
                        continue
                    qq = qm - off   # model joints -> what the motors read
                    m = movers[side]
                    if checker.violation(qq) or not m.lateral_ok(qq):
                        continue
                    if m.path_ok(m.arm.read()[:6], qq, check=m.path_check):
                        continue
                    q = qq
                    print(f"{side}: camera {dist:.2f} m from the target at {elev} deg -> q {q.round(2)}", flush=True)
                    break
                if q is not None:
                    break
            if q is None:
                raise SystemExit(f"{side}: no safe pose looks at {tgt.round(3)}; nothing moved")
            plans[side] = q
        for side in SIDES:
            print(f"moving {side}", flush=True)
            movers[side].move_to(plans[side])
        q_now = {s: movers[s].joints() for s in SIDES}
        cams = {}
        for s in SIDES:
            T = kin.fk(q_now[s][:6] + rig.joint_offsets(s)) @ goals[s][1]
            if s == "left":
                T = T_r_l @ T
            import mujoco
            qq = np.zeros(4)
            mujoco.mju_mat2Quat(qq, T[:3, :3].flatten())
            cams[s] = {"pos": T[:3, 3].round(5).tolist(), "quat": qq.round(5).tolist()}
        grab_zed(out, a.tag)
        grab_wrists(out, a.tag, spec)
        write_meta(out, a.tag, spec, q_now, {"target_in_right_base": tgt_r.tolist(),
                                             "wrist_cam_in_right_base": cams})
    except (safety.SafetyStop, KeyboardInterrupt) as e:
        for m in movers.values():
            m.hold()
        if movers:   # both hold; the operator clears the stop, then both go back to rest
            next(iter(movers.values())).wait_clear(e if str(e) else "interrupted")
    finally:
        for side, m in movers.items():
            print(f"returning {side} to rest", flush=True)
            m.go_home(rest[side][:6])
        for m in movers.values():
            m.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--out", required=True)
    c.add_argument("--tag", default="before")
    c = sub.add_parser("point")
    c.add_argument("--out", required=True)
    c.add_argument("--target", type=float, nargs=2, required=True, help="x y (m) in the right base frame")
    c.add_argument("--tag", default="pointed")
    a = ap.parse_args()
    {"capture": cmd_capture, "point": cmd_point}[a.cmd](a)
