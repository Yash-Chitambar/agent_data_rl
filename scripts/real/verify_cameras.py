"""Check that every camera and arm pose in the spec agrees: all three cameras look at one board.

    .venv/bin/python scripts/real/verify_cameras.py [--out data/calib/verify_<time>]

Put the checkerboard (calib.py's 8 x 7 inner corners, 15 mm) flat on the table between the arms,
where the ZED sees it. Run estop.py (or rigctl.py) first: Enter / space holds both arms.

1. The ZED finds the board: its corners in the right base frame through the spec's
   top.extrinsics_cam_in_right_base.
2. Each arm points its wrist camera at the board from two elevations and finds the corners: base
   frame = FK(q + joint offsets) @ wrist pose_in_link6 @ board pose from the image (intrinsics and
   distortion from the spec). Left arm results go through arms.left_base_in_right_base.
3. Compared in the right base frame: corner distances between every pair of views (mm), and the
   ZED's corners projected into each wrist image (px). Nothing from this run is fitted.

Pass: every view within 4 mm (mean corner distance) of the ZED, and ZED corners within 6 px of the
wrist detections (calibration hold-out: 2.2-2.7 px). Images and numbers go to --out; the summary
also to results/real/camera_verify.json.
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
import calib  # noqa: E402
import estop  # noqa: E402
import rig  # noqa: E402
import safety  # noqa: E402
from snapshot import T_pose  # noqa: E402

REPO = HERE.parents[1]
SIDES = ("right", "left")
PASS_MM, PASS_PX = 4.0, 6.0
ELEVATIONS = (55, 40)   # deg, two views per arm


def corners_in(T, obj=calib.OBJ):
    return (T[:3, :3] @ obj.T).T + T[:3, 3]


def match(a, b):
    """Mean / max distance between two corner sets; a chessboard's corner order can come out
    rotated 180 deg, so the better of the two orders is used."""
    d = [np.linalg.norm(a - b, axis=1), np.linalg.norm(a - b[::-1], axis=1)]
    d = min(d, key=lambda x: x.mean())
    return float(d.mean()), float(d.max())


def wrist_K(spec, side):
    w = spec["wrist"][side]
    i = w["intrinsics"]
    return np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]]), np.array(w["distortion"])


def plan_view(side, mover, kin, checker, tgt, X, elev, rest):
    off = rig.joint_offsets(side)
    R_ref = (kin.fk(rest[:6] + off) @ X)[:3, :3]
    for dist in (0.26, 0.30, 0.22, 0.34):
        for de in (0, 8, -8):
            toward = -tgt[:2] / np.linalg.norm(tgt[:2])
            e = np.radians(elev + de)
            p = tgt + dist * np.array([np.cos(e) * toward[0], np.cos(e) * toward[1], np.sin(e)])
            qm = kin.ik(calib.look_at(p, tgt, R_ref) @ calib.T_inv(X),
                        seeds=[rest[:6] + off, [0, 1.0, 1.0, -0.5, 0, 0], [0, 1.6, 1.3, -0.9, 0, 0]])
            if qm is None:
                continue
            q = qm - off
            if checker.violation(q) or not mover.lateral_ok(q):
                continue
            if mover.path_ok(mover.arm.read()[:6], q, check=mover.path_check):
                continue
            return q, dist, elev + de
    return None, None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"data/calib/verify_{time.strftime('%Y%m%d_%H%M%S')}")
    a = ap.parse_args()
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(rig.SPEC))
    T_rb_zed = T_pose(spec["top"]["extrinsics_cam_in_right_base"])
    T_r_l = T_pose(spec["arms"]["left_base_in_right_base"])
    kin, checker = calib.Kin(), calib.calib_checker(calib.PLAN_CLEARANCE)
    if estop.stopped():
        raise SystemExit(f"{estop.STOP_FILE} is set: clear it first")

    # 1. ZED
    zed = calib.Zed()
    try:
        img = zed.grab()
        c = calib.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if c is None:
            raise SystemExit("the ZED does not see the board: put it on the table between the arms")
        T_zb, rms = calib.board_pose(c, zed.K, None)
        cv2.imwrite(str(out / "zed_raw.png"), img)
        cv2.drawChessboardCorners(img, calib.BOARD, c.astype(np.float32), True)
        cv2.imwrite(str(out / "zed.png"), img)
    finally:
        zed.close()
    P_zed = corners_in(T_rb_zed @ T_zb)
    center = P_zed.mean(0)
    print(f"ZED: board centre in the right base {center.round(4)} m, PnP rms {rms:.2f} px "
          f"(board {np.linalg.norm(T_zb[:3, 3]):.2f} m away)", flush=True)
    views = {"zed": {"corners": P_zed}}

    # 2. wrists
    movers, rest = {}, {}
    try:
        for side in SIDES:
            movers[side] = calib.Mover(side, kin, checker)
            rest[side] = movers[side].joints()
        for side in SIDES:
            m, X = movers[side], T_pose(spec["wrist"][side]["pose_in_link6"])
            K, dist = wrist_K(spec, side)
            T_base = np.eye(4) if side == "right" else T_r_l
            tgt = (calib.T_inv(T_base) @ np.append(center, 1))[:3]
            cam = calib.WristCam(side)
            try:
                for elev in ELEVATIONS:
                    q, d, e = plan_view(side, m, kin, checker, tgt, X, elev, rest[side])
                    name = f"{side}_{elev}"
                    if q is None:
                        print(f"{name}: no safe pose looks at the board; skipped", flush=True)
                        continue
                    print(f"{name}: camera {d:.2f} m away at {e} deg, moving", flush=True)
                    m.move_to(q)
                    qm = m.joints()
                    f = cam.grab()
                    c = calib.detect(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
                    if c is None:
                        print(f"{name}: board not found in the image; skipped", flush=True)
                        cv2.imwrite(str(out / f"{name}.png"), f)
                        continue
                    T_cam = T_base @ kin.fk(qm[:6] + rig.joint_offsets(side)) @ X   # camera in right base
                    ok, rv, tv = cv2.solvePnP(calib.OBJ, c, K, dist, flags=cv2.SOLVEPNP_IPPE)
                    T_cb = calib.T_from_rt(rv, tv)
                    P = corners_in(T_cam @ T_cb)
                    # ZED corners projected into this image through the arm (no PnP in this one)
                    Pc = corners_in(calib.T_inv(T_cam), P_zed)
                    proj, _ = cv2.projectPoints(Pc, np.zeros(3), np.zeros(3), K, dist)
                    proj = proj.reshape(-1, 2)
                    px = min(np.linalg.norm(proj - c, axis=1).mean(), np.linalg.norm(proj[::-1] - c, axis=1).mean())
                    views[name] = {"corners": P, "q": qm[:6].tolist(), "zed_reproj_px": float(px),
                                   "board_dist_m": float(np.linalg.norm(T_cb[:3, 3]))}
                    cv2.imwrite(str(out / f"{name}_raw.png"), f)
                    views[name]["detected_px"] = c.tolist()
                    vis = f.copy()
                    cv2.drawChessboardCorners(vis, calib.BOARD, c.astype(np.float32), True)
                    for p in proj:
                        cv2.circle(vis, tuple(int(v) for v in p), 6, (0, 0, 255), 2)
                    cv2.imwrite(str(out / f"{name}.png"), vis)
                    print(f"{name}: ZED corners land {px:.1f} px from the detected ones", flush=True)
            finally:
                cam.close()
            m.go_home(rest[side][:6])
    except (safety.SafetyStop, KeyboardInterrupt) as e:
        for m in movers.values():
            m.hold()
        if movers:
            next(iter(movers.values())).wait_clear(e if str(e) else "interrupted")
    finally:
        for side, m in movers.items():
            m.go_home(rest[side][:6])
        for m in movers.values():
            m.close()

    # 3. compare
    names = list(views)
    print("\nmean / max corner distance in the right base frame (mm):")
    print(" " * 12 + "".join(f"{n:>14s}" for n in names))
    table = {}
    for i in names:
        row = []
        for j in names:
            mn, mx = match(views[i]["corners"], views[j]["corners"])
            table[f"{i}|{j}"] = [round(mn * 1e3, 2), round(mx * 1e3, 2)]
            row.append(f"{mn * 1e3:6.1f}/{mx * 1e3:5.1f}")
        print(f"{i:>12s}" + "".join(f"{r:>14s}" for r in row))
    wr = [n for n in names if n != "zed"]
    ok = len(wr) == 4 and all(table[f"zed|{n}"][0] <= PASS_MM and views[n]["zed_reproj_px"] <= PASS_PX for n in wr)
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(wr)}/4 wrist views; pass = each within {PASS_MM} mm of the ZED "
          f"and ZED corners within {PASS_PX} px in its image")
    summary = {"time": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "spec": str(rig.SPEC.relative_to(REPO)),
               "out": a.out, "pass": ok, "pass_mm": PASS_MM, "pass_px": PASS_PX,
               "board_center_right_base_m": center.round(4).tolist(), "zed_pnp_rms_px": round(rms, 2),
               "corner_dist_mm_mean_max": table,
               "views": {n: {**{k: v for k, v in views[n].items() if k != "corners"},
                             "corners_right_base_m": np.round(views[n]["corners"], 5).tolist()} for n in names},
               "offset_from_zed_mm": {n: ((views[n]["corners"].mean(0) - P_zed.mean(0)) * 1e3).round(2).tolist()
                                      for n in names if n != "zed"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (REPO / "results/real").mkdir(parents=True, exist_ok=True)
    (REPO / "results/real/camera_verify.json").write_text(json.dumps(summary, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
