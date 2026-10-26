"""Refit the ZED pose in the right base from board positions located by the wrist cameras.

    .venv/bin/python scripts/real/fit_zed.py data/calib/zedfit/pos1 data/calib/zedfit/pos2 ... [--write]

Each directory is a verify_cameras.py run (board at a new spot each time). The board corners in the
right base come from the wrist cameras (through the arms; the mean of the views), the image points
from the ZED's own detection in that run. One PnP over all positions gives the ZED pose: many
corners, spread over the table and in depth, instead of the single board of calib.py's `zed` step.
--write puts it into the camera spec (top.extrinsics_cam_in_right_base and _in_left_base; the old
pose is kept under top.extrinsics_cam_in_right_base_v1). Check it on a held-out position with
verify_cameras.py afterwards.
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
sys.path.insert(0, str(HERE))
import calib  # noqa: E402
import rig  # noqa: E402
from snapshot import T_pose  # noqa: E402

REPO = HERE.parents[1]


def pose_dict(T):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, T[:3, :3].flatten())
    return {"pos": T[:3, 3].round(5).tolist(), "quat": q.round(5).tolist()}


def load(run, T_rb_zed_old):
    """(wrist corners in the right base (56, 3), ZED detections (56, 2), K) for one run."""
    s = json.loads((run / "summary.json").read_text())
    if s.get("zed_moved_mm", 0.0) > 1.5:
        raise SystemExit(f"{run}: the board moved during the run ({s['zed_moved_mm']} mm)")
    if "zed_detected_px" in s:
        c = np.array(s["zed_detected_px"])
    else:   # runs before the detections were saved
        c = calib.detect(cv2.cvtColor(cv2.imread(str(run / "zed_raw.png")), cv2.COLOR_BGR2GRAY))
    if "zed_K" in s:
        K = np.array(s["zed_K"])
    else:
        i = json.loads(rig.SPEC.read_text())["top"]["intrinsics_rectified"]["HD1080"]
        K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.0]])
    ref = np.array(s["views"]["zed"]["corners_right_base_m"])   # the old ZED pose: only to fix the corner order
    W = []
    for name, v in s["views"].items():
        if name == "zed":
            continue
        P = np.array(v["corners_right_base_m"])
        W.append(P if np.linalg.norm(P - ref, axis=1).mean() < np.linalg.norm(P[::-1] - ref, axis=1).mean() else P[::-1])
    W = np.mean(W, axis=0)
    # the ZED's detection order must match `ref` (it is the ZED's own pose, so it does by construction)
    return W, c, K, len(s["views"]) - 1


def reproj(T_zed_rb, W, c, K):
    rv, _ = cv2.Rodrigues(T_zed_rb[:3, :3])
    p, _ = cv2.projectPoints(W, rv, T_zed_rb[:3, 3], K, None)
    return np.linalg.norm(p.reshape(-1, 2) - c, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    spec = json.loads(rig.SPEC.read_text())
    T_old = T_pose(spec["top"]["extrinsics_cam_in_right_base"])
    data = [load(REPO / r, T_old) for r in a.runs]
    K = data[0][2]
    W = np.concatenate([d[0] for d in data])
    c = np.concatenate([d[1] for d in data])
    rv0, _ = cv2.Rodrigues(calib.T_inv(T_old)[:3, :3])
    ok, rv, tv = cv2.solvePnP(W, c, K, None, rv0, calib.T_inv(T_old)[:3, 3].copy(), useExtrinsicGuess=True,
                              flags=cv2.SOLVEPNP_ITERATIVE)
    T_new = calib.T_inv(calib.T_from_rt(rv, tv))
    print(f"{len(a.runs)} positions, {len(W)} corners")
    print(f"{'position':>34s}  {'old px':>7s}  {'new px':>7s}")
    for r, (Wi, ci, _, n) in zip(a.runs, data):
        e0, e1 = reproj(calib.T_inv(T_old), Wi, ci, K), reproj(calib.T_inv(T_new), Wi, ci, K)
        print(f"{r:>34s}  {e0.mean():7.2f}  {e1.mean():7.2f}   ({n} wrist views)")
    d = T_new[:3, 3] - T_old[:3, 3]
    ang = np.degrees(np.arccos(np.clip((np.trace(T_old[:3, :3].T @ T_new[:3, :3]) - 1) / 2, -1, 1)))
    print(f"ZED pose change: {np.round(d * 1e3, 1).tolist()} mm ({np.linalg.norm(d) * 1e3:.1f} mm), {ang:.2f} deg")
    print("new top.extrinsics_cam_in_right_base:", pose_dict(T_new))
    if a.write:
        T_r_l = T_pose(spec["arms"]["left_base_in_right_base"])
        top = spec["top"]
        top.setdefault("extrinsics_cam_in_right_base_v1", top["extrinsics_cam_in_right_base"])
        top["extrinsics_cam_in_right_base"] = {**pose_dict(T_new), "_note": (
            f"ZED HD1080 rectified-left optical frame (OpenCV) in the right base: refit "
            f"{time.strftime('%Y-%m-%d')} by scripts/real/fit_zed.py from {len(a.runs)} board positions located "
            f"by the wrist cameras ({', '.join(a.runs)})")}
        top["extrinsics_cam_in_left_base"] = {**pose_dict(calib.T_inv(T_r_l) @ T_new), "_note": (
            "the right-base pose through arms.left_base_in_right_base")}
        rig.SPEC.write_text(json.dumps(spec, indent=1) + "\n")
        print(f"written to {rig.SPEC.relative_to(REPO)}")


if __name__ == "__main__":
    main()
