"""Rig, no arm motion: where are the PC and the reader? One ZED frame, nominal footprints drawn on it.

The ZED pose in the right arm base (configs/real_cameras_duchamp3.json, checked: the FK tcp at the
ready pose lands between the finger pads in the ZED image) maps table points to pixels. The reader
body and the PC top are the bright blobs on the dark table near their nominal spots; their
centroids, back-projected onto the planes at their top heights, give the positions, and the
blobs' minimum-area rectangles the yaw (mod 90 deg, so |yaw| < 45 deg is read right). Prints the
move to make (mm, base frame) and whether each lies in its trained range; writes an overlay.

    python -m deploy.checks.placement [--image data/task_h/frames/ready1/top_raw.png]
"""
import argparse
import json
import math
import time

import cv2
import mujoco
import numpy as np

from deploy import conventions as C
from deploy.runner import load_config
from blind_insertion.scenes import pc_bench as S
from blind_insertion.scenes import physical_rig as P

OUT = C.REPO / "data/task_h/placement"
READER_TOP = 0.0088          # reader body top above the table (m)
READER_BODY = (0.042, 0.035)  # body length along the plug axis, width (m)


class TopCam:
    def __init__(self):
        t = P.load()["top"]
        self.K = t["intrinsics_rectified"]["HD1080"]
        pose = t["extrinsics_cam_in_right_base"]
        q = np.asarray(pose["quat"], float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q / np.linalg.norm(q))
        self.R, self.t = R.reshape(3, 3), np.asarray(pose["pos"], float)

    def px(self, p):
        c = self.R.T @ (np.asarray(p, float) - self.t)
        return np.array([self.K["fx"] * c[0] / c[2] + self.K["cx"], self.K["fy"] * c[1] / c[2] + self.K["cy"]])

    def ground(self, u, v, h):
        d = self.R @ np.array([(u - self.K["cx"]) / self.K["fx"], (v - self.K["cy"]) / self.K["fy"], 1.0])
        return self.t + (h - self.t[2]) / d[2] * d


def rect(c, yaw_deg, along_x, along_y, z):
    """Corners of a box footprint centred at c (xy), sides along_x / along_y before the yaw."""
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return [np.r_[np.asarray(c) + R @ np.array([sx * along_x / 2, sy * along_y / 2]), z]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def nominal():
    """Nominal reader centre, PC centre (xy) and port (xyz), from scenes/pc_bench.py."""
    a = math.radians(S.PC_YAW_DEG)
    w, h, d = S.PC_SIZE
    # the panel faces -x; the port sits PORT_FROM_LEFT from its +y end (left, seen from the arms)
    pc = np.array(S.PORT_XY) + np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ np.array(
        [d / 2, w / 2 - S.PORT_FROM_LEFT])
    return np.array(S.READER_XY), pc, np.r_[S.PORT_XY, S.PORT_HEIGHT]


def find_blob(img, cam, centre_xy, z, half_m, area_m2):
    """Brightest table blob near centre_xy whose footprint area is closest to area_m2 -> (xy, yaw deg, contour px)."""
    box = [cam.px(np.r_[centre_xy + [dx, dy], z]) for dx, dy in
           ((-half_m, -half_m), (half_m, -half_m), (half_m, half_m), (-half_m, half_m))]
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.array(box, np.int32)], 255)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bright = ((hsv[..., 2] > 140) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright & mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cs:
        g = np.array([cam.ground(u, v, z)[:2] for u, v in c[:, 0]])
        if len(g) < 5:
            continue
        area = cv2.contourArea(g.astype(np.float32))
        score = abs(math.log(max(area, 1e-6) / area_m2))
        if score < 0.7 and (best is None or score < best[0]):
            (cx, cy), _, ang = cv2.minAreaRect(g.astype(np.float32))
            ang = (ang + 45) % 90 - 45
            best = (score, np.array([cx, cy]), ang, c)
    return None if best is None else best[1:]


def grab():
    from deploy.cameras import ZedTop
    cam = ZedTop()
    try:
        t0 = time.monotonic()
        while cam.latest()[0] is None:
            if time.monotonic() - t0 > 5:
                raise RuntimeError("no ZED frame")
            time.sleep(0.02)
        time.sleep(0.3)                      # let exposure settle
        return cam.latest()[0].copy()
    finally:
        cam.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="saved ZED HD1080 rectified-left frame instead of a live one")
    ap.add_argument("--name", default=time.strftime("%Y%m%d_%H%M%S"))
    a = ap.parse_args()
    cfg = load_config()
    img = cv2.imread(a.image) if a.image else grab()
    cam = TopCam()
    r_nom, pc_nom, port = nominal()
    w, h, d = S.PC_SIZE
    vis = img.copy()
    pxs = lambda pts: np.array([cam.px(p) for p in pts], np.int32)  # noqa: E731
    # nominal footprints (green: reader, magenta: PC top/bottom), trained ranges (yellow), port (red)
    cv2.polylines(vis, [pxs(rect(r_nom, S.READER_YAW_DEG, *READER_BODY, READER_TOP))], True, (0, 255, 0), 2)
    cv2.arrowedLine(vis, tuple(cam.px(np.r_[r_nom, 0]).astype(int)), tuple(cam.px(np.r_[r_nom + [0.07, 0], 0]).astype(int)),
                    (0, 255, 0), 2, tipLength=0.2)
    cv2.polylines(vis, [pxs(rect(r_nom, 0, 0.06, 0.10, 0))], True, (0, 255, 255), 1)
    for z in (0.0, h):
        cv2.polylines(vis, [pxs(rect(pc_nom, S.PC_YAW_DEG, d, w, z))], True, (255, 0, 255), 2 if z else 1)
    cv2.polylines(vis, [pxs(rect(port[:2], 0, 0.04, 0.08, port[2]))], True, (0, 255, 255), 1)
    cv2.circle(vis, tuple(cam.px(port).astype(int)), 6, (0, 0, 255), -1)
    rep = {}
    found_r = find_blob(img, cam, r_nom, READER_TOP, 0.14, READER_BODY[0] * READER_BODY[1])
    found_pc = find_blob(img, cam, pc_nom, h, 0.20, w * d)
    lines = []
    if found_r is None:
        lines.append("reader: not found near its nominal spot (bright body on the table)")
    else:
        xy, yaw, c = found_r
        dxy = xy - r_nom
        ok = abs(dxy[0]) <= 0.03 and abs(dxy[1]) <= 0.05 and abs(yaw - S.READER_YAW_DEG) <= 20
        rep["reader"] = dict(xy_mm=(xy * 1e3).round(1).tolist(), off_mm=(dxy * 1e3).round(1).tolist(), yaw_deg=round(yaw, 1), in_range=bool(ok))
        cv2.drawContours(vis, [c], -1, (255, 128, 0), 2)
        lines.append(f"reader: at x {xy[0] * 1e3:.0f}, y {xy[1] * 1e3:.0f} mm, yaw {yaw:+.0f} deg (body rectangle, mod 90) -> "
                     f"move {-dxy[0] * 1e3:+.0f} mm in x, {-dxy[1] * 1e3:+.0f} mm in y, turn {S.READER_YAW_DEG - yaw:+.0f} deg; "
                     f"{'in' if ok else 'OUTSIDE'} the trained range (+-30 x, +-50 y mm, +-20 deg)")
    if found_pc is None:
        lines.append("PC: not found near its nominal spot (silver top)")
    else:
        xy, yaw, c = found_pc
        a_ = math.radians(yaw)
        port_xy = xy + np.array([[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]]) @ np.array([-d / 2, -(w / 2 - S.PORT_FROM_LEFT)])
        dxy = port_xy - port[:2]
        ok = abs(dxy[0]) <= 0.01 and abs(dxy[1]) <= 0.01 and abs(yaw - S.PC_YAW_DEG) <= 10
        rep["pc"] = dict(centre_mm=(xy * 1e3).round(1).tolist(), port_mm=(port_xy * 1e3).round(1).tolist(),
                         port_off_mm=(dxy * 1e3).round(1).tolist(), yaw_deg=round(yaw, 1), within_1cm=bool(ok))
        cv2.drawContours(vis, [c], -1, (255, 128, 0), 2)
        lines.append(f"PC: port at x {port_xy[0] * 1e3:.0f}, y {port_xy[1] * 1e3:.0f} mm, yaw {yaw:+.0f} deg -> move {-dxy[0] * 1e3:+.0f} mm "
                     f"in x, {-dxy[1] * 1e3:+.0f} mm in y, turn {S.PC_YAW_DEG - yaw:+.0f} deg; "
                     f"{'within' if ok else 'NOT within'} 1 cm of nominal (the handoff point assumes the nominal PC)")
    for s in lines:
        print(s)
    print("(+x = away from the right arm base, toward the PC; +y = toward the left arm; turn + = counter-clockwise seen from above)")
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"{a.name}.png"), vis)
    (OUT / f"{a.name}.json").write_text(json.dumps(dict(rep, source=a.image or "live"), indent=1) + "\n")
    print(f"overlay: {OUT / (a.name + '.png')} (green reader, magenta PC, yellow ranges, blue what was found)")


if __name__ == "__main__":
    main()
