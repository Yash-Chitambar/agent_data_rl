"""Real cameras for the ACT stage: one capture thread per camera keeps the newest frame and the
time.monotonic() at which it arrived, so the runner always uses the latest frame and logs its age.

The ACT demos assume a 30 fps frame clock and a frame age of 30-100 ms (scripts/act/gen_demos.py
--cam_age_ms). The age logged here is arrival -> use; the true age adds exposure and transfer
(not measured: the cameras give no hardware timestamp we trust), so 30-100 ms in sim corresponds to a
logged age of roughly 0-70 ms. deploy/checks/timing.py --cameras measures rate and logged age.

Policy images match the sim renders: wrist = undistort + principal-point square crop + resize
(assets/real_camera.real_policy_image); top = ZED HD1080 rectified left, square crop, resize
(scenes/physical_rig.real_top_policy_image). Both 320 x 320 RGB.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "scripts/real"))   # after REPO: its deploy.py must not shadow this package
import rig  # noqa: E402
from blind_insertion.assets.real_camera import load_spec, real_policy_image  # noqa: E402
from blind_insertion.scenes.physical_rig import real_top_policy_image  # noqa: E402


class _Grabber:
    def __init__(self, name):
        self.name, self.frame, self.t, self.n = name, None, None, 0
        self.lock, self.stop_ev = threading.Lock(), threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stop_ev.is_set():
            f = self._grab()
            if f is None:
                continue
            t = time.monotonic()
            with self.lock:
                self.frame, self.t, self.n = f, t, self.n + 1

    def latest(self):
        with self.lock:
            return self.frame, self.t, self.n

    def close(self):
        self.stop_ev.set()
        self.th.join(timeout=2)


class WristCam(_Grabber):
    def __init__(self, side="right"):
        super().__init__(f"wrist_{side}")
        spec = load_spec(rig.SPEC)
        self.res = tuple(spec["wrist"]["sensor_resolution"])     # calibrated (1280, 1024)
        self.cap = cv2.VideoCapture(rig.WRIST_CAM[side], cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res[1])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        assert self.cap.isOpened(), f"{side} wrist camera did not open"
        self.th.start()

    def _grab(self):
        ok, f = self.cap.read()
        return f if ok and f.shape[1::-1] == self.res else None

    def close(self):
        super().close()
        self.cap.release()


class ZedTop(_Grabber):
    def __init__(self):
        super().__init__("top")
        import pyzed.sl as sl
        self.sl = sl
        self.cam = sl.Camera()
        p = sl.InitParameters()
        p.camera_resolution = sl.RESOLUTION.HD1080
        p.camera_fps = 30
        p.depth_mode = sl.DEPTH_MODE.NONE     # see scripts/real/snapshot.py: no depth model on duchamp3
        assert self.cam.open(p) == sl.ERROR_CODE.SUCCESS, "ZED open failed (USB 3? another process?)"
        self.mat = sl.Mat()
        self.th.start()

    def _grab(self):
        if self.cam.grab() != self.sl.ERROR_CODE.SUCCESS:
            return None
        self.cam.retrieve_image(self.mat, self.sl.VIEW.LEFT)
        return self.mat.get_data()[:, :, :3].copy()

    def close(self):
        super().close()
        self.cam.close()


class Cameras:
    """frames() -> ({"top", "wrist"}: (res, res, 3) uint8 RGB policy images, {camera: age s}, {camera: frame count})."""

    def __init__(self, side="right", res=320, max_age_s=0.25, timeout_s=5.0):
        self.spec, self.side, self.res, self.max_age = load_spec(rig.SPEC), side, res, max_age_s
        self.cams = {"top": ZedTop(), "wrist": WristCam(side)}
        t0 = time.monotonic()
        while any(c.latest()[0] is None for c in self.cams.values()):
            if time.monotonic() - t0 > timeout_s:
                self.close()
                raise RuntimeError("no frames from " + ", ".join(k for k, c in self.cams.items() if c.latest()[0] is None))
            time.sleep(0.01)

    def frames(self):
        now = time.monotonic()
        out, age, n = {}, {}, {}
        for k, c in self.cams.items():
            f, t, n[k] = c.latest()
            age[k] = now - t
            if age[k] > self.max_age:
                raise RuntimeError(f"{k} camera frame is {age[k] * 1e3:.0f} ms old (limit {self.max_age * 1e3:.0f} ms)")
            out[k] = (real_top_policy_image(f, self.res) if k == "top"
                      else real_policy_image(f, self.spec, self.res, self.side))
        return out, age, n

    def close(self):
        for c in self.cams.values():
            c.close()


def preprocess_check(frame_top, frame_wrist, side="right", res=320):
    """Policy images from saved raw frames (BGR, as captured), for offline checks."""
    return (real_top_policy_image(frame_top, res),
            real_policy_image(frame_wrist, load_spec(rig.SPEC), res, side))


