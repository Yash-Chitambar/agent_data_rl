"""Capture the calibrated three-camera rig with frame and host timing logs.

No robot commands. Run beside the intended inference workload to measure contention.
The viewer yields each camera through camshare. Camera hardware and host clocks are
kept separate: a host receive timestamp is not an exposure timestamp.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENCV_VIDEOIO_V4L_SELECT_TIMEOUT", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import camshare  # noqa: E402
import rig  # noqa: E402
from blind_insertion.assets.real_camera import load_spec, real_policy_image  # noqa: E402
from blind_insertion.scenes.physical_rig import real_top_policy_image  # noqa: E402

FIELDS = ("camera", "local_frame_id", "host_read_start_ns", "host_receive_ns", "camera_image_ns",
          "camera_clock_domain", "policy_ready_ns", "policy_consumed_ns", "exact_repeat_of_previous",
          "raw_png", "policy_png", "width", "height", "error")
WRIST_SIZE = (1280, 1024)


class ImageWriter:
    """Compress sampled images off the acquisition thread; fail explicitly on backlog."""

    def __init__(self, capacity=8):
        self.queue = queue.Queue(maxsize=capacity)
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                if self.error is None:
                    for path, frame in item:
                        if not cv2.imwrite(str(path), frame):
                            raise RuntimeError(f"failed to write {path}")
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.queue.task_done()

    def submit(self, images):
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.queue.put_nowait([(path, image.copy()) for path, image in images])
        except queue.Full as e:
            raise RuntimeError("image writer backlog exceeded capacity") from e

    def close(self):
        self.queue.put(None)
        self.queue.join()
        self.thread.join()
        if self.error:
            raise RuntimeError(self.error)


class Wrist:
    def __init__(self, side: str):
        self.cap = cv2.VideoCapture(rig.WRIST_CAM[side], cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WRIST_SIZE[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WRIST_SIZE[1])
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"{side} wrist camera did not open")
        size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if size != WRIST_SIZE:
            self.cap.release()
            raise RuntimeError(f"{side} wrist mode is {size}, expected calibrated {WRIST_SIZE}")
        self.settings = {key: self.cap.get(prop) for key, prop in {
            "fps": cv2.CAP_PROP_FPS, "fourcc": cv2.CAP_PROP_FOURCC,
            "exposure": cv2.CAP_PROP_EXPOSURE, "gain": cv2.CAP_PROP_GAIN,
            "white_balance": cv2.CAP_PROP_WB_TEMPERATURE, "focus": cv2.CAP_PROP_FOCUS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
        }.items()}
        self.settings.update(resolution=list(size), device=rig.WRIST_CAM[side])

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame, None

    def close(self):
        self.cap.release()


class Zed:
    def __init__(self):
        import pyzed.sl as sl
        self.sl = sl
        self.cam = sl.Camera()
        p = sl.InitParameters()
        p.set_from_serial_number(rig.ZED_SERIAL)
        p.camera_resolution = sl.RESOLUTION.HD1080
        p.camera_fps = 30
        p.depth_mode = sl.DEPTH_MODE.NONE
        p.open_timeout_sec = 15
        err = self.cam.open(p)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED {rig.ZED_SERIAL} open failed: {err}")
        self.mat = sl.Mat()
        self.settings = {"resolution": [1920, 1080], "fps": 30,
                         "view": "SDK rectified left", "depth_mode": "NONE", "serial": rig.ZED_SERIAL}

    def read(self):
        if self.cam.grab() != self.sl.ERROR_CODE.SUCCESS:
            return False, None, None
        image_ns = int(self.cam.get_timestamp(self.sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
        self.cam.retrieve_image(self.mat, self.sl.VIEW.LEFT)
        return True, self.mat.get_data()[:, :, :3].copy(), image_ns

    def close(self):
        self.cam.close()


def capture_one(name: str, out: Path, deadline: dict, spec: dict, policy_res: int,
                save_every: int, policy_save_every: int, start: threading.Event,
                ready: threading.Event, results: dict) -> None:
    claim = None
    src = None
    rows = 0
    repeats = 0
    prev_hash = None
    image_writer = None
    try:
        claim = camshare.claim(name, "measure_cameras.py")
        src = Zed() if name == "zed" else Wrist(name.removeprefix("wrist_"))
        folder = out / name
        folder.mkdir(parents=True, exist_ok=False)
        image_writer = ImageWriter()
        with (folder / "frames.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            ready.set()
            start.wait()
            while time.monotonic_ns() < deadline["end_ns"]:
                t0 = time.monotonic_ns()
                ok, frame, camera_ns = src.read()
                recv = time.monotonic_ns()
                if not ok or frame is None:
                    writer.writerow(dict(camera=name, local_frame_id=rows, host_read_start_ns=t0,
                                         host_receive_ns=recv, error="read failed"))
                    raise RuntimeError(f"{name}: camera read failed")
                if name == "zed" and frame.shape[:2] != (1080, 1920):
                    raise RuntimeError(f"ZED frame shape {frame.shape}, expected HD1080")
                digest = hashlib.blake2b(frame, digest_size=8).hexdigest()
                repeat = digest == prev_hash
                repeats += int(repeat)
                prev_hash = digest
                policy = (real_top_policy_image(frame, policy_res) if name == "zed" else
                          real_policy_image(frame, spec, policy_res, name.removeprefix("wrist_")))
                ready_ns = time.monotonic_ns()
                raw_path = policy_path = ""
                images = []
                if rows % save_every == 0:
                    raw_path = f"raw_{rows:06d}.png"
                    images.append((folder / raw_path, frame))
                if rows % policy_save_every == 0:
                    policy_path = f"policy_{rows:06d}.png"
                    images.append((folder / policy_path, cv2.cvtColor(policy, cv2.COLOR_RGB2BGR)))
                if images:
                    image_writer.submit(images)
                claim.publish(frame)
                writer.writerow(dict(camera=name, local_frame_id=rows, host_read_start_ns=t0,
                                     host_receive_ns=recv, camera_image_ns=camera_ns,
                                     camera_clock_domain="ZED SDK image clock" if camera_ns is not None else "unavailable",
                                     policy_ready_ns=ready_ns, policy_consumed_ns=None,
                                     exact_repeat_of_previous=int(repeat), raw_png=raw_path,
                                     policy_png=policy_path, width=frame.shape[1], height=frame.shape[0], error=""))
                rows += 1
        results[name] = {"frames": rows, "exact_repeats": repeats, "settings": src.settings}
    except Exception as e:  # still close every camera on partial failure
        results[name] = {"frames": rows, "error": f"{type(e).__name__}: {e}"}
    finally:
        ready.set()  # report both successful startup and startup failure to the coordinator
        if image_writer is not None:
            try:
                image_writer.close()
            except Exception as e:
                results[name] = {"frames": rows, "error": f"{type(e).__name__}: {e}"}
        if src is not None:
            src.close()
        if claim is not None:
            claim.release()


def run(a):
    if a.policy_save_every is None:
        a.policy_save_every = a.save_every
    if not np.isfinite(a.seconds) or a.seconds <= 0 or a.seconds > 300 or a.policy_res <= 0 or a.save_every < 1 or a.policy_save_every < 1:
        raise SystemExit("seconds must be in (0,300], policy-res positive, save-every >= 1")
    if not a.name or "/" in a.name or a.name in (".", ".."):
        raise SystemExit("name must be one path component")
    session = Path(a.session).resolve()
    if not (session / "session.json").is_file():
        raise SystemExit("initialize the session with measure_manual.py init first")
    out = session / "cameras" / a.name
    out.mkdir(parents=True, exist_ok=False)
    spec = load_spec(str(rig.SPEC))
    start = threading.Event()
    results: dict = {}
    ready = {name: threading.Event() for name in ("wrist_right", "wrist_left", "zed")}
    deadline = {"end_ns": 0}
    import signal
    cancelled = {"value": False}
    def request_stop(signum, frame):
        cancelled["value"] = True
        deadline["end_ns"] = 0
        start.set()
    signal.signal(signal.SIGTERM, request_stop)
    threads = [threading.Thread(target=capture_one, args=(name, out, deadline, spec, a.policy_res,
                                                           a.save_every, a.policy_save_every, start, ready[name], results), daemon=True)
               for name in ("wrist_right", "wrist_left", "zed")]
    for thread in threads:
        thread.start()
    readiness_deadline = time.monotonic() + 25
    for name, event in ready.items():
        if not event.wait(max(0, readiness_deadline - time.monotonic())):
            results[name] = {"error": "camera startup did not complete before timeout"}
    start_ns = time.monotonic_ns()
    startup_failed = bool(results) or cancelled["value"]
    deadline["end_ns"] = start_ns if startup_failed else start_ns + int(a.seconds * 1e9)
    if not startup_failed and getattr(a, "ready_file", None):
        Path(a.ready_file).write_text(json.dumps({"host_monotonic_ns": start_ns, "cameras": list(ready)}) + "\n")
    start.set()
    if startup_failed:
        print("Camera startup failed; capture window cancelled. See capture.json.", flush=True)
    else:
        print(f"All three cameras ready; recording for {a.seconds:.1f} seconds.", flush=True)
    join_deadline = time.monotonic() + a.seconds + 25
    for thread in threads:
        thread.join(timeout=max(0, join_deadline - time.monotonic()))
    for name, thread in zip(("wrist_right", "wrist_left", "zed"), threads):
        if thread.is_alive():
            results[name] = {"error": "camera worker did not stop before timeout"}
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "host_start_monotonic_ns": start_ns,
                "requested_seconds": a.seconds, "calibration_file": str(rig.SPEC),
                "calibration_sha256": hashlib.sha256(rig.SPEC.read_bytes()).hexdigest(),
                "save_every": a.save_every, "policy_save_every": a.policy_save_every,
                "startup_failed": startup_failed, "results": results,
                "clock_note": "ZED image timestamps and host monotonic timestamps have not been aligned. "
                              "Wrist hardware exposure time and policy consumption time are unavailable; "
                              "policy_ready_ns is preprocessing completion, not model input time."}
    (out / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if startup_failed or any("error" in r for r in results.values()) or len(results) != 3 or any(t.is_alive() for t in threads):
        raise SystemExit("camera capture incomplete; see capture.json")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ready-file", help="write readiness marker only after all three sources initialize")
    p.add_argument("--session", required=True)
    p.add_argument("--name", required=True, help="new capture name inside session/cameras")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--policy-res", type=int, default=320)
    p.add_argument("--save-every", type=int, default=1, help="save every Nth raw image; log every frame")
    p.add_argument("--policy-save-every", type=int, help="save every Nth policy crop; defaults to --save-every")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
