"""Share the rig cameras between the live viewer (camview.py) and the scripts that need them.

A UVC camera has one owner at a time. A script that opens a camera calls `claim(cam, who)` first:
the viewer sees the claim, releases the device and shows the frames the script publishes with
`Claim.publish(frame)` instead. `Claim.release()` hands the device back. Without a viewer running,
`claim` returns at once, so the scripts work the same with or without it.

Cameras: "zed", "wrist_right", "wrist_left". State lives in /dev/shm/yam_cams/:
<cam>.claim (pid and name of the script), <cam>.held (pid of the viewer while it holds the device),
<cam>.jpg (the newest frame the script published).
"""
import os
import time
from pathlib import Path

import cv2

DIR = Path("/dev/shm/yam_cams")
CAMS = ("zed", "wrist_right", "wrist_left")
PUBLISH_WIDTH = 640
PUBLISH_PERIOD = 0.2   # s


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True


def read_pid(path):
    """(pid, text) from a state file, or (None, None) if it is missing or its process is dead."""
    try:
        pid, _, text = path.read_text().partition(" ")
        pid = int(pid)
    except (FileNotFoundError, ValueError):
        return None, None
    return (pid, text.strip()) if alive(pid) else (None, None)


class Claim:
    def __init__(self, cam, who, timeout=5.0):
        assert cam in CAMS, cam
        DIR.mkdir(exist_ok=True)
        self.cam, self.last = cam, 0.0
        self.path = DIR / f"{cam}.claim"
        self.path.write_text(f"{os.getpid()} {who}")
        held = DIR / f"{cam}.held"
        t0 = time.time()
        while read_pid(held)[0] is not None:   # the viewer still has the device
            if time.time() - t0 > timeout:
                raise RuntimeError(f"{cam}: the viewer did not release the camera in {timeout} s")
            time.sleep(0.05)

    def publish(self, frame_bgr):
        """Show this frame in the viewer (rate limited; costs ~2 ms)."""
        now = time.time()
        if now - self.last < PUBLISH_PERIOD:
            return
        self.last = now
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (PUBLISH_WIDTH, round(h * PUBLISH_WIDTH / w)), interpolation=cv2.INTER_AREA)
        tmp = DIR / f".{self.cam}.{os.getpid()}.jpg"
        cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        tmp.replace(DIR / f"{self.cam}.jpg")

    def release(self):
        if read_pid(self.path)[0] == os.getpid():
            self.path.unlink(missing_ok=True)


def claim(cam, who=None, timeout=5.0):
    import sys
    return Claim(cam, who or Path(sys.argv[0]).name, timeout)
