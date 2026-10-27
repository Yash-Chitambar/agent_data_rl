"""Live view of the rig cameras (ZED + both wrist cameras) in a browser, through viser.

    .venv/bin/python scripts/real/camview.py [--port 8090] [--share]

Open http://duchamp3.elf-altair.ts.net:8090 from any device on the tailnet (the default host is the
Tailscale address, so the page is not on the lab network). --share also asks viser for a public
https link (share.viser.studio, expires in 24 h, new link on each start, written to
/dev/shm/yam_cams/share_url): anyone with that link can see the cameras. --share binds all
interfaces, because the tunnel connects to 127.0.0.1.

Cost: the cameras open only while a browser is connected (released 15 s after the last one leaves),
at low resolution (wrists 640x480 MJPG, ZED 672x376 left half of the UVC stereo stream, not
rectified). Each frame is sent once, as JPEG, at the rate set in the page (default 5 fps).

The viewer gives a camera up when a script claims it (scripts/real/camshare.py: calib.py and
snapshot.py do), and shows the frames that script publishes instead. The status panel shows the
software stop, who has each camera and which rig scripts are running.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import viser

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camshare  # noqa: E402
import estop  # noqa: E402
import rig  # noqa: E402

DEVICES = {   # cam: (device, width, height, fourcc, fps, crop to the left half)
    "zed": ("/dev/v4l/by-id/usb-Technologies__Inc._ZED_2-video-index0", 1344, 376, None, 15, True),
    "wrist_right": (rig.WRIST_CAM["right"], 640, 480, "MJPG", 15, False),
    "wrist_left": (rig.WRIST_CAM["left"], 640, 480, "MJPG", 15, False),
}
LABELS = {"zed": "ZED (top, left eye)", "wrist_right": "Right wrist", "wrist_left": "Left wrist"}
IDLE_RELEASE_S = 15.0
SCRIPT_KEYS = ("scripts/real/", "calibration.", "yam-control")


class Cam(threading.Thread):
    """Reads one camera while `want` is set, keeps only the newest frame."""

    def __init__(self, name):
        super().__init__(daemon=True)
        self.name = name
        self.want = threading.Event()
        self.frame, self.stamp, self.seq, self.error = None, 0.0, 0, None
        self.held = camshare.DIR / f"{name}.held"

    def open(self):
        dev, w, h, fourcc, fps, _ = DEVICES[self.name]
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        return cap

    def run(self):
        crop = DEVICES[self.name][5]
        while True:
            self.want.wait()
            camshare.DIR.mkdir(exist_ok=True)
            self.held.write_text(f"{os.getpid()} camview")
            cap = self.open()
            if cap is None:
                self.error = "cannot open (unplugged, or in use by a script that did not claim it)"
            fails = 0
            while self.want.is_set() and cap is not None:
                ok, f = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        self.error = "no frames"
                        break
                    time.sleep(0.03)
                    continue
                fails, self.error = 0, None
                self.frame = f[:, : f.shape[1] // 2] if crop else f
                self.stamp, self.seq = time.time(), self.seq + 1
            if cap is not None:
                cap.release()
            self.held.unlink(missing_ok=True)
            if self.want.is_set():   # failed: retry in a few seconds
                time.sleep(3.0)


def card(text, w, h=None):
    h = h or round(w * 0.75)
    img = np.full((h, w, 3), 40, np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (12, 30 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def stamp_text(img, text):
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit_width(img, w):
    if img.shape[1] == w:
        return img.copy()
    return cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA)


def rig_scripts():
    """Running rig scripts: (pid, short command line)."""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        if "python" in cmd.split(" ")[0] and any(k in cmd for k in SCRIPT_KEYS) and "camview.py" not in cmd:
            out.append((int(p.name), " ".join(a.split("/")[-1] if i < 2 else a for i, a in enumerate(cmd.split(" ")))[:90]))
    return sorted(out)


def tailscale_ip():
    try:
        return subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3).stdout.split()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=None, help="default: the Tailscale address (0.0.0.0 with --share)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--share", action="store_true", help="also open a public viser share link")
    a = ap.parse_args()
    # the share tunnel connects to 127.0.0.1, so --share needs all interfaces
    host = a.host or ("0.0.0.0" if a.share else tailscale_ip() or "127.0.0.1")
    camshare.DIR.mkdir(exist_ok=True)

    server = viser.ViserServer(host=host, port=a.port, label="duchamp3 cameras", verbose=False)
    server.gui.configure_theme(control_width="large", show_logo=False)
    status = server.gui.add_markdown("starting")
    with server.gui.add_folder("Stream"):
        fps = server.gui.add_slider("fps", min=1, max=15, step=1, initial_value=5)
        width = server.gui.add_dropdown("width", ("320", "480", "640"), initial_value="480")
        quality = server.gui.add_slider("JPEG quality", min=30, max=90, step=5, initial_value=60)
    # layout: [wrists, stacked] [ZED above the status and controls], docked from the right edge
    cams, views, panels = {n: Cam(n) for n in camshare.CAMS}, {}, {}
    for n in camshare.CAMS:
        panels[n] = server.gui.add_panel()
        with panels[n].add_tab(LABELS[n]):
            views[n] = server.gui.add_image(card("waiting for a viewer", 480), format="jpeg", jpeg_quality=60)
        cams[n].start()
    panels["zed"].dock_right()
    panels["zed"].set_width(660)
    panels["wrist_right"].dock_right()
    panels["wrist_right"].set_width(500)
    panels["wrist_left"].dock_below(panels["wrist_right"])
    server.gui.main_panel.dock_below(panels["zed"])
    url, url_checked = None, 0.0

    def share():
        nonlocal url
        new = server.request_share_url(verbose=False)   # the cached link while the tunnel is up
        if new != url:
            url = new
            print(time.strftime("%H:%M:%S"), "share link:", url, flush=True)
            (camshare.DIR / "share_url").write_text(f"{url}\n")

    if a.share:
        share()
    print(f"camview: http://{tailscale_ip() if host == '0.0.0.0' else host}:{a.port}", flush=True)

    sent = {n: None for n in cams}   # what each view shows now, to send each frame once
    last_client, last_status = -1e9, 0.0
    while True:
        t0 = time.time()
        clients = len(server.get_clients())
        if clients:
            last_client = t0
        watching = t0 - last_client < IDLE_RELEASE_S
        w = int(width.value)
        for n, cam in cams.items():
            owner_pid, owner = camshare.read_pid(camshare.DIR / f"{n}.claim")
            if owner_pid is not None or not watching:
                cam.want.clear()
            else:
                cam.want.set()
            if not clients:
                continue
            if owner_pid is not None:
                jpg = camshare.DIR / f"{n}.jpg"
                try:
                    m = jpg.stat().st_mtime
                except FileNotFoundError:
                    m = None
                key = ("pub", m, w)
                if m is not None and t0 - m < 30:
                    img = cv2.imread(str(jpg))
                    if img is None:
                        continue
                    img = stamp_text(fit_width(img, w), f"{owner} (pid {owner_pid})  {time.strftime('%H:%M:%S', time.localtime(m))}")
                else:
                    key = ("busy", owner_pid, w)
                    img = card(f"in use by {owner}\n(pid {owner_pid})\nno frames published yet", w)
            elif cam.error and cam.frame is None:
                key = ("err", cam.error, w)
                img = card(f"{LABELS[n]}:\n{cam.error}", w)
            elif cam.frame is not None and cam.seq != (sent[n] or (None, -1))[1]:
                key = ("live", cam.seq, w)
                img = stamp_text(fit_width(cam.frame, w), time.strftime("%H:%M:%S", time.localtime(cam.stamp)))
            else:
                continue
            if key != sent[n]:
                views[n]._jpeg_quality = int(quality.value)
                views[n].image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                sent[n] = key
        if a.share and t0 - url_checked > 60:   # links expire after 24 h: get a new one
            url_checked = t0
            threading.Thread(target=share, daemon=True).start()
        if t0 - last_status > 1.0:
            last_status = t0
            stop = "**SET** (arms held; `c` in the stop terminal clears it)" if estop.stopped() else "clear"
            lines = [f"**{time.strftime('%Y-%m-%d %H:%M:%S')}**, {clients} viewer(s)"
                     + (f", public link {url}" if url else ""), "",
                     f"software stop: {stop}", ""]
            for n in cams:
                pid, who = camshare.read_pid(camshare.DIR / f"{n}.claim")
                state = f"script: {who} (pid {pid})" if pid else ("live" if cams[n].want.is_set() and cams[n].frame is not None
                                                                   else cams[n].error or "idle")
                lines.append(f"- {LABELS[n]}: {state}")
            procs = rig_scripts()
            lines += ["", "rig scripts running:" if procs else "rig scripts running: none"]
            lines += [f"- `{c}` ({pid})" for pid, c in procs]
            status.content = "\n".join(lines)
        time.sleep(max(0.0, 1.0 / fps.value - (time.time() - t0)))


if __name__ == "__main__":
    main()
