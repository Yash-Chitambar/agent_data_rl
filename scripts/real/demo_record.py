"""Record every streamed rig run as a presentation video: ZED + both wrist cameras + the live GaP stage graph.

    .venv/bin/python scripts/real/demo_record.py            # leave it running (tmux session demorec)

Reads what `run_bimanual --demo` (or `--stream`) publishes to /dev/shm/yam_cams/: the camera frames (camshare, 5 Hz,
640 px) and run_state.json (deploy/bimanual/live.py). A separate process: it adds no work to the control loop.
A new run starts a new file, data/real_runs/<date>/<run>/live_demo.mp4 (H.264, 10 fps, real time; a frame is
repeated until the next one arrives); the file closes 4 s after the run finishes or its runner exits.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import camshare  # noqa: E402
from camview import card, draw_graph  # noqa: E402

ROOT = HERE.parents[1]
FPS, CW, CH, WW = 10, 640, 360, 452          # ZED / graph column 640x360, wrist column 452x360; canvas 1092x720
LABEL = {"zed": "ZED (top)", "wrist_right": "right wrist (insertion arm)", "wrist_left": "left wrist (holds the PC)"}


def label(img, text):
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit(img, w, h):
    """Letterbox into w x h."""
    out = np.full((h, w, 3), 20, np.uint8)
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


class Frames:
    def __init__(self, d):
        self.d, self.img, self.m = d, {}, {}

    def get(self, cam):
        p = self.d / f"{cam}.jpg"
        try:
            m = p.stat().st_mtime
        except FileNotFoundError:
            return self.img.get(cam)
        if m != self.m.get(cam):
            im = cv2.imread(str(p))
            if im is not None:
                self.img[cam], self.m[cam] = im, m
        return self.img.get(cam)


def compose(fr, st):
    z, wr, wl = fr.get("zed"), fr.get("wrist_right"), fr.get("wrist_left")
    zed = label(fit(z, CW, CH), LABEL["zed"]) if z is not None else card("ZED: no frames", CW, CH)
    g = fit(draw_graph(st, CW), CW, CH)
    r = label(fit(wr, WW, CH), LABEL["wrist_right"]) if wr is not None else card("right wrist:\nno frames", WW, CH)
    lw = label(fit(wl, WW, CH), LABEL["wrist_left"]) if wl is not None else card("left wrist:\nno frames", WW, CH)
    return np.hstack([np.vstack([zed, g]), np.vstack([r, lw])])


def open_writer(path, size):
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(camshare.DIR), help="state directory (tests)")
    ap.add_argument("--out", default=str(ROOT / "data/real_runs"))
    ap.add_argument("--once", action="store_true", help="exit after the first recorded run (tests)")
    a = ap.parse_args()
    d = Path(a.dir)
    fr = Frames(d)
    cur, proc, out, end_t, n = None, None, None, None, 0
    print("demo_record: waiting for a streamed run (run_bimanual --demo / --stream)", flush=True)
    while True:
        t0 = time.time()
        try:
            st = json.loads((d / "run_state.json").read_text())
        except (FileNotFoundError, ValueError, OSError):
            st = None
        live = st is not None and (camshare.alive(st.get("pid", -1)) or st.get("finished")) and t0 - st.get("t", 0) < 3600
        if live and st["run"] != cur and not st.get("finished"):
            if proc is not None:
                proc.stdin.close(); proc.wait()
                print(f"saved {out}", flush=True)
            cur = st["run"]
            day = time.strftime("%Y-%m-%d", time.localtime(st.get("t0", t0)))
            run_dir = Path(a.out) / day / cur
            run_dir.mkdir(parents=True, exist_ok=True)
            out = run_dir / "live_demo.mp4"
            img = compose(fr, st)
            proc, end_t, n = open_writer(out, (img.shape[1], img.shape[0])), None, 0
            print(f"recording {cur} -> {out}", flush=True)
        if proc is not None and st is not None and st.get("run") == cur:
            proc.stdin.write(compose(fr, st).tobytes())
            n += 1
            gone = st.get("finished") or not camshare.alive(st.get("pid", -1))
            if gone and end_t is None:
                end_t = t0 + 4.0
            if end_t is not None and t0 >= end_t:
                proc.stdin.close(); proc.wait()
                print(f"saved {out} ({n / FPS:.0f} s)", flush=True)
                proc, end_t = None, None
                if a.once:
                    return
        time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))


if __name__ == "__main__":
    main()
