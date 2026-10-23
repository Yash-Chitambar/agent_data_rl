"""Side-by-side YAM example videos from matched-start rollouts (scripts/report/yam_example_rollouts.sh).

Qualitative only (CPU MuJoCo replay of logged qpos). Each clip shows the same start under the same
test condition for two policies, labelled with the outcome from the eval parquet.

python scripts/report/yam_example_videos.py [--out report/static/videos]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import sys  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.viz.render import save_video  # noqa: E402

EX = ROOT / "results" / "C" / "examples"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SLOW = 4                 # play at 1/SLOW of real time (50 Hz control -> 12.5 fps)
MAX_T = 2.0              # s of a failed episode to show
# clip -> (test condition tag, [(policy tag, panel title)])
PAIRS = {
    "yam_latency": ("latency1", [("base", "Trained without latency"), ("latency1", "Trained with 20 ms latency")]),
    "yam_backlash": ("backlash3", [("base", "Trained without backlash"), ("backlash3", "Trained with backlash")]),
    "yam_gain4": ("gain4", [("base", "Trained at 1x gains"), ("gain4x", "Trained at 4x gains")]),
}
FAIL_TEXT = {"never_found": "did not find the hole", "wedged_partial": "stuck partly inside",
             "jammed_inside": "jammed inside", "slipped": "peg slipped in the hand",
             "left_fixture": "left the fixture", "timeout": "timeout"}


def load(pol, test):
    df = pd.read_parquet(EX / f"{pol}_on_{test}.parquet").set_index("world")
    z = np.load(EX / f"{pol}_on_{test}.rollouts.npz")
    return df, z


def roll(z, w):
    return {k.split("_", 1)[1]: z[k] for k in z.files if k.startswith(f"w{w}_")}


class Cam:
    def __init__(self, m, size):
        self.r = mujoco.Renderer(m, size[1], size[0])
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

    def frames(self, m, lay, d, w, rl, view):
        B.apply_variant_cpu(m, lay, w % len(lay.variants))
        out = []
        n = len(rl["qpos"])
        for t in range(n):
            d.qpos[:] = rl["qpos"][t]
            d.mocap_pos[0] = rl["mocap_pos"][t]
            d.mocap_quat[0] = rl["mocap_quat"][t]
            mujoco.mj_forward(m, d)
            fx = rl["mocap_pos"][t]
            if view == "wide":
                self.cam.lookat[:] = [0.5 * fx[0], 0.5 * fx[1], 0.12]
                self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.95, 215, -22
            else:
                self.cam.lookat[:] = fx
                self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.24, 150, -18
            self.r.update_scene(d, self.cam, self.opt)
            out.append(self.r.render().copy())
        return out


def label(frame, title, sub, ok):
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    f1, f2 = ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 15)
    dr.rectangle([0, 0, im.width, 30], fill=(20, 20, 24))
    dr.text((10, 6), title, font=f1, fill=(240, 240, 240))
    if sub:
        col = (40, 160, 90) if ok else (200, 60, 50)
        tw = dr.textlength(sub, font=f2)
        dr.rectangle([0, im.height - 30, tw + 20, im.height], fill=col)
        dr.text((10, im.height - 25), sub, font=f2, fill=(255, 255, 255))
    return np.asarray(im)


def compose(panels, size):
    """panels: list of (frames, title, outcome_text, ok); pads to equal length, labels the end."""
    n = max(len(f) for f, *_ in panels)
    out = []
    for t in range(n):
        row = []
        for frames, title, sub, ok in panels:
            fr = frames[min(t, len(frames) - 1)]
            done = t >= len(frames) - 1
            row.append(label(fr, title, sub if done else "", ok))
        out.append(np.concatenate(row, 1))
    hold = [out[-1]] * int(12.5 * 1.5)         # hold the result 1.5 s of video
    return out + hold


def outcome(row):
    if row.success:
        return f"seated in {row.time:.2f} s", True
    return FAIL_TEXT.get(row.failure, row.failure), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "report" / "static" / "videos"))
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vs = B.sample_study_pegs(1024, 1000)
    m, lay = B.build("yam", vs, table_top=0.0)
    d, m_bl = mujoco.MjData(m), 0.0
    size = (480, 360)
    cam = Cam(m, size)
    cap = int(MAX_T * 50)
    fps = 50 / SLOW

    if not a.only or "yam_hero" in a.only:
        df, z = load("base", "base")
        logged = sorted({int(k.split("_")[0][1:]) for k in z.files})
        ws = df.loc[logged].sort_values("start_dist", ascending=False).index[:2].tolist()
        panels = []
        for w in ws:
            r = df.loc[w]
            fr = cam.frames(m, lay, d, w, roll(z, w), "wide")[:cap]
            panels.append((fr, f"Start offset {1000 * r.start_dist:.1f} mm", *outcome(r)))
        for w in ws:
            r = df.loc[w]
            fr = cam.frames(m, lay, d, w, roll(z, w), "close")[:cap]
            panels.append((fr, "Close-up", *outcome(r)))
        top = compose(panels[:2], size)
        bot = compose(panels[2:], size)
        n = min(len(top), len(bot))
        save_video([np.concatenate([top[i], bot[i]], 0) for i in range(n)], str(out / "yam_hero.mp4"), fps=fps)
        Image.fromarray(top[-1] if False else np.concatenate([top[n // 2], bot[n // 2]], 0)).save(out.parent / "images" / "yam_hero_poster.jpg")
        print("yam_hero", ws)

    for clip, (test, pols) in PAIRS.items():
        if a.only and clip not in a.only:
            continue
        loaded = [(load(p, test), title) for p, title in pols]
        if any(not (EX / f"{p}_on_{test}.parquet").exists() for p, _ in pols):
            print("skip", clip, "(rollouts missing)")
            continue
        bl = float(test[len("backlash"):]) if test.startswith("backlash") else 0.0
        if bl != m_bl:   # logged qpos includes the hidden backlash hinges
            m, lay = B.build("yam", vs, table_top=0.0, backlash_arcmin=bl)
            d, m_bl = mujoco.MjData(m), bl
            cam.r.close()
            cam = Cam(m, size)
        (df0, z0), (df1, z1) = loaded[0][0], loaded[1][0]
        logged = sorted({int(k.split("_")[0][1:]) for k in z0.files} & {int(k.split("_")[0][1:]) for k in z1.files})
        cand = [w for w in logged if not df0.loc[w].success and df1.loc[w].success]
        if not cand:
            print("no contrasting world for", clip)
            continue
        w = max(cand, key=lambda w: df0.loc[w].start_dist)
        panels = []
        for (dfp, zp), title in loaded:
            r = dfp.loc[w]
            fr = cam.frames(m, lay, d, w, roll(zp, w), "close")[:cap]
            panels.append((fr, title, *outcome(r)))
        frames = compose(panels, size)
        save_video(frames, str(out / f"{clip}.mp4"), fps=fps)
        Image.fromarray(frames[len(frames) // 3]).save(out.parent / "images" / f"{clip}_poster.jpg")
        n_c = len(cand)
        print(clip, "world", w, f"({n_c} of {len(logged)} logged starts fail without / succeed with)")
    cam.r.close()


if __name__ == "__main__":
    main()
