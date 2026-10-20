"""Data, images and videos for the USB-C failure report (report_usbc/).

python scripts/usbc_fail_report.py [--no-media]   -> report_usbc/static/{data,images,videos}

Reads only frozen results in results/E/fail/ (evals, the GPU/CPU slip replay) and the training
log of E-usbc-proprio-curr-s0.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blind_insertion.analysis import stats as S  # noqa: E402

F = ROOT / "results" / "E" / "fail"
OUT = ROOT / "report_usbc" / "static"


def r(x, nd=6):
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return r(x.tolist(), nd)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, (np.integer, np.bool_)):
        return int(x)
    return x


def write(name, obj):
    p = OUT / "data" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r(obj), separators=(",", ":")))
    print("wrote", p.relative_to(ROOT), f"{p.stat().st_size / 1024:.1f} KB")


def rate(k, n):
    lo, hi = S.wilson(k, n)
    return dict(mean=k / n, lo=lo, hi=hi, k=int(k), n=int(n))


def bar(label, d, col, color, note=""):
    return dict(label=label, color=color, note=note, **rate(int(d[col].sum()), len(d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-media", action="store_true")
    a = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    ev = lambda n: pd.read_parquet(F / f"stage1_{n}.parquet")  # noqa: E731

    # 1. outcome of the trained policy at stage 1 (box pads)
    base = ev("last")
    order = ["success", "slipped", "left_fixture", "never_found", "wedged_partial", "jammed_inside", "captured_then_lost"]
    names = {"success": "Seated", "slipped": "Slipped from the grasp", "left_fixture": "Left the fixture",
             "never_found": "Did not find the port", "wedged_partial": "Stuck partly inside",
             "jammed_inside": "Inside, not seated", "captured_then_lost": "Entered, then lost"}
    vc = base.failure.value_counts()
    outcome = [dict(label=names[k], color=0 if k == "success" else 1, **rate(int(vc.get(k, 0)), len(base)))
               for k in order if vc.get(k, 0) > 0]
    write("outcome.json", dict(title="Outcome of 2,048 episodes, trained policy, stage 1, box pads",
                               xlabel="share of episodes", rows=outcome))

    # 2. one-factor sweep with the trained policy (success)
    sweep = [("Env as trained (box pads)", "last", 3), ("No in-hand tilt", "notilt_hand", 2),
             ("No in-hand tilt, gripper rotation or fixture tilt", "notilt_all", 2),
             ("No start offset", "nooffset", 3), ("Episode limit 16 s (not 8 s)", "long16s", 3),
             ("Softer pads (4 ms, default solimp)", "softpads", 1),
             ("Sphere pads, 4 × 4 per pad (same policy)", "spheres4", 0)]
    rows = []
    for lab, n, c in sweep:
        p = F / f"stage1_{n}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            row = bar(lab, d, "success", c)
            row["slip"] = rate(int(d.slip.sum()), len(d))
            rows.append(row)
    write("sweep.json", dict(title="Success of the trained policy, one change at a time (2,048 matched starts)",
                             xlabel="success", rows=rows, extra="slip"))

    # 3. zero action: slip rate per physics setting
    zs = [("Box pads (env)", "zero", 1), ("Box pads, 2x solver iterations", "zero_solver", 3),
          ("Box pads, elliptic cone + impratio 10", "zero_elliptic10", 3),
          ("Box pads, elliptic + more iterations", "zero_elliptic10_solver", 3),
          ("Box pads, pyramidal + impratio 10", "zero_impratio10", 1),
          ("Sphere pads, 4 × 4", "zero_spheres4_logged", 0)]
    rows = [bar(lab, ev(n), "slip", c) for lab, n, c in zs if (F / f"stage1_{n}.parquet").exists()]
    write("zero.json", dict(title="Slip rate with zero action (2,048 matched starts)", xlabel="slipped", rows=rows))

    # 4. slip against pad contact count (GPU, all worlds) and body width
    g = pd.read_parquet(F / "slip_gpu_cpu_gpu_all.parquet")
    from blind_insertion.assets import builder as B  # noqa: PLC0415
    vs = B.sample_usbc(320, 1000, clearance=(3e-4, 3e-4))
    g["width"] = [1000 * vs[k].handle["width"] for k in g.variant]
    bins = [(0, 0, "0"), (1, 3, "1–3"), (4, 6, "4–6"), (8, 99, "8 (full)")]
    crow = []
    for lo, hi, lab in bins:
        d = g[(g.gpu_pad_contacts >= lo) & (g.gpu_pad_contacts <= hi)]
        if len(d):
            crow.append(dict(label=f"{lab} contacts · {len(d)} worlds", color=0 if lo >= 8 else 1,
                             **rate(int(d.gpu_slip.sum()), len(d))))
    write("contacts.json", dict(title="Slip rate against pad–drive contact points after the first step (zero action, GPU)",
                                xlabel="slipped", rows=crow))
    q = pd.qcut(g.width, 5)
    wrow = []
    for iv, d in g.groupby(q, observed=True):
        full = rate(int((d.gpu_pad_contacts >= 8).sum()), len(d))
        wrow.append(dict(label=f"{iv.left:.0f}–{iv.right:.0f} mm", color=0 if iv.left >= 22 else 3, **full))
    write("width.json", dict(title="Worlds with full 8-point pad contact, by drive body width", xlabel="full contact",
                             rows=wrow))

    # 5. CPU replay summary
    c = pd.read_parquet(F / "slip_gpu_cpu.parquet")
    write("replay.json", dict(by=c.groupby("gpu_slip").agg(n=("world", "count"), cpu_slip=("cpu_slip", "mean"),
                                                           gpu_contacts=("gpu_pad_contacts", "median"),
                                                           cpu_contacts=("cpu_pad_contacts", "median")).reset_index()
                              .to_dict("records")))

    # 6. training curve of the stalled run
    d = pd.read_csv(ROOT / "runs" / "E-usbc-proprio-curr-s0" / "train_log.csv")
    k = d.success.fillna(0) * d.episodes
    s = k.rolling(10, min_periods=1).sum() / d.episodes.rolling(10, min_periods=1).sum()
    kc = d.capture.fillna(0) * d.episodes
    sc = kc.rolling(10, min_periods=1).sum() / d.episodes.rolling(10, min_periods=1).sum()
    idx = np.unique(np.linspace(0, len(d) - 1, 160).astype(int))
    x = (d.env_steps.values / 1e6)[idx]
    write("curves.json", dict(xLabel="environment steps (millions)", yLabel="training share",
                              panels=[dict(title="Stage 1 for the whole run (clearance 0.3 mm)", keys=["succ", "cap"])],
                              series={"succ": dict(label="Seated", color=0, seeds=[dict(name="E-usbc-s0", x=x, y=s.values[idx], steps=float(x[-1]), to90=None)]),
                                      "cap": dict(label="Entered the port", color=2, seeds=[dict(name="E-usbc-s0", x=x, y=sc.values[idx], steps=float(x[-1]), to90=None)])}))

    if not a.no_media:
        media()


def media():
    import imageio.v2 as imageio  # noqa: PLC0415
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    from blind_insertion.assets import builder as B  # noqa: PLC0415
    from blind_insertion.viz.replay import render_rollout  # noqa: PLC0415

    vs = B.sample_usbc(320, 1000, clearance=(3e-4, 3e-4))
    m, lay = B.build("yam", vs)
    tmp = ROOT / "report_usbc" / ".build"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    def clip(npz, w, T=40):
        z = np.load(npz)
        roll = {kk.split("_", 1)[1]: z[kk][:T] for kk in z.files if kk.startswith(f"w{w}_")}
        return render_rollout(m, lay, w, roll, tmp / f"{Path(npz).stem}_w{w}", size=(400, 300),
                              azimuth=120, elevation=-12, distance=0.09)

    def label(fr, text):
        im = Image.fromarray(fr)
        dr = ImageDraw.Draw(im)
        dr.rectangle([6, 6, 12 + dr.textlength(text, font=font), 28], fill=(255, 255, 255))
        dr.text((9, 8), text, fill=(20, 20, 20), font=font)
        return np.asarray(im)

    worlds = [16, 45]
    box = [clip(F / "stage1_zero_logged.rollouts.npz", w) for w in worlds]
    sph = [clip(F / "stage1_zero_spheres4_logged.rollouts.npz", w) for w in worlds]
    T = max(len(c) for c in box + sph) + 10
    frames = []
    for t in range(T):
        top = np.concatenate([label(c[min(t, len(c) - 1)], f"box pads, start {w}") for c, w in zip(box, worlds)], 1)
        bot = np.concatenate([label(c[min(t, len(c) - 1)], f"sphere pads, start {w}") for c, w in zip(sph, worlds)], 1)
        frames.append(np.concatenate([top, bot], 0))
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    raw = tmp / "raw.mp4"
    imageio.mimwrite(raw, frames, fps=10, macro_block_size=1)
    out = OUT / "videos" / "slip_vs_spheres.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "24", "-movflags", "+faststart", str(out)], check=True)
    Image.fromarray(frames[min(len(frames) - 1, 12)]).save(OUT / "images" / "slip_vs_spheres_poster.jpg", quality=85)
    for w in worlds:
        shutil.copy(F / "viz" / f"zero_w{w}_sheet.png", OUT / "images" / f"slip_box_w{w}_sheet.png")
    print("wrote", out.relative_to(ROOT))


if __name__ == "__main__":
    main()
