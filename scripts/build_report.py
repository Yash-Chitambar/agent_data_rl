"""Build the data, images and videos for the HTML results report (report/).

python scripts/build_report.py [--no-media]   -> report/static/{data,images,videos}

Reads only frozen results (results/<phase>/) and training logs (runs/*/train_log.csv).
The figure JSON comes from the same numbers as scripts/phase_b_report.py.
"""
from __future__ import annotations

import argparse
import glob
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

OUT = ROOT / "report" / "static"
RB, RA = ROOT / "results" / "B", ROOT / "results" / "A"
PAPER = dict(success=99.4, time=0.80, travel=37, t_partial=0.38, tilt_partial=6.7, tilt_final=1.1)
PAPER_VIS = dict(success=97.2, time=0.44, travel=20, t_partial=0.22, tilt_partial=5.2, tilt_final=1.1)
PAPER_OOD = {"tighter": (93.8, 54.1), "slender": (97.1, 62.7), "very_slender": (93.3, 40.1),
             "mild_size": (97.1, 57.7), "extreme_size": (91.9, 16.4), "random_convex": (36.2, 3.9)}
BINS = ["tighter", "slender", "very_slender", "mild_size", "extreme_size", "random_convex"]
BIN_LABEL = {"in_dist": "Training range", "tighter": "Tighter fit", "slender": "Slender",
             "very_slender": "Very slender", "mild_size": "Mild size", "extreme_size": "Extreme size",
             "random_convex": "Random convex"}
OBS_NAMES = ([f"q{i + 1}" for i in range(7)] + [f"q̇{i + 1}" for i in range(7)] + ["EE x", "EE y", "EE z"]
             + ["roll", "pitch", "yaw"] + ["v x", "v y", "v z", "ω x", "ω y", "ω z"]
             + [f"a{i + 1} (prev)" for i in range(7)])


def r(x, nd=6):
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return r(x.tolist(), nd)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def write(name, obj):
    p = OUT / "data" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(r(obj), separators=(",", ":")))
    print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.1f} KB)")


def rate(k, n):
    lo, hi = S.wilson(k, n)
    return dict(mean=k / n, lo=lo, hi=hi, k=int(k), n=int(n))


# ---------------------------------------------------------------- training curves
def curve(run, smooth=10, n=160):
    d = pd.read_csv(Path(run, "train_log.csv"))
    k = d.success.fillna(0) * d.episodes
    s = k.rolling(smooth, min_periods=1).sum() / d.episodes.rolling(smooth, min_periods=1).sum()
    x = d.env_steps.values / 1e6
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return x[idx], s.values[idx]


def curves():
    groups = {
        "blind": ("Proprioception (blind)", 0, "B-peg-proprio-base-s*"),
        "visual": ("Wrist RGB (visual)", 1, "B-peg-rgb-base-s*"),
        "nocontact": ("Blind, no contact loss", 2, "B-peg-proprio-nocontact-s*"),
        "yam": ("YAM arm, blind peg", 3, "C-peg-proprio-base-s*"),
        "usbc": ("USB-C on YAM, box pads (stopped)", 4, "E-usbc-proprio-curr-s*"),
        "gear": ("Gear meshing, blind", 5, "G-gear-proprio-base-s*"),
    }
    series = {}
    for key, (label, color, pat) in groups.items():
        runs = sorted(p for p in glob.glob(str(ROOT / "runs" / pat)) if Path(p, "train_log.csv").exists())
        seeds = []
        for run in runs:
            x, y = curve(run)
            final = pd.read_csv(Path(run, "train_log.csv")).env_steps.iloc[-1] / 1e6
            seeds.append(dict(name=Path(run).name, x=x, y=y, steps=final,
                              to90=float(x[np.argmax(y >= 0.9)]) if (y >= 0.9).any() else None))
        if seeds:
            series[key] = dict(label=label, color=color, seeds=seeds)
    write("curves.json", dict(xLabel="environment steps (millions)", yLabel="training success",
                              panels=[dict(title="(a) Blind and visual policies", keys=["blind", "visual"]),
                                      dict(title="(b) Contact supervision", keys=["blind", "nocontact"]),
                                      dict(title="(c) Other robot and tasks", keys=["yam", "usbc", "gear"])],
                              series=series))
    return series


# ---------------------------------------------------------------- headline table + bars
def evals():
    pols = {"blind": RB / "eval_blind_s0.parquet", "visual": RB / "eval_rgb_s0.parquet",
            "spiral": RB / "eval_spiral.parquet", "random": RA / "baseline_random.parquet",
            "zero": RA / "baseline_zero.parquet"}
    dfs = {k: pd.read_parquet(p) for k, p in pols.items() if p.exists()}
    table = {}
    for k, d in dfs.items():
        ok = d[d.success > 0]
        t_med, t_ci = S.bootstrap_median(ok.time) if len(ok) else (np.nan, (np.nan, np.nan))
        row = dict(success=rate(d.success.sum(), len(d)), time=t_med, time_ci=t_ci,
                   travel_mm=1000 * ok.travel.median() if len(ok) else None)
        if "t_partial" in d and len(ok):
            row.update(t_partial=ok.t_partial.median(), tilt_partial=ok.tilt_partial[ok.t_partial > 0].median(),
                       tilt_final=ok.tilt_final.median())
            tot = d.time.where(d.success > 0, 5.0).sum()
            row["per_min"] = 60 * d.success.sum() / tot
        if "failure" in d:
            row["failures"] = d.failure.value_counts().drop("success", errors="ignore").to_dict()
        table[k] = row
    # OOD
    ood = {}
    for pol, tag in (("blind", "blind"), ("visual", "rgb")):
        res = {}
        if pol in dfs:
            res["in_dist"] = rate(dfs[pol].success.sum(), len(dfs[pol]))
        for b in BINS:
            f = RB / f"ood_{tag}_s0_{b}.parquet"
            if f.exists():
                s = pd.read_parquet(f).success
                res[b] = rate(s.sum(), len(s))
        if res:
            ood[pol] = res
    write("results.json", dict(table=table, paper=PAPER, paper_visual=PAPER_VIS, ood=ood,
                               paper_ood=PAPER_OOD, bins=["in_dist"] + BINS, bin_label=BIN_LABEL))
    return dfs


def travel(dfs):
    keys = [k for k in ("blind", "visual", "spiral") if k in dfs]
    both = np.ones(len(dfs["blind"]), bool)
    for k in keys:
        both &= dfs[k].success.values > 0
    edges = np.geomspace(3, 1000, 41)
    out = dict(edges=edges, n=int(both.sum()), series={})
    for k in keys:
        x = 1000 * dfs[k].travel.values[both]
        h, _ = np.histogram(np.clip(x, edges[0], edges[-1] * 0.999), edges)
        out["series"][k] = dict(counts=h, median=float(np.median(x)), p10=float(np.percentile(x, 10)),
                                p90=float(np.percentile(x, 90)))
    for k in keys[1:]:
        pr = S.paired(dfs["blind"].travel.values[both], dfs[k].travel.values[both])
        out["series"][k]["vs_blind"] = dict(frac_blind_longer=pr["frac_a_gt_b"], p=pr["wilcoxon_p"],
                                            median_diff_mm=1000 * pr["median_diff"])
    write("travel.json", out)


# ---------------------------------------------------------------- probe
def probe():
    p = json.load(open(RB / "probe_blind_s0.json"))
    modes = ["far", "near", "partial", "moving", "stuck"]   # "lifted" never occurs; "off" has 23 states
    out = dict(n_states=p["n_states"], counts=p["mode_counts"], modes=modes, recall={}, bal={})
    for k in ("latent", "current_obs", "random_encoder"):
        rec = [p[k]["recall"][m] for m in modes]
        out["recall"][k] = rec
        out["bal"][k] = float(np.mean(rec))
        out.setdefault("bal_reported", {})[k] = p[k]["balanced_acc"]
    out["swap"] = p["swap_ratio"]
    out["paper"] = dict(latent=0.70, current_obs=0.56, random_encoder=0.33)
    write("probe.json", out)


# ---------------------------------------------------------------- rollouts: paths, heatmap, video
def rollouts(media=True):
    from blind_insertion.assets import builder as B  # noqa: PLC0415

    files = {"blind": RB / "eval_blind_s0.rollouts.npz", "visual": RB / "eval_rgb_s0.rollouts.npz",
             "spiral": RB / "eval_spiral.rollouts.npz"}
    Z = {k: np.load(f) for k, f in files.items() if f.exists()}
    nw = 8
    vs = B.sample_study_pegs(1024, 1000)[:nw]
    m, lay = B.build("flexiv", vs)
    dfs = {k: pd.read_parquet(RB / n) for k, n in (("blind", "eval_blind_s0.parquet"), ("visual", "eval_rgb_s0.parquet"),
                                                     ("spiral", "eval_spiral.parquet")) if (RB / n).exists()}
    worlds = []
    for w in range(nw):
        v = vs[w]
        ent = dict(world=w, hole=1000 * lay.hole_polys[w][:, :2], clearance_mm=1000 * v.clearance,
                   depth_mm=1000 * v.depth, peg_length_mm=1000 * v.peg_length, paths={})
        for k, z in Z.items():
            if f"w{w}_peg_p" not in z.files:
                continue
            p = z[f"w{w}_peg_p"]
            lab = z[f"w{w}_labels"]
            row = dfs[k].iloc[w] if k in dfs else None
            ent["paths"][k] = dict(xy=1000 * p[:, :2], z=1000 * p[:, 2], inside=(lab == 2).mean(1),
                                   contact=(lab >= 1).mean(1),
                                   success=bool(row.success > 0) if row is not None else None,
                                   time=float(row.time) if row is not None else None)
        worlds.append(ent)
    write("paths.json", dict(dt=0.02, worlds=worlds))

    # heatmap: the blind policy's 33-d observation over its longest logged success
    zb = Z["blind"]
    lens = [len(zb[f"w{w}_hist_obs"]) for w in range(nw)]
    allobs = np.concatenate([zb[f"w{w}_hist_obs"] for w in range(nw)])
    mu, sd = allobs.mean(0), allobs.std(0) + 1e-6
    w = int(np.argmax(lens))
    obs = (zb[f"w{w}_hist_obs"] - mu) / sd
    lab = zb[f"w{w}_labels"]
    write("heatmap.json", dict(world=w, features=OBS_NAMES, window=30, dt=0.02, frames=obs,
                               signal=dict(label="inside", values=(lab == 2).mean(1)),
                               signal2=dict(label="contact", values=(lab >= 1).mean(1))))
    if media:
        video(m, lay, Z, vs)


def video(m, lay, Z, vs):
    import imageio.v2 as imageio  # noqa: PLC0415
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415

    from blind_insertion.viz.replay import render_rollout  # noqa: PLC0415

    tmp = ROOT / "report" / ".build"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    def frames_of(k, w, size=(480, 360)):
        z = Z[k]
        roll = {kk.split("_", 1)[1]: z[kk] for kk in z.files if kk.startswith(f"w{w}_")}
        return render_rollout(m, lay, w, roll, tmp / f"{k}_w{w}", size=size)

    def label(fr, text):
        im = Image.fromarray(fr)
        d = ImageDraw.Draw(im)
        d.rectangle([8, 8, 16 + d.textlength(text, font=font), 34], fill=(255, 255, 255))
        d.text((12, 10), text, fill=(20, 20, 20), font=font)
        return np.asarray(im)

    def tile(clips, labels, ncol, hold=20):
        T = max(len(c) for c in clips) + hold
        out = []
        for t in range(T):
            fs = [label(c[min(t, len(c) - 1)], lb) for c, lb in zip(clips, labels)]
            rows = [np.concatenate(fs[i:i + ncol], 1) for i in range(0, len(fs), ncol)]
            out.append(np.concatenate(rows, 0))
        return out

    def save(frames, name, fps):
        p = OUT / "videos" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = tmp / ("raw_" + name)
        imageio.mimwrite(raw, frames, fps=fps, macro_block_size=1)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "24", "-movflags", "+faststart", str(p)], check=True)
        Image.fromarray(frames[len(frames) // 2]).save(OUT / "images" / (p.stem + "_poster.jpg"), quality=85)
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.0f} KB)")

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    ws = [0, 1, 2, 3]
    clips = [frames_of("blind", w) for w in ws]
    save(tile(clips, [f"start {i + 1}" for i in range(len(ws))], 2), "teaser.mp4", 12.5)
    # same start, three policies
    keys = [k for k in ("blind", "visual", "spiral") if k in Z]
    names = {"blind": "blind", "visual": "visual (RGB)", "spiral": "spiral (privileged)"}
    clips = [frames_of(k, 0) for k in keys]
    save(tile(clips, [names[k] for k in keys], len(keys)), "compare_w0.mp4", 12.5)


# ---------------------------------------------------------------- static images + Phase A numbers
def images():
    copy = {
        "A-S2_drop_study.png": RA, "A-S3_step_flexiv.png": RA, "A-S4_gravity_flexiv.png": RA,
        "A-S5_contact_signal_flexiv.png": RA, "A-S6_dropin_flexiv.png": RA, "A-S6_dropin_yam_stiffpads.png": RA,
        "A-S1_parity.png": RA, "wrist_gpu_nv4.png": RA, "A-S9_throughput.png": RA, "yam_reset_and_0.5s.png": ROOT / "results" / "C",
        "usbc_yam_check.png": ROOT / "results" / "E",
    }
    for f, d in copy.items():
        if (d / f).exists():
            shutil.copy(d / f, OUT / "images" / f)
    thr = json.load(open(RA / "A-S9_throughput.json"))
    write("phase_a.json", dict(throughput=thr,
                               s5={k: json.load(open(RA / f"A-S5_contact_signal_{k}.json")) for k in ("flexiv", "yam")},
                               s6={k: json.load(open(RA / f"A-S6_dropin_{k}.json")) for k in ("flexiv", "yam_stiffpads")}))


def gear():
    RG = ROOT / "results" / "G"
    if not RG.exists():
        return
    for f in ("G-geometry.png", "G-starts.png", "G-drop.png", "G-anchor_spin.png", "G-backlash.png",
              "G-training_curve.png"):
        if (RG / f).exists():
            shutil.copy(RG / f, OUT / "images" / f)
    out = dict(drop=json.load(open(RG / "G-drop.json")), backlash=json.load(open(RG / "G-backlash.json")),
               scripted=json.load(open(RG / "G-scripted_privileged.json")))
    for k in ("zero", "random"):
        f = RG / f"baseline_{k}.parquet"
        if f.exists():
            s = pd.read_parquet(f).success
            out[k] = rate(s.sum(), len(s))
    write("gear.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-media", action="store_true")
    a = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    curves()
    dfs = evals()
    travel(dfs)
    probe()
    images()
    gear()
    rollouts(media=not a.no_media)


if __name__ == "__main__":
    main()
