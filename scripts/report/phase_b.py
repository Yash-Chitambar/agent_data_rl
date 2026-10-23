"""Phase B figures and tables from frozen run logs and eval parquets (whatever exists).

python scripts/report/phase_b.py   -> results/B/*.png, results/B/*.csv, results/B/summary.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.analysis import stats as S  # noqa: E402

OUT = ROOT / "results" / "B"
BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#999999"
PAPER = dict(success=(99.4, 97.2), time=(0.80, 0.44), rate=(65, 82), travel=(37, 20),
             t_partial=(0.38, 0.22), tilt_partial=(6.7, 5.2), tilt_final=(1.1, 1.1))
PAPER_OOD = {"tighter": (93.8, 54.1), "slender": (97.1, 62.7), "very_slender": (93.3, 40.1),
             "mild_size": (97.1, 57.7), "extreme_size": (91.9, 16.4), "random_convex": (36.2, 3.9)}


def runs(pattern):
    return sorted(p for p in glob.glob(str(ROOT / "runs" / pattern)) if Path(p, "train_log.csv").exists())


def curve(run, smooth=10):
    d = pd.read_csv(Path(run, "train_log.csv"))
    # success over a rolling window of episodes, weighted by episode count
    k = d.success.fillna(0) * d.episodes
    s = k.rolling(smooth, min_periods=1).sum() / d.episodes.rolling(smooth, min_periods=1).sum()
    return d.env_steps.values / 1e6, 100 * s.values


def steps_to(run, thr=90.0):
    x, y = curve(run)
    i = np.argmax(y >= thr)
    return float(x[i]) if (y >= thr).any() else np.nan


def band(ax, rs, color, label):
    if not rs:
        return
    cs = [curve(r) for r in rs]
    L = min(len(c[0]) for c in cs)
    x = cs[0][0][:L]
    Y = np.stack([c[1][:L] for c in cs])
    ax.plot(x, Y.mean(0), color=color, label=f"{label} (n={len(rs)} seeds)")
    if len(rs) > 1:
        ax.fill_between(x, Y.min(0), Y.max(0), color=color, alpha=0.25)


def fig_training(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    blind, vis, noc = runs("B-peg-proprio-base-s*"), runs("B-peg-rgb-base-s*"), runs("B-peg-proprio-nocontact-s*")
    band(axes[0], blind, BLUE, "proprioception (blind)")
    band(axes[0], vis, ORANGE, "wrist RGB (visual)")
    axes[0].set_title("B-R1 training success (Fig. 8 replica)")
    band(axes[1], blind, BLUE, "with contact supervision")
    band(axes[1], noc, GRAY, "without contact supervision")
    axes[1].set_title("B-R7 contact supervision (Fig. 14 replica)")
    for ax in axes:
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel("training success (%, stochastic actions)")
        ax.set_ylim(0, 100)
        ax.axhline(90, color="k", lw=0.5, ls=":")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "B-R1_R7_training_curves.png", dpi=130)
    plt.close(fig)
    summary["steps_to_90"] = {"blind": [steps_to(r) for r in blind], "visual": [steps_to(r) for r in vis],
                              "no_contact": [steps_to(r) for r in noc]}


def load(pattern):
    fs = sorted(glob.glob(str(OUT / pattern)))
    return {Path(f).stem: pd.read_parquet(f) for f in fs}


def table5(summary):
    blind = load("eval_blind_s*.parquet")
    vis = load("eval_rgb_s*.parquet")
    if not blind:
        return
    rows = []
    for name, dfs in (("blind", blind), ("visual", vis)):
        if not dfs:
            continue
        succ = [100 * d.success.mean() for d in dfs.values()]
        allr = pd.concat(dfs.values())
        ok = allr[allr.success > 0]
        tot_time = allr.time.where(allr.success > 0, 5.0)   # failures at their full 5 s
        rate = 60 * allr.success.sum() / tot_time.sum()
        r = dict(policy=name, seeds=len(dfs), episodes=len(allr), success=np.mean(succ),
                 success_seeds=succ, time=S.bootstrap_median(ok.time)[0], time_ci=S.bootstrap_median(ok.time)[1],
                 rate_per_min=rate, travel_mm=1000 * ok.travel.median(),
                 t_partial=ok.t_partial.median(), tilt_partial=ok.tilt_partial[ok.t_partial > 0].median(),
                 tilt_final=ok.tilt_final.median())
        rows.append(r)
    t5 = pd.DataFrame(rows)
    t5.to_csv(OUT / "B-R2_table5.csv", index=False)
    summary["table5"] = rows
    # matched starts where both seed-0 policies succeed (paired)
    if "eval_blind_s0" in blind and "eval_rgb_s0" in vis:
        a, b = blind["eval_blind_s0"], vis["eval_rgb_s0"]
        both = (a.success > 0) & (b.success > 0)
        summary["paired_blind_vs_visual"] = {
            k: S.paired(a[k][both].values, b[k][both].values) for k in ("time", "travel", "t_partial")}
        summary["paired_blind_vs_visual"]["n_both"] = int(both.sum())


def fig_travel(summary):
    blind = load("eval_blind_s0.parquet")
    spiral = load("eval_spiral.parquet")
    vis = load("eval_rgb_s0.parquet")
    if not blind or not spiral:
        return
    a, s = blind["eval_blind_s0"], spiral["eval_spiral"]
    both = (a.success > 0) & (s.success > 0)
    dfs = [("blind", a, BLUE), ("spiral (scripted, optimistic)", s, GRAY)]
    if vis:
        v = vis["eval_rgb_s0"]
        both &= v.success > 0
        dfs.append(("visual", v, ORANGE))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bins = np.geomspace(3, 1000, 50)
    for name, d, c in dfs:
        x = 1000 * d.travel[both]
        ax.hist(x, bins=bins, alpha=0.5, color=c, label=f"{name}: median {x.median():.0f} mm")
        ax.axvline(x.median(), color=c, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("lateral distance travelled before seating (mm, log)")
    ax.set_ylabel("episodes")
    ax.set_title(f"B-R3 lateral travel, {int(both.sum())} matched starts where all succeed")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "B-R3_travel_hist.png", dpi=130)
    plt.close(fig)
    summary["B-R3"] = S.paired(a.travel[both].values, s.travel[both].values)
    summary["B-R3"]["blind_median_mm"] = float(1000 * a.travel[both].median())
    summary["B-R3"]["spiral_median_mm"] = float(1000 * s.travel[both].median())
    summary["B-R3"]["spiral_success"] = float(s.success.mean())


def fig_ood(summary):
    bins = ["in_dist", "tighter", "slender", "very_slender", "mild_size", "extreme_size", "random_convex"]
    res = {}
    for pol in ("blind", "rgb"):
        for b in bins:
            fs = sorted(f for f in glob.glob(str(OUT / f"ood_{pol}_s*_{b}.parquet"))
                        if Path(f).stem.split("_", 3)[3] == b)   # "slender" must not match "very_slender"
            if fs:
                ks = [pd.read_parquet(f).success for f in fs]
                res[(pol, b)] = (np.mean([100 * k.mean() for k in ks]), sum(k.sum() for k in ks),
                                 sum(len(k) for k in ks), len(fs))
    if not res:
        return
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(bins))
    for off, pol, c, lab in ((-0.2, "blind", BLUE, "proprioception (blind)"), (0.2, "rgb", ORANGE, "RGB (visual)")):
        vals, lo, hi = [], [], []
        for b in bins:
            if (pol, b) in res:
                m, k, n, _ = res[(pol, b)]
                l_, h_ = S.wilson(k, n)
                vals.append(m)
                lo.append(m - 100 * l_)
                hi.append(100 * h_ - m)
            else:
                vals.append(np.nan)
                lo.append(0)
                hi.append(0)
        ax.bar(x + off, vals, 0.4, yerr=[lo, hi], color=c, label=lab, capsize=2)
    for i, b in enumerate(bins):
        if b in PAPER_OOD:
            ax.plot([i - 0.4, i], [PAPER_OOD[b][0]] * 2, "k--", lw=1)
            ax.plot([i, i + 0.4], [PAPER_OOD[b][1]] * 2, "k:", lw=1)
    ax.set_xticks(x, bins, rotation=15)
    ax.set_ylabel("success (%)")
    ax.set_ylim(0, 105)
    ax.set_title("B-R6 unseen parts (Fig. 11 peg row); dashed/dotted = paper blind/visual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "B-R6_ood.png", dpi=130)
    plt.close(fig)
    summary["ood"] = {f"{p}/{b}": dict(success=v[0], k=int(v[1]), n=int(v[2]), seeds=v[3]) for (p, b), v in res.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    fig_training(summary)
    table5(summary)
    fig_travel(summary)
    fig_ood(summary)
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
