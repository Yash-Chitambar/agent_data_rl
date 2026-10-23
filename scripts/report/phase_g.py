"""Phase G (gear meshing) figures and tables from run logs and eval parquets (whatever exists).

python scripts/report/phase_g.py  -> results/G/G-training_curve.png, G-success.png, G-summary.json
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

OUT = ROOT / "results" / "G"
BLUE, GRAY = "#0072B2", "#999999"
BINS = ["tighter", "taller", "much_taller", "mild_size", "extreme_size", "random_tooth"]
PAPER = {"in_dist": 99.9, **{b: 99.9 for b in BINS}}   # Table 7 gear row: ">99.9" everywhere


def curve(run, smooth=10):
    d = pd.read_csv(Path(run, "train_log.csv"))
    k = d.success.fillna(0) * d.episodes
    s = k.rolling(smooth, min_periods=1).sum() / d.episodes.rolling(smooth, min_periods=1).sum()
    return d.env_steps.values / 1e6, 100 * s.values, d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    runs = sorted(p for p in glob.glob(str(ROOT / "runs" / "G-gear-proprio-*"))
                  if Path(p, "train_log.csv").exists())
    if runs:
        fig, ax = plt.subplots(figsize=(6, 4))
        for i_, r in enumerate(runs):
            x, y, d = curve(r)
            ax.plot(x, y, color=f"C{i_}", label=f"{Path(r).name}: success")
            ax.plot(x, 100 * d.capture.values, color=f"C{i_}", ls="--", lw=0.8,
                    label=f"{Path(r).name}: shaft captured")
            i = np.argmax(y >= 90)
            summary.setdefault("steps_to_90M", {})[Path(r).name] = float(x[i]) if (y >= 90).any() else None
            summary.setdefault("final_train_success", {})[Path(r).name] = float(y[-1])
            summary.setdefault("sps_median", {})[Path(r).name] = float(d.sps.median())
        ax.set_xlabel("environment steps (millions)")
        ax.set_ylabel("training success (%, stochastic actions)")
        ax.set_ylim(0, 100)
        ax.axhline(90, color="k", lw=0.5, ls=":")
        ax.set_title("Gear meshing, blind (Fig. 8 gear panel replica)\nsuccess: rolling 10 iterations; "
                     "12,288 worlds x 32 steps per iteration", fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "G-training_curve.png", dpi=130)
        plt.close(fig)
    rows = []
    for f in sorted(glob.glob(str(OUT / "eval_*.parquet"))):
        tag = Path(f).stem[5:]
        df = pd.read_parquet(f)
        rows.append(dict(tag=tag, bin="in_dist", df=df))
        for b in BINS:
            g = OUT / f"ood_{tag}_{b}.parquet"
            if g.exists():
                rows.append(dict(tag=tag, bin=b, df=pd.read_parquet(g)))
    tab = []
    for r in rows:
        df = r["df"]
        k, n = int(df.success.sum()), len(df)
        lo, hi = S.wilson(k, n)
        ok = df[df.success > 0]
        tab.append(dict(tag=r["tag"], bin=r["bin"], n=n, success=100 * k / n, lo=100 * lo, hi=100 * hi,
                        median_time_s=float(ok.time.median()) if len(ok) else None,
                        median_travel_mm=1000 * float(ok.travel.median()) if len(ok) else None,
                        paper=PAPER.get(r["bin"]), failures=df.failure.value_counts().to_dict()))
    if tab:
        t = pd.DataFrame(tab)
        t.drop(columns=["failures"]).to_csv(OUT / "G-success.csv", index=False)
        summary["success"] = tab
        for tag, sub in t.groupby("tag"):
            fig, ax = plt.subplots(figsize=(8, 3.8))
            xs = np.arange(len(sub))
            ax.bar(xs, sub.success, color=BLUE, yerr=[sub.success - sub.lo, sub.hi - sub.success], capsize=3,
                   label="ours (blind)")
            ax.scatter(xs, sub.paper, marker="_", s=300, color="k", label="paper (blind, >99.9)")
            ax.set_xticks(xs, [f"{b}\n(n={n})" for b, n in zip(sub.bin, sub.n)], fontsize=7)
            ax.set_ylabel("success (%), 95% Wilson")
            ax.set_ylim(max(0, sub.lo.min() - 5), 100.5)
            ax.set_title(f"Gear meshing success, {tag} (Fig. 11 / Table 7 gear row), deterministic actions",
                         fontsize=9)
            ax.legend(fontsize=7, loc="lower left")
            fig.tight_layout()
            fig.savefig(OUT / f"G-success_{tag}.png", dpi=130)
            plt.close(fig)
    json.dump(summary, open(OUT / "G-summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
