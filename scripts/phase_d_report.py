"""Phase D (scale study) grid from training logs and D_eval parquets.

python scripts/phase_d_report.py  -> results/D/D-grid.csv, D-heatmap.png, D-failures.png
Cells without an eval yet show the final training success (marked "train" in the CSV).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blind_insertion.analysis import stats as S  # noqa: E402

OUT = ROOT / "results" / "D"
SIZES = [60, 30, 15, 8]
CLEARS = ["1", "0.5", "0.2", "0.1", "0.05"]
REGIMES = ["scaled", "fixed"]
FAILS = ["never_found", "wedged_partial", "jammed_inside", "captured_then_lost", "slipped", "left_fixture", "timeout"]


def train_info(run: Path) -> dict:
    d = pd.read_csv(run / "train_log.csv")
    hit = d[d.success >= 0.9].env_steps
    tail = d.tail(20)
    return dict(train_steps_M=d.env_steps.iloc[-1] / 1e6, train_success=float(tail.success.mean()),
                steps90_M=hit.iloc[0] / 1e6 if len(hit) else np.nan, wall_h=d.wall.iloc[-1] / 3600,
                done=(run / "DONE").exists())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in sorted((ROOT / "runs").glob("D-peg-proprio-o*_c*_*-s*")):
        m = re.match(r"D-peg-proprio-o(\d+)_c([\d.]+)_(scaled|fixed)(_sph)?-s(\d+)$", run.name)
        if not m or not (run / "train_log.csv").exists():
            continue
        o, c, reg, sph, seed = int(m[1]), m[2], m[3], m[4] or "", int(m[5])
        if o <= 15 and not sph:      # box-pad small cells: grasp slips (superseded by *_sph)
            continue
        r = dict(opening_mm=o, clearance_mm=c, regime=reg, pads="sphere" if sph else "box", seed=seed,
                 **train_info(run))
        ev = OUT / f"eval_o{o}_c{c}_{reg}{sph}_s{seed}.parquet"
        if ev.exists():
            d = pd.read_parquet(ev)
            k, n = int(d.success.sum()), len(d)
            r.update(source="eval", k=k, n=n, success=k / n, lo=S.wilson(k, n)[0], hi=S.wilson(k, n)[1],
                     time_med=float(d.time[d.success.astype(bool)].median()) if k else np.nan,
                     travel_med_mm=1000 * float(d.travel.median()))
            vc = d.failure[~d.success.astype(bool)].value_counts()
            r.update({f"fail_{f}": int(vc.get(f, 0)) for f in FAILS})
        else:
            r.update(source="train", success=r["train_success"])
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "D-grid.csv", index=False)
    print(df[["opening_mm", "clearance_mm", "regime", "source", "success", "steps90_M", "train_steps_M", "wall_h"]]
          .to_string(index=False))

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for ax, reg in zip(axs, REGIMES):
        g = np.full((len(SIZES), len(CLEARS)), np.nan)
        lab = [["" for _ in CLEARS] for _ in SIZES]
        for _, r in df[df.regime == reg].iterrows():
            i, j = SIZES.index(r.opening_mm), CLEARS.index(r.clearance_mm)
            g[i, j] = 100 * r.success
            lab[i][j] = (f"{g[i, j]:.1f}\nn={r.n}" if r.source == "eval"
                         else f"{g[i, j]:.0f}*" + ("" if r.done else "\nrunning"))
        ax.imshow(g, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        for i in range(len(SIZES)):
            for j in range(len(CLEARS)):
                v = g[i, j]
                ax.text(j, i, lab[i][j] or "·", ha="center", va="center", fontsize=8,
                        color="black" if v == v and v > 60 else "white" if v == v else "0.5")
        ax.set_xticks(range(len(CLEARS)), [f"{c} mm" for c in CLEARS])
        ax.set_yticks(range(len(SIZES)), [f"{s} mm" for s in SIZES])
        ax.set_xlabel("clearance per side")
        ax.set_ylabel("opening (long side)")
        ax.set_title(f"{reg} start offset" + (" (∝ size)" if reg == "scaled" else " (3 mm median)"))
    fig.suptitle("Phase D success, YAM, random robot conditions (* = training success, no eval yet)", fontsize=10)
    fig.savefig(OUT / "D-heatmap.png", dpi=150)

    ev = df[df.source == "eval"]
    if len(ev):
        ev = ev.sort_values(["regime", "opening_mm", "clearance_mm"], ascending=[True, False, False])
        fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
        x = np.arange(len(ev))
        bot = np.zeros(len(ev))
        for f in FAILS:
            v = ev[f"fail_{f}"].fillna(0).values / ev.n.values
            ax.bar(x, 100 * v, bottom=100 * bot, label=f.replace("_", " "))
            bot += v
        ax.set_xticks(x, [f"{r.opening_mm}/{r.clearance_mm}\n{r.regime}" for r in ev.itertuples()], fontsize=7)
        ax.set_ylabel("% of episodes failed")
        ax.legend(fontsize=7, ncol=4)
        ax.set_title("Phase D failure modes (opening mm / clearance mm)")
        fig.savefig(OUT / "D-failures.png", dpi=150)


if __name__ == "__main__":
    main()
