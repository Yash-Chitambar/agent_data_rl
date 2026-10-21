"""Phase C (YAM robustness) train-condition x test-condition matrix from eval_phase_c.sh parquets.

python scripts/phase_c_report.py  -> results/C/C-matrix.csv, C-matrix.png, C-summary.json
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blind_insertion.analysis import stats as S  # noqa: E402

OUT = ROOT / "results" / "C"
TRAIN = ["base", "qnoise1e-4", "qnoise5e-4", "quant16", "quant14", "backlash1", "backlash3", "backlash5",
         "latency1", "latency2", "allmid", "allmax", "randall", "gain0.5x", "gain2x", "gain4x"]
TEST = ["base", "qnoise1e-4", "qnoise5e-4", "quant16", "quant14", "backlash1", "backlash3", "backlash5",
        "latency1", "latency2", "gain0.5", "gain2", "gain4", "own"]


def main():
    rows = []
    for f in sorted(glob.glob(str(OUT / "eval_yam_*_s0_*.parquet"))):
        m = re.match(r"eval_yam_(.+)_s0_(.+)\.parquet", Path(f).name)
        d = pd.read_parquet(f, columns=["success"])
        k, n = int(d.success.sum()), len(d)
        lo, hi = S.wilson(k, n)
        rows.append(dict(train=m[1], test=m[2], k=k, n=n, success=k / n, lo=lo, hi=hi))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "C-matrix.csv", index=False)
    t = df.pivot(index="train", columns="test", values="success")
    t = t.reindex(index=[r for r in TRAIN if r in t.index], columns=[c for c in TEST if c in t.columns])

    fig, ax = plt.subplots(figsize=(10, 0.42 * len(t) + 1.6))
    im = ax.imshow(t.values * 100, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j] * 100
            if v == v:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="black" if v > 60 else "white")
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(t.shape[0]), t.index, fontsize=8)
    ax.set_xlabel("test condition")
    ax.set_ylabel("training condition (seed 0)")
    n = int(df.n.max()) if len(df) else 0
    ax.set_title(f"YAM peg success (%), lab-table mount, n = {n} unseen starts per cell", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03, label="success (%)")
    fig.tight_layout()
    fig.savefig(OUT / "C-matrix.png", dpi=150)

    ood = {}
    for f in sorted(glob.glob(str(OUT / "ood_yam_*.parquet"))):
        m = re.match(r"ood_yam_(.+)_s0_(.+)\.parquet", Path(f).name)
        d = pd.read_parquet(f, columns=["success"])
        ood.setdefault(m[1], {})[m[2]] = dict(k=int(d.success.sum()), n=len(d))
    summary = dict(matrix={r: {c: round(float(v), 4) for c, v in t.loc[r].dropna().items()} for r in t.index},
                   ood=ood)
    (OUT / "C-summary.json").write_text(json.dumps(summary, indent=1))
    print(f"{len(df)} cells, {len(t)} training conditions -> {OUT}/C-matrix.{{csv,png}}, C-summary.json")


if __name__ == "__main__":
    main()
