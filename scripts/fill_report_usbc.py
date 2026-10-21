"""Fill the report's USB-C robustness and variant cells (data-ft / data-ft0 attributes) from
results/E/robust/*.parquet and the latch fine-tune eval. Cells without data keep their text.

python scripts/fill_report_usbc.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "E" / "robust"
P = ROOT / "report" / "index.html"


def rate(path):
    if not path.exists():
        return None
    return f"{100 * pd.read_parquet(path).success.mean():.1f}%"


def main():
    s = P.read_text()
    vals = {}
    for attr, tag in (("data-ft0", "sph_s0"), ("data-ft", "noiserandft_s0")):
        for cond in ("base", "lat1", "lat2", "bl1", "bl3", "qn5e-4", "quant14"):
            v = rate(R / f"{tag}_{cond}.parquet")
            if v:
                vals[(attr, cond)] = v
    v = rate(R / "latchft_s0_on_latch_base.parquet")
    if v:
        vals[("data-ft", "latch")] = v
    n = 0
    for (attr, cond), v in vals.items():
        s, k = re.subn(rf'(<td class="n" {attr}="{re.escape(cond)}">)[^<]*(</td>)', rf"\g<1>{v}\g<2>", s)
        n += k
    P.write_text(s)
    print(f"filled {n} cells", vals)


if __name__ == "__main__":
    main()
