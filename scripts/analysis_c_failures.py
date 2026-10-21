"""Phase C failure analysis: why latency, backlash and stiffer gains break the YAM peg policy.

Inputs: results/C/eval_yam_<train>_s0_<test>.parquet (2,048 matched starts per cell) and, where they
exist, results/C/examples/<train>_on_<test>.{parquet,rollouts.npz} (256 matched starts, worlds 0-63
logged per control step; scripts/yam_example_rollouts.sh).

python scripts/analysis_c_failures.py  -> results/C/analysis/*.png, summary.md, metrics.json
"""
from __future__ import annotations

import json
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

C = ROOT / "results" / "C"
EX = C / "examples"
OUT = C / "analysis"
DT = 0.02                                  # control step (s)
CLIP = 3.0                                 # EnvCfg.action_clip; logged actions are pre-clip
# Okabe-Ito
BLUE, ORANGE, GREEN, RED, PURPLE, GREY, SKY, YELLOW = ("#0072B2", "#E69F00", "#009E73", "#D55E00",
                                                       "#CC79A7", "#999999", "#56B4E9", "#F0E442")
FAIL_ORDER = ["success", "never_found", "wedged_partial", "jammed_inside", "left_fixture", "slipped",
              "captured_then_lost"]
FAIL_COL = dict(zip(FAIL_ORDER, [GREEN, GREY, ORANGE, YELLOW, PURPLE, RED, SKY]))
# (factor, test condition, policy trained with it)
FACTORS = [("latency", "latency1", "latency1"), ("backlash", "backlash3", "backlash3"),
           ("gain 4x", "gain4", "gain4x")]


def cell(train, test):
    p = C / f"eval_yam_{train}_s0_{test}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def rate(d):
    k, n = int(d.success.sum()), len(d)
    lo, hi = S.wilson(k, n)
    return dict(k=k, n=n, p=k / n, lo=lo, hi=hi)


# --------------------------------------------------------------------------- 2,048-start cells
def failure_bars(m):
    rows = [("base", "base"), ("base", "latency1"), ("latency1", "latency1"), ("latency1", "base"),
            ("latency2", "base"), ("base", "backlash1"), ("base", "backlash3"), ("base", "backlash5"),
            ("backlash3", "backlash3"), ("base", "gain4"), ("gain4x", "gain4"), ("gain4x", "base")]
    rows = [r for r in rows if cell(*r) is not None]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.4))
    for i, (tr, te) in enumerate(rows):
        d = cell(tr, te)
        vc = d.failure.value_counts(normalize=True)
        x = 0.0
        for f in FAIL_ORDER:
            v = float(vc.get(f, 0.0))
            ax.barh(i, v * 100, left=x * 100, color=FAIL_COL[f], edgecolor="white", lw=0.5,
                    label=f.replace("_", " ") if i == 0 else None)
            if v > 0.06:
                ax.text((x + v / 2) * 100, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if f in ("never_found", "left_fixture", "slipped") else "black")
            x += v
        m["failures"][f"{tr}@{te}"] = {k: int(v) for k, v in d.failure.value_counts().items()}
    ax.set_yticks(range(len(rows)), [f"trained {tr}  /  tested {te}" for tr, te in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("episodes (%)")
    ax.set_title("YAM peg: outcome of each episode (n = 2,048 matched unseen starts per row)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAIL_COL[f]) for f in FAIL_ORDER]
    ax.legend(handles, [f.replace("_", " ") for f in FAIL_ORDER], ncol=4, fontsize=7, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.06 - 0.3 / len(rows)))
    fig.tight_layout()
    fig.savefig(OUT / "failure_modes.png", dpi=160)
    plt.close(fig)


def by_distance(m):
    bins = np.array([0, 4, 6, 8, 10, 12, 14, 18]) / 1000
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (name, test, trained) in zip(axes, FACTORS):
        curves = [("base policy, no factor", "base", "base", GREY, "--"),
                  (f"base policy, {test}", "base", test, RED, "-"),
                  (f"{trained} policy, {test}", trained, test, BLUE, "-")]
        for lab, tr, te, col, ls in curves:
            d = cell(tr, te)
            if d is None:
                continue
            b = np.digitize(d.start_dist, bins) - 1
            xs, ys, los, his = [], [], [], []
            for i in range(len(bins) - 1):
                s = d.success[b == i]
                if len(s) < 20:
                    continue
                lo, hi = S.wilson(int(s.sum()), len(s))
                xs.append((bins[i] + bins[i + 1]) / 2 * 1000)
                ys.append(s.mean() * 100)
                los.append(lo * 100)
                his.append(hi * 100)
            ax.plot(xs, ys, ls, color=col, marker="o", ms=3, label=lab)
            ax.fill_between(xs, los, his, color=col, alpha=0.15, lw=0)
            m["by_distance"][f"{tr}@{te}"] = dict(x_mm=xs, success_pct=[round(y, 1) for y in ys])
        ax.set_title(f"{name} (n = 2,048 per curve)", fontsize=9)
        ax.set_xlabel("start offset from hole centre (mm)")
        ax.set_ylim(0, 102)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    axes[0].set_ylabel("success (%)")
    fig.tight_layout()
    fig.savefig(OUT / "success_vs_offset.png", dpi=160)
    plt.close(fig)


def latency_grid(m):
    """3 x 3 train-latency x test-latency success, from the 2,048-start cells."""
    trains = [("base", 0), ("latency1", 1), ("latency2", 2)]
    tests = [("base", 0), ("latency1", 1), ("latency2", 2)]
    g = np.full((3, 3), np.nan)
    for i, (tr, _) in enumerate(trains):
        for j, (te, _) in enumerate(tests):
            d = cell(tr, te)
            if d is not None:
                g[i, j] = d.success.mean() * 100
    m["latency_grid"] = g.round(1).tolist()
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.imshow(g, cmap="viridis", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{g[i, j]:.0f}", ha="center", va="center", fontsize=11,
                    color="black" if g[i, j] > 60 else "white")
    ax.set_xticks(range(3), ["0 ms", "20 ms", "40 ms"])
    ax.set_yticks(range(3), ["0 ms", "20 ms", "40 ms"])
    ax.set_xlabel("latency at test")
    ax.set_ylabel("latency in training")
    ax.set_title("success (%), n = 2,048 each", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "latency_grid.png", dpi=160)
    plt.close(fig)


def cell_metrics(m):
    out = {}
    for tr, te in [("base", "base"), ("base", "latency1"), ("latency1", "latency1"), ("latency1", "base"),
                   ("base", "backlash3"), ("backlash3", "backlash3"), ("base", "gain4"), ("gain4x", "gain4")]:
        d = cell(tr, te)
        if d is None:
            continue
        ok = d[d.success > 0]
        out[f"{tr}@{te}"] = dict(**{k: round(v, 4) for k, v in rate(d).items()},
                                 median_travel_mm_all=round(float(d.travel.median() * 1000), 1),
                                 median_travel_mm_success=round(float(ok.travel.median() * 1000), 1) if len(ok) else None,
                                 median_time_s_success=round(float(ok.time.median()), 2) if len(ok) else None,
                                 captured=round(float(d.captured.mean()), 3),
                                 median_tilt_final_deg=round(float(d.tilt_final.median()), 1))
    m["cells"] = out


# --------------------------------------------------------------------------- logged rollouts
def load_ex(tag):
    p, z = EX / f"{tag}.parquet", EX / f"{tag}.rollouts.npz"
    if not (p.exists() and z.exists()):
        return None, None
    d = pd.read_parquet(p)
    zz = np.load(z)
    worlds = sorted({int(k.split("_")[0][1:]) for k in zz.files})
    rolls = {w: {k.split("_", 1)[1]: zz[k] for k in zz.files if k.startswith(f"w{w}_")} for w in worlds}
    return d, rolls


def action_stats(rolls, window=25):
    """Per episode, over the first `window` steps (0.5 s): fraction of raw action entries beyond the
    clip (saturated), RMS of the clipped action, lag-1 autocorrelation of the
    action (pooled over joints), fraction of steps where a joint's action changes sign, peak arm
    joint speed (rad/s) and peak peg lateral speed (mm/s)."""
    rows = []
    for w, r in rolls.items():
        raw = r["action"][:window]
        a = np.clip(raw, -CLIP, CLIP)
        if len(a) < 4:
            continue
        ac = [np.corrcoef(a[:-1, j], a[1:, j])[0, 1] for j in range(a.shape[1]) if a[:, j].std() > 1e-6]
        flips = (np.sign(a[1:]) != np.sign(a[:-1])).mean()
        qd = np.abs(np.diff(r["qpos"][:window + 1, :6], axis=0)) / DT
        pv = np.linalg.norm(np.diff(r["peg_p"][:window + 1, :2], axis=0), axis=1) / DT * 1000
        rows.append(dict(world=w, sat=float((np.abs(raw) > CLIP).mean()), rms=float(np.sqrt((a ** 2).mean())), ac1=float(np.nanmean(ac)),
                         flips=float(flips), qd_peak=float(qd.max()), peg_v_peak=float(pv.max()), T=len(r["action"])))
    return pd.DataFrame(rows)


def rollout_figs(m):
    pairs = [("latency1", [("base_on_base", "base policy, 0 ms", GREY), ("base_on_latency1", "base policy, 20 ms", RED),
                           ("latency1_on_latency1", "latency-trained, 20 ms", BLUE)]),
             ("backlash3", [("base_on_base", "base policy, no backlash", GREY),
                            ("base_on_backlash3", "base policy, 3 arcmin", RED),
                            ("backlash3_on_backlash3", "backlash-trained, 3 arcmin", BLUE)]),
             ("gain4", [("base_on_base", "base policy, 1x gains", GREY), ("base_on_gain4", "base policy, 4x gains", RED),
                        ("gain4x_on_gain4", "4x-trained, 4x gains", BLUE)])]
    stats = {}
    for _, lst in pairs:
        for tag, _, _ in lst:
            if tag not in stats:
                d, r = load_ex(tag)
                if d is not None:
                    st = action_stats(r)
                    st = st.merge(d[["world", "success", "failure"]], on="world", how="left")
                    stats[tag] = st
    m["rollout_stats"] = {t: {k: round(float(s[k].median()), 3) for k in ("sat", "rms", "ac1", "flips", "qd_peak", "peg_v_peak")}
                          | dict(n=len(s), success=round(float(s.success.mean()), 3)) for t, s in stats.items()}
    if not stats:
        return
    # (a) action smoothness / oscillation
    metrics = [("ac1", "action lag-1 autocorrelation"), ("flips", "action sign flips per step"),
               ("peg_v_peak", "peak peg lateral speed (mm/s)")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    tags = [t for t in ["base_on_base", "base_on_latency1", "latency1_on_latency1", "base_on_backlash3",
                        "backlash3_on_backlash3", "base_on_gain4", "gain4x_on_gain4"] if t in stats]
    cols = {"base_on_base": GREY, "latency1_on_latency1": BLUE, "backlash3_on_backlash3": BLUE,
            "gain4x_on_gain4": BLUE}
    for ax, (k, lab) in zip(axes, metrics):
        data = [stats[t][k].dropna().values for t in tags]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.6, patch_artist=True, showfliers=False)
        for patch, t in zip(bp["boxes"], tags):
            patch.set_facecolor(cols.get(t, RED))
            patch.set_alpha(0.6)
        ax.set_yticks(range(1, len(tags) + 1), [t.replace("_on_", " @ ") for t in tags] if ax is axes[0] else [""] * len(tags),
                      fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.grid(alpha=0.3, axis="x")
    n = min(len(stats[t]) for t in tags)
    fig.suptitle(f"First 0.5 s of each episode (logged worlds, n = {n} matched starts per row; "
                 "red = policy tested on a condition it did not train with)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "action_oscillation.png", dpi=160)
    plt.close(fig)

    # (b) example traces: latency, one world where base@latency1 fails and latency1@latency1 succeeds
    for fac, lst in pairs:
        have = [x for x in lst if x[0] in stats]
        if len(have) < 3:
            continue
        bad, good = stats[lst[1][0]], stats[lst[2][0]]
        cand = sorted(set(bad.world[bad.success == 0]) & set(good.world[good.success > 0]))
        if not cand:
            continue
        _, r0 = load_ex(lst[0][0])
        _, r1 = load_ex(lst[1][0])
        _, r2 = load_ex(lst[2][0])
        w = cand[0]
        m.setdefault("example_world", {})[fac] = w
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for (tag, lab, col), r in zip(lst, (r0, r1, r2)):
            if w not in r:
                continue
            rr = r[w]
            T = min(len(rr["action"]), 100)
            t = np.arange(T) * DT
            p = (rr["peg_p"][:T, :2] - rr["peg_p"][0, :2]) * 1000
            axes[0].plot(p[:, 0], p[:, 1], color=col, lw=1.5, label=lab)
            axes[0].plot(p[-1, 0], p[-1, 1], "o", color=col, ms=4)
            axes[1].plot(t, np.clip(rr["action"][:T, 0], -CLIP, CLIP), color=col, lw=1.3, label=f"{lab}")
        axes[0].plot(0, 0, "k+", ms=8)
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].set_xlabel("peg x from start (mm)")
        axes[0].set_ylabel("peg y from start (mm)")
        axes[0].set_title(f"peg path, start {w} (first 2 s)", fontsize=9)
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("action, joint 1 (x 0.04 rad)")
        axes[1].set_title("policy action, joint 1 (base yaw)", fontsize=9)
        axes[1].legend(fontsize=7, frameon=False)
        for a in axes:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"example_{fac}.png", dpi=160)
        plt.close(fig)


# --------------------------------------------------------------------------- summary
def summary(m):
    c, f = m["cells"], m["failures"]
    L = []
    pct = lambda key: f"{c[key]['p'] * 100:.1f}% [{c[key]['lo'] * 100:.1f}-{c[key]['hi'] * 100:.1f}]" if key in c else "n/a"
    L.append("# Phase C failure analysis (YAM peg, lab table, seed 0)\n")
    L.append("Generated by `scripts/analysis_c_failures.py`. Cells: 2,048 matched unseen starts, deterministic "
             "actions, 95% Wilson intervals in brackets. Rollout statistics: logged worlds of the 256-start "
             "example runs (`results/C/examples/`), first 0.5 s of each episode.\n")
    L.append("## Success (2,048 starts)\n")
    for k in c:
        v = c[k]
        L.append(f"- `{k}`: {pct(k)}; median travel {v['median_travel_mm_all']} mm (all episodes); "
                 f"captured {v['captured'] * 100:.0f}%")
    L.append("\n## Failure modes (counts of 2,048)\n")
    for k, v in f.items():
        L.append(f"- `{k}`: " + ", ".join(f"{a} {b}" for a, b in sorted(v.items(), key=lambda x: -x[1])))
    L.append("\n## Latency grid (success %, rows = training latency 0/20/40 ms, columns = test latency)\n")
    for row in m.get("latency_grid", []):
        L.append("- " + " / ".join(f"{x:.1f}" for x in row))
    if m.get("rollout_stats"):
        L.append("\n## Action statistics (median over logged episodes)\n")
        L.append("| run | n | success | saturated | RMS action (clipped) | lag-1 autocorr | sign flips / step | peak joint speed (rad/s) | peak peg speed (mm/s) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for t, s in m["rollout_stats"].items():
            L.append(f"| {t} | {s['n']} | {s['success'] * 100:.0f}% | {s['sat'] * 100:.0f}% | {s['rms']:.2f} | {s['ac1']:.2f} | {s['flips']:.2f} | "
                     f"{s['qd_peak']:.2f} | {s['peg_v_peak']:.0f} |")
    (OUT / "summary_auto.md").write_text("\n".join(L) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = dict(failures={}, by_distance={})
    cell_metrics(m)
    failure_bars(m)
    by_distance(m)
    latency_grid(m)
    rollout_figs(m)
    summary(m)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=1, default=float))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
