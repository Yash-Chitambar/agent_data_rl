#!/usr/bin/env python3
"""Generate the server job manifests (one shell command per line) for Phases B-F.

    python3 scripts/server/jobs/make_manifests.py            # (re)write scripts/server/jobs/*.txt
    .venv/bin/python scripts/server/jobs/make_manifests.py --check   # also validate every
        python command against the real argparse parsers and EnvCfg fields (imports torch/warp,
        no GPU work)

Lines keep the runs/queue/jobs.txt format; annotations after ' # ' are read by run_jobs.py and
slurm/submit.sh (`after=RUN` dependencies, `visual`, `mem=GB`, `est_h=hours`) and ignored by bash.
Training lines are ordered seed 0 of every condition first, then seed 1, then seed 2, so a
partial pass still gives one seed of everything.

Edit the CONSTANTS below and re-run (e.g. once Gate C fixes the noise model used in D and E).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# ------------------------------------------------------------------------------- constants
PY = ".venv/bin/python"
YAM = "blind_insertion/configs/phase_c_yam.json"
USBC = "blind_insertion/configs/phase_e_usbc_spheres.json"   # sphere pads: box pads drop the drive (report_usbc/)
SEEDS = (0, 1, 2)
# Phase C noise model carried into Phases D and E (plan §10: "with the Phase C noise model on").
# PLACEHOLDER until Gate C picks it from the robustness plots; regenerate D/E after changing.
NOISE = {"q_noise": 1e-4}
MED_PER_HALF = 0.765          # train.MEDIAN_RADIAL_PER_HALFWIDTH: median |xy| = 0.765 a for U(-a,a)^2

# GPU-hour assumptions, from runs on the local RTX 3090 Ti (runs/queue/queue.log):
# blind 300M at 12,288 worlds = 2.2-2.4 h (36-38k env steps/s incl. PPO, Flexiv and YAM);
# visual with the stop rule stopped at 50M = 1.7 h (8.5k steps/s; 300M would be ~9.8 h);
# eval_phase_b.sh (7 evals + probe) = 0.18 h. USB-C (8,192 worlds, njmax 2048) not yet measured.
H_BLIND, H_VISUAL, H_VISUAL_MAX, H_USBC = 2.3, 1.7, 9.8, 3.0
H_EVAL_B, H_EVAL_C, H_OOD, H_EVAL1, H_EVAL_E_FULL = 0.18, 0.35, 0.12, 0.04, 0.35

# Runs that the LOCAL queue (runs/queue/jobs.txt) also contains as of 2026-10-20: remove them
# from one side before launching, or they train twice.
LOCAL_QUEUE = {"B-peg-proprio-base-s2", "B-peg-proprio-nocontact-s1", "E-usbc-proprio-curr-s0"}
# Runs already finished locally (evaluate where their checkpoints are; on a server first
# `sync_up.sh --runs NAME` and `run_jobs.py --mark-complete NAME`).
LOCAL_DONE = {"B-peg-proprio-base-s0", "B-peg-proprio-base-s1", "B-peg-proprio-nocontact-s0",
              "B-peg-rgb-base-s0",
              "C-peg-proprio-base-s0"}     # (C base s0 was still training locally on 2026-10-20)


def jq(d: dict) -> str:
    """JSON for --env_json, single-quoted for bash."""
    return shlex.quote(json.dumps(d, separators=(",", ":")))


def ann(**kw) -> str:
    parts = []
    for k, v in kw.items():
        if v is None or v is False:
            continue
        parts.append(k if v is True else f"{k}={v}")
    return "  # " + " ".join(parts) if parts else ""


def train_line(name, seed, args, *, visual=False, mem=11, est=H_BLIND, nworld=12288):
    mod = "train_visual" if visual else "train"
    cmd = f"{PY} -m blind_insertion.{mod} --name {name} --seed {seed} --nworld {nworld} --steps 300e6 {args}".strip()
    flags = dict(visual=visual, mem=mem, est_h=est)
    if name in LOCAL_QUEUE:
        flags["also_in_local_queue"] = True
    return f"{cmd} > runs/{name}.log 2>&1" + ann(**flags)


def eval_line(cmd, log, after, est, visual=False, mem=3):
    after = after if isinstance(after, (list, tuple)) else [after]
    needs_sync = any(a in LOCAL_DONE for a in after)
    return f"{cmd} > runs/{log}.log 2>&1" + ann(after=",".join(after), visual=visual, mem=mem, est_h=est,
                                                needs_local_ckpt=needs_sync)


def by_seed(conds):
    """conds: list of (variant, extra args, kw) -> lines ordered seed-major."""
    return [(v, s, a, kw) for s in SEEDS for (v, a, kw) in conds]


# ------------------------------------------------------------------------------- Phase B
def phase_b():
    tr, ev = [], []
    conds = [("blind", "base", "", {}), ("nocontact", "nocontact", "--contact_coef 0", {}),
             ("rgb", "base", "--stop_success 0.985 --min_steps 50e6", dict(visual=True, mem=14, est=H_VISUAL,
                                                                            nworld=4096))]
    for s in SEEDS:
        for tag, var, args, kw in conds:
            mod = "rgb" if tag == "rgb" else "proprio"
            name = f"B-peg-{mod}-{var}-s{s}"
            if name not in LOCAL_DONE:
                tr.append(train_line(name, s, args, **kw))
            # evals that do not exist yet in results/B (blind s0 and rgb s0 are done)
            if name in ("B-peg-proprio-base-s0", "B-peg-rgb-base-s0"):
                continue
            probe = "" if tag == "rgb" else " probe"
            ev.append(eval_line(f"scripts/eval_phase_b.sh {name} {tag}_s{s}{probe}", f"eval_{tag}_s{s}", name,
                                H_EVAL_B, visual=tag == "rgb"))
    allb = [f"B-peg-{m}-{v}-s{s}" for s in SEEDS for m, v in (("proprio", "base"), ("proprio", "nocontact"),
                                                              ("rgb", "base"))]
    ev.append(f"{PY} scripts/phase_b_report.py > runs/phase_b_report.log 2>&1"
              + ann(after=",".join(allb), est_h=0.05, note="needs_all_B_train_logs_in_runs"))
    return tr, ev


# ------------------------------------------------------------------------------- Phase C
def gains(g):
    """§5.2 YAM gains x g, in the format scripts/eval_phase_c.sh passes."""
    kp = [80 * g] * 3 + [30 * g] * 3
    kd = [6 * g] * 3 + [2 * g] * 3
    return {"gains": [kp, kd]}


C_CONDS = [  # variant, env_json overrides (training with that factor on, §9.3 / §9.4)
    ("base", {}),
    ("qnoise1e-4", {"q_noise": 1e-4}), ("qnoise5e-4", {"q_noise": 5e-4}),
    ("quant14", {"q_quant_bits": 14}), ("quant16", {"q_quant_bits": 16}),
    ("backlash1", {"backlash_arcmin": 1}), ("backlash3", {"backlash_arcmin": 3}),
    ("backlash5", {"backlash_arcmin": 5}),
    ("latency1", {"latency": 1}), ("latency2", {"latency": 2}),
    ("allmid", {"q_noise": 1e-4, "q_quant_bits": 16, "backlash_arcmin": 1, "latency": 1}),
    ("allmax", {"q_noise": 5e-4, "q_quant_bits": 14, "backlash_arcmin": 5, "latency": 2}),
    ("gain0.5x", gains(0.5)), ("gain2x", gains(2)), ("gain4x", gains(4)),
]


def phase_c():
    tr, ev = [], []
    for s in SEEDS:
        for var, ej in C_CONDS:
            name = f"C-peg-proprio-{var}-s{s}"
            args = f"--env_file {YAM}" + (f" --env_json {jq(ej)}" if ej else "")
            if name not in LOCAL_DONE:
                tr.append(train_line(name, s, args))
            tag = f"yam_{var}_s{s}"
            # full brittleness matrix: base + each factor alone + gain sweep (scripts/eval_phase_c.sh)
            ev.append(eval_line(f"scripts/eval_phase_c.sh {name} {tag}", f"eval_{tag}", name, H_EVAL_C))
            if var.startswith("all"):     # also under its own combined training condition
                ev.append(eval_line(f"{PY} -m blind_insertion.eval --policy runs/{name}/last.pt --env_file {YAM} "
                                    f"--episodes 2048 --env_json {jq(ej)} --out results/C/eval_{tag}_own.parquet",
                                    f"eval_{tag}_own", name, H_EVAL1))
            if var == "base":             # C-R4 OOD bins on the YAM
                ev.append(eval_line(f"scripts/server/eval/eval_ood.sh {name} {tag} C {YAM}", f"ood_{tag}", name,
                                    H_OOD))
    return tr, ev


# ------------------------------------------------------------------------------- Phase D
D_OPENINGS_MM = (60, 30, 15, 8)
D_CLEARANCES_MM = (1.0, 0.5, 0.2, 0.1, 0.05)
D_DIAGONAL = {(60, 1.0), (30, 0.5), (15, 0.2), (8, 0.05)}     # large-loose -> small-tight


def d_offset(opening_mm, regime):
    """Per-axis half-width a (m) giving the plan's median start offset."""
    med_mm = opening_mm / 6.0 if regime == "scaled" else 3.0    # 10 mm at 60 mm / fixed 3 mm
    return round(med_mm / 1000 / MED_PER_HALF, 6)


def phase_d():
    tiers = {1: [], 2: []}
    ev = []
    cells = [(o, c, r) for r in ("scaled", "fixed") for o in D_OPENINGS_MM for c in D_CLEARANCES_MM]
    for s in SEEDS:
        for o, c, r in cells:
            tier = 1 if (o, c) in D_DIAGONAL or o == 8 else 2
            var = f"o{o}_c{c:g}_{r}"
            name = f"D-peg-proprio-{var}-s{s}"
            fam = f"scale:{o / 1000:g}:{c / 1000:g}"
            ej = dict(NOISE, peg_offset=d_offset(o, r))
            tiers[tier].append(train_line(name, s, f"--env_file {YAM} --family {fam} --env_json {jq(ej)}"))
            ev.append(eval_line(f"{PY} -m blind_insertion.eval --policy runs/{name}/last.pt --env_file {YAM} "
                                f"--family {fam} --env_json {jq(ej)} --episodes 2048 "
                                f"--out results/D/eval_{var}_s{s}.parquet", f"eval_{name}", name, H_EVAL1))
    return tiers[1], tiers[2], ev


# ------------------------------------------------------------------------------- Phase E
E_BASE = f"--family usbc --curriculum usbc --n_variants 320 --env_file {USBC}"
E_CONDS = [  # variant, extra train args, eval env_json (must repeat EnvCfg overrides), eval mode
    ("curr", "", {}, "full"),                                    # E1 (+E2/E3 evals)
    ("nocontact", "--contact_coef 0", {}, "indist"),             # E4 ablations
    ("h10", f"--env_json {jq({'history': 10})}", {"history": 10}, "indist"),
    ("h60", f"--env_json {jq({'history': 60})}", {"history": 60}, "indist"),
    # E1 is currently trained WITHOUT the noise model (phase_e_usbc_up.json has none), so the
    # plan's "no noise model" ablation is E1 itself and this is the "with noise" counterpart.
    ("noise", f"--env_json {jq(NOISE)}", NOISE, "full"),
]


def phase_e():
    tr, ev = [], []
    for s in SEEDS:
        for var, extra, ej, mode in E_CONDS:
            name = f"E-usbc-proprio-{var}-s{s}"
            tr.append(train_line(name, s, f"{E_BASE} {extra}".strip(), nworld=8192, mem=16, est=H_USBC))
            tag = f"{var}_s{s}"
            ejs = f" {jq(ej)}" if ej else ""
            ev.append(eval_line(f"scripts/server/eval/eval_phase_e.sh {name} {tag} {mode}{ejs}", f"eval_{name}",
                                name, H_EVAL_E_FULL if mode == "full" else H_EVAL1))
    return tr, ev


# ------------------------------------------------------------------------------- Phase F
F_TEXT = """\
# Phase F (plan §12): BLOCKED. No line below is runnable yet; every one needs a code change
# in blind_insertion/ (see docs/SERVER.md "Code changes needed"):
#   - train_visual.py has no --env_file / --family / --curriculum, so it cannot train on the
#     YAM USB-C task (it only builds the default Flexiv + study-peg env).
#   - only an RGB-only policy exists (policies/visual.py). Depth (EnvCfg.cam_depth renders it,
#     but no policy reads it), proprio+image fusion, and the depth sensor model (min range,
#     noise, dropout) are not implemented.
#   - no lighting/texture/camera-pose randomization (plan §12.1).
# Intended runs once those exist (all visual, need the CUDA ray tracer, not EGL), 3 seeds each:
#   F-usbc-rgb-curr-s{0,1,2}   F-R   (RGB only)
#   F-usbc-depth-curr-s{0,1,2} F-D   (depth only)
#   F-usbc-pr-curr-s{0,1,2}    F-PR  (proprio + RGB)
#   F-usbc-pd-curr-s{0,1,2}    F-PD  (proprio + depth)
#   F-usbc-prd-curr-s{0,1,2}   F-PRD (proprio + RGB + depth)
#   (F-P = E-usbc-proprio-curr-s*, already in E_train.txt)
# Sketch of the F-R line after train_visual.py gains --env_file/--family/--curriculum:
# .venv/bin/python -m blind_insertion.train_visual --name F-usbc-rgb-curr-s0 --seed 0 --nworld 4096 --steps 300e6 --family usbc --curriculum usbc --n_variants 320 --env_file blind_insertion/configs/phase_e_usbc_up.json > runs/F-usbc-rgb-curr-s0.log 2>&1  # visual mem=16 est_h=10
# Estimated cost: 15 runs x ~10 h (8.5k steps/s, 300M; less if a stop rule is used) = ~150 GPU-h,
# plus F2 capture-radius evals out to 15 mm.
"""


HEADER = """\
# {title}
# Generated by scripts/server/jobs/make_manifests.py -- edit that script, not this file.
# {n} jobs, ~{h:.0f} GPU-h (3090 Ti-equivalent; see the assumptions in the generator).
# Run:  python3 scripts/server/run_jobs.py scripts/server/jobs/{fname} [--gpus ...]
#  or:  scripts/server/slurm/submit.sh scripts/server/jobs/{fname} [...]
{extra}"""


def est(lines, key="est_h"):
    t = 0.0
    for ln in lines:
        _, _, a = ln.partition(" # ")
        for tok in a.split():
            if tok.startswith(key + "="):
                t += float(tok.split("=", 1)[1])
    return t


def write(fname, title, lines, extra=""):
    p = HERE / fname
    p.write_text(HEADER.format(title=title, n=len(lines), h=est(lines), fname=fname, extra=extra)
                 + "\n".join(lines) + "\n")
    return fname, len(lines), est(lines)


def check(paths):
    """Validate each python command's flags with the real parsers and EnvCfg fields."""
    import dataclasses  # noqa: PLC0415
    sys.path.insert(0, str(REPO))
    import argparse as ap_mod  # noqa: PLC0415
    from blind_insertion import train as T  # noqa: PLC0415
    from blind_insertion import train_visual as TV  # noqa: PLC0415
    from blind_insertion.envs.peg_env import EnvCfg  # noqa: PLC0415
    fields = {f.name for f in dataclasses.fields(EnvCfg)}

    class Captured(Exception):
        pass

    def eval_parser_args(argv):
        """Run eval.main()'s argparse on argv without running the evaluation."""
        from blind_insertion import eval as EV  # noqa: PLC0415
        orig = ap_mod.ArgumentParser.parse_args
        out = {}

        def fake(self, args=None, namespace=None):
            out["ns"] = orig(self, argv, namespace)
            raise Captured
        ap_mod.ArgumentParser.parse_args = fake
        try:
            EV.main()
        except Captured:
            pass
        finally:
            ap_mod.ArgumentParser.parse_args = orig
        return out["ns"]

    n_ok, errors = 0, []
    for p in paths:
        for ln in Path(p).read_text().splitlines():
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            cmd = ln.partition(" # ")[0]
            toks = shlex.split(cmd)
            if ">" in toks:
                toks = toks[:toks.index(">")]
            try:
                if toks[:2] == [PY, "-m"]:
                    mod, argv = toks[2], toks[3:]
                    if mod == "blind_insertion.train":
                        ns = T.parser().parse_args(argv)
                    elif mod == "blind_insertion.train_visual":
                        ns = TV.parser().parse_args(argv)
                    elif mod == "blind_insertion.eval":
                        ns = eval_parser_args(argv)
                    else:
                        raise ValueError(f"unknown module {mod}")
                    ej = json.loads(ns.env_json) if getattr(ns, "env_json", "") else {}
                    ef = json.load(open(REPO / ns.env_file)) if getattr(ns, "env_file", "") else {}
                    bad = (set(ej) | set(ef)) - fields
                    if bad:
                        raise ValueError(f"unknown EnvCfg fields {bad}")
                    fam = getattr(ns, "family", "") or ""
                    if fam.startswith("scale:"):
                        _, o, c = fam.split(":")
                        float(o), float(c)
                elif toks[0].endswith(".sh"):
                    if not (REPO / toks[0]).exists():
                        raise FileNotFoundError(toks[0])
                    for t in toks[1:]:
                        if t.startswith("{"):
                            bad = set(json.loads(t)) - fields
                            if bad:
                                raise ValueError(f"unknown EnvCfg fields {bad}")
                elif toks[0] == PY and toks[1].endswith(".py"):
                    if not (REPO / toks[1]).exists():
                        raise FileNotFoundError(toks[1])
                else:
                    raise ValueError("unrecognized command form")
                n_ok += 1
            except SystemExit as e:
                errors.append(f"{p.name}: argparse exit {e.code}: {cmd[:160]}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{p.name}: {type(e).__name__}: {e}: {cmd[:160]}")
    print(f"check: {n_ok} commands OK, {len(errors)} errors")
    for e in errors:
        print("  " + e)
    return not errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate flags against the real parsers")
    a = ap.parse_args()
    b_tr, b_ev = phase_b()
    c_tr, c_ev = phase_c()
    d1, d2, d_ev = phase_d()
    e_tr, e_ev = phase_e()
    dup = ("# Lines tagged also_in_local_queue are ALSO in the local runs/queue/jobs.txt: remove them\n"
           "# from one side first. Lines tagged needs_local_ckpt evaluate runs trained on the local\n"
           "# machine: sync_up.sh --runs NAME and run_jobs.py --mark-complete NAME first, or run there.\n")
    noise = f"# Phase C noise model used here (PLACEHOLDER until Gate C): {json.dumps(NOISE)}\n"
    summary = [
        write("B_train.txt", "Phase B remaining training: blind s2, no-contact s1/s2, visual s1/s2", b_tr, dup),
        write("B_eval.txt", "Phase B evals (eval_phase_b.sh: in-dist 2,048 + 6 OOD bins x 1,024 [+ probe]) "
              "and the Phase B report", b_ev, dup),
        write("C_train.txt", "Phase C YAM: C-R1 base (s1, s2) + §9.3 robustness + §9.4 gain training, 3 seeds",
              c_tr, dup),
        write("C_eval.txt", "Phase C evals: eval_phase_c.sh (factors + gain sweep), own-condition, OOD bins", c_ev,
              dup),
        write("D_train_tier1.txt", "Phase D tier 1: diagonal + 8 mm row, both start regimes, 3 seeds", d1, noise),
        write("D_train_tier2.txt", "Phase D tier 2: remaining grid cells, both start regimes, 3 seeds", d2, noise),
        write("D_eval.txt", "Phase D in-dist evals (2,048 matched starts per run)", d_ev, noise),
        write("E_train.txt", "Phase E USB-C E-up: E1 curriculum + E4 ablations, 3 seeds", e_tr, dup + noise),
        write("E_eval.txt", "Phase E evals: final-stage in-dist; E2 capture radius + E3 clearance for E1/noise", e_ev,
              dup + noise),
    ]
    (HERE / "F_blocked.txt").write_text(F_TEXT)
    print(f"{'manifest':22s} {'jobs':>5s} {'GPU-h':>7s}")
    for f, n, h in summary:
        print(f"{f:22s} {n:5d} {h:7.1f}")
    print(f"{'total':22s} {sum(s[1] for s in summary):5d} {sum(s[2] for s in summary):7.1f}")
    if a.check:
        ok = check([HERE / s[0] for s in summary])
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
