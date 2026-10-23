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
import re
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
# 2026-10-20 (Yash): initial results first. C, D and E train seed 0 only, and blind C/D/E runs
# stop once training success reaches STOP_SUCCESS after MIN_STEPS (Phase B keeps 3 seeds and
# full 300M runs). Add seeds 1-2 later where a result is close or surprising.
SEEDS_CDE = (0,)
STOP = "--stop_success 0.99 --min_steps 50e6"
# Already running on mason with the full 300M command: keep their lines unchanged, because the
# runner matches running jobs by command text (a changed line would start a duplicate).
STARTED_FULL = {"C-peg-proprio-base-s0", "C-peg-proprio-qnoise1e-4-s0"}
# Phase C noise model carried into Phases D and E (plan §10: "with the Phase C noise model on").
# Gate C (2026-10-21): per-world randomized levels. C-peg-proprio-randall-s0 keeps 96-98% under every
# noise, quantization, backlash and latency test level, while each fixed level transfers poorly.
NOISE_RAND = {"q_noise": [0, 1e-4, 5e-4], "q_quant_bits": [0, 16, 14], "backlash_arcmin": [0, 1, 3, 5],
              "latency": [0, 1, 2]}
NOISE = {"noise_rand": NOISE_RAND}
MED_PER_HALF = 0.765          # train.MEDIAN_RADIAL_PER_HALFWIDTH: median |xy| = 0.765 a for U(-a,a)^2

# GPU-hour assumptions, from runs on the local RTX 3090 Ti (runs/queue/queue.log):
# blind 300M at 12,288 worlds = 2.2-2.4 h (36-38k env steps/s incl. PPO, Flexiv and YAM);
# visual with the stop rule stopped at 50M = 1.7 h (8.5k steps/s; 300M would be ~9.8 h);
# eval/phase_b.sh (7 evals + probe) = 0.18 h. USB-C (8,192 worlds, njmax 2048) not yet measured.
H_BLIND, H_VISUAL, H_VISUAL_MAX, H_USBC = 2.3, 1.7, 9.8, 3.0
H_EVAL_B, H_EVAL_C, H_OOD, H_EVAL1, H_EVAL_E_FULL = 0.18, 0.35, 0.12, 0.04, 0.35

# Runs that the LOCAL queue (runs/queue/jobs.txt) also contains as of 2026-10-20: remove them
# from one side before launching, or they train twice.
LOCAL_QUEUE = {"B-peg-proprio-base-s2", "B-peg-proprio-nocontact-s1", "E-usbc-proprio-curr-s0"}
# Runs already finished locally (evaluate where their checkpoints are; on a server first
# `sync_up.sh --runs NAME` and `run_jobs.py --mark-complete NAME`).
LOCAL_DONE = {"B-peg-proprio-base-s0", "B-peg-proprio-base-s1", "B-peg-proprio-nocontact-s0",
              "B-peg-rgb-base-s0"}
# C-peg-proprio-base-s0 from athena5 used the old YAM mount (fixture 10-30 cm below the base);
# the server trains C base s0 again on the lab table setup (builder.YAM_TABLE).


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


def train_line(name, seed, args, *, visual=False, mem=11, est=H_BLIND, nworld=12288, steps="300e6"):
    mod = "train_visual" if visual else "train"
    cmd = f"{PY} -m blind_insertion.{mod} --name {name} --seed {seed} --nworld {nworld} --steps {steps} {args}".strip()
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
            ev.append(eval_line(f"scripts/eval/phase_b.sh {name} {tag}_s{s}{probe}", f"eval_{tag}_s{s}", name,
                                H_EVAL_B, visual=tag == "rgb"))
    allb = [f"B-peg-{m}-{v}-s{s}" for s in SEEDS for m, v in (("proprio", "base"), ("proprio", "nocontact"),
                                                              ("rgb", "base"))]
    ev.append(f"{PY} scripts/report/phase_b.py > runs/phase_b_report.log 2>&1"
              + ann(after=",".join(allb), est_h=0.05, note="needs_all_B_train_logs_in_runs"))
    return tr, ev


def phase_c_bl():
    """Backlash columns only (ONLY=backlash), for the C policies trained without backlash: the redo after
    the 2026-10-21 backlash-model fix (scripts/server/redo_backlash.py archives the old columns)."""
    ev = []
    for s in SEEDS_CDE:
        for var, ej in C_CONDS:
            if "backlash_arcmin" in json.dumps(ej):
                continue                  # retrained: its full eval/phase_c.sh line runs again
            name, tag = f"C-peg-proprio-{var}-s{s}", f"yam_{var}_s{s}"
            base = f" {jq(C_EVAL_BASE[var])}" if var in C_EVAL_BASE else ""
            ev.append(eval_line(f"ONLY=backlash scripts/eval/phase_c.sh {name} {tag}{base}", f"eval_{tag}_bl",
                                name, 0.06))
    return ev


# ------------------------------------------------------------------------------- Phase C
def gains(g):
    """§5.2 YAM gains x g, in the format scripts/eval/phase_c.sh passes."""
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
    # §9.3 "training with the noise randomized": every factor's level drawn per world at each
    # reset (EnvCfg.noise_rand); eval/phase_c.sh then gives its success at each fixed level
    ("randall", {"noise_rand": NOISE_RAND}),
    # 2026-10-21: the base policy is bang-bang (27% of actions at the +-3 clip) and that is what
    # latency/backlash break (results/C/analysis/). Same task with the action clip at 1: does a
    # smooth policy transfer without knowing the latency? Evaluated with the same clip.
    ("clip1", {"action_clip": 1.0}),
]
C_EVAL_BASE = {"clip1": {"action_clip": 1.0}}   # env JSON merged under every eval/phase_c.sh condition


def phase_c():
    tr, ev = [], []
    for s in SEEDS_CDE:
        for var, ej in C_CONDS:
            name = f"C-peg-proprio-{var}-s{s}"
            args = f"--env_file {YAM}" + (f" --env_json {jq(ej)}" if ej else "")
            if name not in LOCAL_DONE:
                tr.append(train_line(name, s, args if name in STARTED_FULL else f"{args} {STOP}"))
            tag = f"yam_{var}_s{s}"
            # full brittleness matrix: base + each factor alone + gain sweep (scripts/eval/phase_c.sh)
            base = f" {jq(C_EVAL_BASE[var])}" if var in C_EVAL_BASE else ""
            ev.append(eval_line(f"scripts/eval/phase_c.sh {name} {tag}{base}", f"eval_{tag}", name, H_EVAL_C))
            if var.startswith("all"):     # also under its own combined training condition
                ev.append(eval_line(f"{PY} -m blind_insertion.eval --policy runs/{name}/last.pt --env_file {YAM} "
                                    f"--episodes 2048 --env_json {jq(ej)} --out results/C/eval_{tag}_own.parquet",
                                    f"eval_{tag}_own", name, H_EVAL1))
            if var == "base":             # C-R4 OOD bins on the YAM
                ev.append(eval_line(f"scripts/eval/ood.sh {name} {tag} C {YAM}", f"ood_{tag}", name,
                                    H_OOD))
    return tr, ev


# ------------------------------------------------------------------------------- Phase D
D_OPENINGS_MM = (60, 30, 15, 8)
D_CLEARANCES_MM = (1.0, 0.5, 0.2, 0.1, 0.05)
D_DIAGONAL = {(60, 1.0), (30, 0.5), (15, 0.2), (8, 0.05)}     # large-loose -> small-tight
# 2026-10-21: D cells cap at 150M steps (small/tight cells may never reach the 99% stop rule; a
# 300M cap would push tier 1 past the first results review). Extend with --resume where needed.
D_STEPS = "150e6"
# 2026-10-21: o15_c0.2 from scratch never found the hole (0 successes in 128M steps: the sparse reward
# gave PPO no signal). Cells tighter than 1 mm now use the clearance curriculum (train.py
# --curriculum scale: 1 mm first, down to the target). o30_c0.5_scaled-s0 had already reached 99% without it.
D_NO_CURRICULUM = {"D-peg-proprio-o30_c0.5_scaled-s0"}
# 2026-10-21 17:00: with box pads the 8 mm and 15 mm pegs slip in the YAM grasp (o8_c1_scaled: 73.7%,
# 159 of 269 failures slipped) and the curriculum never left 1 mm. Small cells use the USB-C sphere pads
# (and its contact buffers) under new names *_sph; the box-pad runs keep the old names.
D_SPHERE_PADS_MAX_MM = 15
SPHERE_PADS = {"pad_spheres": 4, "nconmax": 128, "njmax": 1024}   # 256/2048 OOMs at 12,288 worlds (32 GB)


def d_offset(opening_mm, regime):
    """Per-axis half-width a (m) giving the plan's median start offset."""
    med_mm = opening_mm / 6.0 if regime == "scaled" else 3.0    # 10 mm at 60 mm / fixed 3 mm
    return round(med_mm / 1000 / MED_PER_HALF, 6)


def phase_d():
    tiers = {1: [], 2: []}
    ev = []
    cells = [(o, c, r) for r in ("scaled", "fixed") for o in D_OPENINGS_MM for c in D_CLEARANCES_MM]
    for s in SEEDS_CDE:
        for o, c, r in cells:
            tier = 1 if (o, c) in D_DIAGONAL or o == 8 else 2
            var = f"o{o}_c{c:g}_{r}" + ("_sph" if o <= D_SPHERE_PADS_MAX_MM else "")
            name = f"D-peg-proprio-{var}-s{s}"
            fam = f"scale:{o / 1000:g}:{c / 1000:g}"
            ej = dict(NOISE, peg_offset=d_offset(o, r))
            if o <= D_SPHERE_PADS_MAX_MM:
                ej.update(SPHERE_PADS)
            cur = " --curriculum scale" if c < 1.0 and name not in D_NO_CURRICULUM else ""
            tiers[tier].append(train_line(name, s, f"--env_file {YAM} --family {fam}{cur} --env_json {jq(ej)} {STOP}",
                                          steps=D_STEPS, est=H_BLIND / 2))
            ev.append(eval_line(f"{PY} -m blind_insertion.eval --policy runs/{name}/last.pt --env_file {YAM} "
                                f"--family {fam} --env_json {jq(ej)} --episodes 2048 "
                                f"--out results/D/eval_{var}_s{s}.parquet", f"eval_{name}", name, H_EVAL1))
    return tiers[1], tiers[2], ev


# ------------------------------------------------------------------------------- Phase E
E_BASE = "--family usbc --curriculum usbc --n_variants 320"
USBC_LATCH = "blind_insertion/configs/phase_e_usbc_latch.json"   # E5: + latch spring, 5-15 N
USBC_SIDE = "blind_insertion/configs/phase_e_usbc_side.json"     # E6: port faces the robot
CHAMFER = {"chamfer": [2e-4, 5e-4]}                              # E4: 0.2-0.5 mm lead-in
E_CONDS = [  # variant, env_file, env_json (train and eval), variant_kw, eval mode
    ("curr", USBC, {}, {}, "full"),                                # E1 (+E2/E3 evals)
    ("nocontact", USBC, {}, {}, "indist"),                         # E4 ablations (+ --contact_coef 0)
    ("h10", USBC, {"history": 10}, {}, "indist"),
    ("h60", USBC, {"history": 60}, {}, "indist"),
    # E1 trains WITHOUT the noise model, so the plan's "no noise model" ablation is E1 itself and
    # this is the "with noise" counterpart (the Gate C randomized model)
    ("noise", USBC, NOISE, {}, "full"),
    ("chamfer", USBC, {}, CHAMFER, "full"),                        # E4 with chamfer
    ("latch", USBC_LATCH, {}, {}, "full"),                         # E5
    ("side", USBC_SIDE, {}, {}, "full"),                           # E6
]
# E1's policy evaluated zero-shot on the E4/E5/E6 assemblies ("with vs without" for the paper
# comparison): tag, env_file, variant_kw
E1_CROSS = [("chamfer", USBC, CHAMFER), ("latch", USBC_LATCH, {}), ("side", USBC_SIDE, {})]


def e_eval_opts(ef, ej, vk):
    return (f" --env_file {ef}" + (f" --env_json {jq(ej)}" if ej else "")
            + (f" --variant_kw {jq(vk)}" if vk else ""))


def phase_e():
    tr, ev = [], []
    for s in SEEDS_CDE:
        for var, ef, ej, vk, mode in E_CONDS:
            name = f"E-usbc-proprio-{var}-s{s}"
            extra = (" --contact_coef 0" if var == "nocontact" else "") + e_eval_opts(ef, ej, vk)
            tr.append(train_line(name, s, f"{E_BASE}{extra} {STOP}", nworld=8192, mem=16, est=H_USBC))
            tag = f"{var}_s{s}"
            ev.append(eval_line(f"scripts/eval/phase_e.sh {name} {tag} {mode}{e_eval_opts(ef, ej, vk)}",
                                f"eval_{name}", name, H_EVAL_E_FULL if mode == "full" else H_EVAL1))
        e1 = f"E-usbc-proprio-curr-s{s}"
        for tag, ef, vk in E1_CROSS:
            ev.append(eval_line(f"scripts/eval/phase_e.sh {e1} curr_on_{tag}_s{s} indist"
                                f"{e_eval_opts(ef, {}, vk)}", f"eval_curr_on_{tag}_s{s}", e1, H_EVAL1))
    return tr, ev


# ------------------------------------------------------------------------------- Phase F
USBC_F = "blind_insertion/configs/phase_f_usbc.json"   # USB-C spheres + camera/light/color randomization
F_CONDS = [  # plan §12.2 (F-P = E-usbc-proprio-curr-s*): id, --modality, --proprio
    ("rgb", "rgb", False), ("depth", "depth", False), ("pr", "rgb", True), ("pd", "depth", True),
    ("prd", "rgbd", True),
]
# Direct RL (plan §12.3 route 1). USB-C success plateaus near 90% in training, so the 0.99 stop rule
# would not fire: cap at 150M steps (E1 reached its final stage at 200M; vision may need fewer).
F_STEPS = "150e6"
H_F = 14.0   # 150M steps at ~3k steps/s (1,024 worlds RGB-D+proprio: 10.3 GB, 1.75k steps/s on a shared 3090 Ti)


def phase_f():
    tr, ev = [], []
    for s in SEEDS_CDE:
        for var, mod, prop in F_CONDS:
            name = f"F-usbc-{var}-curr-s{s}"
            args = (f"{E_BASE} --env_file {USBC_F} --modality {mod}" + (" --proprio" if prop else "")
                    + f" --stop_success 0.99 --min_steps 50e6")
            tr.append(train_line(name, s, args, visual=True, mem=20, est=H_F, nworld=2048).replace(
                "--steps 300e6", f"--steps {F_STEPS}"))
            ev.append(eval_line(f"scripts/eval/phase_f.sh {name} {var}_s{s}", f"eval_{name}", name,
                                1.0, visual=True, mem=8))
        # F-P on the same scenes (its camera is unused; same starts as the visual policies)
        e1 = f"E-usbc-proprio-curr-s{s}"
        ev.append(eval_line(f"scripts/eval/phase_f.sh {e1} p_s{s}", f"eval_F-P_s{s}", e1, 0.3))
    return tr, ev


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
            while toks and re.fullmatch(r"[A-Z_]+=\S*", toks[0]):   # env prefix, e.g. ONLY=backlash
                toks = toks[1:]
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
                    for prev, t in zip(toks, toks[1:]):
                        if prev == "--env_file" and not (REPO / t).exists():
                            raise FileNotFoundError(t)
                        if t.startswith("{") and prev != "--variant_kw":
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
    f_tr, f_ev = phase_f()
    dup = ("# Lines tagged also_in_local_queue are ALSO in the local runs/queue/jobs.txt: remove them\n"
           "# from one side first. Lines tagged needs_local_ckpt evaluate runs trained on the local\n"
           "# machine: sync_up.sh --runs NAME and run_jobs.py --mark-complete NAME first, or run there.\n")
    noise = f"# Phase C noise model used here (Gate C): {json.dumps(NOISE)}\n"
    summary = [
        write("B_train.txt", "Phase B remaining training: blind s2, no-contact s1/s2, visual s1/s2", b_tr, dup),
        write("B_eval.txt", "Phase B evals (eval/phase_b.sh: in-dist 2,048 + 6 OOD bins x 1,024 [+ probe]) "
              "and the Phase B report", b_ev, dup),
        write("C_train.txt", "Phase C YAM: C-R1 base (s1, s2) + §9.3 robustness + §9.4 gain training, 3 seeds",
              c_tr, dup),
        write("C_eval.txt", "Phase C evals: eval/phase_c.sh (factors + gain sweep), own-condition, OOD bins", c_ev,
              dup),
        write("C_eval_bl.txt", "Phase C backlash columns redo (fixed backlash model) for policies trained "
              "without backlash", phase_c_bl(), dup),
        write("D_train_tier1.txt", "Phase D tier 1: diagonal + 8 mm row, both start regimes, 3 seeds", d1, noise),
        write("D_train_tier2.txt", "Phase D tier 2: remaining grid cells, both start regimes, 3 seeds", d2, noise),
        write("D_eval.txt", "Phase D in-dist evals (2,048 matched starts per run)", d_ev, noise),
        write("E_train.txt", "Phase E USB-C (sphere pads): E1 curriculum, E4 ablations + chamfer, E5 latch, "
              "E6 side port", e_tr, dup + noise),
        write("E_eval.txt", "Phase E evals: final-stage in-dist; E2 capture radius + E3 clearance (full mode); "
              "E1 zero-shot on chamfer/latch/side", e_ev, dup + noise),
        write("F_train.txt", "Phase F USB-C vision (direct RL, 2,048 worlds): F-R, F-D, F-PR, F-PD, F-PRD", f_tr),
        write("F_eval.txt", "Phase F evals: F1 in-dist, F2 capture radius to 15 mm, F4 degradation; F-P = E1",
              f_ev),
    ]
    (HERE / "F_blocked.txt").unlink(missing_ok=True)
    print(f"{'manifest':22s} {'jobs':>5s} {'GPU-h':>7s}")
    for f, n, h in summary:
        print(f"{f:22s} {n:5d} {h:7.1f}")
    print(f"{'total':22s} {sum(s[1] for s in summary):5d} {sum(s[2] for s in summary):7.1f}")
    if a.check:
        ok = check([HERE / s[0] for s in summary])
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
