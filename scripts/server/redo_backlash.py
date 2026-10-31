#!/usr/bin/env python3
"""Prepare the server to redo every job that used the old (soft-limit) backlash model.

The backlash hinges had MuJoCo's soft default joint limit until commit 3006b51 (2026-10-21), so every
backlash level acted as ~0.6 deg of joint compliance (docs/campaign/FINDINGS.md, "BUG: the backlash model was a
soft spring"). Yash decided to redo the affected runs with the fixed model. This script:

  1. finds the affected training runs: every manifest training line whose command has `backlash_arcmin`
     (C backlash1/3/5, allmid, allmax, randall; all of Phase D; E noise);
  2. finds the affected eval lines: lines that name an affected run (`--policy runs/X/`, `--name X`,
     `eval_phase_*.sh X`, `after=X`);
  3. moves runs/X to runs/_softbl/X and the result files of the affected evals to
     results/<phase>/_softbl/ (kept for comparison, not deleted);
  4. moves the backlash columns of the other Phase C policies (results/C/eval_<tag>_backlash*.parquet)
     to results/C/_softbl/; jobs/C_eval_bl.txt redoes only those columns (ONLY=backlash);
  5. removes the affected commands from the runner's done.txt, so run_jobs.py runs them again.

Stop the runner first (its affected jobs still run the old model in memory), pull main, then:
    python3 scripts/server/redo_backlash.py            # dry run: prints what it would do
    python3 scripts/server/redo_backlash.py --apply
and restart the runner with the same manifests plus scripts/server/jobs/C_eval_bl.txt.
"""
from __future__ import annotations

import argparse
import re
import shlex
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
JOBS = REPO / "scripts/server/jobs"
MANIFESTS = ["C_train.txt", "C_eval.txt", "D_train_tier1.txt", "D_train_tier2.txt", "D_eval.txt",
             "E_train.txt", "E_eval.txt"]


def lines(p: Path):
    for raw in p.read_text().splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            cmd, _, ann = s.partition(" # ")
            yield cmd.strip(), ann


def eval_outputs(cmd: str) -> list[Path]:
    """Result files a command writes (globs for the eval_phase_*.sh wrappers)."""
    tok = shlex.split(cmd)
    out = []
    if "--out" in tok:
        o = tok[tok.index("--out") + 1]
        out += [Path(o), Path(o.replace(".parquet", ".rollouts.npz"))]
    for i, t in enumerate(tok):
        if t.endswith(("eval/phase_c.sh", "eval_phase_c.sh")):
            out += sorted((REPO / "results/C").glob(f"eval_{tok[i + 2]}_*"))
        elif t.endswith(("eval/phase_e.sh", "eval_phase_e.sh")):
            out += sorted((REPO / "results/E").glob(f"*_{tok[i + 2]}_*"))
    return [p if p.is_absolute() else REPO / p for p in out]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="move files and edit done.txt (default: dry run)")
    ap.add_argument("--state", default="runs/queue_server", help="run_jobs.py state dir")
    a = ap.parse_args()

    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    if a.apply and any("run_jobs.py" in x and "redo_backlash" not in x and "--status" not in x for x in ps):
        raise SystemExit("run_jobs.py is running: stop it first (its affected jobs use the old model)")

    all_lines = [(m, c, ann) for m in MANIFESTS for c, ann in lines(JOBS / m)]
    names = {re.search(r"--name (\S+)", c).group(1) for _, c, _ in all_lines
             if "--name " in c and "backlash_arcmin" in c}
    pat = re.compile(r"(runs/|--name |(?:eval_|eval/)phase_[ce]\.sh |after=|,)(" + "|".join(map(re.escape, names)) + r")(/|\s|,|$)")
    affected = [(m, c) for m, c, ann in all_lines if pat.search(c) or pat.search(ann)]
    moves = []
    for n in sorted(names):
        if (REPO / "runs" / n).exists():
            moves.append((REPO / "runs" / n, REPO / "runs/_softbl" / n))
    for _, c in affected:
        moves += [(p, p.parent / "_softbl" / p.name) for p in eval_outputs(c) if p.exists()]
    bl_cols = []
    for _, c, _ in all_lines:   # other C policies: only their backlash columns
        if ("eval/phase_c.sh" in c or "eval_phase_c.sh" in c) and not pat.search(c):
            tag = shlex.split(c)[2]
            bl_cols += sorted((REPO / "results/C").glob(f"eval_{tag}_backlash*.parquet"))
    moves += [(p, p.parent / "_softbl" / p.name) for p in bl_cols]
    moves = list(dict.fromkeys(moves))

    print(f"affected training runs ({len(names)}):", " ".join(sorted(names)))
    print(f"affected manifest lines: {len(affected)}")
    print(f"paths to archive: {len(moves)}")
    for s, d in moves:
        print(f"  {s.relative_to(REPO)} -> {d.relative_to(REPO)}")
    done_f = REPO / a.state / "done.txt"
    done = done_f.read_text().splitlines() if done_f.exists() else []
    drop = {c for _, c in affected}
    keep = [x for x in done if x.strip() not in drop]
    print(f"done.txt: {len(done) - len(keep)} of {len(done)} lines to remove")
    if not a.apply:
        print("dry run: pass --apply to do it")
        return
    for s, d in moves:
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            raise SystemExit(f"{d} exists: archive it by hand first")
        shutil.move(str(s), str(d))
    if done_f.exists():
        shutil.copy(done_f, done_f.with_suffix(".txt.pre_redo_backlash"))
        done_f.write_text("".join(x + "\n" for x in keep))
    print("done: restart run_jobs.py with the same manifests plus scripts/server/jobs/C_eval_bl.txt")


if __name__ == "__main__":
    main()
