# Blind proprioceptive insertion: from a paper reproduction to USB-C on the YAM

A policy that senses only the robot's joints inserts parts, from 60 mm pegs to USB-C drives
(0.05 mm clearance), on a simulated YAM arm. This repo reproduces Bronars et al.,
*Blind Precise Assembly* (arXiv:2610.23849, `docs/`), and extends it.

| Read this | For |
|---|---|
| `report/index.html` (published as an artifact) | Results and key learnings, by research question |
| `docs/RESEARCH_PLAN.md` | The plan: phases A-G, experiments, gates |
| `docs/FINDINGS.md` | Lab notebook: every result and deviation, newest first |
| `docs/SERVER.md` | Running on a multi-GPU server or SLURM |
| `docs/REAL_YAM_LATENCY_BACKLASH.md` | Measuring the real arm's latency, gains and backlash |
| `CLAUDE.md` | Working notes: verified technical facts, commands, conventions |

## Code layout

| Path | What |
|---|---|
| `blind_insertion/assets/` | Part and scene builders: pegs, USB-C, gears; robots (Flexiv, YAM); backlash, sphere pads |
| `blind_insertion/envs/` | Batched MuJoCo Warp environments (`peg_env.py`: peg and USB-C; `gear_env.py`), kinematics |
| `blind_insertion/policies/` | TCN actor-critic with the contact decoder; camera encoders |
| `blind_insertion/train.py`, `train_visual.py` | PPO training (blind, camera); curricula; `--resume` |
| `blind_insertion/eval.py`, `eval_gear.py` | Test a checkpoint or a scripted baseline; one Parquet row per episode |
| `blind_insertion/analysis/` | Probe and latent swap; statistics (Wilson, bootstrap) |
| `blind_insertion/checks/` | Simulation checks A-S1 to A-S10 (contact signal, drop-in, parity, ...) |
| `blind_insertion/viz/` | CPU replays of logged rollouts: videos and contact sheets |
| `blind_insertion/configs/` | Environment configs per phase (`phase_c_yam.json`, `phase_e_usbc_*.json`, ...) |
| `scripts/` | Batched evals, analyses, figures, server runner, real-arm tools (see `scripts/README.md`) |
| `results/<phase>/` | Frozen results (Parquet, CSV, JSON, PNG) |
| `runs/` | Checkpoints and training logs (not in git) |
| `report/` | The HTML report; figures are ES modules in `report/static/js/figures/` |

Run names: `<phase>-<task>-<modality>-<variant>-s<seed>`, e.g. `E-usbc-proprio-h10-s0`.

## Quick start

```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
.venv/bin/python -m blind_insertion.train --name B-peg-proprio-base-s0 --nworld 12288     # 2.3 h on one GPU
.venv/bin/python -m blind_insertion.eval --policy runs/B-peg-proprio-base-s0/last.pt --episodes 2048 --out results/B/x.parquet
```

## Environment setup

One UV environment for developing a project with exact assembly collision
geometry from mjfit and simulation from RoboSimStudio. Both repositories are Git
submodules, installed editably, so changes to either checkout take effect immediately.

From this directory:

```bash
git submodule update --init --recursive
uv sync --locked
uv run --locked python scripts/tools/check_setup.py
```

Python 3.12 is pinned in `.python-version`; UV can download it automatically.
Run UV commands from this parent directory to use its shared `.venv` and `uv.lock`.
Running `uv sync` inside a submodule selects that repository's separate environment.
For a new clone of this parent repository, use `git clone --recurse-submodules <url>`.

The default install includes RoboSimStudio's CPU/SuperDex backend and mjfit's
test and benchmark dependencies. Production dependencies alone can be installed
with `uv sync --locked --no-dev`. The shared MuJoCo constraint is `~=3.12.0`,
matching RoboSimStudio's own override and satisfying mjfit's `>=3.3` requirement.
Newton and MuJoCo Warp are resolved together in the lockfile. Upstream source
and package manifests are unchanged. Local editable sources are configured using
[UV path sources](https://docs.astral.sh/uv/concepts/projects/dependencies/#path).

The NVIDIA path tracer is optional (about 1.8 GB):

```bash
uv sync --locked --extra rtx
uv run --locked --extra rtx python -m robosimstudio.machine
```

Keep `--extra rtx` on later UV commands to keep it installed. GPU execution needs
a working NVIDIA driver; CPU operation is available without it. Other heavyweight
upstream extras such as Genesis, GAP and cuRobo are not configured here.

Run the checks:

```bash
uv pip check
uv run --locked python scripts/tools/check_setup.py
uv run --locked python scripts/tools/test_mjfit.py -q
cd robosimstudio
../.venv/bin/python scripts/testrun.py --tier fast
../.venv/bin/python scripts/testrun.py --tier cpu
```

`check_setup.py` generates a peg and hole, steps the generated XML in MuJoCo,
imports the same assembly into a RoboSimStudio CPU scene and steps it there.
`test_mjfit.py` selects mjfit's test namespace for that process: RoboSimStudio's
editable install also exposes a package named `tests`, which otherwise shadows
mjfit's test helpers. The RoboSimStudio suites need permission to bind local sockets.

In restricted environments, keep runtime caches under this checkout:

```bash
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export XDG_CACHE_HOME="$PWD/.cache"
export ROBOSIMSTUDIO_CACHE="$PWD/.cache/robosimstudio"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
```

Use `git submodule status` to inspect pinned revisions. To update a submodule,
fetch and select a revision inside it, rerun `uv lock` and the checks above, then
commit the changed submodule reference and parent lockfile together. Commit source
changes inside their submodule before recording the new reference in the parent.
