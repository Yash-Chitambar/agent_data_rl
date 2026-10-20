# CLAUDE.md — blind proprioceptive insertion project

Working context for Claude (and humans) in this repo. Keep it current.

## What this is

Research repo reproducing and extending Bronars, Park, Kedia, Agrawal, *Blind Precise
Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse Parts* (arXiv:2610.23849,
PDF in `docs/`). The plan, phases and gates are in `docs/RESEARCH_PLAN.md` (the source of
truth for *what* to do). Discoveries, deviations and results go in `docs/FINDINGS.md`
(the running lab notebook; newest entries at the top of each section). New ideas go in
the "Ideas" section of `docs/FINDINGS.md`.

## Layout

| Path | What |
|---|---|
| `mjfit/`, `robosimstudio/` | Git submodules, editable installs. Do not edit upstream source. |
| `third_party/mujoco_menagerie/` | Sparse clone: `flexiv_rizon4s`, `robotiq_2f85`, `i2rt_yam` (Apache/MIT). |
| `blind_insertion/` | Our code: assets builder, batched env, policy, PPO, eval, analysis. |
| `blind_insertion/checks/` | Phase A signal checks (A-S1..A-S10), one script each. |
| `scripts/` | Entry points (train, eval, figures). |
| `results/<phase>/` | Frozen CSV/Parquet + PNG per figure. |
| `runs/` | Training checkpoints/logs (gitignored). |
| `report/` | HTML results report (research-figures template), built by `scripts/build_report.py`. |

## Environment

- Python 3.12 venv at `.venv`, managed by uv from this directory. Always run
  `.venv/bin/python ...` (or `uv run --locked`) from the repo root.
- Set `export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.python"` before uv.
- Training deps (torch 2.14 cu130, pyarrow, h5py) are in the `train` dependency group.
- GPU: 1x RTX 3090 Ti (24 GB). 16 CPU cores, 60 GB RAM.
- MuJoCo 3.12.0, MuJoCo Warp 3.12.0, Warp 1.18.
- Headless rendering: `MUJOCO_GL=egl`. Call `renderer.close()` explicitly to avoid EGL
  teardown noise.
- **mjlab is not used**: mjlab 1.6 pins `mujoco~=3.11`, which conflicts with the shared
  3.12 stack. We use the plan's fallback: plain MuJoCo Warp env + our own PPO in torch.

## Key technical facts (verified)

- MuJoCo Warp batches `geom_dataid` per world (`Model` field spec starts with `*`), and
  collision reads `geom_dataid[worldid % shape[0]]`. So **one compiled model can give
  every world a different mesh assembly**: put all variants' meshes in the model as
  assets, keep a fixed number of geom slots, and set per-world `geom_dataid`,
  `geom_pos/quat/size/aabb/rbound`, body mass/inertia. Unused slots get parked far away.
- `put_model(mjm, batch_sizes={field: nworld})` allocates per-world copies of batched
  fields; then write them through `wp.to_torch`.
- Paper's action scheme: `q_target = q_meas + scale * a` held for 20 substeps, so the
  command is effectively `tau = kp*scale*a - kd*qdot` at the start of each step.
- mjfit geoms use class `mjfit`: solref (2 ms, 1), solimp (0.99, 0.999, 0.001),
  condim 4, margin/gap 0. Timestep must be <= 1 ms.

- **Static geoms are frozen in MuJoCo Warp**: geoms on world-welded bodies get their pose
  once in `make_data`. The fixture is therefore a **mocap** body (`d.mocap_pos/quat`).
- **Cameras update only in `forward`/`camlight`**, not `kinematics`; `PegEnv.render()` runs
  kinematics + camlight + refit_bvh + render. Wrist camera "wrist" is on the Flexiv gripper.
  Batched 64x64 RGB+depth renders ~160-200k frames/s (4096 worlds in ~25 ms).
- **MuJoCo Warp is not bitwise deterministic** (A-S8); compare distributions, not rollouts.
- Model build time is quadratic in mesh count: keep variants <= ~1024 per model.
- `EnvCfg(auto_reset=False)` + huge `far_*`/`slip_dist` for scripted checks; otherwise
  success/termination silently resets worlds mid-check.
- The CUDA graph (20 substeps) is captured in `PegEnv.__init__` before the first reset.

## How to run

```bash
export PYTHONPATH=$PWD
.venv/bin/python -m blind_insertion.train --name B-peg-proprio-base-s0 --seed 0 --nworld 12288
.venv/bin/python -m blind_insertion.train_visual --name B-peg-rgb-base-s0 --nworld 4096
.venv/bin/python -m blind_insertion.eval --policy runs/<run>/last.pt --episodes 2048 --out results/B/x.parquet [--ood tighter] [--log_worlds 0,1]
.venv/bin/python -m blind_insertion.eval --policy 'spiral{"gain":10,"speed":0.04}' --env_json '{"episode_s":15.0}' ...
.venv/bin/python -m blind_insertion.analysis.probes --policy runs/<run>/last.pt --out results/B/probe
.venv/bin/python -m blind_insertion.viz.replay --rollouts results/B/x.rollouts.npz --worlds 0,1 --out results/B/viz/x
.venv/bin/python -m blind_insertion.checks.signal_checks --robot flexiv   # A-S3..A-S6 (CPU)
.venv/bin/python scripts/build_report.py [--no-media]   # HTML report data/videos -> report/static/
report/tools/preview.sh && <playwright python> report/tools/check_page.py report --page .preview.html
```

- `report/index.html` is in artifact form (no doctype/head; the Artifact publisher adds them).
  Text follows ASD-STE100 (short active sentences, one idea each). Figures are ES modules in
  `report/static/js/figures/` reading `report/static/data/*.json`.

- GPU job queue: `scripts/queue.sh` (running in background) executes lines of
  `runs/queue/jobs.txt` sequentially (done list in `runs/queue/done.txt`, log in
  `runs/queue/queue.log`). Append a line to queue a job. It waits while any
  `blind_insertion.train*` process runs.
- Eval variants default to `variant_seed=1000` (unseen geometries from the training
  distribution; training uses seed 0). OOD bins: `builder.OOD_BINS`.

## Conventions

- Lengths in meters in code; mm in plots/tables.
- Run names: `<phase>-<task>-<modality>-<variant>-s<seed>`.
- Every figure: a script in `blind_insertion/analysis/figures/` or `scripts/`, data in
  `results/<phase>/`, n stated in caption.
- Never use RoboSimStudio for numbers; visuals only.

## Server migration

Moving training to a multi-GPU box or SLURM cluster: see `docs/SERVER.md`. Setup is
`git clone --recurse-submodules` + `bash scripts/server/bootstrap.sh` (uv env, Menagerie sparse
clone, GPU/EGL checks, smoke runs). Jobs are manifests in `scripts/server/jobs/` (generated by
`make_manifests.py`; `--check` validates flags), run by `scripts/server/run_jobs.py` (one job per
GPU) or `scripts/server/slurm/submit.sh` (job arrays); results come back with
`scripts/server/sync_down.sh`. train.py cannot resume yet (docs/SERVER.md §9).
