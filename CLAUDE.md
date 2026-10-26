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
| `yam-control/` | Git submodule: real YAM arm and camera control (CAN, RealSense/ZED streaming, rig config). Do not edit upstream source. |
| `third_party/mujoco_menagerie/` | Sparse clone: `flexiv_rizon4s`, `robotiq_2f85`, `i2rt_yam` (Apache/MIT). |
| `blind_insertion/` | Our code: assets builder, batched env, policy, PPO, eval, analysis. |
| `blind_insertion/checks/` | Phase A signal checks (A-S1..A-S10), one script each. |
| `blind_insertion/scenes/`, `envs/pick_env.py`, `pipeline/` | Task H: mini PC + SD reader bench (`pc_bench`), batched pick env + scripted expert, pick -> insert evaluation. |
| `scripts/` | `eval/` (batched evals), `analysis/` (probes, signal checks, replays), `report/` (figures, report data), `real/` (real-arm tools), `server/` (multi-GPU runner), `tools/` (setup checks, local queue). See `scripts/README.md`. |
| `results/<phase>/` | Frozen CSV/Parquet + PNG per figure. |
| `runs/` | Training checkpoints/logs (gitignored). |
| `report/` | HTML results report (research-figures template), built by `scripts/report/build_report.py`. |

## Environment

- Python 3.12 venv at `.venv`, managed by uv from this directory. Always run
  `.venv/bin/python ...` (or `uv run --locked`) from the repo root.
- Set `export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.python"` before uv.
- Training deps (torch 2.14 cu130, pyarrow, h5py) are in the `train` dependency group.
- Real-arm deps (i2rt, python-can, pyrealsense2, pyzed) are not in uv.lock (i2rt pins numpy 2.2.6):
  `bash scripts/real/setup_real.sh` installs them into `.venv` with uv. Plain `uv sync` removes them.
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
- **Box pads can lose grasp contacts** when a narrow part's face is rotated in the pad plane
  (partial face overlap: MuJoCo box-mesh multi-contact gives an edge or nothing). Use
  `EnvCfg(pad_spheres=4)` for small parts (USB-C); 3x3 gives no contact at all. The env does not log
  the CCD overflow bit (4); per-world pad contact counts are in `checks/e_slip_gpu_cpu.py`.

## How to run

```bash
export PYTHONPATH=$PWD
.venv/bin/python -m blind_insertion.train --name B-peg-proprio-base-s0 --seed 0 --nworld 12288
.venv/bin/python -m blind_insertion.train_visual --name B-peg-rgb-base-s0 --nworld 4096
.venv/bin/python -m blind_insertion.eval --policy runs/<run>/last.pt --episodes 2048 --out results/B/x.parquet [--ood tighter] [--log_worlds 0,1]
.venv/bin/python -m blind_insertion.eval --policy 'spiral{"gain":10,"speed":0.04}' --env_json '{"episode_s":15.0}' ...
.venv/bin/python -m blind_insertion.analysis.probes --policy runs/<run>/last.pt --out results/B/probe [eval.py env flags]
.venv/bin/python -m blind_insertion.train_visual --name F-usbc-pd-curr-s0 --modality depth --proprio --family usbc --curriculum usbc --n_variants 320 --env_file blind_insertion/configs/phase_f_usbc.json
.venv/bin/python -m blind_insertion.checks.signal_checks --env_file blind_insertion/configs/phase_e_usbc_spheres.json --family usbc --only 5,6 --out results/E/signal
.venv/bin/python -m blind_insertion.viz.replay --rollouts results/B/x.rollouts.npz --worlds 0,1 --out results/B/viz/x
.venv/bin/python -m blind_insertion.checks.signal_checks --robot flexiv   # A-S3..A-S6 (CPU)
.venv/bin/python scripts/report/build_report.py [--no-media]   # HTML report data/videos -> report/static/
.venv/bin/python scripts/report/phase_c.py    # C train x test matrix -> results/C/C-matrix.{csv,png}
.venv/bin/python scripts/real/yam_latency_fit.py fit 'data/real_yam/steps_joint*.csv' --kv 5 5 5 2 2 2 --friction 0.01 0.1 0.2 0.4 0.7 1.0 1.5 2.0 3.0 4.0   # real-arm gain + friction fit (read delay from the onset, see FINDINGS)
report/tools/preview.sh && <playwright python> report/tools/check_page.py report --page .preview.html
```

- `report/index.html` is in artifact form (no doctype/head; the Artifact publisher adds them).
  Two sessions publish it (athena5 and the server agent on mason): before each publish, read the live
  artifact and merge onto it; `scripts/report/build_report.py` rebuilds curves.json from local runs only, so
  merge server series (e.g. `yam_table`, blind seed 3) back in instead of overwriting.
  Text follows ASD-STE100 (short active sentences, one idea each). Figures are ES modules in
  `report/static/js/figures/` reading `report/static/data/*.json`.

- GPU job queue: `scripts/tools/local_queue.sh` (running in background) executes lines of
  `runs/queue/jobs.txt` sequentially (done list in `runs/queue/done.txt`, log in
  `runs/queue/queue.log`). Append a line to queue a job. It waits while any
  `blind_insertion.train*` process runs.
- USB-C variants (Phase E): chamfer via `--variant_kw '{"chamfer": [2e-4, 5e-4]}'` (train and eval);
  latch `EnvCfg.latch` (`configs/phase_e_usbc_latch.json`), side port `EnvCfg.fix_base_rpy_deg`
  (`configs/phase_e_usbc_side.json`). Per-world noise levels: `EnvCfg.noise_rand`.
- Visual: `EnvCfg.cam_modality` rgb|depth|rgbd (frame channels), `depth_model` (D405-like sensor),
  `cam_rand`; the YAM camera is Menagerie's `wrist_cam` (D405). `train_visual --proprio` fuses joints.
- Real rig (duchamp3): **clamp the arm bases before any motion** (an unclamped base nearly tipped at
  ~0.58 m reach). Connect arms only through `scripts/real/safety.py` (PD hold from connect; yam-control's
  `Arms` floats and the right arm drifts into the table). Run `scripts/real/estop.py` in a terminal
  first: Enter holds the arm, `c` sends it home. Camera calibration: `scripts/real/calib.py`
  (zed / find / collect / solve; views accumulate per run in `data/calib/run_NNN`, gitignored).
  Scene reference images for the sim (ZED + both wrists, camera poses in the right base frame):
  `scripts/real/snapshot.py` -> `data/sim_ref/<name>/`.
  Open real-vs-sim gaps (latency, gains, joint offsets, fixture, pads, payload, cameras) and their
  tests: `docs/REAL_RIG_GAPS.md`.
- Real-arm deployment loop: `scripts/real/deploy.py` (`check` offline; `run --source hold|step|policy`):
  50 Hz sim action scheme, sim gains (kd capped at 5: DM MIT limit), joint offsets in the observation,
  CAN-level timing logs in `data/real_yam/`. It wraps i2rt driver methods to stamp readings and sends.
- Real-robot training config: `configs/task_h_reader_real.json` (measured per-joint friction via
  `EnvCfg.arm_frictionloss` = one [lo, hi] per joint, kp/kd scale, latency {0, 1}, 14-bit encoders).
  Policies on the real arm: `deploy.py run --source policy --policy <ckpt> --env_file <its config>`.
- Real arm joint offsets: the real YAMs read 0.4-1.3 deg off on joints 2-5 against the Menagerie
  kinematics (`rig.joint_offsets(side)`, spec `arms`): add them to measured joints before FK or a policy.
  The two wrist webcams have different lenses (right ~86x74 deg, strong barrel; left ~99x86 deg).
- Real wrist camera in sim: `EnvCfg.cam_spec = "blind_insertion/configs/real_cameras_duchamp3.json"`
  sets the wrist camera's fovy and mount pose from the spec's `wrist.sim` section and refuses a
  modality the real camera lacks (the real wrist cameras are RGB only). Null fields keep the D405
  and print a placeholder warning (fovy and pose need the camera calibration). The real side feeds
  `assets/real_camera.real_policy_image` (undistort, square crop on the principal point, resize).
- `load_robot` adds backlash hinges before `sphere_pads` (which compiles the spec): the other order
  fails with "more than 6 dofs in body link_1".
- Task H (SD reader -> mini PC USB-C): USB-C parts at spec dimensions (`builder.usbc_spec_variant`,
  `sample_reader(clearance="spec")`), top-down grasp `EnvCfg(grasp_mode="top")`, config
  `configs/task_h_reader.json` (written by `python -m blind_insertion.scenes.pc_bench --write_config`).
  ACT runs in a separate LeRobot venv `.venv-act` (torch cu126 build for the V100s): `scripts/act/`
  (gen_demos -> to_lerobot -> train_act.sh -> export_act -> TorchScript loaded by `pipeline/pick_insert.py`).
- Eval variants default to `variant_seed=1000` (unseen geometries from the training
  distribution; training uses seed 0). OOD bins: `builder.OOD_BINS`.

## Gear meshing (Phase G)

Paper's second task: a spur gear held by an 18 mm boss slides down a shaft into mesh with
1–2 hinged anchor gears (Table 1 gear column). Code: `assets/gear_builder.py` (variants,
`GEAR_OOD_BINS`, build), `envs/gear_env.py` (`GearCfg`, `GearEnv(PegEnv)`), `eval_gear.py`,
`checks/g_checks.py`; results in `results/G/`.

```bash
.venv/bin/python -m blind_insertion.train --task gear --name G-gear-proprio-base-s0 --seed 0 --nworld 12288
.venv/bin/python -m blind_insertion.eval --task gear --policy runs/<run>/last.pt --episodes 8192 --out results/G/x.parquet [--ood tighter|taller|much_taller|mild_size|extreme_size|random_tooth]
.venv/bin/python -m blind_insertion.checks.g_checks [--only render,drop,spin,backlash,scripted]
scripts/eval/phase_g.sh <run> <tag>; .venv/bin/python scripts/report/phase_g.py
```

- `--task gear` sets the gear defaults: kp/kd 40/20, action scale 0.04 rad, 10 s episodes,
  24 PPO minibatches, 128 contact labels (64 shaft rim + 64 tooth tips), critic 401 (+33).
  Checkpoints store `task` and `n_points`; `eval.load_policy` reads `n_points`.
- Fixture frame: origin on the shaft axis at the collar top (seat plane). Gear frame:
  bore centre of the bottom face. Seated <=> all rim points in the bore and gear z < 1 mm.
- Teeth share one mesh per gear (rotated geom slots); unused slots are parked 50 m (gear) /
  80 m (anchors) along the body z axis. Anchors are hinged children of the mocap fixture
  with per-world `body_pos`, `dof_damping`, `dof_frictionloss`.
- CPU rendering needs `geom_sameframe = 0` on slot geoms (done in `gear_builder.build`).
- Keep mesh polygons <= 24 vertices: MuJoCo Warp's multi-CCD buffers scale with the
  largest polygon x `naccdmax` (OOM at 12,288 worlds otherwise). `GearCfg` uses
  `nconmax` 96, `nccdmax` 48, `njmax` 640; the 12,288-world run takes ~10.6 GB.
- `GearCfg(pads_collide=False)` makes pads and housing collision-free (drop tests).
- `scripts/server/jobs/make_manifests.py --check` imports train.py: run it with
  `.venv/bin/python` (system python3 has no numpy).

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
`scripts/server/sync_down.sh`. `train.py` / `train_visual.py --resume` continue from `last.pt`
(optimizer, RNG and curriculum stage included); the runners add `--resume` for partial runs.
