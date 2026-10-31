# Code map

## Data flow (bimanual task)

```
data/real_yam, data/backlash, rig bundles ─> scripts/real2sim/*  ─> results/usbc/bimanual_20261029/{dynamics_id, ...}
                                                                     blind_insertion/configs/*.json
blind_insertion/scenes/pc_bench.build_scene(bimanual=True, measured_support=True)
   ├─ assets/builder.py        robots (YAM, FlexPoint fingers), USB-C reader, collision proxies
   ├─ scenes/physical_rig.py   calibrated cameras, joint offsets, reference pose
   ├─ scenes/measured_support  table and mounting plate from measurements
   ├─ scenes/backdrop.py       fitted wall / raceway / power strip
   └─ scenes/pc_rear.py        PC rear panel and port layout
        │
        v
blind_insertion/envs/bimanual_runtime.py   GPU worlds, CUDA-graph stepping, per-world randomization,
   │                                       driver-gravity residual (envs/arm_residual.py), task layout
   v
blind_insertion/envs/bimanual_task.py      actions, observations, rewards, termination
   v
blind_insertion/train*.py, policies/       PPO, TCN actor-critic with contact decoder  ->  runs/ (not in git)
```

## Package `blind_insertion/`

| Path | What |
|---|---|
| `assets/` | Part and scene builders: pegs, USB-C, gears; robots (Flexiv, YAM); FlexPoint fingers; backlash hinges, sphere pads; real-camera image pipeline (`real_camera.py`) |
| `scenes/` | Physical rig scenes (above); `controller_profiles.py` gains; `bimanual_collision.py` arm proxies |
| `envs/` | `bimanual_runtime.py`, `bimanual_task.py` (bimanual); `peg_env.py`, `gear_env.py`, `pick_env.py` (single-arm); `kinematics.py`, `camera_cadence.py`, `friction_comp.py`, `arm_residual.py` |
| `policies/` | TCN actor-critic with the contact decoder; camera encoders |
| `train.py`, `train_visual.py`, `eval.py`, `eval_gear.py` | PPO training and evaluation (single-arm; base for the bimanual trainer) |
| `checks/`, `analysis/`, `viz/` | Single-arm signal checks (A-S1..A-S10), probes and statistics, rollout replays |
| `pipeline/` | Single-arm task H: ACT pick → PPO insert |
| `configs/` | Env configs per phase; `real_cameras_duchamp3.json`; `backdrop_duchamp3.json` |

## Scripts, results, data

| Path | What |
|---|---|
| `scripts/` | Entry points by purpose: `real2sim/`, `task/`, `checks/`, `eval/`, `analysis/`, `report/`, `server/`, `real/`, `act/`, `tools/` ([scripts/README](/scripts/README.md)) |
| `results/usbc/bimanual_20261029/` | Current bimanual results: dynamics ID, alignment, backdrop fit, placement, layout views ([README](/results/usbc/bimanual_20261029/README.md)) |
| `results/{A..H}/` | Frozen single-arm phase results |
| `data/` | Small real recordings used by fits (step/probe CSVs, backlash, sim reference images); large captures are on `data/*` branches |
| `report/`, `report_usbc/` | HTML results reports (single-arm phases; USB-C) |
| `mjfit/`, `robosimstudio/`, `yam-control/` | Git submodules (do not edit upstream source) |
| `third_party/mujoco_menagerie/` | Sparse clone: YAM, Flexiv, Robotiq models |

## Reproducing the current bimanual results

```bash
export PYTHONPATH=$PWD MUJOCO_GL=egl
.venv/bin/python scripts/real2sim/identify_arm_dynamics.py          # -> dynamics_id/profile.json
.venv/bin/python scripts/real2sim/check_bimanual_dynamics_replay.py # GPU replay vs real
.venv/bin/python scripts/real2sim/fit_gripper_tips.py --sweep data/gripper_sweep/sweep_20261030   # needs the data branch
.venv/bin/python scripts/real2sim/fit_backdrop.py
.venv/bin/python scripts/task/explore_pc_placement.py               # -> placement/report.json
.venv/bin/python scripts/checks/check_bimanual_runtime.py
.venv/bin/python scripts/real2sim/render_layout_views.py
```
