# Real-to-sim: how the simulator matches the duchamp3 rig

Everything below is used by default in the bimanual runtime (`BimanualCfg`). Full numbers,
plots and reproduction commands: [`results/usbc/bimanual_20261029/README.md`](/results/usbc/bimanual_20261029/README.md).
Iterating on any of these belongs on the `sim/fidelity` branch ([branches](../operations/BRANCHES.md)).

## Arm dynamics

- The i2rt driver adds its own gravity feedforward; the sim now applies the same driver model
  (`blind_insertion/envs/arm_residual.py`) plus a per-joint gravity scale and Coulomb friction
  identified per arm from recorded motion (`scripts/real2sim/identify_arm_dynamics.py` →
  `results/usbc/bimanual_20261029/dynamics_id/profile.json`).
- MuJoCo's soft frictionloss crept under sub-threshold load; a stiff friction impedance stops it.
- Held-out tool-position error: left 8.5 → 4.4 mm, right 4.0 → 4.0 mm (CPU); GPU runtime replay of
  six trials 4.9–15.6 → 1.6–4.4 mm (`scripts/real2sim/check_bimanual_dynamics_replay.py`).
- Per-episode randomization: friction ×U(0.5, 1.3), gravity scale ±0.04.
- Not modelled: stick-slip (static > kinetic, 6–12 mrad slips seen in the backlash runs).

## Cameras and fingertips

- Camera intrinsics/extrinsics from the rig calibration (`blind_insertion/configs/real_cameras_duchamp3.json`).
- No FlexPoint CAD: the tip offset is fitted to wrist silhouettes from a gripper opening sweep
  (`scripts/real2sim/fit_gripper_tips.py --sweep`), held to calipers (lens-to-tip 105.25 mm on
  both arms). Held-out tip IoU left 0.70 → 0.89, right 0.59 → 0.81; open jaw 47–48 mm per finger.
- The remaining offset (≤ 4 mm) could be tip geometry or hand-eye, so it moves the wrist
  camera (`tip_offset_target='camera'`): images match, grasp physics unchanged.

## Warp look (what policies see)

The training renderer (MuJoCo Warp) lights the scene differently from the CPU renderer the fits
above used. `BimanualCfg.visual_look` (`configs/warp_look_duchamp3.json`,
`scripts/real2sim/fit_warp_look.py`) sets group colours and a per-camera gain and black level fitted
in Warp to 21 real views; alpha-0 geoms and the wrist lenses are left out of the Warp render.
Real vs Warp scores: `scripts/real2sim/measure_warp_look.py` (fingertip IoU 0.79, L1 19.6).
Per-world visual DR on top: `BimanualCfg.visual_dr` (`envs/visual_dr.py`).

## Background

`blind_insertion/scenes/backdrop.py` + `configs/backdrop_duchamp3.json`, fitted by
`scripts/real2sim/fit_backdrop.py` to 21 real ZED/wrist frames: wall, 8 cm raceway at the
table edge, plugs, power strip with cables, wall device, Ryobi case (edges 0.6–1.9 px at 320 px).
Before/after and side-by-side views: `scripts/real2sim/render_sim_real_alignment.py`,
`scripts/real2sim/render_layout_views.py`.

## Measured, not modelled

| Quantity | Value | Decision |
|---|---|---|
| Backlash | ≤ ~2 arcmin, mostly under camera noise (`data/backlash/README.md`) | not modelled (below encoder-offset noise and stick-slip) |
| PC–table friction | not measurable with the rig tools | μ U(0.3, 0.9) per world |
| Latency | ~8 ms (0.4 control step) | to add: 0–1 step in the bimanual runtime |
| Encoder offsets | 0.4–1.3° on joints 2–5 | to add: ±1.5° observation noise |

## Documents

| Doc | What |
|---|---|
| [PHYSICAL_YAM_SCENE](PHYSICAL_YAM_SCENE.md) | Scene reconstruction: meshes, mounts, cameras |
| [REAL_RIG_GAPS](REAL_RIG_GAPS.md) | Gap list with tests and pass criteria |
| [REAL_SIM_DATA_CONTRACT](REAL_SIM_DATA_CONTRACT.md) | Data exchanged between rig and server |
| [SIM_PHYSICS_AUDIT](SIM_PHYSICS_AUDIT.md) | Measured vs provisional physics inputs |
| [BIMANUAL_MEASUREMENT_PLAN](BIMANUAL_MEASUREMENT_PLAN.md) | Collection protocol for the bimanual rig |
| [REAL_YAM_LATENCY_BACKLASH](REAL_YAM_LATENCY_BACKLASH.md) | How latency and backlash were measured |
