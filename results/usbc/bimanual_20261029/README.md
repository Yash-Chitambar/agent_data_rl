# Bimanual reconstruction — integration in progress

This folder uses UTC date naming; collection was October 28 Pacific time.

`source_inventory.json` identifies the checksum-verified immutable capture bundle.
`calibration_provenance.json` matches all 24 capture calibration hashes. The user confirmed
there is no later adopted calibration to substitute.

## Implemented and checked

- Both arms actuated in the optional bimanual CPU scene; equal pad-contact settings.
- Arm-link and cross-arm collisions restored before compilation; synthetic positive and negative
  contact probes pass. Authored capsule geometry remains approximate.
- Explicit identification/native PD gain profiles prevent conflating temporary test gains with
  native free-settle gains. See `docs/research/KEYBOX_LESSONS_FOR_USBC.md`.
- Free PC with reported mass and dimensions; original connector geometry retained.
- Optional measured support: 480 mm wall-edge to stationary robot-base edge and 23 mm
  mounting rise. Base mesh bottom is the assumed mounting interface (-10 mm model z),
  hence table at -33 mm. Shared plate footprint is explicitly estimated, not surveyed.
- Detailed free-PC visuals preserve collision geometry, inertia and mass versus plain scene.
- `overlays/`: all three recorded camera views, calibrated preprocessing, host-arrival-matched
  encoder poses. Annotation supplies approximate PC lid center; no reader was visible.
  Images are diagnostic comparisons, not a passed photometric or geometric fidelity gate.
- `replay/`: twelve bilateral folded-rest baselines, three left free-settle baselines,
  and six left replays using logged computed driver feedforward. Folded-rest excited-joint
  errors are 2.5–5.5 mrad. Elevated free-settle 001 joint-3 error is 26.2 mrad overall.
  Logged feedforward improves left joint 3 (3.83 to 2.31 mrad), but worsens joints 2 and 4;
  it is not adopted as a validated plant/controller model. Right captures lack this field.
  All metrics are encoder comparisons, not independently measured tool accuracy.

## Reproduce

Run from repository root with the existing environment (do not replace CUDA packages):

```bash
PYTHONPATH=. .venv/bin/python scripts/checks/check_bimanual_scene.py
PYTHONPATH=. .venv/bin/python scripts/real2sim/replay_motion_cpu.py \
  /home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/motion/left_rest_j3_fit \
  --side left --source steps --out-dir /tmp/bimanual-replay-review --output-stem left_j3
```

Rendering: after the required idle-GPU checks, explicitly select one free GPU and run
`scripts/real2sim/render_bimanual_capture.py --bundle BUNDLE --out OUTPUT
--annotation results/usbc/bimanual_20261029/overlays/scene_annotation.json` with
`PYTHONPATH=.` and `MUJOCO_GL=egl`. Output includes per-camera frame IDs and encoder timing deltas.

## Still required

- Resolve visual geometry/material discrepancies using wrist views and independent frames.
- Refine plate footprint; validate full arm/body collision coverage and support behavior.
- Fit and validate gravity/friction across the now-replayed bilateral and elevated captures;
  include failed acquisitions and independent tool measurements where available.
- Identify gripper mapping/contact uncertainty without interpreting normalized position as force.
- Share measured profile and dynamic-port semantics with training environment, reset/rewards,
  camera timing and policy interface. CPU scene is not yet a complete training environment.
- GPU parity, numerical/contact checks and task rollout validation before policy training.

Legacy defaults and older policy worktrees remain separate. No new training launched.

## Measured bimanual GPU runtime

`blind_insertion.envs.bimanual_runtime.BimanualRuntime` now builds the measured
free-PC scene directly on MuJoCo Warp. Its 14D action is right six absolute model-frame
joint targets + gripper, then left six targets + gripper. Optional target-error
feedforward is side-specific and held for the control step; right priors cannot
silently populate the left profile. This remains additional compensation atop the
scene's ideal gravity baseline, not a validated driver/payload model.

`check_bimanual_runtime.py` verifies bilateral actuation, moving-port coordinates,
selective resets, finite stepping, right-only compensation and rejection of legacy
7D actions. `check_bimanual_warp.py` compares 50 one-millisecond steps in two worlds
against CPU MuJoCo: max generalized-position discrepancy 0.000162 (mixed metres/radians).
This is a short-horizon numerical check, not real-robot accuracy. The runtime still
needs task reward/termination, measured reset distributions, actor sensor timing and
camera rendering integration before training acceptance.

The runtime now optionally renders all three calibrated RGB policy views with
`BimanualCfg(render_rgb=True)`. `proprioception()` returns twelve encoder-frame
joint angles/velocities without privileged object poses; hardware-normalized
finger feedback is deliberately not fabricated. `runtime_rgb/` contains actual
Warp renders and checks for camera motion and independently owned frame tensors.
Rendering remains instantaneous; exposure/frame-age simulation is still pending.

The geometric insertion metric checks the entire sampled plug outline against
the live aperture, tilt, and both ends of the seating-depth band. Tests reject
lateral misses, front gaps and over-insertion and preserve results when the PC
and reader translate together. This is not a measured electrical connection test.

GPU review exposed coplanar collision/visual case surfaces causing lid speckling.
Detailed free-PC collision boxes are now in the non-rendered collision group;
contact geometry, masses and inertias are unchanged (construction checks pass).

## Episode semantics

`BimanualTask` wraps the runtime with normalized bilateral joint-increment actions,
continuous geometric-seating success, sparse reward, failure on falling objects or
nonfinite state, timeout truncation and explicit selective resets. Terminal
observations are returned before reset. Hold duration and action scale are design
parameters, not new measurements. The two-world GPU check covers timeout versus
termination, interrupted seating, selective reset and actor/critic separation.
It uses synthetic seated states to test logic and is not a physically solved task.
The fixed reference reset layout remains provisional.

`replay_legacy_motion.py` imports the earlier native-feedback CSV format without
changing raw captures. The right joint-2 step recording is now replayed using its
own gains, encoder offsets and actual send times. Unlike the newer recordings,
this CSV does not record velocity; initial velocity is assumed zero. Its metadata
retains the right-arm gravity-compensation and Coulomb terms for fitting.

## Right joint-2 and sensor corrections

The runtime now uses 1.0 N m frictionloss for **right joint 2 only**. It matches the
existing measured prior and improves both the older step test (10.17 to 6.61 mrad
RMSE) and newer folded-rest test (5.40 to 3.84 mrad). The 0.5/1.0/1.5 sweep and
source hashes are in `replay/friction_cross_pose/`. Left-arm parameters and driver
gravity compensation were not inferred from this result; they remain unresolved.

`camera_timing.json` measures delivered cadence from all post-startup frames:
about 19.95 FPS for both wrists and 27.28 FPS overhead, despite 30 FPS settings.
The reusable profile retains inter-camera timing, preprocessing delays and stalls.
With RGB enabled, the runtime replays this schedule at physics substeps, holds
published frames and exposes their age; reset generations prevent stale frames
from leaking into a new episode. Initial/reset images assume stationary pre-roll.
Host receive times are capture proxies, **not measured exposure times**. The
schedule repeats and is not yet randomized independently across worlds.

Actor joint positions now use native DM 16-bit quantization over +/-12.5 rad;
relative actions use those measured angles rather than ideal simulator angles.
Velocity noise/latency and gripper normalized-feedback mapping remain unmodeled.

`gripper_silhouette/` records a stationary opening sweep with fixed calibration.
Best color-mask overlap is only 0.47 right / 0.66 left. The fit is insufficient to
adopt a new gripper mapping or move camera transforms. These masks are diagnostic,
not independent metrology or a passed alignment gate.

## Identified arm dynamics (2026-10-30)

The runtime default is now the identified real-arm plant
(`BimanualCfg.dynamics_profile`, `dynamics_id/profile.json`). The earlier ideal plant
cancelled gravity exactly. The real i2rt driver adds its own gravity model
(factors 1/1.1/1.1/1.2/1/1, 0.5 kg gripper), which is wrong for these arms.

- Model per arm: plant gravity = `gravity_scale * g_scene`, driver feedforward
  `f_g * g_driver(q_enc)` (`arm_residual.DriverGravity`, matches MuJoCo to 3e-6 N m),
  joint frictionloss = effective Coulomb with a stiff impedance (solref 0.004,
  solimp 0.995/0.9995). Default MuJoCo frictionloss creeps a held joint by
  residual/kp in ~5 s; the stiff impedance leaves ~0.1 mrad/s.
- Fit (`scripts/real2sim/identify_arm_dynamics.py`): static torque balance on 46 free-space
  trials from Oct 25-26 (`data/real_yam`) and Oct 28 (bundle). Friction direction = last
  commanded move (the joints stick-slip). j2/j3 samples on the folded-rest stops excluded.
  Free 20- or 5-parameter inertial fits generalised worse across days than per-joint scales.
- Result: elbow (j3) gravity +11% right / +17% left; left j2 +12%; j4-j6 within 1-8%.
  Coulomb: left 0.31/1.02/1.44/0.11/0.10/0.12, right 0.03/0.55/0.82/0.05/0.04/0.05 N m.
  Left j2 friction was 0.40 on Oct 25-26 and 1.03 on Oct 28: the runtime samples
  friction x U(0.5, 1.3) and gravity scale +/-0.04 per world and joint on every reset.
- Closed-loop CPU replay of recorded commands (`replay_identified_dynamics.py`), mean
  grasp-site error vs real encoders, parameters fitted on the other day only:
  left elevated 8.47 -> 4.41 mm; right elevated 4.02 -> 3.99 mm (2.24 mm with all data;
  the right Oct 28 data has only two elevated trials). Folded rest gets worse
  (0.7 -> 3.0 mm): those poses sit on the joint stops, outside the task workspace.
- GPU runtime replay (`check_bimanual_dynamics_replay.py`, all-data parameters):
  six trials, legacy 4.9-15.6 mm -> identified 1.6-4.4 mm (`dynamics_id/gpu_replay.json`).
- Not modelled: stick-slip sawtooth (~10 mrad on j3; MuJoCo has one friction value),
  payload/contact loads, joint-stop position (sim stop is at model 0, real at encoder 0).
- Cost: none measured. But the bimanual runtime runs ~335 env-steps/s at 2048 worlds on a
  V100 in both versions (6.1 s per control step); this blocks PPO-scale training.

## Fingertip alignment (2026-10-30)

`scripts/real2sim/fit_gripper_tips.py` fits a rigid offset of both FlexPoint finger bodies
(gripper frame) and the opening per frame to the yellow-tip silhouettes in the calibrated
wrist crop. Left uses open and closed-empty frames (tips touching: absolute zero), right
open frames only. Held-out frames from `left_free_settle_001`:
left IoU 0.69/0.58 -> 0.91/0.77, right 0.47/0.48 -> 0.77/0.77. Offsets (thickness toward
camera, lateral, along finger): left -1.5/-0.5/-3.5 mm, right -2.5/-0.5/-1.5 mm.
The closed-empty frame fits opening 2 mm, so the opening map is not the problem.
The silhouettes cannot separate tip geometry from a hand-eye translation error (both move
near-field tips, not the far-field board). So the offset was opt-in
(`BimanualCfg(tip_offset='fitted')`); superseded by the sweep below. Measure camera-lens-to-tip distance with calipers
on both arms to decide: geometry -> enable it; camera -> correct the hand-eye instead.

### Gripper sweep and camera-side alignment (2026-10-30, later)

Rig sweep `data/gripper_sweep/sweep_20261030` (branch `data/gripper-sweep-20261030`; 11 jaw
levels x 3 frames per arm, arms at folded rest). Fit on the open and closed first frames, all
other frames held out (`results/usbc/gripper_sweep/sweep_20261030/`).

The unconstrained fit gave left -2.0/-0.5/-3.5 mm and right -7.0/-0.5/+2.0 mm, i.e. it moved the
right tips ~6 mm toward the lens and shrank the right open jaw to 43 mm. Calipers (jaws open, tip
to lens centre) read **105.25 mm on both arms**; the nominal model has both arms equal to ~1.5 mm
near that value. So the fit now keeps the offset along the lens -> tips direction within 1 mm
(`--lens-tol-mm`), and only moves the tips across the view:

| | offset (x thick, y lat, z finger) mm | held-out IoU before -> after (min) | open jaw per finger |
|---|---|---|---|
| left | -1.0 / -0.5 / -4.0 | 0.70 -> 0.89 (0.85) | 48 mm |
| right | -1.5 / 0.0 / -2.0 | 0.59 -> 0.81 (0.78) | 47 mm |

The arms now agree to ~2 mm and on the opening. Encoder -> opening per finger is linear and
monotonic: left 0.0482 m x enc (rmse 0.40 mm), right 0.0469 m x enc (rmse 0.28 mm). Commanded 0.02
stalls at encoder 0.04 (left) / 0.06 (right): the tips touch. The remaining offset could be tip
geometry or hand-eye, so the runtime applies it to the **wrist camera** by default
(`BimanualCfg.tip_offset_target='camera'`: camera moves by the opposite offset in `link_6`; tips in
the image identical to moving the fingers, checked to 0.01 mm; grasp physics unchanged).
`tip_offset_target='tips'` moves the finger bodies instead (2-4 mm along the fingers: inside the
in-hand pose randomization either way).

## Joint backlash and stiction (2026-10-30)

Measured on both arms (`data/backlash/README.md`): 0.04-2.3 arcmin, mostly under the 0.45-0.8
arcmin camera noise. At most 0.7 mrad, against stick-slip jumps of up to 10-14 mrad and +/-1.5 deg
encoder-offset randomization, so the bimanual runtime does **not** model backlash (the
`add_backlash` hinges would add 12 dofs for a sub-noise effect). The PD error at which a joint
rests after a closed-loop stair gives a static band at or below the identified Coulomb (right j3
0.49 vs 0.82 N m, left j3 0.56 vs 1.44), so the identified friction is not too weak. The real
stick-slip (static > kinetic, 6-12 mrad slips on j2, j3, j5) is still not modelled.

## Task layout: PC placement and reader region (2026-10-30)

`scripts/task/explore_pc_placement.py` searches PC port poses (x 15-40 cm, y -15-60 cm, yaw
+/-30 deg; 25,688 poses, 1 cm / 5 deg) on the CPU model. A pose is feasible when the right arm
reaches the second USB-C port (`usbc_2_target`) along the whole insertion line (60, 30, 6, 0 mm
out), also with +/-8 deg in-hand tilt, and the left arm can hold the PC (closed fingers 3 mm off
the rear face at mid height, or 3 mm above the top, rear half). It must also keep >= 0.05 rad
joint margin and have no arm-arm, arm-table or arm-PC contact (5 mm margin). Score:
min(right, left) smallest Jacobian singular value x joint-margin factor. The nominal maximises
the **worst** score over the per-episode randomization box (+/-3 cm, +/-10 deg).

- 21% of poses are feasible. Most failures: no left hold (left arm reach), then the right arm's
  60 mm approach point.
- **Nominal: port (0.22, 0.36) m, yaw 0 deg** (PC centre
  (0.269, 0.339) m, right arm base frame); left hold: press.
  Robust score 0.0334. Rotating the PC 20 deg toward the left arm scores
  0.0372 (+11%) at the same position; yaw 0 is kept because it is easy to set by hand.
- The legacy runtime pose (`PORT_XY` = 0.321, 0.237) is **infeasible** (the left arm cannot
  hold the PC). The real 2026-10-28 session placement (0.214, 0.331) is feasible but sits 1 cm
  from the edge of the feasible region, so its +/-3 cm box is not.
- Reader (SD/USB-C adapter) region, revised: every (2 cm cell, 20 deg yaw bin) pair, yaw -90..90 deg,
  where a top-down pick (80 mm approach) works, the reader footprint (plug included, +1.5 cm) stays
  clear of every PC pose of the training box and >= 1 cm inside the table's back edge, and it does not
  touch an arm at the reset pose. Cells need >= 3 feasible yaw bins: 490 cells, 0.20 m^2 (was one
  0.12 m^2 rectangle), including the middle of the table and a strip in front of the PC (only
  ~8-14 cm deep there: the PC front is 19 cm from the right base). The runtime samples uniformly
  over the feasible pairs (jittered within the cell and the bin).
  port at the nominal +/- `pc_pose_range` (3 cm, 3 cm, 10 deg) and the reader uniformly in its
  region. On 6 x 2048 worlds both settle within 0.2 mm in 1 s of hold, none non-finite.
- PC feet are now boxes (5 mm square, same height) instead of cylinders: MuJoCo Warp collided
  cylinder-box through single-contact CCD, which returned NaN for ~0.1% of worlds with a yawed PC. `task_layout=None` gives the legacy fixed poses.
- Not checked: paths between poses, the pick-to-insert carry, and the new grasp (in-hand pose
  only through the +/-8 deg tilt). Physical setup: put the port at the nominal, port panel facing the arm bases.

Views: `scripts/real2sim/render_layout_views.py` -> `layout_views/` (real | sim | overlay per camera; on the
ZED view the placement space drawn on the real and sim images: green = single placement works,
yellow = whole training box works, cyan = training box + PC at nominal, magenta = reader region).

## Fitted backdrop (2026-10-30)

`blind_insertion/scenes/backdrop.py` replaces the procedural workbench dressing (pegboard, tools,
monitors, generic raceway/power strip) with the objects the real cameras see: wall, silver
raceway, three plug pairs, power strip + cables, the black clip-on device and the Ryobi bit case.
Parameters (`configs/backdrop_duchamp3.json`) were fitted by `scripts/real2sim/fit_backdrop.py` on 21
real frames (ZED + both wrists; `bimanual_stationary_001`, `free_settle_001-003`) at the logged joint
angles. Each object's sim outline is fitted to Canny edges of the real frame (robots and tips
masked), then its colour to the real median colour. Mean edge distance at 320 px, initial guess ->
fit: raceway 4.9 -> 1.2, plugs 5.4 -> 1.2, power strip 4.5 -> 0.9, device 7.0 -> 1.6, cables 1.6 -> 0.7,
Ryobi 1.8 -> 0.6 px (`backdrop_fit/report.json`, `before.jpg`/`after.jpg`). Before/after overlays:
`alignment_20261030/` (OLD = procedural backdrop). Used by every `measured_support` detailed scene.
Second pass: pass 1 had put the raceway 3 cm behind the table edge and only 2 cm tall (a shelf,
which a top-down camera cannot tell from a vertical face); the sim then showed a white wall line in
the gap. The raceway front is now pinned to the measured table edge, its top kept above the plugs,
and raceway + plugs refitted: an 8 cm vertical face, edges within 1.9 px (plugs 1.2 px). A dark
block (`bd_gap`) also fills any table-raceway gap.
Limits: cables differ between sessions (fitted as an average); wall texture, the sign and the
far-left cable are not modelled; lighting is matched only through per-object colour.

## Runtime speed (2026-10-30)

The runtime stepped `mjw.step` from a Python loop: kernel-launch bound (334 env-steps/s at
2048 worlds, V100). Each substep is now replayed from a captured CUDA graph
(`BimanualCfg.cuda_graph`); the camera cadence and the per-control-step gravity residual
still run between substeps. Buffers shrank from 512 contacts / 3000 rows to 128 / 800 per
world (`nconmax`, `njmax`; random-action peak 8 contacts, 90 rows; `buffer_usage()`).
Throughput: 2048 worlds 28,486, 4096 38,616, 8192 44,256 env-steps/s (was 334; 8192 did
not fit before). All runtime/task/RGB checks pass; the real-trial GPU replay is unchanged.
Contact-rich insertion may need larger buffers: watch `buffer_usage()` in training.
