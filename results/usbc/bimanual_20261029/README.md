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
  native free-settle gains. See `docs/BIMANUAL_SLIDE_EVIDENCE_AUDIT.md`.
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
PYTHONPATH=. .venv/bin/python scripts/tools/check_bimanual_scene.py
PYTHONPATH=. .venv/bin/python scripts/tools/replay_motion_cpu.py \
  /home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/motion/left_rest_j3_fit \
  --side left --source steps --out-dir /tmp/bimanual-replay-review --output-stem left_j3
```

Rendering: after the required idle-GPU checks, explicitly select one free GPU and run
`scripts/tools/render_bimanual_capture.py --bundle BUNDLE --out OUTPUT
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
- Fit (`scripts/tools/identify_arm_dynamics.py`): static torque balance on 46 free-space
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

`scripts/tools/fit_gripper_tips.py` fits a rigid offset of both FlexPoint finger bodies
(gripper frame) and the opening per frame to the yellow-tip silhouettes in the calibrated
wrist crop. Left uses open and closed-empty frames (tips touching: absolute zero), right
open frames only. Held-out frames from `left_free_settle_001`:
left IoU 0.69/0.58 -> 0.91/0.77, right 0.47/0.48 -> 0.77/0.77. Offsets (thickness toward
camera, lateral, along finger): left -1.5/-0.5/-3.5 mm, right -2.5/-0.5/-1.5 mm.
The closed-empty frame fits opening 2 mm, so the opening map is not the problem.
The silhouettes cannot separate tip geometry from a hand-eye translation error (both move
near-field tips, not the far-field board). So the offset is opt-in
(`BimanualCfg(tip_offset='fitted')`). Measure camera-lens-to-tip distance with calipers
on both arms to decide: geometry -> enable it; camera -> correct the hand-eye instead.

## Runtime speed (2026-10-30)

The runtime stepped `mjw.step` from a Python loop: kernel-launch bound (334 env-steps/s at
2048 worlds, V100). Each substep is now replayed from a captured CUDA graph
(`BimanualCfg.cuda_graph`); the camera cadence and the per-control-step gravity residual
still run between substeps. Buffers shrank from 512 contacts / 3000 rows to 128 / 800 per
world (`nconmax`, `njmax`; random-action peak 8 contacts, 90 rows; `buffer_usage()`).
Throughput: 2048 worlds 28,486, 4096 38,616, 8192 44,256 env-steps/s (was 334; 8192 did
not fit before). All runtime/task/RGB checks pass; the real-trial GPU replay is unchanged.
Contact-rich insertion may need larger buffers: watch `buffer_usage()` in training.
