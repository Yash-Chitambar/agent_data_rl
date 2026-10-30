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
