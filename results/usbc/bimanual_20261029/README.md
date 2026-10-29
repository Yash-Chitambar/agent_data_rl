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
