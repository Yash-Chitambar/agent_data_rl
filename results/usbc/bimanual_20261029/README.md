# Bimanual reconstruction — integration in progress

This folder uses UTC date naming; collection was October 28 Pacific time.

`source_inventory.json` identifies the checksum-verified immutable capture bundle.
`calibration_provenance.json` matches all 24 capture calibration hashes. The user confirmed
there is no later adopted calibration to substitute.

## Implemented and checked

- Both arms actuated in the optional bimanual CPU scene; equal pad-contact settings.
- Free PC with reported mass and dimensions; original connector geometry retained.
- Optional measured support: 480 mm wall-edge to stationary robot-base edge and 23 mm
  mounting rise. Base mesh bottom is the assumed mounting interface (-10 mm model z),
  hence table at -33 mm. Shared plate footprint is explicitly estimated, not surveyed.
- Detailed free-PC visuals preserve collision geometry, inertia and mass versus plain scene.
- `overlays/`: all three recorded camera views, calibrated preprocessing, host-arrival-matched
  encoder poses. Annotation supplies approximate PC lid center; no reader was visible.
  Images are diagnostic comparisons, not a passed photometric or geometric fidelity gate.
- `replay/`: left six-joint folded-rest baseline replays using actual sent targets, native
  feedback and logged gains; outputs kept outside source bundle. Excited-joint RMSE is
  3.2–5.5 mrad, peak error 6.4–11.6 mrad. No fit adopted; no contact or elevated-pose claim.

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
- Replay right-arm and elevated/failed acquisitions; separate driver feedforward from friction.
- Identify gripper mapping/contact uncertainty without interpreting normalized position as force.
- Share measured profile and dynamic-port semantics with training environment, reset/rewards,
  camera timing and policy interface. CPU scene is not yet a complete training environment.
- GPU parity, numerical/contact checks and task rollout validation before policy training.

Legacy defaults and older policy worktrees remain separate. No new training launched.
