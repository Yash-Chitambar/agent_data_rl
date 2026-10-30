# Bimanual simulator handoff

Branch: codex/bimanual-calibrated. This is an experimental training baseline,
not a validated sim-to-real insertion model. Existing older training is unchanged.

## Included

- Measured PC/reader geometry, preserved USB-C plug and protected port aperture,
  rear-panel image registration, 23 mm raised bases and 480 mm table-edge offset.
- Adopted camera/arm calibration, camera cadence and observation pipeline.
- Identified per-arm gravity residuals and friction, including right-arm data;
  native gains and bounded accumulated joint targets.
- Free PC, both arms, collision proxies and approximate FlexPoint pads.
- Corrected nominal grasp height and settling before closing in the CPU pick
  diagnostic. Nominal carry retains the reader without an artificial attachment.

See BIMANUAL_POLICY_INTERFACE.md for the policy contract and existing checks.
Do not reuse a checkpoint expecting the old observation/action interface.

## Final grasp audit

Run from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/tools/check_bimanual_grasp_stability.py results/usbc/bimanual_20261029/reach/settled_clearance_identified_feedback_nominal_pick_states.npz
```

The final reader-center shift relative to TCP is 0.974 mm, but its relative
rotation is 27.913 degrees from closure. Therefore position-only retention is
not evidence of a stable grasp. The new check explicitly reports this failure.
Its 1 mm / 1 degree thresholds are engineering diagnostics, not hardware-derived
tolerances. It samples phase endpoints, not every integration step.

Final privileged-feedback preinsertion error is about 0.48 mm and 0.37 degrees
at a 6 mm gap. This exceeds the insertion clearance; it is not successful insertion.
The feedback uses simulator object state and is not a deployed visual policy.

Physical evidence in the curated capture bundle:
- grasp/left_reader_001.json: failed retention.
- grasp/left_reader_002.json: explicitly invalid placement/orientation.
- grasp/left_reader_003.json: operator-confirmed retention, readback 0.33615.
- reference/right_arm_20261027/motion/reader_grasp_003.json: operator confirms
  no slipping during a 20-second powered hold.
- reader_grasp_006.json in the same directory: retained during five 0.005 rad
  elbow triangle repetitions; normalized readback range 0.3927–0.3928.

These observations do not identify squeeze force, torsional friction or pad
compliance. Driver effort is explicitly uncalibrated. No friction parameter was
increased merely to hide the simulated rotation.

## Before a long training run or hardware transfer

Experimental interface/reward testing can use this baseline. Full insertion
training readiness remains unproven: validate stable grasp/contact behavior,
left-arm bracing and successful nonpenetrating insertion, then GPU/runtime parity
for that rollout. Current identified-model throughput (~335 aggregate environment
steps/s at 2048 worlds on the measured V100 run) also needs to be budgeted before
a long PPO run. Reset diversity and background realism remain incomplete.

Use reach/identified_carry_comparison.png under results/usbc/bimanual_20261029
for the latest diagnostic renders. They are simulator renders, not proof of
physical alignment or real-world success.
