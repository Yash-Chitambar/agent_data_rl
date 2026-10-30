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

## Original grasp audit (superseded by solver correction below)

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

## Contact-creep correction

The measured bimanual scene now uses MuJoCo `impratio=100`. This increases
tangential constraint impedance relative to normal contact impedance; it does
not change Coulomb coefficients, pad geometry or motor squeeze. The setting is
supported by the installed Warp model importer. Other scenes are unchanged.

Controlled CPU comparisons with the same identified arm dynamics:
- Stationary 20-second hold plus one second: default impedance produced 30.336°
  relative rotation; CPU noslip diagnostic produced 0.218°; impratio=100 produced
  0.295°. This isolates numerical contact creep from alignment feedback.
- Full nominal carry with impratio=100: 0.104 mm center shift and 0.281° relative
  rotation from closure, no grasp loss.
- Final privileged-feedback preinsertion: 0.0187 mm lateral error, 0.00788° tilt,
  5.9978 mm gap. This is alignment at a gap, not insertion success.

Reproduce the stationary comparison using audit_bimanual_pick.py with
`--identified-dynamics --settle-grasp --passive-hold`, adding
`--diagnostic-impratio 1` for the old solver or `--diagnostic-noslip` for the
CPU-only diagnostic. For carry use `--identified-dynamics --settle-grasp
--clearance-carry`. No hardware-derived stiffness is claimed.
Full GPU grasp/carry parity, insertion and bracing remain required.

## Nominal CPU insertion validation

The corrected solver also completes a scripted insertion using privileged
reader/port feedback. Reproduce with:

```bash
PYTHONPATH=. .venv/bin/python scripts/tools/audit_bimanual_pick.py --identified-dynamics --settle-grasp --clearance-carry --insert
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/tools/render_bimanual_carry_audit.py --insertion
```

At the final hold: depth 5.7779 mm (target 5.8), lateral error 0.0161 mm,
tilt 0.0281°. Geometric seating persisted for 1.4 s at 20 ms sampling,
exceeding the task's 0.2 s criterion. The runtime's actual insertion_metrics
function independently accepts the saved insert/seat_hold endpoints and rejects
the preceding 6 mm-gap state. No reader attachment or PC fixation was used.
PC displacement from preinsertion to final hold was about 18 micrometers;
minimum sampled reader-contact distance was -0.287 micrometers.
Contact penetration/force statistics are sampled at 20 ms, not continuous maxima.

The left arm stayed at its reference pose: this demonstrates nominal insertion,
not bimanual bracing, randomized robustness, GPU task parity, electrical engagement
or hardware success. The older warning about *nominal CPU* insertion being
unproven is superseded by this result; those broader validations remain open.
Artifacts use the insert_settled_clearance_identified_feedback prefix in
results/usbc/bimanual_20261029/reach.

## GPU seated-contact hold

check_bimanual_contact_hold.py initializes the actual training runtime from saved
CPU aligned/seated qpos, qvel and actuator controls, disables parameter
randomization, and holds the same joint targets for one second. Both engines use
identified bilateral gravity/friction; the GPU uses the runtime CUDA graph path.
The seated GPU world satisfies geometric seating at all 50 control samples.
Final GPU depth is 5.7962 mm, CPU depth 5.7352 mm; both satisfy seating.
Peak raw qpos difference is 0.00388 (mixed coordinates); the 0.005 guard is only a
coarse numerical-divergence guard, not an alignment tolerance. Buffer utilization
is 10.94% contacts and 14% constraint rows. All states remain finite.

This is a seeded contact hold, not an end-to-end GPU rollout or a randomized
robustness evaluation. The next remaining integration check is GPU pickup and
insertion with feedback throughout, followed by bimanual bracing/reset coverage.
