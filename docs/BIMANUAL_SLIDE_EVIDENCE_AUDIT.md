# Applying the key-box lessons to the USB-C rig

Reviewed `docs/keybox_sim2real.pptx` (29 slides) against
`origin/codex/bimanual-contact-20261028:docs/LEFT_CONTACT_COMMISSIONING.md`
and the immutable `left_bimanual_session_20261028_curated_v2` bundle.
The deck is evidence from a different task, not USB-C metrology.

| Slide lesson | USB-C implementation or decision | Evidence still needed |
|---|---|---|
| 8: verify reach before policy learning | Both calibrated arms actuated; arm–PC and cross-arm collision proxies restored before compilation | Full task reach/collision sweep, loaded poses |
| 9–10: gravity, friction, controller gains | Separate recorded identification and native free-settle PD profiles; replay sent targets against native encoders; support logged computed driver feedforward | Held-out pose/velocity/direction validation; payload and independent tool motion |
| 10: capped integral | Replay already executed trim commands without applying integral a second time; do not adopt the candidate for contact | Lower-pose tests 002/003 failed the existing attainment gate |
| 12–14: realistic sensor failures | Keep the adopted USB-C camera calibration and preprocessing; USB-C wrist cameras are RGB-only | Actual exposure latency, RGB statistics and any chosen ZED depth pipeline measurements |
| 15–17: matched views and object geometry | Measured PC/reader dimensions, rear-panel photo registration, three-view real/sim overlays | Independent port metrology and pad silhouette agreement; current overlay is not an accuracy certificate |
| 19–20: stable objects and gripper contact | Measured reader mass; protected original USB-C head; FlexPoint visual/collision approximation | Pad CAD unavailable, closing-width mapping, squeeze and compliance measurements |
| 21: insertion friction | Retain geometric insertion constraints; do not invent key-pin friction for USB-C | Force/displacement or bounded insertion/contact trials |
| 22: support physics | Free 284 g PC with approximate feet; measured 23 mm mounting rise and 480 mm table-edge clearance | PC sliding/rocking friction and bracing evidence; do not freeze the PC because the key box was held |
| 23: tab mechanics | Not applicable to USB-C | None |

## Controller profiles

`build_scene(bimanual=True, controller_profile=...)` accepts:

- `identification_20261027` (existing bimanual default): kp 80/80/80/30/30/30,
  kd 5/5/5/2/2/2. These are the **applied** identification gains, not `gains_before`.
- `native_free_settle_20261028`: kp 80/80/80/40/10/10,
  kd 5/5/5/1.5/1.5/1.5. Both arms' controllers recorded these in free-settle trials.

Only the PD coefficients change. Plant friction and driver feedforward are different
quantities. The recorded driver's Coulomb compensation must not be copied into plant
frictionloss. Likewise, the deck's 1.1–1.2 Nm and 35–40 N squeeze do not establish values
for this rig. Scene-perfect gravity compensation remains a baseline approximation.
Legacy single-arm scenes and old training defaults are preserved.

## Reproduction and interpretation

```bash
PYTHONPATH=. .venv/bin/python scripts/tools/replay_free_settle.py \
  /home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/contact/free_settle_001 \
  --out-dir /tmp/new-free-settle-review
```

The adapter verifies the calibration hash, uses executed (not nominal) commands,
applies encoder offsets exactly once, retains CAN send and feedback timestamps,
and reports errors by phase. Raw captures are not modified. Candidate trim is
already present in the recorded commands; replay does not certify that controller.

`replay_motion_cpu.py --logged-feedforward` consumes the recorded
`computed_motor_feedforward_driver` field without importing a different driver.
That field is the last computed command, not measured torque, and its exact
execution timestamp is unavailable. Replay holds it at observation timestamps.
Missing/nonfinite values are errors, never silently replaced by zero.

Folded-rest errors alone are insufficient: the elevated free-settle 001 baseline
has joint-3 RMSE about 26 mrad overall and 28 mrad during baseline hold, considerably
larger than folded-rest identification. Logged feedforward reduces left joint-3 rest
error from 3.83 to 2.31 mrad, but increases joint 2 from 4.70 to 9.05 and joint 4
from 5.52 to 10.58 mrad. This mixed result rules out claiming a general dynamics
correction. All six older right rest captures lack computed feedforward. The original KDL import requires unavailable CAN dependencies. An explicit offline
XML export now supports reconstruction without loading hardware drivers; see the
cross-pose candidate assessment below. No replacement hardware driver was installed. See phase-level JSON reports under
`results/usbc/bimanual_20261029/replay/`.

## Release status

Not yet validated for the next bimanual policy training run. Remaining work includes
gravity/friction identification on held-out poses, gripper/contact uncertainty,
visual registration discrepancies, and integration of the measured scene into the
training reset/action/reward/camera pipeline with GPU parity checks. No new training
or physical robot motion was launched during this audit.

## Contact-branch integration follow-up

The contact branch contains the right-arm evidence in `results/real/step_fit.json`,
`data/real_yam/steps_joint2_meta.json`, and FINDINGS entries for October 25–27.
Right joint-2 effective friction was fit at 1.0 N m (joint 3 at 1.5 N m);
the driver separately used gravity factors 1.1 on joints 2/3 and 1.2 on joint 4.
These are not interchangeable terms. The right-arm target-error feedforward law
from commit b151281 is now available in `envs/friction_comp.py`, with validation
and a guard against implicitly applying the right-arm prior to the left arm.
It is not automatically enabled or declared a validated bimanual dynamics model.

Reviewed contact-branch geometry fixes are integrated: rubber-pad visual sheets,
contact planes derived from the upstream tip mesh, and low-vertex yellow-tip
collision hulls. Sphere pads now choose their inward face with the gripper half
open, and avoid stale MjSpec geom lookup. Opposing-arm collision masks include
these new tip hulls. CAD-derived geometry is still not a measurement of pad
compliance or squeeze force.

World background dressing now follows the measured tabletop height; previously
it remained at its old z=0 datum, leaving the power strip floating. The subpixel
principal point after integer-centered cropping is retained, with an EGL marker
projection check for all three cameras (`check_principal_point.py`). Updated
views are in `overlays_pad_correction/`; no calibration transform was refitted.


## Offline driver reconstruction and cross-pose assessment

The archived clean i2rt source at commit
`4c43932cdb8e74dcb576419a81a76b47b90f2705` can now be exported with
`scripts/tools/export_driver_gravity_model.py`. The exporter loads only XML
construction definitions, removes rendering geometry after compiling explicit
inertias, and verifies static inverse dynamics against the original compiled
model at 100 poses. Maximum torque difference was zero. The export, provenance,
and upstream MIT license are under `results/usbc/bimanual_20261029/driver_model/`.
This verifies export fidelity, not the deployed revision or physical link masses.

The recorded configuration uses linear_4310 with a 0.5 kg gripper mass override,
gravity factors [1,1.1,1.1,1.2,1,1], and velocity-sign compensation
[0.3,0.3,0.3,0.06,0.06,0.06] N·m. These are driver settings, not measured
plant friction. Across 7,800 left folded-rest samples, the difference between
captured computed feedforward and reconstructed gravity has the configured
friction magnitude with worst per-run/per-joint RMSE 0.000410 N·m.
Its sign does not reliably match the separately observed velocity near rest:
the fields are asynchronous and velocity is quantized. This is not measured
motor-torque accuracy, nor evidence for a fixed feedforward delay.

Replay now accepts `--driver-xml` with either reconstruction mode.
`--recorded-state-feedforward` uses recorded encoder states and holds the
reconstruction between observations. It replaces ideal scene gravity
compensation; it does not add it a second time. That replay is an offline
diagnostic, not a deployable policy observation.

The driver-only candidate regresses important joints across poses. Adding the
right friction prior [0.01,1,1.5,0.2,0.2,0.1] N·m improves right pickup joint-3
RMSE from 26.62 to 12.88 mrad and joint-2 from 4.06 to 2.70 mrad. However,
folded-rest joint-4 error rises from 2.29 to 8.93 mrad and joint-2 from 5.40
to 5.67 mrad. Left elevated joint-3 also remains about 27 mrad with driver-only
reconstruction. These comparisons do not support adopting the combined model
as the default. The previously adopted right joint-2-only friction correction
remains separate.

See `replay/driver_cross_pose/comparison.png` and JSON/CSV reports for all
six joints. Further work must distinguish physical link/payload dynamics,
stateful friction, and asynchronous driver feedback using held-out poses.
Do not hide these errors by assuming perfect compensation or tuning camera
transforms to absorb dynamic sag.
