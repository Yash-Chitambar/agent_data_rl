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
correction. All six older right rest captures lack computed feedforward. The optional
KDL reconstruction also requires an unavailable `i2rt` import in this environment;
no replacement driver was installed or adopted. See phase-level JSON reports under
`results/usbc/bimanual_20261029/replay/`.

## Release status

Not yet validated for the next bimanual policy training run. Remaining work includes
gravity/friction identification on held-out poses, gripper/contact uncertainty,
visual registration discrepancies, and integration of the measured scene into the
training reset/action/reward/camera pipeline with GPU parity checks. No new training
or physical robot motion was launched during this audit.
