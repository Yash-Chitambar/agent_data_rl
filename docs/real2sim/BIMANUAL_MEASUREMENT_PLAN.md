# Physical measurements for the next bimanual simulator

Prepared 2026-10-27. Baseline: `real-yam-setup` commit `7059925`, integrated on
`codex/bimanual-calibrated`. This is a collection protocol, not a declaration that the
simulator is validated or that new hardware test runners have been implemented.

Goal: insert the measured SanDisk reader into the **second USB-C port from the left**
when looking at the GMKtec rear panel. Preserve the original simulated plug dimensions;
do not scale the connector to make insertion easier.

## What is already available

Reuse these measurements unless hardware has moved or changed:

- Both wrist intrinsics/distortion, camera-to-link-6 transforms, joint zero offsets,
  left-base-to-right-base transform: `blind_insertion/configs/real_cameras_duchamp3.json`.
- Refitted ZED HD1080 rectified-left pose; independent camera checks and their residuals.
- Both arms' step/hold logs, approximately 8 ms command-loop latency, encoder timing,
  and elbow breakaway trials: `data/real_yam/`, `results/real/`, `docs/campaign/FINDINGS.md`.
- Reader overall length 90.46 mm, body width 34.73 mm, neck width 12.87 mm,
  body thickness 8.83 mm from calipers. Other reader dimensions remain estimates.
- FlexPoint visual assets exist, but the real contact-pad geometry and compliance are
  not identified. No user CAD is available. Calipers and photographs are sufficient
  to improve this; CAD is not a prerequisite.

Earlier decisions deferred pad measurements and backlash tests. They are optional below,
not silently reinstated as prerequisites. Camera recalibration is not necessary just
because this checklist exists. A few-millimetre camera agreement does **not** establish
sub-millimetre connector alignment accuracy.

## Priority and tool availability

| Priority | Measurement | Current collection support | What it changes in sim |
|---|---|---|---|
| P0 | Elbow breakaway, sliding, stopping and reversal | Existing step/hold runner; ramps and richer telemetry need extension | Stateful friction model per elbow, parameter uncertainty |
| P0 | Target-port frame, table plane, PC mounting and start distribution | Manual survey / board observation; no automatic port solver | Fixture transform, collision alignment, resets, movable vs fixed PC |
| P0 | Actual gripper gap, closing speed, grasp/slip behavior | Manual measurements; dedicated command/logger needed | Finger mapping, actuator limits, contacts and grasp physics |
| P0 for visual policy | Frame age, camera skew, drops and processing | Existing capture tools; synchronized timing logger needed | Observation delay, sample-and-hold and camera preprocessing |
| P1 | Reader/PC mass, dimensions; wrist payload | Scale/calipers, manual log | Inertia, support, collision geometry and gravity compensation |
| P1 | Representative-pose dynamics and loaded holds | Existing single-arm step/hold runner | Gains, residual gravity, damping, load dependence |
| P1 | Insertion force/depth and support friction | Manual force gauge/load cell procedure | Contact, seating and extraction model |
| P1 | Both-arm timing and repeatability | Needs coordinated logger; do not launch competing controllers | Command skew, synchronized observations, collision validation |
| P2 | Mechanical lost motion, thermal drift, lighting variation | Optional indicator/telemetry/image sessions | Remaining randomization ranges |

If time is limited, collect P0 plus object/payload masses first. Keep the existing tests
as baseline data; collect new poses and held-out conditions rather than repeating everything.

## Before running motion tests

Follow `scripts/real/README.md`: secure both bases, keep the stop control available, and
use `safety.SafeArm`. Use one controller owner per arm. Check the selected start pose,
return path and other arm clearance locally; an old test pose is not universally safe.
Keep the reader/PC away during free-space tests. Do not disable limits or increase gains
just to force an elbow to move. Stop a trial for unexpected motion, oscillation, contact,
tracking-limit stops or thermal/current limits; retain and label that trial.

Existing `deploy.py run` moves hardware, including start/return motions. New ramp,
gripper, force and synchronized bimanual tests below are **procedures requiring an
appropriate runner**, not modes already present in `deploy.py`.

## 1. Elbow friction identification (both arms)

### Collect with the existing runner

Use the same gains as the recorded calibration: kp `[80,80,80,30,30,30]`,
kd `[5,5,5,2,2,2]`. `--gains sim` applies the hardware kd cap; do not request kd=6
on joints 1–3. Do not pass an accumulated-action config for step/hold tests: the
runner rejects it, and these tests intentionally send fixed targets.

From the repository root, these are examples for a **locally checked clear pose**:

```bash
# In a separate terminal: existing rig console / stop control.
.venv/bin/python scripts/real/rigctl.py

# No --start: hold the current pose. This still connects and commands the arm.
.venv/bin/python scripts/real/deploy.py run --side left --source hold \
  --gains sim --seconds 60 --name left_hold_session01

# Small step example. Review the entire start/return motion before running.
.venv/bin/python scripts/real/deploy.py run --side left --source step \
  --gains sim --joint 3 --amp 0.01 --reps 5 \
  --start 0.0 1.2 1.0 -0.6 0.0 0.0 --name left_j3_pos001_poseA_session01
```

The runner holds one second, steps by `amp` for one second, returns for one second,
and repeats. It is not a constant-speed test. Negative `--amp` reverses the excursion;
the local motion envelope must permit it. Change `--side` to test the right arm and
use a unique `--name` for every run to avoid overwriting logs.

At each selected pose, start with small commands. If the checked envelope permits,
collect positive and negative excursions around the observed transition (existing
0.04, 0.06, 0.08, 0.10 rad data brackets it); refine in 0.005–0.01 rad increments.
Use at least five repeats per direction. Do not automatically sweep larger amplitudes.
Test a compact pose, a representative pick pose and a representative pre-insertion pose;
record the actual joint vectors. Include an empty gripper and the normal reader load.
Reserve one pose and some repeats for validation, not fitting.

Existing outputs: `data/real_yam/<name>.csv`, `timing_<name>.csv`, and `<name>_meta.json`.
The encoder CSV contains `t`, `q_cmd_1..6`, `q_1..6`, `t_cmd`; timing is monotonic host time.
**Current CSVs do not log motor current, torque or temperature.** Do not interpret
`kp * position_error` as measured output torque: damping, feedforward, saturation and
transmission behavior also contribute. Preserve absolute targets: after stiction, the
previous commanded target is not the measured start position.

### Additional friction tests needing a runner extension

1. Slow bounded target ramps from rest in each direction, then stop. Record the first
   sustained encoder motion above the measured stationary noise/quantization floor.
2. Bounded triangular target trajectories at several rates spanning task motion;
   measure actual joint velocity rather than assuming commanded velocity is achieved.
3. Stop for different dwell times, restart, then reverse. This identifies re-sticking
   and history dependence. Keep travel bounded independently of duration.
4. Repeat a subset after normal warm-up, recording temperature if available.

Log raw position/velocity, absolute target, commanded velocity/feedforward, actual gains,
current/torque estimates with units and their documented interpretation, temperatures,
saturation/limit flags, and receive/send timestamps. Use native feedback rate where
available; do not manufacture high-rate observations by repeating 50 Hz samples.
If current/torque telemetry is unavailable, mark it null and retain uncertainty in the fit.

Import: fit static/sliding resistance, velocity transition, re-sticking and viscous terms
per arm, together with delay/gain uncertainty. Compare against a constant-friction baseline.
Validate onset time, position/velocity trajectories, final error, reversals and near-zero
creep on withheld trials. A candidate is not accepted just because it matches one threshold.
The exploratory two-state fit in this branch is not a validated training model.

## 2. Table, PC, target port and reader placement

- Define the world as the calibrated **right arm base**. Record table plane, height,
  slope, base plate heights and table edges in this frame, with measurement uncertainty.
- Survey the target receptacle center, insertion axis and orientation; also measure the
  spacing of the two USB-C ports, panel recess and housing dimensions. Photograph the
  rear panel straight-on with a scale. Do not insert caliper tips into electrical contacts.
- Use a board/tag rigidly associated with the PC plus a measured tag-to-port transform,
  or an independently calibrated probe. A camera-estimated PC bounding box alone does
  not determine the port accurately enough. Repeat from multiple views and report spread.
- Record whether the PC is free, clamped, taped, or held by the second arm. Measure support
  movement during representative contact. A free PC must not remain an immovable mocap body.
- Record at least 20 normal manual placements of reader and PC if starts vary. Save poses,
  not only min/max, to retain correlations. If placements are fixed, measure repeatability.
- Save the reader's full geometry: body length, neck length/thickness, taper, corner radii,
  connector protrusion and connector center relative to the body. Distinguish measured
  values from photo estimates. Preserve the existing plug model; report any measured
  discrepancy for review rather than silently changing its dimensions.

Import: table/fixture transforms, reset distributions, visual/collision meshes and target
frame. Report alignment uncertainty relative to connector clearance; do not treat “within
2 mm” as adequate validation for USB-C insertion. Camera calibration residuals are separate.

## 3. FlexPoint grippers (no CAD required)

For each gripper independently:

- Measure inner pad gap at closed, intermediate and fully open commands, approaching each
  point from both directions. Use a ruler/calipers only while safely stationary.
- Measure pad contact width/height, mount location, thickness, material and surface covering.
  Take front/side views with a scale and wrist-camera views at matching encoder poses.
- Record open/close commands and gap versus time without an object, then around the reader.
  Log any force/current setting, readback, position limits and stall behavior. Do not assume
  the two fingers move symmetrically or that commanded force equals measured pad force.
- If a suitable gauge is available, measure normal gripping force versus command and
  pad deflection at several loads. Record loading and unloading curves.
- With the reader supported to prevent a drop, measure tangential pull force at slip for
  several grip settings/contact locations. Repeat at least five times. Friction inferred
  from pull force requires the **measured sum of normal contact forces**.

Import: command-to-gap mapping, speed/force limits, contact location, effective stiffness,
damping and friction. Rigid contact spheres alone do not establish flexible-pad realism.
If deferred, keep these parameters explicitly estimated and randomize their uncertainty.

## 4. Mass, payload and residual dynamics

Weigh the reader and PC. Record the scale resolution and repeat readings. Measure object
center of mass where practical; otherwise label the geometric-center assumption. Inertia
can initially be computed from measured mass and geometry; it is not a direct measurement.

For each wrist record gripper, webcam, bracket and supported cable mass, center of mass,
and cable routing/tie points. Avoid disturbing calibrated camera mounts solely to weigh them;
use component measurements/spares where possible. Recheck extrinsics if a mount moves.
Record the hardware gravity-compensation payload configuration so sim does not compensate twice.

Use existing hold tests at several clear poses with empty and normal payload conditions.
Add representative small step tests on other joints only where existing data does not cover
pose/load/speed. Log actual gains and available torque/current, including stationary bias.
Import: mass/inertia, effective gains/damping and residual gravity/cable loads. A pose-dependent
bias should not automatically be attributed to Coulomb friction.

## 5. Insertion and support contact (manual first)

Use a supported, aligned reader and suitable low-range force gauge/load cell if available.
Record force versus measured insertion depth and extraction depth, slow speed, repeats,
PC motion, reader orientation and whether the connector is powered. Follow the device's
mechanical limits; do not deliberately jam, force misalignment or run a robot search sweep.
A video alone cannot identify contact force. Without a gauge, record only observable motion
and seating depth, and leave force parameters unmeasured.

Measure reader-on-table and PC-on-table starting/sliding resistance with a horizontal pull
near the support plane; note tipping or rotation rather than fitting those as pure sliding.
Test the actual desk surface and feet. Record material condition and normal load.

Import: support friction, contact compliance, seating travel and any retention force.
Avoid adding a fictional attractive “latch” simply to improve simulated success. Validate
realistic seating and withdrawal separately from visual plug overlap.

## 6. Camera pipeline and synchronization

Keep the calibrated capture modes: wrists 1280x1024, ZED rectified-left HD1080 1920x1080
unless deliberately recalibrating a different mode. Record pixel format, FPS, exposure,
gain, white balance, focus and auto-control settings using supported camera controls.
Save raw frames plus the exact processed policy frames, crop/resize/color conversion and
calibration file. Wrist undistortion and each camera's principal-centered crop matter.

For every camera log frame ID, exposure/capture timestamp if available, host receive time,
policy consumption time, dropped/repeated frames and clock domain. CPU arrival time is
not exposure time. Measure distributions under the intended inference/capture workload.
Use a common visible timed LED event for all cameras and timestamp its trigger; estimate
clock offset if acquisition uses multiple hosts. A phone timer without synchronized timing
only gives an approximate age and cannot establish absolute milliseconds of latency.

Collect paired frames at stationary logged joint poses, a board verification sequence at
new positions, and representative moving-arm sequences for blur/occlusion. No refit should
use the held-out verification frames. Existing verification command (can move arms):

```bash
.venv/bin/python scripts/real/verify_cameras.py --out data/calib/verify_session01
```

Import: distinct camera rates, sample-and-hold, frame-age/jitter distributions, inter-camera
skew, RGB preprocessing and lighting/exposure variation. The approximately 8 ms command
latency is **not** the complete image-to-action latency. Do not double-count delays.

## 7. Bimanual operation and optional residual tests

After single-arm checks, collect both arms holding simultaneously with all cameras and
representative inference load. This needs one coordinated owner/logger; two independent
`deploy.py` instances are not a synchronized bimanual test. Measure observation/send skew,
loop overruns, CAN/USB contention and tracking while the other arm moves through a checked
free-space path. Save matching joint trajectories and images for replay. Do not use arm-arm
contact as a collision calibration test.

Optional: dial-indicator lost-motion tests under reversed loading, repeatability after
warm-up, cable-routing changes and long-session drift. Encoder-only tests cannot identify
all gearbox output backlash. These were previously deferred; collect them only if practical.

Import: per-arm timing/noise, residual offsets, uncertainty ranges and collision-clearance
checks. Retain the measured base transform unless a new independent measurement supports a change.

## Data handoff and acceptance

Create one session folder, e.g. `data/real_yam/bimanual_session01/`, and copy the existing
runner outputs into it (the runner itself writes to `data/real_yam/`). Include:

- `session.json`: git commit, calibration file/hash, local timezone plus UTC start time,
  clock domains, arm/device IDs, units, gains, payloads, mounts, surface condition and operator notes.
- `measurements.csv`: `item,side,quantity,value,unit,uncertainty,method,trial_id`.
  Use SI units for import; explicitly label raw millimetre measurements. Unknown is null, not zero.
- Raw commands and feedback, timing CSVs and metadata; all failed/aborted trials with reasons.
- Camera frames/video and timestamps, photos with scale, force traces if measured.
- Train/validation trial IDs chosen before fitting; keep at least one pose/load condition withheld.
- A README listing exactly what was performed, omitted, or changed since calibration.

Frame convention: `q_model = q_encoder + calibrated_offset`; keep raw encoder values and
state whether exported positions already include offsets. Pose exports must name parent/child
frames, use metres, and label quaternion ordering (the current calibration uses w,x,y,z).
Record tag-to-object transforms rather than leaving them implicit in photographs.

`data/calib/` is gitignored. Transfer large raw captures using an accessible archive/location
and checksum; do not assume a pushed JSON contains the images. Commit small manifests,
measurement tables and derived reports with links to raw data. Review file sizes before adding videos.

On the sim side, replay identical targets with measured timing; compare held-out trajectories,
image landmarks, gap motion and contact traces. Report absolute errors, repeatability and
instrument uncertainty, not just a visual resemblance. Set task-specific tolerances before
accepting the model. If parameters cannot be identified from the data, retain that uncertainty
instead of reporting a precise fitted value. Only promote a model to training after these checks;
the existing older policy/training environment stays unchanged.
