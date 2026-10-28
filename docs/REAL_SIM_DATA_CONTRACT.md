# Real/simulation USB-C data contract

Version: `usbc-transfer-v1` (2026-10-27). This is the normative interface to implement
under [the plan](USB_C_SIM2REAL_PLAN.md), not a claim that current collectors or
checkpoints comply. The real agent owns real exporters/deployment validation; the
server agent owns simulation exporters/converters/package validation. Both review
schema migrations. Reject unsupported schema versions; preserve original artifacts.

## Coordinates, units and missingness

Use SI: metres, radians, seconds, kilograms; timestamps may use integer nanoseconds
with an explicit clock ID. World is `right_arm_base`. Each pose has `parent_frame`,
`child_frame`, `position_m[3]`, `quaternion_wxyz[4]` and uncertainty; quaternion maps
child coordinates into parent coordinates and is unit length. Record tag-to-object
and PC-to-port transforms. Define port axes/sign and depth zero at the surveyed
opening in the profile; never infer insertion sign from an unnamed quaternion.
Maintain a frame graph for table, PC, port_2, reader, plug_tip, wrist and cameras.

Keep `q_encoder_rad[6]` and `q_model_rad[6]` separate with joint names/order and
`q_model = q_encoder + joint_offset_rad`. Convert targets back once at the hardware
boundary: `q_encoder_target = q_model_target - joint_offset_rad`. Record conversion
version/hash and test round trips; never apply offsets again to a model-frame log.

JSON missing values are `null` with a `missing_reason`; CSV missing fields are empty
with a companion reason/validity field. Never use zero, NaN or invented measurements
for unavailable values. Distinguish `measured`, `estimated`, `computed_command`,
`driver_report`, `annotation` and `sim_ground_truth`. Force values remain null without
an instrument; gripper readback and driver effort do not establish output force/torque.

## Immutable session bundle

Bundle root: `data/real_yam/usbc/<session_id>/`. The same logical layout applies to
simulated episodes under `runs/usbc/<profile_version>/sessions/<session_id>/`.
These are proposed paths. UTF-8 JSON/JSONL carries nested records; native logs may
remain CSV/NPZ if an explicit columns/dtype/shape schema accompanies them.

| Path | Required contents |
|---|---|
| `manifest.json` | Schema/session IDs, real/sim origin, host/operator, UTC start, status, code/plan commit, dirty fingerprint, dependencies, profile/calibration/controller hashes, devices, clocks, units, support mode, artifact locations |
| `measurements.jsonl` | Raw repeated readings with item/quantity/value/unit/method/instrument/resolution/uncertainty/trial ID/source and missing reason |
| `poses.jsonl` | Survey/placement IDs, frame pair, pose, uncertainty, method and evidence refs |
| `support.jsonl` | Surface, feet/support observations, rocking/settling/tipping and PC motion |
| `episodes/<episode_id>/episode.json` | Trial IDs, stage/outcome summary, initial condition group, split, termination and files |
| `episodes/<episode_id>/commands.jsonl` | Tick/action chain and timing, as below |
| `episodes/<episode_id>/feedback.jsonl` | Native arrivals, raw/model joints, velocities and available telemetry |
| `episodes/<episode_id>/frames.jsonl` | Per-camera acquisition/arrival/preprocess/write metadata |
| `episodes/<episode_id>/consumption.jsonl` | Exact frame IDs selected per policy tick/query and inference timings |
| `episodes/<episode_id>/events.jsonl` | Contact/stage/intervention/stop/recovery and outcome evidence |
| `episodes/<episode_id>/diagnostics.jsonl` | Privileged simulation state or explicitly sourced real estimates |
| `raw/`, `policy_images/` | Sampled raw images/video and exact lossless 320×320 consumed RGB images |
| `splits.json` | Immutable group/trial/episode assignments and seed sets |
| `notes.md`, `commissioning.json` | Operator observations, limitations and collector checks |
| `checksums.sha256` | SHA-256 of every payload file, including manifest, excluding checksum file itself |

Nonapplicable tables may be absent only when `manifest.files` gives the reason (e.g.
manual-video-only). An action-labeled episode requires commands, feedback, consumption
and referenced policy images. A missing camera frame is logged, never silently dropped
from the attempt count. Capture streams may remain append-only during acquisition;
flush periodically and finalize as `complete`, `aborted` or `partial`, with a reason.
Freeze checksums after finalization; amendments create a new bundle/version referencing
the old one. Interrupted bundles retain recoverable rows and identify any truncated tail.

The baseline fingerprint must cover tracked staged/unstaged differences, submodule
commits/dirty status, and hashes of relevant untracked source/config files, using a
sorted inventory and SHA-256. A bare `git status` or HEAD hash is insufficient. Record
excluded artifacts and do not include secrets. Dependencies include lock hash and
actual installed versions, driver/SDK versions and device serial/mode identities.

## Clocks and camera timing

Each clock entry names source, unit, monotonic/wall nature and synchronization status.
For a known clock mapping record offset, drift, fit interval, method and uncertainty.
Do not subtract ZED/device time from host monotonic time without a validated mapping.
Unknown exposure time stays null; arrival-to-consumption age is still measurable but
must not be labeled exposure-to-consumption age. A visible common event estimates
relative timing and its uncertainty, not automatic exact synchronization.

`frames.jsonl` requires `camera_id`, local `frame_id`, optional hardware frame ID,
source/exposure timestamp and clock (nullable), host arrival, preprocessing start/end,
raw/policy file references and hashes, resolution/color order, crop/preprocess hash,
repeat flag, and write/queue status. Local IDs count captures, not hardware drops.
Record queue drops separately from detected image repeats and unknown hardware loss.
Log supported exposure, gain, focus and white-balance settings with units/auto state;
unsupported readback is null. Use `top`, `right_wrist`, `left_wrist` as logical IDs and
map them explicitly to actual devices (including ZED rectified left RGB).

`consumption.jsonl` requires `episode_id`, `tick_id`, `query_id`, observation cutoff,
selected frame IDs per camera, selection/inference start/end, arrival age, and
exposure age only if known. Log action-chunk source query and offset/ensemble
contributors when ACT reuses predictions. The observation for each control tick is
specified even if no new network query is issued. Preprocess completion alone does
not prove consumption. Saved lossless uint8 RGB must reconstruct the actual network
input after the package's explicit normalization/tensor conversion.

Simulation acquires cameras independently of the 50 Hz controller. At a tick choose
the newest frame whose simulated arrival and required preprocessing completion are
at or before the observation cutoff. Never use a future image. Replay measured delay,
jitter, repeat and drop behavior with seeded temporal sampling. Record the policy for
no frame/stale frame (hold/stop according to supervisor), maximum permitted age and
failure event. Buffer compression/writes off the capture path; bound queues and log
backpressure/overflow so a slow disk is visible in measured timing.

## Commands, feedback and episodes

One command row has `episode_id`, `tick_id`, stage, observation reference and timestamp,
`clean_expert_action` (nullable), `perturbation` (nullable), `requested_action`, decoded
model target, safety-limited target, encoder target, actual sent target, gripper
command, action mode/schema, control deadline and safety decision. Include target
construction, inference, safety, driver enqueue and CAN send times when exposed.
Each timestamp has a clock ID. A requested target is not proof of execution: mark
send status/sequence and missing send telemetry explicitly. Record gains and
compensation profile plus available gravity/friction/feedforward commands separately.

Feedback rows preserve native arrival cadence, device timestamp if present, encoder
and model joints, velocity, gripper raw/normalized readback and optional effort,
temperature or fault reports with provenance/units. Link the exact feedback sample
used at the tick; do not resample away native arrivals in the raw archive.

Events include time/clock, stage before/after, reason, evidence refs, confidence or
operator identity, retry number and intervention. Outcomes include mechanical seating,
seating depth/gap, retention interval after release, electrical enumeration
(`true`/`false`/`null`), termination cause and failure category. Suggested categories:
missed/partial grasp, grasp slip/drop, panel catch, partial entry, PC motion/tip,
alignment timeout, insertion timeout, release displacement, camera/driver failure,
safety stop, operator stop. Preserve natural failures and successful recoveries.

Manual recordings declare `supervision_type: manual_video` and synchronization status;
they cannot claim robot action labels. Synchronized robot demonstrations declare
`robot_action_labeled`. Operator seating labels remain distinguishable from validated
visual verification. Every attempt counts, including interventions and aborted trials.

Privileged simulation diagnostics include reader/PC poses and velocities, insertion
relative pose/depth, contact summaries, grasp drift, full parameter realization,
random seeds and profile/code hashes. Keep them in a separate namespace from actor
inputs. Ground truth may support experts, evaluation and critics; deployed actors and
transition guards must not receive it.

## Rig/task profile

Proposed profile ID: `usbc-rig-v<version>`. Store a JSON profile with immutable file
SHA-256, schema version, controller/camera subprofiles and validation status.
Each parameter contains `value`, `unit`, `source_ref`, provenance class, uncertainty,
randomization distribution/range and validation status. Unknown ranges remain pending,
not numeric defaults masquerading as measurements. Keep nominal measured appearance.

Cover geometry, masses/inertias and their estimation method, feet/table/contact
assumptions, port/reader frames, joint offsets/controller semantics, gripper mapping,
RGB preprocessing and camera timing distributions. All pickup/insertion/combined/real
consumers resolve the same profile. `support_mode` is `free`, `fixed`, or reserved
`left_arm_supported`; the first benchmark requires `free`. Reserved support is not an
alias for fixed. Retain legacy profiles for reproducing older results.

## State/action and policy package

ACT state schema `act-joints-gripper-v1`: six model-frame joint angles in named order
plus dimensionless normalized gripper readback. Define 0 = closed and 1 = open, with
raw endpoints, mapping, clipping and calibration evidence in the profile; validate
hardware polarity. This is a new contract, not a claim that existing `state()[6]` in
metres or current hardware readback already implements it.

ACT output schema `act-target-gripper-done-v1`: six absolute model-frame joint targets
in radians, signed gripper command (-1 close, +1 open; declare continuous vs binary
thresholding), and stage-completion score. Declare chunk length, execution horizon,
temporal ensembling and completion debounce; completion score is a proposal to the
supervisor, not evidence of seating or grasp. Image inputs are top/right-wrist RGB
320×320 with exact rectification, crop, resize, channel order, dtype/range and
normalization; left wrist is diagnostic unless a new manifest explicitly adds it.

Insertion manifests specify ordered observation fields/dimensions/units/frames,
history length/order/fill rule, previous-action semantics, normalization and action
mode. `relative` means measured model joints plus scaled action; `accumulated` means
previous target plus scaled action, capped relative to current measurement according
to manifest. Record action clipping, per-joint scale, target caps, gains, command
holding and control rate (baseline 50 Hz). Do not infer these from wrapper defaults.
At handoff seed the accumulated target from the last safety-approved target, reset
previous action, and initialize history consistently with training. Bound/check target
continuity; if the configured cap would force an unsafe jump, settle/re-align or stop.
Any alternative initialization must be versioned and identically trained/replayed.

Every deployable package contains:

| File | Required information |
|---|---|
| `manifest.json` | Package/schema IDs; architecture, weights hashes, code/plan/profile/controller/calibration hashes; state/action schemas, observation layout, history, normalization, control frequency, gripper mapping, preprocess and timing specification |
| `weights/` | ACT and insertion weights with architecture/export version |
| `normalization.json` | State/action/image normalization or exact embedded-normalization declaration and hash |
| `profile.json`, `supervisor.json` | Adopted profile and guards/retries/timeouts/recovery/stopping behavior |
| `starting_envelope.json` | Validated initial PC/reader/arm/handoff region, uncertainties and support condition |
| `evaluation.json` | Dataset/split provenance, seeds, gates, ablations, failures, confidence intervals and limitations |
| `checksums.sha256` | Full payload integrity |

Validate hashes, architecture, dimensions, action mode, offsets, camera inputs,
controller/profile/support and supervisor compatibility **before connecting an arm**.
Missing or incompatible manifests reject by default; older checkpoints need an
explicit offline migration/validation, never silent loading. Reject accidental double
normalization or image corruption; declare ownership of every rendering/augmentation
operation. Package creation is S7; pre-connect enforcement and commissioning are R6.

## Splits, transfer and validation

Assign immutable groups before fitting/export: real measurement `fit`, `development`,
`validation`; simulated policy `train`, `development`, `final_test`. Group by full
scene/geometry/randomization realization and related trials, never individual frames.
Reserve whole R2 poses; R1 has at least five held-out placements of at least 20 total.
Keep seed sets disjoint and store group-to-split membership. Do not reuse final-test
results for selection; a subsequent development cycle gets a new benchmark version.

Retain all raw episodes. ACT exports may select successful demos and useful recovery
segments, with source episode/tick ranges and selection reasons. Dataset output is a
new version; refuse existing destination/accidental overwrite. Test split leakage,
action reconstruction, observation-before-action, frame eligibility and checksum
integrity. Record rejected episodes and reasons rather than deleting them.

R0 and S0 owners must implement a dedicated resumable session-sync command before
large handoffs. Required behavior: copy finalized manifest/checksum-selected files
without a size filter, resume to a staging directory, verify all destination SHA-256
checksums and only then mark/import the bundle complete. Refuse conflicting session
IDs; interrupted transfer remains incomplete. Validate using an interrupted transfer,
a payload larger than the code-sync cutoff and deliberate corruption. Current
`scripts/server/sync_up.sh` is code sync and its size filtering is not a recording
transfer mechanism. No new sync command is claimed implemented in this plan commit.

Local implementation after the plan commit: `scripts/server/sync_usbc_session.py` creates a SHA-256 manifest for a finalized session directory, supports resumable checksum-based rsync to a staging directory, verifies destination contents, and creates `usbc_session_complete.json` only after verification. It refuses a different manifest or an unrelated nonempty destination and never deletes destination files. Local offline checks covered a 55 MiB interrupted partial file, source mutation, destination corruption and conflicts. Remote behavior still needs verification on mason when its actual address is available. The active duchamp3 session is still being collected: do not create its immutable manifest until it is finalized. Stage the separate `data/sim_ref/usbc_lab_20261027_port/` images, relevant calibration files and camera spec into the bundle before manifesting; they are not automatically included when only `data/real_yam/usbc_lab_20261027/` is synced.

Once collection is finalized, a new staging bundle can be created and verified without altering the source session:

```bash
.venv/bin/python scripts/server/sync_usbc_session.py stage data/real_yam/usbc_lab_20261027 /path/to/new_bundle \
  --reference data/sim_ref/usbc_lab_20261027_port \
  --reference data/calib/usbc_recal_20261027 \
  --reference data/calib/run_001 \
  --metadata blind_insertion/configs/real_cameras_duchamp3.json \
  --metadata docs/USB_C_IMAGE_HANDOFF_20261027.md
.venv/bin/python scripts/server/sync_usbc_session.py manifest /path/to/new_bundle
.venv/bin/python scripts/server/sync_usbc_session.py verify /path/to/new_bundle
.venv/bin/python scripts/server/sync_usbc_session.py sync /path/to/new_bundle /absolute/remote/staging/path --host user@server
```

Add any other calibration or episode references used by the finalized manifest before freezing it. `stage` refuses an existing bundle; after `manifest`, any change to the source bundle fails verification. The destination completion marker indicates that all manifest files and their checksums were verified, not that the measurement or policy gates passed.

Commit code, plans, compact manifests and summaries; transfer raw video/images and
weights through artifact storage. Each compact manifest records immutable artifact
location, size and hash. Collection acceptance requires offline replay and schema/fault
checks; policy acceptance requires recorded-input parity and mismatch rejection before
arm connection. These executable validators are R0/S4/S6/R6 deliverables, not supplied
by this documentation-only release.
