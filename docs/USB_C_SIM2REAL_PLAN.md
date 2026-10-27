# Measurement-driven USB-C sim-to-real transfer

Authoritative task plan, 2026-10-27. Status: planning handoff; no collection,
physics acceptance, training or real-transfer gate is claimed passed by this commit.

## Scope and authority

Insert the **SanDisk USB-C SD reader into the second USB-C port on the GMKtec PC**.
Measurements and deployment belong on **duchamp3**; simulation and training on
**mason**, after verifying each actual runtime. Start with RGB ACT pickup/alignment
→ proprioceptive PPO insertion → release. The stage supervisor, including retries,
recovery and stopping, is part of the complete policy.

The PC rests **freely on the desk**. Model its translation, rotation and tipping.
Use overhead ZED RGB and wrist webcam RGB; depth is outside scope. Observe normal
grasps and reader slip, but defer dedicated gripper force/pad/compliance tests.
Available instruments are calipers and a scale. Quantitative force and precise
compliance claims require additional instruments. Left-arm support is a later phase.

This plan governs this USB-C transfer effort over historical task assumptions in
[BIMANUAL_MEASUREMENT_PLAN.md](BIMANUAL_MEASUREMENT_PLAN.md),
[REAL_RIG_GAPS.md](REAL_RIG_GAPS.md), [RESEARCH_PLAN.md](RESEARCH_PLAN.md) and
[server instructions](../INSTRUCTIONS.md). Historical experiments remain reproducible;
shared-GPU and lab safety rules remain in force. In particular, historical fixed
fixtures, force-collection suggestions, bimanual campaigns and the assertion that no
dataset is needed do not define this effort. See the
[data contract](REAL_SIM_DATA_CONTRACT.md) and [server work queue](SIM_AGENT_HANDOFF.md).

## Source and slide traceability

Source: [keybox_sim2real.pptx](keybox_sim2real.pptx), 29 slides, including speaker
notes, supplied in the working tree and preserved byte-for-byte. SHA-256:

`e5f37bec4382662a639651eab175fe01c6f3f14d849fe85548bcb765e1d8acf4`

The deck supplies methods, not USB-C calibration. Its gains, friction, squeeze and
insertion forces, depth errors, turn angles, pins and release-tab mechanisms must
not become parameters for this cell without evidence. Notes qualify the reported
results: slide 27's agents had not completed their 500-seed final evaluation, and
slide 20's old-slip numbers are historical, not a current rerender.

| Slide | Method or evidence, including notes | USB-C disposition |
|---|---|---|
| 1 | Match motion, perception and physics | R0–R5, S1–S4; paired evidence |
| 2 | Separate development and final simulation results | S7; no borrowed success rates |
| 3 | Human task video establishes sequence and geometry | R1/R4; no key turn or tab |
| 4 | Real rollouts document complete execution | R4/R6; record all attempts |
| 5 | Precision, reach limits and absent force sensing | R1/R2/S5; quantify uncertainty |
| 6 | High simulated success can fail completely in reality | Gates below; preserve failures |
| 7 | Motion section: reach and landing | R2/S3/S5 |
| 8 | Whole-sequence reachability and joint margins | R1/S5; no imported placements/angles |
| 9 | Separate gravity, friction and gain effects | R2/S3; baseline before controller candidates |
| 10 | Reproduce adopted controller in simulation | S3/S4; no copied spring/integral values |
| 11 | Perception failures must be represented | R5/S4; RGB only |
| 12 | Compare camera pipelines on identical observations and latency | R5; depth algorithm selection deferred |
| 13 | Surface-dependent sensor failures | R5; glare/darkness/occlusion in RGB, no depth holes |
| 14 | Model measured sensor/calibration uncertainty | R5/S4; no imported depth or 2× error ranges |
| 15 | Episode appearance and temporal camera variation; nominal view | R5/S4; measured RGB ranges |
| 16 | Validate calibration alternatives on independent poses | R5; retain board calibration unless improved |
| 17 | Separate detector error from geometry/calibration error | R1/R5; port, panel, finger and reader overlays |
| 18 | Diagnose each physical failure using paired evidence | R3/R4/S3; no key-box mechanisms |
| 19 | Resting stability, component mass/COM and reset failures | R1/R3/S2; reader and PC settling/tipping |
| 20 | Grasp geometry and slip matter | R3/S3; dedicated squeeze/pad campaign deferred |
| 21 | Contact onset, partial insertion and recovery | R4/S3/S5; no key pins or force values |
| 22 | Support condition determines motion and failure | R1/S2; free PC, not the deck's hold-down |
| 23 | Recess/border geometry and true completion matter | R1/R4/S3; USB-C seating, no spring/tab |
| 24 | Runs, demonstrations and operator observations ground physics | R4/S5/S6; label evidence provenance |
| 25 | Separate reference, learned and real outcomes | S5/S7/R6 reports |
| 26 | Full real execution recordings (same clips as slide 4) | R4/R6; mechanical/electrical outcomes separate |
| 27 | Distinguish seeded/scratch, development/final results | S7; freeze development selection |
| 28 | Physics → privileged reference → observation policy → real | Dependency order and acceptance gates |
| 29 | Generalize recipe to later tasks | S8; support extension after baseline |

## Responsibilities and artifact locations

The real agent on duchamp3 owns R0–R6 software, collection integrity and reports;
the on-site operator owns physical measurements, trajectory review and motion
supervision. The simulator/server agent on mason owns S0–S8 and simulation evidence.
Both owners review interface changes and held-out comparisons. An unpassed gate
stays pending with evidence and a named blocker; absence of data is never a pass.

Paths below are planned outputs, not claims that those files exist. Real bundles use
`data/real_yam/usbc/<session_id>/` (abbreviated `SESSION/`); simulation artifacts use
`runs/usbc/<profile_version>/`; compact reports use `results/usbc/<version>/`.
Raw recordings/checkpoints stay outside Git. Version and checksum the artifacts
according to the data contract; commit compact manifests and reports when available.

## Real work queue: duchamp3

### R0 — Reproducible baseline and collector commissioning

**Owner:** real agent + operator. **Prerequisites:** documentation/interfaces;
read current safety, rig and camera code, and inspect existing logs without motion.

Record commit, dirty-worktree fingerprint, dependencies, calibration hashes, device
identities, controller settings, joint frames, camera modes and PC support condition.
The existing `scripts/real/measure_manual.py`, `measure_motion.py` and
`measure_cameras.py` are starting points with offline checks only; at plan publication
they are untracked local work, not shipped by this plan commit. Review arm ownership,
stop behavior, connection failures, partial-log preservation and return-to-rest.
A stop must not be automatically cleared to run a return trajectory.

Replace fixed camera startup sleeps with readiness/failure reporting. Add the
calibrated ZED RGB crop from `blind_insertion/scenes/physical_rig.py` and matching
wrist preprocessing from `blind_insertion/assets/real_camera.py`. Save exact 320×320
top/right-wrist ACT inputs plus sampled raw frames. Record images actually selected
at each policy tick; preprocessing completion is not consumption. Buffer image
writing with bounded queues and explicit overflow/write-failure records so compression
does not silently redefine camera timing. Provide offline replay and live no-motion
inference before hardware execution.

**Outputs:** `SESSION/manifest.json`, `commissioning.json`, short synchronized
recording, replay/parity and fault-test report. **Validation:** collection gate:
missing camera, stop, connection loss, writer failure and interrupted/partial trial
all leave diagnostic records. **Blocks:** unreviewed paths or unsafe ownership,
unavailable cameras, missing clocks/schema, failed replay/fault tests. Lab motion
requires an operator present, secured arm bases, available rig console and reviewed
start/excursion/return trajectories. No motion is launched by this documentation commit.

### R1 — Geometry, support and initial conditions (slides 8, 17, 19, 22)

**Owner:** operator measures; real agent records. **Prerequisite:** R0 manual-data
schema; R0 collection gate before robot-assisted surveys.

Use the right arm base as world. Measure PC dimensions, mass, feet/contact locations,
orientation/placement; target port center/axis, rear-panel recess and surrounding
collisions; reader mass and remaining dimensions (body/neck length split, neck
thickness, plug protrusion); table plane/surface. Take three repeated readings for
every accessible dimension and mass, recording instrument resolution. Sample at
least 20 normal reader/PC placements and reserve five for validation before fitting.
Observe resting rocking, tipping and settling and PC translation/rotation during
ordinary manual insertion/withdrawal. Use calibrated board/tag observations or
surveyed references with an independent validation view and explicit uncertainty.
Camera agreement alone is not connector-level accuracy.

Keep current plug dimensions until a measured discrepancy is reviewed. Correct
surrounding geometry and seating without enlarging the receptacle to achieve success.
**Outputs:** `measurements.jsonl`, `poses.jsonl`, `support.jsonl`, reference images,
uncertainty summary and `splits.json` in SESSION. **Validation:** measurement gate,
repeatability and independent pose checks. **Blocks:** unresolved frame/scale,
unknown support or missing held-out samples; estimated quantities stay labeled.

### R2 — Controller response at task poses (slides 9–10)

**Owner:** real agent + operator. **Prerequisites:** R0 collection gate, R1 geometry
and reviewed collision-free paths. Reserve complete poses for validation.

Cover folded/compact, pickup and pre-insertion poses using stationary holds, small
bidirectional steps, bounded ramps, triangles, dwell/reversal. Repeat selected
conditions five times with empty hand and normally held reader; repeat a subset
after normal warm-up. Observe wrist/tool landing and sag externally as well as with
encoders. Log requested and safety-limited targets, CAN sends, native feedback,
position/velocity, gains, compensation and available driver telemetry. Log gravity,
friction and feedforward commands when exposed; computed commands and driver effort
reports are not calibrated output torque.

Characterize the current controller first. Capped integral, revised gravity or
friction feedforward is a separate candidate only when baseline evidence warrants
it. Test on reserved poses and reproduce the adopted profile in simulation before
retraining. **Outputs:** command/feedback logs, `controller_profile.json`,
`controller_comparison.json` with uncertainty. **Validation:** held-out landing,
transient, stop and reversal comparisons within measured repeatability envelopes.
**Blocks:** failed safety checks, ambiguous command timing/frames or unvalidated
candidate; do not compensate for gravity errors by silently increasing friction.

### R3 — Normal grasp stability (slides 19–20)

**Owner:** real agent + operator. **Prerequisites:** R0 gate, R1 and reviewed pickup.
Record 20 normal pickup → lift → carry → alignment trials with existing settings.
Measure/annotate reader pose relative to wrist after pickup and before insertion,
translation/rotation within grasp, location, approach and repeatability. Keep drops,
rocking, partial grasps and contact-induced changes. Log gripper command and
normalized readback, never measured squeeze force. Observe normal open/close to
verify sim gripper-state polarity and mapping. Dedicated force, pad-deflection and
pull-to-slip tests remain deferred.

**Outputs:** `grasp_observations.jsonl`, grasp distribution, failure examples and
unresolved-parameter ledger. **Validation:** all 20 trials accounted for, mapping
checked and pose uncertainty reported. **Blocks:** occlusion or calibration that
prevents identifying drift; do not replace missing data with an attachment constraint.

### R4 — Task/contact outcomes (slides 21–24)

**Owner:** operator + real agent. **Prerequisites:** R0 recording readiness and R1
for manual observations; R0–R3 pass before supervised robot stage trials.

Collect a small manual demonstration set with synchronized views or explicitly
unsynchronized video. Annotate first contact, panel catches, partial entry, actual
seating depth/final gap, PC motion before seating, withdrawal, natural failures and
recovery, including spoken/written observations. Then record supervised pickup,
approach, alignment, insertion and release trials; retain unsuccessful attempts and
operator stops. Record depth/displacement/timing and qualitative resistance. Force
fields stay empty without a gauge: these records cannot identify quantitative
insertion force, contact stiffness or gripping force. Manual videos are strategy and
geometry evidence; only synchronized robot observations/actions are action-labeled demos.

**Outputs:** `episodes/`, `events.jsonl`, seating references and failure taxonomy.
**Validation:** outcome labels tied to evidence, all attempts included, time alignment
and supervision type explicit. **Blocks:** ambiguous seating/clock evidence for the
claimed use; manual recordings must not be converted to fictitious action labels.

### R5 — RGB observations and perception uncertainty (slides 12–17)

**Owner:** real agent + operator. **Prerequisites:** R0 readiness; R1 surveyed targets.
For top, right wrist and left wrist capture stationary/moving scenes in representative
lighting; record supported exposure/gain/focus/white-balance settings. Measure arrival
intervals, repeated frames, processing time and consumption frame age. Use a common
visible event for relative timing; preserve unknown exposures and unaligned clocks.

Compare reader/port/finger/PC real-sim overlays at held-out poses; separate localization
from calibration error. Include glare, dark surfaces, occlusion, blur and compression.
Retain calibration unless independent evidence supports refitting; compare any
robot-outline fit with the board calibration on independent poses.
**Outputs:** `rgb_reference/`, timing distributions, localization residuals and
`appearance_ranges.json`. **Validation:** held-out overlays and timing replay, no
future frame selection. **Blocks:** unavailable raw/crop correspondence or unidentified
clock mapping; depth methods and key-box numeric noise ranges are not substitutes.

### R6 — Real policy supervisor and commissioning

**Owner:** real agent; operator supervises motion; server agent supplies package.
**Prerequisites:** R0–R5, S4 parity and compatible S7 package before real policy trials.
Software scaffolding/replay may proceed earlier.

Implement one controlled interface for ACT and the insertion actor:

`observe → pick → verify grasp → align → settle/handoff → insert → verify seating → release → verify retained seating`

Transitions use deployment-available observations. If the free PC leaves the validated
insertion starting region, re-observe and realign; do not push toward its old pose.
Maintain target continuity and initialize history/previous action/accumulated targets
at handoff. Bound retries and stage timeouts; log every transition reason, stop,
recovery and operator intervention. Keep mechanical seating separate from electrical
enumeration when available. Initial seating confirmation may be operator-labeled
until the visual verifier is validated, and must be reported as such.

**Outputs:** versioned supervisor config, transition logs, parity report and initial
20-trial real report. **Validation:** recorded-input inference → live no-motion
inference → supervised stages → complete trials; count every attempt/intervention.
Mechanical success requires remaining seated for two seconds after release.
**Blocks:** incompatible package, failed parity, unvalidated starting envelope or
missing transition evidence; a timeout cannot masquerade as successful alignment.

## Simulation work queue and dependencies

The detailed owner/procedure/output/validation/blocker specifications for S0–S8 are
in [SIM_AGENT_HANDOFF.md](SIM_AGENT_HANDOFF.md). Their required sequence is:

| Item | Required result | Depends on |
|---|---|---|
| S0 | Actual checkout/runtime/jobs/GPU baseline, preserved historical runs | Published plan commit |
| S1 | One versioned rig/task/controller/camera profile | S0, contract; measured values from R1–R5 |
| S2 | Dynamic free PC, table support and moving port throughout pipeline | S1, R1 |
| S3 | Held-out arm/contact replay and adopted controller | S1/S2, R2–R4 |
| S4 | State/action/frame/camera timing parity | S1, R0/R3/R5; ready before new data |
| S5 | Validated privileged full-sequence reference | Physics gate, S2–S4 |
| S6 | Reproducible collection, replay and disjoint splits | S4/S5, contract |
| S7 | ACT/PPO/supervisor pilot, ablations, frozen final evaluation/package | S6 dataset gate; actual ACT handoffs |
| S8 | Explicit later left-arm-support extension | Single-arm baseline understood |

Execution order: documentation/interfaces and baselines → local R0/R1 and existing-data
replay → R2–R5 alongside S1–S4 preparation → import measurements/physics acceptance →
S5/S6 reference and datasets → S7/R6 training and deployment integration → supervised
real evaluation → versioned discrepancy-driven update → later S8. Preparation with
estimates is allowed; acceptance and scaling cannot skip missing measurements.

## Acceptance gates

These are proposed project requirements, not achieved results. Record the evaluating
owner, date, profile/code hashes, evidence links, thresholds and unresolved limitations
in `results/usbc/<version>/gates.json`. Set task-specific residual tolerances from
measurement uncertainty/repeatability before fitting or opening held-out comparisons.

| Gate | Required evidence and reviewer |
|---|---|
| Collection readiness | Real owner: replay/schema checks and controlled missing-camera, stop, connection/write-failure and partial-trial tests preserve diagnostics |
| Measurement readiness | Both owners: geometry/support manifest complete, calibration uncertainty explicit, fit and held-out trials separated |
| Physics readiness | Both owners: stable resets, no artificial attachment, correct PC/table collisions, held-out arm/object trajectories compared to repeatability envelopes; unmeasured forces explicit |
| Reference readiness | Server owner: 20 deterministic regression scenes pass stability/reachability; target ≥95% pickup/handoff and ≥90% full-task success on 200 development scenes |
| Dataset readiness | Both owners: replay validates observation-before-action, executed vs expert actions, consumed-frame selection; no future images or split leakage |
| Policy readiness | Server owner: three training seeds, frozen 500 reserved simulation scenes, confidence intervals, stage failures and ablations; seating retained two seconds after release |
| Real transfer readiness | Real owner + operator: recorded-input parity and live no-motion pass, then supervised stages and initial 20-trial report counting every attempt/intervention |

A free-PC failure caused by PC motion remains evidence. Never change support inside
the benchmark to improve success. Force gauge and displacement instrument availability
are future gates for quantitative force/compliance fidelity, not prerequisites for
honestly reporting the current limited baseline.

## Publication and handoff

This commit publishes only the three planning/interface documents, source PPTX and
its checksum provenance, and a task pointer in `INSTRUCTIONS.md`. It does not launch
robot motion or training, ship untracked collectors, or establish a live mason baseline.
Verify slide coverage, links, owners, prerequisites and gates, then stage only these
files. Commit as `docs: plan measurement-driven USB-C sim-to-real transfer` and push
without force to `codex/bimanual-calibrated`. Report the resulting commit hash; the
server records that hash as `plan_commit` rather than relying on a moving branch tip.
