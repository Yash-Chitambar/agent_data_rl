# USB-C simulator and training handoff: mason

2026-10-27. Read [USB_C_SIM2REAL_PLAN.md](USB_C_SIM2REAL_PLAN.md) and
[REAL_SIM_DATA_CONTRACT.md](REAL_SIM_DATA_CONTRACT.md) first. This is a work queue,
not a report of completed simulator changes or a command to launch training now.
The publishing agent reports the plan commit hash after pushing; record it as
`plan_commit` in every baseline, profile, dataset and policy package.

## Task and current evidence

Single right-arm SanDisk USB-C SD reader insertion into **port 2 of the GMKtec PC**.
The PC is free on the desk. Start with RGB ACT pickup/alignment, proprioceptive PPO
insertion and release; include supervisor/recovery behavior in policy evaluation.
No depth, no dedicated gripper force/pad campaign, no left-arm support in this first
benchmark. Never fix the PC or enlarge the receptacle to hide a free-PC failure.

At publication the local checkout is on `codex/bimanual-calibrated`, with origin
`https://github.com/Yash-Chitambar/agent_data_rl.git`. Verify rather than assuming this
is mason's checkout. Historical `peg_test`, `jensen`, `athena5`, `main`, `mason-runs`
and GPU status entries in server docs describe earlier work. Preserve historical
runs/checkpoints under their original versions; do not relaunch the legacy job lists
as the USB-C work queue or rebase unrelated work merely to read this plan.

Known implementation starting points (paths relative to repository root):

| Area | Starting point and required follow-up |
|---|---|
| Scene/geometry | `blind_insertion/assets/builder.py`, `blind_insertion/scenes/pc_bench.py`, `blind_insertion/scenes/physical_rig.py`: prescribed fixture and collision/support assumptions need review |
| Environments | `blind_insertion/envs/pick_env.py`, `blind_insertion/envs/peg_env.py`: share measured physics and moving-port semantics |
| Combined policy | `blind_insertion/pipeline/pick_insert.py`: Inserter hardcodes relative action/scale/history defaults; settle uses nominal port; timeout can trigger handoff; seating currently needs stronger retention validation |
| Real interface | `scripts/real/deploy.py`, `rig.py`, `safety.py`, `armshare.py`, `camshare.py`: preserve offsets, arm ownership and stop semantics |
| Cameras | `blind_insertion/assets/real_camera.py`, `blind_insertion/scenes/physical_rig.py`, `blind_insertion/configs/real_cameras_duchamp3.json`: calibrated crop functions/spec |
| Existing configs | `blind_insertion/configs/task_h_reader_real.json`, `task_h_reader_real_acc.json`: legacy baselines, not automatic validated profiles |
| Collection/training | `scripts/act/` and `blind_insertion/envs/pick_env.py`: inspect actual collector/converter/exporter before adapting |
| Checks | `scripts/tools/check_physical_pick.py`, `check_physical_calibration.py`, `check_pc_bench_warp.py`: extend as appropriate |

The previous physical-scene diagnostic achieved **0/4 handoffs**; this is a warning
against scaling data collection before the expert works, not a new evaluation from
this commit. Local uncommitted changes exist, including collectors
`measure_manual.py`, `measure_motion.py`, `measure_cameras.py` and a physics audit;
they are not included in this plan commit. Coordinate their separate reviewed code
handoff. The plan does not certify current wrappers or measurement scripts as ready.

### Lab evidence collected after the plan

On duchamp3, `data/real_yam/usbc_lab_20261027/README.md` indexes the current measurements and stopped trials; `docs/USB_C_IMAGE_HANDOFF_20261027.md` identifies the image bundle and pose-estimation limitations. The reader's reported mass is 17 g, and its body/neck dimensions are recorded from calipers. Do not use the legacy 27 g as a measured mass. The second port center, reader/PC support dynamics, contact forces and compliance remain unresolved.

For controller replay, `motion/right_pickup_mid_empty_hold_001/approach.csv` and `encoder_right.csv` contain a stopped empty-hand approach: joint 3 lagged and settled about 0.0275 rad short of the target. The paired requested/executed targets match and all 285 commands have a logged send time. The one-second final encoder plateau and prior high-pose stop are useful fitting evidence, not proof of a specific friction law or a held-out validation set. The 0.020 rad pose-attainment stop remains active. `motion/reader_grasp_006_arm_motion.csv` is a separate folded-pose loaded joint-3 triangle; the operator reported that the reader remained held, but grip force was not measured.

A diagnostic CPU baseline replay is saved beside the midpoint trial as `cpu_baseline_replay.{csv,json,png}`; `scripts/tools/replay_motion_cpu.py` reproduces it. It replays sent targets at host CAN-send times with logged kp/kd and applies encoder/model offsets once. The current MuJoCo baseline differs from real joint 3 by approximately 0.0266 rad RMS over the recorded feedback. This comparison disables contacts and does not reproduce the driver's complete gravity/friction feedforward; it must not be promoted as calibrated physics or treated as a held-out result.

The adjacent `cpu_dev_friction_sweep.json` contains a development-only constant-friction sensitivity sweep on that same trajectory. Among the tried values, joint-3 `frictionloss=2.0 N m` lowered RMS to 0.00569 rad, while final simulated position still differed by 0.01746 rad. This parameter is not a measured torque, and no candidate is adopted. Evaluate on separate trajectories and poses, compare the stateful-friction candidate, reproduce driver feedforward where available, and check timestep sensitivity before considering promotion.

An exploratory replay of the separate folded-pose triangle is saved in `motion/right_rest_empty_j3_triangle_001/cpu_dev_{baseline,friction_2}.{csv,json,png}`. The 2.0 N m candidate reduced RMS (4.37 to 2.43 mrad), but simulated joint-3 motion spanned 7.54 mrad while the actual encoder spanned only 0.381 mrad. The constant-friction candidate therefore still fails to reproduce the observed sticking. This comparison was not a predeclared held-out gate.

A first CPU-only stick/slip torque-cancellation candidate (`--stiction 3 1`) also failed: 14.45 mrad RMS on midpoint approach and 9.77 mrad on the folded triangle, with 17.91 mrad simulated versus 0.38 mrad observed motion on the triangle. See `cpu_dev_stiction_3_1.*` in both runs. Its simple state transition omits coupled-constraint and driver feedforward behavior. Do not promote it; investigate a numerically stable constrained stick/slip implementation and compare on disjoint poses.

An inverse-mass-coupled variant is available only in the CPU replay tool via `--stiction 3 1 --stiction-mode coupled`. On the same development data, 1 ms / 0.5 ms RMS was 14.659 / 14.654 mrad for the midpoint approach and 0.262 / 0.301 mrad for the folded triangle. Although the sampled position traces differed by at most 0.066 and 0.317 mrad respectively, the folded-triangle stick/slip state counts changed substantially between timesteps. This is not a full constrained contact solver or a reproduction of driver feedforward; it is not promoted and is not a held-out result.

The no-motion RGB reference at `cameras/stationary_rgb_001/` contains host frame arrival times and sampled 320×320 policy crops. The measured collector receive rates were about 29.7 Hz overhead and 20 Hz for each wrist; `timing_summary.json` documents processing times and limitations. Exposure timestamps for the wrists, aligned camera clocks and actual policy-consumed frame IDs remain unavailable. Replay and no-motion policy inference are still required before the collection gate can pass. Raw captures and checkpoints must be transferred as verified artifacts; these local files and uncommitted code are not automatically present on mason.

The provisional measured profile is `data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json`, generated from the measurement CSV with `scripts/tools/build_usbc_profile.py`. It is evidence for a profile, not an adopted training or deployment configuration. The dedicated artifact tool `scripts/server/sync_usbc_session.py` has local 55 MiB partial/resume, corruption and conflict checks; remote verification remains pending because the name `mason` did not resolve from duchamp3. The session is still open, and the separately stored scene images and calibration must be staged with it before freezing the bundle.

An isolated CPU prototype can be built with `pc_bench.build_scene(support_mode='free', profile=profile, detailed=False)`. It has a free PC joint, the reported 0.284 kg mass, four approximate 5 mm feet and PC/table collision bits; the measured reader body, neck and 0.017 kg mass are used while the legacy plug geometry remains unchanged. Uniform-box inertia, foot insets and friction are assumptions, and the neck is still rigid in this prototype. A basic CPU check confirmed it settles on feet and responds to synthetic loads; it has not passed physical calibration or solver-sensitivity gates. Existing Warp pickup/insertion environments and their reset/reward/port calculations still use the fixed fixture and must not be treated as free-PC training.

`data/real_yam/usbc_lab_20261027/baseline_snapshot_20261027.json` is a point-in-time code, dirty-worktree, dependency, calibration and device fingerprint. It records controller settings from a prior run rather than a live connection; later edits and captures require a fresh snapshot before finalization. `cameras/replay_rgb_allcrops_001/` stores a short no-motion reference with every 320×320 crop and sampled raw frames. `scripts/tools/replay_camera_selection.py` generated 388 virtual 50 Hz ticks selecting only already-ready saved crops; sampled raw/crop pairs reconstructed losslessly. These are simulated selections, not ACT-consumed frames, and the 100 ms age threshold is diagnostic rather than a validated supervisor limit.

## S0 — Establish server baseline

**Owner:** server agent on mason. **Prerequisite:** published plan commit.
Before mutation, record hostname, repository root/remote, branch/HEAD, worktree and
submodules, installed dependencies/lock and CUDA stack, active jobs, tmux/runner state,
GPU capacity and disk availability. Obtain the plan in an isolated checkout if needed;
preserve running jobs and their source/checkpoint identities. Read
[INSTRUCTIONS.md](../INSTRUCTIONS.md), [CLAUDE.md](../CLAUDE.md) and
[SERVER.md](SERVER.md) for operational details.

Honor shared-GPU rules: check memory/utilization/processes over multiple samples and
again immediately before every launch, use only free GPUs, explicitly pin device IDs,
never signal other users' processes, no sudo/system packages, keep own directories,
run long work in tmux and release GPUs on completion. Ask Yash before holding **more
than two GPUs** or **any GPU for more than one day**, unless existing authorization
clearly covers the proposed work. Historical approval rows are not automatic approval
for this new campaign. Benchmark actual physics/rendering/workload before choosing
world counts or job sizes; use `--require-idle-gpus` with the runner where applicable.
A live capacity check is not a reservation. Preserve the installed CUDA 12 stack if
present; heed the existing warning against `uv sync`/`uv run` restoring CUDA 13 wheels.

**Outputs:** `results/usbc/<version>/server_baseline.json`, source/runtime/job inventory,
benchmark report and scoped compute budget. Co-own the contract's resumable/checksummed
session-sync implementation and transfer checks with R0 before large recordings arrive.
**Validation:** checkout/plan identity, environment checks and bounded workload
benchmark on authorized free capacity. **Blocks:** inaccessible repo/artifacts,
unidentified running experiments, failed environment checks, busy GPUs or required
resource authorization not obtained. This document publication itself does not run S0
on mason or authorize a training batch.

## S1 — Shared versioned rig/task profile

**Owner:** server agent; real owner reviews measurement references.
**Prerequisites:** S0 and contract; R1–R5 supply measured values, estimates may scaffold.
Introduce one profile for geometry, masses/inertias, controller semantics, contact
assumptions, camera preprocessing/timing and support mode. Every parameter carries
value/unit/source/uncertainty/randomization/validation status. Use it across pickup,
insertion, combined evaluation and deployment; retain legacy configs.

**Outputs:** versioned profile JSON, loader/compatibility checks, parameter ledger and
migration notes under `results/usbc/<version>/`. **Validation:** all four consumers
resolve identical hashes/semantics; unknowns remain explicit. **Blocks:** conflicting
frames or sources, missing required profile fields; do not call estimates calibrated.

## S2 — Dynamic freely resting PC (slides 19, 22)

**Owner:** server agent. **Prerequisites:** S1, R1 geometry/mass/support observations.
Replace the task's prescribed fixture with a free joint; add measured PC mass,
explicit feet/support geometry, estimated-and-labeled inertia and body-attached port
frame. Compute component inertias/COM consistently. Add PC–table collisions: current
fixture/table collision categories cannot simply be reused. Audit collision masks and
contact pairs. Keep fixed fixture only as an explicitly labeled comparison mode.

Update resets, rendering, expert targets, rewards, success checks and `plug_in_port`
to use current PC pose. No stale nominal transforms or teleporting during alignment.
Test settling, translation, yaw/rotation, tipping and contact with table/reader/hand;
verify moving-port transforms in CPU and training implementations.
**Outputs:** dynamic-PC scene/environment changes, deterministic regression scenes,
support-motion comparisons and inertia assumptions. **Validation:** stable starts,
correct collision pairs, physical motion against R1/R4 held-out observations and no
artificial object attachment. **Blocks:** unknown support geometry/mass or unstable
contacts. If free insertion moves the PC before seating, report that result; do not
silently clamp it. Later support can be motivated by this evidence.

## S3 — Controller and contact physics (slides 9–10, 18–23)

**Owner:** server agent; real owner supplies/reviews traces.
**Prerequisites:** S1/S2, R2 command/feedback and R3/R4 object behavior.
Replay real command sequences through CPU MuJoCo and the training implementation.
Fit or bound delay/holding, effective gains/damping, gravity/payload residuals,
breakaway/sliding/stopping/reversal, encoder noise/quantization/offset residuals.
Compare current constant friction with a stateful candidate on held-out poses; adopt
only with better responses and numerical stability. Match whichever real controller
was adopted; any integral/gravity/feedforward revision needs R2 candidate validation
before retraining. Computed effort is not calibrated output torque.

For reader/port/table, match geometry, rest, seating depth, slip and PC displacement.
Inspect contact counts, penetration, solver convergence and timestep sensitivity;
include sliding/rotational resistance only as supported by data. Keep unidentifiable
force/stiffness ranges explicit. Do not import key-box pins, turns, tab springs or
force values. Use the same controller/task physics in pickup and insertion.
**Outputs:** replay tool/results, fit/candidate profiles, held-out residual and
sensitivity report, unresolved-parameter ledger. **Validation:** physics gate against
predeclared measurement-repeatability envelopes, including external tool motion.
**Blocks:** missing native timing, nonidentifiable parameters presented as precise,
held-out regression or instability; force/compliance accuracy remains instrument-gated.

## S4 — Observation and action parity

**Owner:** server agent + real deployment owner.
**Prerequisites:** S1, R0 replay/consumption logs, R3 gripper observations, R5 RGB timing.
Fix the insertion wrapper to honor checkpoint action mode, history, scale, target cap
and ordered observation layout. Apply joint offsets once; use explicit encoder/model
conversion. Version ACT state as six model-frame joints plus normalized gripper
readback, replacing unverified equivalence with finger opening in metres. Define
absolute targets, gripper commands and completion output in the package. Reset
accumulated targets/previous action/history consistently at handoff with target continuity.

Use identical preprocessing for recorded real and simulated RGB inputs. Acquire
cameras independently of 50 Hz control, consuming newest eligible arrived frames with
measured delay/jitter/repeats/drops, never future images. Randomize appearance per
episode and temporal effects over time; preserve nominal paired views and avoid
equivalent corruption in both rendering and augmentation.
**Outputs:** shared adapters/preprocessing, package validator, recorded-input parity
fixtures and timing replay tests. **Validation:** relative/accumulated target and
history equivalence, offset round trip, mismatch rejection before arm connection,
correct frame eligibility/staleness behavior and RGB tensor reconstruction.
**Blocks:** unverified gripper convention, silent legacy defaults, missing camera
selection evidence or incompatible profiles. Complete before new dataset generation.

## S5 — Privileged reference (slides 8, 24, 28)

**Owner:** server agent. **Prerequisites:** physics gate and S2–S4.
Adapt expert to calibrated gripper, measured dynamics and moving PC. Check entire
sequence reachability and joint margins; pick/carry without unrealistic attachment,
align to actual port, perform bounded insertion/evidence-supported recovery, release
and verify seating. Ground-truth geometry/contact may label/reference/evaluate or
supervise critics, but actors and deployed transition guards only use deployment
observations. Preserve separate privileged diagnostic inputs.

**Outputs:** expert, 20 deterministic regression scenes, 200-development-scene report
with stage failures/reachability/joint margins. **Validation:** reference gate:
regressions pass stability/reachability, target ≥95% pickup/handoff and ≥90% complete
task. These are proposed gates, not results. **Blocks:** reference failure including
free-PC motion; the previous 0/4 handoff diagnostic makes validation necessary before
scaling collection. Do not train around invalid reference labels.

## S6 — Episode collection and conversion

**Owner:** server agent. **Prerequisites:** S4/S5 and contract validators.
Save clean expert action, perturbation, requested action and actual executed target;
observation before action, command timing and consumed frame IDs; stage/transition
reason/intervention/termination; privileged reader/PC pose/velocity, relative insertion,
contacts/grasp drift; full parameter realization/seeds/profile/code hashes. Preserve
successes, natural failures and successful recoveries in the raw archive.

Export successful demos/useful recoveries for ACT with source segment IDs and reasons.
Split by scene/geometry/randomization groups, not frames; training/development/final
seeds disjoint. Conversion creates a new dataset version and refuses overwrite.
**Outputs:** collectors/converter, immutable raw sessions, split registry, dataset
manifest and replay validation report. **Validation:** dataset gate (timing, expert vs
executed action, no future frames, no leakage, checksum integrity, refusal of overwrite).
**Blocks:** invalid reference or ambiguous labels/timing; do not discard failed trials
or reuse final-test scenes to improve selection.

## S7 — Complete policy training and evaluation

**Owner:** server agent; real owner integrates R6.
**Prerequisites:** dataset gate, reference gate, measured ACT handoff distribution;
authorized compute budget after benchmark. Start with a small data/training pilot.
Expand only after pilot data and behavior checks pass. Train ACT pickup/alignment
on new RGB/state schema, insertion PPO on actual ACT handoffs (including realistic
PC motion), and supervisor transition/recovery rules on deployment-available signals.
Keep ACT/PPO architecture for the first release; old checkpoints are explicit baselines
and incompatible manifests are rejected.

Compare previous pipeline, updated physics, physics plus timing/appearance, and full
updated data/supervisor. Identify each baseline's original support/physics rather than
mixing it into the new free-PC benchmark. Freeze selection on development data; run
three training seeds for final comparison and evaluate each on the same **500 reserved
simulation scenes**, reporting per-seed and aggregate results without treating correlated
repeated scenes as independent evidence. Report 95% confidence intervals (e.g. Wilson
per-seed binomial intervals), stage failures, interventions, retries and ablations.
Mechanical success includes two seconds retained seating after release; electrical
enumeration is a separate real outcome. Count every attempt, and label operator-assisted
outcomes separately from autonomous ones.

**Outputs:** versioned pilot/training records, dataset/checkpoint identities, evaluation
report, policy package with weights/architecture/state-action/normalization/frequency/
camera/controller/profile/supervisor/starting envelope and hashes.
**Validation:** policy gate, offline package incompatibility checks and R6 parity;
then recorded-input → live no-motion → supervised stages → initial 20 real trials.
**Blocks:** unmet pilot/reference/data gate, resource limits, incompatibility, final-test
selection leakage or unsupported starting envelope. Real deployment is supervised on
duchamp3, not initiated remotely by this handoff.

## S8 — Later left-arm support

**Owner:** server agent + real owner/operator. **Prerequisites:** single-arm baseline
validated or its limitations understood and reported; later support measurements.
Reserve `free`, `fixed`, `left_arm_supported` modes now, with the future mode explicitly
unimplemented until it exists. Later add an actuated colliding left arm, measured PC
support contacts and establish/release-support supervisor stages. Reuse episode and
validation machinery with a new profile/benchmark; do not silently fix the PC.
**Outputs:** support-phase design and later versioned implementation/evaluation.
**Validation:** actual support contact/motion and complete task under separate mode.
**Blocks:** no support measurements or premature substitution for the free baseline.

## Return handoff and completion evidence

For every item report owner, code/profile hashes, artifact paths/checksums, validation
result and unresolved blockers. Keep compact reports in `results/usbc/<version>/`;
weights/recordings stay in artifact storage with resumable verified transfer. Publish
future code changes separately from this plan and preserve legacy experiments.

Return to the real owner: adopted rig/controller profile; held-out real/sim comparison;
validated package and starting envelope; failure taxonomy and retry/stop configuration;
recorded-input parity fixtures; and the frozen evaluation report. R6 must reject an
incompatible package before arm connection. Record real discrepancies to version the
next measurement/model/training cycle. No gate is satisfied merely by a successful
simulated rollout or by copying a numeric value from the key-box deck.
