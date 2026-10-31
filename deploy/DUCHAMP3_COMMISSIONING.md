# Duchamp3 commissioning, 2026-10-27

Model manifest verified. ACT timing on RTX 4090: p99 2.73 ms, max 2.81 ms, with synthetic images. Camera run after selecting ZED serial 20120598: top 29.8 fps, wrist 35.7 fps, host-arrival-to-use p99 32.86 ms and 30.71 ms; exposure time and clock alignment are not measured. Shared handoff and conventions checks passed. Camera intrinsics/spec file differs in overhead extrinsics from the active lab spec; review before geometric overlays.

Gripper calibration with operator supporting reader: normalized open 1.000, closed empty 0.038, closed on reader 0.3371, close command 0.0. A single calibration maps empty to 5.9 mm and reader to 19.0 mm against the checkpoint's assumed map. The current grasp threshold is 5.0 mm, so empty falsely passes. Force, stable retention and a robust threshold are not measured. No policy trial should proceed until this is resolved and validated.

The original real runner calls close after a pick or error, disabling torque while the arm may be raised and the reader held. The original frame capture similarly closes after reaching the raised ready pose. Both real entry points are disabled pending a reviewed powered-hold, retrieval and recovery workflow. The original pick timeout transitioned to insertion even if the gate failed; the real backend now stops instead. The live entry point remains disabled until the full recovery workflow and grasp gate are validated.

The original FlexChecker rejected the recorded folded-rest starting pose (link_2 at -36 mm against +40 mm clearance) because it bounded horizontal capsules by spheres. The revised checker computes capsule and cylinder vertical extents using orientation. The recorded rest and 101 sampled model-space points to ready pass, and the below-table regression still fails as intended. Physical object clearance, trajectory attainment and recovery remain to be reviewed before moving. Actual joint-3 errors previously exceeded 0.02 rad on two approach trials. The gripper driver reported a CAN socket shutdown race after a successful calibration; the cleanup path now stops and joins the motor thread first. Calibration on connect still moves the empty gripper automatically.

No real policy rollout or ready-pose image capture has been performed. Current camera and gripper commissioning changes are local research modifications, not a validated robot deployment package.

Ready1 live attempt: motor 6 communication-loss report followed by failure to attain ready within 0.02 rad. No ready-pose images were saved. Software stop was set; operator confirmed physical support before the capture process was terminated. Operator subsequently confirmed arms stable. No automatic recovery was attempted. A passive CAN trace is archived in data/task_h/frames/ready1_failed. Later traffic on all seven channels does not resolve the fault. The failed attempt had no move_to tracking log; future attempts now preserve target/readback traces and failure images.

The 16-scene simulation dry run completed with 15/16 seated after release and no modeled safety violations. This does not validate real control or free-PC dynamics.

Second ready capture stopped after a motor-4 communication warning. Operator secured the arm before process termination and confirmed stability. Retry recorded 499 movement steps, period p50 19.99 ms, p99 22.31 ms, max 23.27 ms; maximum joint-3 tracking error 0.0402 rad. Both attempts started motor control before potentially blocking ZED SDK initialization. Cameras now initialize before connecting the arm. SDK startup starving control is a hypothesis, not a confirmed root cause; stationary checks with cameras already active are required before another motion.

Camera-first stationary check and 301-step small-motion check completed without reported communication faults. Ready1 retry2 also had no reported communication fault, but ready attainment failed: encoder errors [-0.00401, -0.01019, -0.02304, 0.00231, 0.00171, -0.00439] rad. Joint 3 exceeded the unchanged 0.020-rad threshold. Failure images and target/readback traces are saved under data/task_h/frames/ready1_retry2. Operator confirmed secure support before controller termination. Stop remains set; no accepted ready-pose capture or policy rollout. Next: analyze controller residuals and compare failure images at their recorded approximate pose, retaining synchronization uncertainty.

Offline analysis: ready approach uses SafeArm.command without the configured task friction feedforward, whereas policy sends use it. This is a controller-semantics mismatch, not evidence for changing joint calibration. The approach trace also omitted the one-second settling interval. Added settling samples and an explicit, default-off approach_fric_comp candidate. No new gains, offsets or tracking tolerances were adopted. Candidate must be validated on small motions before a full ready approach. Friction feedforward values in the existing code are estimates/command terms, not measured output torque.

Folded-rest controller comparisons, cameras active: 0.005-rad and 0.015-rad joint-3 pulses completed with no reported communication faults and torque disabled afterward. At 0.015 rad, baseline joint-3 RMS target error was 9.698 mrad and observed range 0.763 mrad; configured feedforward RMS 9.458 mrad, range 0.381 mrad. Single sequential comparisons are not sufficient for controller identification; neither condition tracked the pulse meaningfully. Feedforward is not promoted. Preserve original files data/task_h/approach_controller_001.json and approach_controller_002.json as controller dead-zone evidence, with uncalibrated computed effort distinguished from torque measurements. No further amplitude increase or tracking-tolerance relaxation adopted.

## 2026-10-28: ready pose, grasp gate, camera loop cost

Ready pose attained. Root cause of the misses: static friction, not calibration or gravity model. Out/back
lag on joint 3 at the intermediate pose was +24 / -7 mrad (friction band ~1.2 N m beyond the 0.75 N m
feed-forward, plus ~0.7 N m under-compensation). i2rt's gravity model (linear_4310 + 0.5 kg, x1.1 on
joint 3) and the sim's FlexPoint model differ by 0.05 N m (intermediate) / 0.2 N m (ready) on joint 3,
so a model swap would not help. The sim already randomizes joint-3 friction over 0.6-3 N m.
Fix: `move_to(settle_trim=True)` (deploy/approach_trim.py): bounded integral on all six joints after
the slow move, 10 mrad/s rate, 50 mrad cap, 2 mrad deadband, stops when all joints are within
SETTLE_TOL = 10 mrad for 0.2 s (at most 15 s); every command still passes SafeArm. Used for the ready
pose in runner.py and act_frames.py, not for the return to rest (joints 2-3 sit on their stops).
`checks/ready_settle.py` run 001 (objects clear, jaws empty): stick-slip on joints 2 and 3 (7-10 mrad
overshoots, trim -23 mrad on joint 2), all joints within 3.3 mrad after 13.2 s (tol 5 mrad then);
held 2 s with the trim and 1 s with it dropped (as ACT's first tick): no drift, max 3.3 mrad. PASS,
returned to rest, torque off. `verified_start_envelope` set.

Earlier "measured start fails 0/4 in sim" result withdrawn: its harness edited sim state after reset.
A harness with a fixed nominal start also gave only 6/16, while the stock runner on the same seeds
(2000-2015) gave 14/16 seated: such harnesses are invalid, not the start pose. Moot now that the
real arm reaches the ready pose within 3.3 mrad.

Grasp gate (`checks/grasp_gate.py` run 001, 3 reps, operator holding the reader): empty 0.015,
reader 0.249-0.257, every verdict right. The scale differs from the calibration session (empty
0.038, reader 0.337): i2rt calibrates the gripper on each connect. Config now g_empty 0.015,
g_reader 0.254: gate band 0.134-0.373 classifies both sessions. `verified_reader_gate` set. The
finger map for ACT's state uses the same g_reader (19 mm). Squeeze force still not measured.

Timing: ACT p99 2.86 ms. Cameras 29.8 / 35.7 fps. Preprocessing both policy images took 10-14 ms
inside the 20 ms tick; it now runs once per frame in each capture thread (frames() 0.01 ms in the
loop). Logged age at use now includes preprocessing: top p50 20 / p99 35 ms, wrist p50 22 / p99 41 ms
(in the demos' 0-70 ms logged-age window). Sim dry run after the changes: 4/4 seated.

## 2026-10-28: first ACT pick (pick1), placement, table height

Placement: the first ready capture (ready1) had the reader at +40 / +87 mm and yaw +23 deg, the PC
port at +42 / +81 mm off nominal: outside the trained ranges. `checks/placement.py` (ZED frame, no
arm motion) finds both objects and prints the move; after one correction both were within 6 mm
(ready2). ACT real-vs-sim chunk difference dropped from 0.07-0.30 rad to 0.02-0.16 rad per joint.

pick1 (`runner real --until pick`): ACT descended toward the reader in 1.6 s, never slowed, and the
SafeArm refused a target with the fingertip 15 mm below model z = 0. The operator saw the open
fingertips ~3 cm ABOVE the table. ACT servoes on the wrist image, so it kept descending.
Cause: the real tabletop is ~30 mm below the arm base frame's z = 0. The sim (pc_bench, table_top=0.0)
puts it at z = 0. Evidence: `checks/table_probe.py` run 002 (fingertips hovering, gaps by ruler at
three spots): table at model z -28, -34, -30 mm. The calibration board lying on the table
(verify_lab_20261027*, three spots) fitted at z -22, -23, -20 mm for its top face. The YAMs stand on a
~23 mm mount plate, and the Menagerie base mesh reaches 10 mm below the base origin: ~33 mm.
Consequences in the current sim-trained stack: the reader and the port are 3 cm lower relative to the
arm than in training, the handoff gate's p_nom z (near_z 1 cm) cannot be met, and the placement
back-projections (objects assumed at z = 0) carry ~15 mm of parallax.
Fix (next physical session, docs/REAL_RIG_GAPS.md gap 9): raise the workspace by ~30 mm (flat board under the reader and the PC) and re-probe. No board was available on 2026-10-28.
Proper fix: sim table_top / object heights at -0.033 and the base plate in the scene; retrain ACT and
the insertion policy.

ACT cold start: the first two TorchScript calls took 354 + 298 ms (402 ms on pick1's first tick, arm
holding). ActStage now warms up through step() at load: first tick 2.6 ms.
The probe's first run stopped on an untrimmed intermediate move (joint 2 missed by 20.04 mrad):
every probe move except the return to rest now uses the settle trim.

## TO DO: transfer the 2026-10-28 rig data to the server (for the sim agent)

`data/task_h/` is gitignored (34 MB): it lives only on duchamp3. Copy it to the server by ssh (no
deletes on the destination), into the same path in the server checkout, then tell the sim agent:

    rsync -az --update -e ssh ~/agent_data_rl-physical-validate/data/task_h/ <user>@<server>:<repo>/data/task_h/
    # data/calib/verify_lab_20261027* (board-on-table heights) are in ~/agent_data_rl/data/calib/: copy them too

What the sim agent needs from it:
- `table_probe_002.json`: tabletop at model z -28/-34/-30 mm (gap 9, docs/REAL_RIG_GAPS.md). The sim's
  `table_top=0.0` (scenes/pc_bench.py, PickEnv reader/port heights, task_config fix_z) is 3 cm too high
  unless a ~30 mm board goes under the real workspace. Decide with the operator which side changes.
- `pick1/ticks.npy`: first real ACT pick (descent profile, targets, ACT latency, frame ages), stopped by
  the -15 mm fingertip limit ~3 cm above the reader.
- `frames/ready1`, `frames/ready2` (+ `sim_vs_real.png`): real ZED and wrist policy images at the ready
  pose; ready1 misplaced objects, ready2 within 6 mm of nominal. `placement/*.json|png`: measured poses.
- `ready_settle_001.json`, `intermediate_*.json`, `approach_controller_*.json`: joint stick-slip traces
  (joint 2-3 static friction ~1-2 N m, 7-10 mrad slips) for the sim friction model.
- `grasp_gate_001.json`, `gripper_cal_20261027.json`: gripper readings (scale changes per connect).
- `ctrl_seed2000/`, `dry_*`: sim dry runs of the deploy runner (14/16, 4/4, 4/4 seated).

## 2026-10-28: pick2 on the bare table (diagnostic): where the trained stack fails

Config `deploy/config/task_h_bare_table.json`: task_h.json with `safety.table_z = -0.028` (clearances from
the measured table) and `handoff.p_nom` z lowered by 28 mm. Diagnostic only. Loop clean: ACT max 5.8 ms,
tick p99 20.1 ms.
- **Grasp height (policy, not hardware).** ACT descends to tcp z -21 mm (sim: +4 mm), tracked to ~1 mm
  (lowest target -20.9, lowest measured -22.0), then lifts while closing. Real table ~-30 mm: the tips
  stop ~9 mm above it and the pads close on the top edge of the 8.8 mm reader (operator). Gripper reads
  0.04 = empty. ACT compensated ~25 of the 33 mm from the images; the rest is its learned grasp height.
- **Regrasp.** 5 attempts in 15 s (descend, close, lift ~30 mm, open, re-aim): the recovery behaviour
  runs but never changes the height. Timeout -> "pickup timeout without a valid grasp handoff", hold.
- **Grasp xy (unconfirmed).** tcp at x 195-210 mm; the reader, with the ZED back-projection redone at
  the real table height, is near x ~220 mm: possibly 1-3 cm short along the plug axis.
- **Not reached yet:** carry, handoff gate (p_nom), insertion, release.
For the new policy (sim agent): put the sim tabletop at z = -0.033 in the arm base frame with the
mount plates (or the operator adds a ~30 mm riser, gap 9), randomize table height (+-10 mm) and
port height with it, regenerate ACT demos, retrain ACT; retrain the insertion policy with the port
at table + 25 mm (its fix_z is a single value and its observation has the absolute tcp). Data:
`data/task_h/pick2/ticks.npy` (see the transfer note above).

## 2026-10-28: real runs with the virtual lift (pick3, try1-3): the failure list for the next policy

Runner additions: `virtual_lift` (deploy/conventions.VirtualLift: the policies run in a frame 30 mm
higher; joint positions AND velocities mapped, targets mapped back; sim with `sim_table_z` -0.030:
12/16 seated with the lift vs 0/8 without (pick2 reproduced in sim), baseline 14/16 on the same seeds),
a fingertip floor clamp (targets pressing the tips past SafeArm's limit are moved up instead of stopping
the run), 10 Hz policy-frame recording to `data/task_h/<run>/frames/`, `placement --table_z`.
All runs: ACT max 5.9 ms, tick p99 20.1 ms, no communication faults.

1. **Table height** (gap 9): fixed for the trained policies by `virtual_lift`; the grasp now reaches the
   reader (real tcp z -24..-26 mm, table ~-30).
2. **Placement measured at the wrong height**: the ZED placement assumed objects at z = 0; with the
   table at -30 mm the PC port was really 28 mm too far out (fixed with `--table_z -0.030`).
3. **ACT presses into the table at the grasp** (try1): targets 15 mm below the table -> SafeArm stop.
   Fixed by the floor clamp (try2: 37 clamps, max 3 mm).
4. **Grasp lands 1-2 cm behind the body centre along the plug axis** (operator: "a little too behind";
   wrist frames: pads on the rear edge). ACT's grasp x stays ~190-205 mm whatever the reader x
   (pick3 ~222, try2 217, try3 202): proprio-driven, as the height was. Edge grasps slip out on the
   lift: pick3 2 of 3 attempts, try2 5 of 5, try3 1 of 2. Suspects: real pad position vs the model
   tcp (FlexPoint geometry), the policy's weak visual x correction, the squeeze force (never measured).
5. **Plug hits the PC panel during the carry** (try3, frames 380-415): the rear grasp makes the plug
   stick out ~15 mm more than in training; at contact the reader pivots and levers out of the
   fingers. The carry itself arrived at the port row, close to aligned.
6. **Handoff gate never fired** in any run: ACT's "done" output peaks at 0.15-0.19 near the port;
   pick3 ended 23 mm off p_nom in y, 7 mm in z.
7. **No recovery after a drop**: ACT drifts toward +x/+y, opening and closing the gripper, until the
   15 s timeout.
8. **Not observed yet on the real arm**: the blind insertion, release.
For the new policy: sim table at -0.030 with mount plates (+-10 mm), measure the real pad/tcp offset
(e.g. close on a ruler under the wrist camera) and randomize grasp_along around it, measure the
squeeze force (gripper.close), PC x/y spread as measured (+-15 mm), demos that recover from a drop.
