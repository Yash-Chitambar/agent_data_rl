# Parallel schedule for the capture and home moves (2026-11-15, server, sim and real logs only)

Today the two arms never move at the same time. This work lets the two slow two-arm moves (capture and home) run at
the same time, behind a new config block that is off by default. Nothing ran on the arm.

**Result in short.**

- P1, capture (both arms to the ZED pose at the same time): passes. The arm-arm distance is 116.4 mm at the
  least (n = 32 real runs, every offset from -2 to +4 s). The minimum is at the shared end pose, the same as in
  the serial move. Saving at today's speed: 8.3-9.0 s (n = 7 runs at 0.4 rad/s).
- P2, home: passes only with a left lead. If both start together, the model arms overlap (-15.8 mm, n = 31). With
  the right arm 3 s after the left, the replay never gets closer than the serial home on 31 of 31 runs (planned
  paths) and 7 of 7 runs (measured paths at 0.4 rad/s). Saving: 5.5-6.4 s (n = 7). The serial home itself starts
  with the arms 45-63 mm apart (n = 7). So with the 50 mm rule, 4 of 7 homes run in parallel and 3 run serially.
- P3, the left swing during the pick: fails. The swing brings the left fingers 5-27 mm from the right fingers
  (n = 29). Today's serial swing already comes 9.5-27 mm close, with the right arm at its lift pose. Not
  implemented.
- Total at today's speed: about 14-15 s of a 117-122 s run (P1 + P2 with the left lead). With the 50 mm rule
  the expected saving is about 12 s (home parallel on 4 of 7 runs). This is below the 19-24 s estimate of the
  readings doc, because the home needs the lead and P3 does not pass.
- Sim dry runs, serial against parallel, on the same seeds 7041-7070 (n = 30 each): no new stop; 24 of 30 aligned
  in both; seat detected 23 against 24; seated after the release 19 against 18; hand-over lateral error median
  0.65 against 0.54 mm. The dry run teleports the slow moves, so it tests the code path and the plan check, not
  the timing.

## 1. Stage timings checked against the logs

Source: `events.jsonl` and `moves_*.npz` of the real `full_*` runs. Table: `results/real_align/parallel/tables.json`.

| Run | Capture start -> last move end (s) | Log open -> last move end (s) | Difference (s) |
|---|---|---|---|
| 2026-11-13 `full_165400_all9_full_1` | 109.80 | 117.18 | 7.38 |
| 2026-11-13 `full_171043_all9b_full_1` | 114.25 | 121.83 | 7.58 |

The readings doc's 109.8 and 114.3 s are correct from the capture start. The session doc's 117 and 122 s count from
the log open. The gap of 7.4-7.6 s is before the capture: the net preload (0.9 s), the right arm connect (log
time 4.8-5.0 s), the left arm connect (7.2-7.4 s) and the capture prompt. Over all runs with a capture and a home
the gap is 6.3-80.2 s (median 13.9 s, n = 31). The long gaps are operator prompts. The schedule does not change
this time.

Other stage times agree with the readings doc (runs at 0.4 rad/s, n = 7): serial capture moves (right start to
left end) 17.1-18.6 s, serial home 16.9-18.2 s, pick 19.2-23.2 s, left swing 1.94-1.96 s.

## 2. Clearance replay (CPU)

Script: `scripts/real/parallel_clearance.py` (measured and commanded paths), `scripts/real/parallel_clearance.py
planned` (the rig's plan check on the logged start poses), `scripts/real/parallel_tables.py` (tables).

Method:

- Model: the bimanual scene model (`deploy/bimanual/scene.SceneModel`, the same sets as
  `scripts/real/arm_clearance.TwoArm`): every geom of both arms and both bases (links, grippers, wrist cameras,
  webcam visuals, fingers). Joints = logged motor joints + the rig joint offsets. The PC sits at the run's ZED
  estimate; the reader sits at the run's ZED reader estimate.
- Paths: the logged real joint paths at 50 Hz (`moves_<side>.npz` for capture and home, `ticks.npz` for the pick
  and the swing). The second arm's start is moved in time; before its start it stays at its first sample. Both
  the measured and the commanded path are checked; the tables give the smaller distance.
- Runs: every real `full_*` run of 2026-11-10, 11-11 and 11-13 with logs (33 of 35 folders; 2 are empty).
  Runs before 2026-11-11 `fastmove_1` moved at 0.15 rad/s. For them, the measured replay changes the time scale.
  Thus I also ran the plan check (smoothstep paths at 0.4 rad/s from each run's logged start and goal poses).

### Margin

- Plan check: **50 mm** between the two arms on the planned paths. Reasons: the model has no cables, and the wrist
  camera USB cables hang 10-30 mm off the camera bodies. The geometry and the joint offsets have errors of some mm
  (`arm_clearance.py`: a pose that touched read 32.6 mm when the left pose was assumed). At 0.4 rad/s the hand moves
  up to about 5 mm per 20 ms tick, so the stop takes a few mm. The measured path lags the commanded path.
- Watch during the move: **30 mm** on the measured joints of both arms (both arms hold under it).
- A combined move never gets closer than the serial move did on the same run (checked on the measured paths).

### Table: arm-arm distance (mm), measured or commanded path, whichever is smaller

P1 capture. Offset = left start - right start. n = 32 runs.

| Offset (s) | Min | p10 | Median | >= 50 mm | Time saved, runs at 0.4 rad/s (s, n = 7) |
|---|---|---|---|---|---|
| -2.0 | 116.4 | 117.9 | 118.8 | 32 of 32 | 8.3-9.6 |
| 0.0 | 116.4 | 117.9 | 118.8 | 32 of 32 | 8.3-9.0 |
| +2.0 | 116.4 | 117.9 | 118.8 | 32 of 32 | 6.8-7.1 |
| serial | 116.4 | 117.9 | 118.8 | 32 of 32 | 0 |

The minimum is always the end pose (both arms at the capture pose, left finger to left-arm finger pair). That pose
is the same in the serial move. Reader on the table: 81.1 mm or more from either arm (n = 420 run-offset-arm
samples). Plan check at 0.4 rad/s, lead 0: 116.4 mm at the least, 32 of 32 pass.

P2 home. Offset = right start - left start (the left moves first today). n = 31 runs (measured), n = 7 at
0.4 rad/s.

| Offset (s) | Min | p10 | Median | >= 50 mm | Not closer than serial (n = 7 at 0.4 rad/s) | Time saved (s, n = 7) |
|---|---|---|---|---|---|---|
| -1.0 | -22.1 | -17.5 | -0.5 | 3 of 31 | 0 of 7 | 8.3-8.8 |
| 0.0 | -15.8 | -6.1 | 5.7 | 3 of 31 | 0 of 7 | 8.3-8.8 |
| +1.0 | -7.9 | 9.4 | 17.8 | 3 of 31 | 0 of 7 | 7.5-8.4 |
| +2.0 | 1.8 | 20.1 | 36.3 | 6 of 31 | 1 of 7 | 6.5-7.4 |
| +2.5, +3.0, +3.5 | (7 runs) 45.4 | | | 4 of 7 | **7 of 7** | 5.5-6.4 at +3.0 |
| +4.0 | 23.7 | 40.5 | 51.5 | 16 of 31 | 7 of 7 | 4.5-5.4 |
| serial | 23.7 | 45.4 | 59.1 | 21 of 31 | - | 0 |

The negative values are model overlaps: the right hand (webcam or finger) passes through the left hand's path
near the PC. The serial minimum is the start pose plus the left arm's first second: the right hand has just
retreated from the port, the left hand has just lifted from the PC. Plan check at 0.4 rad/s on all 31 runs:
lead 2 s gives "not closer than serial" on 26 of 31, lead 3 s on 31 of 31 (saving 5.1 s median in the plan
model). Arms to the PC at the home start: 4.1 mm at the least (n = 348). This does not change with the schedule
(each arm keeps its own path).

P3 left swing (park -> stand-off, 1.96 s at speed 0.75) during the pick. n = 29 runs.

| Swing start | Min | p10 | Median | >= 50 mm |
|---|---|---|---|---|
| with the pick start | 5.2 | 7.7 | 11.1 | 0 of 29 |
| 6 s before the pick end | 8.9 | 14.7 | 19.3 | 0 of 29 |
| ends with the pick end | 9.5 | 14.9 | 19.5 | 0 of 29 |
| serial (today: after the pick, right arm at its lift pose) | 9.5 | 15.0 | 19.6 | 0 of 29 |

The closest pair is always the right arm's right finger and the left arm's right finger. Today's serial swing
already comes within 9.5-27 mm in the model. P3 cannot pass a 50 mm rule. Not implemented. One more fact for a
later design: in the right wrist camera cone (fovy 69.7 deg, square crop) no left-arm geom is in view during the
pick in 0 of 29 runs, at every offset.

### Camera views and the press budget

- P1: the ZED PC frame is taken after both arms are at rest, as today (`render(..., fresh=True)` waits for a frame
  after the move). The ZED nets' arm-pose gates see the same poses. No wrist frame is used during the move.
- P2: no frame is used during the home.
- The left press is not on during the capture or the home. The 60 s press clock (`safety.hold_max_s`) does not
  change. For the record (E-A3): press-on time from the latch to the home, minus the align stage's press-off
  intervals, 1.0-50.9 s (median 20.4 s, n = 28); full runs of 2026-11-13: 14.2 and 17.4 s. The larger P3 (descent
  and press during the pick) would add up to about 20 s of pick time: 50.9 + 20 > 50 s on one logged run
  (2026-11-10 `k4_5`). That fails the readings doc's E-A3 test too.

## 3. What changed (all off by default)

| File | Change |
|---|---|
| `deploy/bimanual/parallel.py` (new) | Config parse, smoothstep path (RealArm.move_to's formula), arm-arm model, plan check, combined real move, chain entry. |
| `deploy/bimanual/chain.py` | `RealChain.par` from the config; `move(..., parallel=None)`; `capture_pc` and `home(parallel=False)` call the combined move only when the block is on; `capture_prompt`, `home_prompt`, `plan_home`. Absent block: the same `io.move_to` calls in the same order, the same prompt texts, no new events. |
| `deploy/run_bimanual.py` | The capture and home prompts come from `ch.capture_prompt()` / `ch.home_prompt()` (same text when off). `finish` calls `ch.home(parallel=True)`; the recovery home (`recover`) stays serial always. Dry run: the home plan check is logged only. |
| `deploy/config/bimanual_align11_par_home_capture.json` (new) | `bimanual_align10_all` + `parallel_schedule` (capture lead 0 s, home left lead 3 s, rule absolute 50 mm, watch 30 mm, plan fail -> serial). |
| `deploy/config/bimanual_align11_par_home_capture_rel.json` (new) | The same with rule `start_relative` (needs approval, see 5). |
| `deploy/checks/parallel_realio_mock.py` (new) | The real path on mock hardware (real `RealArm.move_to`, SafeArm, FlexChecker, a private stop file). |
| `scripts/real/parallel_clearance.py`, `scripts/real/parallel_tables.py`, `scripts/eval/parallel_dry_ab.py` (new) | Replay, tables, paired dry runs. |

How a combined move works (`parallel.move_real`):

- Each arm runs its own unchanged `RealArm.move_to` in a thread: the same straight-path check by SafeArm, the same
  smoothstep at `safety.move_max_vel_rad_s` (0.4), every command through `SafeArm.command` (joint box, table and
  body clearances, step cap, deviation limit), the same settle trim and the 0.02 rad goal check. Only the start
  times change. The RealIO preparation (precision controller off, bias cleared) is the same per arm. Both
  grippers open together and the code waits for both (max 1 s), not one after the other.
- Before the go: the plan check runs on the planned paths from the MEASURED start poses at 50 Hz, with the lead.
  Its result is in the prompt text. Under the limit: the serial move of today (`on_plan_fail: serial`) or a stop
  (`stop`).
- During the move: the main thread computes the arm-arm distance from both arms' measured joints every 20 ms
  (`move_parallel_watch` event: the smallest value). Under `watch_min_mm`: both stop.
- One stop stops both: during the move, `RealArm.move_to` reads a shared stop (the operator's estop file OR the
  stop of this move). A SafetyStop of one arm, the estop file, the watch or Ctrl-C sets it. The other arm holds
  at its next 20 ms tick. Then `RealIO.hold()` holds both and the SafetyStop goes to the caller (the normal stop
  and recovery path).
- Events: `parallel_schedule` (config), `parallel_plan` (min_mm, start_mm, pair, planned serial and parallel
  time), `parallel_mode`, `move_to` per arm with `parallel: true` and `delay_s`, `move_parallel_watch`.

Checks (offline, no hardware):

- `deploy.checks.parallel_realio_mock`: `FAILED: none`. Serial and parallel moves reach the goals; the parallel
  move takes 1.60 s against 3.09 s serial; a SafetyStop of the right arm stops the left arm 0.20 rad short of its
  goal; the estop file stops both (0.24 rad short); the watch stops both; during a 1.5 s lead the second arm does
  not move.
- `deploy.checks.bimanual_realio_mock` (serial path): ok.
- Conventions check `deploy.checks.bimanual_conventions`: `FAILED: none` on both new configs.

## 4. Dry runs, serial against parallel

Commands (GPU 4, `PYTHONPATH=. MUJOCO_GL=egl`):
`python scripts/eval/parallel_dry_ab.py run --configs serial=deploy/config/bimanual_align10_all.json --seeds 7041-7070`,
then the same with `par=deploy/config/bimanual_align11_par_home_capture.json`, then `summary --json
results/real_align/parallel/dry_ab.json`. Run folders: `runs/parallel_dry/` (not in git). Sim only.

Seeds 7041-7070, n = 30 per config, the same seeds for both. Summary: `results/real_align/parallel/dry_ab.json`.

| | Serial (`bimanual_align10_all`) | Parallel (`bimanual_align11_par_home_capture`) |
|---|---|---|
| Runs completed, no stop | 30 of 30 | 30 of 30 |
| ZED gate passed / pick held | 30 / 30 | 30 / 30 |
| Aligned (hand-over within tolerance) | 24 of 30 | 24 of 30 |
| Seat detected | 23 of 30 (+1 guard bound) | 24 of 30 |
| Seated after the release (sim truth) | 19 of 30 | 18 of 30 |
| Hand-over lateral error, median | 0.65 mm | 0.54 mm |
| Hand-over gap, median | 5.91 mm | 6.16 mm |
| ZED PC error, median | 3.48 mm | 3.48 mm |
| Alignment time, median | 9.9 s | 10.8 s |
| Rest depth at the end, median | 5.65 mm | 5.63 mm |
| New guard or safety stop | - | none |

- The distributions agree within sim noise. Per seed, 13 of 30 pairs differ in the alignment or seat outcome, in
  both directions (for example 7041: serial seated, parallel not aligned; 7045: serial guard bound, parallel
  seated). The capture move is the same teleport in both configs, so these differences are MuJoCo Warp
  non-determinism, not the schedule. "Not seated after the release" occurs in both configs (serial 4 of 23 seats,
  parallel 6 of 24), as known with K4.
- Seeds 7041-7045: serial seated 7041, 7042, 7043, 7044 (7045 guard bound); parallel seated 7042, 7043, 7044, 7045
  (7041 not aligned); after the release, parallel 7043 and 7045 not seated (the known release effect).
- Capture plan check in the sim: 124.3 mm, 30 of 30 pass. The sim start pose is near the capture pose (right move
  1.1 s), so the sim saving (1.1 s) does not stand for the rig; the rig saving comes from the real start poses
  (section 2).
- Home plan check (logged only, the dry run does not home): 25 of 30 pass, min 40.0 mm, median 64.6 mm. The
  5 failures go to the serial home. Example: seed 7041 ended without an insertion (`end_no_insert`), and the
  arms started the home 40 mm apart.
- `bimanual_align11_par_home_capture_rel`, seeds 7041-7045 (n = 5, `results/real_align/parallel/dry_rel.json`):
  5 of 5 completed, no stop, 4 aligned and seat detected, 2 seated after the release (the known release effect).
  Home plan: 5 of 5 pass (min 49.3 mm, which the absolute rule would send to the serial order).
- No dry run timed the moves (teleports). The run-time saving is the real-log estimate of section 2.

## 5. Approvals needed

**A1. One go for one combined two-arm move.** The rule "one operator go per motion command" stays. The capture move
is already one command for two arms under one go today (the arms move one after the other). The new reading: a
combined move, where both arms move at the same time, is one motion command under one go. The prompts:

- Capture: `[capture: BOTH arms move AT THE SAME TIME to the ZED capture pose, grippers fully open (planned arm-arm
  119 mm, limit 50; parallel)] Enter = go, s = skip the rest, q = abort >`
  What moves: both grippers open fully, then the right and the left arm move at the same time from the start pose
  to the ZED capture pose, each at 0.4 rad/s peak, each on its own straight joint path (the same paths as today).
- Home: `[home: BOTH arms move AT THE SAME TIME to the start pose, left starts, right 3.0 s later (planned arm-arm
  62 mm, limit 50; parallel)] Enter = go, s = skip the rest, q = abort >`
  What moves: the left arm starts home from its lifted pose above the PC; 3.0 s later the right arm starts home from
  its retreat pose; both at 0.4 rad/s peak on today's paths. If the plan shows `FAILS: serial`, the same go starts
  today's serial home (left, then right).
  (The mm values in the prompt are computed from the measured poses at that moment; the values above are from
  `full_165400_all9_full_1`.)
- Run these first runs without `--demo` / `--yes`: these flags answer the prompts automatically.

Variant with separate goes: two goes keep the rule word for word, but the saving needs the second go while the
first arm still moves (within 0 s for the capture, within 3 s for the home). Every operator delay goes into the
run time, and a missing answer leaves one arm moving alone. Thus a two-go variant loses most of the capture saving
and is not implemented. It is possible to add later (the second arm's thread waits for its go instead of a fixed
lead).

**A2 (optional). The `start_relative` rule** (`bimanual_align11_par_home_capture_rel.json`). On 3 of 7 recent runs
the arms start the home 45-49 mm apart, so the 50 mm rule sends those homes to the serial order. The relative rule
lets the combined move run if the planned paths never come closer than the start distance minus 2 mm and never
under 30 mm. Today's serial home has no arm-arm check at all, so neither rule loosens an existing limit. The
relative rule accepts the serial home's own closest pose as the bound. In the plan replay at lead 3 s, 31 of 31
runs are not closer than their serial home, and 30 of 31 are over the 30 mm floor (one run, 2026-11-10 `k4_2`,
has a serial minimum of 23.7 mm; the serial minimum stands in for the start distance here). The 7 runs at
0.4 rad/s all pass. Gain: about 2-3 s more per run on average.

No limit is looser. No speed changes. The 60 s press clock, the guards and SafeArm's checks are unchanged.

## 6. Rig runbook (fewest runs)

Offline on the rig PC first:

```bash
.venv/bin/python -m deploy.checks.bimanual_conventions --config deploy/config/bimanual_align11_par_home_capture.json | tail -1   # FAILED: none
.venv/bin/python -m deploy.checks.parallel_realio_mock | tail -1                                                                # FAILED: none
```

Run 1 (capture only, no press): both arms to the capture pose in parallel, the estimates, then the parallel home.

```bash
.venv/bin/python -m deploy.run_bimanual --stage capture --config deploy/config/bimanual_align11_par_home_capture.json --name par_capture_1
```

- Pass: both prompts show `parallel`; no stop; `move_parallel_watch` min_mm >= 80 for the capture; the ZED PC
  estimate within 2 mm and 1 deg of the last serial capture at the same spot; the capture move ends 8-9 s sooner
  than the serial one (event times).
- Stop the series: any contact or a sound of contact, any SafetyStop, the watch fires, or the measured watch
  minimum is more than 20 mm under the plan's min_mm.

Runs 2 and 3 (full chain, operator goes on):

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --align-mode full --speed 0.75 --config deploy/config/bimanual_align11_par_home_capture.json --name par_full_1   # then par_full_2
```

- Pass: the same outcome as `bimanual_align10_all` (seat, depth, hand-over error inside the usual range); home
  parallel when its plan says `parallel`; capture start to last move end at least 12 s shorter than all10 runs of the
  same session (all9 runs: 109.8 and 114.3 s).
- Stop: as run 1. Also stop if the home plan says `FAILS` on both runs (then the home stays serial; only P1 is
  useful) and report the start_mm values.
- Then `--demo` may be used again for the parallel config, if the operator agrees.

## 7. Limits

- The model has no cables. The 50 mm plan limit and the 30 mm watch cover them only in part.
- The measured replay at today's speed has n = 7 runs, at one PC spot and one reader spot.
- The dry run teleports the slow moves: it cannot show a timing or a collision effect. The real combined move is
  only tested on mock hardware (first-order joints, no CAN). GIL load: the watch thread computes one model distance
  every 20 ms next to two 50 Hz move loops and the 200 Hz precision threads; the per-tick lateness is in the moves
  logs on the rig.
- P3 and the larger P3 are not implemented: they fail the clearance rule, and the larger one fails the press
  budget test.
