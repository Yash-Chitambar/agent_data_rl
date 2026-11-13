# Fast alignment (task 2 of SERVER_TASKS_20261112, 2026-11-13)

Task: make the camera measurements and the align stage much faster. Target: under 15 s, same hand-over accuracy (lateral <= 1 mm, gap 6 +/- 1 mm by the measurement).
Everything here is logs, an offline model, a simplified plant and sim dry runs. **Nothing here has run on the arm.**

New files: `deploy/config/bimanual_align8_fastalign.json` (step stage, the safe deliverable), `deploy/config/bimanual_align8_fastalign_servo.json`
(servo with a stick-slip fix, align mode only, a measurement run), `scripts/real/align_fast_model.py` (time budget and model),
`scripts/real/align_fast_dry.py` (dry-run table), `deploy/checks/align_servo_stickslip.py` (servo against joint stick-slip).
Code: `deploy/bimanual/align.py`, new keys only, every default keeps the old behaviour. `deploy.checks.align_gates`: FAILED: none.
Results: `results/real_align/align_fast/` (`model.json`, `dry_runs.json`, `servo_stickslip*.json`).

## 1. Where the time goes now (real logs, n = 15 step-mode runs of 2026-11-10/11)

Wall clock from `ticks.npz` and the event times (`align_start` to `align_result`). The stage's own `t_stage` counts control ticks and misses the
stalled ticks, so it reads 30-55 s where the wall clock reads 42-69 s.

| Part (median per run) | Now | Why |
|---|---|---|
| Whole stage (wall) | **59.7 s** (42.4-69.4) | |
| Steps | 37.0 s | 123 of 229 steps ran to the 3 s timeout. The reach test is 0.3 mm on the 3-D distance; the plain-PD hand stops near or past the goal and stays there (median moved / asked 1.09 at the timeout). |
| Measurements | 19.8 s, 0.85-0.97 s each | rest wait >= 0.3 s; 5 frames x 87 ms; the second source (wrist net) every cycle. |
| Probe cycles (6, measure + step) | 20.8 s (inside the two rows above) | +/-2-3 mm on each axis and back, every run. |
| Prepare (shell depth, 9 frames) | 0.9 s | |
| Press back on + post-press checks | 3.9 s | 1 s ramp + 3-5 measurements. |

**Per frame (87 ms median, n = 1276 frames):** port_vision itself takes about 26 ms (the same function in the servo's worker process on the rig:
p50 26 ms, p5-p95 22-34 ms, n = 183 frames). The rest is `io.fresh_frames`, which waits for **two** new camera frames (24 frames/s: 42-83 ms) for every
frame of a measurement, plus the control tick. Only the first frame of a measurement needs a frame captured after the arm is at rest.

### Against the data review (`DATA_REVIEW_20261113.md`, findings 11-14)

- Agree: timed-out steps are the largest part (my count 123 of 229 steps including the probes; the review 122 of 212), the probes cost
  10-17 s per run, the loop is late during every measurement (128-204 late ticks, compute p99 92-98 ms), and `a4p_1` (precision controller)
  timed out 1 of 17 steps.
- Disagree in part with "port vision runs in the control thread at about 85 ms per frame": the vision call is about 26 ms; about 60 ms of each
  frame is the two-frame camera wait. Moving the call to a worker process alone saves little in the step stage, because the stage measures only at
  rest and waits for the result anyway. The fix used here: only the first frame waits for two new frames, the others wait for one new frame, and
  3 frames instead of 5. The late ticks during a measurement do not move the arm: the arm is at rest and the target is held.

## 2. What changed (all off by default; on in `bimanual_align8_fastalign.json`)

| Key | Value | What it does |
|---|---|---|
| `reach_mode` | `loose` | Step done when each asked axis is within 0.3 + 0.5 x asked and each other axis within 0.6 mm (`reach_side_mm`) for 3 ticks, or each asked axis has done >= 0.6 of its part and the tip moved < 0.15 mm in 0.5 s (`reach_ok`). Per axis: a first version on the 3-D distance passed a step that closed the gap but left "up" behind (dry run, up moved 0.08 of 1 mm). |
| `reach_freeze` | **false** | Option: after a loose end, the joint goal becomes the measured pose. Rejected: dry run seed 7044 with it, joint 2 still had about 20 mrad of lead at the step end, the hand rose about 2 mm between steps and the stage stopped on 'correction grows' (true up 7 -> 14.7 mm). Without it the joints finish the step during the rest wait, as after a timeout in the old stage. |
| `step_timeout_s` | 2.0 (was 3.0) | |
| `probe_mm`, `slope_prior` | [0, 0, 0], known axes up and gap | No probe cycles. The scale gate at the hand-over uses the motion of the corrections: gap and up must be known (se <= 0.2) or one extra probe on that axis (`extra_probes` 2); across fails only when 2 se out of [0.6, 1.6]. Real slopes, n = 15: across 0.94-1.10, up 0.76-0.98, gap 1.02-1.12. Logic test: an up slope of 0.3 stops at the scale gate (3 of 3 starts). |
| `gap_step_ratio`, `gap_step_max_mm` | 0.5, 3 | While lined up (lateral <= 2 mm): gap step = clip(0.5 x (gap - 6), 1, 3) mm: 3 mm at 12 mm and more, 1.5 at 9, 1 below 8. Not lined up: 1 mm and the funnel as before. |
| `frames`, `min_good` | 3, 2 | Frame noise is under 0.12 mm; the pose residual (0.25-0.74 mm) dominates, and 3 frames do not change it. |
| `fresh_frames` | `first` | Later frames wait for one new camera frame, not two. Real path only (the sim source has no camera). |
| `settle_s` | 0.1 (was 0.3) | The still test (joint speed < 0.02 rad/s) stays. |
| `secondary_near_mm` | 1.0 | The wrist net runs only when the measurement is within tol + 1 mm of the target; it is used only at the hand-over gate. |

Unchanged: target (0, 0, 6) mm, tolerance 1 mm per axis, `confirm_n` 2, three confirms under the press, every gate and limit, the entry gap
limit (12 mm), the approach stop (18 mm).

## 3. Offline time model on the real runs (`align_fast_model.py`, n = 15)

The model replays every logged step through `reach_ok` on its FK trace (new end tick and the fraction done then), drops the probe cycles,
recounts the gap cycles with the new gap steps (each run's median gap fraction at the new step end), and prices each measurement with 3 frames
(first 107 ms, later 67 ms incl. the tick), a 0.1 s settle plus the measured rest wait, and the second source only near the target.

| | Now (logs) | Model |
|---|---|---|
| Stage wall time p50 (range) | 59.7 s (42.4-69.4) | **14.3 s (11.9-18.4)** |
| Steps that time out | 123 of 229 | 42 of 229 (2.0 s timeout) |
| Step length p50 | 150 ticks (3.0 s) | 51 ticks (1.0 s) |
| Probe cycles | 6 per run | 0 (an extra probe only if up or gap did not move enough) |
| Correction cycles p50 | 9 | 6 |
| Cost per correction cycle / per confirm | about 3.9 s / 0.9 s | 1.44 s / 0.34 s |

Model budget, median run: 6 correction cycles x 1.44 s + 3 confirms x 0.34 s + press and post-press 2.0 s + prepare 0.9 s.
9 of 15 runs come out at or under 15 s; the slowest (k4_2, demo_1, kjrc_2, demo_5, k4_4: 16.8-18.4 s) have more correction cycles, or a
failed hand-over with a back-off inside the stage time (k4_4, demo_5). Limits of the model: an earlier step end changes the next pose (the next correction
makes up for it); the later-frame time is an estimate (26 ms vision + one camera frame on average); the stall exit at 0.6 of a step can
add correction cycles (counted through the gap fraction, not for the lateral axes).

Sensitivity (same model, before the asked-axis rule; within about 0.3 s): `reach_rel` 0.3 -> 16.2 s (with the asked-axis rule); 5 frames -> 15.5 s;
two-frame wait on every frame -> 14.9 s; `gap_step_max_mm` 4 -> 12.6 s; the strict test with a 1.2 s timeout -> 15.0 s (but see the dry runs).

## 4. Sim dry runs (`deploy.run_bimanual --dry-run --stage approach_insert --align-mode full`)

Seeds 7041-7048 (`fast` = `bimanual_align8_fastalign.json`; 7041 and 7042 ran with a copy identical in the align keys), 7041-7045 for the old
config; `results/real_align/align_fast/dry_runs.json`, `scripts/real/align_fast_dry.py`.

| Config | n | Aligned (incl. 3 confirms under the press) | Sim stage time p50 (range) | Probe cycles | Step timeouts | True hand-over lateral <= 1 mm | True gap | K4 seat detected | True seated after release |
|---|---|---|---|---|---|---|---|---|---|
| `bimanual_align5_k4_push15_seat6_fastmove` (old) | 5 | 3 (2 x "after the press: step limit") | 39.3 s (33.0-47.1) | 6 each | 4-5 per run | 2 of 3 | 4.8-6.8 mm | 3 of 3 | 3 of 3 |
| **`bimanual_align8_fastalign`** | 8 | **8** | **16.0 s (6.5-37.2)** | 0 (one extra probe on 7043) | 0-4 per run | 6 of 8 | 5.1-6.4 mm | 8 of 8 | 6 of 8 |
| fast + `reach_freeze` true | 4 | 3 (7044: "correction grows") | 17.2 s (11.4-20.7) | 0 | 0-1 | | | 2 of 3 | 2 |
| fast + `reach_rel` 0.3 | 3 (7041, 7044, 7046) | 1 | 29.2 s | 0 | | | | 0 | 1 |
| fast with the strict test, 1.2 s timeout | 3 (7041, 7044, 7046) | 2 (7046: "correction grows") | 28.9 s | 0 | | | | 2 | 2 |

- The fast stage aligns in all 8 dry runs, about 2.5x faster in sim time. The two lateral misses (seeds 7042, 7043: up -1.6 and +1.6 mm by truth)
  are the sim's port-vision error draw; the old config on seed 7042 ended with up -2.0 mm by truth too.
- **Two slow seeds, 7044 and 7046 (37 s, 39 correction cycles).** Their friction draws make the hand rise in "up" after a short step: the FK
  tip moves up 0.5-0.9 mm per cycle on average between the step end and the next measurement (the old config on the same seed: 0.15 mm).
  The real logs show the same effect on some runs (k4_4: +0.58 mm per cycle mean, p90 2.1 mm). The stage still converges and the gates hold.
  Shorter steps by other means are worse on these seeds: the strict test with a 1.2 s timeout, `reach_rel` 0.3 and `reach_freeze` each gave
  one "correction grows" stop with the hand 15-23 mm too high by truth (the gate stopped it). So `reach_rel` 0.5 without freeze is kept.
- K4 seats after 8 of 8 fast hand-overs (FK detector); 2 readers came out at the release (a known dry-run release effect,
  `RETRY_SPEED_20261108.md` section 2).

Sim time counts control ticks (the sim measurement takes no wall time), so the sim times are shorter than real ones; compare old and new
within the sim. The sim starts the align stage about 20 mm out (the real one 17-19 mm). Hand-over truth = sim truth at the last measurement;
the lateral truth error up to 2 mm comes from the sim's port-vision error model (run bias and pose error), the same for both configs.

## 5. Servo sway (real 2026-11-10 `full_144605_servo_1`, `full_145249_servo_2`)

**Cause: joint stick-slip, not the command and not the vision.**
- Commanded tip speed p50 / p99 / max: 4.1 / 8.8 / 8.9 mm/s (servo_1), 7.4 / 11.0 / 11.0 (servo_2). Measured tip speed (FK, 0.1 s window):
  4.5 / 23.8 / 27.7 and 8.7 / 24.8 / 28.0 mm/s. Tip path 61.8 mm for 10.9 mm net (servo_1).
- The joints lag their smooth targets and catch up in jumps: joint 2 stayed still for 3.5 s while its target moved 32 mrad (lead -20 to +12 mrad),
  then moved 12 mrad at once; joint 5 ran 25-35 mrad behind its target the whole time (kp 10: about 0.3 N m of stuck friction); joints 1, 3 and 6
  move in stair steps. In the 0.1 s windows above 15 mm/s, joints 3, 4 and 5 give most of the motion.
- Vision is not the cause: the estimate change at a new frame not explained by FK motion is p50 0.1-0.2 mm and p90 0.2-1.6 mm per axis and does not correlate
  with the tip speed (slope 1-11 ms, |r| <= 0.23), so the latency constant is not the problem; the frame age at use is <= 20 ms.
- Why the sim missed it: the sim joint friction is a Coulomb band (static = kinetic). With that plant the servo stays slow (dry run p99 13.1 mm/s).

**Reproduction** (`deploy/checks/align_servo_stickslip.py`: 6 independent joints with static friction > kinetic, breakaway torques fitted by eye
to the real leads, real kp/kd, 8 ms latency, YAM kinematics, vision 23 frames/s, 40 ms latency, real-like error; n = 32 starts per row):

| Servo | Plant | Done | Time p50 / p90 s | Tip speed p50 / p99 mm/s (worst run p99) | Path ratio |
|---|---|---|---|---|---|
| align6 (gain 1.5, vmax 8) | Coulomb (static = kinetic) | 26/32 | 2.3 / 4.5 | 2.7 / **10.1** (11.1) | 1.19 |
| align6 | stick-slip | 30/32 | 3.1 / 9.8 | 3.2 / **32.3** (37.4) | 1.67 |
| align6 | stick-slip, Fs x 0.7 | 32/32 | 2.4 / 3.4 | 5.7 / **22.5** (30.5) | 1.38 |
| feed-forward off | stick-slip | 23/32 | 3.1 / 8.3 | 3.4 / 32.4 (41.5) | 2.12 |
| vmax 4 | stick-slip | 29/32 | 3.7 / 9.0 | 3.8 / 30.5 (37.0) | 1.67 |
| gain 0.8 | stick-slip | 29/32 | 4.8 / 11.9 | 1.8 / 27.9 (35.9) | 1.65 |
| lead cut 0.5 + vmin 2.5 | stick-slip | 32/32 | 4.1 / 9.8 | 3.4 / **17.6** (21.4) | 1.90 |
| lead cut 0.5 + vmin 2.5 | stick-slip, Fs x 0.7 | 32/32 | 2.5 / 5.2 | 5.0 / **16.2** (23.5) | 1.38 |
| lead cut 0.5 + vmin 2.5 + vmax 5 | stick-slip, Fs x 0.7 | 31/32 | 3.2 / 6.1 | 4.9 / 14.2 (20.2) | 1.50 |
| lead cut 0.3 + vmin 2.5 | stick-slip | 22/32 | 5.0 / 10.2 | 3.8 / 15.1 (23.4) | 2.20 |
| lead cut 0.5 | Coulomb | 26/32 | 2.3 / 4.5 | 2.7 / 10.1 (11.1) | 1.19 |

- The plant reproduces the effect: the same servo gives p99 10 mm/s on the Coulomb plant and 22-32 mm/s with stick-slip (real 24-25).
- Command-side changes (lower gain, lower vmax, feed-forward off, filtering) do not remove it: the jump size is set by the stored spring at breakaway,
  not by the target speed.
- **Lead cut** (`servo.lead_cut`, `deploy/bimanual/align.py servo_lead_cut`, the step stage's creep cut per tick): a joint that was at rest and
  now moves toward its target keeps only `lead_cut_release` of its target lead. It halves the p99 (32 -> 18, 22 -> 16 mm/s). It does nothing on the
  Coulomb plant (same numbers), so it is safe where there is no stick-slip. Release 0.3 cuts more but stalls (22/32 done).
- It does not reach the 15 mm/s pass limit with margin (16-18 mm/s; with vmax 5: 14.2 at Fs x 0.7). The servo stays a measurement, not the
  deliverable: `bimanual_align8_fastalign_servo.json` = lead cut 0.5 + vmin 2.5 (vmax 8), align mode only (`full_allowed` false).
  Servo dry runs (align mode, sim Coulomb plant, so the cut has little to do): seeds 7100 / 7101 aligned in 3.1 / 3.0 s, tip speed p99
  13.8 / 20.0 mm/s; the same config without the cut: 13.3 / 4.3 s, p99 15.1 / 18.5 mm/s. n = 2 each: a code-path check, not a rate.
- The pass check itself: the 15 mm/s p99 limit is not a safety limit in the stage (the stage stops at 60 mm/s). No change proposed.

## 6. Limits

No limit changes. Not applied, for the operator:
- A closer approach stop (`chain_args.stop` 18 -> 14 mm) would save about one 3 mm gap cycle (about 1.4 s). Not done: the approach jump
  (3.6-5.0 mm on 6 runs) and the 12 mm entry gap limit leave too little margin; another worker owns the approach.

## 7. Rig runbook (step stage first)

```bash
C=deploy/config/bimanual_align8_fastalign.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C | tail -1          # FAILED: none
.venv/bin/python -m deploy.checks.align_gates | tail -1                              # FAILED: none
.venv/bin/python -m deploy.run_bimanual --dry-run --stage approach_insert --seed 7042 --config $C --align-mode full --name dry_fastalign   # "align: OK", 0 probes
# rig: one align-mode run first, then full
.venv/bin/python -m deploy.run_bimanual --stage full --config $C --align-mode align --speed 0.75 --name fastalign_a1
.venv/bin/python -m deploy.run_bimanual --stage full --config $C --align-mode full  --speed 0.75 --name fastalign_1
.venv/bin/python scripts/real/align_log.py data/real_runs/<date>/full_<id>            # cycles, steps, decisions
.venv/bin/python scripts/real/align_fast_model.py data/real_runs/<date>/full_<id>     # wall-time budget of the run
```

Pass (each run): align stage wall time under 20 s (target 15); no probe cycles (an extra probe on up or gap is allowed); hand-over within
tolerance and three confirms under the press (as before); scale gate passed; the sources-disagree gate as before.
Watch: step reports `reached` with `ticks` 10-60; the hand stays still during measurements (`align_measure` spread <= 0.5 mm); the "up"
coordinate of `tip_rel` should not rise between a step end and the next measurement by more than 1 mm on two cycles in a row (the slow
dry-run seeds); if it does, finish the run and report it (the gates stop a real runaway).
Stop and go back to `bimanual_align5_k4_push15_seat6_fastmove.json` if: a scale-gate stop with vision on target; "frames disagree" or
"measurement against FK" stops; any step moves more than 3.5 mm; a gap below 4 mm during the stage.

Servo measurement (only after the step runs, align mode, the arm free of the port):
```bash
.venv/bin/python -m deploy.run_bimanual --stage full --config deploy/config/bimanual_align8_fastalign_servo.json --align-mode align --speed 0.5 --name servo_lc_1
```
Pass: tip speed p99 <= 15 mm/s (the 2026-11-10 check), operator sees no sway. Expected from the plant: 16-18 mm/s, so a fail is likely; it tells
whether the lead cut moves the real p99 from 24-25 down as the plant says.
