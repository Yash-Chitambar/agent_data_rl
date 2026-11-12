# Session 2026-11-10, duchamp3: runbook NEXT_RIG_20261109

Branch `real/graph-align` at `89d6ed3`. Run logs `data/real_runs/2026-11-10/` (not in git).

## Setup and no-motion checks

- K4 weights `runs/insert_K4.pt`: sha256 `4d288f234ff1af10` (matches `bimanual_align5_k4.json`). r10 wrist nets `384e7b43b7e0e937`, `78064b5eeab8bbce`. Bundle OK.
- Conventions: `FAILED: none` for align5_precision, align4_kjrc, align5_k4, align6_servo, align7_retry, align7_preload.
- `test_precision_ctrl.py` 15 passed; real-I/O mock ok; bumpless switch-off 0.000 mrad.
- Dry run K-J re-centred (seed 7020): completed, seated, `insert_obs_recenter` logged.
- Servo real path on the rig's copy of `align_full9`: pass (process and inline identical). Servo thread timing on the rig CPU: real_process loop p50/p99/max 20.0/20.3/20.9 ms, 0 ticks over 25 ms, 24.2 frames/s. (Both checks' default paths are mason paths; run with the rig's run dir.)

## Runs

| Time | Run | Block | Outcome |
|---|---|---|---|
| 14:30 | `full_143002_a4p_1` | A | no events logged (start aborted) |
| 14:30 | `full_143024_a4p_1` | A | aligned in 29.5 s, 21 cycles, steps p10/p50/p90 0.92/1.05/1.15; **approach: uncommanded move 4.14 mm (limit 3), joint 2 error 33.2 mrad (limit 25)**; `precision_bumpless_off` logged once |
| 14:40 | `full_144059_servo_1` | A2 | software stop at the align start (`TypeError: log_event() got multiple values for argument 'kind'`, `align.py:959`, real-arm-only path; the sim dry run uses the deferred worker and never reaches it). No servo motion; release and home ran. Fixed: the event field is now `worker`. |

Approach jump on earlier runs, same tool: prec7 8.37 mm, full9 4.84 mm, full10 2.83 mm.

Operator, `a4p_1`: the hand **sagged slightly** when the controller switched off at the end of the align stage (`precision_bumpless_off` was logged). Block A's extra check fails: no full-mode run with the controller. B to D run on plain PD as planned.

`full_144605_servo_1` (A2, plain PD): `align servo: OK` in 7.8 s, 183 frames (23/s), vision worker = process, frame age max 0.02 s, final estimate (across, up, gap) = (0.76, -0.78, 5.51) mm against target (0, 0, 6), approach move 0.97 mm (joint 2/3 max 14.6/3.2 mrad), 1 late tick in 363 (loop p99 22.9 ms). **Fails one pass check: tip speed p50/p99/max 4.5/23.8/27.7 mm/s (limit p99 15).** The commanded speed stays at or below 8.9 mm/s; the excess is a lateral wobble of about 1.4 Hz, ±1-2 mm (tip path 61.8 mm for a 11.0 mm net move; sim dry run 31.0 for 15.9, speed p99 13.1).
Operator, servo_1: the plug swayed a little during the servo alignment (visible).

`full_145249_servo_2` (A2, plain PD): `align servo: OK` in 2.9 s, 60 frames, worker = process, frame age max 0.04 s, final (0.39, 0.72, 6.60) mm, approach move 1.47 mm, 2 late ticks of 119. **Tip speed p50/p99/max 8.7/24.8/28.0 mm/s: fails again** (path 31.3 mm for 11.2 net, lateral wobble rms 0.42 mm). A2 fails 2 of 2 on tip speed only, so no full chain with the servo (C uses the step alignment, D is blocked). Next: block B.

`full_145824_kjrc_1` (B, plain PD, K-J re-centred): approach move 1.86 mm (joint 2/3 max 12.4/8.3 mrad); step align 21 cycles, 54.9 s (slow at the 18 mm entry: the first 30 s stayed near 15-19 mm gap), hand-over gap 5.58 mm, lateral (0.06, 0.60) mm; `insert_obs_recenter` logged (shift about 0.1-0.3); K-J `seated_detected` at 2.22 s, FK depth max 6.62 mm, target up max 13.2 mm, tip up max 2.9 mm; contact head share (free/touching/inside) at 0.2 s 0.01/0.73/0.26, then inside 0.86 at 0.4 s, touching 0.83-0.98 from 0.6 s; plots in the run folder `analysis/`. Operator: seated, but the reader **hangs (droops) more than it should** in the port. FK depth max 6.62 mm against 7.98 mm on `align_full9`, so possibly not fully home.

`full_150628_kjrc_2` (B): **approach move 4.36 mm (limit 3), joint 2 error 28.0 mrad (limit 25)** on plain PD (so the approach wind-up is not the controller alone: plain-PD approaches today 0.43, 0.97, 1.47, 1.86, 4.36 mm); step align 20 cycles, 54.8 s; hand-over gap 6.06 mm, lateral (0.74, 0.40) mm; `insert_obs_recenter` logged; K-J `seated_detected` at 1.76 s, FK depth max 6.43 mm, target up max 29.0 mm (just under the 30 mm mark), contact head inside 0.90 at 0.4 s. Operator: seated, 2 of 2; the reader still hangs; about 1 mm more insertion would make it right. The seat detector stops K-J at depth >= seat_depth - 1 mm (`insert.seat_margin_m` 0.001, seat2 margin 1.0 mm), and the release follows at once, so the plug is never pushed the last millimetre.

## Final push (operator approved, 2026-11-10)

`insert.final_push` (`deploy/bimanual/insert.py` `final_push`): after a detected seat and the 0.2 s hold, the right target moves straight into the port by 5 mm at 4 mm/s (x --speed), holds 0.5 s, then the measured pose for 0.3 s; event `insert_final_push` with the FK advance. New configs `bimanual_align4_kjrc_push.json`, `bimanual_align5_k4_push.json` (originals unchanged). Conventions `FAILED: none` twice. Dry runs: kjrc_push seeds 7020/7021 seated, push advance -0.01/0.04 mm (the sim plug is already at its 5.7 mm seat), true seated after release; k4_push seed 7040 push advance 0.24 mm, true seated at the end false (the plain K4 dry run on seed 7040 also ended unseated after the release).

`full_152149_kjrc_3_push` (B + push): approach 1.90 mm; hand-over gap 5.57, lateral (0.39, -0.39) mm; K-J seated at 1.40 s, FK depth max 6.12 mm, target up max 17.4 mm; push advance 0.93 mm.
`full_152717_k4_1` (C + push): approach 1.69 mm; hand-over gap 5.69, lateral (-0.19, 0.76) mm; `insert_vision_prior` = the hand-over measurement exactly; `insert_rel_pose_obs` logged; K4 seated at 1.52 s, FK depth max 8.02 mm, target up max 32.3 mm; push advance 0.38 mm.
Operator, both: seated and held the reader; still about 3 mm short of fully in after the push; maybe a stronger grasp is needed. Seats today 4 of 4 (kjrc_1, kjrc_2, kjrc_3_push, k4_1).
Push analysis (ticks): the 5 mm push target added only ~9-11 mrad on joints 1, 4 and 6 (joints 2-3 unchanged), joint 1 motor torque -0.35 -> -1.05 N m; roughly 2-3 N along the port axis, far below a USB-C mating force (spec up to 20 N). The gripper reading stayed at 0.302/0.304 (no jaw change). The hand stopped after under 1 mm: the plug resisted ~2-3 N, no sign of slip. The push is too weak, not slipping.

## Final push 15 mm with gates (operator approved)

`bimanual_align5_k4_push15.json`: push 15 mm at 4 mm/s (x --speed), hold 0.5 s. Runs only when K4 reports `seated_detected` and seat-2 reads depth >= 5.0 mm and |lateral| <= 2.5 mm (today's seats: 6.0-8.0 mm, <= 1.83 mm); otherwise event `insert_final_push_skipped`. Stops and holds the measured pose if the right hand advances > 4.0 mm, the left hand (PC press) moves > 2.0 mm, or either gripper reading changes > 0.02; the left arm keeps its press target throughout. Conventions `FAILED: none`. Dry runs: seed 7041 pushed (advance 0.85 mm, left 0.64 mm, grip change 0), seated after; seed 7042 guard stop, no push (correct); seed 7040 align failed (sim), no insertion; skip path checked with min_depth 99 (`final push skipped: depth 5.28 mm < 99.0`). Real 5 mm pushes moved the left hand by roughly <= 0.6-1.3 mm (joint-change bound), so the 2 mm left stop may trip at 15 mm if the PC press yields.

`full_154422_k4_2` (C, push15): approach 0.64 mm; hand-over gap 6.01, lateral (-0.12, -0.30) mm; prior = hand-over. **Outcome `guard_bound` at 4.62 s, no seat detected, so no push; the guard backed off 20 mm with the reader in the jaws.** The plug was in: seat-2 depth >= 4.8 mm for 191 ticks (3.8 s), 8.4-9.0 mm deep, lateral within 1 mm until about 2.4 s. The seat check needs 15 ticks in a row with axial speed <= 3 mm/s; K4 kept moving (|v| p50/p90 1.8/5.8 mm/s while deep), longest run 14. K4 kept searching after it was in (target up to 32 mm up and 21 mm across), and the hand then drifted 8 mm down by FK with the plug still 8 mm deep (the reader turning in the jaws or in the port; jaw reading 0.3029 -> 0.3022), until lateral passed the 6 mm gate and the 5 mm bound stopped it.
Replay of the seat check on all 10 real insertions with guard logs: speed limit 6 mm/s (hold 15 ticks) detects every seat earlier (k4_2 at step 67) and no seat on the three failures (their max depth 3.2-3.6 mm; seats reach 6.1-9.4 mm).
Operator, k4_2: the back-off pulled the reader out of the port. Approved: seat2 speed limit 3 -> 6 mm/s, config `bimanual_align5_k4_push15_seat6.json`.

`full_155541_k4_3` (C, push15_seat6): **approach move 3.58 mm (limit 3)**, joint 2/3 max 19.5/13.4 mrad; step align 17 cycles, 49.0 s; hand-over gap 5.92, lateral (0.08, 0.20) mm; prior = hand-over; K4 `seated_detected` at 1.68 s, depth 8.47 mm (target up max 34.8 mm); **final push advance 2.17 mm** (5 mm pushes: 0.38-0.93), not stopped, left hand 0.12 mm, right jaw reading change 0.0098 (limit 0.02). Operator: seated, worked really well.
Seats today: 5 of 6 insertions (kjrc_1, kjrc_2, kjrc_3_push, k4_1, k4_3; k4_2 was in but undetected and pulled out, fixed by seat6).

`full_160352_k4_4` (C): align stopped at the hand-over, `sources disagree at the hand-over [0.02, 3.54, 0.76] mm` (limit [3.0, 3.5, 2.5]); vision read (-0.53, -0.38, 6.46) mm (aligned); back-off 15 mm, reader lifted, operator held it while the gripper opened, home. No insertion. The wrist net's "up" reads 2-3 mm off vision on nearly every run (2026-11-03..10 hand-overs: 0.3-4.5 mm; today's six seats 1.9-3.0 mm), so the 3.5 mm gate sits at the edge of the normal spread.

`full_161011_k4_4` (C, rerun): approach 2.35 mm; vision-net disagreement (0.50, 2.61, 1.06) mm; hand-over gap 6.50, lateral (-0.17, 0.78) mm; K4 seated at 1.44 s, depth 7.76 mm (target up max 26.7 mm). **Final push stopped by its gripper watch**: right jaw reading 0.3003 -> 0.2794 (closing, -0.020) in the last ~1.5 s of the push, advance 2.74 mm, left hand 0.15 mm. The jaws closed further: the reader moved in the grasp under the push (about 6-9 N).

`full_162427_k4_5` (C): aligned before the press (cycle 18: (-0.15, 0.20, 6.39) mm, 42 s). After the re-press (`align_press_on`), the first post-press check was still on target; it asked a 0.34 mm step, and before the next measurement the hand moved by itself about 2.8 mm (FK tip (-0.14, 2.35, -10.98) -> (-0.72, 4.90, -12.18): 2.5 mm up, 1.2 mm toward the port; vision agrees: up +1.5 then +2.8, gap 6.3 -> 4.8 mm). No contact. Three correcting steps brought it back to (-0.35, 0.45, 6.96), within the post-press tolerance, but the post-press budget (`post_press_cycles` 6, three confirms in a row) ran out one confirm short: `step limit`. Back-off, reader lifted, held by the operator. No insertion.

`full_163515_k4_5` (C, rerun): approach 1.97 mm; hand-over gap 6.69, lateral (0.02, 0.30) mm; prior = hand-over; K4 seated at 1.54 s, depth 8.35 mm; push advance 2.62 mm, not stopped, left hand 0.07 mm, right jaw change 0.009. **Operator: flush and seated.**

**Left wrist camera failed (16:48).** `full_164756_k4_6` stopped in the capture stage: `wrist_left camera frame is 48774 ms old` (no motion beyond the capture pose; arms then homed). `k4_5` (ended ~16:42) logged 280 left frames; the camera dropped off USB at 16:48:05, about 9 s after the k4_6 start, when the runner opened the cameras and before any arm motion. Since then it is not on any USB port: its port 5-2 reads "not attached" (no overcurrent, not disabled); replugging at the PC, another PC port, a new USB-C cable, and the right camera's known-good cable on port 7-2 all gave no device. Conclusion: the camera itself is dead. The left wrist camera is logging only (`--no-left-wrist` exists); a replacement needs its mount pose re-checked (calibration).
Correction (17:02): the left wrist camera came back on its own port 5-2 (with the right camera unplugged), so it is an intermittent fault (most likely its connector), not a dead camera. After re-plugging the right camera on 7-2, both stream 1280x1024 frames from their own ports; views checked (the right camera sees the mini PC and its ports, the left the bench and the ZED).

`full_165527_k4_6`: ended in transport with no stop event (the process ended; no insertion). `full_170521_k4_6` (C): K4 seated at 1.36 s, depth 7.76 mm; push +2.46 mm, not stopped, left 0.25 mm, right jaw change 0.011.

**C with `bimanual_align5_k4_push15_seat6` (K4 + 15 mm push + seat speed 6 mm/s):** 4 of 4 insertions seated (k4_3, k4_4, k4_5, k4_6); two align safety stops at the hand-over (vision-net 3.54 mm; post-press budget after a 2.8 mm hand jump), both reruns seated. K4 over the day: 5 of 6 insertions seated (k4_2 was in but undetected, fixed by seat6).

## Demo mode and live view (added 2026-11-10, operator request)

- `run_bimanual --demo` = `--yes` + `--stream`: the routine prompts answer themselves; it still asks after a stop (recovery menu) and before opening the jaws on a lifted reader (`ch.ask(..., confirm=True)`), and the arms-off `SUPPORTED` prompt at the end stays. The outcome prompt is skipped (tell the agent).
- `--stream`: `RealIO` claims each camera from the viewer (`camshare.claim`) and the camera threads publish frames (`Claim.publish`, 5 Hz, 640 px); `deploy/bimanual/live.py` writes `/dev/shm/yam_cams/run_state.json` from `RealChain.log_event` (stage, done/failed nodes, status line, outcome).
- `scripts/real/camview.py` draws the GaP stage graph (capture, pick, hold, transport, approach, align, insert, release, home; done green, current orange, failed red; the policy of each node). Checked: replay of `k4_6` events through the live state; camera publish test (wrist_right 48 frames in 1.5 s, published frame 0.02 s old, claim released); viewer start; demo dry run (seed 7041 completed; push skipped by the lateral gate at 2.53 mm).
- `scripts/real/demo_record.py` (tmux `demorec`): records every streamed run to `data/real_runs/<date>/<run>/live_demo.mp4` (ZED + both wrists + the GaP graph, 1092x720, 10 fps, real time), from the published frames and run_state.json; a separate process. Tested on a replay of `k4_6`.
