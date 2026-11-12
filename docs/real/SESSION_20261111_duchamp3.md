# Session 2026-11-11, duchamp3: recorded demos

Branch `real/graph-align` (uncommitted work from 2026-11-10, see `SESSION_20261110_duchamp3.md`). Run logs `data/real_runs/2026-11-11/` (not in git).
Config for every demo: `deploy/config/bimanual_align5_k4_push15_seat6.json`, `--speed 0.5 --align-mode full --demo`, same PC spot and reader spot as 2026-11-10.
Each run's `live_demo.mp4` (ZED + both wrists + GaP stage graph) is written by `scripts/real/demo_record.py` (tmux `demorec`).

## Runs

| Time | Run | Operator | Logs |
|---|---|---|---|
| 15:53 | `full_155351_demo_1` | seated, reader hanging slightly (push skipped) | Pick OK, hold latched. **Approach: uncommanded move 4.97 mm (limit 3), joint 2 error 27.6 mrad (limit 25).** Step align 17 cycles, 48.9 s; hand-over gap 6.17 mm, lateral (0.09, -0.03) mm; prior = hand-over. K4 `seated_detected` at 1.42 s, depth 8.23 mm. **Final push skipped: seat lateral (-0.58, -3.50) mm, up beyond the 2.5 mm gate** (2026-11-10 pushes ran at lateral <= 1.83 mm). Loop timing with streaming as on 2026-11-10 (compute p99 89 ms, 173 late ticks; k4_6: 88 ms, 201). Video 230 s, saved. |
| 16:03 | `full_160310_demo_2` | seated | Pick OK, hold latched. **Approach: uncommanded move 3.88 mm (limit 3)**, joint 2/3 error 8.5/14.3 mrad. Step align 21 cycles, 53.9 s; hand-over gap 6.26 mm, lateral (-0.46, -0.14) mm. K4 `seated_detected` at 1.40 s, depth 7.47 mm. Final push +2.35 mm, not stopped, left hand 0.07 mm, right jaw change 0.0006. Compute p99 88 ms, 203 late ticks. Video 244 s, saved. |
| 16:08 | `full_160801_demo_3` | failed: unsuccessful grasp | Pick `closed_empty` (1 attempt; GuardStop 'pick failed', home). Reader estimate and pick servo normal (reader (136.9, 452.0) mm, yaw 90.2 deg; last servo delta 1.6, 1.4, 6.9 mm). **The hand settled 7 mm higher than on every held pick: measured z -17.7 mm at the close against -23.1 to -25.1 mm on the 14 held picks of 2026-11-10/11 (same command, -27 mm).** Wrist frames: reader centred between the open fingers, then the fingers closed over its top (the 2026-11-01 empty-close pattern). Squeeze rotation 2.55 deg (held picks 0.57-1.38). Video 128 s, saved. |
| 16:12 | `full_161223_demo_3` (rerun) | seated (success) | Pick held (close z -23.0 mm), hold latched. Approach move 2.56 mm (within 3), joint 2/3 error 3.4/16.9 mrad. Step align 17 cycles, 47.1 s; hand-over gap 6.55 mm, lateral (-0.50, -0.08) mm. K4 `seated_detected` at 1.12 s, depth 7.36 mm. Final push +2.39 mm, not stopped, left hand 0.13 mm, right jaw change 0.0095. Compute p99 88 ms, 196 late ticks. Video 233 s, saved. |
| 16:17 | `full_161705_demo_4` | **seated, fully inserted (operator: best demo; use this video and data as the fully successful insertion)** | Pick held (close z -24.5 mm), hold latched. Approach move 2.53 mm (within 3), joint 2/3 error 12.9/16.8 mrad. Step align 14 cycles, **33.4 s** (fastest today); hand-over gap 6.62 mm, lateral (0.46, -0.03) mm. K4 `seated_detected` at 1.30 s, depth 8.38 mm. Final push +2.09 mm, not stopped, left hand 0.16 mm, right jaw change 0.0006. Compute p99 85 ms, 152 late ticks. Video 218 s; video clock: align 1:55-2:38, insert 2:38, seated 2:40, push to 2:48, release 2:48, home 2:57. |
| 16:22 | `full_162246_demo_5` | failed (operator: cause not seen) | Pick held, hold latched, approach move 1.00 mm (best today). Step align 17 cycles, 41.7 s; vision at the end (0.58, -0.39, 6.59) mm = aligned. **Align stop at the hand-over: `sources disagree at the hand-over [0.09, 3.51, 1.03] mm` (limit [3.0, 3.5, 2.5]): the wrist net's up was 0.01 mm over its gate.** Back-off 15 mm, release, home; no insertion. Same stop as `k4_4` (2026-11-10, 3.54 mm). Video 219 s, saved. |

Operator after demo_1: continue the demos with the known approach jump (as on 2026-11-10); 5 demos instead of 10 (time).


Wrist net (r10f) up minus vision up at the end of the align stage, 13 runs 2026-11-10/11: net reads +0.7 to +4.4 mm higher on 12 of 13 (median about +3 mm); the 3.5 mm disagreement gate stopped 2 of 13 hand-overs (k4_4, demo_5), both with vision on target. A constant bias, not noise: candidate fix is to subtract the net's up bias before the check, or widen the up gate (config change, operator approval needed; not changed).
## Speed (operator request after the demos)

`--speed` scales only the scripted motions (pick servo, hold swing, transport, approach ramps, release, back-offs, the final push). Not scaled: capture and home moves (`io.move_to`, fixed 0.15 rad/s, about 95 s per run), the step alignment (creep), K4 (own 50 Hz rate). The full chain had never run above 0.5 on the rig.

| Run | Speed | Operator | Stage times (s): capture moves / pick / hold / transport / approach / align / insert / release / home; total | Notes |
|---|---|---|---|---|
| `full_161705_demo_4` | 0.5 | seated, fully in | 53.0 / 18.2 / 9.1 / 9.0 / 19.7 / 43.2 / 9.8 / 8.9 / 42.8; 213.7 | reference |
| `full_163144_speed075_1` | 0.75 | success | 53.5 / 20.1 / 8.1 / 6.0 / 12.8 / 48.6 / 4.1 / 6.2 / 42.5; 202.1 | Pick held; approach move 1.67 mm (within 3); align 16 cycles, 35.7 s; K4 seated at **3.88 s** (others 1.1-1.7), depth 6.77 mm; **final push skipped: seat lateral (-1.03, -3.33) mm, up beyond 2.5** (as demo_1); insert stage shorter because no push. Peak joint speed: right joint 4 in the pick 2.03 rad/s (0.5: 1.81; guard 2.5); transport 1.34 (0.97). Video 206 s. |

### Capture and home moves (2026-11-11)

demo_4 (`moves_*.npz`): four slow moves, one arm at a time, at a 0.15 rad/s peak (`io.move_to` smoothstep: duration = 1.5 x largest joint change / max_vel): right to capture 18.2 s (+1.9 s settle, joint 1 1.82 rad), left to capture and park 29.3 s (+3.0), left home 18.9 s (+1.3), right home 21.4 s (+1.1, joint 2 2.14 rad). 0.15 was the runner's first default (commit 488dbdf), no recorded reason.
Change (no motion): `chain.move` passes `safety.move_max_vel_rad_s` (default 0.15, unchanged for every existing config). New config `bimanual_align5_k4_push15_seat6_fastmove.json` = seat6 + 0.4 rad/s. Conventions `FAILED: none`; dry run seed 7041 completed (the sim teleports these moves, so it only checks the code path). Expected: moves 88 s -> about 33 s. Operator approved 0.4 rad/s (first run `fastmove_1`).

| Run | Config / speed | Operator | Stage times (s): capture moves / pick / hold / transport / approach / align / insert / release / home; total | Notes |
|---|---|---|---|---|
| `full_171905_fastmove_1` | `bimanual_align5_k4_push15_seat6_fastmove` (moves 0.4 rad/s), `--speed 0.75` | **seated, inserted perfectly (operator: a really good demo; keep as the fast showcase)** | **24.1** / 20.9 / 8.1 / 6.0 / 12.8 / 61.6 / 7.5 / 6.2 / **17.0**; **164.3** (demo_4 at 0.5: 213.7; speed075_1: 202.1) | Capture moves 53 -> 24 s, home 43 -> 17 s. Slow moves: 6.8-10.9 s each + 0.3-3.7 s settle, peak measured joint speed 0.50-0.51 rad/s (0.15 setting: 0.21-0.25), tracking error 44-67 mrad (0.15: 40-49), end error 5-26 mrad (same range as before). Pick held; **approach move 5.0 mm (limit 3), joint 2 error 29.7 mrad (limit 25)** (known approach jump, operator accepted). Step align 19 cycles, 49.8 s (stage 61.6 s, the slowest part now); hand-over gap 6.01, lateral (0.19, -0.20) mm. K4 seated at 1.46 s, depth 8.60 mm. Final push +2.04 mm, not stopped, left 0.10 mm, right jaw change 0.0006. Video 168 s, saved. |
