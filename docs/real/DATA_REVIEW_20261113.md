# Data review of the real runs of 2026-11-10 and 2026-11-11 (task 0)

Task 0 of [SERVER_TASKS_20261112.md](SERVER_TASKS_20261112.md). No motion, no sim: logs only.
Tool: `scripts/real/review_runs.py` -> `results/real_runs_review/runs_20261110_11.csv` / `.json` (one row per run, about 150 columns),
`steps_20261110_11.json` (every align step and measurement), `align_time_budget.png`, `push_advance.png`.
Session documents: [SESSION_20261110_duchamp3.md](SESSION_20261110_duchamp3.md), [SESSION_20261111_duchamp3.md](SESSION_20261111_duchamp3.md).

## Data on mason

- 27 `full_*` run folders (dry runs excluded): 19 from 2026-11-10, 8 from 2026-11-11.
- **19 runs have logs.** 11 from 2026-11-10 (14:30 to 16:03) and all 8 from 2026-11-11.
- **7 runs of 2026-11-10 (16:10 to 17:05) are empty folders on mason** (`chunks/` and `frames/` with no files): `k4_4` rerun, `k4_5`, `k4_5` rerun, `k4_6` (four starts).
  Their rows come from the session document only (source = doc). Three of the ten milestone seats are in this group. Copy these folders again.
- 1 folder (`full_143002_a4p_1`) has an empty event file (start aborted).

All distances below are encoder FK in the believed frame, port axes (across, up, into the port), in mm. "Tip" is the plug tip from the mean real grasp.
Forces are estimates: joint PD torque `kp x (target - measured)` through the Jacobian. Joint friction (about 0.3 to 1.5 N m) and the gravity model are not removed, so the error is several N.

## Table 1: stage times (s, wall clock) and outcome

| Run | Config (`bimanual_`) | Speed | Source | Capture | Pick | Hold | Transport | Approach | Align | Insert | Release | Home | Total | Pick | Outcome |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 a4p_1 (14:30:02) | - | - | - | - | - | - | - | - | - | - | - | - | - | - | no events |
| 10 a4p_1 | align5_precision | 0.50 | logs | 54.8 | 24.9 | 9.2 | 9.2 | 19.7 | 42.4 | - | 11.6 | 42.7 | 214.4 | held | align only (OK) |
| 10 servo_1 (14:40) | align6_servo | 0.50 | logs | 56.2 | 39.7 | 9.2 | 9.2 | 19.7 | 10.4 | - | 3.5 | 42.0 | 189.8 | held | software stop at the align start |
| 10 servo_1 (14:46) | align6_servo | 0.50 | logs | 55.3 | 39.6 | 9.2 | 9.2 | 19.7 | 13.8 | - | 14.2 | 42.7 | 203.7 | held | align only (OK) |
| 10 servo_2 | align6_servo | 0.50 | logs | 55.8 | 23.7 | 9.2 | 9.2 | 19.7 | 8.9 | - | 10.7 | 42.3 | 179.4 | held | align only (OK) |
| 10 kjrc_1 | align4_kjrc | 0.50 | logs | 54.6 | 29.9 | 9.2 | 9.0 | 19.8 | 69.5 | 2.4 | 15.0 | 42.5 | 251.9 | held | seated |
| 10 kjrc_2 | align4_kjrc | 0.50 | logs | 54.5 | 27.9 | 9.2 | 9.0 | 19.8 | 65.8 | 2.0 | 13.1 | 41.7 | 242.8 | held | seated |
| 10 kjrc_3_push | align4_kjrc_push | 0.50 | logs | 54.7 | 45.0 | 9.3 | 9.0 | 19.7 | 59.6 | 4.9 | 19.7 | 42.0 | 263.9 | held | seated |
| 10 k4_1 | align5_k4_push | 0.50 | logs | 54.4 | 29.1 | 9.2 | 9.0 | 19.8 | 59.7 | 5.0 | 13.5 | 41.2 | 241.0 | held | seated |
| 10 k4_2 | align5_k4_push15 | 0.50 | logs | 54.5 | 37.2 | 9.2 | 9.0 | 19.7 | 59.8 | 9.0 | 11.5 | 41.3 | 251.3 | held | guard_bound (plug was in, not detected) |
| 10 k4_3 | align5_k4_push15_seat6 | 0.50 | logs | 54.0 | 51.6 | 9.3 | 9.0 | 19.7 | 59.7 | 10.2 | 12.7 | 41.7 | 268.1 | held | seated |
| 10 k4_4 (16:03) | align5_k4_push15_seat6 | 0.50 | logs | 55.9 | 44.7 | 9.2 | 9.0 | 19.7 | 57.9 | - | 11.6 | 41.9 | 249.8 | held | align stop (vision-net 3.54 mm) |
| 10 k4_4 (16:10) | align5_k4_push15_seat6 | 0.50 | doc | - | - | - | - | - | - | - | - | - | - | held | seated (push stopped by the jaw watch) |
| 10 k4_5 (16:24) | align5_k4_push15_seat6 | 0.50 | doc | - | - | - | - | - | - | - | - | - | - | held | align stop (post-press budget) |
| 10 k4_5 (16:35) | align5_k4_push15_seat6 | 0.50 | doc | - | - | - | - | - | - | - | - | - | - | held | seated |
| 10 k4_6 (16:44) | - | - | doc | - | - | - | - | - | - | - | - | - | - | - | not in the session document |
| 10 k4_6 (16:47) | align5_k4_push15_seat6 | 0.50 | doc | - | - | - | - | - | - | - | - | - | - | - | capture stop (left wrist camera) |
| 10 k4_6 (16:55) | align5_k4_push15_seat6 | 0.50 | doc | - | - | - | - | - | - | - | - | - | - | - | process end in transport |
| 10 k4_6 (17:05) | align5_k4_push15_seat6 | 0.50 | doc | - | - | - | - | - | - | - | - | - | - | held | seated |
| 11 demo_1 | align5_k4_push15_seat6 | 0.50 | logs | 54.9 | 20.1 | 9.1 | 9.0 | 19.8 | 60.2 | 1.6 | 8.9 | 42.6 | 226.3 | held | seated (push skipped) |
| 11 demo_2 | align5_k4_push15_seat6 | 0.50 | logs | 53.6 | 19.7 | 9.1 | 9.0 | 19.8 | 66.9 | 9.9 | 8.9 | 42.6 | 239.6 | held | seated |
| 11 demo_3 (16:08) | align5_k4_push15_seat6 | 0.50 | logs | 55.3 | 40.9 | - | - | - | - | - | - | 28.2 | 124.5 | closed_empty | pick failed |
| 11 demo_3 (16:12) | align5_k4_push15_seat6 | 0.50 | logs | 53.1 | 19.6 | 9.1 | 9.0 | 19.9 | 58.0 | 9.6 | 8.9 | 42.2 | 229.4 | held | seated |
| 11 demo_4 | align5_k4_push15_seat6 | 0.50 | logs | 53.0 | 18.3 | 9.1 | 9.0 | 19.7 | 43.2 | 9.8 | 8.9 | 42.8 | 213.7 | held | seated |
| 11 demo_5 | align5_k4_push15_seat6 | 0.50 | logs | 52.8 | 20.4 | 9.1 | 9.0 | 19.8 | 53.2 | - | 8.8 | 42.0 | 215.3 | held | align stop (vision-net 3.51 mm) |
| 11 speed075_1 | align5_k4_push15_seat6 | 0.75 | logs | 53.5 | 20.1 | 8.1 | 6.0 | 12.8 | 48.6 | 4.1 | 6.2 | 42.5 | 202.1 | held | seated (push skipped) |
| 11 fastmove_1 | ..._seat6_fastmove | 0.75 | logs | 24.1 | 20.9 | 8.1 | 6.0 | 12.8 | 61.6 | 7.5 | 6.2 | 17.0 | 164.3 | held | seated |

Notes: "Capture" is the two slow moves and the ZED estimate. "Pick" starts at its prompt; on 2026-11-10 it includes the operator's answer time.
"Align" is the wall time of the whole stage (entry to the insertion start), not the stage's own tick count (see finding 3). "Insert" includes the final push.

## Table 2: approach jump and alignment (runs with logs)

| Run | Jump mm | Jump at (s before the end) | Joint | Release mrad | Jump up mm | Joint 2 error at the end mrad | Joint 2 "error" by the tool mrad | First gap | First across | First up | Mode | Cycles | Align wall s | Align tick s | Steps | Timeouts | Timeout s | Probe step s | Frame s | Other wall s | Late ticks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 a4p_1 | 4.14 | 0.5 | 3 | -9.7 | 2.45 | 17.4 | 33.2 | 17.8 | -0.01 | -0.03 | step, controller | 21 | 42.4 | 29.5 | 17 | 1 | 3.0 | 7.0 | 9.3 | 14.4 | 163 |
| 10 servo_1 (14:40) | 0.43 | 2.5 | 2 | 1.5 | 0.37 | -12.2 | 7.2 | - | - | - | servo (not started) | - | 10.4 | - | - | - | - | - | - | - | 14 |
| 10 servo_1 (14:46) | 0.97 | 4.1 | 2 | 1.6 | 0.83 | -20.5 | 14.6 | - | - | - | servo | 183 | 13.8 | 7.8 | - | - | - | - | - | - | 15 |
| 10 servo_2 | 1.47 | 0.9 | 3 | -3.8 | 0.72 | -20.0 | 8.7 | - | - | - | servo | 60 | 8.9 | 2.9 | - | - | - | - | - | - | 16 |
| 10 kjrc_1 | 1.86 | 5.2 | 1 | -4.2 | 0.18 | -24.7 | 12.4 | 18.3 | -0.39 | -1.04 | step | 21 | 69.5 | 54.9 | 16 | 11 | 33.0 | 15.0 | 11.2 | 15.3 | 195 |
| 10 kjrc_2 | 4.36 | 3.1 | 2 | -11.0 | -4.20 | 8.4 | 28.0 | 17.1 | -0.48 | -0.07 | step | 20 | 65.8 | 54.8 | 15 | 10 | 30.0 | 15.9 | 9.6 | 13.9 | 204 |
| 10 kjrc_3_push | 1.90 | 2.4 | 3 | -3.2 | 1.41 | -8.8 | 5.6 | 16.8 | 0.10 | -0.90 | step | 16 | 59.6 | 47.2 | 15 | 10 | 30.0 | 16.0 | 8.4 | 12.2 | 154 |
| 10 k4_1 | 1.69 | 0.9 | 3 | 5.3 | -0.57 | -25.0 | 16.2 | 17.8 | 2.05 | -2.75 | step | 22 | 59.7 | 46.9 | 17 | 8 | 24.0 | 15.3 | 10.8 | 13.6 | 194 |
| 10 k4_2 | 0.64 | 5.5 | 1 | 1.1 | -0.20 | -24.7 | 14.6 | 17.2 | 1.92 | -1.65 | step | 19 | 59.8 | 45.9 | 16 | 9 | 27.0 | 14.2 | 9.4 | 13.0 | 175 |
| 10 k4_3 | 3.58 | 0.8 | 3 | 7.8 | -2.53 | -31.8 | 19.5 | 18.8 | 2.44 | -4.23 | step | 17 | 59.7 | 49.0 | 15 | 10 | 30.0 | 16.6 | 8.7 | 12.2 | 155 |
| 10 k4_4 (16:03) | 5.46 | 5.5 | 2 | -13.5 | -5.21 | -17.2 | 31.4 | 17.7 | -1.97 | -2.97 | step | 17 | 57.9 | 38.8 | 14 | 8 | 24.0 | 11.1 | 9.6 | 15.7 | 185 |
| 11 demo_1 | 4.97 | 5.2 | 2 | 12.3 | 4.74 | -8.8 | 27.6 | 17.5 | 1.80 | 4.25 | step | 17 | 60.2 | 48.9 | 14 | 10 | 30.0 | 15.5 | 8.9 | 12.5 | 142 |
| 11 demo_2 | 3.88 | 3.7 | 3 | -6.7 | 2.03 | -19.8 | 8.5 | 18.4 | 1.37 | -4.25 | step | 21 | 66.9 | 53.9 | 17 | 12 | 36.0 | 17.2 | 10.5 | 13.5 | 175 |
| 11 demo_3 (16:12) | 2.56 | 0.7 | 3 | -4.4 | 1.98 | -9.3 | 3.4 | 18.1 | -1.16 | -0.37 | step | 17 | 58.0 | 47.1 | 15 | 9 | 27.0 | 12.2 | 9.0 | 12.7 | 177 |
| 11 demo_4 | 2.53 | 0.5 | 3 | -6.1 | 1.80 | -17.0 | 12.9 | 17.1 | 0.65 | -0.76 | step | 14 | 43.2 | 33.4 | 12 | 6 | 18.0 | 10.0 | 7.5 | 10.1 | 135 |
| 11 demo_5 | 1.00 | 4.0 | 2 | 3.8 | 0.91 | -10.3 | 11.2 | 17.6 | 0.28 | -1.23 | step | 17 | 53.2 | 41.7 | 14 | 5 | 15.0 | 13.8 | 7.7 | 13.7 | 128 |
| 11 speed075_1 | 1.67 | 4.7 | 2 | 6.0 | 1.66 | 4.0 | 16.7 | 18.1 | 0.22 | -1.10 | step | 16 | 48.6 | 35.7 | 15 | 4 | 12.0 | 12.1 | 8.9 | 12.1 | 172 |
| 11 fastmove_1 | 5.00 | 3.3 | 2 | -12.1 | -4.73 | -25.2 | 29.7 | 18.8 | 3.87 | -5.25 | step | 19 | 61.6 | 49.8 | 17 | 10 | 30.0 | 15.1 | 9.6 | 13.7 | 158 |

Notes: "Jump" = `scripts/real/approach_jump.py` (largest uncommanded tip move in 0.5 s, last 6 s of the approach, limit 3 mm). "Joint" and "Release" = the joint (1 to 3) whose tracking error changed most in that 0.5 s.
"First" = the first align measurement. "Timeout s", "Probe step s" and "Frame s" are parts of the stage; "Other wall s" is the rest (settle waits, press off and on, in-hand refresh, late ticks).

## Table 3: hand-over and insertion policy

| Run | Gap | Across | Up | Net - vision up | Policy | Seat s | Deep (4.8 mm) s | Seat depth | FK advance at the policy end | Seat up | Lead up | Lead in | F in N | F up N |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 a4p_1 | - | - | - | 0.34 | - | - | - | - | - | - | - | - | - | - |
| 10 kjrc_1 | 5.58 | 0.06 | 0.60 | 1.90 | K-J re-centred | 2.22 | 1.10 | 6.59 | 12.06 | 1.83 | -2.4 | 6.7 | 5.7 | -6.6 |
| 10 kjrc_2 | 6.06 | 0.74 | 0.40 | 2.56 | K-J re-centred | 1.76 | 1.10 | 6.34 | 12.43 | 1.44 | 21.1 | 10.1 | 1.2 | 5.7 |
| 10 kjrc_3_push | 5.57 | 0.39 | -0.39 | 3.03 | K-J re-centred | 1.40 | 0.94 | 5.99 | 11.42 | 0.01 | 16.4 | 6.7 | 0.2 | 1.0 |
| 10 k4_1 | 5.69 | -0.19 | 0.76 | 2.40 | K4 | 1.52 | 0.70 | 8.02 | 13.59 | 0.34 | 30.3 | 20.4 | 4.8 | 11.3 |
| 10 k4_2 | 6.01 | -0.12 | -0.30 | 2.81 | K4 | - | 0.82 | - | 13.63 | -8.43 | 0.5 | 12.0 | 3.5 | -3.2 |
| 10 k4_3 | 5.92 | 0.08 | 0.20 | 2.56 | K4 | 1.68 | 0.66 | 8.47 | 14.35 | -0.80 | 27.0 | 20.4 | 5.2 | 9.4 |
| 10 k4_4 (16:03) | - | - | - | 3.43 | - | - | - | - | - | - | - | - | - | - |
| 10 k4_4 (16:10, doc) | 6.50 | -0.17 | 0.78 | 2.61 | K4 | 1.44 | - | 7.76 | - | - | - | - | - | - |
| 10 k4_5 (16:35, doc) | 6.69 | 0.02 | 0.30 | - | K4 | 1.54 | - | 8.35 | - | - | - | - | - | - |
| 10 k4_6 (17:05, doc) | - | - | - | - | K4 | 1.36 | - | 7.76 | - | - | - | - | - | - |
| 11 demo_1 | 6.17 | 0.09 | -0.03 | -0.19 | K4 | 1.42 | 0.90 | 8.23 | 14.41 | -3.50 | 10.4 | 14.3 | 6.4 | -2.0 |
| 11 demo_2 | 6.26 | -0.46 | -0.14 | 3.40 | K4 | 1.40 | 0.86 | 7.47 | 13.62 | 0.28 | 32.1 | 19.3 | 4.7 | 8.9 |
| 11 demo_3 (16:12) | 6.55 | -0.50 | -0.08 | 3.20 | K4 | 1.12 | 0.66 | 7.36 | 13.88 | 1.48 | 31.8 | 19.0 | 6.2 | 11.1 |
| 11 demo_4 | 6.62 | 0.46 | -0.03 | 2.63 | K4 | 1.30 | 0.76 | 8.38 | 15.18 | -0.65 | 27.6 | 18.3 | 7.6 | 7.6 |
| 11 demo_5 | - | - | - | 3.28 | - | - | - | - | - | - | - | - | - | - |
| 11 speed075_1 | 6.93 | -0.27 | 0.17 | 3.07 | K4 | 3.88 | 3.40 | 6.77 | 13.68 | -3.33 | 8.9 | 13.3 | 5.7 | -2.4 |
| 11 fastmove_1 | 6.01 | 0.19 | -0.20 | 2.32 | K4 | 1.46 | 0.70 | 8.60 | 14.49 | 0.75 | 29.5 | 22.8 | 8.5 | 10.1 |

Notes: "Gap, Across, Up" = the hand-over measurement (target 6, 0, 0). "Net - vision up" = wrist net r10f minus port vision at the last measurement that has both.
"Deep" = first tick with seat-2 depth >= 4.8 mm. "Seat depth" = FK advance minus the vision gap at the detected seat. "Seat up" = seat-2 lateral up at the last policy tick.
"Lead" = policy target minus measured tip at the policy end. "F" = PD force estimate at the policy end (positive in = into the port).

## Table 4: final push

| Run | Push | Target mm | Seat depth | Logged push advance | Seat + push | FK advance after the lead drop | FK advance at the push peak | Gain over the policy end | Push F in N | Tip up at the push peak | Tip up at the end | Peak FK depth | Jaw change | Left hand mm | Operator: fully in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 kjrc_1 | not in the config | - | 6.59 | - | 6.59 | - | - | - | - | - | 0.0 | 6.48 | - | - | no (hangs) |
| 10 kjrc_2 | not in the config | - | 6.34 | - | 6.34 | - | - | - | - | - | 0.1 | 6.37 | - | - | no (about 1 mm short) |
| 10 kjrc_3_push | done | 5 | 5.99 | 0.93 | 6.92 | 10.67 | 11.49 | +0.07 | 4.5 | -2.9 | -3.2 | 5.92 | - | - | no (about 3 mm short) |
| 10 k4_1 | done | 5 | 8.02 | 0.38 | 8.40 | 11.90 | 12.21 | -1.39 | 4.2 | -4.5 | -5.2 | 7.90 | - | - | no (about 3 mm short) |
| 10 k4_2 | no seat detected | - | - | - | - | - | - | - | - | - | - | 7.62 | - | - | pulled out by the guard |
| 10 k4_3 | done | 15 | 8.47 | 2.17 | 10.64 | 12.49 | 14.51 | +0.16 | 9.0 | -9.0 | -9.6 | 8.59 | 0.0098 | 0.12 | not stated ("worked really well") |
| 10 k4_4 (16:10, doc) | stopped (jaw watch) | 15 | 7.76 | 2.74 | 10.50 | - | - | - | - | - | - | - | 0.0200 | 0.15 | not stated |
| 10 k4_5 (16:35, doc) | done | 15 | 8.35 | 2.62 | 10.97 | - | - | - | - | - | - | - | 0.0090 | 0.07 | yes (flush) |
| 10 k4_6 (17:05, doc) | done | 15 | 7.76 | 2.46 | 10.22 | - | - | - | - | - | - | - | 0.0110 | 0.25 | not stated |
| 11 demo_1 | skipped (seat up -3.50) | - | 8.23 | - | 8.23 | - | - | - | - | - | -4.4 | 8.24 | - | - | no (hanging slightly) |
| 11 demo_2 | done | 15 | 7.47 | 2.35 | 9.82 | 12.08 | 14.38 | +0.76 | 9.6 | -6.8 | -7.5 | 8.12 | 0.0006 | 0.07 | not stated |
| 11 demo_3 (16:12) | done | 15 | 7.36 | 2.39 | 9.75 | 12.17 | 14.35 | +0.47 | 9.7 | -6.8 | -7.4 | 7.80 | 0.0095 | 0.13 | not stated |
| 11 demo_4 | done | 15 | 8.38 | 2.09 | 10.48 | 13.60 | 15.31 | +0.12 | 10.7 | -7.3 | -7.5 | 8.69 | 0.0006 | 0.16 | yes |
| 11 speed075_1 | skipped (seat up -3.33) | - | 6.77 | - | 6.77 | - | - | - | - | - | -4.5 | 6.75 | - | - | not stated ("success") |
| 11 fastmove_1 | done | 15 | 8.60 | 2.04 | 10.64 | 12.55 | 14.41 | -0.08 | 9.4 | -7.4 | -8.0 | 8.48 | 0.0006 | 0.10 | yes |

Notes: "FK advance" = plug-tip advance into the port since the hand-over (it includes the hand-over gap of about 6 mm). "Peak FK depth" = the largest FK advance (policy end or push peak) minus the vision gap.
"Tip up" is relative to the hand-over height. Figure: `results/real_runs_review/push_advance.png`.

## What the data says

### Final push and depth

1. **The 15 mm push ends where the policy already was.** At its last tick, K4 has its target 18 to 23 mm into the port and 27 to 32 mm above the tip, and it pushes with about 5 to 8 N in and 8 to 11 N up (estimate, n = 6 K4 seats with a push).
   When the runner drops this lead (the 0.2 s hold), the hand springs back out by 1.5 to 1.9 mm and the tip drops 1.6 to 3.1 mm.
   The 15 mm push then moves the hand in again by 2.0 to 2.4 mm (the logged "advance"). The peak is only -0.08 to +0.76 mm past the point where K4 stopped (n = 5; median +0.16 mm).
   The push target leads by 12.7 to 13.3 mm at the peak: about 9 to 11 N into the port. After the push is dropped, the hand is 1.2 to 1.5 mm back from the peak.
2. **The 5 mm push was weaker than the policy's own push.** About 4.2 to 4.5 N against K4's 4.8 N and its 20 mm lead; on `k4_1` the push peak was 1.39 mm behind the policy end (n = 2).
3. **So the "push advance" number mostly measures the spring-back of the arm and the grasp, not plug travel.** With about 5 to 11 N the FK advance stops at 13.6 to 15.3 mm from the hand-over on every K4 run (n = 9), at every force. Either the plug is at a hard stop, or the remaining travel needs more than about 11 N. The logs cannot tell which: there is no measurement of the true depth.
4. **The push makes the hand go down.** At the push peak the plug tip is 6.8 to 9.0 mm below the hand-over line by FK (15 mm pushes, n = 5); it was -1.3 to +1.5 mm at the K4 seat. The two K-J runs with no push end at 0.0 and 0.1 mm.
   The real plug is in the port and cannot go down 7 mm. So the reader turns in the jaws, or the plug is levered in the port. This is a side load on the port. The push line is straight in the believed frame; the estimated push force has a down part of 2 to 3 N.
   The jaw reading changed by 0.0095 to 0.0098 on 2 of 5 pushes with logs (`k4_3`, `demo_3`) and by 0.020 on `k4_4` (doc): the reader moved in the grasp.
5. **The rule "seat depth + push advance >= 10.4 mm = fully in" cannot be confirmed.** It separates the labelled runs (fully in: 10.48, 10.64, 10.97, n = 3; short: 6.34 to 8.40, n = 5). But every "short" label is a run with no 15 mm push, and the push advance is nearly constant (2.0 to 2.7 mm) when the push runs. So the rule is the same as "the 15 mm push ran".
   Five seats have no depth statement (9.75 to 10.64). A depth that does not use the relaxed pose, "peak FK depth", gives fully in 8.48 and 8.69 (n = 2 with logs), short 5.92 to 8.24 (n = 5), not stated 6.75 to 8.59: no clear limit (8.24 "hanging slightly" against 8.48 "perfect").
   To settle it, the operator must record a depth for every run (for example the neck-to-panel distance with a caliper, or from the side camera).
6. **K4 goes deeper than K-J before any push.** Seat depth 6.77 to 8.60 mm, median 8.1 mm (K4, n = 8 with logs) against 5.99 to 6.59 mm (K-J re-centred, n = 3). FK depth depends on the vision gap at the hand-over (5.57 to 6.93 mm; on 2026-11-11 the median is 6.4 mm, target 6).
7. **Both skipped pushes are runs where K4 did not pull up.** `demo_1` and `speed075_1`: lead up 10.4 and 8.9 mm (other K4 seats 27 to 32 mm), force up -2 N (others +8 to +11 N), tip 3.5 and 3.3 mm below the hand-over line at the seat (others -1.3 to +1.5 mm). The gate (2.5 mm) then stopped the push.
   The seat lateral is FK motion since the hand-over, not a wrist-net value. It does not correlate with the wrist-net bias (Spearman 0.04, n = 12). So the skipped pushes are not caused by the wrist net. `k4_2` (not detected, 8 mm down) is the same pattern, stronger.

### Insertion policy

8. **K4 has the plug at depth in under 1 s.** Seat-2 depth >= 4.8 mm at 0.66 to 0.90 s on 8 of 9 K4 insertions with logs (`speed075_1`: 3.40 s). The seat detection comes 0.46 to 1.02 s later (15 ticks in a row under the speed limit).
9. **The control loop is on time in the insertion.** Compute time p50 4.1 to 5.0 ms, 1 or 2 late ticks per insertion (n = 12). With `--stream` the p50 is 4.5 to 5.0 ms (n = 6) against 4.1 to 4.5 ms without (n = 6). One tick of 41 ms on `demo_1`.

### Alignment

10. **The stage takes 43 to 70 s of wall time, not 34 to 55 s.** Median 59.7 s (n = 14 step-mode runs on plain PD). The 34 to 55 s in the task list is the stage's own tick count; the control loop runs late during the stage (finding 13). The target "under 15 s" must be wall time.
11. **Where the wall time goes (n = 14 runs, 824 s in total):** steps 62 % (steps that timed out 44 %, steps that reached 18 %), camera frames 16 %, the rest 22 % (10 to 16 s per run: settle waits, press off and on, in-hand refresh, 3 to 5 post-press checks of 2.5 to 7.2 s, late ticks). Probe steps are 24 % of the stage (10 to 17 s per run). Figure: `results/real_runs_review/align_time_budget.png`.
12. **58 % of the steps time out (122 of 212), and the hand is at or past the goal when they do.** Median moved / asked = 1.09; 61 % of the timed-out steps moved more than asked; the median distance to the goal is 0.16 mm past it. Each timeout costs 3.0 s, and the number of timeouts sets the align time (Spearman 0.93, n = 15).
    Timeouts by kind (a step is named by its largest part): 1 mm steps toward the port 51 of 87; probes 38 of 84; across 18 of 22; up 15 of 18. The steps that reach take 1.8 s (median 89 ticks).
    With the precision controller (`a4p_1`), 1 of 17 steps timed out and the median step took 1.0 s. So the plain-PD hand does not settle inside the reach window (0.3 mm): it stops past the goal and stays there.
13. **The control loop stalls during every measurement.** In all 15 step-mode runs the align stage has 128 to 204 late ticks and a compute time p99 of 92 to 98 ms. It is the same with and without `--stream` (2026-11-10 had no stream). The port-vision call runs in the control thread: 5 frames at about 85 ms each (425 ms per measurement, n = 298).
    The servo runs used a separate vision process: 35 ms per frame, 15 or 16 late ticks. The session note that put the late ticks on the streaming is not correct.
14. **The probe slopes are the same on every run.** Across 0.94 to 1.10, up 0.76 to 1.01, gap 1.02 to 1.12 (n = 17 with a4p and the servo runs). The six probe steps cost 10 to 17 s per run.
15. **The hand-over accuracy is good.** |across| <= 0.74 mm, |up| <= 0.78 mm, gap 5.57 to 6.93 mm (n = 14 hand-overs, 2 from the doc).

### Approach jump

16. **The jump is over 3 mm on 7 of 18 approaches with logs.** `a4p_1` 4.14, `kjrc_2` 4.36, `k4_3` 3.58, `k4_4` (16:03) 5.46, `demo_1` 4.97, `demo_2` 3.88, `fastmove_1` 5.00 mm. The `k4_4` value is not in the session document.
17. **It is a stick-slip release of joint 2 or joint 3.** In the four largest jumps (4.4 to 5.5 mm), joint 2 lets go of 11 to 13.5 mrad of tracking error within 0.5 s, 3.1 to 5.5 s before the approach ends, and the tip moves 4.2 to 5.2 mm up or down. 11 to 13.5 mrad at kp 80 is 0.9 to 1.1 N m, the size of the measured joint friction. In 8 other runs the largest change is on joint 3: 3 to 10 mrad, 0.5 to 3.7 s before the end (5 of them in the last 0.9 s).
    Before a release, the joint 2 tracking error is 15 to 40 mrad. It is not the align entry: the wind-up at the align entry is 10 to 32 mrad on every run, jump or no jump.
18. **The approach tool's joint error has a constant offset.** `ticks.npz` `q` is the encoder frame and `target_b` is the believed frame. Their difference holds the joint offset: +12.7 mrad on joint 2, +23.2 on joint 4, -19.5 on joint 5 (with the hand held at its measured pose, the logged "error" is exactly these values).
    So "joint 2 error 33.2 mrad (limit 25)" on `a4p_1` is a tracking error of 20.5 mrad, and the sign matters: the tracking error of joint 2 at the approach end is -32 to +17 mrad. The jump in mm is not affected (it is a difference). The 25 mrad limit needs a new definition on the tracking error.
19. **The jump direction shows in the first align measurement.** Jump up against the first measured up: Spearman 0.57, p = 0.03, n = 15. The jump size against the align time: Spearman 0.43, p = 0.09 (not shown). No relation with the speed (0.75: 1.67 and 5.00 mm).
20. The largest uncommanded move of the whole approach is 6 to 12 mm, 11 to 19 s before the end (the first approach steps). The tool looks only at the last 6 s.

### Wrist net against vision

21. **The "up" difference is +1.9 to +3.4 mm on 13 of 15 runs (median +2.8).** Across -0.36 +/- 0.40 mm, gap +0.73 +/- 0.37 mm (mean +/- SD, n = 15). The up difference grows as the gap closes: median +1.8 mm at a gap over 12 mm, +2.6 mm at a gap under 8 mm.
    With the 3.5 mm gate the margin is small; 2 runs stopped (3.54 and 3.51 mm at the hand-over check).
22. **The two runs with no difference are different runs, not noise.** `demo_1` (-0.19 mm): its in-hand estimate (`align_inhand`) is near zero on all six values (after: 0.4, -0.3, 1.1, 0.4, 0.0, -0.2); on the 11 other runs with the event it is about (4.7, 0.3, -1.8, -1.3, 0.4, 1.6). `a4p_1` (+0.34 mm) is the precision-controller run and has no in-hand event.
    So the difference follows the in-hand state. It may be how the two sources treat the real grasp, not a fixed error of the net. A constant subtraction of 2.6 mm would leave `demo_1` and `a4p_1` at -2.8 and -2.3 mm (inside the gate, but with little margin). `demo_1` is also one of the two runs with a skipped push.

### Pick, hold, other stages

23. Pick: 18 held of 19 picks with logs. In demo mode the pick takes 18.3 to 20.9 s (n = 7). The one empty close (`demo_3`, 16:08) is in the session document; nothing new here.
24. The hold latched on every run; the estimated press force is 18 to 22 N for a 25 N command (n = 18).
25. Fixed times at speed 0.5: hold 9.1 to 9.3 s, transport 9.0 to 9.2 s, approach 19.7 to 19.9 s, release 8.8 to 8.9 s in demo mode. At 0.75: 8.1, 6.0, 12.8, 6.2 s. Capture and home: 53 to 56 s and 41 to 43 s at 0.15 rad/s; 24.1 and 17.0 s at 0.4 rad/s (n = 1).
    After `fastmove_1` (164.3 s), the align stage is 37 % of the run and the capture + home moves 25 %.

## What the data cannot say

- The true plug depth and the force that reaches full depth. All depths are FK with a vision gap; all forces are PD estimates with friction not removed.
- Whether the 7 mm drop of the tip during the push is the reader turning in the jaws or the plug bending in the port. The side camera clips may show it.
- Rates. One PC spot, one reader spot, 12 insertions with logs. The numbers above are ranges, not success rates.
- Anything on the three seats and four other starts of 2026-11-10 after 16:10 beyond the session document (no logs on mason).
- The cause of the wrist-net difference. The in-hand link is one run (`demo_1`).

## For the next tasks

- Task 1 (push): a deeper position target does not add depth (finding 1 to 3). Test a higher force along the port axis with no side load, and keep K4's up-pull out of the push. Ask for a true depth measurement per run.
- Task 2 (align): the reach test (finding 12), the measurement in the control thread (13) and the fixed probes (14) are 44 %, 16 % and 24 % of the stage.
- Task 3 (approach jump, wrist net): joint 2 stick-slip (17), the tool's offset (18), the in-hand link (22).
