# Gate replay and depth signals without the operator (W6, 2026-11-14)

Offline experiments 2 and 3 of `/home/yashchitambar/wt/agentic-rl/docs/research/AGENTIC_RL_FORMULA_20261113.md` ("Optimising the graph itself", new ideas 2 and 5).
Logs only, CPU only. **Nothing here ran on the arm. No config or deploy code changed.** Every limit change below is a proposal for the operator.

Tools (new):

- `scripts/real/gate_replay.py` -> `results/real_runs_review/gate_replay.json` (per-gate summary, per-run rows, the gates that cannot be replayed) and `gate_replay.csv` (one row per gate x config x run).
- `scripts/real/depth_signals.py` -> `results/real_runs_review/depth_signals.{json,csv,png}`.

```bash
cd /home/yashchitambar/wt/graph-align
PYTHONPATH=. /home/yashchitambar/peg_test_bimanual/.venv/bin/python scripts/real/gate_replay.py      # about 3 min
PYTHONPATH=. /home/yashchitambar/peg_test_bimanual/.venv/bin/python scripts/real/depth_signals.py    # about 1 min
```

## Data

- `full_*` runs of 2026-11-03, 06, 10 and 11, dry runs excluded: 49 folders. 2026-11-07 has no full runs.
- 37 runs have logs. 4 runs of 2026-11-10 have values in `SESSION_20261110_duchamp3.md` only (folders empty on mason): `k4_4` 16:10 (push jaw watch 0.020), `k4_5` 16:24 (post-press step limit), `k4_5` 16:35, `k4_6` 17:05. 8 folders have no events or no doc values.
- Run labels come from the session documents (in the script, `LAB`). "Fine" is judged per stage: align fine = vision hand-over in tolerance and the reader held; insert fine = the plug was in the port by the operator (also `k4_2`, which the guard then pulled out).
- Two configs: **base** = `bimanual_align5_k4_push15_seat6_fastmove.json` (the milestone), **all9** = `bimanual_align9_all.json`.
- 2026-11-03 and 06 used a 6 mm approach stop. Gates that depend on the stop distance (entry gap) are "not comparable" on those runs and are not counted.

Definitions: **fired** = the limit stops the run (or skips the push) on the logged value. **False stop** = fired on a run whose stage was fine. **Missed stop** = the gate passed, the run shows the failure that this gate is meant to catch, and no other gate of the same stage fired. **Margin** = distance to the limit in the gate's units, positive inside.

## Gate table (base config; all9 where it differs)

Safety = protects the hardware or people. Quality = protects the outcome. n = runs with a logged value.

| Gate | Kind | Why that kind | Limit | n | Good runs min / p50 / max | Fired | False stops | Missed | Nearest good run (margin) | all9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ZED OOD (Mahalanobis) | safety | a wrong PC pose steers the arm toward the case | 21.3 (17.06 x 1.25) | 38 | 16.6 / 17.6 / 19.1 | 0 | 0 | 0 | a4p_1 (2.2) | same |
| ZED presence | quality | no PC, nothing to do | 0.5 | 38 | 1.0 / 1.0 / 1.0 | 0 | 0 | 0 | 0.5 | same |
| Pick held band (gripper reading) | quality | an empty grasp ends the run, no risk | 0.275-0.36 | 36 | 0.303 / 0.313 / 0.328 | 1 (`demo_3` 16:08, 0.426, true) | 0 | 0 | demo_1 (0.028) | same (all9 retries once) |
| Left press time | safety | left joint 2 near its rating | 60 s | 34 | 9.2 / 18.1 / 27.9 (all runs) | 0 | 0 | 0 | 32 s | same |
| Torque peak (0.1 s) | safety | motor rating | 12/12/12/5/5/5 N m | 37 | worst ratio: right joint 4, 3.2 of 5 N m | 0 | 0 | 0 | 1.8 N m | same |
| Torque continuous (20 s) | safety | motor heating | 9/9/9/3/3/3 N m | 37 | at most 0.18 s above | 0 | 0 | 0 | 19.8 s | same |
| Joint speed | safety | runaway or slip | 2.5 rad/s | 37 | 1.37 / 1.82 / 2.19 | 0 | 0 | 0 | servo_1 (0.31) | same |
| Pre-insert lateral | quality | neighbouring port (log only) | none | 34 | 4.0 / 17.9 / 26.5 | - | - | - | - | same |
| Align entry gap | safety | a 7 mm slip into the case (2026-11-03) | >= 12 mm | 15 | 16.8 / 17.8 / 18.8 | 0 | 0 | 0 | kjrc_3 (4.8) | same |
| Align no measurement (in a row) | quality | the stage cannot aim | 5 | 31 | 0 / 0 / 2 | 0 | 0 | 0 | 3 cycles | same |
| Align frame spread | quality | one bad measurement retries | 1.5 mm | 31 | 0.51 / 1.26 / 3.20 (single frames) | 0 | 0 | 0 | never 5 in a row | same |
| **Align sources disagree** | quality | catches a wrong vision fit; vision alone drives the stage | 3.0 / 3.5 / 2.5 mm | 27 | up: 0.23 / 2.58 / 4.49 | **3** | **3** (`align_full7` 4.49, `k4_4` 3.54, `demo_5` 3.51) | 0 | align_full7 (-0.99) | **bias [-0.3, 2.6, 0.8]: fired 0**, nearest demo_1 up -2.79 (0.71) |
| **Align scale (slope)** | quality | catches a wrong camera model | 0.6-1.6 | 24 | 0.57 / 0.88 / 1.00 (worst axis) | **1** | **1** (`align1` gap 0.57, 2026-11-03, after a 4.5 deg turn) | 0 | align1 (-0.03) | slope prior: still fires on align1 (gap se 0.09) |
| Align step limit | quality | cycle budget | 60 | 31 | 11 / 17 / 25 | 0 | 0 | 0 (prec6 caught by the stuck gate) | 35 cycles | same |
| Align post-press budget | quality | the press can move the hand | 6 cycles | 19 | 3 / 3 / 5 | 1 (`k4_5` 16:24, doc) | 0 (fine not judged) | 0 | kjrc_1 (1 cycle) | same |
| Align time | quality | time budget | 120 s | 31 | 21 / 42 / 64 | 0 | 0 | 0 | 56 s | same |
| Align tip lateral | safety | hand stays near the stop line | 20 mm | 31 | 2.0 / 3.8 / 7.2 | 0 | 0 | 0 | 12.8 mm | same |
| Align tip advance | safety | no advance past the hand-over | 16 mm | 31 | -0.3 / 10.3 / 12.9 | 0 | 0 | 0 | demo_2 (3.1) | same |
| Align grip change | quality | reader moving in the jaws | 0.02 | 31 | 0.000 / 0.001 / 0.013 | 3 (11-03: align_full1, both align_full3) | 0 | 0 | prec4 (0.007) | same |
| Align drift (no command) | safety | hand moving by itself near the case | 1 mm, 2 in a row | 27 | 0 / 0.27 / 1.60 | 0 | 0 | 0 | a4p_1 once over 1 mm | same |
| Align measurement vs FK | quality | vision must follow FK | 2.5 mm + 2 sigma, 2 in a row | 31 | max excess 0.35 | 0 | 0 | 0 | 2.1 mm | same |
| Align min clearance | safety | plug must not touch the case | 1 mm | 31 | 3.4 / 5.5 / 6.7 | 0 | 0 | 0 | 2.4 mm | same |
| Align angle budget | quality | correction budget | 7 deg | 31 | 2.25 / 3.75 / 5.25 | 0 | 0 | 0 | 1.75 deg | same |
| Align stuck steps | quality | friction beyond the lead limit | 3 in a row | 30 | 0 / 0 / 1 | 1 (prec6, true) | 0 | 0 | 2 steps | same |
| Step end (time only, not a stop) | quality | - | 0.3 mm strict, 3 s | 15 runs 11-10/11 | 123 of 229 steps time out | - | - | - | - | loose + 2 s: **44 of 229**; step time p50 per run 37.4 -> 16.0 s |
| Seat-2 detection | quality | ends the policy | depth >= 4.8, 6 mm/s, 0.3 s | 17 | max depth 6.1 / 8.1 / 9.4 | 3 (11-03/06 stalls, true) | 0 (`k4_2` now detected at tick 67) | 0 | kjrc_3 (1.3) | same |
| Guard bound + hard | safety | side load on the port | 5 + 3 mm | 17 | 2.0 / 2.9 / 4.7 | 3 (11-03/06 stalls, true) | 0 (`k4_2` exempt with the 6 mm/s seat) | 0 | speed075_1 (3.3) | same |
| Guard press abort | safety | pressing on the case with no progress | 3 s | 17 | 0.1 / 0.17 / 1.72 | 0 | 0 | 0 (full5, full10 caught by the bound) | speed075_1 (1.3 s) | same |
| Insert time | quality | budget | 10 s | 17 | 1.1 / 1.6 / 4.6 | 0 | 0 | 0 | k4_2 (5.4 s) | same |
| Push min depth | quality | push only into a plug in the port | 5 mm | 13 | 6.0 / 8.0 / 9.2 | 0 | 0 | 0 | kjrc_3 (0.99) | same |
| **Push max lateral** | quality | no side load; a skip leaves the plug short | 2.5 mm (last tick) | 13 | 0.37 / 1.44 / 3.50 | **2** | **2** (`demo_1` 3.50, `speed075_1` 3.33) | 0 | demo_1 (-1.0) | same |
| Push watch advance | safety | plug or grasp slipping | 4 mm | 5 | 2.06 / 2.21 / 2.47 | 0 | 0 | 0 | demo_3 (1.5) | same |
| Push watch left hand | safety | the PC press gives way | 2 mm | 8 | 0.07 / 0.12 / 0.25 | 0 | 0 | 0 | 1.75 mm | same |
| **Push watch gripper** | safety | reader turning in the jaws | 0.02 | 8 | 0.001 / 0.009 / 0.020 | **1** (`k4_4` 16:10, doc) | 1 (seated, nothing noted) | 0 | 0.000 | same |
| Push off-line (force mode) | safety | side load | 4 mm | 7 (not comparable) | position pushes: 1.4-6.9 mm | - | - | - | - | on the logged 15 mm position pushes 5 of 5 would fire (5.6-6.9 mm); the force push holds a line, so this only bounds it |
| Full depth report (FULL/SHORT) | quality | report only | 7.2 mm | 11 | fully in 7.32 / 7.36 / 7.56 | - | - | - | fastmove_1 (0.12) | FULL on 3 of 3 fully in, SHORT on 5 of 5 short, SHORT on the 3 not stated (5.90-6.69) |

Not replayed (no per-tick values or the motion changes): TCP box, SafeArm table clearance (one true stop: 2026-11-06 prec2), move_to goal tolerance (two stops on 2026-11-06, both bugs), camera frame age (one true stop, doc only), correction-grows, the no-jump deadband, the second pick attempt, the force push torque cap and end test. The list is in `gate_replay.json` (`not_replayed`).

## What the replay says

1. **Every stop of a fine run comes from 4 quality gates.** Disagree 3, push lateral 2, scale 1, push gripper watch 1 (doc value 0.020 at a 0.02 limit). No safety gate fired on a fine run. No gate missed a failure on its own (the press abort did not fire on `align_full5` and `align_full10`, but the guard bound stopped both).
2. **The safety gates have room on these runs.** Torque peak at most 3.2 of 5 N m (right joint 4), joint speed 2.19 of 2.5 rad/s (servo run), press 28 of 60 s, align lateral 7.2 of 20 mm, guard bound 4.7 of 8 mm on good insertions, push watches at most 2.5 of 4 mm and 0.25 of 2 mm. One place, one PC spot: this says nothing about other placements.
3. **all9 removes the 3 disagree stops** (bias subtraction) and keeps 0 false stops on 27 hand-overs. The nearest good run is then `demo_1` at -2.79 mm up (margin 0.71 mm): the odd-grasp run. So the bias works for the usual grasp and is close to the limit for an unusual one.
4. **all9's loose reach test** cuts the step timeouts from 123 to 44 of 229 logged steps and the step time per run from 37.4 to 16.0 s (median, n = 15). This is the FK of the old motion; it does not include the larger gap steps.
5. **all9's slope prior does not change the scale gate** on the logged levels: `align1` still fires (gap 0.57, se 0.09). All other 23 runs pass with a margin of 0.11 or more. The `align1` value comes from the 6 mm stop and a 4.5 deg turn; the 18 mm stop runs have gap slopes 1.02-1.12.
6. **The seat-2 gate at 6 mm/s now detects `k4_2`** (tick 67, 1.34 s), so the guard would not have pulled it out. The guard bound then exempts it.
7. **The 15 mm position push and the force-push off-line stop do not mix.** On the logged position pushes the hand left a straight line by 5.6-6.9 mm (5 of 5 over 4 mm). The force push is built to hold the line (dry runs 0.13-1.3 mm). The first force pushes on the rig are the real test of this stop.

## Quality gates that are too tight or too loose with this data

Too tight (false stops, n in brackets):

- **`align.disagree_mm` up 3.5 mm** (3 false stops of 27 hand-overs). Cause: a net bias of about +2.6 mm (`ROBUSTNESS_PLAN_20261113.md` part B). all9 handles it with the bias.
- **`insert.final_push.max_lat_mm` 2.5 mm read at one tick** (2 false skips of 13 seats). The median over the last 25 deep ticks passes all 13 (largest 1.81 mm). Both skipped runs stayed short or hanging.
- **`insert.final_push.max_grip_change` 0.02** (1 stop of 8 pushes, value 0.020; doc only, so the exact value is not known). The other 7 pushes read at most 0.011. Not enough data to call it too tight; it is a safety watch, keep it.
- **`align.slope_min` 0.6** (1 false stop, 2026-11-03, 6 mm stop and a large turn). Not seen since the 18 mm stop (0 of 15). No change needed now.

Too loose (no gate fired, but there is a lot of unused room):

- `align.max_time_s` 120 s (good runs at most 64 s, all9 model 12-18 s), `align.max_cycles` 60 (at most 25), `insert.insert_s` 10 s (at most 4.6 s). These are budgets, not checks. Tightening them only shortens a failed run.
- `align.max_lat_mm` 20 mm (at most 7.2 mm) and `align.max_advance_mm` 16 mm (at most 12.9 mm): safety, keep as they are.

Gates with a small margin on a good run (watch them):

- Pick held band: `demo_1` reading 0.303, 0.028 above the lower limit.
- Push min depth 5 mm: `kjrc_3_push` at 5.99 (K-J). K4 seats read 6.8-9.2.
- Full depth 7.2 mm: `fastmove_1` at 7.32 (0.12 mm). The FULL/SHORT split rests on 3 fully-in runs.

## Proposals (text only; each needs the operator's approval)

1. Approve the all9 net bias `align.secondary.bias_mm` [-0.3, 2.6, 0.8] (already proposed in `ROBUSTNESS_PLAN_20261113.md` B1). Replay: 3 false stops -> 0 of 27, nearest good margin 0.71 mm (`demo_1`). Variant inside today's limits: none helps (B1 text).
2. Push lateral gate on the deep median (B3 of `ROBUSTNESS_PLAN_20261113.md`): limit stays 2.5 mm, the statistic changes from the last tick to the median of the last 25 deep ticks. Replay: 2 skips -> 0 of 13. This does push on a reader that hangs (`demo_1`), so it is the operator's call.
3. Do not tighten any safety limit or budget from this data: one placement, 37 runs with logs, and only a few failures per gate.
4. Keep the force push off-line stop at 4 mm. Check it on the first force pushes (the runbook in `NEXT_RIG_20261114.md`).

## Depth signals without the operator (ideas 2 and 5)

Units: rest depth = FK advance of the TCP since the hand-over at the end of the insertion stage minus the vision gap (the units of `insert_depth`). Labels: operator 2 = fully in, 1 = seated, depth not stated, 0 = short or hanging. "Hand band" = the order read by hand on six 2026-11-11 frames (`results/real_align/depth/shell_end_frames.jpg`). Spearman rho, two-sided p; exact permutation p for n <= 8.

### (a) Contact head ("touching" -> "inside")

- K4 only. The K-J head does not work as a sensor: on the seated K-J runs the inside share ends at 0.00-0.08 (K4: 0.74-0.96), and its flip comes at -1.9 to -0.7 mm, before the mouth.
- K4 flip ("mouth") depth: 0.71-3.05 mm (n = 8 seats, median 2.1), at 0.52-0.70 s, while the plug moves at 10-20 mm/s (one 20 ms tick = 0.2-0.4 mm).
- Depth from the mouth at rest, K4: 3.08-5.98 mm.

| Signal (K4 seats unless noted) | vs operator label | vs hand band order |
|---|---|---|
| Rest depth (present) | rho 0.69, p 0.057, n = 8 (all 11 with K-J: 0.74, p 0.009) | rho 1.00, exact p 0.003, n = 6 |
| Depth from the mouth (contact head) | rho 0.37, p 0.37, n = 8 | rho 0.37, exact p 0.50, n = 6 |
| Mouth depth alone | rho -0.15, n = 8 | rho 0.83 in the wrong sense for a mouth marker (exact p 0.058) |

Full (n = 3) against short (n = 5): rest depth separates them (7.32-7.56 against 5.58-6.94). Depth from the mouth does not (4.68-5.34 against 3.08-7.92).

**Result: idea 2 is falsified on this data** (its own test: "rank agreement with the band is no better than the present rest depth"). The flip depth spreads over 2.3 mm, much more than a fixed mouth plus the vision gap error (about 0.5 mm). So the flip tick is not a precise mouth mark: the head lags, and the plug moves 0.2-0.4 mm per tick at that moment. Subtracting it adds noise to the rest depth.

### (b) Plug band detector

Method: the shell is located on the 5 hand-over frames (`port_vision.detect_plug`: rim edges, tip edge, sleeve end). On the last right-wrist stream frame before the release (0.1-0.4 s before), in the shell columns: the neck edge (strongest bright-to-dark step) and the band top (half-way level between the band peak and the case above). mm = px / ((sleeve end - tip edge) / 5.8 mm) from the hand-over frames (4.8-5.4 px/mm). A band with less than 12 grey levels of contrast reads as "no visible band" (0 mm). Only runs with `--stream` have frames in the insertion: n = 11 seats with a rest depth.

| | n | rho | p |
|---|---|---|---|
| Band vs hand band order | 6 | 0.83 | exact 0.058 |
| Band vs rest depth, K4 | 8 | 0.86 | exact 0.011 |
| Band vs rest depth, all (K4 + K-J) | 11 | 0.30 | 0.37 |
| Band vs operator label, K4 | 8 | 0.28 | 0.51 |
| Band vs depth from the mouth, K4 | 8 | 0.41 | 0.33 |

- K4 band: 0 (demo_4, no visible band) to 1.97 mm (speed075_1). Fully in: 0, 1.35, 1.54 mm. Short: 1.22 (demo_1), 1.66 (k4_1).
- The K-J runs (short, rest 5.6-6.0 mm) read 1.15-1.51 mm, the same as K4 runs at 7.3 mm. On those frames a lit area above the bright rim is part of the shell, and the detector measures only the rim. The light on the shell changes from run to run (band contrast 3-103 grey levels).
- Frame to frame (the frame 0.5 s before, same run): within 0.13 mm on 9 of 11 runs. On `demo_1` and `speed075_1` the earlier frame reads 3.4 and 4.4 mm (the hand was still moving then; cause not checked). So only a frame at rest is usable. The idea's own pass bar (0.3 mm against calipers) cannot be checked: there is no caliper truth.

**Result: the band follows the K4 rest depth and the hand order (rho 0.83-0.86), but it does not separate fully in from short on its own, and it fails across the two policies.** It is a weak second signal, not a depth sensor yet.

### What n allows and what it does not

- n = 6 to 11 runs, one PC spot, one reader spot, one camera, two days. A rank correlation of 0.83 with n = 6 has an exact p of 0.058: it is a hint, not a result.
- It allows: the rest depth already orders the six hand-read frames perfectly (rho 1.0, n = 6) and separates the 3 fully-in from the 5 short runs. The contact-head flip does not improve it. The band detector agrees with both on K4.
- It does not allow: a threshold with an error rate, a mm accuracy for the band, any claim at another placement, lighting or policy, or a claim about true depth. All labels are operator words or hand reading; there is no measured depth in any run.

## What to log on the next rig session

For the gates:

1. **Every gate value at its decision, as one event** (`gate`, value, limit, margin, decision), including the gates that passed. Today the disagreement, scale and push values must be rebuilt from other events, and torque, clearance and the TCP box are not logged per decision.
2. The SafeArm clearance minimum per stage (gripper, link_5, link_6) and the TCP box margin of both arms per stage.
3. The push watch values per tick (advance, left hand, gripper readings, off-line distance), not only the worst. The `k4_4` jaw value 0.020 is known only from the doc.
4. The operator outcome per run in the log (`--note` or a prompt at the end): seated yes/no, fully in yes/no, reader slipped yes/no. This makes "fine" a logged label, not a reading of the session document.
5. Copy every run folder to mason the same day (7 folders of 2026-11-10 are still empty).

For depth:

6. **A true depth per run**: caliper of the visible shell (neck to case) after each seat, before the release, or a fixed side camera frame. Without it no depth signal can be calibrated.
7. **Run with `--stream`** so every insertion has wrist frames; better, 3-5 raw right-wrist frames at rest just before the release (tag `insert_rest`), and one after the push drop. Fixed exposure on the right wrist camera, so the band contrast does not change between runs.
8. Idea 5's own test (no robot): plug placed by hand at 0, 1, 2, 3 mm short, 10 frames each from the wrist pose, with calipers. That gives the band's mm scale and error.
9. The contact-head share per tick is logged already. Log the policy's seat-2 depth per tick at 50 Hz in the same row (done) and the plug speed at the flip; with 20+ K4 seats the flip can be retested, but on this data it is not worth more work.

## Files

- `scripts/real/gate_replay.py`, `scripts/real/depth_signals.py` (new).
- `results/real_runs_review/gate_replay.json`, `gate_replay.csv`, `depth_signals.json`, `depth_signals.csv`, `depth_signals.png` (new).
- No config, deploy code or existing file changed.
