# Session 2026-11-03, duchamp3: first live align run stopped on a jump in the approach

Branch `real/graph-align` (`c545340`), config `deploy/config/bimanual_align2.json`, guide [NEXT_RIG_SESSION.md](NEXT_RIG_SESSION.md). Run logs: `data/real_runs/2026-11-03/` (gitignored). Video: `IMG_8788.MOV`.

## Status

Stopped in step 2 after one run, by the guide's stop condition "jumps by more than about 5 mm". `measure2`, step 3 and step 4 did not run. Limits and configs did not change.

## Runs

| Time | Run | Command | Outcome | Operator saw |
|---|---|---|---|---|
| Before 14:48 | setup checks | as in the guide | all pass; timing p99 at most 11.7 ms per stage | |
| Before 14:48 | `camera_check` | step 1, empty jaws | camera opens, 0 of 30 (no plug in view, expected) | |
| 14:48 | `full_144822_measure1` | `--stage full --speed 0.5 --config $C --align-mode measure --name measure1` | pick held, stage measured, backed out, no contact in the log | "very close to usbc port and aligned, then started moving away and down"; at the stop the plug was lower than the port, tilt "slightly down but basically flat" |

## Measurements at the stop (10 cycles, 50 of 50 frames, arm still)

| Source | Across mm | Up mm | Gap mm | rx, ry, rz deg |
|---|---|---|---|---|
| Port vision | -0.49 | -4.96 | 4.59 | -4.77, -1.18, -1.50 |
| `wrist_r10f_ft_abs` | -1.13 | -3.01 | 6.00 | -0.77, +0.10, +0.74 |

Spread at most 0.35 mm per axis, SD at most 0.08 mm. Time per frame: median 85 ms, one first frame 455 ms. Shell depth 121.08 mm, accepted. The port is the one jogged into on 2026-11-02: a `collect` frame at about the same offset shows the same view.

Plug against the panel in the old-stop offline numbers: rx -3.2 to -3.4, ry -1.8 to -2.4, rz -1.1 to -1.3 deg.

## The jump, late in the approach

Port axes, relative to the stop. Encoder FK with joint offsets. Ticks are 20 ms.

| Tick | Across | Up | Gap axis | Commanded up | Joint 2, 3 target - measured mrad | Vision across, up, gap | Net across, up, gap |
|---|---|---|---|---|---|---|---|
| 2740 | +1.9 | +3.6 | +0.1 | +8.5 | +15, +21 | +1.2, -2.0, 4.3 | +0.94, +1.09, 5.96 |
| 2764 | +1.8 | +3.7 | +0.1 | +8.0 | +18, +23 | | |
| 2776 | -1.9 | +6.9 | -0.5 | +7.7 | +20, +11 | | |
| 2800 | -0.5 | 0.0 | +0.1 | +7.5 | +2, +11 | | |
| 2820 | -0.8 | +0.2 | 0.0 | +7.6 | +2, +10 | -1.4, -5.1, 4.4 | -1.90, -2.74, 5.97 |

- Joint 3 lets go first (tip up 3.2 mm, sideways 3.7 mm in 0.24 s). Then joint 2 lets go (tip down 6.9 mm in about 0.4 s). The commanded tip moves 0.5 mm in this time.
- Before it, the arm stayed at about the same place from tick 2339 to 2764 while the commanded height fell from +16.3 to +8.0 mm: stuck in friction while the servo moved the target. Joint 2 error went from -25 to +18 mrad.
- The net read 2.4-3.1 mm above vision in up on these frames, so it still asked to go down when vision read 2 mm low.
- Vision followed FK through the approach within about 1.5 mm in up, on 12 offline frames: this is the first evidence that it tracks the live motion.
- Largest 0.5 s move in the last 300 approach ticks of 2026-11-02: A 3.0, A2 4.1 mm. The 0.5 s window here gives 6.9 mm or more.
- Video: the drop is visible between about 23 and 24 s.
- No contact in the log: the gap axis stayed within 0.5 mm. The commanded tip was 2.2 mm deeper than the measured tip, so the same jump could close the gap by about that much.

## Back-off is not straight

Measured: out 16.4 mm, down 6.6 mm, across 2.4 mm. Commanded: straight out (up constant).

- First 0.5 s: 4.6 mm down. The stage sets the target to the measured pose and so removes the 34 mrad holding lead on joint 5 (kp 10). The wrist sags until the error rebuilds to 15-21 mrad.
- Then joints 2 and 3 never move: errors reach only +4 and +10 mrad, inside the friction band. The wrist joints make the whole motion.
- Lowest plug tip in the back-off 20 mm above the tabletop. No clearance problem in this run.

## Other notes

- **Tilt not settled.** Vision says rx -4.77 deg, the net -0.77, the operator "basically flat". In the video the body's edges slope about 0.5-2.5 deg, tip down, but the view is oblique and hand-held, with no level reference, so this does not separate them. Step 3 turns the plug by the vision value. It needs a square-on side view first.
- **Grip reading.** 0.324 at the squeeze, 0.316 from the approach on. All 13 picks on 2026-11-02 read 0.308-0.315. Inside the held band; no gate fired. One run.
- **Log artefact.** In measure mode `rot_applied`, `fused` angle and `turned` grow without motion (6.75 deg at cycle 10), and `slope` means nothing. Use the raw `rv_deg`. If `measure_n` were higher this would stop the stage at `max_deg`.
- **Jaws at the end.** They open at the prompt "Hold it, then Enter opens the gripper", as written. The guide's "closed jaws" covers the back-off.

## Questions for the server

1. How should the approach end so that the joints do not store this error and let go near the case? The creep controller runs only from the align stage on.
2. Is a stop 5 mm low, with a gap near 4.6 mm, an acceptable entry for the align stage?
3. Why does the net read about 3 mm above vision on these frames? Which one is right, given the label constant?
4. Should the back-off keep the wrist lead?

## Run `full_151615_align1` (operator override of the jump stop, step 3, no insertion)

`--stage full --speed 0.5 --config $C --align-mode align --name align1`. `full_151410_measure1` before it only connected and quit at the first prompt: no motion.

**Outcome: stopped at the hand-over check, `scale gate: gap: measurement moves 0.57 mm per mm of FK motion (allowed 0.6-1.6)`. The position and angle were in tolerance.** 39.4 s, 16 cycles, every cycle 5/5 frames.

| | Across mm | Up mm | Gap mm | rx, ry, rz deg |
|---|---|---|---|---|
| Vision, entry | +1.21 | -2.33 | 5.33 | -5.1, -0.9, -1.2 |
| Vision, end | -0.12 | +0.48 | 6.02 | -0.5, 0.0, -1.2 |
| Net, entry | +0.76 | -1.31 | 6.35 | -0.36, -0.03, +1.24 |
| Net, end | -0.37 | +1.65 | 7.14 | +3.94, +0.96, +1.28 |

Applied turn 4.5, 1.0 deg (limit 7). Slopes: across 1.01, up 0.99, gap 0.57. Moved fraction p10, p50, p90: -0.21, 0.81, 1.09; 3 of 14 below 0.25.

**The stop looks false, from the turn (n = 1).** Vision gap minus FK gap holds at 5.03-5.27 mm through the probes (rot 0), then falls to 4.02 mm as rx turns 0 to 4.5 deg; about 0.24 mm per deg. Across and up minus FK stay steady. Gap slope on cycles 1-7 is 0.99, on all cycles 0.57. The net's gap follows vision, not FK. So the gate regresses a pure translation model over cycles with a rotation in them. FK's tip point or pivot is likely off by a lever of about 14 mm; not separated.

**Tilt is the open risk.** Both sources saw the turn (vision +4.6, net +4.3 deg in rx), so the turn is real. They differ by about 4.4 deg in absolute rx. If vision is right the plug is now flat; if the net is right it is now tipped by about +4 deg. Before the turn the operator saw "basically flat", which fits the net better. No side view of this run.

**The jump repeated:** 6.1 mm in 0.5 s at tick 2459 with 1.4 mm commanded, up about +6 to -2 mm in two steps, late in the approach. In the stage, one slip of 1.7 mm across (cycle 9 to 10) with a near-zero ask. No contact in the log.

**Grip** again high: 0.329 at the squeeze, 0.318 at the stage entry, 0.314 at its end, against 0.308-0.315 on 2026-11-02.

## Run `full_152618_align2` (no insertion, `--align-set slope_min=0.5`, operator approved)

**Outcome: `aligned; mode align: no insertion`.** 32.5 s, 13 cycles, every cycle 5/5 frames. Gap slope 0.84, so this run would also have passed the 0.6 limit. Slopes 0.97, 0.96, 0.84. Moved fraction p10, p50, p90: 0.86, 0.98, 1.39; none below 0.25, one above 3.

| | Across mm | Up mm | Gap mm | rx, ry, rz deg |
|---|---|---|---|---|
| Vision, entry | +0.38 | -4.99 | 5.32 | -3.5, -2.0, -1.3 |
| Vision, end | -0.08 | +0.26 | 5.69 | -0.5, 0.0, -1.9 |
| Net, end | -0.3 | +2.6 | 6.6 | +3.6, +1.9, +0.2 |

Applied turn 3.0, 2.0 deg. Vision minus FK gap: 5.2-5.3 mm before the turn, 4.65 mm after, so the turn moves it here too, by less. Grip 0.3155 at the squeeze, 0.310 in the stage: back in the 2026-11-02 range. Jump in the approach, third run of three: 5.7 mm in 0.5 s with 1.1 mm commanded, tick 2818. Tilt at the end still has no side view: vision and net differ by 4.1 deg in rx.

## Run `full_153808_align_full1` (full mode, `--align-set slope_min=0.5`, operator override of the 3 of 3 rule)

**Outcome: stopped in the align stage, `gripper reading changed 0.299 -> 0.279` (limit 0.02). The insertion policy did not start.** 43.3 s, 15 cycles, every cycle 5/5 frames. Backed out with closed jaws.

Grip reading through the run, as discrete drops: 0.310 squeeze; 0.309 to 0.299 in transport (ticks 1600-1876); 0.2965 at tick 3023; 0.2875 at tick 3808 (first cycle after the first turn step); 0.2785 at tick 4926. Each stage drop takes about 0.2 s. Held band is 0.275-0.36. The gate is doing its job here: this looks like the reader moving in the jaws. Do not loosen it.

| | Across mm | Up mm | Gap mm | rx, ry, rz deg |
|---|---|---|---|---|
| Vision, entry | +1.59 | +0.49 | 5.72 | -6.0, -1.2, -2.0 |
| Vision, last full cycle (14) | -0.28 | +2.49 | 5.36 | -0.5, -0.1, -2.1 |

Up was still out of tolerance when the gate fired. Vision minus FK drifted through the run: up +0.49 to +1.67, gap 5.72 to 3.47 mm. Turn 5.2, 1.2 deg. Net at the end rx +4.48.

Grip and grasp by run (same reader, same tape):

| Run | Squeeze | Stage entry | Stage end | Shell depth mm | Entry rx deg | ZED reader yaw deg |
|---|---|---|---|---|---|---|
| measure1 | 0.324 | 0.316 | 0.316 | 121.1 | -4.8 | 91.8 |
| align1 | 0.329 | 0.318 | 0.314 | 121.4 | -5.1 | 91.7 |
| align2 | 0.3155 | 0.310 | 0.310 | 120.2 | -3.5 | 92.2 |
| align_full1 | 0.310 | 0.299 | 0.2785 | 117.2 | -6.0 | 93.2 |

The reading falls in every run before the stage, 0.006-0.011. n = 4; causes not separated.

## Run `full_155211_align_full2` (full mode, `--align-set slope_min=0.5`): seated

**Outcome: `seated_detected` at 1.86 s of the insertion, depth 9.08 mm by the seat measure. Operator: "got it to work".** The whole chain ran from a real pick: stage OK in 39.0 s and 16 cycles, press on, three measurements under the press, hand-over, K-J, release with the reader left in the port.

| | Across mm | Up mm | Gap mm | rx, ry, rz deg |
|---|---|---|---|---|
| Vision, entry | +2.76 | -6.64 | 4.63 | -4.1, -1.9, -1.1 |
| Vision, stage end | -0.02 | -0.37 | 6.33 | -0.6, -0.1, -1.4 |
| Vision, under the press | -0.08 | -0.62 | 6.49 | -0.7, 0.0, -1.5 |

Applied turn 3.8, 1.9 deg. Slopes 0.99, 0.94, 0.71. Hand-over gap (fused) 5.82 mm. Insertion: tip across -1.5 to +0.8, up -1.2 to +2.9, in 15.3 mm; guard bound active 0 ticks. The commanded target reached about 39 mm above the measured tip by the end, so K-J still winds the target up; the plug was in the port by then. Grip 0.318 squeeze, 0.311 stage, 0.301 at the insertion. Real seats on earlier runs read 6.2-9.5 mm.

This is one seat out of two full-mode tries: n = 1 success, one reader spot, one PC pose. A clip or an operator check of the plug's full depth is not in this record yet.

Operator confirmed for `align_full2`: the plug went fully in, and the reader stayed in the port after the gripper let go. Clip: `IMG_8790.MOV` (hand-held, oblique). In the clip the plug is low at about 25 s, rises to port level by about 46 s, and goes from in front of the port to inside it between about 60 and 61 s. The reader stays in the port through the end of the clip at 124 s. By eye the body's line does not change visibly across the seat; an automatic edge fit was not usable after the turn (the gripper merges with the body in the fit, and the camera moves), so this gives no angle number.

Gap slope against the applied turn, n = 4: 4.5 deg 0.57; 5.2 deg 0.63; 3.8 deg 0.71; 3.0 deg 0.84. It mostly falls as the turn grows, which supports the turn as the cause of the `align1` stop.

## Trials after the first seat: reader lost twice in the align stage

Both runs are named `align_full3`: `full_162735_align_full3` and `full_163516_align_full3`. Neither reached the insertion. Tally of full-mode tries: 1 seat of 4 (`align_full1` slip stop, `align_full2` seated, two reader losses). K-J ran once and seated once.

| Run | ZED reader yaw deg | Squeeze | Shell mm | Stage s | Turn deg | Grip at entry, lowest | Tip speed p99, max mm/s | Result |
|---|---|---|---|---|---|---|---|---|
| `align1` | 91.7 | 0.329 | 121.4 | 39.4 | 4.5, 1.0 | 0.318, 0.314 | 30, 67 | Held; gap slope stop |
| `align2` | 92.2 | 0.316 | 120.2 | 32.5 | 3.0, 2.0 | 0.310, 0.310 | 28, 54 | Held; aligned |
| `align_full2` | 92.0 | 0.318 | 121.1 | 39.0 | 3.8, 1.9 | 0.311, 0.301 | 27, 66 | Held; seated |
| `align_full1` | 93.2 | 0.310 | 117.2 | 43.3 | 5.2, 1.2 | 0.299, 0.279 | 27, 72 | Slip; gate stop |
| `162735` | 93.4 | 0.305 | 116.3 | 8.5 | 0, 0 | 0.289, 0.022 | 22, 29 | Reader lost at about 8 s, in the probes |
| `163516` | 93.2 | 0.316 | 119.5 | 39.6 | 3.8, 1.7 | 0.310, 0.031 | 28, 78 | Reader lost at about 39 s, in 5 drops |

`measure1` (yaw 91.8, no motion in the stage) also held.

Operator on `163516`: "allignemnt took to long the usbc was allined at some points but constat shaking and slight movements causes usbc to slip".

- **The shaking is real.** In every moving stage the 1 mm steps arrive as breakaways: tip speed peaks of 54-78 mm/s, moved fraction up to 2.6, up to 5 mm in 0.5 s. Grip drops happen during moves, at tip speeds of 6-30 mm/s.
- **Speed does not separate held from lost.** Held runs peak at 54-67 mm/s; `162735` lost the reader with a 29 mm/s peak. Stage time does not separate them either: 32-43 s in all long runs, and one loss came at 8 s.
- **Yaw separates them so far:** at most 92.2 deg, held 4 of 4; at least 93.2 deg, slipped or lost 3 of 3. n = 7, a 1-2 deg difference, near the ZED's resolution, not a tested cause. Squeeze and shell depth do not separate them: `163516` read 0.316 and 119.5 mm.
- **Reading:** a marginal grip, which the motion then works loose. Grip force, jaw contact against reader yaw, and the stage's motion are not separated.

`full_164445_align_full4`: pick only (yaw 90.9, squeeze 0.307), third prompt answered `s`; not counted as a try.

## Run `full_164912_align_full5`: good hand-over, no seat

**Outcome: `guard_bound` at 3.74 s of K-J. No seat. Lifted out with closed jaws, as designed. Reader kept.** Tally: 1 seat of 5 full tries; K-J 1 of 2. Yaw 89.9, squeeze 0.310, shell 119.7; stage OK in 32.8 s, slopes 0.94, 0.96, 0.93; grip 0.304 at entry, 0.295 lowest.

Under the press at the hand-over, vision: +0.27, +0.36, 5.28; rx, ry, rz -0.9, +0.2, -1.9 deg (`align_full2`: -0.08, -0.62, 6.49; -0.7, 0.0, -1.5).

| | `align_full2`, seated | `align_full5`, not seated |
|---|---|---|
| Plug stops on the case | about 7.9 mm in, depth 2, t 0.64-0.72 s | about 7.3 mm in, depth 1.3, from t 0.5 s |
| Up from the hand-over at the stall | +1.7, then -0.8 in 0.1 s | +1.6 to +1.9, never comes down |
| Up by vision at the stall (hand-over + motion) | -1.4 at the moment before it enters | about +2 |
| K-J target up at the stall | +1.7 to +2.0 | +20 to +35 |
| What happens next | enters: 15 mm in by 1.3 s | across jump of -5.4 mm at 2.2 s, bound stop at -10.5 mm |

- The seat began with the plug about 1.4 mm below vision's zero. n = 1 against n = 1. Vision's up constant is not known better than 1-2 mm (ALIGN_STAGE limits).
- In `align_full2` the drop that let the plug in took 0.1 s: this looks like a joint letting go, so part of that seat may be luck. Not separated.
- K-J's target climbs in both runs. In the seat it climbed after the plug was going in; here it climbed before, and held the plug up on the case. This is the climb seen on 2026-11-02.
- Reader held at yaw 89.9. Yaw table: at most 92.2 deg, 5 of 5 held; at least 93.2 deg, 3 of 3 slipped or lost.
