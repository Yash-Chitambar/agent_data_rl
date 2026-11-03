# Align stage, seat detector 2, insertion guard

Branch `real/graph-align`, config `deploy/config/bimanual_align1.json`. `bimanual.json` and `bimanual_pick3.json` did not change, and these stages stay off with them (`deploy/checks/align_gates.py`). No robot motion in this work: logged real frames and sim only. Numbers: `results/real_align/`. Which sim grasp each sim number uses is stated next to it, because K-J's rate depends on it.

## What it does

Between the approach and K-J: the plug goes to about 6 mm in front of port 2, lined up within about 1 mm, in small steps. K-J starts only if every gate holds. Otherwise the plug backs out 15 mm with closed jaws and K-J never runs.

1. **Press off.** From the latch until the plug is lined up, the left press has no force and no press clock (as the `collect` stage). The hand stays on the PC. This leaves the 60 s limit unchanged and frees the time: the old chain used about 47 s of it at speed 0.5 before the release.
2. **Shell depth.** Nine still frames at the stop give one depth for this grasp (`port_vision.plug_depth`). Outside 113.5-123.5 mm: refused, nothing moves (picked grasps read 117.7-121.5, hand-placed ai5 and ai2 read 134 and 142).
3. **Probes.** ±2 mm in each axis and back. The camera must follow FK.
4. **Cycles.** Rest, 5 frames, gates, one step of at most 1 mm per axis, FK check of the step.
5. **Press on**, 1 s ramp, target lead clipped to K-J's range. Then three more measurements under the press, with corrections if needed (at most 6 cycles).
6. **In-hand net read again** at the hand-over gap (it trained at 6 mm), then K-J.

**Measurement.** `port_vision` decides across, up and gap. The r10 fine wrist net is a second source: the net cannot refuse a frame and it sits 1.4-1.8 mm off vision across, so it only has a vote at the hand-over (`disagree_mm` 3.0, 3.5, 2.5 mm: **a guess, never run in a loop**). Stub or missing source: the stage stops at its first gate, without moving.

**Fused estimate.** Vision noise at rest is under 0.12 mm, but its residual changes from pose to pose (up 0.74 mm). So the stage does not act on one measurement: it keeps the port in the FK frame, the median of (measurement - FK tip) over the last 6 cycles. FK is used only for this and for the small relative steps.

**Angle: not measured and not corrected.** `port_vision` gives no plug angle: it gives the camera against the panel (logged per frame as `panel_rv_deg`, yaw good to about 1 deg), and the plug in the hand is unknown. The stage holds the orientation the approach left. This matters: K-J drops to 59.5 % at up to 2 deg per axis (sim/grasp-real, v2). Achieved angle: table below; real value unknown.

**ZED yaw (about 1.8 deg high, moderate evidence).** How it reaches this stage: the port axes for the steps, the seat depth and the bound come from it. Effect: small. A 1 mm step gets 0.03 mm of cross motion, closed by vision; along the 12-18 mm advance of a seat it is 0.4-0.6 mm across. The larger concern is the hand's yaw: the transport pose takes it from the ZED, the approach then corrects it only by the net's rotation output, and this stage keeps what it gets. Not measured: PC yaw against a known square is on the rig list.

## Why these choices (real logs)

**Small commanded steps do not simply move** (`results/real_align/tracking.json`, collect1). Joint 2 friction band 1.33 N m (17 mrad at kp 80), constant load -0.87 N m. Sweep moves, target step against measured step: 1-2 mm: gain median 0.83, 37 % moved under a quarter; 2-4 mm: 0.78, 26 %; 8 mm and more: 1.00, 4 %. Operator jogs, n = 42: asked 0.5-1 mm, moved 0, then jumped 4-11 mm.

**Creep controller.** Target = goal + lead on every joint, so the position error always pulls toward the goal and the hand cannot run away. On joints 2-3 the lead ramps slowly while the joint is off its goal and is cut as soon as the joint moves; breaks from rest in both directions give the middle of the friction band (the load), which is the holding lead at the goal. On a one-joint plant with stick-slip (`deploy/checks/align_stickslip.py`, n = 150 per case, random friction and load, a simplified plant fitted by eye to the jogs: an argument, not a measurement): final error for a 1 mm step, p50 and p90: collect-style ramp 1.0, 1.82 mm; creep 0.13, 0.25 mm. In the sim runs the step moved fraction 0.79, 0.93, 1.10 of the ask (p10, p50, p90). Limits: lead at most 45 mrad on joints 2-3 (3.6 N m, below the 9 N m torque guard, which stays on).

**Hand-over target lead.** The real climb's main cause, below. `handover_lead_rad` is K-J's live-state range p1-p99.

### Hand-over gap: 6 mm, measured

K-J's training starts: gap mean 5.86, p1-p99 5.0-6.7, absolute range 4.4-8.7 mm. x p1-p99 -0.85..0.82, y -1.53..1.18 mm, times 1-2 for half the starts, and 30 % hard starts at 1.5-3.5 mm.

Baseline training grasp, n = 1024 per cell, `results/real_align/start_map_*.json`, rows of the chain's angle errors:

| Gap mm | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Seated % | 99.3 | 99.6 | 99.4 | 99.4 | 99.4 | 99.3 | 99.6 | 99.5 | 99.5 | 99.1 | 99.6 | 98.8 | 98.4 |
| Push p99 N | 17.2 | 17.8 | 16.4 | 15.6 | 14.1 | 13.0 | 13.9 | 16.0 | 17.0 | 18.3 | 19.5 | 18.0 | 17.9 |

95 % intervals are about ±0.5 points. This grasp does not separate the gaps. Realistic sim, v2 grasp, in-hand net input (`sim/grasp-real`, their numbers): 6 mm 94.1 %, 8 mm 88.3 %, 11 mm 58.5 %. Going closer than the training range has no measured gain. Against it: vision gap residual 0.49 mm plus a constant of about 0.4 mm, so a target of 5 mm reaches below 4 mm true at 2 sigma, out of training, and the last real step could touch the case. **Choice: 6 mm, ±1 mm.** Push force is also lowest there.

Lateral, gap 6 mm, baseline grasp, n = 1024 per ring radius: 0: 99.2, 1: 99.5, 2: 99.2, 3: 99.5, 3.5: 98.0, 4: 94.6, 5: 81.2, 6: 57.5 %. Up only: +1 99.4, +2 99.2, +3 98.4, +4 94.8, +6 81.8; -1 99.5, -2 99.2, -3 98.3, -4 88.4, -6 52.7 %. Realistic sim, v2: ±1 mm 89.6-94.7 %, ±2 mm 69.7-92.2 %. **Tolerance ±1 mm** follows the v2 numbers: the baseline grasp would allow 3 mm, and it is optimistic. Sim fingers do not yield as real ones do: use these to compare options, not as forecasts.

### Gates

| Event | What happens |
|---|---|
| No measurement, low confidence (< 0.5), sigma > 1.5 mm, frames disagree by > 1.5 mm, fewer than 3 of 5 frames | Retry; 5 in a row: stop |
| Sources disagree at the hand-over | Stop |
| Camera does not follow FK: slope after the probes outside 0.25-2.0; at the hand-over outside 0.6-1.6, or too little motion to tell | Stop. **The old nets' slopes 0.34, 0.17, 0.35 stop here** |
| Measurement jumps against FK by > 2.5 mm + 2 sigma, 2 cycles | Stop |
| Hand moves > 1 mm with no command, 2 cycles (drift) | Stop |
| Correction grows: 4 cycles, or 9 mm above its best | Stop |
| Step does not move the hand under a quarter, at the lead limit, 3 times | Stop (contact, or friction past the limit) |
| Moved > 20 mm lateral or 16 mm in from the stop; 60 cycles; 120 s | Stop |
| Gap below 3 mm | Stop. Not lined up within 2 mm: the plug backs to 9 mm first |
| Gripper reading changes by > 0.02 | Stop |
| Out of tolerance under the press after 6 cycles | Stop |

Every stop backs out 15 mm at 10 mm/s with closed jaws, then asks for the reader by hand. Everything is in `events.jsonl` (`align_measure` with every frame, `align_step`, `align_result`, `align_handover`). `scripts/real/align_log.py <run>` prints it.

## Sim results

`scripts/eval/eval_align_batch.py`: batched chain with the same `AlignLogic` and `creep_law`, encoder offset ±1.5 deg, believed frame, baseline training grasp, old sim PC look and old wrist nets (the r10 nets did not train on this look). Entry at a real-like stop: across -5..1, up 3..10, gap 6..8 mm by truth. Measurement: truth + run constant (0.3, 0.6, 0.4 mm) + pose residual (0.25, 0.74, 0.49 mm) + frame noise 0.1 mm + 10 % frame loss. Truth at the hand-over:

| Case | Worlds counted | Handed over [95 %] | Across abs p50, p95 | Up abs p50, p95 | Gap p5, p50, p95 | K-J seated, handed-over worlds |
|---|---|---|---|---|---|---|
| Port-vision-like, nominal friction | 100 | 70 [60, 78] | 0.33, 0.86 | 0.49, 1.65 | 5.28, 6.27, 7.33 | 70/70 [95, 100] |
| Same, joints 2-3 friction x 2.4 | 19 | 18 [75, 99] | 0.27, 0.61 | 0.81, 1.96 | 5.27, 6.38, 7.45 | 15/18 [61, 94], 3 grasp slip |
| Old net scale (slope 0.34, 0.17, 0.35), gates on | 15 | **0** [0, 20] | | | | none: stage stops (scale 8, gap limit 6, step limit 1) |
| Old net scale, scale gate off, to show why | 17 | 7 [22, 64] | 1.74, 3.54 | 2.63, 8.68 | 8.47, 9.95, 10.84 | 7/7, small n: training grasp |

Achieved against the target of about 6 mm and about 1 mm (port-vision-like, n = 70 handed over): within 1 mm in both across and up 56 (80 % [69, 88]), within 2 mm 68, gap 5-7 mm 62, gap range 4.96-7.64. So the p95 of up (1.65 mm) is past the target: K-J numbers for that range (`sim/grasp-real`, v2): ±1 mm 89.6-94.7 %, ±2 mm 69.7-92.2 %. The up error is mostly the run constant in the noise model (0.6 mm), which no averaging removes.

**Angle at the hand-over (truth, not corrected):** this table's grasp and look: rx 0.68, 2.69; ry 0.16, 0.44; rz 0.24, 0.96 deg (abs p50, p95).

**Stops are mostly false stops here, by design of the gates.** Of 30 stops in the first row: 20 scale gate (noise of 0.74 mm against 2 mm probes, not a bad camera), 6 step limit, 2 correction grows, 1 too far, 1 gap limit. A stop costs a try, not a part. If real noise is like that, widen `slope_min` or lengthen `probe_mm` on the rig: the runbook asks for the slopes.

**Time:** in the one-world runs the stage takes 19-21 s median, p95 31-44 s, about 15 cycles.

The batched eval loses the reader before the stage in about half of the worlds (92 of 192): not counted above. Not investigated.

### One-world dry runs, the real path, fair look

`deploy/run_bimanual.py --dry-run --stage full --speed 0.5` with this config: r10 nets, `pc_real_look`, `inhand_real` v2, encoder offset ±1.5 deg, entry offset across ±1, up 1-5, gap 0-1 mm, same noise, seeds 7100-7107 and 7200-7206. n = 15.

- Hand-over 12 of 15 [80 % : 55, 93], 3 scale gate. No crash, no guard stop before the hand-over.
- Truth at the hand-over: across abs p50 0.35, up 0.7, gap 5.4-6.6 mm.
- **Angle (truth): rx p95 1.4-1.6, ry p95 2.3-2.7, rz p95 2.0-2.1 deg.** This is what the approach left: ry -1.1 to -2.7 deg on every run, and the stage keeps it. Fine r10 net rotation output against truth, last frames, n = 21 of 7 runs: mean error +0.1, **+1.95**, +0.9 deg, SD 0.8, 0.4, 1.0: the net reads about 0 where the plug is at -2 deg. So in this sim the net's rotation has a constant bias. Real value unknown: `align_measure` and `net` events log it.
- **K-J after the hand-over: 0 of 12 seated by the sim's truth [0, 24 %].** Outcomes press abort 7, guard bound 4, seat detect 1 (false: true depth 5.3 mm, tilt above the sim's limit). Plug depth at the end: 5.6, 5.4, 5.3, 4.5, 4.4, 3.7 mm in six of them; on the case in the others. Without the guard, same seeds (7200-7206; the sim is not bitwise repeatable): 1 of 7 seated [3, 51 %], 6 time limits, end depth 4.4, 3.4, 2.2 mm in three. n is small, but K-J stalls without the guard too: the guard is not the main cause. The sim's seated test also has a 2 deg tilt limit.

This is the main open result: at this angle, K-J does not finish in this sim, as `sim/grasp-real`'s up-to-2-deg number warned (59.5 %). The stage fixes position and not angle.

## Seat detector 2 (`insert.seat2`)

Seated = plug past the MEASURED gap by seat depth - 1 mm (depth = advance - gap), across within 3 mm and up within 6 mm of the port (up is wide: a real seated hand read up to 4.5 mm lower, the reader pitches in the jaws), speed under 3 mm/s and lateral range under 2 mm for 0.5 s.

Real logs (`results/real_align/seat_real.json`, n = 2 + 2, the same runs that set the values: not a held-out test):

| Run, operator | Old | New, gap from vision | New, gap from the label | New, assuming 6 mm |
|---|---|---|---|---|
| A, not seated | no | no (depth 1.8, needs 4.8) | no (-1.8) | no (4.2) |
| A2, not seated | **seated at 8.84 s** | no (2.7, 3.6 with PORT_VISION values) | no (1.0) | no (5.6; the up gate rejects it) |
| ai4, seated at about 4 s | seated at 1.18 s: too early | seated at 4.44 s | 4.44 s | not run |
| collect1, jogged seat | | seated | seated | not run |
| collect1, second seat | | seated | seated | not run |

The gap between the pressed-on-case readings (at most 3.6 mm) and the threshold (4.8 mm) is only 1.2 mm. Real seats read 6.2-9.5 mm: the arm and PC give under a push, about 1 mm by PORT_VISION. Grip reading falls at a pushed seat (0.303 to 0.293) and not on the case in A and A2 (n = 2 each): logged, not used.

Sim, training grasp, measured-gap noise 0.63 mm and lateral 0.39, 0.95 mm (`seat_final_*.json`), a detection counts if within 3 s after the sim's seated hold:

| Set | Not seated: false seats, new [95 %] | Old (6 mm assumed) | Seated: missed within 3 s, new [95 %] | Old |
|---|---|---|---|---|
| Live hand-over, 1024 | 0 of 1 | 0 of 1 | 152 of 1023 = 14.9 % [12.8, 17.2] | 1 of 1023 |
| Plug on the case, 512 | 2 of 142 = 1.4 % [0.4, 5.0] | **138 of 142 = 97 %** | 65 of 370 = 17.6 % [14.0, 21.8] | 0 |

True depth at a new detection: p1 5.6 mm, none under 5 mm in the live set. The cost of 0.5 s at rest: more seats are found late or not at rest. Late ones are caught at the time limit: a plug at depth on the axis counts as seated there, and the press abort does not fire. So a "miss" here is rarely a lost seat; the sim's early ones were the worse fault (dry run s7100, 0.2 s hold: stopped at true 4 mm).

## Why K-J climbs

**Answer: mostly the hand-over target, not the search.** `results/real_align/climb_real.json`, `guard_case*.json`.

1. **Target offset at the hand-over.** During the approach the arm sticks while the target moves on (the servo integrates a net error and the arm does not follow). At t = 0, before any action, the target tip is already above the measured tip: A +22.8, A2 +12.2, ai4 +28.2, ai2 +14.7, ai5 +5.1 mm; dry runs 0.2-4.0 mm. Joint 2 target error up to -44 mrad, against the sim's p1-p99 of -5..+16.
2. **What the policy does.** Mean action on joint 5 is +0.10 to +0.12 in real and in dry runs alike (0.09-0.135), and is cut by the 0.05 rad cap (actions +150 mm up over 5 s, cap -140 mm). This is learned and the same in sim, and alone it does not explain the climb.
3. **Offline input changes** on logged observations (mean up rate about 35 mm/s, first 1 s and later): in-hand input to zero: -1 to -5; encoder offset ±1.5 deg: ±3-4; TCP pose to nominal: -2 to -8; velocity to zero: large in the first second only. None removes it. So it is not the encoder offset and not the in-hand input.
4. **Sim test, plug on the case**, gap 11.5 mm, 5-14 mm high, training grasp, n = 512, up mean at 4 s and seated:

| Condition | Up at 4 s (not seated) | Seated |
|---|---|---|
| Nominal | -7.8 mm: it searches DOWN | 366/512 |
| Encoder offset 0 | -8.0 mm | 400/512 |
| All joint friction x 2-3 | -9.6 mm | 218/512 |
| **Real-like target offset at the hand-over (joint 2 -27, 3 +17, 5 +29 mrad)** | **+4.2 mm, p90 +6.6: it climbs** | **0/512** |

And from aligned live hand-over states, n = 1024: 99.90 % without the offset, 75.6 % with it [72.9, 78.1], seat time p50 0.84 to 4.92 s, up to 17.7 mm up before the seat.

So in this sim the real-like target offset alone reproduces the climb. The friction test applies to the whole arm and lowers the rate, but does not make it climb; real friction is what makes the offset appear. What is not separated: gravity deficit in the sim against the real one. **Fix in this config:** the stage's lead at the hand-over is clipped to K-J's range. Not verified on the arm.

Aligned at 11.5 mm with friction x 2-3: 258/512 seated (507/512 at nominal friction). So K-J in the sim loses much under real-like friction by itself, and aligned does not fix that. (`live_friction` is not a valid friction test: live states restore their own friction values.)

## Insertion guard (`insert.guard`)

Bound 5 mm across and up from the hand-over: at the bound, the target loses its outward lead. At 8 mm (5 + 3): stop. Press abort: no axial progress (0.3 mm) for 3 s with the target at least 3 mm ahead, after 1 s, no seat: stop. Value from the data, measured tip before the seat:

| Set, training grasp | n | Across p99, p99.9, max | Up p99, p99.9, max mm | Without guard | With guard |
|---|---|---|---|---|---|
| Live hand-over | 1024 | 1.2, 1.7, 1.8 | 2.0, 2.9, 3.3 | 1023 = 99.90 % [99.45, 99.98] | 1023 = 99.90 % [99.45, 99.98] |
| Round 3 x 1.5 | 1024 | 2.7, 3.4, 3.6 | 3.3, 3.9, 5.3 | 1015 = 99.12 % [98.34, 99.54] | 1015 = 99.12 % [98.34, 99.54] |
| Gap 11.5, n = 512 | 512 | 2.9, 3.7, 4.1 | 5.0, 5.7, 5.7 | 507 = 99.0 % [97.7, 99.6] | 505 = 98.6 % [97.2, 99.3] |

Guard stops before a seat: 1 of 1024 (live), 2 of 1024 (round 3). It is not free when the hand-over is bad: a plug on the case 5-14 mm high, 366/512 without and 38/512 with the guard (the search down crosses the bound; it stops instead); real-like target offset, 774/1024 without, 1/1024 with. It relies on the stage's alignment and lead clip.

Real logs, open loop (`guard_real.json`): A: clamp at 3.7 s, press stop at 5.46 s. A2: clamp at 2.9 s, bound stop at 4.12 s, at +8.4 mm up (it reached +14). ai4 (seated): clamp from 0.4 s (+6.8 mm at 0.5 s), seat 2 at 4.44 s comes first, press stop would be at 7.64 s. So the guard would have acted in ai4's first second: the logged run does not show how that plays out.

### The reader's fall in `IMG_8784` (A)

The reader fell when the chain opened the jaws, not from the push. Video: fall between 30.6 and 30.7 s; the plug reaches the case about 10 s before. Log: the time limit ends the insertion at 10.15 s after its start, then the jaws open (reading 0.311 to 0.415). The reading stayed at 0.310-0.311 through the whole insertion. PD force estimate in the last second: across -2.8, up +7.4, in 2.1 N; peak 9.5, 10.9, 6.1 N; it ignores friction, so it is not the contact force. Fix (`insert.keep_reader_unless_seated`): no seat means out with closed jaws, 20 mm at 10 mm/s; the reader goes to the operator's hand.

## Offline replay at the real stops (`results/real_align/replay.json`)

With the plug depth of each run, median of the stop frames, across, up, gap:

| Run | Port vision | Label from the collect anchor | `wrist_v18ws` | First command, across, up, gap |
|---|---|---|---|---|
| A | -0.87, +5.41, 8.42 | +1.57, +7.26, 12.04 | -0.06, -1.25, 6.11 | +0.87, -1.0, +0.58 (backs out) |
| A2 | -4.24, +7.46, 8.94 | -3.27, +14.11, 10.61 | -0.17, -0.19, 6.17 | +1.0, -1.0, +0.06 |

The label has a constant offset of about -0.5, -2.3, -1.7 mm (PORT_VISION). A2's frames span several servo poses; PORT_VISION's stop values: A -0.3, +5.0, 8.4, A2 -5.2, +8.9, 8.0. ai4 (2.19, -2.39, 9.55) is similar. ai2 and ai5 (hand-placed) read plug depths 142 and 134 mm: the depth gate refuses them; ai3: no shell, refused. Old net on the same frames: nearly the same value at A and A2, and its step would have been wrong by about 8 mm in up; the scale gate rejects it. Port vision against the collect sweep (n = 97, 88 detected, one depth): slope 1.00, 0.98, 1.03 (all sweep frames under 15 mm gap, tuned on half of them; PORT_VISION has the held-out numbers).

A real-frame check with a fake arm (`deploy/checks/align_realframes.py`): measure mode at A's stop frames, median -0.06, +5.11, 8.45, no step; align with a frozen frame stops at the probes' scale gate; no shell: stops without a step.

## Parameters

`deploy/config/bimanual_align1.json`, defaults in `deploy/bimanual/align.py` (`DEFAULTS`). This config is pick3 + batch A (level roll) + r10 nets + these stages. `--align-mode measure | align | full`, `--align-set KEY=JSON`.

## Checks

- `deploy/checks/align_gates.py`, every gate and the configs: `FAILED: none`. Old configs keep the stages off; `bimanual_align1.json` = pick3 + batch A + r10 nets + the new keys, limits equal.
- `deploy/checks/bimanual_conventions.py --config bimanual_align1.json`: `FAILED: none`. `bimanual_realio_mock`: last line ok.
- `deploy/checks/align_realframes.py`: ok. `align_stickslip`: creep p90 under 1 mm for steps of 0.5 mm and more.
- `deploy/checks/bimanual_timing.py --config bimanual_align1.json` on mason, machine shared with this work's sim jobs: compute p50, p99 ms: align 8.6, 22.2 (2 % over 20 ms; the stage calls IK per step; a late tick during the slow stage does no harm by itself, but the rig decides); insert 13.3, 15.5 (with the guard and seat 2); pick p99 28.9, approach p99 53.8 ms, as older runs of this check on a loaded machine. **Reported, not passed: the rig decides.**
- Vision at 82 ms per frame stops the 50 Hz ticks for that long: there is one control tick between frames, the driver holds the last target.

## Limits

- **Never run on the arm.** Port vision has never run in a live loop. Creep has run only on a sim and a simplified plant. The r10 nets have not run in a loop.
- One PC, one session's lighting, one reader spot.
- Real up tolerance relies on port vision's constant (offset range 1.6 mm on y, 11 frames at two moments). If the constant is off by 1-2 mm, the stage hands over 1-2 mm off and no gate sees it. K-J's real margin for this is unknown. Light-contact anchor in the runbook.
- **Angle is not measured or corrected.** See the one-world result: K-J stalled at about 2 deg in the fair sim. Real angle unknown; real pitch at first seat is reported up to 2-4 deg (`docs/real/INHAND_REAL.md`).
- K-J itself: in the sim, under real-like friction, it loses about half even when aligned. Sim fingers do not yield as real ones do. The stage gets the plug to the place K-J trained from. It does not make K-J work on the real arm.
- The batched eval loses the reader before the stage in about half of its worlds (`reader_lost_before_stage`): those are not counted. Not investigated. It did not appear in the one-world dry runs.
- Seat values come from 2 failed and 2 seated runs. A missed seat leads to a press abort, which pulls the plug out: safe, but a lost try.
- The press clock is unchanged at 60 s; re-press and insertion run under it.
- Stick-slip plant is one joint, fitted by eye.
