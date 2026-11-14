# Force-push spring-back and the depth label (server task, 2026-11-14)

Task: the force push of 2026-11-13 springs back; find the mechanism in the logs, give a fix (new config, off by default),
and re-fit the FULL/SHORT label. Data (read-only): `data/real_runs/2026-11-13/full_165400_all9_full_1` (10 N,
`bimanual_align9_all`) and `full_171043_all9b_full_1` (14 N + centring, `bimanual_align9_all_b`). For comparison: the
position pushes of 2026-11-10/11 (8 runs with a push, including the re-sent `k4_4`, `k4_5`, `k4_6`).
**Nothing here has run on the arm. The force-push numbers are n = 2 runs.** Sim numbers are sim only.

Tools (new): `scripts/real/springback_timeline.py` (per-tick timeline from the seat to the release; JSON
`results/real_align/depth/springback_all.json`, CSV per run in `results/real_align/depth/springback/`).
`scripts/real/depth_analysis.py` (existing) for the rest depths.

## 1. Timeline of the push (n = 2 force pushes)

Hand = right TCP from encoder FK (believed angles), port axes from the ZED yaw (as the chain). Advance is along the port
axis since the push start (after K4's 0.2 s seat hold). `F_pd` = PD force `kp (target - q)` through `J^-T` along the axis
(what the controller asks). `F_eff` = the same from the measured joint torques minus their seat-hold mean (gravity).
20 ms ticks.

| Phase | 10 N (`all9_full_1`) | 14 N (`all9b_full_1`) |
|---|---|---|
| Push start (after the seat hold) | tick 2790, advance 0 | tick 2867, advance 0 |
| Load ramp (3 N/s at speed 0.75) | advance follows the force: 0.29 mm/N (linear fit, 2 N to full) | centring up 2.6 mm first, then 0.25 mm/N |
| Push end, under load | 2.60 mm, `F_pd` 10.0 N, `F_eff` 9.9 N, PD lead 82 mrad | 4.06 mm, up +2.41 mm, `F_pd` 13.9 N, `F_eff` 14.3 N, PD lead 117 mrad |
| Drop, tick +1 (load cut in one tick: target = measured pose) | 2.34 mm (`F_pd` 0.1, `F_eff` 5.7 N) | 3.28 mm (`F_pd` 0.5, `F_eff` 6.0 N) |
| Drop, tick +2 (40 ms) | **0.97 mm** | **1.66 mm** |
| Drop, tick +4 (80 ms) | 0.53 mm | 1.05 mm |
| Drop end (0.3 s), jaws closed | **0.67 mm** (back 1.93) | **0.68 mm** (back 3.37) |
| Release start: jaws open (0.4 s) and the left press is released in the same tick | left hand jumps 2.3-2.7 mm in 2 ticks; right hand sinks 1.6 mm down, 1.0 mm back, PD lead grows to 11 mrad (the plug no longer carries part of the arm) | left 2.5-2.8 mm in 2 ticks; right hand 1.7 mm down, 0.4 mm back, lead 13 mrad |
| Jaw reading | 0.3126 closed until the release; first change tick +1 of the release; full open (23.5 mm) after 0.72 s | 0.3123 -> **0.3025 during the push** (the jaws close 0.4 mm more under the load; limit 0.02); full open after 0.74 s |
| Left hand (press) during push and drop | <= 0.16 mm | <= 0.20 mm |
| Retreat starts | 0.4 s after the release start, jaws at 22.8 mm (still opening, 4.1 mm wider than the grasp) | same, jaws at 22.7 mm |

Answers:
- **When does the hand come back?** In the first 40-80 ms of the 0.3 s "drop the push" (`final_push_force`,
  target = the measured pose each tick). There is no unload ramp: the load goes from 10-14 N to about 0 in one tick. The
  hand comes back before the mode switch to the release and before the jaws open (at the jaw opening it is already at
  0.67 / 0.68 mm). The jaw opening adds 0.4-1.0 mm of "back" and 1.6-1.7 mm of "down", but that is the hand sagging
  after the plug stops holding it, with the jaws already wider than the reader.
- **Is it the arm, the left press or the port?** Not the left press (its hand moves <= 0.2 mm until the release). With
  the target equal to the measured pose, the controller asks for no force: the arm does not pull back. The hand is
  pushed back by energy stored in elastic parts during the push. Load and unload are both close to linear:
  0.25-0.29 mm/N on the way up, 0.19-0.24 mm/N on the way down, and the hand returns to within 0.7 mm of its start.
  The PD alone is far softer than this (axial PD stiffness at the seat pose 0.38 N/mm, from `kp` and `J`), so the
  4-5 N/mm part is the arm drive train, links, wrist, fingers and the reader in the jaws, or the port side.
- **The same spring-back is on every position push**, also on the four runs the operator called fully in: 1.39-1.53 mm
  back at 8.9-10.4 N (n = 7 pushes at >= 8.9 N; `demo_4` 1.53, `k4_3` 1.53, `fastmove_1` 1.44, `k4_5` 1.45, all
  "fully in" or "flush"). So a spring-back of this size does not by itself leave a 2 mm band. The force push springs back
  more (1.93 mm at 10 N, 3.37 mm at 14 N) because it loads only the axis, and the hand rests at the same place (0.67,
  0.68 mm; position pushes 0.56-1.59 mm).
- **Does the plug come back with the hand?** The logs cannot show it: FK sees the hand, not the plug. What they show:
  (a) the hand's load curve is linear from 2 N up, with no step where a plug would slide or a detent would give;
  (b) by hand (session step 0), a plug 1-3 mm out stays where it is (it hangs): the port does not push a plug out;
  (c) the right wrist camera moves only about 0.8 px per mm along the port axis (case-slot row: +2 px over the 2.6 mm
  push, -2 px at the drop): too coarse to see the plug. The most likely reading: **the push loads the arm and the grasp,
  and the plug does not travel the last ~2 mm at 10-14 N.** The spring-back is then the elastic chain relaxing, and it
  moves the hand, not the plug. Confidence: moderate (n = 2 force pushes, n = 7 position pushes for comparison, no
  plug measurement).
- **Does the retreat drag the plug?** Not likely: the retreat starts with the jaws 4.1 mm wider than the grasp and the
  last wrist frames before the release already show the 2 mm band. The release order is the same as on the fully-in
  runs. One real order issue: `retreat_right` sets the left target to its measured pose in the first release tick, so
  the PC press is removed in the same tick the right jaws start to open (left hand +2.3-2.8 mm in 40 ms). The PC is
  unclamped while the right jaws still touch the reader.
- Options from the operator, checked against these numbers (and the dry runs of section 3):
  - "Re-target the position hold to the loaded pose": no effect. The PD is 0.38 N/mm against 4-5 N/mm of elastic
    chain, so a stiff hold at the loaded pose keeps only about 8 % of the deflection; the hand still comes back about
    92 %.
  - "Ramp the load down slowly": removes the 40 ms jerk; the hand still ends near 0.7 mm. It tests whether the speed
    of the unload matters.
  - "Keep the load while the jaws open": keeps the plug loaded until the fingers let go. With the full 14 N this is a
    force source that loses its resistance when the jaws release (it moves fast; the 4 mm advance watch is already
    almost used at 4.06 mm). In the sim even 3 N kept on made the hand jump 1.8-2.5 mm when the jaws let go
    (section 3). Not in the config.

## 2. The change (code off by default, new config)

`deploy/bimanual/insert.py`:
- `final_push_force`: the per-tick load law is now a local function `load_step` (same arithmetic as before, used by the
  push loop). New optional key `insert.final_push.unload` (absent = the 0.3 s drop as before) and `depth_report`.
- `unload_force` (new, runs only with `unload`): after the push, with the left press target kept:
  1. ramp the axial load from the reached force down to `open_force_n` at `ramp_n_s` 4 N/s x `--speed` (the
     approved up-ramp rate), jaws closed, the same load law (axial + lateral spring + rotation spring);
  2. open the right jaws for `open_s` 0.6 s / `--speed`: with `open_force_n` > 0 the load law keeps that force on; with
     `open_force_n` 0 (the config) the right target is held fixed at the pose reached at 0 N (stiff position hold);
  3. with `open_force_n` > 0 only: relax the rest to 0 over `relax_s` (jaws open); then hold the measured pose 0.2 s;
  4. then the normal release (`retreat_right`: the jaws are already open; retreat; left unload).
  So in the config the load goes 14 -> 0 N in 4.7 s at speed 0.75 instead of one tick, the jaws open only at 0 N, and
  the left press stays on until the right jaws are open (the base removes it in the same tick the jaws start to open).
  Watches every tick: hand advance since the push start > `max_advance_mm` (4), left hand > `max_left_mm` (2), hand off
  the push line > `max_lat_err_mm` (4) (all as the push); gripper readings > `max_grip_change` (0.02): both jaws in step
  1, the left jaw in steps 2-3 (the right jaws open on purpose there); new: the hand must not advance more than
  `open_max_adv_mm` 1.0 mm beyond its pose at the start of the opening. A stop holds the measured pose (no load).
  Event `insert_unload` (ticks of each step, hand advance at each step, spring-back slope in mm/N, sim truth depth).
- `run_insertion`: passes the depth at the push start to `final_push_force` when `depth_report` is set.
- With `depth_report`, the run prints `depth (seat-2 units, FK of the hand, not the plug): push start .., loaded ..,
  jaws closed at 3.0 N .. mm. FULL/SHORT: NOT VERIFIED by FK ...` and logs `insert_depth_report`.

New config `deploy/config/bimanual_align9_all_b_unload.json` = `bimanual_align9_all_b` + `final_push.unload` +
`depth_report` + `full_depth_mm: null` (`unload.open_force_n` 0). Nothing else differs. Why `all9_b` and not `all9`: both are approved; 14 N
stores the largest deflection (3.4 mm), so a change in the band is easiest to see, and the 4.5 mm lateral gate with
centring avoids the skipped pushes of the 2.5 mm gate (2 of 8 K4 seats on 2026-11-11). Limits: force 14 N, centring
4.5 mm, lateral gate 4.5 mm, `tau_max_nm` as approved on 2026-11-13; no limit is looser. A residual load with the jaws
open (the operator's "keep the load while the jaws open") is in the code (`open_force_n` > 0) but **not in the config**:
the dry runs below show it is a fast force source.

Conventions check: `FAILED: none`.

## 3. Dry runs (SimIO, K4, `--speed 0.75`, 2026-11-14; they check the code path, not a rate)

`deploy.run_bimanual --dry-run --stage full --align-mode full --speed 0.75 --seed N --config C`, GPU 1. "v1" = the first
version of the new config with `open_force_n` 3 N (replaced); "new" = the config as delivered (`open_force_n` 0).
`SIM_MATE_FORCE_N` = the optional sim mating spring (`io_sim.py`). True depths are sim truth (sim seat 5.8 mm).

| Config | Seed | Mate | Push | Unload | True depth: loaded / low load, jaws closed / end of insert | Seated after release |
|---|---|---|---|---|---|---|
| base `all9_b` | 7043 | 0 | 14 N, adv 0.86, rest 0.85 mm | 0.3 s drop | 5.77 / - / 5.72 | yes |
| base `all9_b` | 7044 | 0 | 14 N, centred (0.47, -2.10), adv 0.52, rest 0.34 | 0.3 s drop | 5.80 / - / 5.62 | **no** (known sim release effect) |
| base `all9_b` | 7043 | 4 N | skipped: seat-2 depth 4.97 < 5.0 | - | - / - / 2.71 (spring) | no |
| v1 (3 N at the opening) | 7043 | 0 | 14 N, adv 0.85 | 14 -> 3 N in 3.68 s; **jaws open: hand +2.47 mm, watch fired at 1.64 mm (one tick late)** | 5.80 / 5.80 / 5.66 | yes |
| v1 (3 N at the opening) | 7044 | 0 | 14 N, centred (0.25, -1.54), adv 0.61 | 14 -> 3 N; **jaws open: hand +1.83 mm, watch fired at 1.06 mm** | 5.79 / 5.77 / 5.64 | yes |
| new | 7043 | 0 | 14 N, centred (-0.10, 1.59), adv 0.67 | 14 -> 0 N in 4.68 s, no stop; jaws open: hand -1.17 mm (sags) | 5.78 / 5.75 / 5.65 | yes |
| new | 7044 | 0 | 14 N, centred (0.62, -2.02), adv 0.51, off line 2.38 mm | 14 -> 0 N in 4.68 s, no stop; jaws open: hand -1.16 mm | 5.79 / 5.76 / 5.67 | yes |
| new | 7044 | 0 | (first try: sim align stop "step limit" after 21 cycles, no insertion; not counted) | | | |
| new | 7043 | 4 N | 14 N, centred (-0.51, 1.42), adv 0.62, off line 2.28 | 14 -> 0 N, no stop; jaws open at 0 N | 5.73 / **5.70** / -84 (plug pushed out when the jaws opened) | no |

- The new path runs without a stop on both seeds (n = 2 per seed setting). Push watches and the unload watches stay
  quiet; the hand stays within 2.4 mm of its line (limit 4).
- **The sim cannot show the rig spring-back.** The sim arm and grasp are stiff: the hand moved 0.02-0.04 mm per N
  during the ramp-down (rig: 0.19-0.24 mm/N), and the base drop gave back 0.01-0.18 mm (rig 1.9-3.4 mm). So the sim
  does not test the mechanism; it tests the code path.
- With the 4 N mating spring the plug stays at 5.70 mm while the load ramps to 0 with the jaws closed, then the spring
  pushes it out as soon as the jaws open (true depth -84 mm: the reader fell). The spring has no friction hold (FULL_DEPTH
  section 5), so "stays in after the release" cannot be checked with it; it would fail the base the same way after its
  release. It does show that a slow unload with closed jaws keeps the plug where it is.
- The 3 N residual at the jaw opening (operator option "keep the load on while the jaws open", at a reduced force) made
  the hand jump 1.8-2.5 mm in the sim, and the 1.0 mm watch fired one 20 ms tick late. On the rig the stored elastic
  deflection would add to this. So the config opens the jaws at 0 N with the arm held still.
- 7044 base "not seated after release" vs new "seated" is the known sim release variation (7041/7042 also), not an
  effect of the change (n = 1 each).

## 4. The depth label

Rest depth = FK advance since the hand-over minus the vision gap (seat-2 units), at the last tick before the release
(`depth_analysis.py`). Operator labels, n = 11 insertions:

| Label | Runs (rest depth, mm) |
|---|---|
| fully in / flush (n = 4) | `demo_4` 7.56, `k4_5` 7.59 (re-sent folder, "flush and seated"), `k4_3` 7.36, `fastmove_1` 7.32 |
| short: band, hangs or "mm short" (n = 7) | `all9b_full_1` **7.33** (2 mm band), `demo_1` 6.94 (hangs slightly), `all9_full_1` 6.75 (2 mm band), `k4_1` 6.13 (~3 mm), `kjrc_1` 6.00 (hangs), `kjrc_3_push` 5.77 (~3 mm), `kjrc_2` 5.58 (~1 mm) |
| "seated", no depth word (not used) | `k4_4` 7.93, `k4_6` 6.98, `demo_2` 6.69, `demo_3` 6.65, `speed075_1` 5.90 |

- **No FK threshold separates full from short.** The full range is 7.32-7.59 mm, and a short run sits at 7.33 mm. Any
  threshold that passes `fastmove_1` (7.32) passes `all9b_full_1` (7.33). The rest depth also carries the vision gap
  of the hand-over (5.68 mm on `all9b_full_1`, the smallest of the 11), which FULL_DEPTH section 1 found 1.5-2 mm off
  with an unsettled cause; a 1 mm gap error moves the depth by 1 mm.
- Other FK signals checked: the push gain at rest (rest after the push minus rest after the K4 hold): full 0.56-1.17,
  short force pushes 0.54 and 0.72 mm: no separation. Spring-back size: same on full runs (1.44-1.53 mm). The hand
  advance after the jaws open: meaningless for the plug (the hand is free and sags).
- In the last right wrist frame before the release, the silver band looks thicker on the two force-push runs and on
  `k4_1` than on the four full runs (by eye). A rough automatic band measure (this task, not kept) did not separate
  them. A vision label needs the port-vision code, which another worker owns: not done here.
- In the new config the FULL/SHORT label is off (`full_depth_mm: null`). The run prints the FK depths with "NOT
  VERIFIED", and the operator's band check is the label. Existing configs are unchanged (they still print FULL/SHORT
  from 7.2 mm; read that line as not verified).

## 5. Rig runbook (3 runs, `bimanual_align9_all_b_unload`)

Rules as always: bases clamped, `estop.py` open, one go per run. Same PC and reader spots.

Approval needed before run 1: **the new unload motion** (load ramps 14 -> 0 N in about 4.7 s with the jaws closed, then
the jaws open with the right arm held still and the left press kept until the jaws are open). No limit is looser than
the approved `all9_b`.

```bash
git pull   # real/graph-align
C=deploy/config/bimanual_align9_all_b_unload.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C | tail -1        # FAILED: none
.venv/bin/python -m deploy.run_bimanual --dry-run --stage full --align-mode full --seed 7043 --config $C --name dry_unload
#   prints "final push (force 14.0 of 14.0 N) ...", "unload: load 14.0 -> 0.0 N ...", "depth (seat-2 units ...) NOT VERIFIED"
.venv/bin/python -m deploy.run_bimanual --stage full --config $C --align-mode full --speed 0.75 --demo --name unload_1   # then _2, _3
```

Per run, report: the three printed lines (`final push`, `unload`, `depth`), the operator's band (mm) and "hangs or
straight", and whether the reader moved when the jaws opened.

- Pass (per run): push reaches 14 N, no watch stop; `unload` line: hand back during the ramp about 0.19-0.24 mm/N
  (2.7-3.4 mm for 14 N; this is the slow version of the 2026-11-13 spring-back), `jaws opened ... hand moved` within
  +/- 1.0 mm; the operator sees **no band**. Also note the band before the jaws open (the wrist stream shows it).
- Read-out after 3 runs:
  - band gone on 3 of 3: the unload was the cause; keep the config for the depth block.
  - band still about 2 mm with the slow unload: the unload is not the cause (the plug did not go deeper under the
    load). Stop changing the unload. Next candidates (each needs its own approval): K6d (trained with a mating force),
    or a push that is not along the measured axis only (the four full runs all had a 15 mm position push that also
    drove the hand 4-9 mm down).
- Stop the block: any watch stop in the unload twice; `open_move` beyond 1.0 mm (the hand jumps when the jaws open);
  the reader turns or moves when the jaws open; the PC moves; anything looks wrong to the operator.
- Not in this config: a residual load at the jaw opening (`open_force_n` > 0). In the sim 3 N made the hand jump
  1.8-2.5 mm when the jaws let go. Do not use it on the rig without a new server check and an approval.
