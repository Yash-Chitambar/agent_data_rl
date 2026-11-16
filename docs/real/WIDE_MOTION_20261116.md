# Wide PC placement: collision-aware motions and rig runbook (2026-11-16)

Task: make the full chain work over the wide PC box (dx, dy +/-25 mm, dyaw -15..+15 deg from the K7 box centre) without looser limits, for the rig session of 2026-11-17. Worker on mason, GPUs 3 and 4, no rig access. **Nothing here has run on the arm.** No weights changed. No existing config changed. No safety limit changed. Sim numbers are sim only; every number has its n.

Coordinates as in `WIDE_PLACEMENT_READY_20261116.md`: right-base frame, x away from the bases, y toward the left arm, +yaw counter-clockwise from above. dx, dy, dyaw = offsets from (231, 394) mm, 4 deg. The 2026-11-10 spot B is at about dy +4.

## Answer

**Ready for a wide test over most of the box, not all of it.** New motion code (off by default) and two new configs (`bimanual_align12_wide_k7`, `_k4`) move the right hand out of the way before the left press, carry the reader to the port through a high joint-space path, and choose the left press yaw per PC estimate from a collision check of the whole run. No limit is looser.

- Offline whole-chain check (scene meshes, reader, PC, table, SafeArm checks; n = 175 PC poses): **147 pass** (today's motions: 61). The whole square **dx, dy -25..+25 mm passes at dyaw -5..+15 deg** (125 of 125), and at dyaw -10 for dy <= 0. Reader spots +/-40 mm, yaw 70-110 deg: 144 of 162 (the 18 failures are the dyaw -10, dy +25 row).
- Sim full chain (real edge fit; truth-model fit only at dy >= +20, where the sim render loses panel edges): K7 39 runs: **31 of 31 that reached the insertion seated before the release**, 26 after it (4 release artefact, 1 insertion guard stop with the plug at 5.7 mm); 2 align step limits (known sim noise); no guard, torque or SafeArm stop. The 2 runs at dyaw -15 outside the box stopped before the hold as designed. The 4 runs with the reader at its diagonal corners (+/-40, +/-40 mm with yaw 70 / 110) failed the pick (reader estimate 3-11 mm off): a pick limit, not a motion limit. K4: 2 of 3 seated.
- **Not reachable with clearance by any motion of this chain: dyaw <= -10 with the PC toward the left arm, and dyaw -15.** At those poses the right wrist itself reaches the left shoulder at the insertion pose; only a layout change fixes that (section "Needs the operator").
- Rig box for tomorrow: **dx, dy -20..+20 mm, dyaw -5..+12 deg, and down to -8 deg with dy <= 0.** Eight tape poses, corners included, all pass the offline check; 7 of them ran in the dry run (all seated before the release; P0 ran at (0, 0, 0)).

## What was wrong (offline check of today's motions)

The previous grid flagged the problems but the sim does not collide the visual meshes. The new check (`scripts/eval/wide_motion_check.py`) replays the planned motions of the chain on the bimanual scene model with every arm mesh, both grippers, the D405 camera bodies and the webcams, the fingers, the reader in the right hand, the PC case, the table, and SafeArm's own table and joint check. It found three causes:

1. **Hold: the lifted right hand and the left press fingers.** After the pick the right hand waits above the reader at its lift pose while the left arm swings in and presses. The right finger meshes and the left finger meshes meet there: 1 mm at the box centre, -24 mm at dx -25, -14 mm at dyaw -15 (model; real runs on the same model -0.6..+14.6 mm, n = 13). This is the "arm-arm in the hold" flag of the previous grid. The cause is the right-arm wait pose, not the left press point.
2. **Transport: right joint 6.** At negative yaw the Cartesian transport goes through the via pose 40 mm out along the port axis. Right joint 6 loses margin as the plug moves away from the port along a turned axis (0.37 rad at the look gap, 0.23 rad at 70 mm, at dyaw -15). Only one IK branch of the right arm exists at the pre-insert pose (random-seed search, 4000 seeds, 25 poses): no wrist flip is possible, and a 180 deg roll puts the hand under the reader.
3. **Negative yaw toward the left arm: the right wrist reaches the left shoulder.** At dyaw <= -10 with the PC toward the left arm, the right link_4 (wrist) comes within 5 mm of the left link_2 (shoulder, at the left base) and overlaps it by up to 16 mm, at the pre-insert, seat and retreat poses. The left shoulder does not move with the press point: across, along and tool yaw changed this distance by less than 3 mm (64 combinations at four poses). **No motion change fixes this region.** Only a layout change does (see "Needs the operator").

The left wrist camera table margin (+x / -y) is the press TARGET (contact + the force offset dq = Kp^-1 J^T F, which SafeArm also checks), not the measured pose. The tool yaw changes it by up to 6 mm.

## What changed

New code, off by default (absent block = today's code path):

- `deploy/bimanual/wide_motion.py`: config block `wide_motion`; `ClearModel` (scene-model distances and SafeArm checks), `Planner` (the chain's motions as kinematic paths), `select` (hold choice per PC estimate).
- `deploy/bimanual/chain.py` (RealChain), only when the block is on:
  - **right clear pose**: at the end of `stage_pick`, the right hand moves from the lift pose to a fixed TCP point (79, 297, 94) mm in the right-base frame (= the nominal lift + (-60, -100, +40) mm), lift orientation, joint smoothstep, 1.5 s / speed. The transport starts there. A point relative to the lift failed with the reader 40 mm closer to the PC (first version, below).
  - **joint_via transport**: joint smoothstep from the clear pose to a point 60 mm above the via pose (`via_high_mm` 60), peak joint speed <= 0.8 rad/s (duration max(transport_s[0], 1.5 dq / 0.8)), then two Cartesian legs as in eval_chain: down 60 mm to the via pose, which is 30 mm straight above the look pose (`via_back_mm` 0, `via_up_mm` 30), then down 30 mm to the look pose in front of the panel. Keeping the via at the look gap keeps right joint 6 at the margin of the look pose (cause 2).
  - **hold selection**: before the left swing, `wide_motion.select` scores hold candidates (along -0.02, -0.0275, -0.015, -0.035 m x tool yaw -90, -75, -105 deg, in this preference order; the yaws are the set LeftHold searched on every real run) against the planned right path of this run and takes the first that passes every floor. The real path keeps the 50 Hz hold stream alive while it computes (thread). **If no candidate passes, the run stops before the hold** (`GuardStop`, `on_fail: stop`): the chain refuses a PC estimate outside the checked geometry.
  - log events `wide_motion` (config), `wide_right_clear`, `wide_hold_select` (choice, time, the minima of each scored candidate), `wide_transport`.
- `deploy/bimanual/wide_motion.py` refuses hold candidates outside the K7 / K4 training range (along -0.04..-0.015 m, yaw +/-30 deg, across -0.0208 m). Both policies mask every left-arm observation (`mask_left_obs: all`), so the hold choice does not change the policy input.

Floors (planning floors on the scene model, believed angles; not SafeArm limits): arm-arm >= 10 mm in every stage; SafeArm joint-box margin >= 0.30 rad; SafeArm table margin >= 3 mm for all bodies and >= 8 mm for the left wrist_camera (its limit is 30 mm; with 3 mm the dry-run replay with the encoder-offset draw read 1.0 and 2.8 mm at (+25, -25, 0) and (+20, -20, 0)); left arm without the press fingers to the PC case >= -3 mm (real runs -2.8..+1.3); reader to the table >= 15 mm.

New configs:

| Config | Content | Conventions |
|---|---|---|
| `deploy/config/bimanual_align12_wide_k7.json` | `bimanual_align11_wide_k7` + `wide_motion` (enabled, right_clear to (79, 297, 94) mm, joint_via with via_high 60 mm and via 0 / 30 mm, hold_select, on_fail stop, floors above) | FAILED: none |
| `deploy/config/bimanual_align12_wide_k4.json` | the same with K4 (A/B twin) | FAILED: none |

Dry-run-only copies with the truth-model fit (for dy >= +20, where the sim render loses panel-edge inliers): `results/placement/wide_motion/configs/wide12_k7_tm.json`, `wide12_k4_tm.json`.

New scripts: `scripts/eval/wide_motion_check.py` (offline whole-chain check), `scripts/eval/wide_motion_grid.py` (dry-run tables).

## Offline whole-chain check over the full box

`scripts/eval/wide_motion_check.py`, 175 PC poses (dx, dy in -25, -12.5, 0, +12.5, +25 mm x dyaw -15, -10, -5, 0, +5, +10, +15 deg), reader at its nominal spot (131, 448) mm, 92 deg, both motion sets on the same floors. Kinematic: planned paths, believed angles, no encoder-offset error, no dynamics. Results: `results/placement/wide_motion/final_yaw15/check.json` (final configs, wide_motion only) and `final/check.json` (both motion sets, earlier floors: camera 3 mm, yaw +/-30; the same pass map). `final_cam8/` = camera 8 mm with yaw +/-30 (same pass map).

| | today's motions | wide_motion (align12) |
|---|---|---|
| Poses that pass every floor (n = 175) | 61 | **147** |
| First failing node | hold 109, transport 5 | transport 26, release 2 (all at dyaw <= -10) |
| Arm-arm, hold (right waiting, left swing and press), min / median | -38.7 / 0.5 mm | 41.2 / 89.0 mm |
| Arm-arm, transport | -33.1 / 2.9 mm | 10.1 / 25.3 mm (passing poses) |
| Arm-arm, pre-insert to seat | -13.7 / 26.1 mm | 15.0 / 30.6 mm (passing poses) |
| Arm-arm, retreat and home | -16.4 / 38.2 mm | 14.4 / 43.1 mm (passing poses) |
| Right joint margin to the SafeArm box, transport | 0.13 rad (min) | 0.37 rad (passing poses) |
| Left SafeArm table margin, press target (wrist_camera) | 4.7 / 14.7 mm | 8.0 / 15.8 mm (final 8 mm camera floor) |
| Reader to the table, transport | 19.1 mm | 19.1 mm |
| Hold choice | LeftHold over -105, -90, -75 deg: -105 | (-0.02 m, -90 deg) at 139 of 147 poses; -75 deg at 6 (the +x / -y corner, camera floor); along -0.0275 or -0.035 at 3 |
| Selection time (mason CPU) | - | median 3.3 s, max 7.3 s |

Pass map, wide_motion / today (P, p = pass; x, . = fail). Rows dy +25, +12.5, 0, -12.5, -25; in each cell dx -25, -12.5, 0, +12.5, +25:

| dyaw | dy +25 | dy +12.5 | dy 0 | dy -12.5 | dy -25 |
|---|---|---|---|---|---|
| +15 | PPPPP / ..ppp | PPPPP / ..ppp | PPPPP / ..ppp | PPPPP / ..ppp | PPPPP / ..ppp |
| +10 | PPPPP / ..ppp | PPPPP / ...pp | PPPPP / ..ppp | PPPPP / ..ppp | PPPPP / ..ppp |
| +5 | PPPPP / ...pp | PPPPP / ...pp | PPPPP / ...pp | PPPPP / ...pp | PPPPP / ...pp |
| 0 | PPPPP / ...pp | PPPPP / ...pp | PPPPP / ...pp | PPPPP / ...pp | PPPPP / ...pp |
| -5 | PPPPP / ....p | PPPPP / ....p | PPPPP / ....p | PPPPP / ...pp | PPPPP / ...pp |
| -10 | xxxxx / ..... | PPPxx / ..... | PPPPP / ....p | PPPPP / ....p | PPPPP / ....p |
| -15 | xxxxx / ..... | xxxxx / ..... | xxxxx / ..... | xxPxx / ....p | xxPPP / ....p |

Today's motions pass only at dx >= +12.5 (and the floors are stricter than the proven real runs: the box centre itself fails with 0.5-1 mm in the hold, real runs -0.6..+14.6 mm on the same model). Every failure of wide_motion is at dyaw <= -10: the right wrist (link_4) against the left shoulder (left link_2), the right webcam against the left gripper, or right joint 6 (0.22-0.28 rad to the SafeArm box at dx -25 / -12.5).

Reader variants (`final/reader.json`): 27 PC poses (dx, dy in -25, 0, +25; dyaw -10, 0, +15) x 6 reader spots (body +/-40 mm in x and y with yaw 70 / 110 deg, and +/-40 mm in x at 92 deg), n = 162: **144 pass**; the 18 failures are the 3 poses (dyaw -10, dy +25) at every reader spot. The right lift pose for a moved reader is the nominal lift moved rigidly with the reader (IK error <= 0.04 mm); the real pick may choose another grasp tilt.

The tape poses of the runbook (8 poses, `tape/`): all pass (wide_motion); today's motions fail 5 of 8 (hold).

A first version (right clear pose relative to the lift pose, no high via; `final_v0/`) passed 151 of 175 at the nominal reader but failed at 55 of 108 reader-variant poses: with the reader 40 mm closer to the PC the clear pose stayed near the left arm, and with reader yaw 110 deg the joint transport dipped the reader to 13 mm (floor 15 mm) and below the table top. The final settings fix both (fixed clear point, joint move to 60 mm above the via).

## Sim full-chain dry runs (real edge fit)

`scripts/eval/placement_grid.py` (one `deploy.run_bimanual --dry-run --stage full --align-mode full --speed 0.75` per pose, unique `--name` per run), configs `bimanual_align12_wide_k7` / `_k4` with the **real edge fit** (sim panel fix in), except 8 runs at dy >= +20 with the dry-run-only truth-model copies (`_tm`). Seeds 7043 (all) and 7045 (3 poses). Tables: `results/placement/wide_motion/grid_summary.json`, made by `scripts/eval/wide_motion_grid.py`. "Seated before the release" = strict sim seat or true plug depth >= 5.3 mm at the end of the insertion; "after" = strict seat after the release (the sim release is a known artefact, `SIM_RELEASE_PANEL_20261116.md`).

| Region (K7) | n | First failing node | Seated before release | Seated after release |
|---|---|---|---|---|
| yaw 0..+15, centre and +y (incl. 4 truth-model runs at dy +20..+25) | 8 | ok 4, pick 2 (reader corners), align 1 (step limit), insert 1 (guard stop, plug at 5.71 mm, seed 7045) | 5 of 5 | 4 |
| toward the bases (dx -20..-25) | 10 | ok 8, release 1, pick 1 (reader corner) | 9 of 9 | 8 |
| +x / -y corner (dx +20..+25, dy -20..-25, yaw 0..+15) | 6 | ok 6 | 6 of 6 | 6 |
| yaw -5 | 3 | ok 2, release 1 | 3 of 3 | 2 |
| yaw -8..-10 | 8 | ok 5, release 1, pick 1 (reader corner), align 1 (step limit, (+25, -25, -10), final config) | 6 of 6 | 5 |
| yaw -15 | 4 | stop before the hold 2 (planned: outside the box), ok 1 (dy -25), release 1 (dy -25) | 2 of 2 | 1 |

All K7 (n = 39): 33 reached the hold; **31 reached the insertion and 31 of 31 were seated before the release**; 26 seated after the release; 4 "release" (depth 5.56-5.58 mm, not strict after the release); 1 insertion guard stop (`guard_bound`) with the plug truly at 5.71 mm; 2 align step limits at (+25, +25, 0) and (+25, -25, -10) (the known sim align-noise class, "after the press: step limit"). **No guard, torque or SafeArm stop in any run.** Error of the used PC estimate 0.25-2.42 mm on the 33 real-fit runs (n = 40 runs in all, K7 and K4). Align 5.0-18.4 s (n = 36).

Config history inside the grid (all runs use the final motions; only the hold floors changed): the camera floor went 3 -> 8 mm and the yaw candidates +/-30 -> +/-15 deg during the grid. Runs before the change chose (-0.02, -90) everywhere, which the final config also chooses there, except at the +x / -y corner: `k7_pc+25_-25_+0` and `k7_pc+25_-25_-10` (3 mm floor, -90 deg) were rerun as `cam8_*` (-75 and -60 deg) and `yaw15_*` (final config: (-0.0275, -75) at (+25, -25, -10), (-0.02, -75) at (+20, -20, -8)). The -60 deg run is why the yaw set is now +/-15 (see the replay below). Selection 2-10 s; the right clear move and the joint transport ran in every run that passed the pick.

K4 (n = 3, seed 7043): (0, 0, 0) and (-20, -20, 0) seated before and after the release; (+20, -20, +5) align step limit.

Reader at its diagonal corners (body +/-40 mm in x and y together, yaw 70 or 110 deg; n = 4): all 4 failed the pick (closed empty; reader estimate 3.1-11.2 mm off). The previous grid passed 15 of 16 with the reader moved along one axis at a time. This is a pick / reader-net limit, not part of this task; keep the reader within +/-20 mm and 80-100 deg of its nominal spot tomorrow.

Tick replay of the dry runs on the real-run clearance model (`scripts/eval/wide_clearance.py`, true angles with the encoder-offset draw, measured AND target poses; `clearance_replay*.json`): n = 40 runs + 2 (`clearance_replay.json`, `clearance_replay_yaw15.json`). The model of the real-run replay (`WIDE_PLACEMENT_READY_20261116.md`; real runs, n = 13, in brackets):
- arm-arm, measured poses: hold >= 55.1 mm (real -0.6), transport >= 11.5 (11.4, at (+25, -25, -15), outside the box), approach >= 14.2 (42.4, at (0, +12, -10)), align / insert / release >= 18.4 mm at the final-config corner runs. The insert-stage numbers of the replay (to -28 mm) are the TARGET poses of the force push (virtual: 14 N over the PD stiffness); the measured poses of the same runs are 18-24 mm.
- the -60 deg left yaw (`cam8_k7_pc+25_-25_-10`) measured 8.5 mm between the right webcam and the left gripper in the release (yaw -90: 23 mm). It passed the plan; the plan does not see the K7 / align motion. The final yaw set (-105..-75) gives 18.4 mm at (+20, -20, -8).
- left wrist_camera SafeArm table margin (press target, with the sim's +/-1.5 deg encoder-offset draw): 3.2 and 2.7 mm at (+20, -20, -8) and (+25, -25, -10) with the final config (3.3 / 3.9 mm with the 8 mm floor and yaw +/-30; 1.0 / -0.6 mm with the first 3 mm floor and -90 deg; real runs 10.8-17.0). Positive: SafeArm does not stop. The rig has calibrated offsets, so the plan's 8 mm is the better estimate there, but this corner has the least margin: watch the left camera at P3.
- right joint margin to the SafeArm box >= 0.35 rad (real 0.58).

A first version of the grid (right clear point relative to the lift, no high via; `grid_v0/`, n = 11 finished): 9 of 9 that reached the insertion seated before the release; 2 stopped at the fit gate (dy +25).

Default path: `bimanual_align11_wide_k7` (no `wide_motion` block) with the edited `chain.py`, seed 7043: completed, seated before and after the release (`default_check/`).

## New box

Planned box (offline check, all floors pass), from the K7 box centre (231, 394) mm, 4 deg:

- **dyaw -5..+15 deg: dx -25..+25, dy -25..+25 mm** (whole square; 125 of 125 poses at the nominal reader; 72 of 72 reader-variant poses at dyaw 0 / +15).
- **dyaw -10 deg: dy -25..0** (dy +12.5 passes for dx <= 0 only; dy +25 fails at every dx and every reader spot).
- **dyaw -15 deg: not in the box** (passes only at 4 of 25 poses, dy <= -12.5).

Rig box for 2026-11-17 (5 mm and 2-3 deg inside the planned box): **dx, dy -20..+20 mm, dyaw -5..+12 deg; and dyaw down to -8 deg only with dy <= 0** (from B: y <= -4 mm). The old rig box was x 0..+15, y -10..+10, yaw 0..+12.

Perception limits stay as in `WIDE_PLACEMENT_READY_20261116.md`: the sim render loses panel-edge inliers at dy >= +12 (the fit gate stops the run at the capture); on real frames the fit passed at the +9.5 mm map pose. Step A of the runbook measures it.

## Needs the operator

No pose of the rig box needs a looser limit. Nothing is proposed as a default.

1. **dyaw <= -10 with the PC toward the left arm (dy >= +12) cannot be reached with clearance** by any motion of this chain. The pre-insert, seat and retreat poses of the right arm are fixed by the port pose (one IK branch; a 180 deg roll would put the hand under the reader), and at those poses the right wrist reaches the left shoulder: 5 mm at (0, +25, -10), -13 to -16 mm at (*, +25, -15). The left press point and tool yaw change this by < 3 mm (64 combinations at 4 poses). No limit change helps. Options for the operator, none tested: keep the PC out of that corner (today's plan); move the left arm base about 50 mm toward +y or +x; raise the PC by a riser (changes the port height: sim and nets need it too).
2. **dyaw -15 at dx <= -12.5**: right joint 6 is 0.22-0.28 rad from the SafeArm box at the retreat and the pre-insert (0.07-0.13 rad inside the Menagerie range, where the IK clamps). This is inside the limits; the plan floor (0.30 rad) keeps it out of the box. If the operator wants dyaw -15, a shorter retreat (`insert.retreat_m` 0.05 -> 0.03) gains about 0.05 rad; that is a motion change, not a limit change, and it is not in these configs.
3. The planning floors (arm-arm >= 10 mm etc.) are stricter than what ran on the rig (hold -0.6 mm, insert -6.3 mm on the same model, no contact). Loosening them would open more of dyaw -10 / -15; the server does not propose it.

## Rig runbook, 2026-11-17

Give this section to the local agent: "Read `docs/real/WIDE_MOTION_20261116.md`, section Rig runbook, and guide me through it." Branch `real/graph-align`. **Nothing of this has run on the arm.** Operator preference: few runs, one combined config. Expected: 10 minutes by hand, 8 captures (1 minute each), 4 to 6 full runs.

### Rules

- Bases clamped, `estop.py` open, nothing moves without the operator's go.
- One config for the whole session: `bimanual_align12_wide_k7.json` (K7). The K4 twin only for the A/B of step C.
- Do not edit a config on the rig. Do not change a limit.
- Per full run write down: run name, tape pose (measured), outcome, the printed `wide_motion hold:` line, and the operator's band check (band mm, "hangs" or "straight").

### Approvals needed (before step B)

| Item | Change | Data |
|---|---|---|
| A1-A5 | As in `WIDE_PLACEMENT_READY_20261116.md` (edge fit in use, zed_v5 gate and yaw, reader_zed_v2, K7 on the arm, PC off the proven spot) | that doc |
| A6 | **New motions on the arm**: (1) after the pick the right hand moves to a fixed clear point (79, 297, 94) mm with the reader in hand (about 100 mm toward the right base, 40 mm up from the nominal lift); (2) the transport is a joint move to a point 60 mm above the via pose, then two straight descents (60 mm, 30 mm) in front of the panel; (3) the left press point and tool yaw come from the clearance check (in the dry runs mostly along -0.02 m with yaw -90 deg, -75 deg at the +x / -y corner; today LeftHold chose -105 deg from the same yaw set). The joint moves keep the peak joint speed at or under 0.8 rad/s (rated `qd_max` 2.5) | sections "What changed", "Offline check", "Dry runs" |
| A7 | PC poses over the new box (x +/-20, y +/-20 mm, yaw -8..+12 deg; next section) instead of the old rig box (x 0..+15, y -10..+10, yaw 0..+12) | section "New box" |

No limit is looser than `bimanual_align10_all`. Step A (captures) needs no approval.

### Before

```bash
git pull
sha256sum runs/perception/zed_v5s0.pt runs/perception/zed_v5s1.pt runs/perception/zed_v5_gate.pt runs/perception/reader_zed_v2.pt runs/insert_K7.pt
# first 16: e486acdce74b045f 0adaf0b87fc62860 b937820a1fb248d0 e0baaa66df16661b 9a6e23b2f225ccc1
for c in bimanual_align12_wide_k7 bimanual_align12_wide_k4; do
  echo "$c: $(.venv/bin/python -m deploy.checks.bimanual_conventions --config deploy/config/$c.json | tail -1)"; done   # FAILED: none, twice
.venv/bin/python -m deploy.run_bimanual --dry-run --stage full --align-mode full --seed 7043 --config deploy/config/bimanual_align12_wide_k7.json --name dry_wide12
```

The dry run must print `wide_motion hold: ok choice (-0.02, -90.0, -0.0208)` and end seated.

### Tape poses

Mark spot B (the 2026-11-10 spot) as in `WIDE_PLACEMENT_READY_20261116.md`: an L at the left-front case corner and a 150 mm tape line along the port panel, port 2 marked. Move the PC by its port-2 mark with a ruler. **Positions below are from B** (x away from the bases, y toward the left arm). Yaw: keep port 2 on its mark, turn the case, measure the gap between the panel face and the B line 100 mm along the line from port 2: 5 deg = 8.7 mm, 8 deg = 14.1 mm, 12 deg = 21.3 mm. + = counter-clockwise from above (the panel turns toward the right arm); - = clockwise.

| Pose | From B: x, y (mm), yaw | Box offset (dx, dy, dyaw) | Why |
|---|---|---|---|
| P0 | 0, 0, 0 | (0, +4, 0) | proven spot, first K7 run on the arm |
| P1 | 0, 0, -5 (8.7 mm gap, clockwise) | (0, +4, -5) | first negative yaw |
| P2 | -20, 0, 0 (toward the bases) | (-20, +4, 0) | was blocked (arms 19 mm overlap in the model with today's hold) |
| P3 | +20, -24, 0 | (+20, -20, 0) | +x / -y corner (left camera table margin) |
| P4 | -20, -24, -8 (14.1 mm gap, clockwise) | (-20, -20, -8) | corner: bases, right arm, negative yaw |
| P5 | +20, -24, +12 (21.3 mm gap) | (+20, -20, +12) | corner |
| P6 | +20, +16, -5 (8.7 mm gap, clockwise) | (+20, +20, -5) | corner toward the left arm (edge fit at dy +20 untested on the rig) |
| P7 | -20, +16, +12 (21.3 mm gap) | (-20, +20, +12) | corner |

Do not place the PC at yaw beyond -8 deg (clockwise), nor at yaw -6..-8 deg with y more than -4 mm from B (dy > 0): the right wrist reaches the left shoulder there (section "New box"). Keep the reader at its usual spot (within +/-20 mm and 80-100 deg; the sim pick failed 4 of 4 at the diagonal corners of the reader box). Write down each pose as measured.

### Step A: capture only, all eight poses, nearest first (P0, P1, P2, P3, P4, P5, P6, P7; about 1 minute each)

```bash
S='{"port_xy": [<x m>, <y m>], "yaw_deg": <d>, "method": "tape from B", "unc_mm": 1}'
.venv/bin/python -m deploy.run_bimanual --stage capture --config deploy/config/bimanual_align12_wide_k7.json --name wide12_cap_P<n> --pc-spot "$S"
```

- Pass per pose: gate True; `edge fit ... -> ok`; the fit change from P0 equals the tape move within 2 mm and the yaw change within 1 deg; reader estimate ok; no motor or CAN error.
- A reject: answer `r` once. Never `c`. A second reject is a result: write it down, go on.
- Stop step A: a CAN or motor error; the fit misses the tape move by more than 3 mm at two poses (then no step B).
- The capture moves nothing after the capture pose. The hold choice is not printed here (it runs after the pick).

### Step B: full runs (needs A1-A7), at most 6, nearest first, only at poses whose capture passed

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --align-mode full --speed 0.75 --demo --config deploy/config/bimanual_align12_wide_k7.json --name wide12_k7_P<n>
```

Order: **P0, P2, P1, P3, P4**, then P5 or P6 or P7 (one, whichever capture was best).

- Pass per run: estimate = the edge fit, gate True; pick held; **right clear move without contact**; `wide_motion hold: ok choice (...)`; hold latched; transport and approach without a guard stop; aligned; seated; push and unload lines as in `NEXT_RIG_20261115.md`; operator band check.
- Watch: (1) the right clear move (the right hand with the reader moves toward the right base and up, about 1.5 s / 0.75); (2) the left swing to the press with the right hand away; (3) the transport joint move (the hand turns above the PC, then two straight descents in front of the panel); (4) at P2 and P4 the left gripper and the right wrist during the approach (closest pair of the plan).
- Stop step B: any guard or SafeArm stop in the hold, transport or approach; any contact between the arms; `wide_motion: no hold candidate passes` at a pose inside the box (send the run: the estimate is probably off); a plug pulled out; the PC moves; two runs without a seat.
- `wide_motion: no hold candidate passes` is a planned stop before the hold (the arms are still apart): the chain refuses a pose outside the checked box. Answer the home prompt; nothing else to do.

### Step C (only if a K7 run failed in the insertion)

The same pose with `bimanual_align12_wide_k4.json`, only at yaw <= +6 deg (P0, P1, P2, P3, P4 not P5 / P7).

### What the server needs back

All run folders (captures too), the tape poses with their gaps, the band per full run, and every reject or stop line.

## Limits

- Kinematic check: planned paths at believed angles. No encoder-offset error (the rig applies calibrated offsets; the sim draws +/-1.5 deg, worth up to 10 mm at the hand), no tracking error, no K7 motion beyond the axis poses (seat, retreat). The dry runs carry the dynamics, n = 1 per pose.
- The scene meshes overlap by up to 6 mm on proven real runs (finger meshes; `WIDE_PLACEMENT_READY_20261116.md`), so the absolute numbers are conservative by a few mm. The left D405 body is checked as SafeArm checks it (FlexChecker), whether or not it is on the real arm.
- Reader variants use a rigid move of the nominal lift pose. The real pick picks its grasp tilt per reader pose.
- The hold choice is the default point (-0.02 m, -90 deg) at 139 of 147 planned poses. Today's LeftHold picked -105 deg (best joint margin): the left tool yaw changes by 15 deg on every run. K7 and K4 were trained with +/-30 deg yaw DR and masked left observations, so the policy input does not change; the PC press physics changes a little.
- The right clear move and the joint transport were never on the arm. The real arm tracks joint moves well (move_to), but the reader is in the hand: watch it in step B.
- Selection takes 2-10 s on mason's CPU; the rig PC may be slower. The arms hold through it (50 Hz hold stream in a thread).
- Sim numbers are sim only. MuJoCo Warp is not bitwise deterministic; n = 1 per dry-run pose is a code-path and failure-mode check, not a rate.

## Files

- Code: `deploy/bimanual/wide_motion.py` (new), `deploy/bimanual/chain.py` (wide_motion hooks only; the zed_fp hooks in the same file are another worker's)
- Configs: `deploy/config/bimanual_align12_wide_k7.json`, `deploy/config/bimanual_align12_wide_k4.json`
- Scripts: `scripts/eval/wide_motion_check.py`, `scripts/eval/wide_motion_grid.py`
- Results: `results/placement/wide_motion/` (`final/check.json` offline box, `final/reader.json` reader variants, `tape/` the tape poses, `grid/` dry runs, `grid_summary.json`, `configs/` dry-run-only copies)
