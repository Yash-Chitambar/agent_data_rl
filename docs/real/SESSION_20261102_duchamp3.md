# Session 2026-11-02, duchamp3: pick fixed, alignment is the next problem

Branch `real/bimanual-deploy`, config `deploy/config/bimanual_pick3.json`. Run logs: `data/real_runs/2026-11-02/` (gitignored, on mason after the sync). Test plan: [PICK_BATCH_RUNBOOK.md](PICK_BATCH_RUNBOOK.md).

## Result

1. **Pick: batch A (level roll only) held 10 of 10** (Wilson 95 % 72-100 %), reader on one taped spot. Last session: 2 of 8. In `full` and `collect` the pick also held, 3 of 3. Batches B, C, D did not run: A passed.
2. **Skipped with operator approval:** the long run and the opening comparison (runbook step 7). The operator went directly to `full`.
3. **Full chain: 0 of 2 inserted.** Pick, press, transport and approach ran without a stop. The plug stopped too high and the policy climbed past the port. The grip reading stayed steady (0.31) through the insertion, so the grasp did not slip. Operator: the grasp is consistent and flat, and it can go in if the alignment is right.
4. **Real data:** one `collect` run, 150 labeled wrist frames with a really picked reader, from a plug jogged into the port by the operator. Labels have one open problem: see "Label quality".

## Runs

| Time | Run | Command | Outcome | Operator saw |
|---|---|---|---|---|
| 15:55-17:38 | `pick_*_batchA` (10 runs) | `--stage pick --speed 0.5 --batch A` | 10 held | "worked perfectly", all 10 |
| 17:48 | `full_174850_full_A` | `--stage full --speed 0.5 --batch A --name full_A` | timeout, seat depth 3.3-4.1 mm | failed to insert, reader dropped, very close |
| 17:57 | `full_175742_full_A2` | same, `--name full_A2` | false `seated_detected` at 5.76 mm | went too high and kept going up; the grasp is consistent and good |
| 18:21 | `collect_182100_collect1` | `--stage collect --speed 0.5 --batch A --collect-seed 1 --name collect1` | 150 frames, 0 skipped, no stop | inserted twice by jog |

## Batch A


| try | run | open_mm | outcome | attempts | roll_close_deg | rise_to_stall_mm | grip_stall | grip_squeeze | grip_test | grip_end | touch_d_mm | tip_fk_mm | operator | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | collect_182100_collect1 | 24 | held | 1 | 0.182 | 1.26 |  | 0.309 | 0.309 | 0.308 |  |  | seated | both were succefully insereted by jogging |
| 2 | full_174850_full_A | 24 | held | 1 | -0.094 | 1.01 |  | 0.312 | 0.312 | 0.312 |  |  | failed | usbc failed to insert and eventyally droped from gripper was very close though |
| 3 | full_175742_full_A2 | 24 | held | 1 | 0.111 | 1.24 |  | 0.31 | 0.31 | 0.31 |  |  | failed | yeah it tryid to insert it to high and it kept going up, evne if the grasp does not match what the grasp is in sim everytime the robot graps it is a consitent grasp and highquality grasp its flat and definlty can b inserted into usbc if allihged right during allighment phase |
| 4 | pick_155557_batchA | 24 | held | 1 | -0.123 | 1.08 |  | 0.309 | 0.309 | 0.308 |  |  | seated | worked perfectly |
| 5 | pick_170747_batchA | 24 | held | 1 | 0.127 | 1.33 |  | 0.315 | 0.315 | 0.314 |  |  | seated | worked perfectly |
| 6 | pick_171054_batchA | 24 | held | 1 | -0.284 | 1.11 |  | 0.308 | 0.308 | 0.308 |  |  | seated | worked perfectly |
| 7 | pick_171422_batchA | 24 | held | 1 | -0.186 | 1.04 |  | 0.313 | 0.313 | 0.312 |  |  | seated | worked perfectly |
| 8 | pick_172019_batchA | 24 | held | 1 | -0.0549 | 1.47 |  | 0.314 | 0.314 | 0.313 |  |  | seated | worked perfectly |
| 9 | pick_172356_batchA | 24 | held | 1 | -0.226 | 1.13 |  | 0.313 | 0.312 | 0.312 |  |  | seated | worked perfectly |
| 10 | pick_172750_batchA | 24 | held | 1 | -0.101 | 0.923 |  | 0.314 | 0.313 | 0.313 |  |  | seated | worked perfectly |
| 11 | pick_173136_batchA | 24 | held | 1 | -0.256 | 1.21 |  | 0.314 | 0.314 | 0.314 |  |  | seated | worked perfectly |
| 12 | pick_173502_batchA | 24 | held | 1 | -0.423 | 0.753 |  | 0.312 | 0.312 | 0.312 |  |  | seated | worked perfectly |
| 13 | pick_173821_batchA | 24 | held | 1 | -0.224 | 1.28 |  | 0.313 | 0.313 | 0.312 |  |  | seated | worked perfectly |

held 13 of 13 (Wilson 95 % 77-100 %); {'held': 13}

`operator` reads "seated" because the runner's key `s` means held and lifted for a pick. Servo roll at the last frame before the fix: mean -3.21, SD 0.67, range -4.32 to -2.27 deg, in the range that closed empty on 2026-11-01. The servo nets' roll bias is still there: the fix removes it from the command. All tries use one reader spot.

## Why the insertion fails

Same grasp type in A and A2 (in-hand estimate first value +3.69 and +3.84 mm), PC not moved (ZED frames within 0.5 px):

| At the stop | Hand height (FK) | Hand sideways (FK) | `wrist_v18ws` up, across, gap |
|---|---|---|---|
| A | 4.7 mm | 380.2 mm | -1.08, +0.38, 6.31 mm |
| A2 | 11.0 mm | 385.1 mm | +0.21, -0.63, 5.88 mm |

The hand moved 6.3 and 4.9 mm. The net moved 1.3 and 1.0 mm.

Net against real labels (`data/real_training/port_v1/summary.json`; slope = least-squares net output per mm of label; not sensitive to a constant label offset):

| Net | Frames | Slope across | Slope up | Slope gap |
|---|---|---|---|---|
| `wrist_v18ws` (fine) | 88, collect, gap 3-10 mm | 0.34 | 0.17 | 0.35 |
| `wrist_v14w` (coarse) | 150, collect, all gaps | 0.50 | 0.32 | 0.59 |

An exact net has slope 1. The nets see the direction but report a third or less of the real motion on the fine net. Likely causes, not separated: the real in-hand pose (below), the 41.3 mm body against the real 29.6 mm, sim-to-real appearance. ai4 at its stop: real x +3.6, y +1.7, gap 11.2 mm (own seated anchor), net +0.95, -1.3, 5.65 mm, and the real plug was 3-4 deg off.

**Seated detector:** it assumes a 6 mm gap at the hand-over. The real gap was about 11-12 mm. A plug pressing on the case therefore reads about 6 mm "seated". This explains A2 and probably the false seats of 2026-11-01.

## Label quality (read this before training)

`deploy/bimanual/collect.py` (header) gives the method: encoder FK displacement from the pose where the plug was jogged into the port. `scripts/real/build_port_dataset.py` builds the data set.

**The recheck failed its 1 mm limit.** After the sweep the operator jogged the plug in again. The labels read x -0.38, y **-4.49**, z -4.88 mm (expected 0, 0, -5.8), rotation -1.0, +0.7, +0.6 deg. So the two seated poses differ by 4.5 mm in height.

- The PC did not move: ZED shift at most 0.5 px from the approach to the end.
- The sweep itself looks sound: the plug and reader body do not move in the wrist frames from the approach stop to the last sample, and the grip reading is steady (0.293 to 0.290).
- Under the push the reader moves in the jaws: the grip reading falls during both seats (0.303 to 0.293 first, 0.290 to 0.278 second), and the reader body looks pitched in the frame at the second seat.
- At the 3 mm stand-off of the first anchor, the plug looks about 3 mm above the port row in the wrist frame (rough, by eye).

**How to read it:** differences between frames of the sweep are good (dry run against sim truth: 0.3-0.65 mm near the port, up to about 1.8 mm at long range). The constant height offset of the run's labels is **not known within about 4.5 mm**; the first anchor is probably the high one, because the reader pitches when pressed in. Do not use absolute y of this run until that offset is fixed. Across and gap are less affected (0.4 and 0.9 mm in the recheck). Fix: one scalar per run, from the port and plug edges in the frames, using the calibrated wrist camera; or re-collect with the anchor taken at light contact (plug entered 1-2 mm, no push).

Sources in the manifest: `collect` 150, `same_spot` 102 (A and A2 frames with the collect anchor; add about 1 mm of grasp difference), `self` 53 (ai4, own seated anchor, hand-placed grasp), `unlabeled` 283 (ai1, ai2, ai3, ai5: hand-placed grasps without a seated pose).

## Server work

1. **Sim in-hand pose (operator).** The real grasp is consistent and good but differs from the sim: in the sim the reader lines up exactly with the bottom of the gripper; on the robot its butt angles out of the gripper by less than 10 deg, and it still looks straight. Images and video come from the operator. Real in-hand estimates on picked grasps: first value +3.35 to +3.84 mm, third -1.40 to -2.54 mm.
2. **Fix the wrist nets on real frames.** Correct the sim in-hand pose and the 29.6 mm body, retrain, and fine-tune on this data. Acceptance on real frames: slope near 1 on across, up and gap on `collect`; A and A2 report the 6 mm height difference.
3. **Train over the real stop:** the real stop gap is 11-12 mm while the fine net trained on 3-10 mm.
4. **Port edge detection (operator).** Add a check from the wrist camera alongside the GaP policy: find the USB-C port edges in the image so the exact port position is known before insertion. It does not depend on the net. The labeled frames can test it, and it can also fix the label offset above.
5. **Seated detector:** do not assume a 6 mm gap; see above.
6. **Insertion policy search:** it climbs in every run (ai4 too). Starting too high, it goes past the port and onto the case (A2: tip about +12 mm).
7. **Servo roll bias:** the real servo nets still ask for about -3 deg; level roll hides this.

## Code in this session

`deploy/bimanual/collect.py` (stage `collect`), `deploy/run_bimanual.py` (`--stage collect`, `--collect-n`, `--collect-seed`), `scripts/real/build_port_dataset.py`. Limits and configs did not change. In `collect` the left press lasts about 1 s, then its force goes to zero with the hand left on the PC; the operator braces the PC during the jogs.
