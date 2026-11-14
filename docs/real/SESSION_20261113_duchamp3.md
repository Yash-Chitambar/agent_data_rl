# Session 2026-11-13, duchamp3: runbook NEXT_RIG_20261114 (short session)

Branch `real/graph-align` at `fa7aca62`. Run logs `data/real_runs/2026-11-13/` (not in git).
Operator plan: "This session in short": combined config `bimanual_align9_all` first, single-change blocks (A-D) only if
something fails; no repeat of a passing run; C3 (K6d) and G skipped. Same PC spot and reader spot as 2026-11-11.
Note: the `_about` of `bimanual_align9_all.json` says to run it only after each part passed alone; the runbook's operator
decision (combined first) replaces that for this session.

## Setup and no-motion checks

- K4 `runs/insert_K4.pt` sha256 `4d288f234ff1af10` (matches).
- Conventions `FAILED: none` on all 7 configs: align8_nojump, align8_nojump_retry2, align8_fastalign, align8_depth_a,
  align8_depth_b, align8_netbias, align9_all. `deploy.checks.align_gates`: `FAILED: none`.
- Dry run `full_161703_dry_s7041_dry_all9` (align9_all, seed 7041): completed, no stop. `approach_nojump` logged, 2
  `approach_deadband` events; pick held (1 attempt); align 7.2 s, 11 cycles, hand-over gap 6.70 mm, lateral (0.15, 0.38) mm;
  insert `press_abort` (sim), true seated at the end (5.69 mm), so the force push did not run (no `seated_detected`).
  Same seed with the base config on 2026-11-11 also ended in `press_abort`.
- Dry run `full_161810_dry_s7043_dry_all9_b` (seed 7043, to exercise the force push): completed, K4 `seated_detected` at
  2.62 s, true seated after release. Printed `final push (force 10.0 of 10.0 N): advance 0.62 mm, at rest 0.57 mm, left
  hand 0.21 mm, off line 0.32 mm, grip change [0, 0]` and `depth at rest 6.45 mm (fully in from 7.2 mm): SHORT` (the
  7.2 mm FULL threshold is in rig depth units, where the real port mouth sits about 1.5-2 mm deeper than in the sim:
  FULL_DEPTH section 1. The sim label is not a rig prediction; the run checks the code path).

- Dry run `full_165923_dry_s7043_dry_depth_b` (`bimanual_align8_depth_b`, seed 7043, before its first rig run): completed, K4 seated at 1.76 s, `final push (force 14.0 of 14.0 N): advance 0.57 mm, at rest 0.66 mm, centred [-0.1, 0.47] mm, left hand 0.23 mm, off line 0.32 mm, grip change [0, 0]`, true seated after release. The centring path ran without a stop.

- New config `deploy/config/bimanual_align9_all_b.json` (operator approved, option a after `depth_b_1`): `bimanual_align9_all` + `insert.final_push` force_n 14, centre_max_mm 4.5, centre_v_mm_s 2, max_lat_mm 4.5 (the depth_b keys; nothing else differs). Conventions `FAILED: none`. Dry run `full_170907_dry_s7043_dry_all9_b_cfg` (seed 7043): completed, K4 seated at 1.74 s, `final push (force 14.0 of 14.0 N): advance 0.52 mm, at rest 0.49 mm, centred [-0.25, -0.49] mm, left hand 0.28 mm, off line 0.50 mm`; sim true seated at the insert end, **not seated after the release** (the same seed with depth_b was seated after the release; the base config showed the same sim variation on 2026-11-11). Code path checked; not a rate.

## Step 0: depth reference by hand (no robot)

Operator, reader pushed into port 2 by hand:
- Fully in: **no silver band (0 mm), no gap**.
- 0.5 mm out: straight.
- 1 mm out: hanging. 2 mm out: hanging.
- 3 mm out: hanging so far that it touches the table ("touching ground").
So on the rig any visible band (>= ~1 mm) means short, and a hanging reader means at least 1 mm short.
No force gauge reading.

## Approvals

- 16:4x: operator confirmed bases clamped, `estop.py` open, and started run 1 with `bimanual_align9_all` after being told it uses D (wrist-net offset [-0.3, 2.6, 0.8] mm before the disagreement check): logged as **D approved**. Operator on the approvals: "not entirely sure, I think so since we are getting successful insertions".
- **C1 (10 N force push) approved** by the operator ("C1 is yes"), during run 1.
- 17:0x, after `all9_full_1` was short (band about 2 mm): operator chose to go straight to C2 (`bimanual_align8_depth_b`), skipping C1 alone ("lets go straight to b", after being told it needs a yes to both changes): **C2 approved: push force 10 -> 14 N; push lateral gate 2.5 -> 4.5 mm with centring.** Reason given to the operator: C1 alone runs the same 10 N push as `all9_full_1`.

- After the depth step: operator "sounds [good], lets get to it" to the E1 plan (approval E: moving the PC up to 2 cm / 15 deg for capture-only mapping): logged as **E approved**.

## Runs

| Time | Run | Block / config | Outcome | Operator: real depth (silver band, mm) | Logs |
|---|---|---|---|---|---|
| 16:33 | `full_163318_all9_align_1` | run 1, `bimanual_align9_all`, `--align-mode align` | **no motion: camera open failed** (`AssertionError: left wrist camera did not open`, `cameras.py:80`, in `RealIO._open_cameras` before the motors take commands) | - | The left wrist camera is connected and streams (frame checked: the left wrist view of the table, PC and fingers), but it re-enumerated at 14:38 on another USB port, `pci-0000:13:00.0-usb-0:4` (`/dev/video0`), not the `pci-0000:16:00.3-usb-0:2` (USB 5-2) of `scripts/real/rig.py` `WRIST_CAM`. Nothing is on 5-2 now. A ZED 2i (not the rig's ZED 2, S/N 20120598) was plugged on `usb-0:5` at 14:39. Same intermittent left-camera port fault as 2026-11-10. C1 / D approvals still open at this start (operator started the command). |
| 16:37 | `full_163740_all9_align_1` | run 1 again, `bimanual_align9_all`, `--align-mode align` | **no motion: software stop before the stage graph** (`KeyError: 'mm'`, `run_bimanual.py` `node_policies`, the `--demo` viewer label read the position-push key; the force push has none). Arms connected and held (PD hold from connect), operator typed SUPPORTED, torques off. | - | Fix (label only): `node_policies` shows `+ 10 N push` for `final_push.mode` force, `+ 15 mm push` otherwise; checked on align9_all, depth_a, the base and three align8 configs. The insert code already dispatches on the mode (`insert.py` line 415); `live.py` reads event fields with `.get`. Not seen in the dry runs: they do not use `--demo`. |
| 16:39 | `full_163924_all9_align_1` | run 1, `bimanual_align9_all` (D approved), `--align-mode align` | **align stop: `no measurement (plug depth 102.9 mm from its width: another grasp or not the shell), 5 times`** after 7 cycles, 5.2 s; back-off 15 mm, release, home. No insertion (as planned for run 1). | - (no insertion) | Capture PC (230.8, 395.8) mm, yaw 4.4 deg, gate ok; reader (133.3, 453.1) mm, yaw 87.2 deg. Pick held (1 attempt), hold latched. **Approach: `approach_nojump` logged, 5 iterations, `approach_deadband` at iterations 3-4; at the stop max uncommanded move 2.86 mm (<= 3: pass)**, joint 2 error -17.6 mrad; `approach_iters.py` also reports 8.14 mm at tick 1765, 232 ticks before align (the end of the coarse ramp, which the runbook says not to read as the jump). Align: prepare plug_z 119.2 mm accepted; cycle 1 (2.35, -0.94, 18.1) mm, step reached in 1.0 s; cycle 2 (0.93, -0.60, 16.6), 3 mm gap step, timed out at 2.0 s (moved 3.8 mm); cycles 3-7: every frame rejected. **Cause (offline, `port_vision.measure` on the saved frames): the shell's left edge stays at u 619 px and its end row at v 534 px in every frame, but the detected right edge jumps from 666 to 673-675 px after the step, so the width reads 54-57 px (46-47 before; fastmove_1: 45.7-46.4 px over its whole align) and the depth 100-104 mm (gate 108-145). The crop shows a bright smear (likely the port lip or a reflection) touching the right side of the shell.** The plug did not move in the jaws. Here the plug entered align higher than on the base runs (first align measurement up -0.9 mm at 18 mm; fastmove_1: -5.2 mm; the no-jump deadband leaves up alone within 3 mm), which may put that smear next to the shell at a 13-16 mm gap. Detector issue for the server; not a config limit. |
| 16:45 | `full_164514_fastalign_align_1` | block B alone, `bimanual_align8_fastalign`, `--align-mode align` (runbook: align stop -> B alone) | **B pass: aligned in 13.14 s, 8 cycles** (align stage wall time 17.6 s incl. the back-off; was 33-62 s), no probe cycles, final (-0.25, +0.03, 6.33) mm; steps 6, of them 3 ended at the new 2.0 s timeout; back-off 15 mm, release, home. No insertion (align mode). | - (no insertion) | Capture PC (230.9, 395.9) mm, yaw 4.3 deg; reader (132.9, 454.9) mm, yaw 91.7 deg. Pick held (1 attempt), hold latched. **Approach (old, no `approach_nojump`): uncommanded move at the stop 5.88 mm (> 3)**, joint 2 error up to 37.6 mrad, 11 iterations: the known approach jump that block A removes (run 1: 2.86 mm). Align crossed gaps 14.9 -> 11.1 mm with across 1.7 -> 0.7, up 0.0-0.2 mm: port vision shell width 46.5-47.1 px in every frame (left edge 619-620, right edge 666-667 px), so the run 1 smear did not appear here although "up" was also near 0: the run 1 trigger is not "up" alone. |
| 16:50 | `full_165005_all9_align_2` | run 1 repeated, `bimanual_align9_all`, `--align-mode align` | **step 1 pass: aligned in 10.10 s, 8 cycles**, no probe cycles, no "sources disagree" stop, final (-0.30, +0.67, 5.2) mm; back-off, release, home. | - (no insertion) | Capture PC (230.9, 396.2) mm, yaw 4.1 deg; reader (133.5, 454.3) mm, yaw 90.9 deg. Pick held (1 attempt), hold latched. **Approach: `approach_nojump`, 5 iterations, deadband held all three axes at both stop iterations (error (-0.4, 1.1, -0.35) mm); uncommanded move at the stop 0.21 mm** (command path 0.0 mm). Port vision ok on every cycle (3/3 frames). Release stage 32 s with no events (probably the operator prompt before opening the jaws on the lifted reader). |
| 16:54 | `full_165400_all9_full_1` | step 2 run 1 of 3, `bimanual_align9_all` (C1 + D approved), `--align-mode full` | **seated, not fully in** (operator: "successfully inserted but still not pushed all the way"). Printed rest depth **6.75 mm: SHORT** (full from 7.2). | operator: not fully in, **silver band about 2 mm** | Pick held (1 attempt), hold latched. Approach: deadband held all axes at both stop iterations (error (-1.1, -2.3, 0.5) / (-0.9, -2.5, 0.4) mm), uncommanded move at the stop 0.24 mm. Align 10.64 s, 7 cycles, press re-check passed; hand-over gap 6.79, lateral (-0.16, 0.00) mm; no disagreement stop. K4 `seated_detected` at 1.12 s, depth 7.28 mm. **Force push: 10.0 N reached, advance 2.64 mm under the load, ended `still` (no 0.15 mm progress in 0.6 s), no watch stop (left hand 0.16 mm, grip change 0.0001, off line 0.32 mm); at rest the advance fell back to 0.71 mm.** So the hand went about 1.9 mm back when the load ended. |
| 17:02 | `full_170158_depth_b_1` | C2, `bimanual_align8_depth_b` (no D, old approach and align) | **no insertion: align stop at the hand-over, `after the press: sources disagree at the hand-over [1.53, 3.56, 0.79] mm`** (up limit 3.5); back-off 15 mm, release, home. Push not tested. | - (no insertion) | Capture PC (229.3, 395.1) mm, yaw 4.6 deg; reader (133.4, 453.6) mm, yaw 89.6 deg. Pick held, hold latched. Align 53.3 s, 19 cycles. Vision at the end (0.34, 0.10, 6.38) mm = on target; wrist net (-1.20, 3.66, 7.17) mm: the known net "up" bias (+2.6 mm) that D removes (with D the up residual would be about 1.1 mm). Same stop as `k4_4`, `demo_5`. |
| 17:10 | `full_171043_all9b_full_1` | C2 in the combined config, `bimanual_align9_all_b` (14 N, centring) | **seated, not fully in: operator band about 2 mm** although the printed label says `depth at rest 7.33 mm (fully in from 7.2 mm): FULL` (**label disagrees with the operator**). | **2 mm band** | Pick held, hold latched. Approach uncommanded move at the stop 2.24 mm. Align 13.86 s, 8 cycles, no disagreement stop; hand-over gap 5.68, lateral (0.14, 0.26) mm. K4 `seated_detected` at 1.26 s, depth 8.34 mm (all9_full_1: 7.28). **Force push: 14.0 N reached, centred (-0.53, 2.59) mm, advance 3.91 mm under the load (the advance stop is 4.0 mm), ended `still`, at rest 0.54 mm; left hand 0.23 mm, off line 1.20 mm, grip change 0.01 (limit 0.02).** |

Fix (operator: re-cabled the USB for a ZED 2i used as a 3D scanner in another project; approved option A): `scripts/real/rig.py`
`WRIST_CAM["left"]` -> `/dev/v4l/by-path/pci-0000:13:00.0-usb-0:4:1.0-video-index0`. No-motion check through
`deploy.cameras.WristCam`: right and left open at 1280x1024 and stream; the left frame is the left wrist view (PC at the
lower right, as in its last published frame), the right frame the right wrist view. The ZED is opened by serial
(20120598), so the ZED 2i is not picked by mistake.

## Depth block result (C1 10 N, C2 14 N), operator band per run

| Run | Push | K4 seat depth | Advance under load | Advance at rest | Printed rest depth | Operator |
|---|---|---|---|---|---|---|
| `all9_full_1` | 10 N, no centring | 7.28 mm | 2.64 mm | 0.71 mm | 6.75 SHORT | band about 2 mm |
| `all9b_full_1` | 14 N, centred (-0.5, 2.6) mm | 8.34 mm | 3.91 mm | 0.54 mm | 7.33 FULL | band about 2 mm |

- Both pushes reach their force and move the hand 2.6-3.9 mm deeper under the load, then the hand comes back 1.9-3.4 mm when the load ends; the operator sees the same 2 mm band after both. More force did not change the result at rest. The hand holds the reader in closed jaws when the load ends, so the arm's spring-back may pull the plug back out (not measured: FK sees the hand, not the plug).
- The printed FULL/SHORT label follows K4's seat depth more than the push: 7.33 FULL with a 2 mm band. The 7.2 mm threshold from 3 labelled runs does not hold.
- Runbook stop rule: two failures of the same kind (short after the push) -> stop the depth step; base config for the rest. Not run: C1 alone (same push as `all9_full_1`), K6d (C3, skipped by plan).
- Question for the server: keep the load on while the jaws open (or ramp it down slowly), so the plug is not pulled back.

## Placement mapping (E1, capture only)

Operator decision: light version. No tape marks or ruler (the operator cannot measure the angle accurately); the PC is moved by eye, about 1 cm or a small turn, 6 poses instead of 12 (the runbook allows the first 6). Truth for each pose: the server fits the PC case outline in the saved ZED frames (`fit_pc_edges.py`, as for `data/zed_gt` on 2026-11-01, about +/-2 mm). No tape measurement of port 2 at the current spot.

| Time | Capture | PC move (by eye) | ZED PC estimate (v1): port x, y (mm), yaw (deg) | Gate | Notes |
|---|---|---|---|---|---|
| 17:31 | `capture_173123_map_real` | none (current spot) | (228.9, 395.3), 4.51 | ok (Mahalanobis 17.9) | reader estimate rejected (gate false, (145.3, 437.1) mm, yaw 79.5 deg): the reader was not on its usual spot |
| 17:34 | `capture_173419_map_x_plus` | about +1 cm x (away from the bases) | (239.0, 393.7), 3.82; change from map_real (+10.1, -1.6) mm, -0.7 deg | ok (Mahalanobis 18.4) | |
| 17:37 | `capture_173700_map_x_minus` | about -1 cm x (toward the bases), by eye | (225.1, 396.7), 3.57; change from map_real (-3.8, +1.4) mm, -0.9 deg | ok (Mahalanobis 17.8) | estimate moved 3.8 mm; **operator: the move was closer to 5 mm than 10** (by eye); the edge fit decides |
| 17:40 | `capture_174030_map_y_plus` | about 1 cm sideways, by eye (operator: "1 cm left of center", from where the operator stands; side in the right-base frame not checked) | (223.8, 396.5), 4.40; change from map_real (-5.1, +1.3) mm, -0.1 deg | ok (Mahalanobis 18.2) | **Estimate barely moved in y.** Rough check from the ZED image (phase correlation of the PC crop against map_real; the x_plus pose gives the scale and direction, 1.36 px per mm): the PC moved about 6 mm toward the bases and about 4 mm sideways (sign not checked), so it was not put back after map_x_minus and the y move was small. If so, the y estimate is off by up to about 3 mm (under the 5 mm stop rule). Rough; the edge fit decides. |
| 17:44 | `capture_174453_map_y_minus` | about 1 cm to the operator's right of the original spot (the opposite of map_y_plus), by eye | (231.6, 391.8), 3.23; change from map_real (+2.7, -3.5) mm, -1.3 deg | ok (Mahalanobis 15.7) | Image check (same rough method): about 7 mm along x and 8 mm sideways. The ZED y change has the right sign on both sideways poses (map_y_plus +1.3, map_y_minus -3.5) but may be smaller than the move. **The image check is not reliable at the 3-5 mm level** (the static left gripper is in the crop and pulls the phase correlation toward zero; on map_x_minus it disagrees with the ZED in both axes), so it does not trigger the 5 mm stop rule; the edge fit decides. |
| 17:49 | `capture_174911_map_yaw` | turned in place about 5-10 deg, by eye | (233.0, 394.1), 10.09; change from map_real (+4.1, -1.2) mm, **+5.6 deg** | **rejected (Mahalanobis 23.1)**, presence 1.0, arm pose ok | A gate stop is a result (runbook). The estimate turned +5.6 deg for a 5-10 deg turn. The sim expected the v1 gate to pass up to about yaw -6..+8 deg from the box centre; the real spot is already at +4.5 deg, so this pose is about +10 deg absolute. Operator asked to put the PC back at the original spot afterwards (not yet confirmed). |

## To send to mason

Packed on the rig (no file changed): `~/transfer/real_runs_2026-11-13.tar` (1.1 GB, all of `data/real_runs/2026-11-13`, dry runs included) and `~/transfer/real_runs_2026-11-10_resend.tar` (1.6 GB: `full_160352_k4_4`, `full_161011_k4_4`, `full_162427_k4_5`, `full_163515_k4_5`, `full_164410_k4_6`, `full_164756_k4_6`, `full_165527_k4_6`, `full_170521_k4_6`, the folders empty or partial on mason).

| Run folder | What | Operator depth |
|---|---|---|
| `full_163318_all9_align_1`, `full_163740_all9_align_1` | no motion (camera port, viewer label) | - |
| `full_163924_all9_align_1` | align mode, stop: port vision shell width (right edge +8 px) | - |
| `full_164514_fastalign_align_1` | block B alone, aligned 13.1 s | - |
| `full_165005_all9_align_2` | align mode, aligned 10.1 s, approach 0.21 mm | - |
| `full_165400_all9_full_1` | combined, 10 N push, printed 6.75 SHORT | **band about 2 mm** |
| `full_170158_depth_b_1` | depth_b, align stop (wrist-net up 3.56 mm) | - (no insertion) |
| `full_171043_all9b_full_1` | combined + 14 N, printed 7.33 FULL | **band about 2 mm** |
| `capture_173123_map_real` ... `capture_174911_map_yaw` | E1 light mapping, 6 poses | - |

Depth reference by hand (step 0): fully in = no band, no gap; 0.5 mm out straight; 1-2 mm out hanging; 3 mm out touches the table.
Tape shifts: none (light version, by eye: +x about 1 cm, -x about 0.5 cm, sideways about 1 cm each way, yaw about 5-10 deg); truth from the case-edge fit on mason.
Tape-measured port 2 at the current spot: **not taken** (operator chose the light version), so the v1 / v5 question at the real spot stays open; the case-edge fit on `capture_173123_map_real` is the nearest substitute.
Phone / side clips: none this session.
