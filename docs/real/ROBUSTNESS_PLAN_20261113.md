# Placement robustness plan, wrist-net up bias, approach jump (task 3, 2026-11-13)

Task 3 of [SERVER_TASKS_20261112.md](SERVER_TASKS_20261112.md). Worker T3 on mason, GPU 4. No rig access.
Read with [DATA_REVIEW_20261113.md](DATA_REVIEW_20261113.md) (task 0). Sim numbers are sim only. Every number has its n.

Coordinates (as the rig logs them, right-base frame):
- **PC**: port 2 position (x, y) in mm and yaw in deg (ZED convention `atan2(-R12, -R02)`). Real spot (2026-11-03..11): about (231, 394) mm, yaw 3-6 deg. Sim training box: (220, 360) +/- (10, 15) mm, +/- 5 deg.
- **Reader**: the ZED reader estimate `p` = the reader frame origin at the plug tip (x, y) in mm, and its yaw. Real strip: x 121-142, y 438-459 mm, yaw 87-99 deg. The reader body centre is 70 mm behind the plug tip (at yaw 90: y - 70 mm).

## Summary

- **Reach is not the limit.** Both arms reach every PC pose in x 181-281, y 344-444 mm, yaw -20..+28 deg (1555 of 1573 grid poses; joint margin >= 0.47 rad inside +/-30 mm, -12..+20 deg) and every reader pose in +/-60 mm, yaw 40-140 deg except where the reader would touch the PC.
- **The ZED stage is the first wall, for the PC and for the reader.** The PC net and its OOD gate were trained around the sim nominal (220, 360) mm, 0 deg. The real spot already sits near the gate edge (real Mahalanobis 17.5-19.1, limit 21.3, n = 18). In the sim, the gate stops the run at PC yaw +5 deg or x +20 mm, and the estimate is wrong (7-53 mm) toward +y even when the gate passes. The reader net reads yaw 70 / 110 deg with 5 / 13 deg error and x +40 mm with 17 mm error, and its gate does not catch it; the servo pick still held in every sim dry run (x +40 mm with a 20.6 mm estimate error, yaw 70-110 deg; n = 1 each).
- **K4 is the second wall: it fails at positive PC yaw** (sim, real-like plant, n = 256 per cell): yaw 14 deg 58.6 %, 19 deg 1.2 %, against 98.0 % at the real spot (n = 512); x +/-30 mm 94.9-99.4 %, y +30 mm 85.9 %. K4p (K4 fine-tuned on the re-centred, wider PC box) is training.
- **Plan:** retrain the ZED PC net + gate (zed_v5, proposed), the reader ZED net (reader_zed_v2 + presence head: done, sim and real-frame checks pass) and K4 (K4p: training) on a layout centred on the real spot; then fine-tune the approach wrist nets on real frames from the new spots. Config only: pick retry, approach fix (`bimanual_align8_nojump*`). Rig session A: capture-only mapping of the ZED gate, then full runs at PC x +/-10, y -20..+10 mm, yaw 0..+6 deg (not 15: K4 and the ZED gate). Session B (+/-25 mm, +/-15 deg) after zed_v5 and K4p pass.

## Part A: placement robustness

### What ran (sim; tools in the repo)

| Check | Tool | n |
|---|---|---|
| Reach, PC (both arms, margins, contacts) | `scripts/task/placement_reach_real.py --only pc` -> `results/placement/reach/pc_map.json` | 1573 poses (11 x 11 x 13) |
| Reach, reader (pick plan, margins, PC clearance) | same, `--only reader` -> `reader_map.json` | 3549 poses (13 x 13 x 21) |
| ZED PC and reader estimates + gates | `scripts/eval/zed_placement.py` -> `results/placement/zed_placement.json` | 46 poses x 8 worlds |
| Full real path, dry run (`deploy.run_bimanual --dry-run --stage full`, `bimanual_align8_nojump`) at forced poses | `scripts/eval/placement_grid.py` -> `results/placement/grid_v1/table.json` | 48 runs (n = 1-2 per point), plus 12 A/B runs at the real spot (Part C) |
| K4 against the PC pose (real-like plant, real hand-over lead) | `scripts/eval/k4_placement.py` -> `results/placement/k4_placement.json` | 512 or 256 per cell |
| demo_3 empty close | real logs (`grasp_v3` events) | 17 held picks, 1 empty |

The dry runs use `align.sim_source` (sim truth + noise) for the align stage: the sim does not render port vision. So "align" in the sim checks the motion and the gates, not the camera view of the port.

### Reach map (kinematics only)

- PC (port x, y, yaw), both arms: right IK on the insertion line (0, 6, 30, 60 mm out, +/-8 deg in-hand tilt), left press/brace IK, joint margin >= 0.05 rad, 5 mm contact margin. **1555 of 1573 feasible** over x 181-281, y 344-444 mm, yaw -20..+28 deg. The 18 failures are all at yaw <= -8 deg with x <= 191 and y >= 414 (no left hold) or at x >= 271, y 444, yaw -20 (right IK with tilt).
- Smallest joint margin of either arm, about the real spot (PC yaw absolute): +/-20 mm, yaw -8..+16 deg: >= 0.55 rad (175 of 175 poses); +/-30 mm, -12..+20 deg: >= 0.47 rad (441 of 441); +/-50 mm, -20..+28 deg: >= 0.28 rad (1555 of 1573 feasible).
- Reader (pick plan of `PickExpert.reset`: 55 deg tilt grasp, pre-grasp / grasp / lift, margin >= 0.05 rad): every pose converges within +/-60 mm, yaw 40-140 deg. With the PC at the real spot and >= 15 mm reader-PC clearance: +/-30 mm and 70-110 deg 441 of 441 (margin >= 0.35 rad, clearance >= 22 mm); +/-40 mm and 60-120 deg 1018 of 1053; the blocked poses are high x with high yaw (the plug end swings toward the PC).
- So kinematics allow a much larger box than the perception does.

### ZED perception (sim, 8 worlds per pose, policy/v1 visual DR)

| Pose (from the real spot) | PC gate ok (of 8) | PC err p50 / max mm | PC yaw err p50 / max deg | Reader gate ok | Reader err p50 / max mm | Reader yaw err p50 / max deg | reader_zed_v2 err p50 / max mm, yaw p50 / max deg |
|---|---|---|---|---|---|---|---|
| real | 8 | 2.9 / 4.5 | 0.5 / 1.4 | 8 | 2.0 / 4.2 | 0.2 / 1.4 | 1.1 / 2.8, 0.7 / 1.3 |
| pc dx -40 | 7 | 3.9 / 6.4 | 0.6 / 1.1 | 8 | 2.1 / 4.6 | 0.8 / 1.7 | |
| pc dx -30 | 8 | 4.3 / 7.0 | 0.6 / 1.4 | 8 | 2.6 / 7.0 | 0.5 / 1.3 | |
| pc dx -20 | 8 | 3.5 / 5.5 | 0.4 / 1.0 | 8 | 1.6 / 3.1 | 0.4 / 0.7 | |
| pc dx -10 | 8 | 5.0 / 6.7 | 0.3 / 0.8 | 8 | 2.1 / 3.9 | 0.3 / 0.6 | |
| pc dx +0 | 8 | 4.2 / 5.4 | 0.5 / 1.2 | 8 | 2.7 / 3.9 | 0.3 / 1.2 | |
| pc dx +10 | 8 | 5.9 / 6.4 | 0.3 / 0.9 | 8 | 2.4 / 3.6 | 0.2 / 1.3 | |
| pc dx +20 | 8 | 4.0 / 11.5 | 0.4 / 0.7 | 8 | 2.2 / 3.4 | 0.3 / 1.0 | |
| pc dx +30 | 7 | 8.2 / 15.8 | 0.6 / 2.0 | 8 | 2.0 / 7.0 | 0.3 / 0.8 | |
| pc dx +40 | 6 | 28.3 / 85.9 | 0.4 / 1.8 | 8 | 2.0 / 3.1 | 0.3 / 0.8 | |
| pc dy -60 | 8 | 3.0 / 7.1 | 0.5 / 1.3 | 8 | 1.6 / 3.1 | 0.5 / 1.2 | |
| pc dy -45 | 8 | 3.8 / 5.7 | 0.3 / 1.0 | 8 | 0.8 / 1.9 | 0.6 / 1.8 | |
| pc dy -30 | 8 | 3.6 / 7.7 | 0.2 / 2.4 | 8 | 1.4 / 3.8 | 0.5 / 1.7 | |
| pc dy -20 | 8 | 3.5 / 9.3 | 0.4 / 0.9 | 8 | 1.8 / 2.7 | 0.4 / 0.7 | |
| pc dy -10 | 8 | 4.6 / 5.4 | 0.3 / 0.7 | 8 | 1.6 / 3.6 | 0.2 / 0.5 | |
| pc dy +10 | 7 | 7.4 / 15.3 | 0.5 / 2.7 | 8 | 3.1 / 5.6 | 0.4 / 1.8 | |
| pc dy +20 | 6 | 19.8 / 35.4 | 0.7 / 2.2 | 8 | 3.2 / 5.0 | 0.3 / 1.3 | |
| pc dy +30 | 3 | 35.4 / 51.2 | 1.3 / 4.1 | 8 | 3.1 / 6.8 | 0.5 / 1.4 | |
| pc dy +40 | 0 | 52.9 / 86.2 | 1.9 / 4.9 | 8 | 2.8 / 3.7 | 0.2 / 0.7 | |
| pc yaw -20 | 2 | 5.3 / 9.5 | 2.9 / 4.8 | 8 | 2.0 / 2.9 | 0.3 / 0.7 | |
| pc yaw -15 | 8 | 2.8 / 6.7 | 0.4 / 1.0 | 8 | 2.0 / 4.1 | 0.2 / 0.6 | |
| pc yaw -10 | 8 | 5.2 / 5.9 | 0.3 / 0.8 | 7 | 2.5 / 3.9 | 0.5 / 2.3 | |
| pc yaw -5 | 8 | 4.8 / 6.8 | 0.7 / 1.6 | 8 | 2.1 / 7.6 | 0.5 / 1.1 | |
| pc yaw +5 | 8 | 3.2 / 6.6 | 0.3 / 1.0 | 8 | 2.4 / 4.3 | 0.2 / 1.8 | |
| pc yaw +10 | 3 | 6.2 / 11.3 | 3.1 / 5.2 | 8 | 2.1 / 4.2 | 0.4 / 0.9 | |
| pc yaw +15 | 1 | 4.9 / 12.1 | 7.4 / 10.2 | 8 | 1.5 / 4.0 | 0.3 / 1.4 | |
| pc yaw +20 | 2 | 6.8 / 12.1 | 11.8 / 14.2 | 8 | 1.8 / 3.3 | 0.4 / 0.8 | |
| pc yaw +25 | 0 | 8.7 / 14.1 | 17.7 / 21.5 | 8 | 2.3 / 3.1 | 0.1 / 0.7 | |
| rd dx -60 | 8 | 3.3 / 5.1 | 0.6 / 1.1 | 7 | 15.3 / 43.6 | 0.7 / 7.4 | 2.2 / 4.5, 0.8 / 2.0 |
| rd dx -40 | 8 | 5.4 / 7.9 | 0.6 / 1.2 | 8 | 2.5 / 5.0 | 0.4 / 1.7 | 1.9 / 3.2, 0.7 / 1.3 |
| rd dx -20 | 8 | 4.5 / 6.6 | 0.6 / 1.4 | 8 | 2.3 / 4.1 | 0.6 / 1.3 | 1.6 / 2.9, 0.7 / 1.6 |
| rd dx +20 | 8 | 3.8 / 5.5 | 0.2 / 0.8 | 8 | 4.5 / 5.7 | 1.2 / 1.4 | 1.4 / 3.6, 0.6 / 1.8 |
| rd dx +40 | 8 | 3.1 / 9.0 | 0.5 / 1.0 | 8 | 17.2 / 22.7 | 4.0 / 5.2 | 1.7 / 3.4, 0.4 / 0.8 |
| rd dx +60 | 8 | 3.8 / 8.5 | 0.4 / 1.2 | 1 | 49.8 / 64.5 | 6.1 / 7.4 | 5.9 / 10.3, 0.9 / 1.7 |
| rd dy -60 | 7 | 5.3 / 8.8 | 0.5 / 1.3 | 8 | 7.1 / 15.5 | 0.4 / 0.6 | 2.9 / 4.2, 0.5 / 1.0 |
| rd dy -40 | 7 | 2.2 / 6.2 | 0.8 / 2.7 | 8 | 1.8 / 4.3 | 0.8 / 1.7 | 1.6 / 6.1, 0.4 / 1.6 |
| rd dy -20 | 8 | 2.9 / 4.4 | 0.4 / 1.7 | 8 | 2.2 / 3.4 | 0.5 / 2.0 | 2.2 / 3.1, 0.4 / 0.8 |
| rd dy +20 | 8 | 3.9 / 6.4 | 0.5 / 1.5 | 8 | 2.3 / 4.0 | 0.3 / 0.7 | 1.2 / 2.0, 0.2 / 1.1 |
| rd dy +40 | 8 | 4.5 / 8.3 | 0.7 / 2.1 | 8 | 2.2 / 2.5 | 0.5 / 1.0 | 1.3 / 3.0, 0.7 / 1.6 |
| rd dy +60 | 8 | 3.8 / 5.8 | 0.5 / 0.8 | 8 | 3.0 / 3.9 | 0.3 / 0.8 | 2.7 / 5.3, 0.6 / 1.3 |
| rd yaw 50 | 8 | 2.1 / 4.3 | 0.5 / 1.4 | 8 | 22.0 / 35.5 | 31.4 / 36.9 | 0.8 / 2.7, 0.4 / 1.4 |
| rd yaw 60 | 8 | 3.0 / 6.9 | 0.5 / 1.2 | 8 | 13.6 / 15.2 | 16.9 / 19.7 | 1.1 / 2.4, 0.3 / 1.0 |
| rd yaw 70 | 8 | 3.6 / 7.0 | 0.3 / 1.0 | 8 | 4.2 / 5.7 | 5.3 / 5.7 | 0.4 / 1.7, 0.2 / 0.5 |
| rd yaw 80 | 8 | 2.9 / 8.4 | 0.4 / 0.9 | 8 | 2.0 / 3.3 | 0.2 / 0.6 | 1.2 / 3.8, 0.4 / 1.0 |
| rd yaw 100 | 8 | 2.5 / 4.5 | 0.8 / 1.3 | 8 | 1.7 / 3.8 | 3.0 / 3.9 | 1.5 / 4.1, 0.2 / 0.9 |
| rd yaw 110 | 8 | 3.2 / 4.6 | 0.1 / 0.3 | 8 | 6.2 / 7.8 | 13.0 / 13.7 | 1.1 / 4.0, 0.2 / 2.7 |
| rd yaw 120 | 8 | 4.3 / 5.8 | 0.3 / 0.8 | 6 | 15.0 / 19.5 | 24.9 / 38.5 | 1.0 / 11.0, 0.3 / 16.4 |
| rd yaw 130 | 8 | 4.1 / 5.2 | 0.3 / 0.7 | 8 | 23.6 / 28.2 | 40.5 / 43.2 | 1.4 / 2.6, 0.4 / 1.1 |

"pc dx/dy" move the PC port (mm), "pc yaw" adds to its 4 deg; "rd dx/dy" move the reader plug tip (mm), "rd yaw" is the absolute plug yaw (deg). PC gate = presence + OOD (Mahalanobis <= 21.3) + arm pose. Reader gate (v1) = presence + arm pose. Sim real-spot Mahalanobis 19.7 (dry run seed 7041), real 17.5-19.1 (n = 18 runs).

### Full chain dry runs at forced poses

| Run | PC (x, y mm; yaw) | Reader tip (x, y mm; yaw) | ZED PC err mm | Reader est err mm | Pick | Hold | End |
|---|---|---|---|---|---|---|---|
| c_s7041 | (231, 394; 4) | (131, 448; 92) | 0.2 | 3.3 | held | latched | insert: guard_bound |
| c_s7043 | (231, 394; 4) | (131, 448; 92) | 2.4 | 3.8 | held | latched | seated detected (true seated) |
| pcx-30_s7041 | (201, 394; 4) | (131, 448; 92) | 1.4 | 1.8 | held | latched | insert: press_abort |
| pcx-30_s7043 | (201, 394; 4) | (131, 448; 92) | 2.6 | 2.1 | held | latched | align: after the press: step limit |
| pcx-20_s7041 | (211, 394; 4) | (131, 448; 92) | 2.0 | 2.0 | held | latched | seated detected (true seated) |
| pcx-20_s7043 | (211, 394; 4) | (131, 448; 92) | 2.9 | 2.5 | held | latched | seated detected (true seated) |
| pcx+20_s7041 | (251, 394; 4) | (131, 448; 92) | 2.8 | - | - | - | ZED gate stop (est err 2.8 mm) |
| ng_pcx+20_s7041 | (251, 394; 4) | (131, 448; 92) | 2.8 | 4.2 | held | latched | seated detected |
| pcx+30_s7041 | (261, 394; 4) | (131, 448; 92) | 8.4 | - | - | - | ZED gate stop (est err 8.4 mm) |
| ng_pcx+30_s7041 | (261, 394; 4) | (131, 448; 92) | 8.4 | 4.1 | held | latched | align: moved too far from the stop |
| pcy-30_s7041 | (231, 364; 4) | (131, 448; 92) | 5.4 | 2.0 | held | latched | insert: guard_bound |
| pcy-30_s7043 | (231, 364; 4) | (131, 448; 92) | 3.1 | 2.3 | held | latched | seated detected (true seated) |
| pcy-20_s7041 | (231, 374; 4) | (131, 448; 92) | 5.2 | 2.4 | held | latched | insert: guard_bound |
| pcy+20_s7041 | (231, 414; 4) | (131, 448; 92) | 27.1 | 3.8 | held | latched | align: moved too far from the stop |
| pcy+30_s7041 | (231, 424; 4) | (131, 448; 92) | 44.0 | - | - | - | ZED gate stop (est err 44.0 mm) |
| pcyaw-15_s7041 | (231, 394; -11) | (131, 448; 92) | 3.2 | - | - | - | ZED gate stop (est err 3.2 mm) |
| ng_pcyaw-15_s7041 | (231, 394; -11) | (131, 448; 92) | 3.2 | 3.1 | held | latched | insert: press_abort |
| pcyaw-10_s7041 | (231, 394; -6) | (131, 448; 92) | 0.6 | 2.9 | held | latched | seated detected (true seated) |
| pcyaw-5_s7041 | (231, 394; -1) | (131, 448; 92) | 0.7 | 3.1 | held | latched | seated detected (true seated) |
| pcyaw+5_s7041 | (231, 394; 9) | (131, 448; 92) | 1.5 | - | - | - | ZED gate stop (est err 1.5 mm) |
| ng_pcyaw+5_s7041 | (231, 394; 9) | (131, 448; 92) | 1.5 | 3.1 | held | latched | insert: press_abort |
| pcyaw+10_s7041 | (231, 394; 14) | (131, 448; 92) | 2.7 | - | - | - | ZED gate stop (est err 2.7 mm) |
| ng_pcyaw+10_s7041 | (231, 394; 14) | (131, 448; 92) | 2.7 | 3.5 | held | latched | insert: press_abort |
| pcyaw+15_s7041 | (231, 394; 19) | (131, 448; 92) | 5.6 | - | - | - | ZED gate stop (est err 5.6 mm) |
| pcc-20-20+10_s7041 | (211, 374; 14) | (131, 448; 92) | 5.2 | - | - | - | ZED gate stop (est err 5.2 mm) |
| pcc+20-20-10_s7041 | (251, 374; -6) | (131, 448; 92) | 6.1 | 3.8 | held | latched | seated detected |
| pcc-20+20-10_s7041 | (211, 414; -6) | (131, 448; 92) | 29.8 | 2.6 | held | latched | align: moved too far from the stop |
| pcc+20+20+10_s7041 | (251, 414; 14) | (131, 448; 92) | 33.3 | - | - | - | ZED gate stop (est err 33.3 mm) |
| rdx-40_s7041 | (231, 394; 4) | (91, 448; 92) | 0.2 | 1.9 | held | latched | align: after the press: moved too far from the stop |
| rdx-20_s7041 | (231, 394; 4) | (111, 448; 92) | 0.2 | 3.2 | held | latched | insert: press_abort |
| rdx+20_s7041 | (231, 394; 4) | (151, 448; 92) | 0.4 | 3.7 | held | latched | insert: press_abort |
| rdx+40_s7041 | (231, 394; 4) | (171, 448; 92) | 1.4 | 20.6 | held | latched | seated detected (true seated) |
| rz_rdx+40_s7041 | (231, 394; 4) | (171, 448; 92) | 1.4 | 0.8 | held | latched | seated detected (true seated) |
| rdy-40_s7041 | (231, 394; 4) | (131, 408; 92) | 0.2 | 4.0 | held | latched | insert: guard_bound |
| rdy-20_s7041 | (231, 394; 4) | (131, 428; 92) | 0.3 | 4.0 | held | latched | insert: press_abort |
| rdy+20_s7041 | (231, 394; 4) | (131, 468; 92) | 0.1 | 3.1 | held | latched | insert: press_abort |
| rdy+40_s7041 | (231, 394; 4) | (131, 488; 92) | 0.2 | 3.0 | held | latched | insert: press_abort |
| rdyaw-22_s7041 | (231, 394; 4) | (131, 448; 70) | 0.2 | 6.1 | held | latched | insert: guard_bound |
| rz_rdyaw70_s7041 | (231, 394; 4) | (131, 448; 70) | 0.2 | 0.4 | held | latched | insert: press_abort |
| rdyaw-12_s7041 | (231, 394; 4) | (131, 448; 80) | 0.2 | 2.5 | held | latched | seated detected (true seated) |
| rdyaw+8_s7041 | (231, 394; 4) | (131, 448; 100) | 0.3 | 2.5 | held | latched | align: after the press: step limit |
| rdyaw+18_s7041 | (231, 394; 4) | (131, 448; 110) | 0.7 | 5.5 | held | latched | align: moved too far from the stop |
| rz_rdyaw110_s7041 | (231, 394; 4) | (131, 448; 110) | 0.7 | 1.1 | held | latched | align: moved too far from the stop |
| rz_c_s7041 | (231, 394; 4) | (131, 448; 92) | 0.2 | 1.5 | held | latched | insert: press_abort |

Tags: `ng_` = the same point with the ZED OOD gate opened (sim study only, `--zed-ood-scale 3`), to see the stages behind it; `rz_` = config `bimanual_align8_nojump_retry2_rzv2` (reader_zed_v2). All runs `bimanual_align8_nojump` (except `rz_`), speed 1.0, seed 7041 unless `_s7043`. n = 1 per point: the dry run is a code-path and failure-mode check, not a rate.

Reading:
- **Pick and hold never failed** (all 38 runs past the ZED stage: held, latched), including reader x +40 mm with a 20.6 mm reader estimate error (v1 net): the servo pick absorbs it. Reader yaw 70-110 deg also held.
- **The ZED PC gate stopped 9 of 22 off-centre PC runs** (x +20/+30, y +30, yaw +5/+10/+15/-15, two corners). With the gate opened, x +20 (seated) and yaw -15..+10 got past the approach and the align stage (their ends are the sim insert noise below); x +30 (8.4 mm estimate error) stopped in the align stage, so the gate (not the estimate) was the limit there; y +20 / y +30 and the (-20, +20, -10) corner had a 27-44 mm estimate error, and the run ended in an align stop ("moved too far from the stop") or at the presence gate: the safe stops worked.
- **The ends after the align stage are sim-noise dominated:** at the real spot itself 1 of 3 runs is seated (c_s7041 guard_bound, rz_c_s7041 press_abort, c_s7043 seated). Seed 7041 ends in `press_abort` or `guard_bound` at most points (also on 2026-11-11: `dry_s7041_dry_fastmove` press_abort). Align stops "moved too far from the stop" (-16.1 mm advance, limit 16) come from the sim wrist net's gap bias at the 18 mm stop. So the grid shows where each stage *stops*, and K4's rate comes from the batched eval below.

### K4 against the PC pose

| Cell | PC port (x, y mm; yaw deg) | Seated | Rate [Wilson 95 %] | Main failures |
|---|---|---|---|---|
| real | (231, 394; 4) | 502 / 512 | 98.0 % [96.4, 98.9] | lateral_miss 9, grasp_slip 1 |
| dx-30 | (201, 394; 4) | 509 / 512 | 99.4 % [98.3, 99.8] | lateral_miss 2, grasp_slip 1 |
| dx-20 | (211, 394; 4) | 507 / 512 | 99.0 % [97.7, 99.6] | lateral_miss 5 |
| dx+20 | (251, 394; 4) | 245 / 256 | 95.7 % [92.5, 97.6] | lateral_miss 11 |
| dx+30 | (261, 394; 4) | 243 / 256 | 94.9 % [91.5, 97.0] | lateral_miss 10, tilt 2, deep_unseated 1 |
| dy-30 | (231, 364; 4) | 254 / 256 | 99.2 % [97.2, 99.8] | grasp_slip 1, lateral_miss 1 |
| dy-20 | (231, 374; 4) | 255 / 256 | 99.6 % [97.8, 99.9] | lateral_miss 1 |
| dy+20 | (231, 414; 4) | 238 / 256 | 93.0 % [89.2, 95.5] | lateral_miss 14, tilt 2, grasp_slip 1, jam 1 |
| dy+30 | (231, 424; 4) | 220 / 256 | 85.9 % [81.1, 89.7] | lateral_miss 32, tilt 3, jam 1 |
| yaw-15 | (231, 394; -11) | 211 / 256 | 82.4 % [77.3, 86.6] | grasp_slip 21, tilt 14, lateral_miss 9, jam 1 |
| yaw-10 | (231, 394; -6) | 247 / 256 | 96.5 % [93.5, 98.1] | grasp_slip 6, lateral_miss 3 |
| yaw+10 | (231, 394; 14) | 150 / 256 | 58.6 % [52.5, 64.5] | lateral_miss 102, grasp_slip 2, tilt 2 |
| yaw+15 | (231, 394; 19) | 3 / 256 | 1.2 % [0.4, 3.4] | lateral_miss 248, grasp_slip 3, tilt 2 |
| yaw+20 | (231, 394; 24) | 0 / 256 | 0.0 % [0.0, 1.5] | lateral_miss 255, grasp_slip 1 |
| sim_nominal | (220, 360; 0) | 252 / 256 | 98.4 % [96.1, 99.4] | lateral_miss 4 |
| c-20-20-10 | (211, 374; -6) | 253 / 256 | 98.8 % [96.6, 99.6] | lateral_miss 3 |
| c+20+20+15 | (251, 414; 19) | 2 / 256 | 0.8 % [0.2, 2.8] | lateral_miss 251, grasp_slip 2, tilt 1 |
| c+20-20+15 | (251, 374; 19) | 0 / 256 | 0.0 % [0.0, 1.5] | lateral_miss 253, grasp_slip 3 |

Setup: `scripts/eval/k4_placement.py` = `k3_robust.py`'s real-like plant and 'real_ho' hand-over cell (gap 5.5 mm, lateral ring 0.5 mm, real right-arm lead, vision prior as trained), all worlds of a cell at one PC pose. First episode per world. The first three cells ran with n = 512, the rest with n = 256 (shared GPU).

Reading:
- **K4 holds in x and in -y** (94.9-99.6 %), and at the sim nominal (98.4 %). It drops toward +y (y +20 mm 93.0 %, +30 mm 85.9 %).
- **K4 fails at positive yaw.** PC yaw 14 deg: 58.6 %; 19 deg: 1.2 %; 24 deg: 0 % (n = 256 each), almost all `lateral_miss`. Negative yaw holds better (-6 deg 96.5 %, -11 deg 82.4 %). K4 trained with the PC at 0 +/- 5 deg (BimanualCfg.pc_pose_range); its observation includes absolute right-arm joint angles, so a turned PC is out of its training range. The real spot (4 deg) is at the edge already.
- So the operator's "yaw 0-15 deg" test would fail in the insertion at the upper end with today's K4. Keep yaw <= +8 deg on the rig until K4p passes.
- Retraining started (clearly needed): **K4p** = K4 + the v2 layout (PC nominal = real spot) + `pc_pose_range` +/-30 mm, +/-15 deg; init `runs/insert_K4.pt`, 30M steps, same env and PPO settings as K4 (`runs/P4-insert-k4-s0/config.json`), seed 449, 4096 worlds. Run `runs/P4-insert-k4p-s0` (env `runs/k4p_env.json`, log `results/placement/train_k4p_s0.log`). Early training success 0.97-0.98 (iterations 2-8). Expected 2.5-3 h on a free GPU 4 (K4: 2.3 h). Acceptance: `k4_placement.py --policy runs/P4-insert-k4p-s0/last.pt --out results/placement/k4p_placement.json` >= 95 % in every cell of the step 2 box, the real cell not worse than K4 (98.0 %), and the K4 acceptance cells of K2_PROGRESS (real-like, rings, roll) and the real hand-over replay (`replay_real_handovers.py`). Then copy to a new name (`runs/insert_K4p.pt`) and a new config; K4 stays.

### Grasp: the demo_3 empty close

- Real data: hand height at the close (`close_from_settle` z): 17 held picks of 2026-11-10/11 at -23.0 to -25.3 mm; `demo_3` (16:08) at **-17.7 mm** (same -27 mm command). The squeeze then read 0.03 (empty), the test lift said not held, and `max_attempts` 1 (batch A) ended the run with "pick failed".
- Nothing upstream marks it: reader estimate (137, 452) mm, 90.2 deg (strip normal), last servo delta (1.6, 1.4, 6.9) mm (other runs 1.0-7.2, 5.1-9.1), 5 servo frames as on every run. So it is a chance touch-down on the reader top, not a placement effect. It can happen at any spot.
- Fix 1 (config only, made): `deploy/config/bimanual_align8_nojump_retry2.json` = `bimanual_align8_nojump` + `pick.batches.A.grasp_v3.max_attempts` 2. The existing retry opens, rises 40 mm, returns to the hover and takes new wrist frames; no press (`press_from_attempt` stays null). Sim record of the retry (PICK_GRASP_V3.md): "retry only" rows recover most first-attempt misses. Conventions `FAILED: none`. Needs operator approval only in the sense that a second descent is new on the rig (same limits).
- Fix 2 (new logic, proposal): a pre-close height check in `blind_insertion/pipeline/grasp_v3.py`: if the measured TCP z at `close_from_settle` is more than `close_max_above_cmd_mm` (proposed 5.0) above the commanded grasp height, do not close; reopen, rise and retry. Real data: held picks sit 1.7-4.0 mm above the -27 mm command (n = 17), demo_3 9.3 mm. It saves the squeeze and the test lift. Off by default (null). Not done here: grasp_v3 is shared sim code; the sim close heights vary with the encoder-offset DR (-20.0 to -30.9 mm in 3 dry runs), so the check must be relative to the command, and it needs a sim eval first.
- New reader poses and the grasp: the reach map shows the pick plan solves everywhere in the target box. The in-hand pose spread at other yaws is not known on the real rig; the wrist nets learned one grasp family (Part B). Check the hand-over disagreement on the first runs at new reader yaws.

### Failure map: where each stage fails as the placement moves

| Stage | Limit found | Evidence (sim unless stated) | Fix class |
|---|---|---|---|
| ZED PC estimate + gate | OOD gate: passes the real spot only (real Mahalanobis 17.5-19.1, limit 21.3, n = 18 runs); sim gate stops at yaw +5 deg (21.5), x +20 mm (22.8), yaw -15 deg (23.7). Estimate wrong toward +y: y +20 mm err 20 mm p50 and 6 of 8 pass the gate; y +30 mm 35 mm; yaw >= +10 deg yaw error 3-18 deg | zed_placement (8 per pose), grid_v1 (zed stage) | **retrain** (zed_v5 + gate on a box centred on the real spot) |
| ZED reader estimate + gate | x +40 mm (plug tip x 171): 17 mm error, gate passes; yaw 70 deg 5 deg error, 110 deg 13 deg, 60/120 deg 17/25 deg; y +/-60 mm fine (y -60: 7 mm) | zed_placement | **retrain** (reader_zed_v2, started) |
| Pick reach | none inside +/-60 mm, yaw 40-140 deg (except reader-PC clearance) | reach map | none |
| Pick grasp | chance empty close (demo_3), not placement-bound; in-hand pose at new yaws unknown on the rig | real logs, grid_v1 | config (retry2) + new check (proposal) |
| Left hold | none in +/-50 mm, yaw -8..+28; fails at yaw <= -8 with x <= 191, y >= 414 | reach map, grid_v1 (all latched) | none |
| Transport / approach IK | none inside the reach map box | reach map, grid_v1 | none |
| Approach servo (wrist nets) | follows the ZED error: with a 27-30 mm ZED error (y +20, or x -20 / y +20 / yaw -10) the approach ends far off and align stops "moved too far from the stop" (safe stop); wrist nets were fine-tuned on real frames at one spot (background, lighting, grasp) | grid_v1 | real-frame fine-tune after the first new-spot runs |
| Align (port vision, yaw) | not testable in the sim (sim_source); the view is relative to the port, so the PC pose changes only the background and the light on the panel; the panel yaw estimate covers 7 deg (`angle.max_deg`); reader yaw changes the in-hand pose | geometry, Part B | rig check |
| K4 | holds in x (94.9-99.4 %) and -y; drops at y +30 (85.9 %), yaw -11 (82.4 %); **fails at yaw >= +14 deg** (58.6, 1.2, 0 %) | k4_placement (256-512 per cell) | **retrain** (K4p, started) |
| Sim-only noise at every spot | the dry run fails 1 of 2-3 runs even at the real spot (align post-press budget "step limit", insert guard_bound / press_abort); these are sim align-noise and plant effects, not placement | grid_v1 c_s7041, nojump dry runs | not placement |

### Retraining started: reader_zed_v2 (done)

- Layout `results/placement/task_layout_real_v2.json` (new file): PC nominal = the real port (231, 394) mm, 4 deg; reader body centres x 85-175, y 335-425 mm (plug tip about y 405-495 at yaw 90), plug yaw 50-130 deg.
- `scripts/perception/train_reader_zed_v2.py` (wrapper; `train_reader_zed.py` unchanged): same net, crop, labels and strip centre as v1; PC box +/-30 mm, +/-15 deg. n = 49152 train, 4096 test, 40 epochs, 2546 s on the shared GPU 4. New file `runs/perception/reader_zed_v2.pt` (v1 untouched); report `results/usbc/policy/pick/reader_zed_v2.json`.
- Held-out (DR): xy p50 / p99 1.31 / 5.45 mm, yaw 0.38 / 2.65 deg (v1 on its narrower set: 1.10 / 4.38 mm, 0.32 / 2.08 deg).
- Placement sweep (`zed_placement.py --reader-net reader_zed_v2`, 8 per pose): x -60..+40 mm err p50 1.4-2.2 mm (x +60: 5.9), y -60..+60 mm 1.2-2.9 mm, yaw 50-130 deg yaw err p50 0.2-0.7 deg (one 16 deg outlier of 8 at 120 deg). v1 on the same poses: 17-50 mm at x >= +40, 5-40 deg yaw error outside 80-100 deg.
- Real frames (no labels): v2 against v1 on the 40 real `zed_reader` frames of 2026-11-02..11: xy difference p50 / p90 / max 2.1 / 2.8 / 3.5 mm, yaw 0.3 / 0.8 deg. v1 is the rig-proven net, so v2 transfers at the real spot.
- Not ready for the rig yet: v2 has no presence head (v1 had one from `train_reader_presence.py`); without it the gate is the arm-pose envelope only. Next: train the presence head on the v2 layout, then a config `pick.reader_estimator: reader_zed_v2`, bundle copy, conventions, dry run.

### Proposed target range

Two steps. Step 1 needs no retraining; step 2 needs the new ZED nets.

| | PC port 2 (x, y; yaw) | Reader plug tip (x, y; yaw) |
|---|---|---|
| Today (proven, n = 10 seats) | (231, 394) mm, 3-6 deg | x 121-142, y 438-459 mm, 87-99 deg |
| **Step 1** (configs of today, rig test next) | x 221-241, y 374-404 mm (-20..+10), yaw 0..+6 deg | x 111-151, y 428-468 mm (+/-20), yaw 82-102 deg |
| **Step 2 target** (after zed_v5 + reader_zed_v2 + real-frame fine-tune of the wrist nets) | x 206-256, y 369-419 mm (+/-25), yaw -11..+19 deg (+/-15) | x 91-171, y 408-488 mm (+/-40), yaw 70-110 deg |
| Stretch (reach allows; check K4 and the left hold first) | +/-40 mm, yaw -16..+24 deg | +/-50 mm, yaw 60-120 deg |

Why step 1 stops there: K4 drops from 98.0 % at 4 deg to 58.6 % at 14 deg (trained at 0 +/- 5 deg), and the ZED OOD gate already sits at 17.5-19.1 of 21.3 at the real spot, and in the sim it closes at about yaw +5 deg and x +20 mm from it. A gate stop is safe (the run ends at the capture), but it is not a placement test. Inside step 1 the sim ZED error stays at 1.5-5 mm, which the wrist servo and the align stage absorb (the align stage starts from an 18 mm stop and handles +/-20 mm lateral).
Why the step 2 box: reach margin >= 0.47 rad on both arms, left hold everywhere, reader_zed_v2 at 1.2-2.9 mm, K4p over +/-30 mm and +/-15 deg (to be checked). The y direction is the one to watch: today's ZED net is wrong toward +y even when its gate passes.

### What needs what

| Stage | Change | Kind | Status |
|---|---|---|---|
| ZED PC net + presence + OOD gate | zed_v5: train on a box centred on the real spot (+/-40 mm, +/-20 deg); move `NOMINAL_PORT_XY` / the crop with it (a new estimator key, the old nets stay), refit the gate (`train_zed_gate_v2.py`), arm envelope, real check (`check_real_zed.py` on the 2026-11-02..11 ZED frames: v5 must agree with v1 at the real spot) | **retrain + small code** | proposed |
| ZED reader net | reader_zed_v2 + presence head | **retrain** | net done; presence head training (`train_reader_presence_v2.py`) |
| Pick servo nets (hover, near, near6) | check the yaw 70-110 deg cells in the sim pick eval (`scripts/policy/grasp_v3_eval.py` on the v2 layout); retrain only if the hover / near errors grow | eval first | proposed |
| Pick grasp | `max_attempts` 2 (config, done); close-height check (new logic, off by default) | config + logic | config made |
| Left hold | none (plans from the ZED estimate) | none | - |
| Transport, approach IK | none | none | - |
| Approach jump | `approach_nojump` (config + chain.py, done) | config + code | made (Part C) |
| Approach wrist nets (r10w, r10f) | fine-tune on real frames from the new spots (`finetune_wrist_real.py`, worktree `~/wt/wrist-real`); fold in the up-bias fix (Part B: B2) | **retrain on real rollouts** | after the first step 1 runs |
| Align (port vision) | none in code; check the panel and the plug in view at each new spot (`align_measure` conf, sigma) | rig check | - |
| Wrist-net disagreement gate | `align.secondary.bias_mm` (Part B: B1) | config + small code (align.py owner) | spec given |
| K4 | K4p: fine-tune with `pc_pose_range` +/-30 mm, +/-15 deg on the v2 layout (new name `insert_K4p.pt`) | **retrain** | training (`runs/P4-insert-k4p-s0`) |

### Order and cost (GPU 4, V100 32 GB, shared)

1. Now (done, 0 GPU h more): `bimanual_align8_nojump`, `bimanual_align8_nojump_retry2`, reader_zed_v2 (0.7 GPU h).
2. Rig session A (step 1 box, today's nets): see the runbook below. Collect wrist frames at the new spots.
3. zed_v5 + gate (about 2 GPU h: 2 seeds x 0.6 h as zed_v4, gate 0.5 h, checks 0.3 h) and the reader presence head (0.3 h). Code: the estimator key for the new nominal and crop (half a day).
4. Wrist-net fine-tune on the real frames of session A and the earlier align frames (B2: under 1 GPU h; half a day of code).
5. K4p (started): 30M steps, 2.5-3 h on a free GPU, + the placement eval and the K4 acceptance cells (1.5 h).
6. Rig session B (step 2 box).
Total: about 4-7 GPU h and 1.5 days of code before session B.

### Rig test plan (operator: PC +/-1-2 cm and yaw 0-15 deg first)

Session A, today's nets. Mark the current PC spot on the table first (tape at two feet), so each shift is measured against it (ruler, +/-1 mm; yaw from the two marks, +/-1 deg). Shift the PC only; the reader stays on its strip.

A1. Capture-only mapping (no motion beyond the capture pose, about 1 min per pose). It shows where the ZED gate passes and how well the estimate follows the shift:

```bash
C=deploy/config/bimanual_align8_nojump.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C | tail -1                     # FAILED: none
# per pose: move the PC, then
.venv/bin/python -m deploy.run_bimanual --stage capture --config $C --name map_dx+10 \
    --pc-spot '{"port_xy": [0.241, 0.394], "yaw_deg": 4, "method": "tape shift", "unc_mm": 1}'
```

Poses, in this order: real, x +10, x -10, y +10, y -10, yaw +5, yaw +10, yaw +15, x +20, x -20, y +20, y -20 (12 poses).
Read: `estimate zed_pc` (port_xy, yaw_deg, ok, mahalanobis) in events.jsonl. At a rejected gate answer Enter (stop), never `c`.
Pass per pose: gate ok, and the estimate moves by the shift within 3 mm and 1.5 deg. Expected from the sim: ok up to about x -30..+10, y -30..+10, yaw -6..+8 (gate); rejects or wrong beyond (worst toward +y and +yaw). Stop the mapping at the first pose where the gate passes but the estimate is off by more than 5 mm: that is the dangerous case.

A2. Full runs at the poses that passed A1 (start with the nearest), `bimanual_align8_nojump_retry2` at `--speed 0.75 --align-mode full --demo`:
- Keep the PC yaw at 0..+6 deg in full runs until K4p is on the rig (sim K4: 14 deg 58.6 %, 19 deg 1.2 %).
- 2 runs at the real spot first (checks the approach fix and the retry config at a known spot).
- Then 1-2 runs per passed pose, nearest first; yaw before x/y.
- Reader: move it within its strip only (x +/-10, y +/-10 mm, yaw 85-100 deg) until reader_zed_v2 is in the bundle.

Watch, per run: approach (`approach_iters.py` "at the stop": jump <= 3 mm), align entry (first `align_measure`: |across|, |up| <= 8 mm, gap 15-22 mm), align `conf` / `sigma` (port vision at the new panel pose), hand-over disagreement (Part B), K4 seat, push, operator note.
Stop the session: ZED gate passes with an estimate > 5 mm off the tape shift; a "moved too far from the stop" or "entry gap" align stop at a pose where the real-spot runs passed; any guard stop in the approach; an approach jump > 3 mm with `approach_nojump` on.

Session B (step 2 box): only after zed_v5 and the reader presence head pass their real-frame checks and dry runs.

## Part B: wrist-net "up" bias

Tool: `scripts/real/wristbias_table.py` (logs only, no motion). Data: every `full_*` run of 2026-11-03, 06, 10, 11 with align events, dry runs excluded: 31 runs, 552 align cycles, 13 seats. No events (not analysed): 2026-11-07 (all runs), 2026-11-10 `161011_k4_4`, `162427_k4_5`, `163515_k4_5`, `164410_k4_6`, `164756_k4_6`, `165527_k4_6`, `170521_k4_6` (partial copy: ticks only).

### How the check works (align.py, read only)

- Primary source: port vision, median of 5 frames. Secondary: `WristNetSource` with no `net` key, so it uses `ch.fine` = `wrist_r10f_ft_abs`, one frame per cycle. Both give the plug tip in port axes (x across, y up, z gap), mm.
- Step mode: `AlignLogic.update` (line 409 on 2026-11-13 07:20, was 345), `dis = |vision - net|` per axis. It is tested only when the confirm count is full (line 422) against `disagree_mm` [3.0, 3.5, 2.5]. Servo mode: same test at line 1171.
- The dry run has no secondary source (`secondary_sim` is null).

### Net minus vision at the hand-over

| Day | n | Up, median (min..max), mm |
|---|---|---|
| 2026-11-03 | 8 | +1.55 (-4.16..+2.96) |
| 2026-11-06 | 8 | +1.55 (-0.18..+4.49) |
| 2026-11-10 | 8 | +2.60 (+0.29..+3.54) |
| 2026-11-11 | 7 | +3.07 (-0.19..+3.51) |

Clean hand-overs (reader held, align ended by tolerance or the disagreement stop), n = 25: across -0.27, up **+2.58**, gap +0.76 mm (median). SD 0.35, 1.12, 0.32 mm. Up is positive on 23 of 25. The gate stopped 3 hand-overs: `align_full7` (4.49), `k4_4` (3.54), `demo_5` (3.51). On 2026-11-10/11, n = 15: up median +2.88, SD 1.07 mm.

### Which source is biased: the net

1. **Seated geometry (independent of both sources).** Seat-2 lateral = vision's lateral at the hand-over + FK tip motion since the hand-over. With the plug deep in the port (seat-2 depth >= 5 mm), the true lateral is about 0. Median "up" over the deep ticks, per seat, n = 13:
   - vision as the reference: median **+0.33 mm**, mean +0.32, SD 0.85.
   - net as the reference (same FK, net's hand-over value): median **+2.91 mm**, mean +2.75, SD 1.50.
   So vision is right within about 0.3 mm. The net reads about 2.6 mm high.
2. **Slopes against FK motion (all cycles, per run, n = 29 runs).** Up: vision 0.93 median (p10 0.80), net 1.03 (p10 0.94). Vision has the small slope deficit (7 % of a few mm: under 0.3 mm). It does not explain a 2.6 mm constant.
3. **Gap dependence: small.** Net minus vision up, all cycles: gap 3-6 mm +1.81 (n = 199), 6-8 mm +2.20 (n = 122), 8-12 mm +2.33 (n = 67), 12-25 mm +1.90 (n = 164).
4. **Grasp dependence: strong.** Over 15 hand-overs with an in-hand refresh, net minus vision up correlates with the in-hand net's dx (r = 0.84). `demo_1` had an unusual grasp (in-hand dx 0.43 mm; others 3.9-5.5 mm) and the net agreed with vision there (-0.19 mm). Without it, n = 14: median +2.64, SD 0.61 mm.

Cause: `r10f_ft_abs` took its absolute "up" from FK plus vision's offset on one collect run (one grasp, 2026-11-02, `docs/real/WRIST_NETS_REAL.md`). The net reads the hand more than the plug. When the reader sits differently in the jaws, its "up" error changes. The bias is not constant: it rose from about +1.5 (2026-11-03/06) to about +2.9 mm (2026-11-10/11).

Side note (not in Part B's scope): the coarse net `r10w_ft_abs` at the approach end reads up +1.71 mm (median, SD 1.95, n = 15) above the first align vision measurement. So the 18 mm approach stop is also about 1.7 mm high.

### Seat check: not the net bias

- `insert.py` `make_guards` gives `SeatDetector2` the align result `lat_mm` (vision, fused). `insert_guard.py` line 49: `lat = lat0 + FK tip motion`. The net is not used. So the net bias cannot reach the seat check.
- The final push gate (`insert.py` lines 307-321) reads `lat2` at **one tick**: the last tick of seat 2. While the plug is deep, FK "up" swings by 3.0 mm per run (median range, n = 13), because K4 moves the hand and the reader turns in the jaws.
- The two skips were dips at that tick. `demo_1`: deep median -1.71, seat tick -3.50 mm. The value fell from -0.4 to -3.5 mm in the last 15 ticks at a constant 8.1 mm depth. This is the reader turning in the jaws (operator: "reader hanging"). `speed075_1`: deep median -0.74, seat tick -3.33 mm.
- Median over the deep ticks, all 13 seats: |across| <= 1.20, |up| <= 1.78 mm. 0 of 13 are over 2.5. The last 25 deep ticks give the same result: <= 1.33 and 1.81 mm, 0 of 13 over.

### Comparison with T0 (DATA_REVIEW findings 21-22, 7)

- Same data, same answer on the size: T0 +1.9 to +3.4 mm on 13 of 15 (median +2.8); here +2.58 median over 25 clean hand-overs (2026-11-03..11), +2.88 on 2026-11-10/11 (n = 15).
- Same answer on the cause: it follows the grasp (in-hand dx, r = 0.84, n = 15; `demo_1` -0.19 mm with an odd grasp). The seated geometry adds the independent check that vision, not the net, is right (deep-seat residual +0.33 mm with vision as the reference, +2.91 mm with the net, n = 13).
- Seat check: agree that the net bias does not reach it (seat-2 lateral = vision hand-over + FK motion). T0 reads the two skips as a real low tip (K4 did not pull up: lead up 8.9-10.4 mm against 27-32 mm, tip 3.3-3.5 mm low); the deep-median view here says the value fell in the last 15 ticks at constant depth (reader turning in the jaws, operator: "hanging"). Both say the tip was really low at the end. So B3 below would push a reader that hangs: an operator decision, not a bias fix. I keep B3 as an option only.

### Fix specification (for the coordinator; align.py and insert.py not edited)

**B1: subtract the net bias (recommended now).**
- Key: `align.secondary.bias_mm`, net minus vision (across, up, gap) in mm. Default when absent: [0, 0, 0], which keeps every existing config as before.
- Value: **[-0.3, 2.6, 0.8]**. Data: median over the 25 clean hand-overs (-0.27, +2.58, +0.76). The seated geometry gives the same value (net 2.91 - vision 0.33 = 2.58, n = 13).
- Place: `WristNetSource.__init__` (align.py, class at line 208 on 2026-11-13 07:20; align.py is being edited, so find it by name): `self.bias = np.asarray(spec.get('bias_mm', [0, 0, 0]), float)`. `WristNetSource.measure` (line 233 on the same date): `v = self.predict(fr, ch.ndev) - self.bias`, and keep the raw value in `extra['raw']`. This one change covers both checks (`AlignLogic.update` and the servo check) and the logged `second` values. The dry run is not affected (no secondary in the sim).
- Effect on the logs: residual up on 2026-11-10/11 (n = 15): -2.79..+0.94 mm. `k4_4` 0.94, `demo_5` 0.91. 0 of 25 clean hand-overs exceed 3.5. Both stopped good runs would pass.
- Approval: the gate width stays 3.0/3.5/2.5. The window moves: the raw net-vision "up" can now be -0.9..+6.1 mm, before -3.5..+3.5. This allows a larger raw "up" difference, so **the operator must approve it**. Variant inside today's limits: up bias 2.6 with an up gate of 0.9 keeps the raw upper limit at 3.5. It would stop 4 of 15 hand-overs on 2026-11-10/11 (a4p_1 -2.31, demo_1 -2.79, k4_4 +0.94, demo_5 +0.91), so it does not help.
- Limits: this is a calibration for one grasp family. Recheck it after every change to the pick or the reader spot (Part A). It drifted by +1.5 mm in one week. A grasp like `demo_1` gives a residual of -2.8 mm, which is near the gate.

**B2: retrain the net (the real fix, after B1).** Fine-tune `wrist_r10f_ft_abs` -> new name `wrist_r11f_ft_align` with `finetune_wrist_real.py` (~/wt/wrist-real). Add a dataset of the real align frames: 31 runs, 552 cycles x 5 frames, many grasps. Use the vision 5-frame median as the label (vision matches the seated geometry within 0.3 mm). Keep the run split by day for hold-out. Work: about half a day of code (a new loader, the same loss), less than 1 GPU hour (4000 steps). Pass bar: up bias within ±0.5 mm and SD under 0.8 mm on held-out days. Then set `bias_mm` back to 0.

**B3: seat lateral gate (final push).** Key: `insert.final_push.lat_stat`, values `"last"` (default, as today) or `"deep_median"`, with `insert.final_push.lat_window_ticks` (25). With `"deep_median"`, the gate reads the median `lat2` over the last 25 seat-2 ticks at depth >= `min_depth_mm`. Place: `insert.py` final push block, lines 307-317 (record the `s2['lat']` and `s2['depth']` history in the insert loop at line 282). `max_lat_mm` stays 2.5. Data: 13 of 13 seats pass, including `demo_1` (1.71) and `speed075_1` (0.74). Today, 11 of 13 pass. The gate value does not change, but the gate reads a different statistic. Propose it to the operator with this table. Do not subtract a bias here: the seat check has no net bias (vision-referenced deep median +0.33 mm, n = 13).

## Part C: approach jump

### What happens (ticks, 2026-11-03..11, 37 real approaches with logs)

Tool: `scripts/real/approach_iters.py <run> ...` (no motion): every servo iteration of the approach (frame, net estimate, commanded tip step, measured tip move in the ramp and in the settle, command-tip lag, joint 2/3 tracking error in the believed frame), and the worst uncommanded move located in its iteration. JSON: `results/placement/approach_iters.json`.

1. **Every real approach ran 11 servo iterations: 5 coarse and 6 at the 18 mm stop.** The configs ask for `chain_args.iters` 4 and `final_iters` 2 since 2026-11-03 (`bimanual_align3`, note `_stop`: "the servo integrated its target while joints 2 and 3 stuck, 6 iterations at the stop"). `deploy/bimanual/io_sim.chain_argv` never passes these two keys to eval_chain, so the eval_chain defaults (6, 6) ran. Check: approach length 980 ticks at speed 0.5 on every run = 100 settle + 11 x (50 ramp + 30 settle); 11 `net` events per approach.
2. **At the stop, the servo chases frame-to-frame net noise.** The coarse net (`wrist_r10w_ft_abs`, one frame) reads 0.3-6 mm off the stop target in each stop iteration with the hand still. The servo then asks a step of the same size each time (2026-11-10/11, n = 18 runs, 124 stop iterations: 97 asked less than 3 mm).
3. **The command integrates; the joints stick.** eval_chain plans each step from FK of the *command*, not of the measured pose. While joints 2/3 stay in Coulomb friction (real 1-1.5 N m on joints 2-3, feed-forward 0.55 / 0.82 N m, kp 80), each small step adds to the command-hand lag. Example `k4_4` (16:03): the command moved 5.7 then 4.0 mm (base z) in two stop iterations, the tip stayed within 0.1 mm, the joint 2 tracking error grew to 17-19 mrad, then joint 2 broke free at 134 mrad/s and the tip moved 7.8 mm in 0.3 s. `fastmove_1`: the same, joint 2 error 17.1 mrad before the release.
4. **The jump size follows the wind-up.** Command-tip lag growth at the stop before the worst window: jump runs `kjrc_2` 14.8, `k4_4` 14.8, `fastmove_1` 10.3, `demo_2` 5.8, `a4p_1` 4.1, `k4_3` 4.1, `demo_1` 2.7 mm; runs without a jump `k4_2` 0.1, `k4_1` 0.6, `kjrc_3_push` 0.8, `demo_5` 1.4 mm (Pearson r = 0.41 with the jump size, n = 18; `demo_3` 9.0 and `demo_4` 4.9 mm built a lead but released it in the last iteration, inside the 0.5 s window limit). The jump runs also asked larger steps at the stop: mean 2.1-3.9 mm per iteration against 1.0-1.8 mm on most runs without a jump.
5. **Not the left press.** The left hand moved <= 0.15 mm in every worst window (n = 18). **Not the align entry**: the worst window is 1.5-6 s before the align stage starts (T0 finding 17 agrees: 3.1-5.5 s for the four largest).

Agreement with T0 (DATA_REVIEW 16-18): same runs over 3 mm (7 of 18 with `k4_4` 16:03), same joint-2 release of 11-20 mrad. T0's tool-offset finding is right: `scripts/real/approach_jump.py` reports `target_b - q` with `q` in the encoder frame, so joint 2 reads +12.7 mrad high (right-arm offset 12.7 mrad); the mm value is a difference and is right. `approach_iters.py` uses the believed frame on both sides. New here: the cause of the stick-slip is the unused `iters`/`final_iters` plus the integrating servo at the stop.

### Fix (off by default): `approach_nojump`

Code: `deploy/bimanual/chain.py` (`RealChain.__init__`, `_StopDeadband`, `hstep` keeps `_stage_k0`). Without the config block nothing changes.

```json
"approach_nojump": {"enabled": true, "iters": 4, "final_iters": 2, "stop_deadband_mm": [3.0, 3.0, 1.0]}
```

- `iters` / `final_iters`: applied to the approach (schedule 30 -> 23.2 -> 18 mm, then 2 iterations at 18 mm; 5 iterations instead of 11). `final_pos_only` is capped at `final_iters` (2: no rotation step at the stop, as before).
- `stop_deadband_mm` (across, up, gap): in the iterations at the stop, a port axis of the net estimate within its band of the stop target reads as the target, so the servo asks no step on that axis and the command stays put. The align stage (port vision, creep law) owns these residuals. The raw net output stays in the log (`net` events); each stop iteration logs `approach_deadband` (estimate, error per axis, which axes were held).
- Gap band 1 mm, not 3: the align stage may advance at most 16 mm (`align.max_advance_mm`) from the stop. The first dry run with a 3 mm band on all axes (seed 7041) entered align at a true 21 mm gap and failed `moved too far from the stop ([1.9, -8.0, -16.1] mm)`. So the gap axis keeps its correction.
- Also logged once: `approach_nojump` (applied settings).

New configs:
- `deploy/config/bimanual_align8_nojump.json` = `bimanual_align5_k4_push15_seat6_fastmove.json` + `approach_nojump`.
- `deploy/config/bimanual_align8_nojump_retry2.json` = `align8_nojump` + one pick retry after an empty close (Part A, grasp).

Checks: conventions `FAILED: none` on both configs. Dry runs (`--speed 0.75 --align-mode full`): seed 7041 completed (align then failed `after the press: step limit`, the sim post-press budget; the baseline config on the same seed seated), seed 7043 seated (true seated after release). A/B at the real spot, speed 0.75, seeds 7100-7105 (`scripts/eval/placement_grid.py`, `results/placement/nojump_ab/`):

| | `align5_k4_push15_seat6_fastmove` (base) | `align8_nojump` |
|---|---|---|
| Completed, K4 `seated_detected` | 6 / 6 | 6 / 6 |
| True seated after the release (sim) | 3 / 6 | 4 / 6 |
| Approach iterations (at the stop) | 11 (6-7) | 5 (2) |
| Command path at the stop, mm | 5.9-14.5 | 0.0-1.8 |
| Largest uncommanded move at the stop, mm | 1.8-3.5 | 0.15-0.5 (one 2.1 that includes the last coarse ramp) |
| Align stage, s | 36.5-48.4 | 34.8-45.4 |

The deadband held all three axes in 9 of 12 stop iterations and across / up in the other 3 (the gap stepped, as intended) (`approach_deadband` events). n = 6 per config: no rate difference can be claimed; the sim joints do not stick like the real ones.

Limits: the sim cannot show the stick-slip (its joints do not stick like the real ones; dry-run "jumps" 1.1-2.0 mm, n = 3, 2026-11-11). Only the rig can show that the jump is gone. The approach also gets shorter (5 iterations: 6 fewer iterations: about 9.6 s less at speed 0.5, 6.4 s at 0.75).

### Rig check for the fix

```bash
C=deploy/config/bimanual_align8_nojump.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C | tail -1          # FAILED: none
.venv/bin/python -m deploy.run_bimanual --stage full --config $C --speed 0.75 --align-mode full --demo --name nojump_1
.venv/bin/python scripts/real/approach_iters.py data/real_runs/$(date +%F)/full_*nojump_1 --quiet   # "at the stop" line
```

Pass: `approach_nojump` event logged; 5 `net` events in the approach; at the stop `max_uncommanded_mm` <= 3 and command-tip lag growth <= 3 mm; hand-over and K4 as before. Note: `approach_jump.py --ticks 300` now also covers the end of the last coarse ramp (the approach is shorter), where a lagging hand still catches up with a commanded ramp; read the "at the stop" line of `approach_iters.py` for the jump. Stop: a jump over 3 mm at the stop again, or the align stage fails `moved too far from the stop` or `entry gap`.


## Files (this task)

- Doc: `docs/real/ROBUSTNESS_PLAN_20261113.md` (this file).
- Code (off by default): `deploy/bimanual/chain.py` (`approach_nojump`, `_StopDeadband`).
- Configs (new): `deploy/config/bimanual_align8_nojump.json`, `bimanual_align8_nojump_retry2.json`, `bimanual_align8_nojump_retry2_rzv2.json` (conventions `FAILED: none` on all three; rzv2 needs `runs/perception/reader_zed_v2.pt` copied to the rig).
- Tools: `scripts/real/approach_iters.py`, `scripts/real/wristbias_table.py`, `scripts/task/placement_reach_real.py`, `scripts/eval/placement_grid.py`, `scripts/eval/zed_placement.py`, `scripts/eval/k4_placement.py`, `scripts/perception/train_reader_zed_v2.py`, `scripts/perception/train_reader_presence_v2.py`.
- Data: `results/placement/` (reach maps, ZED sweeps, `grid_v1/table.json`, `nojump_ab/`, `k4_placement.json`, `approach_iters.json`, `task_layout_real_v2.json`, training logs).
- Weights (new names): `runs/perception/reader_zed_v2.pt` (+ presence head; report `results/usbc/policy/pick/reader_zed_v2.json`, `reader_presence_v2.json`); `runs/P4-insert-k4p-s0/` (K4p, training on GPU 4 at the time of writing).
