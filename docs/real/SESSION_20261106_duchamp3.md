# Session 2026-11-06, duchamp3: precision controller, gravity system ID, a plain-PD seat

Branch `real/graph-align`. Details, numbers and code: [PRECISION_CONTROL.md](PRECISION_CONTROL.md). Run logs `data/real_runs/2026-11-06/` and raw samples in `data/real_yam/{precision,gravity_id}/*/samples.npz` are not in git.

## Result

1. **Plain-PD full chain seated** (`full_173941_align_full9`, `bimanual_align2`): K-J `seated_detected` 2.2 s after the hand-over, depth 7.98 mm. Operator: fully in, the reader stayed in the port. Video: `data/real_runs/2026-11-06/full_173941_align_full9/demo_3cam.mp4`. Plain-PD tally since 2026-11-03: K-J 2 seats of 4 hand-overs, full chain 2 of 7 tries. The repeat (`align_full10`) failed like `align_full5`: a good hand-over, then the plug stalled on the case while K-J's target climbed. Both runs that stalled and both that seated started from hand-overs within 0.4 mm, so the hand-over alone does not decide it: the contact policy's search does.
2. **Precision controller** (features of the athena3 controller, values from this rig): right arm, encoder FK, hand error 0.13 mm median against 4.2 mm on plain PD, 12 of 12 steps settled.
3. **Gravity system ID, both arms** (our own tool). Held-out torque rms, joints 2/3:
   - Right arm: 0.13 / 0.19 N m (i2rt default 0.82 / 0.61).
   - Left arm: 0.14 / 0.15 N m (default 1.01 / 0.67).
4. **In the chain:** the controller made the align stage the best so far, once the creep offsets were off and the integral window was 12 mrad: steps 0.94-1.09 of the ask, stage 21.8 s, hand-over within 0.15 mm. It does not fit the insertion policy: K-J was trained on plain PD and climbed at once (`align_full8`).

## Runs

| Time | Run | What | Outcome |
|---|---|---|---|
| 14:11 | `precision/20261106_141135_right` | precision check, default values | controller 1.0 mm hand error, hunting on joints 2-3 |
| 14:22 | `gravity_id/20261106_142246_right` | gravity collect | **pose 20 pressed on the limp left arm**, operator stop; 20 poses kept |
| 14:55 | `gravity_id/20261106_145458_right` | gravity collect, two-arm checks | false stop (joint 6 lag), 22 poses kept |
| 15:04 | `gravity_id/20261106_150421_right` | resume at pose 22 | joint 6 resisted at pose 22, guard stop; fit of all three runs |
| 15:09 | `precision/20261106_150932_right` | fitted gravity, measured friction, bands | 0.13 mm, 12/12 settled and locked |
| 15:34 | `gravity_id/20261106_153409_left` | left gravity collect | 32 poses, clean |
| 15:45 | `full_154548_align_prec1` | controller in the chain | software crash (config `_about` read as an arm) |
| 15:51 | `full_155102_align_prec2` | same, fixed | **controller on in the pick: gripper on the table**, safety stop |
| 16:03 | `full_160310_align_prec3` | controller per stage | capture stopped: settle trim skipped (bug) |
| 16:07 | `full_160705_align_prec4` | align only | aligned; motion worse (approach 16.9 mm jump) |
| 16:14 | `full_161426_align_prec5` | integral window 6 mrad | aligned; motion like plain PD |
| 16:25 | `full_162512_align_prec6` | creep offsets off | stalled at step 9 (joint 2 outside the window) |
| 16:33 | `full_163339_align_prec7` | creep off, window 12 mrad | aligned, cleanest steps, 21.8 s |
| 16:45 | `full_164558_align_full7` | + controller on in contact | stopped at the hand-over (vision-net up 4.49 mm); home stopped (bug) |
| 16:52 | `full_165244_align_full8` | same, vision-net up limit 5 mm | hand-over 0.15 mm; K-J climbed 9.7 mm in 0.34 s, guard stop |
| 17:39 | `full_173941_align_full9` | plain PD (`bimanual_align2`) | **seated, confirmed by the operator** |
| 17:50 | `full_175030_align_full10` | plain PD repeat | **not seated**: hand-over 0.26 / 0.29 mm, gap 5.38 mm; plug stalled on the case at 2-3 mm depth, K-J target climbed to +37 mm, a 5 mm sideways jump at 1.5 s, guard bound at across -9.9 mm (4.3 s) |

## Safety incidents

- **Right hand onto the limp left arm** (gravity pose 20). Fixed: the other arm is connected and its measured joints checked (80 mm) before every motion; planning solves joint 1 to keep 100 mm from it; contact guard on joint 1.
- **Controller on during the pick: gripper on the table.** Fixed: `arm_control.<side>.active_stages`; every other stage runs as before the controller.
- Three software stops from my code (a config note read as an arm; settle trim skipped when the controller was off; the stage switch missing on slow moves). Each now has a test or a mock check.

## Open

- **The contact policy's search (K-J) and the alignment policy need more work** (operator, after align_full10): the failures from good hand-overs are the most useful data for it.

- K-J under the controller: model the controller in the sim and fine-tune or re-evaluate, or change how the hand-over sets the target.
- The approach servo wind-up (4.4-8.5 mm uncommanded moves); the server's 18 mm stop (`bimanual_align3`) has not run.
- Joint 6 resistance at right-arm gravity pose 22 (cause unknown).
- Camera check of accuracy (needs the checkerboard); the left arm's precision check.
