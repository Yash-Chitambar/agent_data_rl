# Real in-hand pose and real reader look in the sim

Status: v1. Off by default. Measured on 13 real grasps of 2026-11-02, one reader spot. Results below grow as work continues.

## How to enable

Presets: `blind_insertion/configs/inhand_real_v2.json` (use this; one row per measured grasp, default of `grasp_real` and `inhand_real_preset`) and `inhand_real_v1.json` (same mean, independent uniform box, frozen for the K-J numbers). Both have the same look. Module: `blind_insertion/envs/inhand_real.py`.

| What | Setting | Effect |
|---|---|---|
| Reader look | `BimanualCfg.inhand_real = 'v1'` | Silver plate and shoulder as on the real reader. Visual only: no change to the collision shape or mass. |
| In-hand pose, insertion | `InsertionCfg.grasp = 'real'` (`grasp_real = 'v1'`, `grasp_real_spread = 1`) | Reader drawn from the measured grasp. The reference pose stays the Part 1 nominal, so the policy's in-hand input keeps its meaning. |
| In-hand pose, render data | `InsertCfg.inhand = 'real'` (`RightInserter`) | The same draw, for wrist and in-hand data. |
| Wrist net training | `scripts/perception/train_approach_estimators.py --inhand-mode real --reader-look v1` | Both of the above. |
| Direct use | `inhand_real.sample_real(n, 'v1', device=dev)` | Site in reader, the signature of `inhand.sample_measured`. |

Do not set `reader_body_length_m` (see Reader): the body stays 41.26 mm.

## The reader: the 29.6 mm is not the body length

Claim checked: "reader body 41.3 mm in sim, 29.6 mm real". **The body is about 41 mm on the real reader. 29.6 mm is the width of the silver plate.**

- Source of 41.26: `pc_bench.build_scene` computes `READER_TOTAL_LENGTH - plug_protrusion - neck_length` = 90.46 - 5.8 - 43.4 mm. It is a config-derived value, not a mesh.
- Source of 29.6: `data/real_yam/usbc_lab_20261027/measurements.csv`, "29.6 mm for silver body only", with 34.6 mm "across". `blind_insertion/assets/pc_rear/README.md` already says that 29.6 is the silver inset. The caliper axes are swapped (below).
- Measured with the calibrated wrist camera, 26 hover frames over 13 grasps, the reader flat on the table, frames rectified to straight down at the top-face height (6.8 px/mm, FK height):

| Quantity | Measured (mean, SD over frames) | Caliper or model |
|---|---|---|
| Silver across the reader | 30.9 mm, 0.11 (blown-out edge adds about 1 mm) | "length" 29.6 |
| Silver along the reader | 35.9 mm median, 0.6 (same edge effect) | "across" 34.6 |
| Plug tip to silver front | 48.0 mm, 0.14 | 5.8 + 43.4 = 49.2 |
| Plug exposed | 6.0 mm, 0.07 | 5.8 |
| Plug tip to butt | about 90.9 mm, by eye on one rectified frame | 90.46 |

- **Sim old look:** silver 34.6 wide, 29.6 long, 54 mm from the tip. Real: 29.6 wide, 34.6 long, 49.2 mm from the tip. The old look is wrong in width, length and place.
- **`grasp_fidelity` v2 sets `reader_body_length_m = 0.0296`.** That cuts the collision body and the look to 29.6 mm. It is wrong: do not use that entry for geometry. I did not change v2. `grasp_fidelity_v2_real.json` is v2 without it.
- **Nets:** all wrist and in-hand nets in the bundle trained on the old look (41.26 mm body, silver 34.6 x 29.6). No net trained on a 29.6 mm body: v2 is not in their training configs (`ApproachRenderer` has no `grasp_fidelity`). The look mismatch stays a candidate cause of the weak real slopes, but not the length.

## The real grasp

Method: `scripts/real2sim/inhand_measure.py`. Held-still frames after the lift, 13 grasps, 24 of 40 frames pass the fit (RMS under 5 px; best 2.5 px). Features: 4 corners of the silver plate and 4 of the plug, 6-dof fit on the camera. The camera to site transform has no joints in it.

Deltas about the Part 1 nominal on the TCP axes, hand-eye camera (median, SD over grasps; n = 13):

| Quantity | Value | Within-grasp SD |
|---|---|---|
| x (mm) | +6.66, 0.67 | 0.08 |
| y (mm) | +2.66, 0.40 | 0.11 |
| z (mm) | +0.27, 0.48 | 0.11 |
| Pitch about the closing axis (butt out of the gripper) | **4.1 deg, 0.6** | 0.20 |
| Yaw about the reader's thick axis | +1.6 deg, 0.33 | 0.08 |
| Roll about the reader's axis | not resolved (0 used): SD 3.5 over grasps | 2.3 |

Independent check of pitch: the reader lies flat on the table when the jaws close, so reader pitch in hand = hand tilt at the close minus 55 deg. FK at the end of the squeeze gives a tilt of 59.02 deg, SD 0.53. Reader axis against the table, using the camera fit and that FK: 0.06 deg, SD 0.20 (flat). Plug tip height above the table: 5.17 mm, SD 0.90 (flat on the table: about 4.4 mm). So the camera fit and the encoders agree on pitch within 0.2 deg, and on height within 0.8 mm.

**Cause of the pitch:** the hand closes at 59 deg, not at 55 deg. The butt angle is the hand's tilt error at the close. The reader stays flat on the table, so the grasp keeps the same angle. Over the 13 grasps: pitch = FK tilt at the end of the squeeze - 55 deg, r = 0.96, mean difference -0.04 deg, residual SD 0.17 deg. **So the encoders give each grasp's pitch** without a net. x follows it (r = 0.80 with the tilt).

## Spread between grasps

Port vision (`docs/real/PORT_VISION.md`, branch `real/port-vision`) found that A and A2 are not the same grasp: shell end 12.7 px and rear edge 12.3 px lower in A2 at the stop. This measurement agrees, and shows when it happened:

| A2 - A | This fit, held frames after the pick | Port vision, at the stop |
|---|---|---|
| Shell row (port vision's own detector) | +12.8 px (531.4 and 544.2, from the end of the pick on) | +12.7 px |
| Plug in the hand, height | -1.56 mm | -1.39 mm (tip +3.88 minus camera +5.27) |
| Pitch | -1.27 deg | not an output |

- The difference exists before the transport: no slip on the way. FK tilt at the close: A 58.44, A2 59.52 deg, so it comes from the servo's tilt at the close.
- A and A2 are the lowest and highest of the 13 for the shell row. Their pitch lies inside the range (3.0 to 5.0 deg). Spread over the 13 picked grasps (SD): x 0.67 mm, z 0.48 mm, pitch 0.59 deg, tip row 3.5 px. x and pitch are coupled, r = -0.88.
- **`v1`'s box:** covers A and A2, but not the lowest z of the 13 (-1.12 mm against -0.69). Its draws are independent, so x and pitch lose their coupling. **`v2`** draws one measured grasp, stretched about the mean by U(1, 1.25), plus jitter at the within-grasp SD: it covers all 13 and keeps the coupling (checked in `tests/test_inhand_real.py`).
- Left out: ai1 to ai5, hand-placed (ai1 rolled edge-on, ai3 with the shell above the window). A real pick cannot be assumed to stay in this range at another reader spot.

Where the jaws hold it: pad centroid 58.9 mm from the plug tip along the reader (nominal 61.5 mm) and 10.2 mm above its mid-plane (nominal 8.8 mm).

**Frame choice in the preset:** the sim's wrist camera is moved from the hand-eye value by the tip fit (`BimanualCfg.tip_offset_target='camera'`; 1.5 mm on x, -2 mm on z of the TCP). This moves the reader 2.5 mm along its axis and leaves its height unchanged. The preset uses the runtime camera, so the sim image matches: x +8.16, z -1.73 mm. The bundle's in-hand net read about +3.6 and -2 on real frames: z agrees, and x is under-read. **The physical value along the plug axis is known within about 2.5 mm: not separated.**

Preset spread: uniform, half range = 2 SD over grasps. Real spread may be larger: one reader spot only.

## Validation: sim against real at the same joint angles

`scripts/real2sim/inhand_compare.py`, images `results/real/inhand_real/wrist_*.jpg`. Corners projected in the sim's camera against corners detected in the real frame, RMS over 13 grasps, one frame each:

| | Plug | Silver plate | All |
|---|---|---|---|
| Old pose and look | 52.7 px | 60.1 px | 56.5 px |
| `inhand_real_v1` | 5.3 px | 7.0 px | 6.3 px |

`v1` was fitted on the same frames: this shows agreement, not a held-out test. Leave-one-grasp-out (`v2`, the mean of the other 12 grasps, `--mode loo`): plug 5.4 px, silver 7.2 px. Each grasp's own row (`--mode own`): 3.8 and 4.3 px. In image scale 1 px is about 0.17 mm at the silver plate and 0.18 mm at the plug. Tip silhouette IoU 0.73 with this script's fixed colour mask; the tips do not depend on the preset.

Side view: `results/real/inhand_real/side_real_sim.jpg` (phone photo `IMG_8786`, sim at A's joints). The phone is uncalibrated, so this is a check by eye and one angle: tip lower edge against the reader's neck, about 7-9 deg in the photo (+-3 deg guess, perspective). The sim's edge sits 13.7 deg from the reader axis at the nominal grasp, and pitch closes it: 13.7 - 4.1 = 9.6 deg. Old pose: 13.7 deg. The operator's "under 10 deg at the butt" fits 4.1 deg; "lines up exactly in sim" is not literally so.

`render_lateral_mm = 1.45`: in the real image the reader lies 1.45 mm from the sim's centre between the fingers. I did not separate a real offset from a camera offset. It is used for rendering only; in `InsertionTask` the pads centre the reader.

## The reader under push

The camera is rigid to the hand, so a change of the shell row is a change of the plug in the hand. `scripts/real2sim/inhand_push.py`, with port vision's shell detector and depth from shell width (the medians of shell width scatter by about 0.5 px, about 1.3 mm in depth, so height here is rough), still frames:

| Collect run | Grip reading | Shell row | Shell width | Tip in the hand, from free |
|---|---|---|---|---|
| Free | 0.304-0.298 | 537-535 | 46.3-45.9 px | - |
| After the first seat | 0.292-0.290 | 531 | 47.5 px | up about 2-4 mm: about 2-4 deg more pitch (port vision: 2.3 mm) |
| After the second seat | 0.282-0.278 | 526 | 48.0-49.1 px | larger |
| Before the release | 0.270 | 521 | 49.6 px | up about 7 mm, back about 1 mm: about 7 deg (+-2 deg guess) |

- It does not spring back when the push ends: the reading and the row hold through the sweep.
- A reading drop of 0.01 is 0.47 mm per finger (`gripper_map`). The real fingers yield. In the sim the fingers barely move (see the K-J table).
- A: reading steady at 0.312-0.310 through the stop, shell row 531-532, no change at the stop. A2: 0.304, 544. Their frames near the push are few (the arm is moving); the drop of the reader in `IMG_8784` is near 31 s, and I did not resolve the moment in the logs.
- Port vision reports vision and FK differ by 0.9-1.3 mm along the panel normal under load and agree free. That is the arm or PC giving. It is separate from the in-hand move above, which uses no FK.
- A free 6-dof fit on the silver corners alone falls into the planar ambiguity (37 deg): I did not use it.

## Roll bias of the servo net

`reader_servo_near6` gives the waypoint rotation (hover and near nets give no rotation in this config). Base: 13 grasps, last waypoint, target roll -3.08 deg, SD 0.70 from the saved frames (logged -3.09 on these frames; the session note gives -3.21, and I did not trace the 0.12 deg). `scripts/real2sim/servo_roll_source.py`, results `results/real/inhand_real/roll_source_paste.json`.

| Test (mean target roll over 13 frames, deg) | Result | Reading |
|---|---|---|
| Sim render, same joints, reader at the measured table pose | +0.29 (new look +0.64) | The bias does not come from the geometry of this view |
| Whole real image turned, -6 to +6 deg | -2.0 to -3.8 with no trend (slope 0.03 deg/deg); to reach 0 it would take about 90 deg | **Camera mount roll excluded** |
| Principal point +-8 px, bilinear resize, distortion x0.8 | -2.6 to -3.2 | Crop excluded |
| Distortion k2 = 0 (a gross error) | +0.36 | The net reacts, but such an error would show at the plug |
| Real + sim tips; real, tips grey | -3.68; -2.72 | **Fingertips excluded** |
| Sim, reader yaw +-3 deg; roll +-3 deg | -0.37 to +0.65; +0.19 to +0.63 | Reader's rest pose excluded (in sim) |
| Real + sim reader | -0.02 | **Bias gone: the real reader's pixels carry it** |
| Sim + real reader | -1.29 | Reader alone gives part; the real scene adds |
| Real scene + sim tips and reader | -0.26 | Scene alone does not |
| Real, reader mirrored left-right | +1.81 | Sign flips: it reads a left-right asymmetry |
| Mirror neck and plug only; body only | +2.14; -4.29 | **In the neck and plug, not the body** |
| Symmetrized neck and plug only; body only | +1.57; -3.82 | Same |
| Reader dimmed x0.6 | -2.85 | Not brightness |

**Conclusion: the bias comes from the real neck and plug's left-right appearance in the frame. Not the camera mount, not the crop, not the fingertips, not the plate.** How sure: high on the exclusions and on "neck and plug" (several edits agree), moderate on what in the neck. Shading on one side of the neck, or the sim's plain rectangular neck, are guesses; the mirror edit also moves the neck if it sits off the mask's centre, and these are unnatural inputs on one reader spot, n = 13. The sim value here (+0.29, CPU render, nominal look) is not the +1.74 of the earlier note (Warp, DR); I did not reproduce that. Level roll stays. If the servo nets are retrained: use the real neck shape and randomize its shading.

## Camera roll in the port frames: no

Sim render against real frame at the same joints (`scripts/real2sim/camera_roll_check.py`; atan2 in the image):

| Probe | Real minus sim | n |
|---|---|---|
| Table-side rail edge (held frames) | +0.05 deg, SD 0.21 over runs | 8 frames, 6 runs. The backdrop was partly fitted on wrist frames, so this may not be fully independent |
| PC panel line (port approach frames) | +1.48 deg, SD 0.17 over runs | 61 frames, 3 runs |
| Fingertips (mask fit) | -1.40 deg, SD 0.19 | 13 |

The three do not fit one camera roll: a camera roll turns all of them the same way. What fits:

- **The panel's 1.48 deg is the PC's yaw estimate.** Changing the sim's PC yaw moves this line by 0.83 deg per deg; the difference reaches zero at ZED yaw - 1.8 deg (all 3 runs: -1.8, -1.7, -2.1). So the real panel is about 1.8 deg off the ZED's 3.9 deg (one PC placement, 3 runs). **The graph and wrist work should expect this.** The earlier 3-4 deg plug error in ai4 may be related (a guess: hand-placed grasp, other day's PC estimate).
- **The tips and the reader turn together against the camera** by about 1.4-1.6 deg in the same sense (the reader's yaw is measured independently). The likely reading is that the fingers sit turned on the real hand against the CAD, so the plug's lateral offset (about 1.5 mm at the tip) is physical. This rests on the rail probe, n = 8 frames: moderate confidence. Joint 6's offset is not observable by hand-eye (spec `arms`), and would not change the tips in the frame.
- **Wrist nets:** turning real collect frames (gap under 10 mm, n = 88) by +-1.5 deg moves `wrist_v18ws` by at most 0.3 mm and `wrist_v14w` up to 0.7 mm, and 1.5 mm on gap in one direction: a small part of their error (`camera_roll.json`). A camera roll is not what makes them read low.

Also for the wrist retrain: the sim's camera differs from hand-eye by the tip fit (1.5 mm, -2 mm), see "Frame choice".

## K-J on the real grasp

`scripts/eval/eval_kj_inhand.py`. Realistic sim = K-J's env + `grasp_fidelity_v2_real` (pad friction and time constant, weak wrist, rounded body; body 41.26 mm). n = 1024 per cell, 95 % Wilson, one seed. Hand-over at 6 mm gap, no rotation error, unless stated. `true` = true in-hand pose + K-J's training noise. `net` = the constant the real in-hand net gave on 3 real grasps (+3.6, 0, -2.0, -0.65, -0.75, +1.0), which is what the robot feeds today.

| Hand-over offset (across, up; mm) | Old grasp, true | `v2`, true | `v2`, net |
|---|---|---|---|
| 0, 0 | 97.2 (96.0-98.0) | 95.8 (94.4-96.9) | 94.1 (92.5-95.4) |
| 1, 0 | 95.5 (94.1-96.6) | 85.9 (83.7-87.9) | 90.2 (88.3-91.9) |
| -1, 0 | 92.0 (90.2-93.5) | 88.8 (86.7-90.6) | 92.3 (90.5-93.8) |
| 2, 0 | 94.8 (93.3-96.0) | 76.3 (73.6-78.8) | 80.2 (77.6-82.5) |
| -2, 0 | 89.6 (87.6-91.4) | 82.6 (80.2-84.8) | 91.3 (89.4-92.9) |
| 0, 1 | 95.0 (93.5-96.2) | 90.0 (88.0-91.7) | 94.7 (93.2-95.9) |
| 0, -1 | 94.5 (93.0-95.8) | 92.8 (91.0-94.2) | 89.6 (87.6-91.4) |
| 0, 2 | 92.8 (91.0-94.2) | 82.5 (80.1-84.7) | 92.2 (90.4-93.7) |
| 0, -2 | 93.1 (91.3-94.5) | 85.4 (83.0-87.4) | 69.7 (66.8-72.5) |
| 0, 0, gap 8 mm | 95.6 (94.2-96.7) | 87.7 (85.5-89.6) | 88.3 (86.2-90.1) |
| 0, 0, gap 11 mm | 93.8 (92.1-95.1) | 87.5 (85.3-89.4) | 58.5 (55.5-61.5) |
| 0, 0, rotation up to 2 deg per axis | 80.2 (77.6-82.5) | 66.7 (63.7-69.5) | 59.5 (56.4-62.4) |

Failures: mainly `grasp_slip` (reader moved over 10 mm or 15 deg in the hand), then `lateral_miss`. Full classes: `kj_inhand_v2.csv`. Base sim (no fidelity), old grasp: 98.9-99.6 % over 8 cells; the new grasp was not run in the base sim.

**Reading.** At an exact hand-over at 6 mm, K-J is robust to this grasp: `v2` net 94.1 %, against 97.2 % for the old grasp. It gets worse than the old grasp as the hand-over gets worse, and the gap matters as much as the offset: at 11 mm with today's input it falls to 58.5 %. Vision reads the real stop at about 8 mm (`PORT_VISION.md`).

**Recommendation (no training was run):** make the graph hold the hand-over within about 1 mm and about 6 mm gap first, and do not fine-tune K-J for that case. Fine-tune on `v2` if the graph cannot, or if the real runs stay slip-limited. Do not feed the true in-hand pose without fine-tuning: x +8 mm is outside K-J's training range (about +-3.7 mm), and `true` is often not better than the wrong `net` value (for example -2, 0: 82.6 against 91.3).

**Why these numbers are optimistic:** sim fingers do not yield. Sim finger closure in a push: median 0.01-0.05 mm; real: 0.3-0.5 mm at the first seat, 1.3 mm by the end. Sim reader pitch at exact hand-over: median 0.8 deg, p90 1.6 deg; real: about 2-4 deg at the first seat. The sim also does not give the real drop on its own terms. The sim's success is therefore an upper bound on contact, not a forecast of the real rate.

**Breakdown, exact hand-over, `kj_inhand_parts.csv`:** pitch only 99.4 % (true), 89.5 % (net); `v1` mean without spread 97.2 %; `v1` mean position and pitch with `v1`'s independent box 71.6 %. So `v1`'s box costs 25 points; `v1` total 71.5 %, net 69.0 %. Which axis of the box does this was not separated. Use `v2`. Position only: 0 of 1024, ended before any motion was logged: an invalid setup (a guess: the reader off the pads), ignore it.

## Not verified

- One reader spot, one PC placement, one session. 13 grasps.
- Roll about the reader's axis in the hand: not resolved by the camera. Yaw and the 1.45 mm lateral offset: not separated from the camera to tip geometry with certainty.
- Position along the plug axis: within about 2.5 mm (tip fit against hand-eye).
- The silver sizes come from a blown-out edge plus the caliper values. The neck is the old rectangle, slightly wider than real and without taper; the real shoulder is approximated. The look was compared under one renderer (CPU, not Warp with DR).
- No net has been trained or tested on the new look.

## Next rig session

1. Calipers on the reader: housing length and width, silver along and across, neck width near the plug and near the shoulder, with a photo. This settles the swapped labels.
2. 5 picks at other reader spots, with frames at a fixed hand pose: does pitch still equal the tilt at the close minus 55 deg?
3. Side photo with a ruler at the hand's plane, square to the reader, for the axial position (the 2.5 mm).
4. Slow push with the plug against the case: log grip reading and shell row at 10 Hz or more, to give the sim a finger compliance.
5. Fingers closed on a flat gauge, wrist view, to separate a gripper turn on the flange from a camera roll (1.4 deg). A level or board in view at two joint-6 angles for the joint 6 offset.
6. PC yaw from a square on the table or from the panel line, against the ZED's yaw (1.8 deg here).
