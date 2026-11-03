# Real in-hand pose and real reader look in the sim

Status: v1. Off by default. Measured on 13 real grasps of 2026-11-02, one reader spot. Results below grow as work continues.

## How to enable

Preset: `blind_insertion/configs/inhand_real_v1.json`. Module: `blind_insertion/envs/inhand_real.py`.

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

Method: `scripts/real2sim/inhand_measure.py`. Held-still frames after the lift, 13 grasps, 26 of 40 frames pass the fit (RMS under 5 px; best 2.5 px). Features: 4 corners of the silver plate and 4 of the plug, 6-dof fit on the camera. The camera to site transform has no joints in it.

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

**Cause of the pitch:** the hand closes at 59 deg, not at 55 deg. The butt angle is the hand's tilt error at the close. The reader stays flat on the table, so the grasp keeps the same angle.

Where the jaws hold it: pad centroid 58.9 mm from the plug tip along the reader (nominal 61.5 mm) and 10.2 mm above its mid-plane (nominal 8.8 mm).

**Frame choice in the preset:** the sim's wrist camera is moved from the hand-eye value by the tip fit (`BimanualCfg.tip_offset_target='camera'`; 1.5 mm on x, -2 mm on z of the TCP). This moves the reader 2.5 mm along its axis and leaves its height unchanged. The preset uses the runtime camera, so the sim image matches: x +8.16, z -1.73 mm. The bundle's in-hand net read about +3.6 and -2 on real frames: z agrees, and x is under-read. **The physical value along the plug axis is known within about 2.5 mm: not separated.**

Preset spread: uniform, half range = 2 SD over grasps. Real spread may be larger: one reader spot only.

## Validation: sim against real at the same joint angles

`scripts/real2sim/inhand_compare.py`, images `results/real/inhand_real/wrist_*.jpg`. Corners projected in the sim's camera against corners detected in the real frame, RMS over 13 grasps, one frame each:

| | Plug | Silver plate | All |
|---|---|---|---|
| Old pose and look | 52.7 px | 60.1 px | 56.5 px |
| `inhand_real_v1` | 5.3 px | 7.0 px | 6.3 px |

The preset was fitted on the same frames: this shows agreement, not a held-out test. In image scale 1 px is about 0.17 mm at the silver plate and 0.18 mm at the plug.

`render_lateral_mm = 1.45`: in the real image the reader lies 1.45 mm from the sim's centre between the fingers. I did not separate a real offset from a camera offset. It is used for rendering only; in `InsertionTask` the pads centre the reader.
