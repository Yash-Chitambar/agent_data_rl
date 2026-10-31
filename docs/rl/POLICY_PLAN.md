# Policy plan: pick → approach → hold → insert (draft, 2026-10-31)

Guide: the key-box study ([slides](../research/keybox_sim2real.pptx)). Status and gates roll up
into [CURRENT](../campaign/CURRENT.md). This page is the scaffold: four parts, what each must
achieve, what to explore, and how it is measured. Each part is developed and validated in sim
first; real tests start only after its sim gate passes.

## v1 priority (user, 2026-10-31)

Maximize sim success first. Shrink any variation (layout, reader yaw, in-hand rotation, start
errors, grasp tilt) that costs success; widen it again only after the chain passes. Friction
values are unknown: pick values that work now and refine them from real runs of the first policy.

## Target accuracy

91.5 % (single-arm USB-C) is not enough. The tasks the self-learning pipeline counts as solved
score 98–100 % on held-out finals (500/500 on peg insertion, toolbox, hanging; 493/500 coffee
machine); the key-box vision graph reached 496/500 (99.2 %) in sim and still failed on the real
robot until motion, perception and physics were matched. So, in sim, on held-out randomized
starts with full domain randomization:

| | Gate |
|---|---|
| Each part | ≥ 99.5 % (n ≥ 1,000, Wilson 95 % lower bound ≥ 99 %) |
| End to end (all four chained) | ≥ 98 % (n ≥ 500) |

## Shared: domain randomization (all parts, all data collection)

Every training episode and every demo is drawn from this. Seed 0 reproduces the real look.

| Group | Randomized | Range (source) |
|---|---|---|
| Arm dynamics | Coulomb friction, gravity scale | ×U(0.5, 1.3), ±0.04 (identified profile) |
| Control | action delay; encoder offsets | 0–1 control step (measured ~8 ms); ±1.5° on joints 2–5 (done: `BimanualTaskCfg.latency_steps`, `enc_offset_deg`) |
| Contact | plug–port friction; pad friction and stiffness; PC-feet friction | to set; to set; μ U(0.8, 1.0) for v1 (real value unknown; tune after the first policy) |
| Layout | PC pose; reader pose | PC nominal x ±1 cm, y ±1.5 cm, ±5° for v1 (was ±3 cm, ±10°); reader: 5 × 10 cm strip in front of the PC (x 0.095–0.145, y 0.335–0.435 m), heading = right-base bearing + 15° ± 10° (user: front or middle, not beside the PC); defined by the pick agent from reach + lift reliability |
| Grasp | in-hand reader position and tilt | to set from real grasp captures (new grasp) |
| **Appearance** (slide 15) | 16 groups (PC case/dark parts, reader body/trim/plug, table, wall, room, raceway, plates, backdrop black, accents, mount, arm dark/light, gripper, fingertips): random HSV colour (p 0.75; tips 0.25) else nominal ±20 %; specular U(0, 1), shininess U(0, 1); table and wall patterns p 0.5 (8 procedural, 6 real-photo patches, 40 % real), 2–12 repeats | per episode (done: `BimanualCfg.visual_dr`, `envs/visual_dr.py`) |
| **Lighting** (slide 15) | key point light: elevation 35–90°, azimuth 0–360°, 0.9–1.6 m, intensity 0.3–1.0 (0.65 = nominal), 3200–6500 K, shadows p 0.5; fill ×0.2–1.2; ambient ×0.5–1.5; headlight fixed (Warp compile-time) | per episode (done) |
| **Camera image** (slide 15) | per-channel gain 0.93–1.07, brightness ±12, exposure ±6 %, Gaussian noise σ 1–4 (per-episode level), blur σ 0–1 px, radial k1 ±0.02 | gain/brightness/exposure/noise per frame, blur/k1 per episode (done) |
| **Camera pose** (slides 14, 16) | extrinsic error, random direction, magnitude U(0, 2× real): ZED 4.3 mm / 0.6°, right wrist 5.1 mm / 1.2°, left wrist 3.4 mm / 1.2° (mean = real); fx, fy ±1.5 %, principal point ±0.5 % of the sensor | per episode (done) |
| Camera timing | frame age and cadence | measured cadence model |

Rule from the deck: sim sensors are at least as bad as the real ones.
Appearance and lighting stay mild (user, 2026-10-31): the scene must still look like the real rig
(`VisualDRCfg` defaults: nominal colours ±20 %, a random colour (saturation ≤ 0.5) in 25 % of
groups, yellow fingertips kept, patterns on 25 % of episodes (80 % real-photo patches), light
45–90° up, 0.4–0.95, 3600–6200 K, shadows 40 %). Camera-image noise and camera-pose error stay at real scale.
**RGB only: no depth cameras or depth data anywhere in the policy pipeline for now.**

Visual DR is off by default: `BimanualCfg(render_rgb=True, visual_dr=True)` (or a dict of
`VisualDRCfg` ranges). 10 % of episodes keep the nominal look (`nominal_prob`). Seed 0 = real look:
`BimanualCfg.visual_look` (`configs/warp_look_duchamp3.json`, fitted to the real frames in the Warp
renderer). Check: `scripts/checks/check_visual_dr.py`; contact sheet
`results/usbc/policy/visual_dr/contact_sheet.jpg`.

## Part 1: Pick (ACT, scripted fallback)

- **Goal:** the right arm picks the reader from anywhere in its region with the **new grasp**
  and lifts it with a known, small in-hand pose spread.
- **New grasp (confirmed by the user 2026-10-31, render: `results/usbc/policy/grasp/grasp_60deg_confirmed.png`):**
  the gripper reaches forward tipped **60° from vertical** (tool axis 30° below horizontal, tipped
  about the finger-closing axis toward the reader), so its underside faces the table. The fingers
  close horizontally on the reader's **two side faces** (8.8 mm tall) near the front of the body,
  behind the neck; the reader stays flat; the plug points across the closing direction.
  In `grasp_site` terms (reader at yaw 0, plug along +x): site y (finger opening) = world y,
  site z = (sin 60°, 0, -cos 60°); pad centroid ≈ 9.6 mm behind the site along -z; fingertips
  1 mm above the table. CPU check at reader (0.30, 0.0): exact IK; pads cover ~3.7 of the 8.8 mm
  side face (more tilt = less contact); beyond ~60-65° the arm cannot reach and the finger plates
  hit the table first. Main risk to test: slip or tip on lift with this small contact height.
- **Approach:** scripted IK expert in sim (privileged reader pose) → demos with full DR → ACT on
  wrist + ZED RGB + encoders (sim-only demos, as in task H).
- **Explore:** scripted (ZED pose estimate + IK) vs ACT; ACT chunk size and ensembling; grasp
  height/tilt; regrasp or retry on a failed grasp.
- **Metrics:** grasp success; reader still flat in hand (tilt); in-hand position spread; time.

### Part 1 status (2026-10-31, branch `policy/pick`; reader strip in front of the PC)

- **Grasp as code:** `blind_insertion/scenes/reader_grasp.py`, tilt **55°** (60° confirmed by the
  user; Part 2: 55° reaches all port poses), `flip=False` (hand reaches along the plug; the flipped
  grasp cannot be turned to the port). `grasp_poses(...)`: pre-grasp 6 cm back along the tool axis
  and **10 cm up** (clears the PC case on the move from rest), grasp, lift 8 cm. Fingertips 1.0 mm
  above the table; touching pad spheres over the reader COM (10.3 mm behind the body front); pad
  contact band 5.9-8.9 mm of the 8.8 mm face at every tilt 45-60°. **`inhand_pose()`** returns the
  measured in-hand pose from `results/usbc/policy/pick/inhand_spread.json`.
- **Reader region (user: in front of the PC):** `results/usbc/policy/pick/reader_region.json`
  (+ `task_layout.json`, `reader_region.png`, ZED view `layout_views/zed.png`): body centre
  **x 0.095-0.145 m (5 cm) × y 0.335-0.435 m (10 cm)**, reader yaw = **right-base bearing + 15° ± 10°**
  (world 73-102°). Bounded by the arm mounting plate (front edge x 0.035 m, 2.3 cm tall: the open
  fingers spread along x) and the PC front. Each 1 cm cell passes at its 4 corners × yaw edges: IK of
  pre-grasp/grasp/lift with ≥ 0.05 rad joint margin; reader footprint ≥ 1.5 cm from the PC hull;
  right hand ≥ 3 mm from the PC hull and the plate, ≥ 1.5 cm from the parked left arm. Needed for
  5 cm depth: PC x box ±1 cm (pick task), approach opening 24 mm per finger, heading +15° off radial
  (29 mm opening or PC x ±1.5 cm: 3 cm). Positions further from the radial heading or the old
  60°/world-yaw windows: `reach_map.*`, FINDINGS.
- **Post-pick reorientation:** `scripts/policy/pick_reorient_check.py`: from every start to the
  pre-insert pose (plug 10 mm out, aligned) for 27 PC poses: IK with margin, hand clear of the PC and of
  the left arm in its press pose: **27/27** (`reorient_check.json`).
- **Scripted expert:** `blind_insertion/envs/bimanual_pick.py` (`PickTask`, `PickExpert`): left arm
  parked (joint 1 = 0) during the pick, right gripper opened to 24 mm per finger, joint-space move to
  the pre-grasp, Cartesian approach, settle (integral 0.03/step, ≤ 3 s), close until the fingers stop,
  lift until the tool orientation settles (< 1°), hold. DR: arm dynamics, PC pose ±1 cm / ±1.5 cm /
  ±5° and feet friction, reader region, latency 0-1 step, encoder offsets ±1.5°, reader-table μ
  U(0.4, 1). Contact: pad μ 1.2, squeeze 12 N, impratio 30 (chosen for the least in-hand creep).
  **Gate: 2048/2048 held-out (seed 20000), Wilson 95 % lower bound 99.81 %; 10 s hold 1024/1024.**
  Reader tilt after lift p50 0.98°, max 1.58°. Episode 9.3 s (p50).
- **In-hand pose spread** (grasp_site frame, about the mean, n = 2048): position p50 0.30 mm,
  p99 0.77 mm; rotation p50 0.07°, p99 0.24°; a 10 s hold adds p99 0.16 mm / 0.19°. Sim only.
- **ACT:** pipeline ready (`gen_pick_demos.py` with visual DR, `pick_to_lerobot.py`,
  `scripts/act/train_act.sh`, `export_act.py`, `pick_act_eval.py`); demos on the strip wait for the
  user's approval of the window. ACT v1 (old 60° window, 650 demos, 20k steps): 26.0 % (n = 512).
- **Check:** `scripts/checks/check_pick_part1.py`.

## Part 2: Approach (IK to pre-insert)

- **Goal:** bring the held plug to a pre-insert pose a few mm in front of port 2, with an error
  inside the insertion policy's capture region.
- **Approach:** PC/port pose from the ZED (and the wrist camera near the port); in-hand reader
  pose from the wrist camera; IK to the pre-insert pose.
- **Explore:** pose estimators (ZED only, wrist only, both); a visual-servo refinement step at
  the end; how far out to stop (3–10 mm).
- **Metrics:** lateral and angular pre-insert error distribution vs the capture region of Part 4.
- **Status (2026-11-01, round 3, `policy/approach-hold`): tilt/roll target met; lateral p99 2.0 mm (target
  1.5 mm).** Deliverable for Part 4: `results/usbc/policy/approach/preinsert_error.json`
  (+ `preinsert_error_samples.npz`, per-episode errors to sample RL starts from).
  - **Pipeline (RGB only):** ZED frame before the arms move -> CNN PC pose (x, y, yaw) -> left hold
    (Part 3) -> IK move to 30 mm in front of the estimated port (mean measured in-hand pose) -> visual
    servo on the right wrist RGB (CNN plug-to-port pose; 6 waypoints to the 6 mm stop gap + 6 iterations
    there with roll correction, the last 2 position only; smooth 0.5 s moves; Coulomb feed-forward on the
    right arm). `scripts/task/eval_approach.py --fric-comp --tilt 55 --final-rot roll --coarse-net wrist_v8`;
    estimators `blind_insertion/perception/`, `scripts/perception/train_approach_estimators.py` (DAgger:
    `--states` from `eval_approach.py --save-states`). Weights `runs/perception/wrist_v8.pt`, `zed.pt`
    (not in git).
  - **Current numbers (DR of 3829c31 + Part 1 grasp e94a268, n = 1024):** lateral p50 0.37 / p99 1.98 /
    p99.5 2.21 mm (x p99 1.04, y p99 1.80), tilt p99 1.12 deg, roll p99 0.68 deg, gap error p99 1.22 mm;
    90.5 % inside 1 mm / 2 deg. Floor with a perfect estimate: lateral p99 0.11 mm, tilt p99 0.31 deg.
  - **What moved it:** the measured (small) in-hand spread; fixing a one-step command jerk at the
    teleport; Coulomb feed-forward (servo floor 2.8 -> 0.9 mm in round 1); DAgger on in-loop states
    (lateral p99 4.9 -> 2.1 -> 2.2 -> 2.0 mm over three rounds); roll correction at the stop gap.
    Did not help: fine-stage crop net, multi-frame averaging, a 2-net ensemble.
  - **ZED:** CNN on DR renders xy p50 0.79 / p99 3.1 mm, yaw p99 1.6 deg (sim); real frames 0.41 mm from
    the manual landmark; classical silhouette fit 11.8 mm p50 with 21 % failures. Needs a presence check.
  - **Grasp at the port:** 55 deg reaches every PC pose of the box (60 deg failed at one corner).
  - **Next:** the residual is vertical (camera looks along the plug): a second, rig-mountable RGB view
    from the side of the port, or a stereo baseline across the wrist; more DAgger rounds give ~0.1 mm each.

## Part 3: Left-arm hold

- **Goal:** the PC does not move or tip during insertion.
- **Approach:** scripted IK brace (rear face) or press (top) from the placement search, set
  before Part 2.
- **Explore:** brace vs press; target offset into the PC (holding force); timing.
- **Metrics:** PC translation/rotation during insertion; contact force; collisions with the right arm.
- **Status (2026-11-01, round 3, `policy/approach-hold`): recommended hold: press 25 N, guarded;
  gate (99.5 %) met at 1 N, 99.3 % at 3 N.** Code: `blind_insertion/envs/left_hold.py` (`LeftHold`,
  `Harness`, `RightInserter`), `scripts/task/eval_left_hold.py`, `scripts/task/sweep_left_hold.sh`;
  results `results/usbc/policy/hold/` (`r3_*` = current DR).
  - **Recommended hold:** press the closed fingertips down on the PC top, front half (2 cm in front of
    the case centre), in line with port 2, **25 N** (left joint 2 ~8.5 N m, within the DM4340 rating;
    35 N needs ~12 N m and is no better); **guarded engage**: descend at 10 mm/s from a 15 mm stand-off
    until the encoder-estimated force (PD torque through J^+T, bias at the stand-off) exceeds 3 N,
    latch the measured pose, then ramp dq = Kp^-1 J^T F over 0.4 s (~4-6 s in all). Engage it before
    the approach; it moves the PC by p99 0.06 mm.
  - **PC still (< 0.5 mm, < 0.5 deg over 2 s; fixed push at the port along the insertion axis, +/-15 %
    lateral; n = 1024 each; current DR: PC +/-1.5 cm / +/-5 deg, feet mu U(0.8, 1.0), Part 1 grasp contact
    impratio 30):**

    | Press | 1 N | 3 N | 5 N | 10 N |
    |---|---|---|---|---|
    | 15 N | 100 % | 98.1 % | 97.2 % | 91.2 % |
    | **25 N** | **100 %** | **99.3 %** | **98.3 %** | **95.5 %** |
    | 35 N | 100 % | 99.0 % | 98.1 % | 95.9 % |

    No arm-arm contact, no non-finite world, no right-arm/reader contact with the PC in these runs.
  - **Why guarded:** an absolute IK target with the force offset pushes the PC sideways with 2.7-7 N
    (encoder-offset error through the coupled arm stiffness of 0.3-0.5 N/mm); guarded gives 0.4-1.4 N.
    A rear-face brace cannot hold (6.5 %, round 1); the front half of the top beats the rear half.
  - **History:** round 1 (PC +/-3 cm / +/-10 deg, feet mu U(0.3, 0.9), impratio 100) at 35 N: 1 N 99.3 %,
    3 N 96.0 %, 10 N 84.4 %. The narrower DR and impratio 30 each helped (r2: impratio 100, 3 N 97.0 %).
  - **Limit:** a few % of PCs still slide at 3-10 N although Coulomb capacity is > 20 N: soft friction
    on the 5 mm box feet (sim artifact; MuJoCo Warp has no noslip). The scripted privileged insertion
    still jams (no chamfer), so the real insertion load is set by Part 4.

## Part 4: Insertion (RL)

- **Goal:** seat the plug from Part 2's pre-insert distribution despite pose errors, latency,
  encoder offsets and contact variation.
- **Baseline:** the blind proprioceptive assembly recipe (Bronars et al.): PPO, relative joint
  actions, temporal-convolution actor over joint history, contact-prediction auxiliary head,
  privileged critic; start errors at least as large as real.
- **Explore (new): what the wrist camera adds.** Variants, all with the same DR:
  (a) proprio only; (b) proprio + wrist RGB in the actor; (c) wrist RGB only for the first
  approach, proprio for contact; (d) proprio + wrist RGB + ZED. Measure success, capture region
  and robustness to larger start errors and camera pose errors.
- **Metrics:** seated rate (0.2 s hold), time to seat, PC motion, peak contact force, failure classes.

**Status (2026-10-31, branch `policy/insertion`; details and numbers in FINDINGS 2026-10-31):**

- Built: `envs/insertion_task.py` (episode: reader in the right hand at a pre-insert pose, starts
  from Part 2's measured pre-insert errors `results/usbc/policy/approach/preinsert_error_samples.npz`
  scaled by a curriculum, in-hand ±3 mm/±8° around the confirmed 60° grasp (`inhand_pose`), Part 3's
  guarded 35 N press as the left hold, full physics DR + contact DR), `train_bimanual.py` (PPO,
  TCN actor, contact head, privileged critic), `distill_insertion.py` (DAgger from a privileged
  teacher into blind / wrist / wrist-approach / wrist+ZED students), `eval_insertion.py` (held-out
  seeds, Wilson CI, failure classes, strict and tolerant seating).
- Decisions forced by the sim: accumulated joint targets (relative targets drift 23 mm in 2 s),
  elliptic friction cones (grasp creep), seating with a 25 µm aperture tolerance (soft-contact
  penetration of fully inserted plugs is up to 14 µm), action std 0.2.
- Results (held-out seed 1000, Part 2 start errors × scale, guarded hold, seating = 0.2 s hold with a
  25 µm aperture tolerance; n per row; Wilson 95 %):

  | Policy | Conditions | Scale | n | Seated | Notes |
  |---|---|---|---|---|---|
  | Scripted privileged (Jacobian) | wide DR, 6 s | 1.0 | 512 | 41.4 % [37.2, 45.7] | feasibility only |
  | PPO privileged teacher | wide DR, 60° grasp, 6 s | 1.0 | 1024 | 80.0 % [77.4, 82.3] | first-episode eval (biased low); +6.3 % deep-unseated |
  | DAgger students (blind / wrist / wrist-approach / wrist+ZED) | students' training conditions | 1.0 | 1024 each | 7.1 / 6.5 / 3.5 / 6.3 % | cameras add nothing measurable via DAgger |
  | PPO blind v3 (10 s, early ckpt) | success-first DR, Part 1 grasp | 0.25 | 2048 | 93.4 % [92.2, 94.4] | 97.1 % incl. deep-unseated |
  | **PPO blind v5 s0 (final)** | round-3 Part 2 starts, 25 N hold, Part 1 grasp, success-first contact DR, 10 s | 0.5 / **1.0** / 1.5 | 2048 each | 99.90 % / **99.71 %** [99.36, 99.87] / 99.17 % [98.67, 99.48] | gate passes at Part 2's distribution, misses at ×1.5; PC fixed at 1.5: 99.22 % (hold is not the limit) |

  Final per-level table: FINDINGS 2026-10-31 (Part 4). Approach accuracy needed: success is 100 % for
  start lateral ≤ 0.5 mm, 99.86 % for 0.5-1 mm, 99.3 % for 1-1.5 mm, ~94 % for 1.5-3 mm.
- Open: widen the capture region beyond ~1.5 mm (longer training at ×1.5, search behaviour, chamfered
  port model if the real port has one); the wrist camera did not help through DAgger (students 3.5-7 %)
  and needs a PPO-trained or better-distilled visual policy to answer; contact values (plug-port μ
  0.10-0.20, port solref 3-4 ms) and the 25 µm seating tolerance are choices, not measurements.

## Order of work

1. Shared DR (control, contact, appearance, lighting, camera) in the bimanual runtime.
2. Part 4 baseline (blind) from synthetic pre-insert starts → wrist-camera variants.
3. Part 3 in the GPU runtime under insertion forces.
4. Part 1 with the new grasp → Part 2 → chain all four; end-to-end gate.
5. Real: Part 4 from a hand-placed pre-insert pose, then the full chain (`real/duchamp3`).
