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
| Layout | PC pose; reader pose | PC nominal ±1.5 cm, ±5° for v1 (was ±3 cm, ±10°); reader: a compact window between the right arm base and the PC, yaw ±10° for v1; defined by the pick agent from reach + lift reliability |
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
- **Status (2026-10-31, `policy/approach-hold`): pipeline built and measured; target (1 mm / 2 deg at
  99.5 %) not reached.** Deliverable for Part 4: `results/usbc/policy/approach/preinsert_error.json`
  (+ `preinsert_error_samples.npz`, per-episode errors to sample RL starts from).
  - **Pipeline (RGB only):** ZED frame before the arms move -> CNN PC pose (x, y, yaw) -> left hold
    (Part 3) -> IK move to 30 mm in front of the estimated port -> visual servo on the right wrist RGB
    (CNN plug-to-port pose, 6 waypoints to the stop gap + 6 iterations there, smooth 0.5 s moves,
    position-only at the stop gap) with a Coulomb feed-forward on the right arm.
    `scripts/task/eval_approach.py`, `scripts/task/sweep_approach.sh`, estimators in
    `blind_insertion/perception/`, training `scripts/perception/train_approach_estimators.py`.
  - **Grasp at the port:** the confirmed 60-deg grasp reaches the pre-insert pose over the PC box
    except the near-left corner (port x 0.19 m, yaw -10 deg) with +8 deg in-hand pitch; 55 deg reaches
    everywhere (`results/usbc/policy/approach/preinsert_grasp_clearance.json`).
  - **ZED:** CNN on DR renders xy p50 0.79 / p99 3.1 mm, yaw p99 1.6 deg (sim held-out, n = 2048);
    real frames: 0.41 mm from the manual landmark, 0.05-0.08 mm frame jitter (n = 123). Classical
    silhouette: p50 11.8 mm, 21 % failures. ZED alone (open loop) gives lateral p99 21 mm (in-hand pose,
    encoder offsets): not usable without the wrist servo.
  - **Execution floor** (servo with a perfect estimate): lateral p99 0.89 mm, tilt p99 1.8 deg, with
    the Coulomb feed-forward; 2.8 mm without (joint-friction deadband).
  - **Achieved (right wrist CNN, stop 6 mm, n = 1021):** lateral p50 0.96 / p99 4.5 mm, tilt p50 1.1 /
    p99 5.7 deg, gap error p99 3.4 mm, roll p99 8.9 deg; 46 % inside 1 mm / 2 deg. Averaging 4 frames
    does not help (the error is a per-episode bias: camera pose, appearance, in-hand pose).
    Stop distance barely matters: lateral p50 / p99 3 mm 1.05 / 4.6, 6 mm 0.96 / 4.5, 10 mm 0.92 / 5.5;
    tilt p99 6.7 / 5.7 / 5.8 deg (n = 1021-1023 each). Recommended: stop at 6 mm. Part 4 must absorb
    at least lateral +/-4.5 mm, tilt 5.7 deg, roll 9 deg (p99) -- or the approach needs a better view.
  - **Next:** a second view that sees the plug from the side (the left wrist at the press pose sees
    only the PC top); an estimator trained on the in-loop distribution (DAgger-style); roll correction
    at the stop gap; a presence check for the ZED CNN.

## Part 3: Left-arm hold

- **Goal:** the PC does not move or tip during insertion.
- **Approach:** scripted IK brace (rear face) or press (top) from the placement search, set
  before Part 2.
- **Explore:** brace vs press; target offset into the PC (holding force); timing.
- **Metrics:** PC translation/rotation during insertion; contact force; collisions with the right arm.
- **Status (2026-10-31, `policy/approach-hold`): recommended hold found; gate met only without load.**
  Code: `blind_insertion/envs/left_hold.py` (`LeftHold`, `Harness`, `RightInserter`),
  `scripts/task/eval_left_hold.py`, `scripts/task/sweep_left_hold.sh`; results `results/usbc/policy/hold/`.
  - **Recommended hold:** press the closed fingertips down on the PC top, front half (2 cm in front of
    the case centre), in line with port 2; 35 N; **guarded engage**: descend at 10 mm/s from a 15 mm
    stand-off until the encoder-estimated force (PD torque through J^+T, bias at the stand-off)
    exceeds 3 N, latch the measured pose, then ramp dq = Kp^-1 J^T F over 0.4 s. Engage takes
    ~4-6 s; engage it before the approach (it moves the PC: p99 0.27 mm, max 3.1 mm, n = 1024)
    and take the ZED estimate after it, or re-estimate.
  - **Why guarded:** an absolute IK target with the force offset puts 2.7-7 N of parasitic
    sideways force on the PC (encoder-offset error x the coupled arm stiffness); the guarded
    engage gives 0.4-1.4 N. Arm Cartesian stiffness is only 0.3-0.5 N/mm (kp 80 Nm/rad), so a
    position offset is not a force command.
  - **PC motion (full DR, n = 1024 each, 1.5 s hold + 0.5 s ramp, +/-15 % lateral load):**
    fixed push at the port along the insertion axis 0 N **99.8 %** (2 non-finite worlds), 1 N 99.3 %,
    2 N 97.3 %, 3 N 96.0 %, 5 N 94.0 %, 10 N 84.4 %. No arm-arm contact in any press run.
    Brace (rear face): 6.5 % (n = 1024, push U(0, 20) N). Rear-half press: 54 % at 35 N (n = 2048,
    push U(0, 20) N); front half beats rear half (pitch moment of the port load).
  - **Limit:** the sim lets the PC slide at 1-5 N with 35 N down and feet mu >= 0.3 (Coulomb capacity
    >= 11 N); 2-6 % of worlds break loose. Stiffer feet contacts (solref 5 ms) made it worse. This
    is the soft-friction / box-feet contact, not the hold. Left joint 2 holds ~15 N m at 35 N
    (DM4340 rated ~9 N m): keep the hold short or use 25 N.
  - **Open:** privileged scripted insertion jams at the panel (45 um clearance, no chamfer;
    3.8 % seated under full DR), so the insertion load is undefined until Part 4 exists; real
    USB-C insertion is 5-20 N, above what this sim hold keeps still.

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

## Order of work

1. Shared DR (control, contact, appearance, lighting, camera) in the bimanual runtime.
2. Part 4 baseline (blind) from synthetic pre-insert starts → wrist-camera variants.
3. Part 3 in the GPU runtime under insertion forces.
4. Part 1 with the new grasp → Part 2 → chain all four; end-to-end gate.
5. Real: Part 4 from a hand-placed pre-insert pose, then the full chain (`real/duchamp3`).
