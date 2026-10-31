# Policy plan: pick → approach → hold → insert (draft, 2026-10-31)

Guide: the key-box study ([slides](../research/keybox_sim2real.pptx)). Status and gates roll up
into [CURRENT](../campaign/CURRENT.md). This page is the scaffold: four parts, what each must
achieve, what to explore, and how it is measured. Each part is developed and validated in sim
first; real tests start only after its sim gate passes.

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
| Contact | plug–port friction; pad friction and stiffness; PC-feet friction | to set; to set; μ U(0.3, 0.9) |
| Layout | PC pose; reader pose | PC nominal ±3 cm, ±10°; reader: a compact window between the right arm base and the PC (user decision 2026-10-31: smaller is fine if it raises success), yaw ±30° to start; defined by the pick agent from reach + lift reliability |
| Grasp | in-hand reader position and tilt | to set from real grasp captures (new grasp) |
| **Appearance** (slide 15) | 16 groups (PC case/dark parts, reader body/trim/plug, table, wall, room, raceway, plates, backdrop black, accents, mount, arm dark/light, gripper, fingertips): random HSV colour (p 0.75; tips 0.25) else nominal ±20 %; specular U(0, 1), shininess U(0, 1); table and wall patterns p 0.5 (8 procedural, 6 real-photo patches, 40 % real), 2–12 repeats | per episode (done: `BimanualCfg.visual_dr`, `envs/visual_dr.py`) |
| **Lighting** (slide 15) | key point light: elevation 35–90°, azimuth 0–360°, 0.9–1.6 m, intensity 0.3–1.0 (0.65 = nominal), 3200–6500 K, shadows p 0.5; fill ×0.2–1.2; ambient ×0.5–1.5; headlight fixed (Warp compile-time) | per episode (done) |
| **Camera image** (slide 15) | per-channel gain 0.93–1.07, brightness ±12, exposure ±6 %, Gaussian noise σ 1–4 (per-episode level), blur σ 0–1 px, radial k1 ±0.02 | gain/brightness/exposure/noise per frame, blur/k1 per episode (done) |
| **Camera pose** (slides 14, 16) | extrinsic error, random direction, magnitude U(0, 2× real): ZED 4.3 mm / 0.6°, right wrist 5.1 mm / 1.2°, left wrist 3.4 mm / 1.2° (mean = real); fx, fy ±1.5 %, principal point ±0.5 % of the sensor | per episode (done) |
| Camera timing | frame age and cadence | measured cadence model |

Rule from the deck: sim sensors are at least as bad as the real ones.
Appearance and lighting stay mild (user, 2026-10-31): the scene must still look like the real rig
(`VisualDRCfg` defaults: nominal colours ±15 %, a muted random colour in 10 % of groups, yellow
fingertips kept, real-photo patterns on 15 % of episodes, light 55–90° up, 0.5–0.85, 4000–6000 K,
shadows 30 %). Camera-image noise and camera-pose error stay at real scale.
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

## Part 2: Approach (IK to pre-insert)

- **Goal:** bring the held plug to a pre-insert pose a few mm in front of port 2, with an error
  inside the insertion policy's capture region.
- **Approach:** PC/port pose from the ZED (and the wrist camera near the port); in-hand reader
  pose from the wrist camera; IK to the pre-insert pose.
- **Explore:** pose estimators (ZED only, wrist only, both); a visual-servo refinement step at
  the end; how far out to stop (3–10 mm).
- **Metrics:** lateral and angular pre-insert error distribution vs the capture region of Part 4.

## Part 3: Left-arm hold

- **Goal:** the PC does not move or tip during insertion.
- **Approach:** scripted IK brace (rear face) or press (top) from the placement search, set
  before Part 2.
- **Explore:** brace vs press; target offset into the PC (holding force); timing.
- **Metrics:** PC translation/rotation during insertion; contact force; collisions with the right arm.

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
