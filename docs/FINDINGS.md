# Findings, deviations and ideas (lab notebook)

Running log for `docs/RESEARCH_PLAN.md`. Dated entries; newest first within each section.
Figures and data referenced here live in `results/<phase>/`.

## Status summary

| Phase | Status |
|---|---|
| 0 | Passed (per plan). The Phase 0 scripts/configs (`blind_usbc/`, `configs/usbc_flash_drive.yaml`) are **not in this repo**; recreated as needed. |
| A | Env, policy, PPO, eval harness built. Signal checks: see table below. |
| B | Blind s0, s1 (300M) and visual s0 (50M, stop rule) trained and evaluated; no-contact s0 running; s2 and no-contact s1 queued. Results match the paper (table below). |
| C | YAM model, workspace and fixture ranges set up (`configs/phase_c_yam.json`); A-S3..A-S6 run on YAM. |
| E | USB-C flash-drive variants + pool sampler in the builder; visual check done. |
| G | Gear meshing (paper's second task) built: assets, `GearEnv`, `--task gear` in train/eval, checks in `results/G/`. 15-min smoke: shaft capture 2% -> 50% in 12M steps, success not yet rising. Full run `G-gear-proprio-base-s0` + eval queued after `C-peg-proprio-base-s0`. |
| Report | `report/` (built by `scripts/build_report.py`): HTML results page in ASD-STE100 style. |

### Phase B results (seed 0 final checkpoints, 2,048 matched starts, unseen geometries)

| Metric | Blind (ours) | Blind (paper) | Visual (ours) | Visual (paper) |
|---|---|---|---|---|
| Success | 99.9% (2046/2048) | 99.4% | 99.95% (2047/2048) | 97.2% |
| Median time to seat | 0.64 s | 0.80 s | 0.38 s | 0.44 s |
| Insertions / min | 88 | 65 | 139 | 82 |
| Median lateral travel | 26 mm | 37 mm | 15 mm | 20 mm |
| Median time partly inside | 0.28 s | 0.38 s | 0.10 s | 0.22 s |
| Tilt partly inside / seated | 5.3° / 1.5° | 6.7° / 1.1° | 4.9° / 1.4° | 5.2° / 1.1° |
| Steps to 90% training success | 39.7M, 41.7M (s0, s1) | — | 7.9M | — |
| No contact loss: steps to 90% / success at 180M | 88.9M / 97.8% (s0, running) | slower (Fig. 14) | | |

OOD (seed 0, 1,024 starts each), blind / visual (paper blind / visual):

| Family | Blind | Visual | Paper blind | Paper visual |
|---|---|---|---|---|
| Training range (2,048) | 99.9% | 99.95% | 99.4% | 97.2% |
| Tighter | 97.4% | 85.0% | 93.8% | 54.1% |
| Slender | 98.1% | 84.3% | 97.1% | 62.7% |
| Very slender | 93.9% | 63.8% | 93.3% | 40.1% |
| Mild size | 96.3% | 88.7% | 97.1% | 57.7% |
| Extreme size | 85.8% | 31.4% | 91.9% | 16.4% |
| Random convex | 22.8% | 3.0% | 36.2% | 3.9% |

Blind beats visual in every held-out family by 12–54 points (B-R6 reproduced). Our visual
policy generalizes better than the paper's (plain renders, no lighting/texture DR is a
possible cause). Note: `phase_b_report.py` globbed `ood_*_slender` and also
matched `very_slender` (fixed 2026-10-20).

Probe (final s0, 52,289 states, 1,536 worlds): mean recall over the 5 occurring modes
(far, near, partial, moving, stuck): latent 70%, current obs 53%, random encoder 52%
(paper 70 / 56 / 33). The script's `balanced_acc` (58% / 44% / 43%) averages in the "off"
class (23 states, recall 0), so it understates; its bootstrap CI is unreliable for the same
reason. Swap ratios: far↔near 0.99–1.01, search↔inside 1.19–2.22 (stuck receiver highest).
B-R3: blind travels less than the privileged spiral on 96% of 1,582 starts where all succeed
(Wilcoxon p ≈ 1e-256).

Earlier 110M-checkpoint numbers: 98.5% success, 0.88 s, 23 mm; probe 70 / 50 / 51%.

The blind policy never lifts the peg off the plate (the probe's "lifted" mode never occurs),
unlike the paper's deployed policy (lifted recall 84%). Plausible cause: the uncompensated
peg weight (5 cm sag) makes lifting costly, so pressing-and-sweeping dominates.

### Phase A check table (Flexiv Rizon 4S, study pegs)

| Check | Result | Pass? |
|---|---|---|
| A-S1 parity | One 20 ms control step: CPU vs GPU ≤ 0.04 mm (all 100 starts). 2 s arm-only (peg parked): median 0.048 mm, p90 0.069 mm, 98% < 0.1 mm (2 outliers hit the fixture with the gripper). With the peg grasped: median 0.22 mm, but p90 93 mm (chaotic grasp stick-slip under random shaking; CPU float64 even drops the peg in 2/20 worlds where GPU does not). | Pass for arm dynamics; trajectory-level parity with contact is not achievable (chaos), see below |
| A-S3 step response | 0.05 rad steps, ≤ 1 oscillation cycle, max overshoot 12% | Pass |
| A-S5 contact signal | Vertical press: deflection monotonic in force; F_min = 0.30 N at σ_q = 1e-4 rad, 0.38 N at 5e-4. Lateral wall press: noisy/non-monotonic (peg pivots in the grasp) | Pass (vertical) / partial (lateral) |
| A-S6 drop-in | Capture visible as lateral joint velocities snapping to zero; joint-velocity peak-to-noise 3.2 at σ_q = 1e-4 (EE-z PNR only 1.2 because the scripted descent is slow) | Pass (by eye), quantified |
| A-S7 leak | Actor obs and actions bitwise identical when peg pose/vel, fixture pose and critic inputs are corrupted | Pass |
| A-S8 determinism | **Fails bitwise**: replaying the same CUDA graph from the same state differs by up to 1e-3 after 20 ms | Fail (engine property), documented |
| A-S9 throughput | 18k / 34k / 43k / 43k env steps/s at N = 1k / 4k / 8k / 12k (physics only); 36–38k with PPO at 12,288 worlds -> 300M steps in about 2.3 h | Report |
| Baselines | 2,048 matched starts: zero action 2.9%, random (σ 0.5) 6.1% success | Report |

## Next steps (as of 2026-10-20 ~03:15)

- Queue (`runs/queue/jobs.txt`, runner `scripts/queue.sh` in background): eval + probe of
  blind s0 and `scripts/phase_b_report.py` -> blind s1 -> visual s0 -> no-contact s0 ->
  YAM C-R1 s0 -> USB-C E1 s0 (curriculum) -> blind s2 -> no-contact s1.
- After each run: `scripts/eval_phase_b.sh <run> <tag>` (Phase B) or
  `scripts/eval_phase_c.sh <run> <tag>` (Phase C robustness + gain sweep), then the report.
- Not yet done: visual-policy evaluation and B-R2/B-R6 comparisons (need the visual run),
  Phase C training-with-noise runs, Phase D grid (`--family scale:<o>:<c>`; ~12 GPU-days,
  server), Phase E E2–E8, Phase F, RoboSimStudio replay (no `MjfitSpec` in this checkout).

## Discoveries

### 2026-10-20 — Gear meshing (Phase G)

- **One mesh per gear, rotated by geom slots.** All teeth of a gear are identical, so each
  gear set needs only ~7 meshes (gear tooth, gear hub sector, shaft, and per anchor a tooth
  and a hub); T geom slots use the same mesh with different `geom_pos/quat`. 1,024 gear
  sets build in 6.8 s (6,213 meshes). Random-tooth OOD gears need one mesh per tooth.
- **Hinged anchors work as children of the mocap fixture**, with per-world `body_pos`
  (centre distance and azimuth), `dof_damping` and `dof_frictionloss` (all batched in
  MuJoCo Warp; verified to take effect per world).
- **CPU MuJoCo ignores per-variant geom frames when `geom_sameframe` is set.** Slots
  compiled with identical placeholder meshes are flagged "same frame as the body", and
  `mj_kinematics` then skips `geom_pos/quat`: every anchor tooth rendered at the anchor
  centre (a plain disc). MuJoCo Warp does not use the flag, so GPU physics was correct.
  Fix: `gear_builder.build` clears `geom_sameframe`/`body_sameframe` on part bodies.
  (The peg builder may have the same latent issue for CPU renders of multi-piece pegs.)
- **MuJoCo Warp sizes multi-CCD scratch buffers by the largest mesh polygon x `naccdmax`.**
  mjfit's anchor hub (a polygon through every root point, ~4T vertices) gave
  `npolygonmax` = 72 and a 2.7 GB allocation that ran out of memory at 12,288 worlds. Hub
  -> 24-gon inscribed in the root circle, shaft 24-gon, `nconmax` 96, `nccdmax` 48 (both are
  pooled across worlds): `npolygonmax` 24 (as the peg), 10.6 GB for the 12,288-world run.
- **Newton is stable for the hinged anchors** (no PGS needed in MuJoCo Warp): contacts
  are always between an anchor and the free gear, whose inverse weight regularizes them.
  Anchors at rest stay at rest; a free anchor spun at 3 rad/s follows
  I w' = -b w - f within 0.11 rad/s (median max error; p95 0.41), time constant 24 ms
  (`G-anchor_spin.png`, n = 256).
- **Drop test (coaxial, level, 1 mm above the shaft tip, gripper collision-free;
  `G-drop.png`, n = 256 per phase):** seats in 100% / 99.6% / 98.4% / 81% / 10% / 0% / 0% of
  drops at tooth-phase errors 0 / 0.02 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 pitch. Out-of-phase gears
  stop on the anchors' tops (median 1.5 mm below the top face, because a gear resting on one
  anchor tilts within the bore play). The capture band in phase is about +-0.1 pitch.
- **Soft fixture contacts make tooth contact squishy.** With the paper's 3.5–7.1 ms fixture
  time constant, gear-anchor penetration is 0.06 mm (p99) when seated in phase but up to
  0.8 mm for a gear resting tilted on a tooth edge, comparable to the 0.4–0.6 mm backlash.
  Measured play of a meshed anchor (+-torque, `G-backlash.png`): 1.05x the designed
  backlash at 3 mN m (p10–p90 0.82–1.29; the spread is the gear's bore play, which moves the
  centre distance), 1.97x at 20 mN m.
- **The task is feasible with the robot's grasp**: a privileged scripted controller
  (aligns the axis to 0.2 mm, turns to zero phase with anchor 0, then turns to fix the worst
  anchor while pressing 1 N) seats 89.0% of 1,024 unseen gear sets (1 anchor 92.4%, 2 anchors
  85.3%), median 4.6 s. Its first version (no lift above the tip, gain 10) reached 13%:
  aligning a 0.375–0.5 mm radial bore play with the compliant kp = 40 arm needs care.
- **Baselines (2,048 matched starts, unseen gear sets):** zero action 0.05% success, random
  (sigma 0.5) 0.73%.
- **Smoke training (12,288 worlds, 15 min, shared GPU):** 14.4k env steps/s (YAM training
  running alongside), 10.6 GB. In 12.2M steps training capture rose from 2% to 50%;
  training success stayed at 0–0.4%. The 9.8M checkpoint (deterministic, 2,048 starts):
  0% success, 18% captured (zero action: 5.6%). The peg had 11% success at the same step
  count, so the gear learns slower at first; whether it reaches the paper's >99.9% is open
  until the queued 300M run finishes.

### 2026-10-20 — Visual pipeline, YAM, USB-C

- **USB-C plateau (78% on curriculum stage 1 for 160M steps) is a contact-generation artifact,
  not a learning problem** (full analysis: report_usbc/, results/E/fail/). The run was stopped at
  160M steps.
  - Trained policy, stage 1, 2,048 matched starts: 80.9% success; 87% of failures are grasp slips
    (median 0.18 s into the episode); only 7 episodes never find the port. Start offset and the
    16 s episode limit change nothing.
  - Zero action slips in 12.5% of resets (median 0.08 s); 73% of those starts also slip under the
    policy. Solver iterations (100/50) do not help (12.2%); elliptic cone + impratio 10 helps a
    little (10.8%); pyramidal + impratio 10 is 4x worse on the GPU (43.7%).
  - Cause: pad-drive contact count. With 8 contacts (4 per pad) 1.3% of worlds slip; with 0-6
    contacts 75% slip (277 of 2,048 worlds). The in-hand tilt turns the drive face up to 5 deg in
    the pad plane; narrow bodies (< ~20 mm) then overlap the 14.4 mm pad face only partly, and
    MuJoCo's box-mesh multi-contact returns an edge (2 points) or nothing. Bodies >= 22 mm always
    get 8 points. Removing in-hand tilt removes all zero-action slips. CPU MuJoCo also produces the
    partial-contact states (in different worlds), so it is not only MuJoCo Warp.
  - Fix: `EnvCfg.pad_spheres = 4` (`builder.sphere_pads`): each pad's contact is a 4x4 grid of
    1.2 mm spheres flush with its inner face (box kept as visual, collision off). Zero-action slip
    0/2,048; the SAME box-pad policy seats 98.3% on stage 1 (slips 2/2,048). A 3x3 grid gives NO
    contact (probably a GJK degeneracy with spheres on the drive's symmetry plane): use 4x4.
  - The compiler fuses the joint-less YAM pad bodies into the fingers; spec-local geom poses are
    then in the wrong frame and a spec contype edit on right_pad did not reach the compiled model.
    `sphere_pads` uses compiled poses, and `build` switches the boxes off after compiling.
  - Retraining queued: E-usbc-proprio-currsph-s0 (configs/phase_e_usbc_spheres.json).
  - Not pursued (Yash, 2026-10-20): the CPU-replay drift and a sphere-pad re-check of the peg
    tasks do not affect the results.

- **YAM blind peg policy (C-peg-proprio-base-s0) trains faster and seats faster than the Flexiv.**
  300M steps in 2.45 h; training success 22.5% at 3.9M, 82.5% at 11.8M steps (Flexiv: 57% at
  15.7M) and 99.9% at 299M, with a mean episode of 16 steps (0.32 s) against 36 on the Flexiv.
  Plausible causes: the YAM is ~3x stiffer (14 mm sag vs 48 mm), its action scale is 0.04 rad,
  and its arm bodies are gravity compensated. Eval (scripts/eval_phase_c.sh, tag yam_s0) queued.
- **USB-C curriculum run (E-usbc-proprio-curr-s0)** at 86.5M steps: 73.8% training success.

- **The visual policy learns ~5x faster in steps but is ~4.4x slower per step** (8.5k vs
  37.7k env steps/s; the PPO update with the CNN dominates, 10 s vs 5 s rollout per
  iteration). It reached 90% training success at 7.9M steps (blind ~40M) and stopped at 50M
  under the stop rule. In distribution it is faster than blind (0.38 s vs 0.64 s), as in
  the paper.
- `results/A/wrist_rgb_grid.png` and `wrist_compare.png` are stale renders from before the
  camera fix (identical views); use `wrist_gpu_nv4.png`.

- **Batched rendering is fast enough for direct RL**: MuJoCo Warp's ray-traced renderer
  does 4,096 worlds of 64x64 RGB+depth in ~25 ms (~160–200k frames/s while sharing the GPU
  with training), 8–10x the plan's 20k frames/s threshold. Per-world mesh swaps
  (`geom_dataid`) render correctly (it builds BVHs for all meshes).
- **Camera poses update only in `forward`/`camlight`** (not `kinematics`); the first
  render test showed identical stale views in every world.
- **Visual PPO needs care with history state**: storing 30-frame histories per sample would
  need ~TB, so frames are stored once per step and windows rebuilt by index, minibatched by
  world. A built-in check compares recomputed vs rollout log-probs on the first minibatch:
  it exposed that carried-over embeddings computed by the previous CNN weights broke
  consistency (up to 0.4 nats); re-embedding the carried window each rollout fixes it
  (≤1e-6).
- **YAM workspace with the gripper pointing down is small and yaw-constrained**: the
  wrist limits only allow a downward gripper near 180° yaw (J6 mid-range). Down-pointing TCP
  reach: x 0.15–0.5 m, |y| ≤ 0.25 m, z −0.3–0.1 m. Phase C fixture ranges: x 0.25–0.45,
  y ±0.2, z −0.30–−0.10 m (top face), nominal gripper yaw 180°.
- **YAM is ~3x stiffer than the Flexiv at its assumed gains**: TCP sag under the study
  peg is 14 mm (Flexiv 48 mm); F_min (vertical press, σ_q 1e-4) 0.79 N vs 0.30 N (coarse
  grid); step response passes (≤1.5 cycles, 11% overshoot).
- **Soft gripper-pad contacts let parts pivot or pass through the pads.** MuJoCo's soft
  contact is mass-normalized, so a light part squeezed hard behaves like a ~2,000 N/m
  spring: with Menagerie-style pad parameters (solref 4 ms, solimp 0.95/0.99) a 3–15 g
  USB-C drive gripped at 28–57 N had the pads sink through it (2,431 slip terminations
  per 51k env steps; CPU replay reproduces GPU exactly, so it is physics, not the GPU).
  On the YAM the same softness let 0.35–0.68 kg study pegs pivot in the small 14 x 16 mm
  pads. **Fix: pad solref 2 ms, solimp (0.99, 0.999, 0.001)**, plus a 6 N base grip
  (x 1.41–2.83) for USB-C: slips drop 10x, median pad penetration 0.035 mm; ~1% of
  randomized (tilted) grasps still pass through, tracked as a Phase E validity item.
  With stiff pads the YAM A-S6 slide captures cleanly and its **joint-velocity
  drop-in peak-to-noise is 10.7 vs 3.2 on the Flexiv** (σ_q = 1e-4 rad): the lighter,
  stiffer YAM shows contact events more clearly.
- **Backlash** is modelled as a second hinge on the same body and axis as each motor joint
  (dead zone ±b/2, 0.05 N m/rad), invisible to the encoder FK (`EnvCfg.backlash_arcmin`).
- **YAM fingertips extend ~15 mm past the pad centre**: holding a USB-C drive mid-body
  puts the fingertips at the port housing. Finger links were collision-free (for speed),
  which would have hidden this; distal finger geoms now collide with the fixture (bit 8),
  and the drive is held near the top of its body.

### 2026-10-20 — Physics and control properties

- **MuJoCo Warp is not bitwise deterministic** for this scene. Replaying the same captured
  CUDA graph from an identical state gives qpos differences up to 1e-3 (peg quaternion)
  after one 20 ms control step, and two same-seed runs diverge by O(1) over 1000 steps.
  Probably atomic-order-dependent contact/constraint assembly feeding a float32 solver.
  Consequence: "matched starts" give paired statistics, not identical rollouts. A-S8
  cannot pass as written; the meaningful test is distribution-level reproducibility.
- **The arm sags ~5 cm under the peg's weight at kp = 80.** With kp = 80 N m/rad on every
  Flexiv joint the TCP Cartesian stiffness is only ~100–300 N/m, so a 0.35–0.675 kg peg
  (not gravity-compensated: the payload is unknown to the controller) pulls the TCP down
  by ~48 mm in free space (A-S5). Holding the peg's weight costs an action of ~2 units at
  the shoulder (action scale 0.02 rad). Pegs therefore always press on the plate with
  their full weight unless the policy lifts. Because the arm sags, the free-to-contact
  transition sits ~5 cm above the nominal target in scripted tests.
- **Scripted Cartesian control is hard on this arm.** With `q_target = q_meas + 0.02 a`,
  torque is proportional to the action and the arm is very compliant, so plate friction
  (μ 0.5–2 times 3–7 N weight) stalls velocity-style scripted motions; stick-slip
  dominates. Scripted checks use moving references or absolute IK targets with a
  weight feed-forward. This is also why the spiral baseline needs care (A-S10, B-R3).
- **The grasp is a pivot.** Under vigorous motion the peg rotates about the pad contact
  (the same DOF the paper randomizes as "in-hand tilt"). This makes long-horizon
  CPU-vs-GPU parity chaotic but is physically plausible.
- **Training statistics need desynchronized episodes.** All worlds start together, so
  early iterations only see short (often successful) episodes or simultaneous timeouts;
  success swung between 2% and 75% across iterations. Fixed with random initial episode
  phases (`reset_all(desync=True)`).

### 2026-10-20 — Environment and tooling

- **MuJoCo Warp computes static geom poses once.** Geoms of bodies welded to the world
  get `geom_xpos/xmat` only in `make_data`; per-world `body_pos` or `geom_pos` changes are
  ignored afterwards. The first env version let pegs fall through plates. Fix: the
  fixture is a **mocap** body (pose via `d.mocap_pos/quat`), whose geoms are recomputed
  every step, which also makes per-world geom slot frames work.
- **Per-world geometry is natively supported.** `Model.geom_dataid` is batched, and every
  narrowphase reads `geom_dataid[worldid % nbatch, geom]`. One model holds all variants'
  meshes; each world selects its assembly by writing `geom_dataid` and the mesh-dependent
  frame fields (`geom_pos/quat/size/aabb/rbound`) plus the peg's inertial fields. No
  rebuild, so the plan's O1/O2/O3 trade-off disappears.
- **Model build time is quadratic in the number of meshes** (MjSpec): 1,024 variants
  build in 36 s, 2,048 in 132 s; 12,288 would take over an hour. We use K = 1,024
  distinct assemblies shared by 12,288 worlds (12 worlds per assembly, different poses,
  grasps and contact parameters).
- **mjlab conflicts with our MuJoCo pin** (mjlab 1.6 requires `mujoco~=3.11`). Using
  plain MuJoCo Warp + our own PPO.
- **The paper's arm is in MuJoCo Menagerie** (`flexiv_rizon4s`): used for Phase B instead
  of a Panda stand-in. Armatures 3.17 / 1.38 / 0.13 kg m^2 vs the YAM's 0.032 / 0.0018.
- RoboSimStudio's YAM models (and Menagerie's) set `gravcomp="1"` on all arm bodies;
  A-S4 gravity sag therefore comes only from the held part.
- MuJoCo Warp prints `linesearch iterations limit reached` from inside kernels at
  `ls_iterations=20`; CPU MuJoCo needs 1–3 Newton iterations on the same states and the
  results agree. Printing is disabled (`opt.warn_overflow=False`) and the per-world
  overflow bitmask is counted and logged instead (`ov_*` columns in `train_log.csv`).
- `njmax=320` overflowed (nefc up to 337); raised to 512, `nconmax` 64.

## Deviations from the paper / plan

| Item | Paper / plan | Ours | Why |
|---|---|---|---|
| Gear: bore clearance | "bore 0.75–1.0 mm" | Diametral (0.375–0.5 mm radial play) | Radial 1 mm play would exceed the 0.25-module tip–root clearance and let teeth jam into anchor roots |
| Gear: backlash | 0.4–0.6 mm per flank | Centre distance solved so the circular backlash on the operating pitch circles is 2x the per-flank value (exact involute relations); verified 1.05x in sim at low torque | Concrete definition |
| Gear: anchors | 1–2 anchors, tooth count not stated | 1 or 2 (50/50), 16–24 teeth, tooth width 0.45–0.47, same module, pressure angle and face width as the gear; azimuth 70–290 deg apart; printed density 1,250 kg/m^3 | Not stated |
| Gear: mass | 128–202 g with boss | Linear in volume (pitch disc minus bore + boss), mapped so the study range spans 128–202 g; inertia from the uniform-density geometry scaled to that mass | "set by their geometry" (App. A.2) |
| Gear: boss | 18 mm printed boss | 18 mm, 30 mm tall, 16-segment ring with a 6.6 mm bore (never touches the shaft) | Height not stated; shaft sticks out into the boss |
| Gear: collar / plate | collar the gear seats on | Collar r 10 mm x 5 mm, plate 200 mm square; no chamfers on shaft or bore | Not stated |
| Gear: polygonal round parts | involute gears, round shaft | Bore = T-gon of inradius r_bore (one hub sector per tooth); shaft 24-gon on r_shaft (max extra play 0.04 mm); anchor hub 24-gon in the root circle | Mesh count / MJW polygon buffers |
| Gear: start phase | "teeth out of phase" | Each anchor's hinge angle set so the tooth-phase error is +-U(0.2, 0.5) pitch | Concrete definition (drop test: errors <= 0.1 pitch seat by themselves) |
| Gear: start height | "resting on the shaft tip" | Lowest gear point over the shaft top (or over an anchor, when the tilt makes it lower) 0.5 mm above it | Concrete definition |
| Gear: mesh bonus | "first mesh with every anchor" | One +1 bonus the first step all anchors are meshed (a tooth-tip point inside an anchor's tip circle 1 mm below its top face) | Ambiguous (could be one per anchor) |
| Gear: labels | free / on gear / inside bore; free / on anchor / in mesh | Rim point inside bore: radial < r_bore and above the gear's bottom plane; on the gear: within 1 mm below it. Tip point in mesh: inside an anchor's tip circle and > 0.5 mm below its top; on anchor: within 0.5 mm of the tip circle and < 1 mm above the top | Concrete definition |
| Gear: critic | 434 inputs | 33 obs + gear pos/6D rot/lin+ang vel (15) + 128x3 one-hot labels + 2 anchor phase errors (pitches) = 434 | The paper's count leaves 2 values; phase errors are the natural privileged state |
| Gear: termination | "far from the fixture" | Gear > 35 mm off the shaft axis or > 60 mm above the seat; slip > 20 mm | Concrete definition |
| Gear: distinct sets | one per world | 1,024 gear sets, each used by 12 worlds | As for pegs |
| Framework | mjlab + rsl_rl | MuJoCo Warp + own PPO (rsl_rl-style GAE, timeout bootstrap) | mjlab pins MuJoCo 3.11 |
| Phase B robot | Flexiv Rizon 4S (real, identified) / plan: Panda stand-in | Menagerie Flexiv Rizon 4S; armature from Menagerie; no identified damping/friction | Exact arm available |
| Gripper | Robotiq 2F-85 | Simplified 2F-85: two coupled slide fingers, 22 x 37.5 mm pads, constant-force motors (20 N x scale 1.41–2.83), 0.925 kg | Linkage gripper is costly; force-capped grasp is what matters |
| Distinct assemblies | 12,288 (one per world) | 1,024, each used by 12 worlds | Quadratic MjSpec build time |
| Hole | through-hole? (not stated) | Closed hole, 5 mm floor | "Seated" needs a floor |
| Success | "entire bottom face seated" | All 64 bottom-edge points inside the opening and within 1.5 mm of the floor | Concrete definition |
| Capture bonus | "first capture by the opening" | First step with all 64 bottom-edge points inside the opening, below the top face | Concrete definition |
| Termination | "far from the fixture" | Peg xy beyond plate edge + 20 mm, or 50 mm above the top face, or >20 mm grasp slip | Concrete definition |
| Actions | not stated | clipped to ±3 | Safety |
| EE orientation obs | Euler angles | Euler xyz of R_nom^T R (R_nom = gripper pointing down), avoiding the ±π wrap | Continuity |
| Critic input | 240 values | 33 obs + peg pos (3) + 6D rot + lin/ang vel (6) + 64x3 one-hot labels = 240 | Matches count |
| TCN | not specified | 3 Conv1d (64 ch, k=5, strides 1/2/2) + MLP(128) -> 32 | |
| Contact loss weight | not specified | 1.0 | |
| Resets | not specified | Per-world start-state pool (4 per world, IK on GPU), refreshed every PPO iteration | Per-step IK made rollouts 3x slower |

## Compute budget (for moving work to a server)

### 2026-10-20 — Server migration to mason (bootstrap passed; validation running)

- Actual host: `mason`, 5 × Tesla V100-PCIE-32GB, driver 580.82.07, compute
  capability 7.0. Setup is pinned to GPU 0 after repeated idle capacity checks;
  all five GPUs initially used <50 MiB with 0% compute and no compute processes.
- Branch: `jensen-runs`. Bootstrap completed in tmux `peg-setup`, log
  `runs/bootstrap.log`; `runs/bootstrap.exit` is 0.
- Pinned submodules and Menagerie installed. CUDA 12 stack: torch
  `2.14.1+cu126`, Warp `1.18.0+cu12`, MuJoCo/MuJoCo Warp `3.12.0`.
  Call `.venv/bin/python` directly; `uv run` / `uv sync` restore CUDA 13 binaries
  that do not support V100.
- Server verification passed all six checks, including GPU matrix multiplication,
  EGL rendering and MuJoCo Warp stepping (`runs/_verify.json`).
  `uv pip check` passed (151 packages); all 381 manifest commands passed parser
  validation (`runs/manifest-check.log`). CPU assembly integration also passed
  (`runs/check-setup.log`). Bootstrap blind training (2 PPO iterations), 64-episode checkpoint evaluation,
  and visual training (2 PPO iterations) all passed; bootstrap ended `ALL OK`.
- Full-size blind benchmark passed (12,288 worlds, 12 PPO iterations): median
  **20,490 steps/s**, peak **10,706 MiB**, no OOM, 327.1 s including model build.
  Estimated 300M-step runtime **4.07 hours**, about 1.83× the 37.5k local reference.
  Evidence: `results/server_setup/mason/benchmark_blind_12288.json`.
- Full-size visual benchmark passed (4,096 worlds, 12 PPO iterations): median
  **3,922 steps/s**, peak **28,558 MiB**, no OOM, 528.1 s including model build.
  Evidence: `results/server_setup/mason/benchmark_visual_4096.json`.
  A 300M-step cap would take **21.25 hours**; 50M takes ~3.54 hours if the existing
  stop-success rule triggers there. Peak memory is much higher than the manifest's
  informational `mem=14` estimate; keep one visual job per 32 GB V100.
  GPU 1 released on completion.
- Prescribed 300M-step validation `B-peg-proprio-base-s3` started 17:59 UTC on
  GPU 0 in tmux `peg-validation`. Runner state is `runs/queue_validation/`, log
  `runs/B-peg-proprio-base-s3.log`; completion marker is written only on exit 0.
  No retries (a failure needs inspection). Compare success >90% at 35–50M and
  >=99% from 150M before starting a production batch.
  First checkpoint at 9,830,400 steps (iteration 25) loaded strictly on CPU;
  all model tensors and policy/value forward outputs were finite. Training success
  at this checkpoint was 40.2%. Evidence:
  `results/server_setup/mason/validation_checkpoint_smoke.json`. This is a checkpoint
  integrity check; the learning-curve gate is still pending.
- All README test suites passed: mjfit **265 tests** (528.70 s), RoboSimStudio fast
  **1,728 tests, 27 skipped** (160.19 s), RoboSimStudio CPU **19 tests** (111.59 s).
  Logs: `runs/test-mjfit.log`, `runs/test-robosimstudio-fast.log`,
  `runs/test-robosimstudio-cpu.log`.
- Shared-server runner hardened with opt-in `--require-idle-gpus`: three idle
  samples followed by a launch-time recheck; busy or unreadable GPUs keep jobs
  queued without moving partial runs or counting failed attempts. Six regression
  tests passed. Live queue smoke skipped the active validation GPU 0 and dispatched
  a CPU-only environment check on idle GPU 1; recorded `CUDA_VISIBLE_DEVICES=1`
  and exited 0 (`runs/_runner-idle-smoke/runner.log`). Training code is unchanged.
- No production batch has started. D/E retain their Gate C noise decision
  requirement; local-queue jobs and missing local checkpoints must be skipped.

One 300M-step run = ~2.3 h on the local RTX 3090 Ti (12,288 worlds, ~37k env steps/s).

| Work | Runs | Local time | Where |
|---|---|---|---|
| B-R1/B-R7 blind with/without contact loss, 3 seeds each | 6 | ~14 h (+ no-contact runs may need 2x steps) | local, sequential |
| B visual policy (rendering) | 3 | unknown; renderer throughput not yet measured | likely server |
| C YAM blind + noise/backlash/latency + gain sweeps (train) | ~20 | ~2 days | **server** |
| D size x clearance grid, 40 cells x 3 seeds | 120 | ~12 days | **server** |
| E USB-C curriculum + ablations | ~15 | ~1.5 days | local or server |
| F six sensing variants x 3 seeds (rendering) | 18 | days | **server** |

## Ideas to explore


- (2026-10-20) Gear: the meshed anchor turns with the gear and the other one slips under
  it only if the press is light and the hinge friction high; a probe on "pressing hard vs
  lightly while meshing" could show whether the blind policy learns this force regulation.
- (2026-10-20) Gear: train with stiffer fixture contacts (2 ms, mjfit default) as an
  ablation; the paper's 3.5–7.1 ms lets teeth sink up to 0.8 mm into each other.
- (2026-10-20) Payload-aware gravity feed-forward as an ablation: compensate the nominal
  peg weight in the controller (as a real Flexiv with a configured payload would) and
  see whether learning speeds up and search gets gentler.
- (2026-10-20) Since every world can select a geometry by index, OOD evaluation sets
  (Table 6 bins) can live in the same compiled model as extra variants.
- (2026-10-20) Quantify A-S8 as distribution-level reproducibility: two same-seed
  evaluations of a fixed policy should give success rates within binomial noise.
