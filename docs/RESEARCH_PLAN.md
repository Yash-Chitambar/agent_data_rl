# Blind Proprioceptive Insertion on YAM: From Reproduction to USB-C

**Owner:** Yash (AutoLab, UC Berkeley)  **Status:** Phase 0 complete, Phase A next  **Last updated:** 2026-10-19

This document defines the research experiment end to end: what we are trying to show, the tools used for each job, every component that must be implemented, and the exact statistical and visual results that must exist before moving from one phase to the next. Each phase ends in a **gate**: a short list of results that must be produced and pass before the next phase starts. Nothing is skipped; if a gate fails, we diagnose inside that phase.

Reference paper: Bronars, Park, Kedia, Agrawal, *Blind Precise Assembly: Proprioceptive Sim-to-Real Peg-in-Hole for Diverse Parts*, arXiv:2610.23849 (MIT, 2026). Below, "the paper" means this work, and "Fig. N / Table N / §N" refer to it.

---

## Contents

1. [Research goals](#1-research-goals)
2. [Tools and their roles](#2-tools-and-their-roles)
3. [Standards used in every phase](#3-standards-used-in-every-phase)
4. [Fixed simulation settings](#4-fixed-simulation-settings)
5. [Robot model and assumed joint gains](#5-robot-model-and-assumed-joint-gains)
6. [Phase 0: Asset verification (done)](#6-phase-0-asset-verification-done)
7. [Phase A: Training infrastructure and signal checks](#7-phase-a-training-infrastructure-and-signal-checks)
8. [Phase B: Reproduce the paper's results](#8-phase-b-reproduce-the-papers-results)
9. [Phase C: Port blind peg insertion to YAM](#9-phase-c-port-blind-peg-insertion-to-yam)
10. [Phase D: Scale-down study, from large pegs to connector size](#10-phase-d-scale-down-study-from-large-pegs-to-connector-size)
11. [Phase E: USB-C flash drive insertion](#11-phase-e-usb-c-flash-drive-insertion)
12. [Phase F: Adding wrist RGB and depth](#12-phase-f-adding-wrist-rgb-and-depth)
13. [Later: hardware gains and real-robot work](#13-later-hardware-gains-and-real-robot-work)
14. [Repository layout](#14-repository-layout)
15. [Risk register](#15-risk-register)
16. [Master checklist](#16-master-checklist)

---

## 1. Research goals

### 1.1 Central question

Can a policy that senses only the robot's own joints insert **connector-scale** parts (USB-C: 8.3 × 2.6 mm opening, ~50 µm clearance per side) on a **low-cost backdrivable arm (YAM)**? If so, how much does adding wrist RGB and depth improve it, and in which regime?

The paper shows blind insertion for large pegs (30–90 mm openings, 0.5–1.5 mm clearance) on a 7-DOF Flexiv arm. It never tests small parts on the policy side, and its limitations section never mentions scale. Our work fills that gap.

### 1.2 Research questions and hypotheses

| ID | Question | Hypothesis | Answered in |
|---|---|---|---|
| RQ1 | Do the paper's results reproduce with our own implementation? | Yes, within the tolerances in §8.9. | Phase B |
| RQ2 | Does blind insertion transfer from a 7-DOF arm to the 6-DOF YAM with realistic gains? | Yes for large pegs; YAM's low gearing gives a usable contact signal. | Phase C |
| RQ3 | How does blind success degrade as part size and clearance shrink? | Graceful down to about 10 mm openings; below that, search cost and the size of the contact signal dominate. | Phase D |
| RQ4 | Can a blind policy plug a USB-C flash drive into a port in sim? | Yes from small start offsets (≤ 2 mm); success falls sharply with larger offsets. | Phase E |
| RQ5 | Where does vision help? | RGB/depth extends the capture radius (larger start offsets) but hurts generalization to unseen parts, as in the paper. Proprioception + vision beats either alone. | Phase F |

### 1.3 What counts as a contribution

1. An independent reproduction of the paper's core claims (positive or negative).
2. The first systematic **scale study** of blind proprioceptive insertion: success, search cost and contact signal vs. part size and clearance.
3. A blind USB-C insertion policy in sim on a low-cost arm.
4. A controlled comparison of proprioception, RGB, depth, and their fusions at connector scale.

### 1.4 Out of scope for this document

- Real-robot deployment (planned later, §13).
- Deformable cables and wires: the flash drive is rigid by design.
- Gear meshing. Nut threading is a stretch goal only (§8.8).

---

## 2. Tools and their roles

Each tool has exactly one job. This is a hard rule: training never touches the visualizer, and the visualizer never produces numbers.

| Job | Tool | Notes |
|---|---|---|
| Part geometry (pegs, holes, USB-C) | **mjfit** (`pip install git+https://github.com/younghyopark/mjfit`) | Exact convex pieces, clearance kept intact. Emits MJCF. |
| Physics for training and evaluation | **MuJoCo Warp** (GPU, batched) | Same physics as the paper. |
| Environment framework + PPO | **mjlab** (Isaac Lab-style API on MuJoCo Warp) with **rsl_rl** PPO | The paper's own framework (Zakka et al., 2026). Custom TCN actor is plugged into rsl_rl. If mjlab blocks a requirement (§7.2), fall back to a plain MuJoCo Warp env + rsl_rl. |
| Single-env debugging, unit tests | **MuJoCo** (C library, CPU) + `mujoco.viewer` | Ground truth for every physics check. |
| Analysis and plots | numpy, pandas, scipy, matplotlib | Every figure produced by a script from logged data. |
| Experiment tracking | Weights & Biases (or TensorBoard) | Every run logs config, seed, git hash. |
| **Qualitative visualization only** | **RoboSimStudio** | Replay logged trajectories in the browser viewer, inspect scenes, render presentation videos. **Not used for training, evaluation, or any number in a result.** |

### 2.1 How RoboSimStudio is used, concretely

- Logged rollouts (`qpos` per step for robot and parts) are replayed kinematically into an RSS scene (bench + YAM + mjfit assembly via `MjfitSpec` from `rss_mjfit.py`) for viewing.
- If any physics is ever stepped in RSS for a demo, it must use `TIGHT_FIT` (rigid gap 0, 1 kHz). RSS's default `MUJOCO_500HZ` has a 2 mm contact gap that prevents a USB-C plug from entering its port (Phase 0 finding).

---

## 3. Standards used in every phase

### 3.1 Seeds and episode counts

| Item | Standard |
|---|---|
| Training seeds per condition | **3** (paper used 2). |
| Evaluation episodes per seed per condition | **≥ 2,048** (in-distribution); **≥ 1,024** per OOD condition. The paper used 8,192 where compute allows. |
| Evaluation actions | Deterministic (policy mean), as in the paper. |
| Matched starts | All conditions compared on the **same** set of start states (same reset seeds), so paired statistics apply. |

### 3.2 Statistics

| Quantity | Report | Interval |
|---|---|---|
| Success rate | mean over seeds; also per seed | 95% **Wilson** interval per seed; seed range as error bar on the mean |
| Times, distances (skewed) | **median** | 95% bootstrap CI (10,000 resamples) |
| Paired comparisons on matched starts | fraction of starts where A beats B; median paired difference | Wilcoxon signed-rank p-value; bootstrap CI on the median difference |
| Success-rate differences | difference of proportions | Newcombe interval; two-proportion z-test |
| Probe accuracy | **balanced** accuracy over modes | split by world (even/odd worlds, as in the paper); report chance level |
| Training speed | env steps to reach 90% training success | mean ± range over seeds |

Multiple comparisons: when a figure makes more than 5 pairwise claims, apply Holm–Bonferroni.

### 3.3 Required plot standards

- Every figure states **n** (episodes and seeds) in its caption.
- Training curves show the mean over seeds with a shaded min–max band; x-axis in environment steps (millions).
- Success bars carry error bars as defined above.
- Use the same color for a modality in every figure: proprioception blue, RGB orange, depth green, fusions in mixed shades. Use a colorblind-safe palette.
- Every figure has a generating script in `analysis/figures/` and a frozen data file in `results/<phase>/`.

### 3.4 Run naming and logging

`<phase>-<task>-<modality>-<variant>-s<seed>`, e.g. `C-peg-proprio-kp80-s2`.

Each run logs: full config (YAML), git hash, mjfit version, MuJoCo/MuJoCo Warp version, wall-clock time, env steps/s, training success, value loss, entropy, policy std, contact-decoder accuracy, and a 10-episode rollout video every 50M steps.

---

## 4. Fixed simulation settings

These are fixed for every phase unless a phase explicitly sweeps them.

| Setting | Value | Source / reason |
|---|---|---|
| Physics timestep | **1 ms** (1 kHz) | Paper §A.1; mjfit tunes contacts for ≤ 1 ms |
| Control rate | **50 Hz** (20 physics steps per action) | Paper |
| Integrator / solver | implicit-fast, Newton | mjfit benchmark settings (Appendix B) |
| Contact solref / solimp | mjfit defaults: solref (2 ms, 1), solimp (0.99, 0.999, 0.001) | mjfit, Appendix B |
| Geom margin and gap | **0** | Phase 0: a nonzero gap destroys sub-mm fits |
| Condim | 4 (mjfit default) | |
| Sim resolution floor | **~20 µm**; no claim may rest on effects smaller than this | Phase 0; Appendix B ("S1 cannot resolve below about 20 µm") |

---

## 5. Robot model and assumed joint gains

### 5.1 Model

- **Geometry and gripper:** the i2rt YAM model (`assets/embodiments/yam_i2rt/` in RoboSimStudio, MIT/BSD-licensed per its `LICENSE`), with the linear_4310 gripper that matches the real hardware. Copy the MJCF into our repo; do not import RSS at training time.
- **One arm only** for Phases B–F. Bimanual is future work.
- The gripper is closed on the part at reset and held at a constant command for the whole episode, as in the paper.

### 5.2 Assumed gains (until real gains are measured)

We use the Damiao-motor gains baked into RoboSimStudio's YAM arena model (`yam_bimanual_arms.xml`), which that repository describes as modeling the real hardware. These are close to the paper's kp = 40–80, so the arm is compliant enough for joint deflection to carry contact information.

| Joints | Motor | kp (N·m/rad) | kv (N·m·s/rad) | armature (kg·m²) | damping | frictionloss |
|---|---|---|---|---|---|---|
| J1–J3 | DM4340 | 80 | 6 | 0.032 | 0.5 | 0.01 |
| J4–J6 | DM4310 | 30 | 2 | 0.0018 | 0.5 | 0.01 |

- Implementation: MuJoCo position actuators with these kp/kv; the policy's action is a joint-position increment added to the **measured** joint position (the paper's scheme): `q_target = q_meas + action_scale · a`.
- Action scale: 0.04 rad for all joints (the paper's value for its gear/nut policies and deployed peg policy, which share YAM-like gains of kp 40 / kd 20).
- Torque limits: set from DM4340/DM4310 datasheet peaks. **Verify the datasheet values before Phase C**; record the source in the config.
- The i2rt model's own kp = 300 servo is **not** used: it is ~4× stiffer, which shrinks the contact-induced deflection the policy reads.

These values are an **explicit assumption**. Phase C includes a gain sweep (§9.4) so we know how sensitive results are to them. Measured gains replace them in §13.

---

## 6. Phase 0: Asset verification (done)

**Goal:** confirm mjfit's parts keep their fit in our physics stack.

**Done (scripts in `blind_usbc/`, CPU, MuJoCo 3.12, Newton 1.6):**

| Check | Result |
|---|---|
| mjfit USB-C/USB-A/RJ45/DB9 drop test, plain MuJoCo | All fall in without touching the walls; USB-C also at 20 µm clearance. |
| Rim overlap released from rest | 10 µm overlap slips in (soft contact); 50 µm catches on the rim. Defines the ~20 µm resolution floor. |
| Rim overlap released from 1 mm height | Overlaps up to ~100 µm are pushed sideways into the port (penetration resolved laterally). **Approach speed must stay realistic.** |
| mjfit through Newton (what RSS uses) | Authored solref survives. A 2 mm `rigid_gap` alone breaks the fit; 500 Hz, multi-CCD and ke/kd overrides do not. |
| Full RSS scene with YAM, `TIGHT_FIT` | Plug seats at the same height as plain MuJoCo (2.90 mm). Flash-drive config: 0.42 vs 0.45 mm. |
| USB-C flash-drive config | `configs/usbc_flash_drive.yaml`: 8 g drive, 20 × 6.5 × 30 mm body, blind Type-C port, 50 µm clearance. Builds and seats. |

**Gate 0: passed.**

---

## 7. Phase A: Training infrastructure and signal checks

**Goal:** build the batched training environment and prove, before any RL, that (1) the physics is right, (2) the controller behaves as specified, and (3) a contact signal actually exists in the joint readings. Training on a broken environment is the most expensive mistake in this project, so this phase is deliberately thorough.

### 7.1 Components to implement

| # | Component | Spec |
|---|---|---|
| A-1 | `assets/` builder | Python function that composes one MJCF: robot (§5) + mjfit assembly + table. Takes an assembly config (YAML) and a robot config. Deterministic given its inputs. |
| A-2 | Variant pool | Builds K assemblies by sampling geometry (Table 1 ranges for Phase B; `configs/` for Phase E). Each variant saved as YAML + MJCF with a content hash. |
| A-3 | Batched env (mjlab task) | N parallel worlds; reset sampler; step at 50 Hz with 20 physics substeps; per-world episode timers; done/timeout logic. |
| A-4 | Per-world geometry mechanism | See §7.2. |
| A-5 | Observation module | 30-step history ring buffer of [q (6), q̇ (6), EE pose (6: position + Euler, or 9 with 6D rotation), EE twist (6), previous action (6)]. EE pose/twist computed by forward kinematics from joint state only. Normalization by running mean/std, frozen at evaluation. |
| A-6 | Privileged state module | For the critic and labels only: part pose relative to fixture, contact labels, fixture pose. Never reaches the actor (unit-tested, §7.4). |
| A-7 | Contact labeler | 64 points on the part's bottom edge; each labeled **free** (> 1 mm above the fixture top face), **contact** (on rim/top face within 1 mm), or **inside** (below the top face, within the opening). Exactly the paper's §A.1 definition. |
| A-8 | Reward | Paper's sparse reward: +1 on the part's first capture by the opening, +1 on success; episode ends on success, timeout (5 s), or the part moving far from the fixture. |
| A-9 | Success check | Peg: entire bottom face seated in the hole. USB-C: plug within 0.2 mm of the port floor, axis aligned within 2°. |
| A-10 | Domain randomization | Table 1 (paper) for Phase B; per-phase tables below. Geometry per variant, poses and contact parameters per reset, dynamics per world. |
| A-11 | Rollout logger | Saves per-step joint state, action, part pose, contact labels, and policy latent for selected episodes (HDF5). Feeds analysis and RSS replay. |
| A-12 | Evaluation harness | Runs a frozen checkpoint on a fixed list of start states; writes one row per episode (success, time, lateral travel, max tilt, failure mode) to Parquet. |

### 7.2 Per-world geometry: decision to make in this phase

The paper gives each of its 12,288 worlds a different assembly but does not say how. MuJoCo Warp batches numeric model fields across worlds, but mesh shapes differ per variant. Options, in order of preference:

| Option | How | Pros | Cons |
|---|---|---|---|
| **O1. Variant groups** | K separate compiled models, each batched over N/K worlds; resample K variants every M PPO iterations | Simple, exact | Rebuild cost; K limited by GPU memory |
| O2. Parked pool | All K assemblies in every model; inactive ones parked far away per world via batched body positions | One model | Collision broadphase cost grows with K |
| O3. Fixed-topology meshes | All variants share vertex counts; per-world mesh vertex buffers | Fast | Depends on MuJoCo Warp support; engineering-heavy |

**Procedure:** implement O1 first. Benchmark env steps/s for K ∈ {1, 8, 32, 64} at N = 4,096 on one GPU. Pick the largest K that keeps throughput ≥ 50% of K = 1. Record the table in `results/A/geometry_throughput.csv`. For USB-C (Phase E), K ≈ 16–64 is enough because the connector is standardized; most variation is numeric (pose, grasp, friction, gains, noise).

### 7.3 Signal checks (must pass before any training)

Each check produces one plot and one pass/fail number.

| ID | Check | Procedure | Required visual | Pass criterion |
|---|---|---|---|---|
| A-S1 | Physics parity, CPU vs GPU | Same 100 scripted trajectories in MuJoCo (CPU) and MuJoCo Warp (GPU) | Overlay of joint trajectories; histogram of max position difference | Max EE difference < 0.1 mm over 2 s, no contact; < 0.5 mm with contact |
| A-S2 | Drop tests in batched env | Phase 0 drop tests repeated in MuJoCo Warp for every variant in the pool | Table: variant × fell-in (yes/no) | 100% fall in when centered |
| A-S3 | Controller step response | 0.05 rad step on each joint, free space | Step-response curves per joint (target vs measured) | Rise time, overshoot, steady-state error logged; no oscillation > 3 cycles |
| A-S4 | Tracking under gravity | Hold 20 random poses for 2 s | Bar chart of steady-state error per joint | Error recorded (gravity sag is expected and realistic; it must be **consistent**, not near zero) |
| A-S5 | **Contact-signal curve** | Press the held part straight down on the plate with commanded offsets giving forces 0.1–20 N; also lateral presses against the hole wall | Plot of joint deflection (commanded − measured) per joint vs contact force, log-log; plus EE-space deflection vs force | Deflection is monotonic in force; the force at which deflection exceeds 3× the injected encoder noise (§9.3) is reported. This is the minimum detectable force, F_min |
| A-S6 | Drop-in signature | Slide the part across the plate until it falls into the hole (scripted) | Time series of joint positions, velocities, EE z, with the moment of capture marked | The capture event is visible in q̇ and EE z by eye; quantify its peak-to-noise ratio |
| A-S7 | Information leak test | Zero all privileged inputs; confirm the actor's output is unchanged | n/a | Bitwise-identical actions |
| A-S8 | Determinism | Same seed twice | n/a | Identical rollouts for 1,000 steps |
| A-S9 | Throughput | Steps/s at N ∈ {1k, 4k, 8k, 12k} | Line plot steps/s vs N | Report; estimate time to 300M steps |
| A-S10 | Scripted baseline sanity | Run the spiral search (§8.6) on 512 starts | Success rate, median lateral travel | Spiral succeeds from flush starts (it should; if not, the env is wrong) |

A-S5 and A-S6 are the most important checks in the project. They answer, before any learning: **is the information the policy needs actually present in the joint readings?** They are rerun at every scale in Phase D and for USB-C in Phase E.

### Gate A

- [ ] All A-S1…A-S10 plots exist in `results/A/` and pass.
- [ ] Geometry mechanism chosen and benchmarked (§7.2).
- [ ] A dummy policy (random actions and zero actions) runs 10M steps without NaN or contact-buffer overflow.

---

## 8. Phase B: Reproduce the paper's results

**Goal:** answer RQ1. Reproduce the paper's core claims in simulation, in its own setting as closely as possible, **before** changing anything. If we cannot reproduce them, every later result is uninterpretable.

### 8.1 Setting

- **Robot:** a 7-DOF arm with joint impedance control, standing in for the Flexiv Rizon 4S. Use a Flexiv Rizon 4 MJCF if one is available (check MuJoCo Menagerie); otherwise the Franka Panda (Menagerie, also in RSS). Gains kp/kd = 80/20 (study peg) and an action scale of 0.02 rad, per Table 1. Document the choice; the paper also identified armature, damping and friction from its real arm, which we cannot copy exactly.
- **Task:** study peg insertion: stadium pegs, Table 1 peg column, exactly.
- **Not reproduced:** the real-robot results (Fig. 12). Gear meshing is skipped. Nut threading is optional (§8.8).

### 8.2 Policy (exact match to §2.3, §A.1)

| Component | Spec |
|---|---|
| Actor input | 33 values/step × 30 steps (7 q, 7 q̇, 12 EE pose + twist, 7 previous action) |
| Encoder | Temporal convolution over 30 steps → 32-d latent z_t |
| Actor head | MLP (256, 256, 128) on [z_t, o_t] → 7 joint-position increments |
| Contact decoder | 2 hidden layers × 128 on z_t → per-point 3-class logits (64 points); class weights free 0.25, contact 3.0, inside 2.0; cross-entropy added to the PPO loss; **dropped at deployment** |
| Critic | MLP on actor obs + part pose relative to fixture + contact labels (240 inputs for the peg in the paper) |

### 8.3 PPO (exact match to §A.1)

12,288 worlds (or the max that fits; scale minibatches to keep batch size per update constant), 32-step rollouts, 5 epochs × 20 minibatches, lr 3e-4 fixed, clip 0.2, γ 0.98, λ 0.95, entropy 0.003, value coefficient 1 with clipping, grad-norm clip 1, initial action std 0.5. Budget: 300M env steps per run (Fig. 8 converges by ~250M).

### 8.4 Visual baseline (§2.3)

Wrist-mounted camera rigidly attached to the gripper; 64 × 64 RGB; same 30-step history; compact CNN encoder replacing the joint-history encoder; everything else identical (task, resets, randomization, reward, actions, success check). Requires batched rendering in MuJoCo Warp. **Verify renderer availability and throughput in Phase A**; if rendering is too slow for 300M steps, train the visual policy for as long as needed to reach ≥ 97% training success and report its step count.

### 8.5 Results to reproduce, in this order

Each row is one experiment. "Paper value" is the target; tolerances are in §8.9.

| ID | Paper claim | Our experiment | Required statistical result | Required visual |
|---|---|---|---|---|
| **B-R1** | Proprioception alone learns peg insertion to near-perfect training success (Fig. 8) | Train blind and visual, 3 seeds each | Training success vs steps; final in-dist success (paper: blind 99.4%, visual 97.2%, Table 5) | Fig. 8-style curves, blind vs visual |
| **B-R2** | Blind is slower but reliable (Table 5) | Evaluate both on 2,048 matched starts/seed | Median time to seat (paper 0.80 vs 0.44 s), insertions/min (65 vs 82), lateral travel (37 vs 20 mm), time partly engaged (0.38 vs 0.22 s), tilt partly engaged/seated (6.7/1.1° vs 5.2/1.1°); paired tests on matched starts | Table 5 replica; Fig. 9a/b-style top-view paths and descent-vs-time for one matched start |
| **B-R3** | Blind search is more efficient than spiral search (Fig. 9c) | Spiral baseline (§8.6) vs blind policy, same starts | Median lateral travel (paper 37 vs 95 mm); Wilcoxon test | Histogram of lateral travel (log x), medians marked |
| **B-R4** | The history latent encodes contact mode (Fig. 10a) | Linear probe on z_t vs current obs vs random encoder (§8.7) | Balanced accuracy (paper 70% vs 56% vs 33%; chance 14%); recall per mode (lifted/moving/stuck 84–90%) | 7×7 confusion matrix; per-step true vs decoded mode strip for one episode |
| **B-R5** | The policy uses that encoding (Fig. 15a) | Latent-swap test (§8.7) | Action change ratio cross-mode / within-mode (paper 1.4–1.9× searching↔inside; 0.99–1.11× near↔far) | 5×5 heat map of ratios |
| **B-R6** | Blind generalizes better than visual to unseen parts (Fig. 11, Table 7) | Evaluate both on the 6 OOD bins of Table 6 (tighter, slender, very slender, mild size, extreme size, random convex) | Success per bin with intervals (paper blind/visual: tighter 93.8/54.1, slender 97.1/62.7, very slender 93.3/40.1, mild size 97.1/57.7, extreme size 91.9/16.4, random convex 36.2/3.9); difference-of-proportions test per bin | Fig. 11 peg row replica |
| **B-R7** | Contact supervision speeds learning (Fig. 14) | Blind with vs without contact loss, 3 seeds | Steps to 90% training success (paper: 2.7× more without) | Fig. 14 peg panel replica |

### 8.6 Spiral search baseline (scripted)

Archimedean spiral in the plane of the fixture surface, starting **flush with the hole opening's edge**, given the true peg orientation and the best starting direction (the paper's "optimistic" spiral). Pitch equal to the clearance-adjusted peg width; constant downward force; success when the peg drops in. Same start states and success check as the policy. This is a deliberately strong baseline.

### 8.7 Probe and latent-swap procedures

**Linear probe (§3.2, App. C):**
1. Roll out the policy in 1,536 worlds with deterministic actions; log z_t, o_t, and the privileged mode label every step.
2. Modes (peg): lifted, beside the plate (off), searching far (> 5 mm from the opening), searching near (≤ 5 mm), partly engaged, inside moving, inside stuck (< 0.5 mm progress in 0.2 s).
3. Train multinomial logistic regression on even-numbered worlds; test on odd-numbered worlds.
4. Repeat with input = current observation only, and with a randomly initialized encoder.
5. Report balanced accuracy and per-mode recall, with bootstrap CIs over test worlds.

**Latent swap:** for pairs of states (receiving, donor), keep the receiving state's current observation but substitute the donor's z_t; compute ‖Δaction‖. Normalize by the mean change for donor states in the **same** mode as the receiver. Report the 5×5 (far, near, partial, moving, stuck) matrix of ratios.

### 8.8 Optional: nut threading

Only if B-R1…B-R7 pass with time to spare. Same pipeline with mjfit's thread generator and Table 1 nut column. Reproduce the nut row of Fig. 8 and Fig. 11.

### 8.9 Reproduction tolerances

| Metric | Reproduced if |
|---|---|
| In-dist success (blind) | ≥ 97% |
| Time and travel medians | Within ±30% of the paper, **and** the same ordering (blind slower / travels more than visual; blind travels less than spiral) with p < 0.01 |
| Probe accuracy | z_t beats current obs beats random encoder; z_t balanced accuracy ≥ 55% |
| Latent swap | cross-mode/within-mode ratio > 1.2 for searching↔inside; ≈ 1 for near↔far |
| OOD | Blind > visual in every OOD bin with non-overlapping 95% intervals in ≥ 4 of 6 bins |
| Contact supervision | ≥ 1.5× fewer steps to 90% with supervision |

**Failure handling.** If a claim does not reproduce, do not proceed on assumption. Diagnose using the Phase A checks, try the paper's exact configuration on that axis, and write a short report in `results/B/deviations.md`. A well-documented non-reproduction is itself a result.

### Gate B

- [ ] B-R1…B-R7 results and figures in `results/B/`, each with a pass/fail against §8.9.
- [ ] `results/B/deviations.md` lists every difference from the paper's setup (robot model, dynamics ID, geometry mechanism, compute).

---

## 9. Phase C: Port blind peg insertion to YAM

**Goal:** answer RQ2. Same task and parts as Phase B, now on the 6-DOF YAM with the assumed gains (§5.2).

### 9.1 Changes from Phase B (and nothing else)

| Item | Phase B | Phase C |
|---|---|---|
| Robot | 7-DOF stand-in | YAM i2rt, one arm, 6-DOF |
| Actor obs/step | 33 | 30 (6 q, 6 q̇, 12 EE pose+twist, 6 previous action) |
| Gains | 80/20 | §5.2 table |
| Action scale | 0.02 rad | 0.04 rad (re-tuned once on 1 seed if training stalls; record it) |
| Workspace / fixture poses | Table 1 | Table 1 ranges scaled to YAM reach (780 mm); fixture placed where the YAM has full 6-DOF dexterity. Record the final ranges. |
| Grasp randomization | Table 1 | Same, limited by the YAM gripper's 95 mm stroke |

### 9.2 Experiments

| ID | Experiment | Required statistical result | Required visual |
|---|---|---|---|
| C-R1 | Train blind YAM peg policy, 3 seeds | Training curve; in-dist success | Curves vs Phase B blind |
| C-R2 | Table 5 metrics on YAM | Same metrics as B-R2, paired against nothing (different robot); compare distributions with Mann–Whitney | Side-by-side table Flexiv-stand-in vs YAM |
| C-R3 | Probe + latent swap on YAM | As B-R4/B-R5 | Confusion matrix, swap heat map |
| C-R4 | OOD bins on YAM | As B-R6 (blind only, plus visual if compute allows) | Fig. 11-style bars |
| C-R5 | Contact-signal curve on YAM vs 7-DOF arm | A-S5 on both robots | Deflection vs force, both robots on one plot |

### 9.3 Robustness additions the paper did not test

The paper adds no observation noise (Table 1). Real encoders and gearboxes are not perfect, and this matters more as parts shrink. Add each separately, then all together:

| Factor | Model | Levels |
|---|---|---|
| Encoder quantization | Round q to the encoder resolution | 14-bit and 16-bit per revolution (verify the DM motor spec) |
| Encoder noise | Gaussian on q, q̇ | σ_q ∈ {0, 1e-4, 5e-4} rad |
| Backlash | Dead-zone hinge in series with each joint (small passive hinge with a narrow range limit, low stiffness inside the zone) | {0, 1, 3, 5} arcmin |
| Latency | Action applied k control steps late | k ∈ {0, 1, 2} |

For each factor, report:
- success when **evaluating** a noise-free-trained policy under the noise (brittleness);
- success after **training** with the noise randomized (robustness).

Required visual: a grouped bar chart per factor.

### 9.4 Gain sensitivity

Evaluate the C-R1 policy, and train new policies, at gain multipliers {0.5, 1, 2, 4} × §5.2 (×4 ≈ the i2rt kp 300 servo). Report success and F_min (A-S5) at each. Required visual: success vs gain multiplier, with F_min on a second axis. This tells us how much the real gains (§13) will matter.

### Gate C

- [ ] YAM blind in-dist success ≥ 95%.
- [ ] Probe shows z_t > current obs on YAM.
- [ ] Noise, backlash and gain sensitivity plots exist; the noise level used for all later phases is chosen and justified from them.

---

## 10. Phase D: Scale-down study, from large pegs to connector size

**Goal:** answer RQ3. Map how blind insertion degrades as parts shrink, on YAM, with the Phase C noise model on. This bridges the paper's regime (30–90 mm) and USB-C (8 mm), and tells us where the difficulty comes from.

### 10.1 Grid

Stadium pegs, aspect ratio fixed at 2.0. Two axes:

| Axis | Levels |
|---|---|
| Opening long side | 60, 30, 15, 8 mm |
| Clearance per side | 1.0, 0.5, 0.2, 0.1, 0.05 mm |

Two start-offset regimes:
- **Scaled:** start offset proportional to opening size (paper's ~10 mm median at 60 mm → 1.3 mm at 8 mm).
- **Fixed absolute:** median 3 mm for all sizes.

Train one policy per (size, clearance, regime) cell: 4 × 5 × 2 = 40 cells × 3 seeds. If compute is short, prioritize the diagonal (large-loose → small-tight) and the 8 mm row.

### 10.2 Measurements per cell

| Measurement | Statistic |
|---|---|
| In-dist success | mean ± Wilson |
| Median time to seat, lateral travel | median + bootstrap CI |
| F_min (A-S5) and drop-in peak-to-noise ratio (A-S6) | from scripted probes, no policy |
| Contact-decoder accuracy at end of training | balanced accuracy |
| Probe balanced accuracy | as §8.7 |
| Failure-mode breakdown | fraction: never found hole / jammed / tilted-wedged / timeout while searching |

### 10.3 Required visuals

1. **Success heat map:** size (rows) × clearance (columns), one per start regime, with the value and n in each cell.
2. **Search cost vs size:** median lateral travel vs opening size, one line per clearance, log-log.
3. **Signal vs size:** drop-in peak-to-noise ratio vs opening size; overlay policy success on a second axis. This shows whether failures track signal loss.
4. **Failure-mode stacked bars** per cell along the diagonal.

### 10.4 Analysis questions this phase must answer, in writing

1. At what size/clearance does success fall below 90%? Below 50%?
2. Is the drop driven by **search area** (fixed-offset regime much worse than scaled) or by **signal** (both regimes fail together, and the drop-in signal ratio collapses)?
3. Does the latent still encode contact mode at small scale (probe accuracy vs size)?

### Gate D

- [ ] Heat maps and the three plots exist with ≥ 3 seeds for the diagonal and the 8 mm row.
- [ ] `results/D/findings.md` answers the three questions with numbers.

---

## 11. Phase E: USB-C flash drive insertion

**Goal:** answer RQ4. Train a blind policy that plugs a USB-C flash drive into a port in simulation, and characterize where it works.

### 11.1 Assets

- Base: `configs/usbc_flash_drive.yaml`. Receptacle opening 8.34 × 2.56 mm stadium with a 6.5 × 0.7 mm tongue; blind port 6.6 mm deep; plug shell length 8.5 mm; drive body 20 × 6.5 × 30 mm; 8 g.
- Variant pool (K = 32–64), each drawn from:

| Parameter | Range |
|---|---|
| Clearance per side | 0.02–0.10 mm (log-uniform); training curriculum below |
| Port housing footprint | 15–40 mm |
| Port recess (port opening below the housing surface) | 0–1.5 mm |
| Drive body width / thickness / length | 12–25 / 4.5–9 / 20–40 mm |
| Drive mass | 3–15 g |

- **Chamfer:** real ports have lead-ins. Phase E trains **without** chamfers first (the hard case), then adds a 0.2–0.5 mm lead-in as an ablation, built with mjfit's outline offsets.
- **Latch force:** omitted in E1–E4; added in E5 (§11.5).

### 11.2 Task definition

| Item | Spec |
|---|---|
| Port orientation | **E-up:** port facing up (hub style). **E-side:** port facing sideways (laptop style), introduced after E-up passes |
| Start state | Plug touching the housing surface near the port, lateral offset per curriculum, tilt ±5°, yaw ±5° about insertion axis |
| Grasp | Drive body held across its thickness; position along the body ±3 mm, in-hand tilt ±5° (smaller than the paper's peg; justified by the part size) |
| Episode length | 8 s |
| Success | Plug within 0.2 mm of port floor and axis within 2° |
| Contact labels | 64 points on the plug's leading rim: free (> 0.5 mm above housing), contact (≤ 0.5 mm), inside (below the port opening) |
| Reward | +1 first capture (plug tip below the port opening), +1 success; same structure as the paper |
| Speed limit | EE speed cap at 50 mm/s near the port, applied by penalty, because of the Phase 0 impact artifact |

### 11.3 Curriculum

Advance a stage when training success ≥ 90% over 5M steps. Log the step count at every transition.

| Stage | Clearance | Start offset (median) |
|---|---|---|
| 1 | 0.30 mm | 0.5 mm |
| 2 | 0.15 mm | 1.0 mm |
| 3 | 0.08 mm | 1.5 mm |
| 4 | 0.05 mm | 2.0 mm |
| 5 | 0.02–0.10 mm | 3.0 mm |

### 11.4 Experiments

| ID | Experiment | Required statistical result | Required visual |
|---|---|---|---|
| E1 | Blind USB-C, E-up, curriculum, 3 seeds | Final success; steps per curriculum stage | Training curve with stage transitions marked |
| E2 | **Capture radius** | Evaluate E1 at start offsets 0.5, 1, 2, 3, 4, 5 mm (1,024 episodes each) | **Success vs start offset**, with the offset at 50% success (logistic fit + CI). The headline plot of the project. |
| E3 | Clearance sweep at fixed offset | Success at 0.02, 0.05, 0.1, 0.2 mm | Success vs clearance |
| E4 | Ablations (3 seeds each) | No contact supervision; history 10 / 30 / 60 steps; no noise model; with chamfer | Ablation bar chart vs E1, paired on matched starts |
| E5 | Latch / detent force | Add a spring constraint resisting the last 1.5 mm of travel (peak 5–15 N, randomized) | Success with vs without latch; force–displacement curve of the latch model |
| E6 | E-side orientation | Retrain or fine-tune for sideways ports | Success, capture radius vs E-up |
| E7 | Behavior analysis | Probe + latent swap (modes: lifted, searching far/near, tongue-collision, entering, seated, stuck); search path plots | Confusion matrix; top-view search paths (10 episodes); EE z vs time |
| E8 | Failure taxonomy | Classify every failed episode | Stacked bar: never found / plug on tongue / wedged tilted / jammed partial / timeout |

### 11.5 Validity checks specific to small parts

- Re-run A-S5 and A-S6 on the USB-C assembly with the YAM. If the drop-in signal is below noise, record it; it predicts E1 failure.
- Check that no result depends on sub-20 µm geometry: re-evaluate E1 at clearance +20 µm and −10 µm. Success must change smoothly (no cliff).
- Visually inspect 20 successful and 20 failed episodes (via RSS replay) for physics artifacts: plug tunneling through the tongue, sideways "snaps" into the port. Log any artifact with its episode ID.

### Gate E

- [ ] E1 success ≥ 80% at stage 4 (50 µm, 2 mm offset), or a documented analysis of why not, using Phase D results.
- [ ] E2 capture-radius curve with fit.
- [ ] E8 failure taxonomy; §11.5 validity checks pass.

---

## 12. Phase F: Adding wrist RGB and depth

**Goal:** answer RQ5. Measure what vision adds to blind USB-C insertion, and where.

### 12.1 Sensor models

| Sensor | Model in sim |
|---|---|
| Wrist RGB | Mounted on the gripper, looking along the fingers at the part tip. 64 × 64 for training (paper), ablate 128 × 128. Randomize lighting, textures, camera pose ±2 mm / ±2°. |
| Wrist depth | Same mount. Model a real close-range stereo depth camera: minimum range (e.g. ~70 mm for D405-class sensors; **verify the spec of the camera we will actually use**), Gaussian depth noise growing with distance, dropout at edges and on specular surfaces. Pixels closer than the minimum range are invalid (zero). |

The minimum range matters: at the moment of insertion the port may be closer than the depth camera can see. The mount distance must be chosen so the port stays in valid range, and this choice is reported.

### 12.2 Policy variants (identical everything else)

| ID | Inputs |
|---|---|
| F-P | Proprioception only (= E1) |
| F-R | RGB only |
| F-D | Depth only |
| F-PR | Proprioception + RGB |
| F-PD | Proprioception + depth |
| F-PRD | Proprioception + RGB + depth |

Fusion: each modality has its own encoder (TCN for joints, CNN for images, each over the same 30-step history); latents are concatenated before the actor MLP. The contact decoder reads the fused latent.

### 12.3 Training strategy

Rendering at thousands of worlds is expensive. Two routes, chosen by the Phase A throughput benchmark:
1. **Direct RL** with images, if batched rendering ≥ ~20k frames/s.
2. **Teacher–student distillation:** the blind (or privileged) teacher from Phase E generates actions; the vision student is trained by DAgger on its own rollouts. Report which route was used for each variant.

### 12.4 Experiments

| ID | Experiment | Required statistical result | Required visual |
|---|---|---|---|
| F1 | All 6 variants, in distribution | Success, median time to seat, lateral travel; paired comparisons vs F-P | Grouped bars |
| F2 | **Capture radius** per variant | As E2, offsets up to 15 mm | Success vs start offset, one line per variant. Main hypothesis: vision lines extend much further right. |
| F3 | OOD parts | Unseen drive body shapes, port housing shapes, colors/textures, clearance outside training | Fig. 11-style bars per variant |
| F4 | Sensor degradation | Depth noise ×2, depth dropout 20%, lighting shift, camera mount offset ±5 mm | Bars per degradation per variant |
| F5 | What vision is used for | Probe each modality's latent for (a) port position relative to plug, (b) contact mode | Probe accuracy table: modality × target |
| F6 | Hybrid hand-off (optional) | Vision-driven approach from 15 mm, switch to F-P inside 2 mm | Success vs start offset vs F-PRD |

### 12.5 Questions this phase must answer, in writing

1. Does vision extend the capture radius, and by how much (offset at 50% success)?
2. Does adding vision hurt OOD generalization relative to proprioception alone, as in the paper?
3. Is depth worth it over RGB at this scale, given its minimum range?
4. Is the best system fused (F-PRD) or a hand-off (F6)?

### Gate F

- [ ] F1–F5 complete with 3 seeds per variant.
- [ ] `results/F/findings.md` answers the four questions with numbers.

---

## 13. Later: hardware gains and real-robot work

Not part of the current plan, but designed for:

1. **Measure real YAM gains and dynamics:** read the MIT-mode kp/kd actually used on our hardware; identify armature, damping and friction from free-space trajectories (fit sim to real joint traces, as the paper did). Replace §5.2 and re-run the Phase C gain-sensitivity check at the measured point.
2. **Measure backlash and encoder noise** on hardware (dial indicator or calibrated camera vs encoder readings). Set the Phase C noise model from these measurements.
3. **Retrain** the Phase E policy with measured values, and evaluate in sim first.
4. **Real deployment** of the blind USB-C policy on YAM: separate plan.
5. **Bimanual extension:** one arm holds the port, the other the drive.

---

## 14. Repository layout

```
blind-insertion/
  assets/
    robots/yam_i2rt/            # copied MJCF + meshes, license kept
    robots/panda/               # Phase B stand-in (or flexiv)
    assemblies/                 # mjfit YAML configs; generated MJCF cached by hash
  envs/
    peg_insert.py               # mjlab task: Phase B/C/D
    usbc_insert.py              # Phase E/F
    obs.py  contact_labels.py  rewards.py  randomization.py  success.py
  policies/
    tcn_actor_critic.py         # TCN encoder + MLP + contact decoder (rsl_rl compatible)
    cnn_encoders.py             # RGB / depth encoders (Phase F)
  baselines/spiral.py
  train.py  eval.py             # eval writes Parquet rows per episode
  analysis/
    stats.py                    # Wilson, bootstrap, Newcombe, Wilcoxon helpers
    probes.py  latent_swap.py
    figures/                    # one script per figure
  checks/                       # Phase 0 and Phase A signal checks (A-S1..A-S10)
  viz/rss_replay.py             # RoboSimStudio replay of logged rollouts (visual only)
  configs/                      # one YAML per run family
  results/<phase>/              # frozen CSV/Parquet + PNG/PDF per figure
  RESEARCH_PLAN.md
```

---

## 15. Risk register

| Risk | Signal that it is happening | Mitigation |
|---|---|---|
| Per-world geometry too slow | A-S9 throughput too low | Option O1 with small K; fewer variants are fine for USB-C |
| Reproduction fails | Gate B fails | Deviation report; match paper config axis by axis |
| No contact signal at USB-C scale | A-S5/A-S6 on USB-C below noise | Report as a finding (Phase D explains it); add joint torque (motor current) to the observation, still proprioceptive |
| Sim artifacts at small clearance | §11.5 checks fail; sideways snaps in replays | Speed cap; smaller timestep (0.5 ms) for the USB-C phase; report the resolution floor |
| Batched rendering unavailable/slow | Phase A benchmark | Teacher–student distillation (§12.3) |
| Assumed gains far from real | Gain sensitivity (§9.4) steep | Prioritize §13 step 1 early |
| Policy exploits impact | Fast approaches in rollouts; success drops with speed cap | Keep speed penalty; report |

---

## 16. Master checklist

- [x] **Phase 0:** mjfit verified in MuJoCo, Newton and RoboSimStudio; USB-C flash-drive config
- [ ] **Phase A:** env components A-1…A-12; geometry mechanism benchmark; signal checks A-S1…A-S10
- [ ] **Phase B:** B-R1…B-R7 reproduced or deviations documented
- [ ] **Phase C:** YAM port; noise/backlash/latency robustness; gain sensitivity
- [ ] **Phase D:** size × clearance heat maps; signal vs size; written findings
- [ ] **Phase E:** USB-C E1…E8; capture radius; validity checks
- [ ] **Phase F:** six sensing variants; capture radius per variant; written findings
- [ ] **Later:** measured gains; real deployment plan
