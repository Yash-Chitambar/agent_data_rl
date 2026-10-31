# Findings, deviations and ideas (lab notebook)

Running log for `docs/archive/RESEARCH_PLAN_single_arm.md`. Dated entries; newest first within each section.
Figures and data referenced here live in `results/<phase>/`.

## 2026-10-25 (Pacific): physical Task H reconstruction

On `codex/physical-yam-scene`, integrated `real-yam-setup` e7b843e. Both arms now
use upstream I2RT flexible_4310 meshes, measured left-base transform, bilateral
wrist calibration and ZED policy crop. Object placements estimated from calibrated
ZED rays; original measured reader and plug unchanged. See
`docs/real2sim/PHYSICAL_YAM_SCENE.md` for source revision, preprocessing and limitations.
Camera replay transforms match saved positions to <5 um numerically. Calibrated
render/step and manifest checks pass. Existing expert diagnostic: **0/4 held or
completed handoffs**, finite states, final phases 4/5/4/4 under measured friction
ranges. Do not carry over v2 success claims or launch bulk demo generation without
contact/expert validation. FlexPoint compliance and payload are still unmeasured.
No hardware motion, training or deployment performed in this update.

## Status summary

| Phase | Status |
|---|---|
| 0 | Passed (per plan). The Phase 0 scripts/configs (`blind_usbc/`, `configs/usbc_flash_drive.yaml`) are **not in this repo**; recreated as needed. |
| A | Env, policy, PPO, eval harness built. Signal checks: see table below. |
| B | Blind s0, s1, s3 (300M), visual s0-s2 (50M, stop rule), no-contact s0, s2 trained and evaluated. Blind s2 and no-contact s1 are in the local queue. Results match the paper (table below). |
| C | 16 seed-0 YAM runs on the lab-table mount (mason) trained and evaluated: 16 x 13 train/test matrix (`results/C/C-matrix.*`). **Gate C closed (2026-10-21 10:15 UTC): D/E use per-world randomized noise/quant/backlash/latency** (`randall` 96-99% on every such test, corrected backlash model 2026-10-22). Corrected backlash costs <= 4 points for every policy (base 99.9% at 5'); latency and gains remain the conditions that need training. All 17 conditions redone on the fixed model (2026-10-22 19:30 UTC). |
| H | Task H (SD reader into the mini PC): scene, spec-accurate USB-C, scripted expert (512/512 demos), LeRobot dataset, fine-tuned insertion 93% (1 mm) / 90% (3 mm); expert + insert 97.7%, ACT 200k + insert + release 85.5% inserted / 84.0% seated after release (2026-10-24 20:40 entry). v2 scene + gain/friction DR: insertion v2 97.3% / 95.1% / 89.5% at 1.3 / 3.9 / 6 mm (2026-10-25 06:30); ACT v2 + insertion v2 + release 90.6% inserted / 87.5% seated with settle handoff (2026-10-25 19:00). |
| D | Tier 1 (16 runs, 150M cap, randomized noise model) on mason (`runs/queue_server/stateD`). o60_c1 and o30_c0.5 (scaled) reach 99% training success; o15_c0.2 from scratch never succeeded, so cells tighter than 1 mm now use a clearance curriculum (`--curriculum scale`). Grid: `scripts/phase_d_report.py`. |
| E | USB-C with sphere pads: full curriculum, 91.5% at 20-100 um (athena5, old mount). E4-E6 variants built; E1 zero-shot 77.9% on the table mount. Mason: curr, chamfer, latch, side training from scratch on the table mount (no noise model, ~7.5 h each); noise (randomized), nocontact, h10, h60 queued after D tier 1. |
| G | Gear meshing (paper's second task) built: assets, `GearEnv`, `--task gear` in train/eval, checks in `results/G/`. 15-min smoke: shaft capture 2% -> 50% in 12M steps, success not yet rising. Full run `G-gear-proprio-base-s0` + eval queued after `C-peg-proprio-base-s0`. |
| Report | `report/` (built by `scripts/build_report.py`): HTML results page in ASD-STE100 style. |
| Real rig | duchamp3 ready (2026-10-26), both arms: CAN (udev autostart), safety layer, rig console `rigctl.py` (stop, arms on/off, cameras), live browser camera view. Wrist cameras, joint offsets, base poses and the ZED calibrated and cross-checked at new board positions (wrists 1-3 mm, ZED ~2 px held out). Latency ~8 ms; joint 2-3 friction 1-1.5 N m, elbows stick-slip 3-5 N m static (accumulated-target actions added). Training configs `task_h_reader_real.json` / `_acc.json` (not trained yet). Open: fixture pose, payload, camera exposure, ZED in sim; two-arm deploy loop when needed. Branch `real-yam-setup` (PR #1). |

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

## Next steps (as of 2026-10-21 11:00 UTC)

- D tier 1 (randomized noise model) and E batch 1 on mason; then the E ablations and D tier 2.
- Gains are not in `randall`: gain x4 still gives 52%. Measure real gains or add gain
  randomization (0.5-2x) to the noise model if the real step tests disagree with kp 80/30.
- Measure the real YAM's command-to-encoder latency (in 20 ms control steps) and backlash: these
  two set the D/E noise model more than q noise does. The rig is ready for it (2026-10-25 entry).
- Real-rig gaps for sim realism (2026-10-25 entry, "Open"): gains (real wrist kp 40/10/10 vs sim
  30/30/30), joint offsets in the deployment loop, real fixture pose, finger-pad geometry, wrist
  payload, camera exposure and frame latency. Plan and tests: `docs/real2sim/REAL_RIG_GAPS.md`.
- Blind s2 and no-contact s1 (local queue) for the 3-seed B tables.
- Phase F after E, with Yash's OK (~70 GPU-h).

## Discoveries

### 2026-11-01 (mason, `policy/approach-hold`) — Round 6: why the arm-robust ZED nets miss on real frames

Real test: the 123 ZED frames of `bimanual_stationary_001` (both arms folded, all joints ~0 by the encoders)
against the manual lid landmark (port (0.2142, 0.3308) m, about +/-5 mm). One factor at a time from the v1
recipe (each net n = 16,384 renders, 25 epochs; `scripts/perception/zed_ablation.sh`, results
`results/usbc/policy/approach/ablation/`):

| Training change from v1 | Real offset (mm) | Sim rest / parked p50 (mm) |
|---|---|---|
| none (v1 recipe, today's moderate visual DR) | 0.6 | 0.9 / 7.5 |
| mild visual DR (what v1 used) | 3.0 | 1.0 / 7.5 |
| Part 1 layout (reader in the front strip) | 2.4 | 0.9 / 7.3 |
| reader varied (strip / elsewhere / out of view) | 1.9 | 0.9 / 7.7 |
| **arm poses and gripper openings varied** | **13.9** | 1.6 / 2.1 |
| arms + reader varied | 8.2 | 1.7 / 2.1 |
| arms masked from the joint angles (sim segmentation; real mask from the encoders) | 10.9 (bases masked), 8.9 (bases kept) | -- |
| arms varied + masked | 8.1 | -- |

- **Cause: the arm-pose variation itself.** Visual DR level, PC box, reader placement and layout move the real
  estimate by <= 3 mm (inside the landmark's uncertainty); every net trained with varied (or masked) arms
  lands 7-15 mm off, always toward +y. Masking the arm pixels (the real mask from the encoders aligns well
  with the real arms) does not help, so it is not the arm pixels' look alone.
- **Contradiction that points at sim-vs-real arm context:** in sim with both arms folded (the real frames'
  pose) the v1-type nets are 7.5-8 mm off (p50, n = 512) while the arm-varied nets are 2.1-2.7 mm; on the real
  folded frames it is the reverse (v1 0.4 mm, arm-varied 8-14 mm). The real folded scene does not look to the
  nets like the sim folded scene; with one real PC placement and a +/-5 mm landmark we cannot tell more.
- A tight-crop refinement stage (64 px around the coarse estimate) did not train (sim error unchanged); not
  pursued further.
- **Decision unchanged:** v1 + presence/OOD + arm-pose gate, ZED frame at the rest pose with the gripper
  open (round 5). The arm-robust nets stay sim-only.
- **Needed to go further:** real ZED frames with known PC poses under several arm poses (e.g. the PC at 3-4
  marked spots, arms folded / parked / pressing, encoders logged). Without them no arm-robust net can be
  validated on real data.

### 2026-11-01 (mason, `policy/approach-hold`) — Round 5: chain-surfaced issues (elliptic cones, ZED arm poses, hold contact, plug bias)

All with elliptic cones + impratio 30 (the chain and Part 4 solver) unless noted.

- **Hold in the chain context** (`scripts/task/diag_chain_hold.py`: reset -> Part 1 expert pick -> guarded 25 N press,
  right arm holding the reader at the lift pose; n = 1024 = 2 x 512, seeds 7100-7101):
  2 cm press: arm-arm 0/1024, PC still 1024/1024 at 3 N and 1024/1024 at 5 N, left joint 2 <= 8.9 N m.
  3.5 cm press: still 1024/1024 at 3 and 5 N, but **arm-arm 4/1024** (chain: 5/1024): the left forearm (link_5 /
  link_3 capsules) hits the right arm's link_4 and the two fingertip hulls touch, with the right arm at the
  post-pick lift pose. A higher stand-off (+5 cm) made it worse (8/1024, PC still 77 %). Elliptic cones hold
  the PC better than pyramidal (2 cm, 3 N: 100 % vs 99.3 %). **Use the 2 cm press** (= Part 4's training
  hold): it passes 3 and 5 N with no contact, and 10 N (1024/1024). 15 N at 2 cm: 3 N 100 %, 5 N 100 %, 10 N
  99.9 %, sink 0.56 mm, left joint 2 <= 5.4 N m -- a lower-torque option if 25 N heats the motor.
- **PC sink:** with elliptic cones the 25 N press pushes the PC 0.77 mm (p50; p99 0.86) into its feet (3.5 cm:
  0.89 mm). It is soft-contact compression of the 5 mm box feet, settles during the engage (before the
  approach), and is almost all vertical (engage translation p99 0.89 mm vs sink p99 0.86 mm; one world in
  1024 moved 6.1 mm): no effect on the PC-still result; the approach servo measures the live port.
- **ZED vs arm pose:** v1 (trained at rest, right fingers open) by condition (n = 512 each): rest 0.79 / 2.63 mm,
  right fingers at 24 mm 2.95 / 5.71, left parked 7.10 / 20.1, left pressing 8.58 / 21.9. Retrained with varied arm
  poses and openings (`render_data.vary_arms`: right rest/perturbed, fingers closed..open; left rest / parked /
  pressing; reader in the strip / elsewhere / out of view; PC box widened to +/-3 cm / +/-10 deg): v4 two-net
  ensemble rest 0.95 / 4.27, parked 1.70 / 5.47, pressing 1.69 / 5.17 mm (accepted, n = 671-703) -- robust in sim
  but **real frames regress**: 8.7 / 14.1 mm from the manual landmark (v1: 0.41 mm), and the two nets disagree
  there, so the ensemble gate rejects 122/123 real stationary frames (it knows it cannot handle them).
  A heteroscedastic (predicted-std) variant was no better (sim p50 1.45 mm, real 15.8 mm).
- **Gate that rejects what it cannot handle:** the arm configuration is known from the encoders, so the
  gate now checks it (`ZedEstimator(frames, arms=(q_left, q_right, right_finger))`, envelope in the gate
  checkpoint). v1 + presence/OOD + arm-pose gate (`runs/perception/zed_v1_gate_arm.pt`, +/-0.05 rad of the rest
  pose, right fingers >= 40 mm): rest accepted 100 % (0.79 / 2.63 mm), 24 mm / parked / pressing rejected 100 %,
  arms beyond training (x2.5) 97.7 % rejected. Recommendation: take the ZED frame at the rest pose with the right
  gripper fully open (as the chain does) with this gate; the v4 ensemble is the fallback when the arms cannot be
  at rest, not yet for real frames.
- **Approach under elliptic cones** (2 cm hold, chain layout): v10w + v10ws lateral p50 0.39 / p99 1.67 mm,
  plug low by 0.20 mm on average (chain: 0.36). One DAgger round on elliptic in-loop states (both holds) ->
  v12w + v12ws: p50 0.30 / p99 1.41 mm, mean (x, y) = (+0.06, +0.10) mm. **Bias removed** by a servo target
  offset (-0.06, -0.10) mm calibrated on seeds 1-2, tested on seeds 3-4: **lateral p50 0.27 / p99 1.24 /
  p99.5 1.63 mm, tilt p99 0.69 deg, roll p99 0.50 deg, mean (x, y) = (-0.03, -0.04) mm, 96.8 % inside 1 mm / 2 deg
  (n = 1024)**. Deliverable updated (`preinsert_error.json`). 3.5 cm hold: p99 1.49 mm.

### 2026-10-31 (mason, `policy/chain`) — End to end: pick → hold → approach → insertion in one simulation, 99.80 % (n = 1024)

`scripts/eval/eval_chain.py` runs all four parts in one continuous MuJoCo Warp simulation per world:
no teleports between stages. Each world draws one layout (PC box ±1 cm / ±1.5 cm / ±5°, reader on the
Part 1 strip) and full DR at reset (arm dynamics, latency 0-1 step, encoder offsets ±1.5° kept through
all stages, PC feet μ U(0.8, 1), reader-table μ U(0.4, 1), plug-port μ U(0.1, 0.2), port solref 3-4 ms,
visual DR moderate defaults, RGB only). Stages: ZED CNN + gate (frame at the rest pose) → Part 1
scripted expert pick → Part 3 guarded 25 N press (planned from the ZED estimate) → new Cartesian
transport to the look pose → Part 2 wrist servo (wrist_v10w + wrist_v10ws below 7 mm, 6 mm stop) →
Part 4 blind PPO from the live state (InsertionTask success test: 0.2 s seated, 25 µm aperture tolerance).

**Result (held-out seeds 7000-7001, 2 × 512 worlds):**

| Insertion policy | Hold press point | n | End to end | Wilson 95 % | Failures |
|---|---|---|---|---|---|
| **v6 s2 (Part 4 best)** | **2 cm in front of the case centre (Part 4 training value)** | 1024 | **99.80 %** | [99.29, 99.95] | 2 stuck at the mouth |
| v6 s2 | 3.5 cm (Part 3 round 4) | 1024 | 96.00 % | [94.61, 97.04] | 41 stuck at the mouth |
| v10dr (hold DR 1.5-4 cm) | 2 cm | 1024 | 99.22 % | [98.47, 99.60] | 7 mouth, 1 jam; 0 arm-arm |
| v10dr | 3.5 cm | 1024 | 99.22 % | [98.47, 99.60] | 8 mouth; 5 arm-arm (hold engage) |
| v5 s0 | 2 cm | 1024 | 98.54 % | [97.60, 99.11] | 14 mouth, 1 slip |
| v5 s0 | 3.5 cm | 1024 | 97.66 % | [96.54, 98.42] | 23 mouth, 1 slip |

v10dr (policy/insertion 7e9ef6e, trained with the press point randomized 1.5-4 cm) removes the hold-position
sensitivity (99.22 % at both points) but is 0.6 points below v6 at 2 cm; its own eval on chain starts gave 100 % /
99.95 %. Its 16 chain failures again start low (pre-insert y -0.2 to -4 mm, warm-up lateral 0.3-4.6 mm) and stall
at the mouth: the live hand-over (warm-up drift, true in-hand state) is harder than sampled chain starts.
Hold yaw for all runs: InsertionTask's rule (best yaw at the nominal PC pose +/- 15 deg, not LeftHold's -90..90).
Every stage before the insertion passed 1024/1024 in all six runs (ZED gate, pick, hold latch,
transport, approach). Gate (≥ 98 %, n ≥ 500) passes with v6 + the 2 cm hold.

**Mismatches (most valuable):**
- **The blind insertion policy reads the left arm.** Its observation has all 12 encoder angles. With
  LeftHold's default tool-yaw search (-90..90°) the left wrist (joint 6) sat 0.29 rad from the training
  value and the policy seated 5/128, against 128/128 from teleported InsertionTask starts in the same
  runtime. Restricting the yaw to the training set (nominal best ± 15°) fixed it. Moving the press point
  from 2 to 3.5 cm (Part 3's round-4 hold) costs 3.8 points with v6 (99.80 → 96.00 %) and 0.9 with v5.
  The failures start well inside the capture region (pre-insert lateral 0.2-1 mm) and stall at the
  port mouth. Part 3 and Part 4 each pass alone; together the hold must be the one Part 4 trained with,
  or Part 4 must train with the 3.5 cm hold (or mask the left arm from the actor).
- **ZED estimate depends on the right gripper opening.** The pick task sets the approach opening (24 mm)
  in the reset pose. With it in view the ZED CNN is biased: xy p50 3.0 vs 0.8 mm (n = 256). The chain
  takes the ZED frame with the gripper fully open (the training condition), then closes it. With the left
  arm parked (Part 1's start) the estimate is xy p50 8.0 / p99 20.6 mm and **the presence/OOD gate still
  accepts 100 %** of those frames: the gate does not catch this arm-pose shift. Frame at the rest pose:
  p50 0.77 / p99 2.79 mm, yaw p99 1.49° (Part 2: 0.79 / 3.3 mm).
- **No transport existed.** Part 2 teleports to the look pose. A joint-space move from the lift pose to
  the look pose swung the plug 3 cm into the table (in-hand rotation p50 16°, n = 16). A Cartesian path
  (turn the reader at lift height → via pose 40 mm further out and 30 mm up → look pose; IK each step)
  keeps the reader: in-hand change p99 0.35 mm / 0.12°, no hand-PC contact (n = 1024). Arrival at the
  look pose: lateral p50 7.6 / p99 18 mm (Part 2's teleported look pose: 7.5-7.8 / 19.6-20.4 mm).
- **Pre-insert error matches Part 2.** Chain: lateral p50 0.46 / p99 1.48 mm, tilt p99 0.86°, roll p99
  0.72°, gap 5.76 mm p50 (Part 2 deliverable: p99 1.64 mm / 0.87°). The plug sits 0.36 mm low on average
  (y mean -0.36, std 0.45 mm). In the 0.2 s insertion warm-up (feed-forward off, zero action) the error
  grows by 0.3 mm (p50 0.46 → 0.76, p99 1.48 → 1.91 mm): the policy starts from that, not from Part 2's
  numbers. Both v6 failures started low (y -0.9 and -0.3 mm).
- **Solver cones.** Part 4 needs elliptic cones (pyramidal goes non-finite under insertion); Parts 1-3
  were validated with pyramidal. The chain runs elliptic throughout. The pick still passes 1024/1024, but
  the in-hand spread about the mean is wider (position p99 0.91 vs 0.77 mm, rotation p99 0.54 vs 0.24°;
  rotation about the tool axis p99 0.55 vs 0.01°) and the PC sinks 0.77 mm (p50) into its feet when the
  press engages (Part 2 reports 0.01 mm PC motion with pyramidal).
- **Hold at 3.5 cm:** 5/1024 worlds had arm-arm contact during the engage (0 at 2 cm); PC moved
  ≤ 1.1 mm (p99), one world 6.7 mm.
- Insertion time to seat p50 1.0 / p99 2.5 s; PC motion during insertion p99 1.4 mm.

**Shared-file change:** `BimanualTask(cfg, runtime=None)` and `PickTask(cfg, runtime=None)` can share an
existing runtime (default unchanged). Weights outside git: `runs/perception/{zed,zed_presence,wrist_v10w,
wrist_v10ws}.pt`, `runs/chain/P4-insert-best_v6s2.pt` (= policy-insertion `runs/P4-insert-best/best.pt`;
`P4_INSERT_BEST` overrides). Results `results/usbc/policy/chain/` (`chain_p4v6_hold020.*` = headline);
demo videos `results/usbc/policy/chain/demos/`. Smoke test `scripts/checks/check_chain.py`.
Narrowed for success: the hold press point (2 cm, not 3.5 cm), the left tool yaw (training set), the
ZED frame pose (rest pose, gripper open). Layout and DR ranges are the parts' own.

### 2026-11-01 (mason, `policy/approach-hold`) — Round 4: hold 3 N gate met; ZED presence gate; approach DAgger/specialist

- **Hold:** pressing 3.5 cm in front of the case centre (instead of 2 cm) passes 3 N: 5113/5120 PC still (99.86 %,
  five runs of n = 1024, two starting from the folded/parked left pose), 5 N 99.5-99.8 %, 10 N 97.0 %, 25 N, left
  joint 2 <= 8.8 N m. 4.5 cm in front: 98.2 %; longer settle, earlier latch, centred press: no gain.
- **ZED gate** (`perception/zed_estimator.py`): presence CNN + Mahalanobis OOD on the pose CNN's features
  (1.25 x sim p99.9). Sim: accepts 100 % with a PC, 0 % without (n = 1024 each). Real first frames of the 24
  curated sessions: rejects 7/7 without a PC, accepts 14/17 with one (rejected: tape reel, hand, sticky notes on
  the PC). The 1.25 margin is set from those real distances. Reader in the new strip: no change (xy p99 2.9 mm).
- **Approach** (lateral p50 / p99 mm, n = 1024, press 2 cm): v9 DAgger 0.37 / 1.84; wider CNN v9w 0.37 / 1.74;
  v9 + stop-gap specialist v9s 0.34 / 1.68 (93.8 % inside 1 mm / 2 deg); 12-iteration dwell 0.43 / 1.81.
  Error grows with the DR wrist-camera pose error: with that error inside the real calibration (<= 5.1 mm,
  <= 1.2 deg; n = 331) v9 + v9s gives p99 1.02 mm, tilt p99 0.72 deg. The DR samples up to 2x real.
- The 3.5 cm press puts the left fingers next to the port in the wrist image: the round-3/4 nets fail there
  (v9w p99 11 mm). DAgger at the new hold: v10w 0.42 / 2.04 mm; **v10w + stop-gap specialist v10ws (current deliverable):
  lateral p50 0.32 / p99 1.64 / p99.5 1.88 mm (x p99 0.93, y p99 1.49), tilt p99 0.87 deg, roll p99 0.57 deg,
  gap error p99 0.83 mm, 95.3 % inside 1 mm / 2 deg (n = 1024, all finite, ZED gate accepted all)**; with the wrist
  camera error inside the real calibration (n = 331): lateral p99 1.13 mm, tilt p99 0.70 deg. One more round
  (v11w + v10ws): p99 1.62 mm, 94.3 %: plateau.
### 2026-10-31 (mason) — Part 4 insertion: episode, grasp, contact DR, solver; training status

**Result (blind PPO v5 seed 0, `runs/P4-insert-proprio-v5-s0`, ~19 M steps):** held-out seed 1000,
Part 2 round-3 pre-insert errors (`preinsert_error_samples.npz`, n = 1024 episodes) × scale, Part 1
grasp (in-hand spread ×1.25), Part 3 guarded 25 N hold, full DR, 10 s episodes, seated = 0.2 s
geometric hold with a 25 µm aperture tolerance; n = 2048 per row:

| Scale | Lateral p99 | Seated | Wilson 95 % | Failures | Part 4 gate |
|---|---|---|---|---|---|
| 0.5 | ~1.0 mm | 99.90 % | [99.64, 99.97] | 2 slip | pass |
| **1.0 (Part 2 as measured)** | 1.98 mm | **99.71 %** | [99.36, 99.87] | 3 slip, 3 stuck at mouth | **pass** |
| 1.5 (requested margin) | ~3.1 mm | 99.17 % | [98.67, 99.48] | 2 slip, 15 stuck at mouth | miss (lower bound 98.7 %) |

**PC fixed** (mass ×30, `mj_setConst`; PC motion p99 0.11 mm) at 1.5: 99.22 % [98.73, 99.52] (n = 2048,
15 stuck at mouth, 1 slip), the same as with the 25 N hold: the hold is not the limit, the lateral
capture region is. (×1000 mass without `mj_setConst` went non-finite in every world.)
Other seeds at 1.5: s1 98.68 % [98.09, 99.09], s2 99.07 % [98.56, 99.41].
**Round 2 result (2026-11-01): gate passed at scale 1.5.** Blind PPO v6 seed 2 (continued from v5 seed 2
with curriculum 1.5 → 2.0 and 30 % of starts drawn at 1.5-3.5 mm lateral, `hard_frac`; 10 M steps, ~1 h on
one V100), n = 2048 per row:

| Scale | Eval seed | Seated | Wilson 95 % | Failures |
|---|---|---|---|---|
| 0.5 | 1000 | 99.95 % | [99.72, 99.99] | 1 slip |
| 1.0 | 1000 | 99.95 % | [99.72, 99.99] | 1 stuck at mouth |
| **1.5** | 1000 | **99.76 %** | [99.43, 99.90] | 4 stuck at mouth, 1 slip |
| **1.5** | 2000 (independent) | **99.85 %** | [99.57, 99.95] | 2 stuck at mouth, 1 slip |

At 1.5 (seed 1000) by start lateral error: ≤ 1.5 mm 100 % (1855), 1.5-2 mm 99.0 % (96), 2-3 mm 95.8 % (71),
> 3 mm 100 % (24). Seated plugs 5.63-5.80 mm deep, tilt ≤ 1.92°, aperture violation median 15.7 µm.
Median time to seat 0.88 s; PC motion p99 1.3 mm. v6 seed 0 at 10 M: 99.27 % [98.80, 99.56] (misses).
Exported as `runs/P4-insert-best/best.pt`.

**Left-arm dependence (chain feedback, 2026-11-01).** v6s2 reads the left arm (12 encoders etc.): in this
env, scale 1.5, n = 2048: hold at 2 cm 99.76 %, at 3.5 cm **82.76 %** [81.07, 84.34] (349 stuck at the mouth).
A 0.26 rad change of left joint 6 alone (tool yaw choice) dropped it to 0.2 % (n = 1024). Diagnosis:
- Press point drawn in 1.5-4 cm (v6s2, no retraining): 93.5 %, no slips, so the hold is physically fine and
  the loss is observation dependence.
- Masking the left arm from the actor (all left features, or only joint angles + previous actions) and
  fine-tuning makes grasp slips appear and grow (to 30 % in training): the left arm is the actor's insertion
  **force sensor** (the plug pushes the PC, the PC pushes the left hand, its encoders move). Without it the
  policy pushes harder and the reader slips.
- Fix used: keep the left observations and randomize the hold per start (press point 1.5-4 cm in front of the
  case centre; optional tool yaw ±15°) and train on Part 2 round-3 + the chain's 1024 actual starts
  (`results/usbc/policy/insertion/aux/starts_round3_plus_chain.npz`; chain starts sit 0.36 mm low).

- **Result (v10dr, exported as best.pt):** fine-tuned from v6s2 with the press-point DR (no yaw DR), 5.2 M
  steps. n = 2048 each: chain starts, hold 2 cm 100 % [99.81, 100], hold 3.5 cm 99.95 % [99.72, 99.99];
  round-3 × 1.5, hold 2 cm 99.61 % [99.23, 99.80], hold 3.5 cm 99.61 % [99.23, 99.80]. Gate met at both holds.
- **Tool yaw is still a sensitivity:** v10dr with random hold yaw ±15° (best-margin yaw of the shifted set):
  47.85 % (n = 1024). Yaw changes the whole left-arm IK, not only joint 6; at ±30° there is no left-right or
  left-reader contact (n = 512) but 10.9 % slips: behavioural (misread force signal), not physical. Training
  with yaw DR (v11: left joint 6 masked; v12: unmasked) is in progress; the hold yaw must match between the
  chain and this env until then (this env: best of nominal ± 15° by joint margin).

- **v11 (exported as best.pt):** left joint 6 masked from the actor (keeps the force signal in joints 2-5),
  hold DR press point 1.5-4 cm + tool yaw ±30°, from v10dr, 10.7 M steps. Round-3 × 1.5, n = 2048 each: hold 2 cm
  99.66 % [99.30, 99.83]; hold 3.5 cm 99.46 % [99.04, 99.70]; random yaw ±15° 99.76 % [99.43, 99.90]; ±30° 99.90 %
  [99.64, 99.97] (v10dr at ±15°: 47.85 %). Yaw DR without the joint-6 mask collapsed (28.8 % in training).
  The loader applies the mask itself.

**Round 3 (2026-11-01): the wrist camera trained directly with PPO.** VisualAC (64 px right-wrist RGB, moderate
visual DR, zero-init residual on v6s2), 4.1 M camera steps, same curriculum/hard starts as the blind control
(v6s2 continued to 13 M steps). n = 2048 each: scale 1.5 wrist 99.56 % [99.17, 99.77] vs blind 99.76 %
[99.43, 99.90]; scale 2.0 wrist 99.56 % [99.17, 99.77] vs blind 99.32 % [98.86, 99.59]. No significant
difference at either scale (intervals overlap); the camera does not help at this error level in sim.

**Round 3 (2026-11-01): robustness of the exported policy (v6s2, no retraining).** One change at a time
against the baseline (scale 1.5, seed 1000: 99.76 %), n = 2048 each:

| Change | Seated | Wilson 95 % | Failures |
|---|---|---|---|
| ×2.0 start errors | 99.51 % | [99.10, 99.73] | 9 mouth, 1 slip |
| Encoder offset ±2° (was ±1.5°) | 99.85 % | [99.57, 99.95] | 3 mouth |
| PC box ±3 cm / ±10° (was ±1/1.5 cm, ±5°) | 99.61 % | [99.23, 99.80] | 7 mouth, 1 slip |
| Latency 0-2 steps (was 0-1; measured ~0.4) | 94.24 % | [93.14, 95.17] | 70 mouth, 47 slip |
| Plug-port μ U(0.10, 0.30) (was 0.10-0.20) | 94.43 % | [93.35, 95.35] | 114 mouth |
| Wide contact DR (port μ 0.15-0.6, 3-8 ms; pad μ 0.8-2.0, 2-6 ms; squeeze ×0.6-1.4) | 69.53 % | [67.50, 71.49] | 421 slip, 202 mouth |

Margins: start error, encoder offsets and PC placement have margin; latency beyond 1 step, plug-port
friction above 0.2 and pad friction/squeeze below the Part 1 values do not. The contact values are the
first thing to measure on the rig (or to train over once measured ranges exist).

**Round 3 (2026-11-01): why the DAgger students looked good in training.** The distiller updates the
student every control step on a replay buffer that holds the current episode's teacher labels, so the
privileged teacher steers the student mid-episode through the gradient steps. Same student (wrist,
1.02 M steps), same env, β = 0, 256 worlds: with updates (lr 3e-4) 100 % / 90.5 % seated (n = 320, 359);
frozen (lr 0) 3.1 % / 0.0 % (n = 258, 243). The training logs measured online teacher leakage, not the
student; the held-out evals (≤ 7 %) were right. Not an env mismatch (memorization was also ruled out:
the wrist student scores 0.6 % on its own training start pool vs 0.4 % on seed 1000). Frozen-policy
evals, as the chain uses, are unaffected.

**Round 2 checks (2026-11-01):**
- *Seated means fully inserted:* seed 0 at 1.5, re-run (99.07 %, n = 2048): the 2029 seated plugs are
  5.63-5.80 mm deep (median 5.72; seat 5.8 mm at the neck stop), tip centre within 0.08 mm of the
  axis, tilt median 1.03°, max 1.74°; aperture violation median 13.4 µm, p99 23.6 µm (cap 25). No
  seated plug is near the rim; failures end at +0.3 mm (outside the panel) with mm-scale violations.
- *Encoder offsets are not the cause:* the max offset is ≥ 1.13° in 90 % of all episodes; success by
  bin 99.4 / 98.8 / 99.3 / 99.1 % (flat), latency 0 vs 1: 99.1 vs 99.0 %. The driver is the start
  lateral error.
- *Seed ensemble* (mean action of s0-s2): 98.78 % [98.20, 99.17], worse than seed 0 alone.
- *Export:* `runs/P4-insert-best/best.pt` (+ README.json) and `eval_insertion.load_insertion_policy()`
  (`InsertionPolicy`, env var `P4_INSERT_BEST` from other worktrees). At 1.5 (seed 0), by start
lateral error: ≤ 0.5 mm 100 % (888), 0.5-1 mm 99.86 % (692), 1-1.5 mm 99.27 % (275), 1.5-2 mm 94.8 % (96),
2-3 mm 93.0 % (71), 3-4.5 mm 87.5 % (24); every failure had an encoder-offset error ≥ 1.09°. Median
time to seat 0.6 / 0.84 / 0.9 s; PC motion p99 ≤ 1.3 mm. Strict (0 µm) aperture test: ≤ 0.25 % at
every level (soft-contact penetration; see below). Results: `results/usbc/policy/insertion/eval_blind_v5s0_*`.

Branch `policy/insertion`. Code: `envs/insertion_task.py`, `train_bimanual.py`, `eval_insertion.py`,
`distill_insertion.py`; checks `scripts/checks/check_insertion_task.py`, `check_insertion_clearance.py`.
Results: `results/usbc/policy/insertion/`.

- **Episode:** reader in the right hand at a pre-insert pose 2-8 mm in front of port 2 (GPU IK start
  pool), PC pose nominal ±3 cm / ±10°, left arm pressing on the PC top (scripted), full physics DR
  (arm friction/gravity, latency 0-1 step, encoder offset ±1.5°, PC feet μ, new contact DR).
  Start errors (scale 1): plug lateral ±2 mm per axis, ±2° per axis, in-hand ±3 mm (plug axis, grasp
  height) and ±8° about the closing axis (±1.5° on the other two; the coupled fingers centre the reader).
- **Grasp (confirmed 60° grasp):** TCP in the reader frame (0, +1.75, 48.9) mm, solved so the lowest
  finger point is 1 mm below the reader bottom; pad sphere rows cover 6.1 mm of the 8.8 mm side face.
  At port 2, seated, randomized in-hand pose (n=64 PC poses): fingers ≥ 35 mm from the PC case and
  ≥ 13.7 mm above the table, gripper housing ≥ 27 mm above the table. **No tilt change is needed near
  the port** (30-60° all clear).
- **Bug fixed:** in-hand rotation noise rotated the reader about its plug tip (57 mm from the pads): an
  8° pitch moved the body 8 mm out of the fingers and the reader fell. Now about the pad centroid.
- **Grasp creep is a solver artifact:** pyramidal cones (scene default; `impratio` acts on elliptic
  cones only) let the 17 g reader creep in the pads: 0.5 mm median / 7 mm max in 4 s, 5-9 % slip
  terminations in a 2 s zero-action hold (n=512), worst at low pad μ. **Elliptic cones** (`cone=1`, new
  `BimanualCfg.opt_overrides`): 0.09 mm median / 0.4 mm max in 4 s, no slips for pad μ 0.8-2.0.
  Needs `njmax` 2000 (1400 overflowed once under random actions).
- **Contact DR** (`BimanualCfg.contact_dr`, batched `geom_friction`/`geom_solref`/`actuator_gainprm`):
  plug-port μ U(0.15, 0.6) (port geoms have contact priority), port solref 3-8 ms, pad μ U(0.8, 2.0)
  (reader geoms lowered to 0.5 so the pad sets the pair), pad solref 2-6 ms, squeeze ×U(0.6, 1.4).
- **Left press holds only ~3-5 N:** at 6 mm press depth only 66 % of worlds touch the PC (arm sag +
  encoder offset error); at 20 mm all touch, and a horizontal force along the insertion axis moves the
  PC > 1 mm at 5 N (median 0.85 mm, max 16 mm) and tens of mm at 8 N (n=32 per force). The insertion
  policy must stay under a few N or Part 3's hold must be stronger (brace on the far face).
- **Relative actions are unusable on this arm model:** zero action = zero stiffness; in a 2 s
  zero-action hold the plug drifts 23 mm median (71 mm max) and tilts 8° (accumulated targets:
  2.1 mm, 1.7°; n=64). A relative-action teacher ended episodes with the reader tilted 35° on the panel.
  Accumulated targets need a low initial action std (0.2; 0.5 random-walks the targets into the PC).
- **Reward bug fixed:** potential shaping γφ' − φ with φ < 0 pays (γ − 1)φ > 0 per step for staying far
  from the port; PPO learned to hover. Now φ' − φ.
- **Scripted privileged controller** (Jacobian, true reader/port state): 26.2 % seated (n=512, Wilson
  [22.6, 30.1] %), feasibility only: it is slow (median seat 4.9 s of 6 s) and many "jams" sit
  at −5.78 mm when time runs out. Clearance per side: 45 µm (width), 80 µm (height).
- **Starts from Part 2** (`preinsert_error_samples.npz`, 1021 episodes; 6 approach failures with lateral
  > 8 mm or gap outside 0.5-15 mm dropped): each start draws a row, the curriculum scales lateral and
  angles (0.5 → 1.25), the gap is kept (median 5.9 mm). In-hand ±3 mm/±8° is sampled on top (it
  changes the arm configuration, not the plug-port error).
- **Left hold = Part 3's guarded 35 N press** (end state: latched at the true contact as the encoders
  read it, plus Kp⁻¹JᵀF, not capped by the policy's 0.1 rad tracking cap): touches in 100 % of worlds;
  a 1 s axial push at the port moves the PC ≤ 0.3 mm (max) up to 8 N, ≤ 0.85 mm at 12 N, 9 mm at 16 N
  (n = 32 per force). The PC sinks 0.9 mm into its feet under the press.
- **Seating test too strict for soft contacts:** a privileged teacher left 90 of 119 fully inserted
  plugs (5.5-5.8 mm deep, roll 0.2-0.4°, tilt < 1°) failing the aperture test by 3 µm median, 14 µm max
  (contact penetration in a 45 µm-clearance slot). `insertion_metrics(aperture_tol_m=...)` (default 0,
  unchanged); the insertion task uses 25 µm and every eval also reports the strict rate.
- **Scripted privileged controller with Part 2 starts and the guarded hold: 41.4 % seated** (n = 512,
  Wilson [37.2, 45.7] %; 234 stuck at the mouth, 38 jams, 28 slips): real seating happens in this env.
  (26.2 % earlier with uniform starts, simple press and the strict test.)
- **Training success is biased high right after a synchronized reset** (the first episodes to finish
  are the quick successes): a teacher logged 98.8 % that settled to ~80-88 % at steady state. Use the
  held-out evals below, not the training curves.
- **Bug (eval bias):** resets skip `mjw.forward` for speed; after `reset_all` the first step used the
  gravity compensation of the runtime's folded reference pose. First episodes seated 0.8 % vs 14 % for
  later ones (blind student, n = 256 each); fixed (forward after resets of > N/8 worlds), equal after.
  Evals with one episode per world before this fix are biased low and were re-run.
- **Success-first contact DR** (user directive: maximize sim success): the privileged teacher lost
  85 → 71 % at high plug-port μ and jammed more with soft ports (wide DR, n = 1024). Now plug-port μ
  U(0.10, 0.20), port solref 3-4 ms. The same blind PPO policy: 23.6 % (wide DR) → 42.8 % (new DR + new
  PC ranges ±1.5 cm / ±5°, feet μ 0.8-1.0), scale 1.0, n = 1024 each (first-episode evals, biased low).
- **Pads now match Part 1** (55° grasp, measured in-hand pose ±(p1..p99)×1.25, pad μ 1.2, 12 N squeeze,
  impratio 30). Part 1's pyramidal cones go non-finite under insertion actions (197 of 256 worlds in
  60 steps at impratio 30); elliptic cones + impratio 30: 0. The insertion env uses elliptic.
- **Episode length:** at the smallest error level a 10 s episode trained to 99.9 % vs 83.5 % at 6 s
  (training success, same policy init): failures are mostly time spent searching at the mouth.
- **Reward hacking (fixed):** the per-step seated reward (+1 while geometrically seated, success after
  10 consecutive steps) was exploited: policies hovered just short of the hold and collected it every
  step. Training reward rose 4 → 17 per rollout while success fell 100 → 50 %; held-out at scale 0.25
  the degraded checkpoint seated 49.0 % with 438/1024 plugs fully inserted but never holding (vs 93.4 %,
  n = 2048, for its earlier checkpoint). Longer episodes made it worse. Now a one-time success bonus (10).
- **Blind PPO v5 (round-3 Part 2 starts, 25 N hold, Part 1 grasp, 10 s episodes, curriculum 0.5 → 1.0
  → 1.5 × Part 2 errors, 3 seeds):** seed 2 mid-training (stage 1.0, ~8 M steps) held-out at scale 1.0:
  **98.2 % [97.6, 98.7] (2012/2048)**; failures are 35 stuck at the mouth (1 jam). Success by start
  lateral error: ≤ 0.5 mm 99.9 % (n = 1276), 0.5-1 mm 98.1 % (579), 1-1.5 mm 95.8 % (119), 1.5-2 mm
  79 % (48), 2-3 mm 67 % (24); starts > 1 mm below the port are worst (80 %). The blind capture region
  is ~1 mm: **the approach must be within ~1 mm laterally for ≥ 99 %**. Final per-level numbers below.
- **Slip termination** relaxed to a drop (10 mm / 15° in the hand); a few mm of in-hand slip that still
  seats is not a failure.
- **What the wrist camera adds (DAgger from the privileged teacher, n = 1024 each, scale 1.0, in the
  students' training conditions: 60° grasp, wide DR):** see the table in the status below. DAgger
  needs a replay buffer longer than an episode: with 2.4 s per world it forgot earlier phases and
  collapsed as soon as the teacher stopped acting.
- Throughput (V100, elliptic cones, 4096 worlds): ~6.3k env-steps/s alone, ~4k with two runs per GPU.
  Resets no longer call `mjw.forward` (`runtime.reset(forward=False)`): 250 → 30 ms per reset batch.


### 2026-10-31 (mason, `policy/pick`) — Part 1 strip in front of the PC: 5 × 10 cm, gate 2048/2048

- User rejected windows beside the PC. With a world-fixed yaw the tipped grasp needs x ≳ 0.30 m; with
  the reader yaw tied to the right-base bearing (radial reach) the front of the PC is reachable, but:
  (1) the recorded left-arm rest pose holds its hand at (0.03, 0.44, 0.17) m over that zone: the left
  arm is now parked (joint 1 = 0) during the pick; (2) the open fingers spread along x and dipped into
  the arm mounting plate (front edge x 0.035 m, 2.3 cm tall) or hit the PC front: with the full
  opening only reader centres x 0.11-0.14 m remain. Widened to **x 0.095-0.145 × y 0.335-0.435 m**
  with PC x box ±1 cm (pick task), 24 mm approach opening, heading bearing + 15° (± 10°), 3 mm hand
  margin; 29 mm opening or PC x ±1.5 cm gives 3 cm. The flipped grasp (plug toward the arm) does not
  widen it and cannot be turned to the port.
- Expert fixes: pre-grasp 10 cm up (the joint-space move from rest hit the PC case in ~2 % of
  episodes; seen as 10-130 mm settle errors); wait for the tool orientation (< 1°) after the lift
  (weak wrist gains under load: 3-4° at the start of the hold at bearing ≈ 95°). A 24 mm opening in
  the earlier middle window tipped 6 % of readers when the approach settled 2-3 mm off; in the strip,
  with the PC collisions removed, 0/2048.
- **Gate (seed 20000): 2048/2048, Wilson lower bound 99.81 %; 10 s hold 1024/1024.** In-hand spread
  about the mean: position p50 0.30 / p99 0.77 mm, rotation p50 0.07 / p99 0.24°; 10 s creep p99
  0.16 mm / 0.19°. Reorientation to the pre-insert pose: 27/27 PC poses.
- Superseded windows (not for use): beside the PC x 0.305-0.405 (gate 2048/2048), x 0.225-0.305 with
  heading bearing − 50° (gate 2048/2048).
### 2026-11-01 (mason, `policy/approach-hold`) — Round 3: hold 25 N; approach with the measured grasp and DAgger

Current DR (policy/v1 3829c31 + e94a268): PC +/-1.5 cm / +/-5 deg, PC feet mu U(0.8, 1.0); Part 1 grasp:
55 deg, pad mu 1.2, impratio 30, measured in-hand poses (bootstrap of the n = 2048 post-pick poses).

- **Bug (fixed):** after teleporting the right arm (Part 2/3 harness), the latency buffer replayed the
  pre-teleport command for one step: a jerk towards the home pose. With the measured grasp it threw
  the reader out (look-pose error 40-90 mm); with the old grasp it made the reader slip (the "~10 deg
  in-hand rotation" of round 1 was this artifact). All round-1/2 approach numbers carry it. Fix:
  `Harness.fresh` set at every teleport.
- **Hold (n = 1024 per cell, PC still < 0.5 mm / < 0.5 deg over 2 s, axial push at the port):**
  15 N: 1 N 100 %, 3 N 98.1 %, 5 N 97.2 %, 10 N 91.2 %; **25 N: 100 / 99.3 / 98.3 / 95.5 %**; 35 N: 100 / 99.0 /
  98.1 / 95.9 %. 25 N matches 35 N with left joint 2 at ~8.5 N m instead of ~12 N m: recommended. impratio 30
  helps the feet too (35 N, 3 N: 99.0 % vs 97.0 % at impratio 100). No arm-arm contact, no non-finite world.
- **Approach floor** (servo with a perfect estimate, roll corrected at the stop gap): lateral p99 0.11 mm,
  tilt p99 0.31 deg, roll p99 0.12 deg (n = 1024). The estimator is the whole error.
- **DAgger:** re-rendering the states the servo visits (fresh visual DR each time) and fine-tuning:
  v5 (old in-hand, no in-loop data at this grasp) lateral p50 1.21 / p99 4.9 mm -> v6 (one round) 0.52 / 2.08 ->
  v7 (two rounds) 0.43 / 2.21, tilt p99 1.27 deg, roll p99 < 1.1 deg (n = 1024 each).
  **v8 (three rounds, 32,768 in-loop renders): lateral p50 0.37 / p99 1.98 / p99.5 2.21 mm, tilt p99 1.12 deg,
  roll p99 0.68 deg, gap error p99 1.22 mm, 90.5 % inside 1 mm / 2 deg (n = 1024, all finite).** Tilt and roll
  meet the <= 3 deg target; lateral p99 misses 1.5 mm by ~0.5 mm. Ensemble v7+v6: p99 1.96 mm (no gain).
  Deliverable: `results/usbc/policy/approach/preinsert_error.json` (+ `preinsert_error_samples.npz`).
- Did not help: a full-resolution centre crop "fine" net for the last 12 mm (in loop 0.91 / 4.3 mm); multi-frame
  averaging (round 1). The residual is a per-episode bias, mostly vertical (y p99 2.0 mm vs x 1.2 mm): the wrist
  camera looks along the plug. A second RGB view from the side (e.g. a small camera on the table or on the
  left gripper looking across the panel) is the next lever; not added (needs a rig mount and a scene change).

### 2026-10-31 (mason, `policy/pick`) — Part 1 final config: tilt 55°, yaw ±10° window, fixed pad μ; gate 2048/2048

- After the "maximize sim success" directive: grasp tilt 55° (no fallback), reader window x
  0.305-0.405, y 0.055-0.155 m, yaw ±10° (`results/usbc/policy/pick/reader_region.json`), PC box
  ±1.5 cm / ±5°.
- **In-hand spread sweep** (n = 1024 each, seed 7000, 10 s hold, success = lift, tilt < 3°, in-hand
  change < 1 mm / 1° over the hold): random pad μ U(1, 2) + squeeze ×U(0.6, 1.2) 94.3 %; pad μ 1.2
  99.8 % (squeeze ×0.8 / 1.0 / 1.2 alike); μ 1.6 96.8 %; μ 2.0 70.2 %; μ 1.0 100 %; μ 1.2 with
  impratio 30 100 % (creep p50 0.11° vs 0.22° at 10, 0.66° at 3); slower lift (5 cm/s) no gain.
  Creep grows with pad μ: a solver artifact, not slip. Chosen: pad μ 1.2, squeeze ×1, impratio 30.
- **Gate: 2048/2048 (seed 8000, Wilson lower bound 99.81 %); 10 s hold 1024/1024.** In-hand spread
  about the mean: position p50 0.45 / p99 2.7 mm, rotation p50 0.11 / p99 0.80°; 10 s creep p99
  0.47 mm / 0.69°. Tilt after lift max 1.58°. `results/usbc/policy/pick/inhand_spread.json`.
- **ACT v1** (old config: 60°, yaw -30..+10, 650 demos with visual DR): 20k steps 133/512 = 26.0 % held-out
  (seed 5000; failures: no lift 225, tilt 105; stopped at the config change). ACT v2 on the final config: see below.

### 2026-10-31 (mason, `policy/pick`) — Part 1 pick: 60° side grasp, Part 1 reader region, expert gate 2048/2048

- **Grasp (user-confirmed 60° tipped side grasp) as code:** `scenes/reader_grasp.py` (frames,
  `grasp_poses`, `grasp_offset`, `inhand_pose()`). With the FlexPoint sphere pads (4 × 4, r 1.2 mm)
  and fingertips 1.0 mm above the table, pad spheres reach the side face between 5.9 and 8.9 mm
  (the face is 8.83 mm tall): one sphere per side on the face, one or two straddling the top edge.
- **Reach:** only 47 % of the old reader region (n = 21,035 IK points) is reachable at 60° with
  the confirmed hand roll (74 % with the hand turned 180° about the tool axis, 84 % with tilt
  50-65°). The reader must be at x ≳ 0.30 m; the band between the right base and the PC is out.
- **New Part 1 region** (user allowed a smaller window): 10 cm square, body centre x 0.315-0.415,
  y 0.015-0.115 m, yaw -30..+10°. The first 12 cm / ±30° windows failed the expert at 0.9-1 %:
  yaw > +10° (arm tracking limit, settle error 10-35 mm) and the corner nearest the PC (the gripper
  hit a PC shifted 3 cm towards the reader: the region search now checks hand-PC clearance).
- **impratio 100 breaks this grasp.** With the scene's impratio 100 (set for top-down grasp creep)
  the reader stays on the table whenever pad μ < ~1.3, CPU and GPU, and more squeeze makes it worse
  (CPU, μ 1.0: 0/6 lifts at 7.2 or 14.4 N; impratio 1 or 10: 6/6). It also gave 1.5-2 % non-finite
  worlds with low reader-table friction. `PickTaskCfg.impratio = 10` (task-level override of
  `opt.impratio_invsqrt`); no non-finite worlds since.
- **Close until the fingers stop.** With the squeeze randomized ×0.6-1.2 and 200 N s/m finger
  damping, closing takes up to 0.9 s; a fixed 0.6 s close lifted before contact (35 % failures at
  squeeze ×0.6-0.74). The expert now waits for the opening to stop changing.
- **A fast Cartesian integral oscillates** with the identified dynamics (gain 0.25/step: hold error
  27-57 mm); 0.03/step is stable.
- **Expert gate (all DR, held-out seed 3000): 2048/2048, Wilson 95 % lower bound 99.81 %**; reader
  tilt p50 1.08°, max 2.40°; in-hand spread (grasp_site frame) std 0.35 / 0.10 / 0.44 mm and
  0.12 / 0.14 / 0.02°. Sim-only: real pad compliance, friction and squeeze are unmeasured, so Part 4
  should randomize the in-hand pose wider until real grasp captures exist.
- **Longer holds creep (impratio 10):** a 10 s hold after lift (n = 1024, seed 4000) keeps every
  reader (no drops, tilt max 3.1°) but the in-hand rotation drifts median 0.47°, p99 1.5°, max
  2.1° (7.3 % exceed 1°); it grows with pad μ, so it is numerical (soft friction), not slip.
  Part 2/4 carries last several seconds: randomize the in-hand pose for that drift.
- **Open:** the pad model (rigid spheres, 2 ms) loads one sphere per side; a box pad collides with
  nothing against the reader mesh (CPU, `mj_geomDistance` -0.2 mm but no contact). A pull test on
  the real gripper (μ, squeeze, slip) is the measurement this grasp depends on.

### 2026-10-31 (mason, `policy/approach-hold`) — Part 3 left hold and Part 2 approach in the GPU runtime

All numbers: GPU bimanual runtime, full DR (PC layout +/-3 cm/+/-10 deg, PC feet mu U(0.3, 0.9), arm
Coulomb x U(0.5, 1.3), gravity jitter, encoder offsets +/-1.5 deg on joints 2-5, latency 0-1 step) plus,
for Part 2, policy/v1 visual DR (camera pose up to 2x real) and the confirmed 60-deg grasp with a
provisional in-hand pose +/-3 mm / +/-8 deg.

**Part 3 (hold).** `blind_insertion/envs/left_hold.py`, `scripts/task/eval_left_hold.py`, results
`results/usbc/policy/hold/`.
- The left arm is soft: Cartesian stiffness 0.3-0.5 N/mm at the hold poses (kp 80 Nm/rad). A brace
  behind the rear face cannot stop the PC (6.5 % still, n = 1024, push U(0, 20) N). The hold must
  come from feet friction, so press down.
- An absolute IK target plus a force offset puts 2.7-7 N of sideways force on the PC (encoder-offset
  error through the coupled stiffness). A **guarded engage** (descend until the encoder-estimated force
  > 3 N, latch, then add Kp^-1 J^T F) cuts it to 0.4-1.4 N: idle 7 s still in 99.5 % (n = 1024) vs 98.1 %.
- Front half of the top beats the rear half (pitch moment of the load at the port).
- **Recommended:** press, 2 cm in front of the case centre, in line with port 2, 35 N, guarded. PC
  still (< 0.5 mm, < 0.5 deg over 2 s) under a fixed axial push at the port: 0 N 99.8 %, 1 N 99.3 %,
  2 N 97.3 %, 3 N 96.0 %, 5 N 94.0 %, 10 N 84.4 % (n = 1024 each); 0 arm-arm contacts. **Gate
  (99.5 %) met only without load.**
- Sim limit: 2-6 % of worlds slide at 1-5 N although the Coulomb capacity is >= 11 N (35 N down,
  mu >= 0.3); force and position barely change it; stiffer feet contacts (5 ms) made it worse. Soft
  friction on the 5 mm box feet, not the hold. MuJoCo Warp has no noslip solver.
- Left joint 2 holds ~15 N m at 35 N (DM4340 rated ~9 N m): keep the hold short.
- The privileged scripted insertion jams under full DR (45 um clearance, no chamfer; 3.8 % seated),
  so a realistic insertion load in sim is still undefined.

**Part 2 (approach).** `blind_insertion/perception/` (render data, CNNs), `scripts/perception/`,
`scripts/task/eval_approach.py`, results `results/usbc/policy/approach/`.
- Grasp clearance at the pre-insert pose (`check_preinsert_grasp.py`, 27 PC poses x 9 in-hand
  extremes x 4 gaps): 60 deg reaches 24/27 poses; it fails at port x 0.19 m, yaw -10 deg with the
  in-hand +8 deg pitch (joint limit). 55 deg reaches all 27 (margin 0.014 rad), 50 deg margin 0.11 rad.
  Gripper >= 7.8 mm above the table, >= 20 mm from the left arm, >= 37 mm from the case.
- **ZED (RGB) PC pose**: learned CNN on 24,576 DR renders: held-out xy p50 0.79 / p99 3.1 mm, yaw
  p99 1.6 deg (n = 2048, camera error up to 2x real). Classical silhouette fit: 21 % failures, xy p50
  11.8 mm. **Real frames**: CNN vs the manual landmark 0.41 mm (reference good to ~5 mm), yaw +1.9 deg
  (reference yaw not fitted), frame-to-frame std 0.05/0.08 mm (n = 123); classical 6.5 mm. The
  CNN answers on frames without a PC too: it needs a presence check.
- Open loop (ZED + IK, nominal in-hand pose) misses by lateral p50 8.9 / p99 21 mm, tilt p50 4.5 deg:
  in-hand pose and encoder offsets dominate. A wrist-camera servo is required.
- Servo with a perfect relative estimate: lateral p99 2.8 mm without, **0.89 mm with** a Coulomb
  feed-forward on the right arm (`friction_comp`, identified Coulomb, 5 mrad ramp): the joint-friction
  deadband (~12-19 mrad) otherwise stalls/limit-cycles the arm. n = 1021-1022.
- Right-wrist RGB plug-to-port CNN: closed loop lateral p50 1.07 / p99 4.5 mm, tilt p50 1.1 / p99 5.6 deg
  (n = 1021). The vertical (port y) is the weak axis: the camera looks along the plug.
- The reader pivots in the 60-deg grasp: in-hand rotation grows by up to ~10 deg (p99) after placement.
- Retrained with in-hand rotation +/-20 deg (wrist_v2, 32,768 renders; held-out lateral p50 0.64 mm):
  **lateral p50 0.96 / p99 4.5 mm, tilt p50 1.1 / p99 5.7 deg, roll p99 8.9 deg, 46 % inside 1 mm / 2 deg**
  at a 6 mm stop (n = 1021 finite of 1024; 3 non-finite, 3 more off by > 10 mm). 4-frame averaging: no change (the error is a per-episode bias). A fine
  net for the last 12 mm: no better in the loop (n = 512). Stop 3 / 6 / 10 mm: lateral p99 4.6 / 4.5 /
  5.5 mm. The left wrist at the press pose sees only the PC top, so it adds no plug view.
- **Deliverable:** `results/usbc/policy/approach/preinsert_error.json` (+ per-episode samples). Target
  1 mm / 2 deg at 99.5 % is not reached; achievable here is ~1 mm / 1.1 deg median, 4.5 mm / 5.7 deg p99.

### 2026-10-31 (mason) — Visual DR for the bimanual Warp runtime; Warp nominal look; left wrist camera was blind

- **Bug:** in the MuJoCo Warp render (what policies see) the left wrist camera sat inside its own lens
  cylinder: every left-wrist frame was a dark blur. Warp also draws alpha-0 geoms (hidden in the CPU
  renderer): the Menagerie D405 boxes and 104 hidden PC detail geoms. Both are now left out of the
  Warp render context (`visual_dr.render_hidden`).
- **Nominal look (seed 0 = real):** the scene colours were fitted in the CPU renderer; in Warp the wall
  saturated, the arm black rendered grey (82 vs 40) and the gripper white (226 vs 28). New
  `configs/warp_look_duchamp3.json` (`scripts/real2sim/fit_warp_look.py`): group colours plus a per-camera
  RGB gain and black level (right wrist gain 0.74, left wrist lifted blacks +25). On the 21 recorded
  views (`scripts/real2sim/measure_warp_look.py`): fingertip IoU 0.39 -> 0.79 (right 0.78, left 0.0 -> 0.79),
  L1 61.1 -> 19.6, region colour error 39.3 -> 9.6 (0-255). Edge distance 4.16 px over all three
  cameras (before: 3.99 over ZED + right wrist only; ZED 2.84 -> 3.22, right wrist 5.14 -> 5.23):
  darker, real-contrast objects give Canny fewer dark-on-dark edges.
- **Visual DR** (`BimanualCfg.visual_dr`, `envs/visual_dr.py`, off by default), per world, resampled on
  reset: 16 appearance groups (random colour or jittered nominal, flat to glossy), table and wall
  patterns (8 procedural + 6 real-photo patches), key/fill point lights (elevation 35-90 deg, intensity
  0.3-1.0, 3200-6500 K, shadows half the episodes), camera extrinsics (U(0, 2x real): mean = ZED
  4.3 mm / 0.6 deg, wrists 5.1 / 3.4 mm, 1.2 deg) and intrinsics (fx ±1.5 %, c ±0.5 %); per frame on the
  GPU: gain 0.93-1.07, brightness ±12, exposure ±6 %, noise σ 1-4, blur, radial k1 ±0.02.
- **What Warp batches per world:** geom_matid, geom/mat rgba, specular, shininess, emission, texid,
  texrepeat, all light_* fields, cam pos/quat/fovy/intrinsic. **Global:** texture images (we pick per
  world among the model's textures), headlight (compile-time), use_shadows (per-world via
  light_castshadow), alpha. Textures sample only on meshes and planes, so patterns live on two
  visual-only textured quads on the table and wall.
- **Throughput** (V100, 3 cameras, 320 px): legacy 2139 / 2178 frames/s at 512 / 2048 worlds; nominal
  look 2431 / 2471 (fewer geoms in the BVH); DR on 1307 / 1501. Shadows cost most (DR without
  shadows: 1955 at 512); the image model ~12 %.
- Checks: `scripts/checks/check_visual_dr.py` (ranges, per-world, reset, nominal = DR off, finite, real
  frames); contact sheet `results/usbc/policy/visual_dr/contact_sheet.jpg`.

### 2026-10-31 (mason) — Reader region around the PC; PC feet NaN; raceway refit

Reader start region is now every (cell, yaw bin) pair where a top-down pick works and the reader
clears every PC pose (0.20 m^2, yaw +/-90 deg, incl. the middle of the table and in front of the PC).
Yawed PCs went NaN in ~0.1% of GPU worlds: cylinder feet vs box table go through MuJoCo Warp's
single-contact CCD; box feet fix it (0 / 12,288). The backdrop raceway had fitted as a low shelf
(top-down ambiguity) and showed a white line; pinned to the table edge it is an 8 cm vertical face.

### 2026-10-30 (mason) — PC placement search, reader region, fitted backdrop

Reachability search (25,688 PC poses): the legacy runtime PC pose is infeasible for the bimanual
task (left arm cannot hold the PC). New nominal: port (0.22, 0.36) m,
yaw 0, chosen for the best worst-case score over the +/-3 cm / +/-10 deg reset box. The reader is
now sampled over a 34 x 34 cm region, yaw +/-60 deg. Both are wired into `BimanualCfg.task_layout`.
The procedural backdrop is replaced by a backdrop fitted to 21 real frames (object edges within
0.6-1.6 px at 320 px). Details: `results/usbc/bimanual_20261029/README.md`.

### 2026-10-30 (mason) — Gripper sweep fit, backlash <= 2 arcmin

Gripper sweep (both arms, 11 jaw levels) + calipers (lens-to-tip 105.25 mm, both arms): the
unconstrained fit wrongly pulled the right tips 6 mm toward the lens; held to the caliper
distance, held-out IoU left 0.70 -> 0.89, right 0.59 -> 0.81, arms agree to ~2 mm, open jaw
47-48 mm per finger, encoder -> opening linear (<= 0.4 mm RMSE). Applied to the wrist camera by
default (images match, grasp physics unchanged). Backlash on
both arms is <= ~2 arcmin, mostly under camera noise: not modelled. Real joints stick-slip
(6-12 mrad jumps on j2/j3/j5); the identified Coulomb friction is at or above the measured
static band, but static > kinetic is not modelled. Details: `results/usbc/bimanual_20261029/README.md`.

### 2026-10-30 (mason) — Identified real-arm dynamics: tool error 8.5 -> 4.4 mm held-out; fingertip offsets

The bimanual runtime cancelled gravity exactly; the real driver uses its own gravity model.
Static torque-balance fit on 46 free-space trials (two days), per arm: elbow gravity
+11/+17% (right/left), Coulomb up to 1.44 N m (left j3), day-to-day friction spread up to
2.5x. MuJoCo frictionloss creeps under sub-threshold load; a stiff impedance fixes it.
Held-out (other day) closed-loop replay: left elevated 8.47 -> 4.41 mm, right 4.02 ->
3.99 mm. GPU runtime replay 4.9-15.6 -> 1.6-4.4 mm. Now the runtime default, with
per-world friction/gravity randomization. Fingertip silhouette fit: few-mm offsets per
arm (IoU 0.5-0.7 -> 0.77-0.91 held-out), opt-in until a caliper check separates tip
geometry from hand-eye error. Runtime speed (~335 env-steps/s) is the next blocker.
Details: `results/usbc/bimanual_20261029/README.md`.

### 2026-10-26 (22:30 UTC, duchamp3) — ZED pose refit from 3 board positions; held-out check passes

Follow-up to the entry below. `scripts/real/fit_zed.py`: one PnP over 168 corners from three board
positions (`data/calib/zedfit/pos1-3`): 3D corners from the wrist cameras through the arms (mean of
3-4 views), 2D from the ZED's own detections. Written to the spec (`top.extrinsics_cam_in_right_base`
and `_in_left_base`; the old pose kept as `extrinsics_cam_in_right_base_v1`). Change: 22 mm, 1.2°
(mostly rotation traded against translation: 1.2° at 1 m is 21 mm). `results/real/zed_refit.json`.

| Wrist corners in the ZED image (mean px) | pos 1 | pos 2 | pos 3 | pos 4 (held out) |
|---|---|---|---|---|
| old ZED pose | 0.98 | 1.51 | 2.09 | 3.86 |
| new ZED pose | 0.33 | 0.32 | 0.35 | 2.04 |

- Held out (position 4, near the left base: right arm views only): 2.3 / 1.8 px in the ZED image,
  ZED corners 3.2 / 2.4 px in the right wrist image (old pose: 8-11 px). Passes.
- The held-out residual (~2 px, ~2 mm at 1 m) is the right-vs-left arm difference (2-3 mm): the fit
  used both arms' mean, position 4 only the right arm. The ZED pose is now as good as the arms agree.
- **The mm test was the wrong bar:** the ZED's single-board PnP depth of a 12 cm board at 1 m wobbles
  ~5 mm between positions (+2 to +4 mm, then -4.7 mm at position 4), so it measured that, not the poses.
  `verify_cameras.py` now passes on pixels: wrist corners <= 3 px in the ZED image, ZED corners <= 6 px
  in each wrist image, board still (ZED re-shot after the arm views; 10 frames dropped after open).
- A stale ZED frame once showed the board before it was rotated (25° off from both wrists, which
  agreed): the check now catches this.
- The right arm's CAN adapter (`canable2`, behind two chained USB 2 hubs on the ZED's controller)
  dropped off USB twice (14:29, 14:43) for ~7 s, both while idle. The udev rule brought it back up.
  Plug it straight into the PC: a drop during motion leaves the arm limp.

### 2026-10-26 (21:40 UTC, duchamp3) — Three-camera check: wrists agree to 1-3 mm, the ZED is 3.5-5.5 mm off

`scripts/real/verify_cameras.py`: board (8 x 7, 15 mm) at a new spot on the table, (0.30, 0.34) m in
the right base. The ZED finds it (1.05 m away); each wrist camera finds it from two elevations
through FK(q + offsets) @ mount @ PnP; the left arm via the base-to-base transform. Nothing fitted.
`results/real/camera_verify.json`.

| mean / max corner distance (mm) | ZED | right 55° | right 40° | left 55° | left 40° |
|---|---|---|---|---|---|
| ZED | - | 5.2 / 5.6 | 5.5 / 5.8 | 3.5 / 3.9 | 3.9 / 4.1 |
| right 55° | | - | 1.8 / 2.0 | 2.7 / 3.2 | 2.0 / 2.5 |
| left 55° | | | | - | 0.7 / 1.0 |

ZED corners projected into the wrist images: right 11.1 / 8.5 px, left 4.5 / 3.1 px (calibration
hold-out 2.7 / 2.2 px).

- **The two arms and their cameras agree** (1.8-3 mm across arms, through the base-to-base transform;
  0.7-1.8 mm between one arm's two views). Robot base poses, joint offsets and wrist mounts hold up
  at a new board position.
- **The ZED is the outlier:** a uniform shift (the red ZED corners sit ~10 px up-left of the
  detections in the right wrist image), not noise. Its pose came from one board position
  (run_004, a 12 x 10 cm board 1.07 m away: ~100 px in the image, so a 0.5% size error is 5 mm
  of depth). It is right there and drifts elsewhere. Square size is not it: the calibration
  distance (1.03-1.07 m) matches this one, so a scale error cancels.
- Fails the 4 mm / 6 px bar on the right arm only. For the ZED-free proprio policies this does not
  matter; for a policy or a sim scene that uses the ZED pose, refit the ZED from several board
  positions located by the wrist cameras, then check on a held-out position.

### 2026-10-26 (20:40 UTC, duchamp3) — Accumulated-target actions (`EnvCfg.action_mode`); MuJoCo friction creeps

- **Why:** with relative actions (q_target = q_meas + scale * a) the target re-centres on the measured
  joint every step, so a Task H action (scale 0.01, clip 3) never pushes the elbow with more than
  80 x 0.03 = 2.4 N m: below the real left elbow's 3.2-4.8 N m breakaway (entry below).
- **What:** `EnvCfg.action_mode = "accumulate"`: q_target = clip(q_target + scale * a, q_meas +- target_cap),
  `target_cap` 0.1 rad (8 N m on the elbow, past breakaway; equals safety.MAX_STEP). q_target starts at
  the reset pose. The actor observation gets q_target - q_meas (obs 30 -> 36): without it the
  policy cannot see the pending push. GearEnv refuses the mode. `deploy.py` mirrors it from the
  config (`action_mode`, `target_cap`); hold/step sources stay relative.
  Config: `configs/task_h_reader_real_acc.json` (= task_h_reader_real + accumulate).
- **Checks** (`deploy.py check`, PASSED): loop vs direct rollout 3e-16 rad in both modes; observation vs
  `PegEnv._actor_obs_now` 1.2e-7 (incl. the 36-dim accumulate obs with encoder offsets); PegEnv's
  q_target and actuator targets equal the formula exactly over 40 steps x 8 worlds (363 capped
  entries). Elbow frictionloss 4 N m, a = +3 for 1 s: relative moves it 26 mrad, accumulate 499 mrad.
- **Smoke training** (Task H flags, 2,048 worlds, 2.6M steps): runs, 33k steps/s (relative 40k).
  Episodes end earlier under random exploration (mean length 203 vs 433): a random action walk
  integrates into the target. A relative checkpoint cannot warm-start it (obs dim): train from scratch,
  or widen the first layer by hand.
- **MuJoCo friction is soft:** `frictionloss` is a constraint with dof_solref 0.02 s, so a pushed joint
  creeps ~26 mrad/s whatever the friction (same at 4 and 8 N m); the real elbow does not move at all.
  dof_solref 0.005 / 0.002 s cut the creep to 7 / 3 mrad/s and make 0.04-0.1 rad steps stall as on the
  real arm. Not changed in the env yet: a relative-action policy trained on the soft friction will
  overrate how well it can nudge joints 2-3.

### 2026-10-26 (19:50 UTC, duchamp3) — Left YAM Tests 1-2: same latency as the right arm; elbow does not move for 0.04 rad

The right arm's Tests 1-2 repeated on the left arm (same gains, start pose and step sizes).
Data `data/real_yam/left_*.csv`, fit `results/real/step_fit_left.json`.

- **Test 1 (60 s hold):** loop latency 3.75 ms mean, p99 5.8 (right 3.97 / 6.1); encoder 364 Hz;
  hold error < 1.5 mrad. Same as the right arm.
- **Test 2 (0.01 rad x 3, then 0.04 rad x 10 per joint):** no ringing, < 0.4 mrad coupling.

| Joint | Reached at 0.04 rad, left (right) | First motion after send, left (right) | Fitted friction N m, left (right) | Gain scale, left (right) |
|---|---|---|---|---|
| 1 | 100% (103%) | 10.7 ms (11.4) | 0.1 (0.01) | 0.85 (0.7) |
| 2 | 70% (66%) | 13.3 ms (14.4) | 1.0 (1.0) | 0.85 (0.7) |
| 3 | **1%** (9%) | no motion (28.5) | >= 3 (1.5) | not identifiable |
| 4 | 88% (88%) | 7.9 ms (8.5) | 0.2 (0.2) | 1.2 (1.5) |
| 5 | 94% (79%) | 8.0 ms (8.5) | 0.1 (0.2) | 1.5 (1.0) |
| 6 | 78% (94%) | 8.0 ms (8.5) | 0.1 (0.1) | 0.5 (1.5) |

- **Latency is the same on both arms** (onsets within 1 ms): total ~8 ms, latency {0, 1} stands.
- **The high joint 2-3 friction is the YAM, not the right arm's cable.** Both arms stick on joints 2-3
  for 0.01 rad. The left elbow moved 1 encoder count (0.38 mrad) in 20 steps of 0.04 rad: its static
  friction is above kp x 0.04 = 3.2 N m. The fit's 1.0 N m on joint 3 is meaningless (it barely moved).
- **Consequence for Task H (action_scale 0.01, clip 3):** one action moves the target at most 0.03 rad
  from the measured joint, i.e. <= 2.4 N m on the elbow. That is below the left elbow's static
  friction, so **a Task H policy cannot move the left elbow at all**, and the right elbow only barely.
  The target is re-centred on the measured joint every step, so the error never builds up.
  Options: (a) accumulate the target (q_target += scale * a, the absolute-target scheme), (b) raise
  kp on joint 3, (c) larger action scale on joint 3. Each needs a sim retrain; (a) is the standard
  fix for stiction.
- **Left elbow breakaway** (`data/real_yam/left_break_joint3_*.csv`, 5 reps each way, same pose):

  | Step (rad) | Command torque kp x step (N m) | Reached | Left short (mrad) | kp x short (N m) |
  |---|---|---|---|---|
  | 0.04 | 3.2 | 1% | 40 | 3.2 |
  | 0.06 | 4.8 | 48-57% | 29-31 | 2.4 |
  | 0.08 | 6.4 | 74-79% | 17-20 | 1.5 |
  | 0.10 | 8.0 | 84-89% | 11-15 | 1.0 |

  Right elbow, same test (`data/real_yam/right_break_joint3_*.csv`): 0.04 rad 9% (Test 2), 0.06 rad
  64% (22 mrad short, 1.7 N m), 0.08 rad 81% (15 mrad, 1.2 N m), 0.10 rad 89% (11 mrad, 0.85 N m).
  The right elbow breaks away at ~3 N m (just under kp x 0.04); sliding friction ~0.9 N m, as the
  left. Both elbows: static 3-5 N m, sliding ~1 N m.

  Left: static friction (breakaway) is 3.2-4.8 N m; once moving, the joint stops short by an amount that
  shrinks with the step (faster motion coasts further): sliding friction ~1 N m. Up and down steps
  match within a few mrad, so gravity compensation is fine. Stick-slip, not a fault. MuJoCo's
  `frictionloss` has one value for both, so the sim range [0.6, 4.0] spans the two; an
  accumulated-target action scheme removes the dead band in either case.
- `task_h_reader_real.json`: joint 3 friction range widened to [0.6, 4.0] N m to cover the left arm.
  At the top of this range the sim elbow also stalls, so training will show whether the policy copes.
- Joint 1 friction 0.1 N m (right 0.01); joints 5-6 gain scales swap (1.5/0.5 vs 1.0/1.5): wrist gains
  are poorly identified at these friction levels. Both inside the config's kp range except joint 6 at
  0.5 (range 0.6-1.6).

### 2026-10-26 (00:20 UTC, duchamp3) — Encoder offset randomization: `EnvCfg.enc_offset_deg`

- New `EnvCfg.enc_offset_deg` (deg half-range, per arm joint or one value): per world, uniform at every
  reset, added to the observed joints only (q and the FK in the observation). The command
  `q_meas + scale * a` stays in the encoder frame the motors servo in, as on the real arm, where
  `deploy.py` adds the calibrated offsets to the observation only.
- `task_h_reader_real.json`: [0, 1.5, 1.5, 1.5, 1.5, 0] deg (joints 2-5; the measured offsets are
  0.4-1.3 deg, and joints 1 and 6 are not observable by the calibration).
- Checks (512 worlds): observed q - true q = the drawn offset (1e-7); observed TCP shifts 7.7 mm median,
  20 mm max; zero-action drift over 50 steps identical with and without offsets (52.507 vs 52.509 mrad
  median: the start-pose settling), so the offsets do not move the arm.

### 2026-10-25 (23:55 UTC, duchamp3) — Training config with the measured real YAM: `task_h_reader_real.json`

- `EnvCfg.arm_frictionloss` now takes one `[lo, hi]` per arm joint (log-uniform per reset) as well as
  one shared range. Checked on 512 worlds: each joint's draws stay in its own range.
- `configs/task_h_reader_real.json` = `task_h_reader.json` + the measured robot (2026-10-25 entries):
  friction per joint [0.01, 0.4] / [0.4, 2.0] / [0.6, 3.0] / [0.08, 0.5] / [0.08, 0.5] / [0.04, 0.3] N m
  (fit 0.01 / 1.0 / 1.5 / 0.2 / 0.2 / 0.1), `arm_kp_scale` [0.6, 1.6], `arm_kd_scale` [0.7, 1.4] (real kd
  5 vs 6), `noise_rand` latency {0, 1}, quantization 14 bit (measured 0.381 mrad), encoder offsets ±1.5° on joints 2-5 (next entry), q noise {0, 1e-4,
  5e-4}, backlash {0, 1, 3, 5}' (not measured yet), history 10 (as `H-reader-proprio-v2-s0`, for `--init`).
- Smoke run (1,024 worlds, 2 M steps from scratch): 21.6k steps/s, finite; the friction does not change
  the solver overflow counts under random actions (base 48k vs 51k).
- With the reader's `action_scale` 0.01, joint 3's dead band (~19 mrad) is ~2 action units: expect the
  v2 policy to drop on the first evals with this config.
- Server run (branch `real-yam-setup`):

  ```
  .venv/bin/python -m blind_insertion.train --name H-reader-proprio-real-s0 --robot yam --family reader \
    --variant_kw '{"clearance": "spec"}' --env_file blind_insertion/configs/task_h_reader_real.json \
    --env_json '{"peg_offset": 0.006}' --init runs/H-reader-proprio-v2-s0/last.pt --nworld 8192 --steps 150e6
  ```

  Evaluate the v2 policy on this config first (`blind_insertion.eval --env_file ...`): it shows how much
  the real friction costs before any retraining.
- `deploy.py run --source policy --env_file <training config>` takes the action scale (0.01 for the
  reader), history and grasp frame from the config; a history-10 reader checkpoint loads and acts.


### 2026-10-25 (23:40 UTC, duchamp3) — Real YAM step test: latency ~8 ms; joint friction 10-150x the sim's

Test 2 of `docs/real2sim/REAL_YAM_LATENCY_BACKLASH.md` on the right arm, at the sim gains (kd 5 on joints 1-3,
the DM limit), from q = (0, 1.2, 1.0, -0.6, 0, 0) (wrist 0.28 m above the table). Each joint alone:
0.01 rad x 3 (probe), then 0.04 rad (one full action) x 10. No stop, no ringing, < 1 count of
coupling into other joints (joint 1: 4 mrad, the base swing). Data `data/real_yam/steps_joint*.csv`,
fits `results/real/step_fit.json` (with friction) and `step_fit_nofriction.json`.

| Joint | Reached at 0.04 rad (0.01 rad) | First motion after the bus send | Fitted friction (N m) | Fitted gain scale |
|---|---|---|---|---|
| 1 | 103%, overshoot 9-12% (48%) | 11.4 ms (sim: 9-10) | 0.01 | 0.7 |
| 2 | 62% (0%) | 14.4 ms | 1.0 | 0.7 |
| 3 | 8% (0%) | 28.5 ms | 1.5 | 0.35 (not identifiable) |
| 4 | 87% (13%) | 8.5 ms (sim: 4-5) | 0.2 | 1.5 |
| 5 | 79% (42%) | 8.5 ms (sim: 4) | 0.2 | 1.0 |
| 6 | 94% (77%) | 8.5 ms (sim: 3) | 0.1 | 1.5 |

- **Latency.** Command on the bus -> first encoder motion is 1.5-5.5 ms later than in the sim (joints
  1, 4-6; the encoder period is 2.9 ms, so that is the resolution). With the 4.0 ms loop part (Test 1):
  **total ~8 ms = 0.4 control steps**, jitter 1.6 ms. Between the sim's k = 0 and k = 1: per the
  protocol, randomize `latency` over {0, 1} (or add a sub-step delay to the env). The fit script's own
  delay comes out 0 ms: the onset is a small part of the fitted curve and gain and friction absorb
  it, so read the delay from the onset, not from `delay_ms`.
- **Friction is the big gap.** The sim has `frictionloss` 0.01 N m; the fit gives 1.0 (joint 2) and
  1.5 N m (joint 3), 0.1-0.2 on the wrist. A command closer than friction / kp does not move a
  joint: ~12 mrad on joint 2, ~19 mrad (half an action unit) on joint 3, 3-7 mrad on the wrist. Small
  corrective actions, the ones insertion needs, are lost on joints 2-3. Without friction the fit is
  meaningless on joints 2-3 (delay 100 ms, gain x4 / x0.25, rms 0.26 / 0.51 vs 0.09 / 0.035).
  i2rt's Coulomb compensation (0.3 N m x sign(velocity) on joints 1-3) does nothing at rest.
- Joint 3's friction (8% of a 0.04 rad step, ~3 N m of command torque) is high for a DM4340: check
  the elbow cable for drag. Joint 2 moved 110% on the first step after the move and 62% after:
  hysteresis.
- **Gains.** Where friction lets it be measured (joints 1, 4-6) the stiffness is 0.7-1.5x the sim's.
  Friction and gain trade off in the fit, so these are rough.
- **For training**: `EnvCfg.arm_frictionloss` exists but draws one log-uniform range for all joints;
  the real arm needs per-joint ranges (joints 2-3 up to ~1.5-2 N m, wrist ~0.3). Plus
  `arm_kp_scale` ~(0.6, 1.6) and latency {0, 1}. A policy trained on 0.01 N m will likely stall in the
  dead band on joints 2-3.


### 2026-10-25 (23:05 UTC, duchamp3) — Real YAM deployment loop; Test 1: loop latency 4.0 ms, encoder 343 Hz

`scripts/real/deploy.py` runs the sim's scheme on the real arm: 50 Hz, `q_target = q_meas + 0.04 a`
through `safety.SafeArm`, observation in the model joint frame (`q + rig.joint_offsets`). The
offsets cancel in the target (both terms are motor readings) and act only through the observation.

- **Offline check** (`deploy.py check`): loop + a CPU sim arm with the right arm's offsets vs a direct
  rollout of the scheme, 300 random-action steps: max difference 5.6e-16 rad. Loop observation vs
  `PegEnv._actor_obs_now` (axial `phase_c_yam` and top-grasp `task_h_reader`): 1.2e-7.
- **Gains**: the real arm now runs the sim's kp 80/80/80/30/30/30. kd is 5/5/5/2/2/2, not the sim's
  6/6/6: the DM motors' MIT kd range ends at 5.0. No oscillation at the sim gains in a 60 s hold
  (wrist joints moved 2 encoder counts).
- **Test 1** (right arm, hold 60 s, a random-weight policy-sized TCN forward pass per step;
  `data/real_yam/hold_60s*`). Timing comes from hooks in i2rt's threads (CAN-burst times of every
  encoder reading and of the first send of each command):

| | mean | std | p99 |
|---|---|---|---|
| Control period (ms) | 20.00 | 0.54 | |
| Age of the reading used, at the step start (ms) | < 1.1 | | |
| Compute: read, observation, policy, safety checks (ms) | 1.3 | 1.0 | 2.7 |
| Action ready -> command on the CAN bus (ms) | 3.0 | 1.1 | 5.1 |
| **Loop latency `t_send - t_obs` (ms)** | **4.0** | **1.6** | **6.1** |

- So the software loop adds 0.2 control steps. The motor's own delay (bus to motion) comes from the
  step test (Test 2); total latency = 4 ms + that.
- One hiccup in 3,000 steps: step 230 took 41 ms of compute (cause unknown, maybe Python GC); the
  next command was replaced before it reached the bus.
- Encoders: DM chain loop 343 Hz (p99 period 3.1 ms). Resolution 0.381 mrad (16 bits over the DM
  ±12.5 rad range), the same as `q_quant_bits` = 14 in the sim. Hold noise is pure quantization
  (flips of one count).
- Static hold error (command - reading): joint 2 -0.7 mrad, joint 4 -1.4 mrad, others 0: i2rt's
  gravity compensation factors (1.1 on joint 2, 1.2 on joint 4) overshoot by ~0.05 N·m. Negligible.

### 2026-10-25 (22:30 UTC, duchamp3) — Real YAM rig: safety, cameras calibrated, joint offsets 0.4-1.3°

duchamp3 (RTX 4090) is the real-arm PC: two YAMs (i2rt, DM motors) on CANable2 adapters, two
wrist webcams, a ZED 2 above the table. Branch `real-yam-setup` (PR #1). Tools in `scripts/real/`
(`setup_real.sh`, `rig.py`, `safety.py`, `estop.py`, `calib.py`); results in
`blind_insertion/configs/real_cameras_duchamp3.json` (sim reads it via `EnvCfg.cam_spec`).

**Calibration** (checkerboard 8 x 7 inner corners, 15 mm; arm views from planned poses plus
hand-held frames over the whole image; hand-eye by Park-Martin + least squares on the Menagerie
`link_6` frame, because OpenCV 5 has no `calibrateHandEye`):

| | Right wrist (policy arm) | Left wrist |
|---|---|---|
| Lens | fx 682.6, strong barrel (k1 -0.33), ~86 x 74° | fx 544.3, almost none (k1 -0.004), ~99 x 86° |
| Intrinsics RMS | 0.74 px (36 arm + 267 hand-held frames) | 0.23 px (41 + 150) |
| Policy crop / sim fovy | 952 px / 69.7° | 932 px / 81.1° |
| Hand-eye hold-out | 2.7 px with joint offsets (8.0 without) | 2.2 px (3.0 without) |
| Sim camera vs real corners | 0.83 px median at 480 px | 0.95 px |

- The two "identical" webcams (Realtek 0bda:5883, same iSerial) have different lenses. The sim
  camera (Menagerie D405, ~58° fovy) sat 29 mm and 7° from the real right camera.
- **Real arms read 0.4-1.3° off the Menagerie kinematics** on joints 2-5 (right: 0.73, 0.01, 1.33,
  -1.11°; left: 1.20, -0.41, 0.38, -0.03°). On the right arm this, not the camera, was most of the
  hand-eye error (8.0 -> 2.7 px). 1° on a wrist joint is several mm at the gripper. Joints 1 and 6
  are not observable this way (they trade off with the board pose and the camera mount).
- Rig geometry: left base 0.608 m to the left of the right base, level to 0.8 mm, yaw < 1°; ZED
  0.87 m above both bases.
- Arm-only views covered just the image centre (15 of 64 cells): focal length and distortion
  traded off (left fx 546 vs right 677 looked like a fit artefact). Hand-held frames (50/64
  cells) settled it.

**Hardware and safety lessons** (each cost a stop or a near miss):

- **Clamp the bases.** An unclamped base nearly tipped at ~0.58 m reach. Calibration reach is now
  capped at 0.42 m.
- **i2rt's default zero-gravity mode floats the right arm into the table**: its 1.1x gravity
  feed-forward beats that shoulder's friction (CAN log: kp = 0, shoulder 0 -> 0.79 rad in 1.3 s).
  `safety.connect` uses PD hold from connect (2.9 ms float window); hold drift 0.008 rad in 5 s.
- One CANable2 adapter (0045006F) was bad: motors timed out at random on any arm cabled to it.
- i2rt holds the wrist at kp 10 (joints 5-6), so it trails commands by up to 0.19 rad in moves.
  Capping each command step from the *measured* pose made that lag snowball into a stop; the cap
  now builds on the previous command, and lag limits are per joint (0.15 / 0.2 / 0.3 rad).
- Safety checks need real margins between levels (choose a pose 90 mm above the table at the
  wrist, path 75, live stop 50, homing 20): with equal levels, a millimetre of sag stopped runs and
  trapped the arm where it could not go home.
- ZED rectified board pose: 0.05 px reprojection, 0.09 mm jitter; single readings scatter up to
  ~1 mm while the arm shades the board, once 2.07 mm with the board still in place.

**Open (sim realism)**, roughly by impact:

1. Latency and backlash (`docs/real2sim/REAL_YAM_LATENCY_BACKLASH.md`) not measured; the rig is ready.
2. Gains differ: real i2rt PD kp 80/80/80/40/10/10, kd 5/5/5/1.5/1.5/1.5; sim 80/80/80/30/30/30,
   kd 6/6/6/2/2/2. Set the real gains to the sim's, or train with the real ones.
3. Joint offsets are in the spec (`rig.joint_offsets`) but nothing applies them yet (no deployment
   loop); they could also be randomized in sim.
4. Real fixture and part poses not measured (the calibrated ZED can now locate them).
5. Real finger pads (yellow, 3D printed) differ from Menagerie's fingertips: grasp contact and
   what the wrist camera sees.
6. Wrist payload: sim carries the 109 g D405 bracket; the real webcam and cable are not weighed.
7. Wrist cameras run auto exposure and white balance; frame latency not measured.
8. The ZED is not in the sim (only needed for top-camera policies).

Tests, pass criteria and the order to close these: `docs/real2sim/REAL_RIG_GAPS.md`.

### 2026-10-25 (19:00 UTC, mason) — Task H v2: ACT v2 + insertion v2 + release, 90.6% inserted / 87.5% seated

- **ACT v2** `H-act-pick-v2`: 200k steps (12.1 h, V100) on `data/act/reader_pick_v2` = 1,536 expert demos in the v2
  scene (all ok), a new spec reader per batch of 64, DART OU noise (tau 0.25 s) on 60% of episodes: sigma 0.01 rad in
  approach/carry, 0.004 rad in the first 0.8 s of alignment at the port. Pick DR: friction, grip force, kp/kd,
  frictionloss, latency 0-2 steps, encoder noise, cameras, light, colours.
- **Pipeline** (`pipeline/pick_insert.py`, ACT -> insertion v2 -> release, spec parts, n = 256, 95% Wilson CI):

  | ACT ckpt | Handoff | Inserted | Seated after release | Held | Lateral at handoff (median / p90) | Gap (median / p90) |
  |---|---|---|---|---|---|---|
  | 150k | done only | 82.8% (77.7-86.9) | 79.3% | 96.1% | 3.0 / 5.7 mm | 7.8 / 11.1 mm |
  | 150k | settle | 88.3% (83.8-91.7) | 85.9% | 97.3% | 2.6 / 5.1 mm | 7.4 / 10.7 mm |
  | 200k | done only | 79.3% (73.9-83.8) | 77.7% | 92.6% | 3.2 / 6.6 mm | 6.4 / 9.6 mm |
  | **200k** | **settle** | **90.6% (86.4-93.6)** | **87.5%** | 96.5% | 2.5 / 5.1 mm | 6.1 / 8.9 mm |

  Old ACT 200k (v1 scene, no gain DR): 85.5% / 84.0%, lateral 3.6 / 9.6 mm, 34% of handoffs at the 12 s timeout.
- **Gap fix 1 (settle handoff)** adds 5.5-11 points: without it ACT keeps drifting at the port until done fires
  (or the timeout), and the reader is lost more often (held 92.6% vs 96.5%). Handoff time in the video: 5.1-6.2 s (was ~7 s).
- **Gap fix 2 (DART + per-batch parts)** halves the p90 lateral handoff error (9.6 -> 5.1 mm), inside the range where
  insertion v2 gets 95-97%. Remaining losses: reader not held (3.5%) and failed insertions from large gaps.
- Video: `results/H/act_v2_200k_pipeline.mp4` (8 episodes, 7 seated after release).

### 2026-10-25 (06:30 UTC, mason) — Task H v2: insertion on the measured reader with gain/friction DR, 97%/95%/90%

- **Change**: v2 scene (measured reader 34.73 x 8.83 x 42 mm body, 90.46 mm long; lit lab room; top camera
  between the arms) and new randomization: servo kp x U(0.8, 1.25), kd x U(0.7, 1.4), joint frictionloss
  U(0.005, 0.1), on top of the per-world encoder noise / quantization / backlash / latency (`noise_rand`).
- **Run** `H-reader-proprio-v2-s0`: 150 M steps from `H-reader-proprio-topft-s0`, 8,192 worlds, start offsets up to
  6 mm, history 10 (3.6 h on one V100). Training success 86-88% at the end.
- **Eval**: unseen spec parts, the same v2 DR, n = 1,024 each (`results/H/eval_{topft,v2}_dr_*.parquet`):

  | Start offset (peg_offset) | topft (before) | v2 |
  |---|---|---|
  | 1.3 mm | 73.5% (70.7-76.1) | **97.3%** (96.1-98.1) |
  | 3.9 mm | 68.1% (65.1-70.8) | **95.1%** (93.6-96.3) |
  | 6.0 mm | 60.9% (57.9-63.9) | **89.5%** (87.4-91.2) |

  95% Wilson CIs. topft fails mostly by leaving the fixture (22-34%); v2 failures are almost all
  `left_fixture` (2.7% / 4.5% / 10%), `never_found` <= 0.4%. The gain/friction DR was the main loss for the old
  policy, and training on it recovers it fully. Video: `results/H/insert_v2.mp4` (3/3 seated).
- ACT v2 (1,536 DART demos, new parts per batch) is training; pipeline numbers follow.

### 2026-10-24 (20:40 UTC, mason) — Task H: ACT 200k + release, pipeline 85.5%

- Pipeline now releases after seating (`pick_insert.Release`: hold joints, open 0.4 s, lift 8 cm).
  Expert pick: 127/128 inserted, 126/128 still seated after the hand is clear.
- ACT + fine-tuned insertion + release, n = 256 each (`results/H/pipe_act{160,200}k_ft_release.json`):

  | ACT steps | Inserted | Seated after release | Held | Handoff lateral (median / p90) | Timeout handoffs |
  |---|---|---|---|---|---|
  | 120k (n = 128, no release) | 72.7% | – | 95% | 5.2 / 14.5 mm | – |
  | 160k | 84.8% | 84.4% | 99% | 5.8 / 14.8 mm | 41% |
  | 200k | 85.5% (CI 80.7–89.3) | 84.0% | 100% | 3.6 / 9.6 mm | 34% |

- The remaining failures are ACT's lateral error at handoff. One third of handoffs still happen at the
  10 s limit (ACT does not raise `done`). Next options: handoff on arm-at-rest, more demos, or a wrist-camera crop near the port.
- Video: `results/H/act_200k_ft_release.mp4`.

### 2026-10-24 (16:40 UTC, mason) — Task H: fine-tuned insertion + ACT 120k, pipeline 73%

- **Insertion fine-tune** `H-reader-proprio-topft-s0`: 150 M steps from the stage-2 checkpoint (231 M),
  spec USB-C parts only, start offsets up to ±6 mm. Unseen spec parts (n = 1,024 each, `results/H/eval_ft_final_*.parquet`):

  | Median start offset | Before (231 M) | Fine-tuned |
  |---|---|---|
  | 1 mm | 74.9% | 92.6% |
  | 3 mm | 48.0% | 89.6% |
  | 4.6 mm | – | 81.2% |

  The failure mode changed from "never found" (434 of 1,024 at 3 mm) to "left fixture" (105; 1 never found).
- **Expert pick + fine-tuned insertion**: 97.7% (n = 256, CI 95.0–98.9%), handoff lateral error 0.45 mm, gap 6.1 mm.
- **ACT 120k (ensembled) + fine-tuned insertion**: 72.7% (n = 128, CI 64.4–79.6%), from 29.7% (ACT 60k + old policy).
  Held 95% (from 73%). Handoff lateral error is still 5.2 mm median (p90 14.5), so the insertion policy now
  recovers from most of ACT's error. Many handoffs happen at the 10 s time limit: ACT does not always raise `done`.
- Videos: `results/H/act_120k_ft_rollouts.mp4`, `results/H/insert_ft_final.mp4`. ACT continues to 200k (GPU 3).

### 2026-10-24 (07:45 UTC, mason) — Task H: pick the SD reader (ACT) and plug it into the mini PC (blind policy)

New task close to the planned real experiment (Yash, 2026-10-24): YAM bench as RoboSimStudio
`yam_i2rt`; GMKtec G10 mini PC (103 x 42 x 98 mm) back right with its rear panel facing the
arms; SanDisk USB-C SD reader lying flat in front, plug towards the PC; target port: second
USB-C from the left. The right arm picks the reader top-down (ACT on top + wrist cameras), lines
the plug up 3-10 mm in front of the port, and hands off to the blind TCN policy.

- **USB-C at spec dimensions** (USB-IF Type-C R2.x): receptacle 8.34 x 2.56 mm, plug shell
  8.25 x 2.40 mm and 6.65 mm long, tongue 6.69 x 0.70 mm. The fit is anisotropic (45 um per
  side across the width, 80 um across the thickness), so the plug is the opening offset by the
  thickness clearance and widened (`Variant.plug_widen`). Final-stage parts draw opening and
  plug within tolerance (20-90 um / 50-110 um per side). Earlier E runs used 8.5 mm plugs,
  isotropic clearance and a 6.5 mm tongue. Cavity 6.50 mm: mjfit needs the hollow plug longer
  than the cavity + clearance, so the plug stops on the floor with the overmold 0.15 mm short of
  the panel. Checks: an aligned push seats every part; 0.9-1.2 mm offsets are blocked at the
  face; 0.5-0.6 mm offsets enter by tilting (free part, no hand), as with the E physics.
- **Top-down grasp** (`EnvCfg.grasp_mode="top"`): part axis horizontal along tcp +x, pads
  across the 36 mm body. The mirror grasp (tcp y = +part x) is outside the YAM wrist limits.
  In-hand yaw is 0 (parallel pads force the faces flat). `grasp_along` acts along the part axis
  in this mode (along the closing axis it pushed the part off-centre: 45/256 dropped). The pads
  sit 2-5 mm above the body centre (`grasp_y`) because the fingertips must clear the table.
- **Pick scene physics** (`scenes/pc_bench.build_scene`): same builder, pads, port and case as
  PegEnv. Two fixes for the pick: the insertion env's stiff pad contact (time constant 2 ms,
  solimp 0.99/0.999) is needed or the 27 g part sinks millimetres into the pads; finger damping
  200 N s/m (about 6 cm/s closing) or the constant-force fingers slam shut and throw the part.
- **Expert** (`envs/pick_env.Expert`, privileged): 512/512 demos reach the pre-insert pose,
  lateral error 0.15 mm, tilt 0.16 deg (after waiting for the lagging arm before closing, and an
  integral correction of 0.85 mm servo sag). 50% of episodes have DART noise (0.01 rad) during
  reach and carry. Data: `data/act/raw_v1` -> LeRobot `data/act/reader_pick_v1` (H.264).
- **Insertion policy** `H-reader-proprio-top-s0` (GPU 1): usbc curriculum, contact loss,
  history 10, randomized noise model, start gap 0.3-10 mm. 69% training success at 31 M steps
  (stage 1).
- Pipeline check (expert pick + stage-1 checkpoint, 64 episodes): handoff 64/64, plug 0.48 mm
  from the axis, gap 3.7-9.1 mm; insertion 3% (expected: stage-1 policy, spec clearance).

### 2026-10-22 (19:30 UTC, mason) — Redo batch: C specialists, D fixed row + diagonal, E noise stalls

All on the corrected backlash model, 2,048 starts per cell, random robot conditions for D.
- **C allmid / allmax / backlash5 retrained** (50.3 M, stop at 99%): own condition 99.1 / 98.6 / 99.6%.
  allmid (latency 1) gets 36.2% on base, allmax (latency 2) 14.4% (old model: 45.4 / 20.7%). backlash5 base
  99.6% (old 97.5%). Same conclusion as before: allmid/allmax are latency-matched specialists; use `randall`.
  `results/C/C-matrix.*` now has all 17 training conditions.
- **D fixed 8 mm row** (median start 3 mm), 1.0 / 0.5 / 0.2 / 0.1 / 0.05 mm: 98.9 / 95.8 / 91.7 / 88.4 / 83.7%
  (scaled: 98.5 / 97.5 / 98.7 / 98.1 / 94.2%). The wide start costs up to 10 points at tight clearance, almost all
  never_found (191 / 165 / 126 at 0.05 / 0.1 / 0.2 mm). 90% training success at 38-44 M steps (scaled 7-9 M);
  0.1 and 0.05 mm never reached 90% in 150 M.
- **D diagonal**, scaled / fixed: o60_c1 99.6 / 99.7%, o30_c0.5 98.2 / 98.1%, o15_c0.2 97.6 / 97.8%. Size alone
  does not lower success; the start area matters only at 8 mm and <= 0.2 mm.
- **D tier 2 scaled so far:** o60 c0.5 / 0.2 / 0.1 -> 98.7 / 98.0 / 98.6%; o30_c1 97.3%. o60_c0.05, o30 c0.2 / 0.1 /
  0.05, o15 c1 running; o15 c0.5 / 0.1 / 0.05 queued.
- **E noise (USB-C, randomized noise/quant/backlash 0-5'/latency 0-2) stalls in curriculum stage 2**: 276 M steps
  at 85-91% training success, rolling 5 M-step success never >= 0.9 (the advance rule); 300 M cap. Final-range test 21.6%
  (curr, no noise: 79.0%); never_found 963, wedged_partial 421, slipped 150. Capture 54% at 0.5 mm (curr 97%);
  clearance sweep 3.0% at 20 um to 69% at 200 um. The peg shrugs the random model off (randall 98.6%); the USB-C
  does not. Consistent with athena5's USB-C backlash and latency results. Options for Yash: fine-tune from
  `E-usbc-proprio-curr-s0` with the noise model (as fullft3 did for backlash), or narrow the ranges (latency 1 only).

### 2026-10-23 (10:00 UTC, mason) — Report v42 rewrite; scripts/ reorganized

- Report v42: summary of key results first, one claim per RQ, tables with the old backlash model removed,
  new probe / signal / replay findings in RQ3 and RQ4.
- `scripts/` split by purpose: `eval/` (was `eval_phase_*.sh`, `usbc_*_eval.sh`, `server/eval/*`),
  `analysis/`, `report/` (was `phase_*_report.py`, `build_report.py`, demos), `real/` (was
  `real_yam_latency.py`), `tools/` (setup checks, `local_queue.sh`, was `queue.sh`). Paths in docs,
  manifests, runner and figures updated; this notebook keeps the old paths (history). Empty
  `blind_insertion/baselines/` removed (the spiral baseline is in `eval.py`). Guides: `README.md`
  (project overview + layout), `scripts/README.md`.
- Checks: all scripts compile / `bash -n`; `test_run_jobs.py` passes; `make_manifests.py --check` 171 OK;
  `scripts/report/phase_{c,d}.py` rebuild identical outputs.

### 2026-10-23 (09:00 UTC, mason) — Plan analyses: C-R3 probe, D signal and probe vs size, E7 probe, §11.5 replays

Scripts: `scripts/analysis/run_probes.sh`, `run_signal_vs_size.sh`, `run_replay_check.sh`.
`analysis/probes.py` got `--near_mm` (far/near split, scaled with part size: 5 mm x opening / 60 mm; USB-C 1 mm).

**Probe (balanced accuracy over occurring modes; latent / current obs / random encoder), 1,536 worlds:**

| Policy | Latent | Current obs | Random enc. | Max swap ratio inside<-search |
|---|---|---|---|---|
| C YAM base (C-R3) | 0.60 | 0.48 | 0.43 | 1.50 |
| D 60 mm, 0.05 mm | 0.64 | 0.53 | 0.54 | 2.46 |
| D 30 mm, 0.05 mm | 0.58 | 0.44 | 0.34 | 1.87 |
| D 15 mm, 0.05 mm | 0.54 | 0.45 | 0.37 | 2.13 |
| D 8 mm, 0.05 mm | 0.67 | 0.50 | 0.46 | 2.03 |
| E USB-C base (E7) | 0.70 | 0.54 | 0.47 | 2.28 |
| E USB-C h10 | 0.64 | 0.56 | 0.53 | 1.94 |
| E USB-C no contact loss | 0.33 | 0.36 | 0.32 | 1.32 |

- Gate C probe item passes (YAM latent > current obs). The latent encodes contact at every size (D Q3:
  yes, no trend with size). Far<->near swap ratios stay 0.93-1.18 everywhere (as in the paper).
- Without the contact loss the USB-C latent holds no contact information (0.33 = random encoder): this is
  why that run does not learn.

**Signal vs size (D §10.3 (3); CPU, YAM lab table, sigma_q 1e-4):** A-S5 F_min vertical 0.053 / 0.068 /
0.050 / 0.054 N (60 / 30 / 15 / 8 mm, results/D/signal/*.json), lateral ~0.05 N. A-S6 drop-in (2 N press,
1 mm clearance) joint-velocity peak-to-noise 7.3 / 9.6 / 20.9 / 24.0. The signal does not weaken as parts
shrink (the drop-in peak grows). So the 0.05 mm losses at 30 and 8 mm are not a signal limit.
(The Phase A YAM value 0.79 N came from the old setup with 14 mm gravity sag; not comparable.)

**§11.5 replay check (h10 policy, 2,048 starts, 95.2%; 20 successes + 19 failures rendered):**
no physics artifacts. Successes seat upright (final tilt <= 1.9 deg, seat depth -7.3 +- 0.4 mm). Wedged
cases rest 2 mm above the port with one corner in the opening (no tunneling). But 81 of 98 failures
(base policy: 311 of 431) end with the drive tilted > 15 deg in the grasp: the drive turns between the
pads and the policy cannot recover. The slip detector checks translation only, so these count as
"left fixture" / "never found". The grasp is the main USB-C failure, not the search.

### 2026-10-23 (05:00 UTC, mason) — E4 ablations: contact loss is necessary for USB-C; history 10 best

Port-up USB-C, lab table, 300 M steps, seed 0, final-stage test (2,048 matched starts, no robot errors):

| Run | Final stage | Wilson 95% | Stage reached | Final stage entered | Median time | Travel |
|---|---|---|---|---|---|---|
| E-usbc-proprio-curr-s0 (base, h30) | 79.0% | 77.1-80.7 | 5 | 260 M | 2.00 s | 28 mm |
| E-usbc-proprio-nocontact-s0 | 10.2% | 9.0-11.6 | 2 (plateau 62-68%) | - | 4.94 s | 87 mm |
| E-usbc-proprio-h10-s0 | 95.3% | 94.3-96.1 | 5 | 190 M | 1.62 s | 22 mm |
| E-usbc-proprio-h60-s0 | 80.6% | 78.8-82.3 | 5 | 273 M | 2.18 s | 30 mm |

- No contact loss: 1,377 of 1,839 failures never find the port. On the peg (RQ1) the loss only sped up
  learning 2.2x; for USB-C it is the difference between learning and not.
- h10: confound, it spent 110 M steps on the final stage vs 40 M for the base. A fair check is the base
  continued to 370 M steps (not queued: user asked to cut training).
- nocontact stage-2 stop is a plateau (flat 62-68% for 280 M steps), not the window bug. h10/h60 reached stage 5.
- Runner finished 04:46; 12 tier-2 D evals wait on deferred trainings (expected). All GPUs free.
- Report v41: RQ4 ablation paragraph and table.

### 2026-10-23 (04:00 UTC, mason) — D 0.05 mm second seeds: the 3-4 point loss holds

- `D-peg-proprio-o30_c0.05_scaled-s1`: 1,954/2,048 = 95.4% (s0 94.4%). Failures 60 wedged partly inside,
  28 never found, 6 left fixture: wedging dominates again.
- `D-peg-proprio-o8_c0.05_scaled_sph-s1`: 1,945/2,048 = 95.0% (s0 94.2%). Failures 65 never found,
  36 wedged, 2 jammed: search failures dominate again. Median time 0.96 s (s0 0.94 s).
- So at 0.05 mm the 30 and 8 mm pegs lose 3-5 points against 60 mm (99.0%) and 15 mm (97.6%) on two seeds.
- `phase_d_report.py`: D-grid.csv keeps every seed; heatmaps and scale.json use seed 0.
- Report v40: seed result in RQ3; "RQ3 seeds" next-step removed.

### 2026-10-23 (00:35 UTC, mason) — D scaled grid complete: 15 mm / 0.05 mm 97.6%

- `D-peg-proprio-o15_c0.05_scaled_sph-s0` (corrected backlash, 150 M steps, 3.3 h): 1,999/2,048 = 97.6%
  (Wilson 96.9-98.2%). Failures 28 wedged partly inside, 20 never found, 1 jammed.
- The 0.05 mm column is 99.0 / 94.4 / 97.6 / 94.2% (60 / 30 / 15 / 8 mm), so the loss at 0.05 mm does not
  follow size. One seed per cell: a second seed of 30 and 8 mm at 0.05 mm would show if the 3-4 points are real.
- E ablations (nocontact / h10 / h60) at 126-146 M of 300 M, stages 2-3; ETA ~04:00 UTC.
- Report v39: RQ3 grid filled; fixed "all 18 cells" to 16 (4 sizes x 4 clearances down to 0.1 mm).

### 2026-10-22 (23:40 UTC, mason) — D scaled grid 19/20 cells; report v38 (RQ2/RQ3/RQ4 all on corrected backlash)

- D scaled grid (2,048 starts each): down to 0.1 mm all 18 cells 96.3-99.6%. 0.05 mm: 60 mm 99.0%, 30 mm 94.4%
  (74 of 115 failures wedged), 8 mm 94.2% (70 of 119 never found). 15 mm / 0.05 mm still training (GPU 3).
- E noise clearance sweep (2 mm median offset, 1,024 each): 0.2 / 0.1 / 0.07 / 0.05 / 0.04 / 0.02 mm ->
  69 / 61 / 48 / 34 / 25 / 3%; capture 54% already at 0.5 mm offset.
- Report v38: RQ3 fully on the corrected model (old-model tables removed), RQ2 figure/matrix with all 17 policies,
  RQ4 adds the E noise result. Runner now on E_later ablations (nocontact, h10, h60) + D 15/0.05.

### 2026-10-22 (19:30 UTC, mason) — Redo batch: C specialists, D fixed row + diagonal, E noise stalls

All on the corrected backlash model, 2,048 starts per cell, random robot conditions for D.
- **C allmid / allmax / backlash5 retrained** (50.3 M, stop at 99%): own condition 99.1 / 98.6 / 99.6%.
  allmid (latency 1) gets 36.2% on base, allmax (latency 2) 14.4% (old model: 45.4 / 20.7%). backlash5 base
  99.6% (old 97.5%). Same conclusion as before: allmid/allmax are latency-matched specialists; use `randall`.
  `results/C/C-matrix.*` now has all 17 training conditions.
- **D fixed 8 mm row** (median start 3 mm), 1.0 / 0.5 / 0.2 / 0.1 / 0.05 mm: 98.9 / 95.8 / 91.7 / 88.4 / 83.7%
  (scaled: 98.5 / 97.5 / 98.7 / 98.1 / 94.2%). The wide start costs up to 10 points at tight clearance, almost all
  never_found (191 / 165 / 126 at 0.05 / 0.1 / 0.2 mm). 90% training success at 38-44 M steps (scaled 7-9 M);
  0.1 and 0.05 mm never reached 90% in 150 M.
- **D diagonal**, scaled / fixed: o60_c1 99.6 / 99.7%, o30_c0.5 98.2 / 98.1%, o15_c0.2 97.6 / 97.8%. Size alone
  does not lower success; the start area matters only at 8 mm and <= 0.2 mm.
- **D tier 2 scaled so far:** o60 c0.5 / 0.2 / 0.1 -> 98.7 / 98.0 / 98.6%; o30_c1 97.3%. o60_c0.05, o30 c0.2 / 0.1 /
  0.05, o15 c1 running; o15 c0.5 / 0.1 / 0.05 queued.
- **E noise (USB-C, randomized noise/quant/backlash 0-5'/latency 0-2) stalls in curriculum stage 2**: 276 M steps
  at 85-91% training success, rolling 5 M-step success never >= 0.9 (the advance rule); 300 M cap. Final-range test 21.6%
  (curr, no noise: 79.0%); never_found 963, wedged_partial 421, slipped 150. Capture 54% at 0.5 mm (curr 97%);
  clearance sweep 3.0% at 20 um to 69% at 200 um. The peg shrugs the random model off (randall 98.6%); the USB-C
  does not. Consistent with athena5's USB-C backlash and latency results. Options for Yash: fine-tune from
  `E-usbc-proprio-curr-s0` with the noise model (as fullft3 did for backlash), or narrow the ranges (latency 1 only).

### 2026-10-22 (23:40 UTC, mason) — Redo complete: D grid, C allmid/allmax/backlash5, E noise (corrected backlash)

- (A scheduled check-in missed 12:16-23:20 UTC; the runner kept going and finished the approved redo order.)
- D scaled grid, 19 of 20 cells (15 mm / 0.05 mm still training), 2,048 starts each: down to 0.1 mm all 96.3-99.6%.
  0.05 mm: 60 mm 99.0%, 30 mm 94.4% (74 of 115 failures wedged), 8 mm 94.2% (70 of 119 never found).
- D fixed 3 mm start: 60/1.0 99.7, 30/0.5 98.1, 15/0.2 97.8; 8 mm row 98.9 / 95.8 / 91.7 / 88.4 / 83.7% (scaled
  98.5 / 97.5 / 98.7 / 98.1 / 94.2). Extra failures are search failures (191 vs 70 never_found at 0.05 mm).
- C retrains: backlash5 99.6% base, 99.5-99.6% under backlash, 45 / 53% at 20 / 40 ms latency (base policy: 16 / 24%).
  allmid (20 ms fixed) 99.1% own, 36% ideal arm; allmax (40 ms) 98.6% own, 14% ideal. Matrix now 17 x 14.
- E-usbc-proprio-noise-s0 (USB-C, random noise/quant/backlash/latency 0-40 ms from step 0, 300 M): curriculum stuck in
  stage 2 (0.15 mm) from 24 M steps at ~86% train (< 0.9 thr): a plateau, not the window bug. Final stage 21.6%;
  clearance 0.2 / 0.1 / 0.05 / 0.02 mm: 69 / 61 / 34 / 3%. Capture at 0.5 mm offset only 54%. Random errors from the
  start do not work for USB-C; train clean, then fine-tune at the measured latency.
- Runner now on E_later ablations (nocontact, h10, h60) + D 15/0.05; report v38.

### 2026-10-22 (11:15 UTC, mason) — Phase D 8 mm scaled row on the corrected backlash model

- o8 scaled, sphere pads, curriculum, random robot conditions, 2,048 starts each (`results/D/D-grid.csv`,
  `report/static/data/scale.json`): 1.0 / 0.5 / 0.2 / 0.1 / 0.05 mm -> 98.5 / 97.5 / 98.7 / 98.1 / 94.2%
  (old model: 97.0 / 98.6 / 95.6 / 93.0 / 89.7%). Median time 0.12 -> 0.94 s.
- Failures: never_found 29 / 44 / 20 / 27 / 70, wedged_partial 2 / 3 / 4 / 9 / 47 (old 0.05 mm: 115 wedged).
  0 slips in 10,240 starts. The old soft-limit flex (~0.6°) was what made the tight fit wedge.
- 0.1 and 0.05 mm ran to the 150 M cap (train 97% / 93% at end); 0.2 mm also 150 M; 1.0 and 0.5 mm stopped at 99%.
- Report v37: RQ3 leads with the new row (own table + scale figure, scaled only); the diagonal and fixed-start
  table stay under an "old backlash model" note until redone. scale.js now skips a regime with no cells.

### 2026-10-22 (08:10 UTC, mason) — Phase C on the corrected backlash model: backlash does not matter on the peg

- Backlash-only evals of 12 C policies + full evals of the retrained backlash1, backlash3, randall (YAM, lab table,
  2,048 starts per cell; `results/C/C-matrix.csv`, `report/static/data/robust.json`).
- Base policy at 1/3/5 arcmin: 99.9 / 100 / 99.9% (old model: 46.7 / 44.3 / 43.1%). No policy loses more than
  4 points to backlash (clip1 98.9 -> 95.2% at 5'; gain2x 85.2 -> 81.7%). The old drop was the ~0.6° joint flex of the
  soft limit, not backlash.
- Retrained (fixed model): backlash1 / backlash3 / randall train 99.4 / 99.3 / 98.4% (last 20 iters), base test
  98.6 / 99.1 / 98.6% (old: 97.5 / 96.0 / 96.2%). randall: 96.3-99.0% on every noise/encoder/backlash/latency test,
  86.4 / 94.0 / 60.8% at gains x0.5 / x2 / x4 (old x4: 52%). Its cost on the ideal arm is now 1.2 points (was 3.6).
- RQ2 conclusion changes: latency (and gains) need training; backlash up to 5' does not (peg). USB-C still feels
  backlash (athena5, RQ4).
- allmid, allmax, backlash5 retrain later (priority step 4); out of the figure/matrix until then.
- E sideft (side port, +200 M steps from E-usbc-proprio-side-s0): final-stage 47.2% (side from scratch: 35.8%).
  Stuck in curriculum stage 3 (0.08 mm, 1.5 mm) at ~88% training success < 0.9 threshold for 190 M steps: a
  plateau, not the window bug, so not reopened. Failures: never_found 577, slipped 238, wedged_partial 190 of 1,082.

### 2026-10-22 (06:05 UTC, mason) — Backlash-model redo started (Yash OK)

- Runner stopped at 06:00 (5 old-model jobs: D tier-2 o60_c0.05, o30_c0.2/0.1/0.05, o15_c0.5; E noise).
  Merged main (3006b51 fix). `redo_backlash.py --state runs/queue_server/stateD --apply`: 47 runs, 174
  paths to `runs/_softbl/` and `results/*/_softbl/`, 44 done.txt lines dropped. By hand to `_softbl/`:
  the D 10 s-episode test, D grid/plots, D demos, box-pad and no-curriculum D runs.
- Runner restarted 06:04 (tmux `runnerD2`, GPUs 0-4) with `runs/queue_server/priority.txt` first, each
  eval right after its training: (1) C backlash1/3, randall + `C_eval_bl.txt` + E sideft eval;
  (2) E noise; (3) D scaled 8 mm row + 60/30/15 mm diagonal; (4) C allmid/allmax/backlash5, D fixed
  8 mm row + diagonal; (5) D tier-2 scaled. Tier-2 fixed is deferred. All D entries below this one are
  old-model numbers; the report's RQ3 carries a note until the new ones replace them.

### 2026-10-21 (athena5) — USB-C: training with the fixed backlash model buys 3' tolerance, not more

- `E-usbc-proprio-fullft3-s0`: 100M steps from the stage-1 policy (stiff friction, 20 ms) with
  latency {20, 40 ms} + backlash {0, 1, 3'} per world (fixed model) + `--std_max 0.8`. Training
  success 47% at 13M, then flat (52% at 100M); std 0.58 -> 0.67 (under the cap).
- Tests (stiff friction, final stage, 2,048 each), stage-1 policy -> fullft3:

  | latency | 0' | 1' | 3' |
  |---|---|---|---|
  | 20 ms | 67.2 -> 61.8 | 59.0 -> 60.3 | 35.6 -> **55.3** |
  | 40 ms | 64.2 -> 52.0 | 54.7 -> 54.3 | 29.7 -> **46.8** |

  No latency: 12.6 -> 12.1 (both policies are tuned to delay). Mean of the six cells: 51.7 -> 55.1.
- Reading: backlash training gives +17-20 points at 3' and costs 5-12 points without backlash. At
  the likely real condition (<= 1', 20-40 ms) the two policies are equal (59 / 55 vs 60 / 54). So the
  real arm's backlash only needs training if it is near 3'. Measure it first
  (`docs/real2sim/REAL_YAM_LATENCY_BACKLASH.md`). The remaining gap to the no-delay policy (91%) is latency,
  not backlash. Baseline cells: `results/E/robust_fixedbl/latstiffft_*`; fullft3:
  `results/E/robust/fullft3_s0_*`.

### 2026-10-21 (athena5) — BUG: the backlash model was a soft spring, not a dead zone (all earlier backlash results)

- **What was wrong.** `builder.add_backlash` adds a limited hinge in series with each joint, but
  the limit used MuJoCo's default soft constraint (solref 0.02 s, solimp 0.9/0.95). CPU check
  (YAM, 4x4 pads, 5 N on link_6, 3 s): every hinge turns **~0.011 rad (~38 arcmin)** whatever the
  nominal zone: 7,505x the zone at 0.01', 76x at 1', 26x at 3'. TCP moves 14.5 mm against 11.4 mm
  with no hinges. So "backlash 1 / 3 / 5 arcmin" was really ~0.6 deg of series compliance at every
  level. This explains why Phase C's base policy scores almost the same at 1, 3 and 5' (47 / 44 /
  43%) and why even a 0.01' hinge broke the USB-C policies.
- **Fix (default now):** `BACKLASH_LIMIT_SOLREF = (0.002, 1)`, `SOLIMP = (0.99, 0.999, 0.001)`. Hinge
  stays within 1.1-1.2x its zone at 1-3'; 0.01' adds 0.01 mm of TCP motion. `add_backlash(...,
  legacy_soft_limit=True)` reproduces the old model.
- **Corrected USB-C numbers** (final stage, 2,048 each): wide-start policy (no backlash in
  training) at 1 / 3': **90.4 / 77.2%** (old model: 8.6% at 1'; no backlash: 91.4%). Stage-1
  policy (stiff friction, 20 ms, no backlash in training) at 20 ms + 0.01 / 1 / 3': 63.1 / 59.0 /
  35.6% (old model: 0.0%). So realistic backlash (<= 1') costs USB-C little; 3' costs 14-30 points.
- **Affected results (all trained or tested with the old model):** Phase C backlash rows and the
  `allmid`, `allmax`, `randall` policies (mason); every Phase D run (the random robot conditions
  include backlash 0-5'); the queued E `noise` run; tonight's USB-C backlash rows (deployft,
  stiffblft, deploystiffft, fullft, fullft2, the latency x backlash tables). Their numbers describe
  robustness to ~0.6 deg of extra joint compliance, not to arcmin backlash. Which runs to redo is
  Yash's call (GPU cost); the latency, noise, quantization and gain results are not affected.
- Result of `E-usbc-proprio-fullft3-s0` (stage 1 + latency {1, 2} + backlash {0, 1, 3}, fixed
  model): see the entry above.

### 2026-10-21 (athena5) — USB-C full recipe in one step fails; staged instead

- `E-usbc-proprio-deploystiffft-s0` (150M from the stiff+backlash policy; stiff friction +
  backlash {0,1,3} + latency {1,2}): training success <= 4.5%, action std 0.6 -> 8.7 (PPO's entropy
  term widens the noise when no episode succeeds). Tests: 1.7-3.1% in every latency x backlash cell.
- Cause: the starting policy scored 0.1% under latency, so the combined change gave no reward
  signal. The successful fine-tunes changed one factor at a time. Training note: `log_std` is
  unbounded; a clamp (or lower entropy coefficient) would stop the blow-up in no-signal runs.
- Queued, staged: `E-usbc-proprio-latstiffft-s0` (from the 20-40 ms policy, add stiff friction),
  then `E-usbc-proprio-fullft-s0` (add backlash {0,1,3}); tests `scripts/usbc_deploy_eval.sh`.
- **Stage 1 result** (`latstiffft`, 100M, training success 0 -> 61% and rising): stiff friction,
  latency 0 / 20 / 40 ms: 12.6 / **67.2 / 64.2%**; with any backlash 0.0%.
- **Stage 2 failed** (`fullft`, + backlash {0,1,3}): 2.6-6.2% in every cell; training success
  0.3 -> 6%, action std 0.75 -> 3.9. It started at 0.3% even though 1/3 of the worlds are at
  "0" arcmin: those keep the backlash hinges (0.01 arcmin zone), so the hinges themselves, with
  latency, break the policy. Added `train.py --std_max` (caps the action std; off by default).
  Retry queued: `fullft2` from stage 1, backlash {0, 0.5, 1}, `--std_max 0.8`.

### 2026-10-21 (athena5) — USB-C: with stiffer friction, training with backlash works (2.5 -> 76%)

- `E-usbc-proprio-stiffblft-s0`: 100M steps from the stiff-friction fine-tune, stiff friction +
  backlash {0, 1, 3} arcmin per world. Training success 0 -> 70% (still rising at 100M).
- Stiff friction, test backlash 0 / 1 / 3 arcmin: **61.8 / 76.2 / 72.1%** (before: 89.8 / 2.5 /
  1.7%). Soft friction: 40.2% (580 slips); soft + 1 arcmin 41.3%. Stiff + 20 ms latency: 0.1%.
  Latch: soft 13.8%, stiff 20.8%.
- So once the soft-friction grasp creep is removed, USB-C learns backlash as the YAM peg did; the
  earlier failure (deployment recipe, 13%) was the slips. Caveat: training worlds with "0" carry
  the backlash hinges with a 0.01 arcmin zone, while the 0-arcmin test has no hinges, which may
  explain part of the lower no-backlash score.
- Queued: `E-usbc-proprio-deploystiffft-s0` (150M), stiff friction + backlash {0, 1, 3} + latency
  {1, 2}: the full real-arm recipe; tests `scripts/usbc_deploy_eval.sh`.

### 2026-10-21 (athena5) — USB-C adapts to stiffer friction; under it, backlash breaks the search

- `E-usbc-proprio-stiffft-s0`: 100M steps from the wide-start policy with `mj_opt` cone elliptic +
  impratio 10 (grasp creep point ~7 -> ~12 N per mason's CPU test). Training success 87%.
- Final stage (2,048 each), wide-start policy -> stiff fine-tune: soft friction 91.0 -> 60.0%,
  **stiff friction 36.1 -> 89.8%**; stiff + 1 / 3 arcmin backlash 0.3 / 0.2 -> 2.5 / 1.7% (never
  found ~1,650); stiff + 20 ms latency 0.2 -> 0.1%; latch with stiff friction 6.2 -> 21.5%
  (wedged partial 1,183). Soft + 1 arcmin: 4.3 -> 8.3% (slips 718).
- So the soft-friction dependence is removable in 100M steps (a policy for the stiffer, more
  realistic grasp exists). Under stiff friction, backlash failures are search failures, not slips.
  Next (queued): `E-usbc-proprio-stiffblft-s0`, stiff friction + backlash {0, 1, 3} per world, to
  test whether training with backlash works for USB-C once the slips are gone (it does for the
  YAM peg: 98.5-99.6%).

### 2026-10-21 (athena5) — USB-C: training over a 1-step latency range covers both steps

- `E-usbc-proprio-lat12ft-s0`: 100M steps from the matched 20 ms policy with latency drawn from
  {1, 2} steps per world (resumed once after the reboot). Training success 72%.
- Test (2,048 each), latency 0 / 1 / 2 steps: **20.8 / 78.8 / 74.8%** (matched 1-step policy:
  10.9 / 78.3 / 38.1%). Covering the extra step costs nothing at 1 step and gains 37 points at 2.
- So the real latency must be known to within one control step (20 ms), not exactly: train over
  [L, L+1]. Backlash still breaks it (1 / 3 arcmin: 0.3-0.7%, ~1,100 slips of 2,048), which the
  stiff-friction fine-tune (queued) tests.

### 2026-10-21 (athena5) — Resume after a real crash: one bug fixed

- athena5 rebooted at ~13:10 while `E-usbc-proprio-lat12ft-s0` trained (7.9M steps). `--resume`
  first crashed: the kill left a partial last row in `train_log.csv` (`int(None)`). Fixed in
  `train.py` / `train_visual.py`: rows whose `iter` is missing or not a number are dropped. The run
  then resumed at it 25 (6.6M steps, the last checkpoint) with a continuous log. Lost: ~2 h of GPU
  time (the local queue runner does not restart after a reboot; restart it with
  `setsid nohup bash scripts/queue.sh &`).

### 2026-10-21 (athena5) — USB-C: backlash breaks the grasp and the search; the policy leans on soft friction

- **Matched 20 ms policy under backlash** (`scripts/usbc_latency_eval.sh`, 2,048 each): latency
  1 step + 0 / 1 / 3 arcmin: 78.3 / 2.6 / 2.1%; latency 2 + 1 / 3: 1.6 / 1.1%. Mostly slips
  (1,160 of 2,048 at 1 arcmin).
- **Deployment recipe failed in 100M** (`E-usbc-proprio-deployft-s0`: latency fixed 1, backlash
  0-5, q noise, 0/16/14 bit randomized; from the matched policy): training success flat at ~13%
  from 20M on, action std rising 0.55 -> 1.46 (no learning signal: most episodes end in slips).
  Eval at latency 1: backlash 0 / 1 / 3: 10.7 / 17.1 / 17.1%.
- **Diagnostic: stiffer friction removes the slips but not the failure** (matched policy, 1,024
  each; `mj_opt` cone elliptic + impratio 10, which mason found raises the grasp creep point from
  ~7 N to 11.6 N): latency 1 + 1 arcmin: 2.6% (558 slips) -> 0.4% (0 slips, 783 never found).
  Without backlash, the stiff friction alone drops the policy from 78.3% to 20.9%.
- Reading: (1) under backlash the default soft friction lets the drive creep out of the grasp,
  which is a sim artifact (a real 8-17 N rubber-pad grip should hold ~2 mu F >> 7 N); (2) with
  that removed, backlash still breaks the search (never found), as on the YAM peg; (3) the USB-C
  policies depend on MuJoCo's soft friction: a sim-to-real risk. Next (queued): fine-tune the
  wide-start policy under stiff friction (`E-usbc-proprio-stiffft-s0`), then test backlash and the
  latch under it (`scripts/usbc_friction_eval.sh`).

### 2026-10-21 (athena5) — USB-C trained at a matched 20 ms latency recovers to 79%

- `E-usbc-proprio-lat1ft-s0`: 100M steps from the wide-start fine-tune (91.4% clean) with a fixed
  latency of 1 control step, lab table, final stage (3 mm median). Training success 75%, rising.
- Test at latency 0 / 1 / 2 steps (2,048 each): **11.1 / 78.9 / 38.8%**. The wide-start policy
  (no latency in training) at 1 / 2 steps: 0.7 / 0.6% (never found, slipped, left fixture).
- Same pattern as the YAM peg (Phase C): a matched latency recovers most of the clean success, but
  the policy then fails without the latency. Combined with the random-error fine-tune (22% at
  20 ms, 31% clean), the practical route for USB-C is to measure the real latency and train at it,
  not to cover 0-40 ms. `results/E/robust/lat1ft_s0_lat*.parquet`, `wide_s0_lat*.parquet`.

### 2026-10-21 (athena5) — USB-C E5 latch: a 100M fine-tune pushes through it (28 -> 90%)

- `E-usbc-proprio-latchft-s0`: 100M steps from E1 on `phase_e_usbc_latch.json` (lab table, final
  stage, 3 mm median start, latch 5-15 N over the last 1.5 mm). Training success 88% at the end.
- On the latch port (2,048 starts): **89.7%** (E1 zero-shot: 28.0%), median 1.96 s; failures
  slipped 84, never found 55, left fixture 32. q noise 5e-4: 90.5%, 14-bit: 90.8%. Latency 1 / 2:
  3.0 / 1.0%, backlash 1 / 3: 2.3 / 2.2% (no robot errors in this fine-tune).
- **Reconciled with mason's 7.2 N limit (below):** at a fixed peak the fine-tune seats 94.3% (5 N)
  and **78.7% (15 N)**, above the static limit. It strikes: in 8 logged episodes the plug enters
  the last 1.6 mm at 40-80 mm/s (`results/E/robust/latchft_s0_peak15N.rollouts.npz`). Energy check:
  past the 0.72 mm where 7.2 N balances a 10 N/mm spring, the plug needs ~1.7 mJ to reach 1.3 mm,
  i.e. ~0.9 kg effective mass at 60 mm/s, plausible for the YAM wrist + gripper. So the static
  limit is real but a dynamic insertion beats it. Caveats: the latch model is a pure spring with
  no detent, and success counts the moment the plug reaches the seat (a real latch would click
  and hold); the strike speed is near the 50 mm/s speed cap; slips rise at 15 N (238 of 2,048),
  matching the grasp creep limit mason found. Starting from E1 (28% zero-shot) gave the dense
  reward that the from-scratch run never saw.

### 2026-10-21 (athena5) — USB-C: fine-tuning on wider starts grows the capture radius to 7.1 mm

- `E-usbc-proprio-wide-s0`: 100M steps from E1 (old mount) on the lab table, final-stage parts
  (20-100 um), start offsets uniform +-6.5 mm per axis (median 5.0 mm, was 3.0 mm), `--init`.
- E2 (8,192 starts, 50 um, +-9 mm; `scripts/usbc_capture.py --halfwidth 9e-3`): r50 **5.56 mm**
  (95% CI 5.45-5.66) before -> **7.13 mm** (7.05-7.22) after. 4-4.5 mm bin 70 -> 87%, 6-6.5 mm
  59 vs 37%. E3 at 2 mm median: 20 um 82.3 -> 91.7%, 50 um 90.0 -> 96.6%, 100 um 91.0 -> 97.0%.
- Final-stage test (3 mm median, 2,048): 77.9 -> **91.4%**, median 2.14 -> 1.90 s; failures now
  never found 105, left fixture 56.
- Caveat: the gain mixes two effects, wider starts and the move to the lab-table mount (E1 trained
  on the old mount). The server's E1 trains on the table from scratch; compare its r50 with 7.1 mm.
### 2026-10-22 (04:30 UTC, mason) — Latch fine-tune at reachable forces pushes but slips; 60 mm ignores clearance

- **E-usbc-proprio-latchlowft-s0** (lab-table curr policy + 150M steps on a 2-7 N latch, which the
  7.2 N push limit can beat): 90.7% on its own 2-7 N latch (left fixture 85, never found 70).
  On the 5-15 N latch: 57.5% (lab-table curr policy zero-shot: 26.5%); slipped 479 of 870 failures,
  wedged 155, jammed 94. athena5's 5-15 N fine-tune reached 89.7% by striking at 40-80 mm/s. So a
  policy that learns only to push loses the drive at peaks it cannot push through; the strike
  (momentum) is needed for strong latches. `results/E/eval_latchlowft_s0{,_on_latch}_final.parquet`.
- **D o60 row (scaled):** c1 98.2, c0.5 96.7, c0.2 97.9, c0.1 98.6%. The 60 mm peg does not lose
  success down to 0.1 mm; the 8 mm peg loses 4-5 points by 0.1 mm. Tight clearance hurts small parts.

### 2026-10-21 (23:45 UTC, mason) — Phase D fixed regime: a 3 mm start gives the same success, a longer search

Fixed regime (median start 3 mm at every size) against scaled, test success on 2,048 starts:
o60_c1 99.7 / 98.2, o30_c0.5 98.4 / 96.9, o15_c0.2 97.3 / 97.6, o8_c1 96.9 / 97.0, o8_c0.5 96.9 / 98.6.
At 8 mm and 1.0 mm the fixed start (2.3x the scaled 1.3 mm) takes 0.44 s and 22 mm of travel against
0.12 s and 3.9 mm, with the same failures (never found 55 vs 55). So at >= 0.5 mm clearance the
search area is not the limit. Tighter fixed 8 mm cells (00:45 UTC): c0.2 95.0% (scaled 95.6%; 0.96 s,
35 mm), c0.1 89.1% (scaled 93.0%; never found 147 vs 64, wedged 68 vs 80). So at 0.1 mm the larger
start area costs 4 points, all in search failures. c0.05 fixed: 81.9% (scaled 89.7%; never found 199
vs 88, wedged 162 vs 115, 1.52 s, 36 mm). First tier-2 cell o60_c0.5_scaled: 96.7% (wedged 39, never
found 25; o60_c1 98.2%). o60_c0.2_scaled: 97.9% (wedged 26, never found 10). At 60 mm the clearance
barely matters down to 0.2 mm; at 8 mm it costs 3 points by 0.2 mm. Once the D_train_tier1 queue emptied, the runner
ran these evals itself; E latchlowft and sideft started at 23:16.

### 2026-10-21 (21:15 UTC, mason) — Phase D diagonal and 8 mm row: clearance, not size, lowers success

Scaled regime, 2,048 unseen starts, random robot conditions. Diagonal: o60_c1 98.2%, o30_c0.5 96.9%,
o15_c0.2 97.6% (0.68 s, 19.6 mm travel; never found 20, wedged 27). Fixed regime o60_c1: 99.7%
(0.46 s, 13 mm). 8 mm row (sphere pads, median start 1.3 mm):

| Clearance | Success (95% CI) | Median time | Never found | Wedged partial |
|---|---|---|---|---|
| 1.0 mm | 97.0% (96.1-97.6) | 0.12 s | 55 | 3 |
| 0.5 mm | 98.6% (98.0-99.1) | 0.26 s | 26 | 2 |
| 0.2 mm | 95.6% (94.6-96.4) | 0.54 s | 62 | 26 |
| 0.1 mm | 93.0% (91.8-94.0) | 0.84 s | 64 | 80 |
| 0.05 mm | 89.7% (88.3-90.9) | 1.28 s | 88 | 115 |

o8_c0.05 with 10 s episodes (`results/D/extra_o8_c0.05_ep10_s0.parquet`): 92.6% against 89.7% at 5 s; wedged 115 -> 64, 43 seats after 5 s. About half of the wedged pegs are slow, not stuck.
At fixed size, tighter clearance makes the seat slower and turns failures from search (never
found) into alignment (wedged partial). This is the trend the USB-C runs (0.02-0.1 mm) continue.
o8_c0.05 (USB-C clearance) reached its final stage by the step cap and 92% training success. Fixed regime o30_c0.5: 98.4% (0.52 s, 20.9 mm). The 12 tier-2 scaled cells are queued in E_later_train.txt after the latch/side fine-tunes and the random-error USB-C run, before the E ablations. Demo grids D_o60_c1 / D_o8_c1 / D_o8_c0.1 are in the report's
example figure (8 mm uses a close camera, DEMO_CAM in make_demos.sh). build_demos_data.py now keeps
items rendered on the other host instead of dropping them.

### 2026-10-21 (evening, mason) — First Phase D tests: 97-98% at 1.0 mm from 60 mm down to 8 mm

Scaled regime, 2,048 unseen starts with random robot conditions (noise, quantization, backlash, latency):

| Cell | Test success (95% CI) | Median time | Median travel | Main failure |
|---|---|---|---|---|
| o60_c1 | 98.2% (97.6-98.7) | 0.72 s | 35.1 mm | never found 21 of 36 |
| o30_c0.5 | 96.9% (96.0-97.5) | 0.54 s | 28.6 mm | never found 25, wedged 17 of 64 |
| o8_c1 (sphere pads) | 97.0% (96.1-97.6) | 0.12 s | 3.9 mm | never found 55 of 62 |

8 mm training success at the target clearance (no tests yet): c0.5 98%, c0.2 92%, c0.1 86%. The
evals ran beside the training jobs on GPU 1 (marked done in stateD/done.txt first, so the runner
skips them). o15_c0.2 stopped at the 0.5 mm stage (old stop_success bug) and is queued to resume.

### 2026-10-21 (afternoon, mason) — Two from-scratch runs never found the goal; curricula fix queued

- **D o15_c0.2_scaled (15 mm opening, 0.2 mm clearance, 2.5 mm median start offset):** 0 successes
  and 0 captures in 150M steps. Capture was 0.2-0.4% in the first 1M steps (as in o30_c0.5, which
  then learned), then fell to 0; the value loss went to ~1e-8. The reward is sparse (capture +
  seat), so PPO lost the few lucky episodes and stopped exploring. o60_c1 (99.1% at 50M, 90% at
  24M steps) and o30_c0.5 (99.0% at 140M, 90% at 37M) learned from scratch.
  **Fix:** `train.py --curriculum scale` trains on looser clearances first (1.0, 0.5, 0.2, 0.1 mm
  down to the target; same start offset; advance at 90% training success). All D cells tighter
  than 1 mm use it from now on, except o30_c0.5_scaled-s0 (already done). The failed run is kept
  as `runs/D-peg-proprio-o15_c0.2_scaled-s0_nocurriculum`. So the steps-to-90% numbers of
  curriculum cells are not comparable with from-scratch cells.
- **D small cells with box pads: the peg slips in the grasp.** With the clearance curriculum all
  8 mm cells and the 15 mm rerun stayed at stage 1 (1 mm clearance) at 53-70% training success.
  A 1,024-start test of o8_c1_scaled at 143M steps: 73.7%; failures slipped 159, left fixture 85,
  never found 24. So at 8 mm the limit was the box-pad grasp (as for USB-C), not the search. Cells
  <= 15 mm now use the USB-C sphere pads (`pad_spheres` 4, nconmax 256, njmax 2048) under new
  names `*_sph`; the box-pad runs keep the old names and are excluded from `phase_d_report.py`.
  The scale curriculum also advances a stage after 0.6 x steps / (stages - 1) at the latest.
  With sphere pads (nconmax 128, njmax 1024; 256/2048 ran out of memory on a 32 GB V100 at 12,288
  worlds) the 8 mm cells train to 93-99% and the 15 mm cell to 97.8% (18:25 UTC).
- **Curriculum bug (fixed in 4ba3986):** at 12,288 worlds one PPO iteration is 393k env steps. The
  5M-step success window then spans at most 12 iterations (4.72M), below the 4.75M (95%) the
  advance test required. So D stages advanced only on the step cap (30-45M steps), never on 90%
  success, although stage-1 success was 97-99%. (The USB-C runs, 262k steps per iteration, fill
  the window and are not affected.) Second bug: `--stop_success 0.99` also fired at a non-final
  stage, so o8_c0.5_scaled_sph stopped at 1 mm clearance. Now stop_success needs the final stage,
  and a run that stopped early resumes. The runs in flight keep the old code; their final-stage
  numbers are valid but their steps-to-stage numbers are set by the cap.
- **E5 latch from scratch (E-usbc-proprio-latch-s0):** 0.0% training success in 300M steps,
  never past curriculum stage 1 (0.3 mm clearance). The final-stage eval is 0% (never found 1,576,
  wedged partial 265, slipped 139 of 2,048). Without the latch, stage 1 reaches 90% in 16M steps.
  The latch spring (5-15 N over the last 1.5 mm) blocks every seat, so the sparse seat reward never
  appears. **Why (CPU MuJoCo, one plug, 4x4 sphere pads):** with action_scale 0.01 rad and clip 3,
  the YAM can press only **7.2 N** along the port axis (stalled torque kp x 0.03 rad through J^T;
  7.27 N computed, 7.2 N measured against the port floor). Every failed latch push stopped where
  peak x e/travel = 7.24 N. The plug must pass 1.3 of the 1.5 mm, so only peaks <= 8.35 N can be
  beaten: 33% of U[5, 15], matching the 28% zero-shot. The grasp is a second limit: under a slowly
  rising axial force the plug creeps 1 mm at 6.4-7.8 N and tilts > 2 deg at ~8 N, almost
  independent of grip force (8.5 -> 17 N per pad moves it < 0.5 N; MuJoCo soft friction:
  impratio 10 raises the 1 mm point to 11.6 N, noslip_iterations 10 to 12.9 N). Raising
  grip_force_base does not help. **Fix queued:** `E-usbc-proprio-latchlowft-s0` fine-tunes the
  finished no-latch policy (`--init runs/E-usbc-proprio-curr-s0/last.pt`) with a 2-7 N latch
  (`configs/phase_e_usbc_latch_low.json`). A 5-15 N latch needs action_scale ~0.02 (14.5 N push)
  plus a stiffer grasp; open question for the real YAM: measure its maximum axial push and a real
  USB-C port's insertion force. Scripts: scratchpad only (latch_forces.py, creep.py, fix.py).
- **USB-C from scratch on the table mount (300M cap, no noise model), evals:**

  | Run | Final stage (2,048) | Clearance 20 / 50 / 100 um (2 mm offset) | Median offset at 50% |
  |---|---|---|---|
  | curr (port up) | 79.0% (left fixture 234, never found 153) | 87.4 / 94.0 / 93.8% | ~4.9 mm |
  | chamfer 0.2-0.5 mm | 96.2% (never found 52) | 97.5 / 98.5 / 98.6% | ~7.0 mm |
  | side port | 35.8% (never found 879, slipped 196) | 26.9% at 20 um | — |

  curr from scratch on the table (79.0%) is no better than the athena5 old-mount policy tested on
  the table (77.9%) and below athena5's table fine-tune on wider starts (91.4%). The side port reached only
  curriculum stage 3 in 300M steps.

### 2026-10-21 (athena5) — Phase B with three seeds

- Blind s0-s2: 99.90 / 99.66 / 99.76% in distribution (6,144 starts, 99.77%), median 0.64 s,
  travel 25.8 mm; 90% training success at 39.3 / 40.5 / 38.1M steps. Visual s0-s2: 99.95 /
  99.90 / 100%. No contact loss s0 / s2: 99.17 / 99.02%, median 0.78 / 0.80 s.
- OOD (3 seeds x 1,024 starts, blind / visual): tighter 97.9 / 87.3, slender 97.8 / 82.7, very
  slender 94.6 / 64.5, mild size 96.7 / 88.4, extreme size 86.3 / 30.2, random convex 22.0 / 2.6%.
  Blind beats visual in all 18 seed-family pairs (B-R6 reproduced with 3 seeds).
- Probe (mean recall over the 5 occurring modes): blind 69.9 / 67.6 / 70.1% (current obs
  48-53%); no contact loss 64.9 / 62.7%. Contact supervision makes the latent more decodable.
- Files: `results/B/summary.json`, `B-R2_table5.csv`, `B-R6_ood.png`, `probe_*.json`.

### 2026-10-21 — Gate C: one randomized policy covers noise, backlash and latency

`C-peg-proprio-randall-s0` (per-world `noise_rand`: q_noise {0,1e-4,5e-4}, quant {0,16,14} bit,
backlash {0,1,3,5}', latency {0,1,2} steps; 66.8 min on a V100, stop rule) on the 13 test
conditions, 2,048 starts each: base 96.2%, noise/quant 96.3-97.0%, backlash 1/3/5' 98.1/98.0/97.1%,
latency 1/2 steps 96.3/96.8%, gains 0.5x/2x/4x 73.6/91.1/52.1%. Every fixed-level policy with
latency fails off its own latency (C matrix), so randomization costs only 3.6 points on the ideal
arm and removes the need to know the latency exactly. Decision: `NOISE = {"noise_rand": NOISE_RAND}`
in `make_manifests.py` for D and the E "noise" ablation (the separate E "noiserand" line is gone).
Contrast with the USB-C random-error *fine-tune* (athena5, entry below): 31% clean after 100M from a
noise-free policy. Training with the randomization from scratch is the better path; E-noise tests it.
D runs are capped at 150M steps (most C runs stopped at 50-57M).

`C-peg-proprio-clip1-s0` (action_clip 1.0, else base; tests "is smoothness the mechanism?"): own
98.9%, noise/quant 98.4-98.8%, backlash 1/3/5' 40.6/35.4/31.5%, latency 1/2 42.1/47.8%, gains
0.5x/2x/4x 65.4/71.5/46.1%. Small actions alone help latency (16 -> 42%) but not backlash (worse
than base). So the low-gain behaviour of perturbation-trained policies is not sufficient: the policy
must see the perturbation in training.

Example clips (`scripts/yam_example_rollouts.sh`, `scripts/yam_example_videos.py`): matched starts,
64 logged worlds. Base fails / own-trained succeeds on 50/64 (latency 20 ms), 38/64 (backlash 3'),
54/64 (gain x4) starts. The backlash replay needs the CPU model built with the same
`backlash_arcmin`, since the logged qpos includes the 6 hidden hinges.

### 2026-10-21 (night, athena5) — USB-C under robot errors; wrist-camera view; demos

- **The USB-C policy breaks under the robot errors that Phase C found** (athena5 E1 s0, sphere
  pads, lab-table mount, final-stage distribution, 2,048 matched starts each;
  `scripts/usbc_brittleness.sh`, `results/E/robust/`). Fine-tune `E-usbc-proprio-noiserandft-s0`:
  100M steps from E1 with latency 0-2, backlash 0-5 arcmin, 0/16/14 bit and q noise 0-5e-4 drawn
  per world at each reset.

  | Test condition | E1 (no errors in training) | Fine-tuned, random errors |
  |---|---|---|
  | none | 78.0% | 31.1% |
  | latency 1 step (20 ms) | 5.2% | 22.6% |
  | latency 2 steps | 5.0% | 22.0% |
  | backlash 1 arcmin | 8.6% | 45.4% |
  | backlash 3 arcmin | 7.2% | 44.6% |
  | q noise 5e-4 rad | 77.3% | 32.0% |
  | 14-bit encoders | 78.4% | 29.6% |

  E1's failures under latency/backlash are mostly "never found" plus slips. Noise and quantization
  cost nothing (as in Phase C). The fine-tune is 4-6x more robust but loses more than half of the
  clean success; its training success was 40% at 100M and still rising (not converged). As in
  Phase C, covering every latency with one policy is costly, and at USB-C clearances very costly:
  measure the real latency and train matched (or over a narrow measured range).
- **The YAM wrist D405 does not see the port at the start states** (`results/F/camera_views.png`):
  the camera sits behind the drive and looks along it, so the drive body hides the opening. The
  housing shows behind the drive at ~9 cm; the gripper is closer than the 7 cm minimum range and
  reads invalid. Phase F policies can use the housing edge and the drive's motion, not the port,
  unless the mount changes (plan §12.1 asks to report the mount choice).
- **Demonstrations.** `eval.py --demo_k K` records every world on the GPU and keeps K successes and
  K episodes of each failure type; `viz/demos.py` renders a captioned grid video;
  `scripts/make_demos.sh` builds 16 cases (`results/demos/`, report figure `demos`). A demo run is a
  new run of the test, so its rate can differ slightly (e.g. blind in-dist 99.7% vs 99.9%).

### 2026-10-21 — Phase C on the lab-table YAM: noise and quantization are harmless; latency, backlash and gains are not

15 seed-0 runs (12,288 worlds, `configs/phase_c_yam.json`, table mount, box pads) on mason. Every
run cross-evaluated on 13 test conditions, 2,048 unseen starts per cell (`scripts/phase_c_report.py`
-> `results/C/C-matrix.{csv,png}`, `C-summary.json`).

- **Training.** All 15 reach >= 99% training success; test success under the own condition is 98.3-100% (allmid 99.0%, allmax 98.5%). 90% at 14.2M (base) to 24.8M (allmax) steps;
  backlash and latency slow learning a little (16-21M). The 99% stop rule works: 13 runs stop at
  the 50M minimum (~46 min each on a V100), allmax at 56.6M. Base and qnoise1e-4 ran the full 300M.
- **Joint noise (1e-4, 5e-4 rad) and encoder quantization (14, 16 bit) cost nothing**: every
  policy keeps its own level under them (base: 99.8-100%).
- **Backlash breaks a policy that never felt it**: base 47 / 44 / 43% at 1 / 3 / 5 arcmin
  (failures: never found 479, wedged 405 of 2,048 at 1 arcmin). Training with any backlash level
  covers all three (98.5-99.6%) and costs only 2-3 points without backlash.
- **Latency is the worst factor, and it does not transfer in either direction.** Base: 16% at 1
  step, 24% at 2 steps (never found, left fixture, wedged). A latency-1 policy gets 99% at latency
  1 but 33% without latency and 69% at latency 2; latency-2 gets 99% / 26% / 45%. allmid (latency 1)
  and allmax (latency 2) behave the same way: they are latency-matched policies (allmax 20.7% on
  base). So a fixed latency must match the robot to within one 20 ms step.
- **Gain mismatch**: base at 0.5x / 2x / 4x gains: 61 / 80 / 20% (4x: 1,556 slips). Gain-trained
  policies are 99% at their own gain; 2x and 4x transfer to each other (89%, 94%), but no gain
  policy covers the whole range (gain0.5x: 55% at 1x).
- **OOD (base s0)**: tighter 99.6, slender 99.0, very slender 97.6, mild size 99.7, extreme size
  97.0, random convex 0.5% (1,024 each). The YAM generalizes better than the Flexiv blind policy
  on every geometric bin except random convex (Flexiv 22.8%; failures there are mostly slips).
- **Implication for Gate C**: q noise is not the decision. D/E need latency and backlash in
  training, either matched to the measured robot or randomized per world; `randall` (running)
  answers whether one randomized policy covers the whole matrix. A single fixed setting such as
  allmid is the wrong choice: it fails at 45% on a robot without latency.

### 2026-10-21 — Phase B server runs (mason)

- Visual s1 / s2 (50M, stop rule): 99.9 / 100% in distribution, 90% at 6.9M steps both.
  OOD tighter 87.7 / 89.3, slender 84.2 / 79.6, very slender 67.6 / 62.1, mild 88.6 / 88.1,
  extreme 29.2 / 29.9, random convex 2.3 / 2.3%: seeds agree with s0 (blind still wins every bin).
- No-contact s2 (300M): 90% at 88.9M (blind ~40M), 98.8% at 300M, eval 99.0%; OOD tighter 92.3,
  slender 95.4, very slender 93.4, mild 94.1, extreme 82.7, random convex 15.9%.

### 2026-10-21 — Server unblock: resume, Phase E variants (E4-E6), Phase F code, USB-C validity checks

- **Merged `mason-runs` into `main`** (YAM lab table, stop rule, idle-GPU runner), then built the
  missing code on top. The server can now run all of Phases B-F from `main`
  (`docs/operations/SERVER.md` §9 lists what changed; `make_manifests.py --check`: 160 commands, 0 errors).
- **Resume.** `--resume` restores model, Adam, RNG, curriculum stage and log; tested by stopping a
  USB-C curriculum run at 57k steps and continuing to 106k (log rows continuous, no duplicates).
  Checkpoints are written atomically; `runs/<name>/DONE` marks a finished run. The runners append
  `--resume` for partial runs instead of discarding them.
- **Per-world noise randomization** (`EnvCfg.noise_rand`, plan §9.3). Verified: per-world levels,
  per-world backlash `jnt_range`, per-world latency queue. New C line `C-peg-proprio-randall-s0`.
- **Bug found: backlash + sphere pads did not compile** ("more than 6 dofs in body link_1"):
  `sphere_pads` compiles the spec, and joints added afterwards land on the wrong body. Backlash is
  now added first. No finished run was affected (Phase C used box pads; E used no backlash), but
  any USB-C run with backlash in the noise model would have crashed at start.
- **E4 chamfer** (`sample_usbc(chamfer=...)`): the plate wedges' inner top edge is cut at 45 deg;
  neighbouring wedges share the new vertices, so the lead-in has no gaps
  (`results/E/variants/views.png`). Drawing the chamfer last keeps the same parts as the no-chamfer
  pool for the same seed.
- **E5 latch** (`EnvCfg.latch`): a spring along the port axis over the last 1.5 mm, peak 5-15 N per
  reset, plus 5 N s/m damping, launched inside the 20-substep CUDA graph before each step. Force
  read back from `xfrc_applied` matches the law to 4 uN with no lateral component.
- **E6 side port** (`EnvCfg.fix_base_rpy_deg = (0, -90, 0)`, port facing the robot). YAM IK reach
  map (256 starts per cell): housing lifted <= 6 cm above the table fails 3-97% of starts; lifted
  10-15 cm at x = 0.30-0.45 m reaches 100%. The config uses that range (a stand under the housing,
  not modelled; the fixture is fixed in space).
- **Phase F code.** Depth and RGB-D frames (`cam_modality`), a D405-like depth model (7 cm minimum
  range, noise ~ d^2, random and edge dropout), camera/light/colour randomization (`cam_rand`), and
  fused policies (`--proprio`: joint TCN + image TCN, latents concatenated). The YAM uses
  Menagerie's wrist D405 (`wrist_cam`). From that mount the scene spans 7-18 cm, **so the plug tip is
  at the edge of the D405's minimum range**, and the port is only a few pixels wide at 64 x 64
  (plan §12.1 asks to report this; the 128 x 128 ablation matters here). Memory: 10.3 GB at 1,024
  worlds (RGB-D + proprio), after cutting the rollout re-embedding chunk from 1,024 to 256 worlds
  (the 1,024 chunk needed a 3.75 GiB activation and ran out of memory). Two bugs fixed while
  testing: `"depth" in "rgbd"` is False (depth was never rendered for RGB-D), and
  `train_visual.py` reshaped frames to 3 channels.
- **§11.5 validity checks, USB-C on the YAM (CPU, sphere pads, 50 um).**
  A-S5: the contact signal is strong: F_min = 0.13 N vertical and 0.036 N lateral at
  sigma_q = 1e-4 (0.46 / 0.17 N at 5e-4), against 0.79 / 1.13 N for the YAM peg; the light 6 N
  grip and the soft USB-C gains make the arm deflect more per newton.
  A-S6 (scripted slide into the port): with the default 0.5 N press the plug never drops, at 50 um
  or 300 um: through the soft gains 0.5 N lowers the target only ~0.5 mm, and the plug stops 0.34 mm
  down (the YAM peg's A-S6 also never captured). With a 2 N press at 300 um the drop-in shows as
  a joint-velocity spike at 3.7 s: **peak-to-noise 5.0 at sigma_q = 1e-4 and 1.0 at 5e-4** (Flexiv
  peg: 3.2 at 1e-4). So the drop-in event is visible at 1e-4 and buried at 5e-4. At 50 um the
  scripted slide still does not drop (a constant-orientation slide must align within ~0.4 deg);
  the trained policy does (91.5%), so this is a limit of the script, not of the signal.
  Two check fixes: `CpuScene.place` gripped at 0.75 x plug length (the YAM fingertips then hit the
  housing before the plug); it now uses the variant's grasp height, as the GPU env does. The A-S6
  signal window is anchored at the drop onset (plug 0.2 mm down), not at the 1 mm capture time.
- **E1 zero-shot on the new assemblies** (athena5 sphere-pad E1 s0, trained on the old mount;
  2,048 matched final-stage starts):

  | Assembly | Success | Main failures (of 2,048) |
  |---|---|---|
  | Port up, lab-table mount (reference) | 77.9% | never found 363, left fixture 55 |
  | E4 chamfer 0.2-0.5 mm | 83.6% | never found 263 |
  | E5 latch 5-15 N | 28.0% | slipped 466, wedged partial 430, never found 246 |
  | E6 port facing the robot | 0.0% | left fixture 2,045 (median 0.4 s) |

  The reference drops from 91.5% (old mount, fixture 12-25 cm below the base) to 77.9% on the
  table mount: the fixture pose shift alone costs 14 points, so the server's E1 retrains on the
  table. The chamfer helps (+5.7 points, fewer "never found"). The latch defeats a policy that never
  felt it: it slips the drive in the grasp or wedges. **If E5 training stalls on slips, the 6 N grip
  base may be too weak for a 15 N latch**: raise `grip_force_base` rather than lower the latch.
  The side port fails at once: the port-up policy drives the plug ~14 mm along the housing in
  0.25 s (joint-space actions do not transfer to the rotated arm pose); zero action holds still, so
  the env is sound. E6 needs its own training (as the plan expects). Script:
  `scripts/usbc_variants_zeroshot.sh`; data `results/E/variants/`.

### 2026-10-20 — YAM on the lab workcell table (mason)

- **The YAM scene now matches the lab mount** (Yash's instruction): RoboSimStudio's
  `yam_i2rt` workcell, without its background. The arm base is level with the table top, at
  0.3475 m behind the table centre and 0.31 m to its right (the task arm of the bimanual
  mount; we use it for one arm too). In the arm frame the table centre is (0.3475, 0.31),
  half size 0.2975 x 0.65 m, height 0.75 m (`builder.YAM_TABLE`). The Menagerie and i2rt YAM
  base frames are the same (arm along +x, z up; joint origins agree to ~8 mm), so only the
  mount height changed.
- **Before, the YAM was effectively mounted 10–30 cm above the work**: Phase C fixture tops were
  at z −0.30 to −0.10 m in the base frame. Now `EnvCfg.table_top = 0` puts the table top
  level with the base, and the (tilted) fixture rests on the table at its lowest corner
  (`fix_z` = extra lift, now 0). Fixture tops are then at z +19 to +45 mm. x 0.25–0.45 and
  y ±0.2 m stay. The table collides like the fixture (peg, pads, fingertips). Also applied to
  `phase_e_usbc_up.json` and `phase_e_usbc_spheres.json`.
- Checks (GPU 1, 2,048 starts): 0 IK failures; lowest fixture corner at z = 0 in every
  world; all arm links stay ≥ 63 mm above the table top. IK failure at table height is
  0–0.5% for x ≤ 0.45 m (3% at x 0.40–0.50, top 50 mm). Image:
  `results/C/yam_table_reset.png`.
- Smoke training (12,288 worlds, 25M steps, `_smoke-C-yam-table`): 90% training success at
  18.1M steps, 97.4% at 25.2M, 19.2k steps/s on a V100. The YAM learns faster than the
  Flexiv (90% at ~40M).
- **Runs trained before this change use the old (floating) YAM setup**: athena5's
  C-peg-proprio-base-s0 and E1 s0. They are not comparable with the new C/D/E runs.

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

- **USB-C with sphere pads (E-usbc-proprio-currsph-s0) completes the curriculum.** Stage 2 at
  20.2M, 3 at 96.5M, 4 at 164.4M, 5 (final, 20-100 um, 3 mm median offset) at 200.3M steps;
  final training success 89% (plateau over the last 25M). Deterministic eval, 2,048 matched starts
  per stage (results/E/stages/): stage 1 81.7%, 2 99.0%, 3 98.9%, 4 98.1% (Gate E: >= 80%, pass),
  5 91.5%.
  - Final-stage failures are search failures: 105/175 never find the port (all 8 s timeouts);
    success by start distance 98-99% (<= 2 mm), 96% (2-3), 92% (3-4), 82% (4-5), 55% (5-6);
    clearance quartiles 90-93%. A 16 s limit gives 91.9% (never_found 46, but left_fixture 92).
  - E2 capture radius (50 um, 8,192 starts in a +-6 mm square, results/E/capture/): r50 =
    5.98 mm (bootstrap 95% CI 5.91-6.04); 98-100% up to 3 mm, 28% at 7-8 mm.
  - E3 clearance sweep (2 mm median offset, 2,048 each): 20 um 93.9%, 40 um 97.0%, 50 um 98.0%,
    70 um 98.4%, 100 um 98.5%, 200 um 95.7%. Validity check (50 um -10/+20) passes: smooth.
  - Stage 1 drop (81.7%): with 0.3 mm play the drive tilts inside the port (jammed_inside, median
    tilt 3.9 deg > 2 deg success limit); the tight-fit policy no longer straightens it.
  - Server E manifests pointed at the box-pad config; `make_manifests.py` on main now uses
    `phase_e_usbc_spheres.json` (note added to docs/archive/SERVER_AGENT_INSTRUCTIONS_2026-10.md for the mason agent).

- **Blind gear policy (G-gear-proprio-base-s0) reproduces the paper's gear row.** 300M steps in
  3.2 h; training success 0.1% at 12M, 99.5% at 75M, 100% at 299M. Deterministic eval: 99.94% of
  8,192 unseen gear sets (median 0.90 s; paper 0.58 s), OOD bins 99.71% (tighter), 100% (taller,
  much taller, mild size, random tooth), 99.90% (extreme size); paper >99.9% for all.

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
| Seeds and run length, C/D/E (2026-10-20) | 3 seeds, 300M steps | Seed 0 only; blind runs stop at training success >= 99% after >= 50M steps (`--stop_success 0.99 --min_steps 50e6`). C base s0 and C qnoise1e-4 s0 were already running and keep 300M | Yash: initial results first; ~815 -> ~160 GPU-h (3090 Ti-eq.). Add seeds 1-2 where a result is close or surprising |
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
- Branch: `mason-runs` (renamed from `jensen-runs` on 2026-10-20; the host is mason). Bootstrap completed in tmux `peg-setup`, log
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
- Validation crossed **90% training success at 37.36M steps** (iteration 95,
  success 90.47%), within the required 35–50M window and comparable to local seeds
  (39.7M, 41.7M). Early learning gate passed; >=99% from 150M and completion
  of the 300M run are still pending. Evidence:
  `results/server_setup/mason/validation_early_gate.json`.
- **Validation passed both gates** and finished 300M steps (exit 0, 4.08 h): min training
  success 99.21% from 150M (mean 99.64%), final 99.79%. Logs:
  `results/B/train_logs/B-peg-proprio-base-s3/`; summary
  `results/server_setup/mason/validation_final.json`.
- **Production batch started 2026-10-20 22:06 UTC on GPUs 0,1** (Yash's OK: 2 GPUs), tmux
  `jobs`: B_train, B_eval, C_train, C_eval with `--require-idle-gpus --skip-tag
  also_in_local_queue --skip-tag needs_local_ckpt`. First jobs: B-peg-rgb-base-s1 (28.6 GB,
  ~4.0k steps/s) and B-peg-proprio-nocontact-s2 (10.7 GB, ~21.4k steps/s), as benchmarked.
  C jobs use the new YAM table scene.
- **2026-10-20 22:17 UTC: batch widened to GPUs 0–4** (Yash's OK). The runner was restarted
  and adopted the two running jobs. **C-peg-proprio-base-s0 now trains on mason** on the
  table setup (removed from `LOCAL_DONE` in `make_manifests.py`: athena5's run used the old
  mount). Yash's plan: athena5's old-setup YAM runs are not re-validated; mason's
  table-setup C runs confirm the result, then D and E. New jobs start as benchmarked
  (visual 28.6 GB / 4.1k steps/s, YAM blind 10.7 GB / 20.3–20.7k steps/s).
- **Plan cut to initial results** (Yash, 2026-10-20 ~22:40 UTC): C, D, E seed 0 only, with a
  99% stop rule (ported from `train_visual.py` to `train.py`). 136 jobs, ~161 GPU-h
  3090 Ti-equivalent (~290 V100 GPU-h, ~2.5 days on 5 GPUs; less where the stop rule fires).
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
