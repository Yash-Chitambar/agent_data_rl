# Findings, deviations and ideas (lab notebook)

Running log for `docs/RESEARCH_PLAN.md`. Dated entries; newest first within each section.
Figures and data referenced here live in `results/<phase>/`.

## Status summary

| Phase | Status |
|---|---|
| 0 | Passed (per plan). The Phase 0 scripts/configs (`blind_usbc/`, `configs/usbc_flash_drive.yaml`) are **not in this repo**; recreated as needed. |
| A | Env, policy, PPO, eval harness built. Signal checks: see table below. |
| B | Blind s0, s1, s3 (300M), visual s0-s2 (50M, stop rule), no-contact s0, s2 trained and evaluated. Blind s2 and no-contact s1 are in the local queue. Results match the paper (table below). |
| C | 16 seed-0 YAM runs on the lab-table mount (mason) trained and evaluated: 16 x 13 train/test matrix (`results/C/C-matrix.*`). **Gate C closed (2026-10-21 10:15 UTC): D/E use per-world randomized noise/quant/backlash/latency** (`randall` 96-98% on every such test). `clip1` (action clip 1.0) eval running. |
| D | Tier 1 (16 runs, 150M cap, randomized noise model) on mason (`runs/queue_server/stateD`). o60_c1 and o30_c0.5 (scaled) reach 99% training success; o15_c0.2 from scratch never succeeded, so cells tighter than 1 mm now use a clearance curriculum (`--curriculum scale`). Grid: `scripts/phase_d_report.py`. |
| E | USB-C with sphere pads: full curriculum, 91.5% at 20-100 um (athena5, old mount). E4-E6 variants built; E1 zero-shot 77.9% on the table mount. Mason: curr, chamfer, latch, side training from scratch on the table mount (no noise model, ~7.5 h each); noise (randomized), nocontact, h10, h60 queued after D tier 1. |
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

## Next steps (as of 2026-10-21 11:00 UTC)

- D tier 1 (randomized noise model) and E batch 1 on mason; then the E ablations and D tier 2.
- Gains are not in `randall`: gain x4 still gives 52%. Measure real gains or add gain
  randomization (0.5-2x) to the noise model if the real step tests disagree with kp 80/30.
- Measure the real YAM's command-to-encoder latency (in 20 ms control steps) and backlash: these
  two set the D/E noise model more than q noise does.
- Blind s2 and no-contact s1 (local queue) for the 3-seed B tables.
- Phase F after E, with Yash's OK (~70 GPU-h).

## Discoveries
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
- **USB-C from scratch on the table mount at the 300M cap:** training success curr 76% (stage 5),
  chamfer 91% (stage 5), side port 75% (stage 3). Evals follow.

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
  (`docs/SERVER.md` §9 lists what changed; `make_manifests.py --check`: 160 commands, 0 errors).
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
    `phase_e_usbc_spheres.json` (note added to INSTRUCTIONS.md for the mason agent).

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
