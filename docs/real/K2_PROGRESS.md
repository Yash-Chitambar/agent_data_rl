# K2 contact policy and alignment: progress log

Running log of the work after [CONTACT_ALIGN_PLAN_20261107.md](CONTACT_ALIGN_PLAN_20261107.md). Newest entries at the top of each section. Every number has its n; sim numbers are sim only.

## Operator decisions (2026-11-07)

- K2 spec: yes, including the climb penalty.
- Real-world learning: allowed in this form: run the real policy, learn from its failures, fix in the sim. No hundreds of manual demos.
- π0.5 / VLA: dropped for now.
- Alignment: explore both the continuous vision servo and an RL alignment policy.
- No rig until Monday 2026-11-09. Work on the policies until then.

## Learnings

- **2026-11-08. A stronger climb penalty does not change the climb.** K4c = K4 with `climb_thresh_mm` 4, `climb_coef` 0.002 (30M steps from K3 final). n = 1024 per cell: real-like 99.1 % [98.3, 99.5], ring 4 mm 98.3 % [97.4, 99.0], ring 5 mm 98.2 % [97.2, 98.9], roll ±3 deg 99.0 % [98.2, 99.5], mix 98.7 % [97.8, 99.3]; climb still 8.0-8.2 mm in every cell, as K-J, K3 and K4. In the sim the climb is joint 5 held at its 50 mrad target cap, which the policy uses for the push; the reward does not move it. The real 35-40 mm came from joints 2-3 saturating in stalls. Open: watch the climb with K4 on the rig (`kj_real_ref.py`); if it still stalls with the climb, train with a lower wrist target cap. K4c is a backup to K4 (same rates within the intervals; not tested on the real-hand-over replay).
- **2026-11-08. RL alignment (A3, Coulomb feed-forward, fixed stillness test) reaches 53 % training success at 10M steps** (A2 never passed 46 %). Still far behind the servo (99.6 %); it keeps training.

- **2026-11-08. K4 final is the rig candidate.** K4 = K3 + all left-arm inputs masked, 30M steps from K3 seed 0 final. Real-like plant, real right-arm lead, n = 1024 per cell, Wilson 95 %:

| Cell | K-J | K3 s0 final | K4 s0 final | K4 s1 final |
|---|---|---|---|---|
| real-like hand-over | 94.8 % | 98.5 % | 99.1 % [98.3, 99.5] | 98.8 % [98.0, 99.3] |
| ring 3 mm | 91.5 % | 97.9 % | 98.6 % [97.7, 99.2] | |
| ring 4 mm | 81.8 % | 96.4 % | 98.2 % [97.2, 98.9] | 98.4 % [97.5, 99.0] |
| ring 5 mm | 62.2 % | 93.8 % | 97.9 % [96.9, 98.7] | 97.6 % [96.4, 98.3] |
| roll ±3 deg | 65.0 % | 98.6 % | 98.4 % [97.5, 99.0] | 98.7 % [97.8, 99.3] |
| roll ±4 deg | 50.0 % | 98.5 % | 97.6 % [96.4, 98.3] | |
| pitch ±4 deg | 85.7 % | 98.2 % | 97.5 % [96.3, 98.3] | |
| gap 8 mm | 97.0 % | 97.0 % | 96.5 % [95.2, 97.4] | |
| mix | 79.9 % | 98.1 % | 97.9 % [96.9, 98.7] | 97.9 % [96.9, 98.7] |

Replay of all ten real hand-overs with the rig's left lead (n = 256 each, `results/real_align/replay/replay_k4_leftlead.json`): the five align-stage runs 91.8-96.5 %; the five older runs without an align stage 0-77.7 % (ai2 0 %, ai4 34 %, ai5 23 %, A 48 %, A2 78 %: they started further off than any policy handles; the align stage is needed). Chain dry run (seed 7040, `bimanual_align5_k4.json`): completes, loads `runs/insert_K4.pt` (sha256 4d288f234ff1af10), logs `insert_rel_pose_obs` and `insert_vision_prior`; seat detected, sim truth not seated after release (the known sim relax under 5.55 mm). Not met: the climb (8.0-8.2 mm, joint 5 at its cap); K4c (climb threshold 4 mm) is training.

- **2026-11-08. K4 (left-arm inputs masked) removes the left-lead sensitivity.** Replay of the five real align-stage hand-overs with the rig's real left press lead in the observation (n = 256 per run, real-like plant, K4 seed 0 early snapshot at 6.6M): align_full2 95.7 % [92.5, 97.6], full5 95.7 % [92.5, 97.6], full8 97.7 % [95.0, 98.9], full9 96.1 % [93.0, 97.9], full10 97.3 % [94.5, 98.7]. K3 and K-J (sim FK inputs) with the same real lead: 0.0 % on full9 and full10. The replay ranks policies; it does not predict real rates (it gives seats and stalls the same rate).
- **2026-11-08. K3 final, mix cell (ring 2, roll ±2, pitch ±3): 98.1 % [97.1, 98.8]** (n = 1024; 79.9 % at 9M; K-J 79.9 %). K3 passes every acceptance cell; the climb is the only open item.

- **2026-11-08. K3 final (seed 0, 40M) passes the sim acceptance bar** (real-like plant, real right-arm lead, n = 1024 per cell, `results/real_align/k2/k3s0_final.json`): real-like hand-over 98.5 % [97.6, 99.1] (bar 98); ring 3 mm 97.9 % [96.8, 98.6]; ring 4 mm 96.4 % [95.1, 97.4] (bar 95); ring 5 mm 93.8 % [92.2, 95.2]; roll ±3 deg 98.6 % [97.7, 99.2] (bar 95); roll ±4 deg 98.5 % [97.6, 99.1]; pitch ±4 deg 98.2 % [97.2, 98.9]; gap 8 mm 97.0 % [95.7, 97.9]. K-J on the same cells: 94.8, 91.5, 81.8, 62.2, 65.0, 50.0, 85.7, 97.0 %. Not yet met: the climb (8.0-8.2 mm = joint 5 at its cap, unchanged from K-J). K3 still reads the left press lead (fork C replay: 0 % with the real lead), so the rig candidate is K4, not K3.

- **2026-11-08. The left arm's press lead is a hidden port locator (fork C replay).** Both K-J and K3 read the left target error (`obs[44:49]`) to find the port. The rig reads +74-79 / -57 to -64 mrad on joints 2-3 against the sim's +101 / -71 with almost no spread. With the real values in the observation only, K3 drops from 89.8 % to 0.0 % on align_full9 and align_full10 (n = 256 each), mostly lateral misses. K-J seated 2 of 4 on the rig with the real lead, so the sim's response is too strong: a sensitivity, not a rate. Fix: **K4 = K3 + all left-arm inputs masked** (`mask_left_obs` 'all': angles, velocities, previous action, lead; the policy loader masks on the robot too), fine-tuned from K3 seed 0 final, 2 seeds, 30M steps. Acceptance adds the real-hand-over replay with the real left lead (`scripts/eval/replay_real_handovers.py`).

- **2026-11-08. The camera prior works in the sim.** Real-like plant, real hand-over lead, n = 512 per cell (K-J n = 1024), Wilson 95 %:

| Cell | K-J | K2 seed 1 @25M | K3 seed 0 @9M |
|---|---|---|---|
| real-like hand-over | 94.8 % [93.3, 96.0] | 98.4 % [96.9, 99.2] | 97.7 % [95.9, 98.7] |
| roll ±3 deg | 65.0 % [62.1, 67.9] | 90.4 % [87.6, 92.7] | 90.0 % [87.1, 92.3] |
| ring 4 mm | 81.8 % [79.4, 84.1] | 39.1 % [34.9, 43.4] | 93.0 % [90.4, 94.9] |
| ring 5 mm | 62.2 % [59.2, 65.1] | 18.9 % [15.8, 22.6] | 88.9 % [85.8, 91.3] |
| mix (ring 2, roll 2, pitch 3) | 79.9 % [77.3, 82.2] | 57.6 % [53.3, 61.8] | 79.9 % [76.2, 83.1] |

K2 learned roll (90 % against 65 %) but cannot find an offset port blind. K3 has both. K-J's numbers here use the sim's exact absolute pose; on the rig its pose features are wrong (21.8 % in the offset test). K3 uses no absolute FK, so the rig offsets cannot hurt it the same way. **Weak spot: combined errors.** In the mix cell K3 is no better than K-J (79.9 %), failures mostly lateral misses (91 of 103). Each error alone is fine; together they are not. Still to check: the acceptance bar (≥ 98 % real-like, ≥ 95 % ring 4 / roll 3), the final checkpoints, a second seed, and the climb (still 8.0-8.2 mm: joint 5 at its cap).

- **2026-11-07. K-J found the port partly from absolute FK.** K2 (relative pose only) at 10M steps, real-like plant, n = 512: real-like hand-over 89.8 % [86.9, 92.2], roll ±3 deg 47.3 % [43.0, 51.6], ring 4 mm 18.4 % [15.2, 21.9], against K-J's 94.8, 65.0, 81.8 % (n = 1024). Almost all K2 failures are lateral misses. With the absolute pose features, the sim tells K-J where the port is within the PC pose spread; on the rig that information is wrong (the stalls). Without it the policy has to search blind and finds a 4 mm offset 18 % of the time. So the camera prior (K3) is required, not optional: it replaces the wrong absolute information with a measured relative one.
- **2026-11-07. The align env's early speed peaks were a sim reset artefact.** With zero action the tip reaches 38-43 mm/s in the first 0.3 s (the arm settles under the ±12 % gravity error from targets set at the reset). Fixed: 0.5 s settle with actions ignored and no metrics (`align_settle_s`). During servo motion (2-8 s) the true tip speed is p99 13-17 mm/s, max 36 mm/s. Friction off did not remove the peaks (true speed p99 54 mm/s with the transient included), so stick-slip is not their only source. The first comparison tables (with the transient) are superseded by `results/real_align/align_cmp2/`.

- **2026-11-07. Absolute-FK observation features break K-J on the rig.** `obs[50:56]` (TCP position x10 and Euler, relative to a fixed nominal pose) read about 3 sd outside the sim's hand-over spread on every real hand-over. With these offsets in the sim, K-J: 21.8 % [19.4, 24.4] (real-like plant), re-centred 89.4 % [87.3, 91.1], none 96.6 % (n = 1024 each). The in-hand estimate offset alone has no effect (99.6 %).
- **2026-11-07. The plant explains little.** Real friction x1.3-2.0, gravity error 12 %, real hand-over lead: 95-97 % against 99.3 % (n = 1024 each).
- **2026-11-07. Roll matters, pitch does not (within 3 deg).** Roll ±2 deg 94.5 %, ±4 deg 56.3 %; pitch ±3 deg 99.4 % (n = 1024). The align stage corrects pitch and yaw only.
- **2026-11-07. A climb penalty on joint target lead would fight the push.** kp is 80 N m/rad on joints 1-3, 30 on 4-6; the push needs 25-40 mrad of lead. K2 penalises the lateral (across/up) offset of the commanded plug tip from the measured tip instead.
- **2026-11-07. Real velocity telemetry is noisy at rest** (0.005-0.04 rad/s rms, joints 4-6 highest; sim exact). Filtering it does not change K-J's wrong "inside" belief, so it is not the cause; K2 trains with matched noise anyway.

## K2 contact policy

### Runs

| Run | Init | Env | Seed | Status |
|---|---|---|---|---|
| `P4-insert-k2-s0` | K-J (`P4-insert-best/best.pt`, sha256 1ced0550) | `runs/k2_env.json` | 444 | training, 40M steps |
| `P4-insert-k2-s1` | K-J | `runs/k2_env.json` | 445 | training |
| `P4-insert-k2nc-s0` | K-J | `runs/k2_env_noclimb.json` (climb_coef 0) | 444 | stopped at 12.5M to free the GPU for K3 |
| `P4-insert-k3-s0` | K2 seed 1 at ~15M (`runs/k2s1_snap_for_k3.pt`) | `runs/k3_env.json` = K2 + `vision_prior_obs` | 446 | training, 40M steps |

**K3 = K2 + the wrist-camera prior** (`InsertionCfg.vision_prior_obs`): 3 inputs appended after the in-hand estimate, the port-vision plug position at the hand-over (across, up, gap; mm / 10), held for the episode, = truth + N(0, (0.4, 0.95, 0.63) mm) per episode (run bias, pose-dependent error and the 5-frame median of the align stage together). Robot side: `deploy/bimanual/insert.py` appends the align stage's fused hand-over measurement (`align_result` lat_mm, gap_mm; event `insert_vision_prior`); it needs `align.mode` full.

K2 env = K-J's training env + (new `InsertionCfg` fields, all off by default):
- `rel_pose_obs`: `obs[50:56]` relative to the believed TCP pose at the reset (the hand-over).
- `vel_obs_noise` [0.006, 0.003, 0.003, 0.018, 0.016, 0.028] rad/s, per-world scale x0.5-1.5.
- `extra_ang_deg` [4, 4, 3] (rx, ry, roll) on 50 % of pool starts.
- `handover_lead_rad` = mean real hand-over lead, x0.5-1.5 + N(0, 3 mrad), 50 % of resets.
- `climb_coef` 0.001 per mm per step above `climb_thresh_mm` 8.
- `friction_scale_range` [0.5, 2.0], `gravity_scale_jitter` 0.12, `hard_lat_m` [1.5, 4] mm.

Smoke test (K-J itself in the K2 env, n = 256): 32.4 %. The obs, noise and climb measure work.

### K-J baseline on the acceptance matrix (real-like plant, real hand-over lead, n = 1024 per cell; `results/real_align/k2/kj_real.json`)

The sim's own observation (no real FK offsets), so this is K-J's best case.

| Cell | K-J |
|---|---|
| real-like hand-over (gap 5.5, ring 0.5 mm) | 94.8 % [93.3, 96.0] |
| ring 3 mm | 91.5 % [89.6, 93.1] |
| ring 4 mm | 81.8 % [79.4, 84.1] |
| ring 5 mm | 62.2 % [59.2, 65.1] |
| roll ±3 deg | 65.0 % [62.1, 67.9] |
| roll ±4 deg | 50.0 % [46.9, 53.1] |
| pitch ±4 deg | 85.7 % [83.5, 87.8] |
| gap 8 mm | 97.0 % [95.7, 97.9] |
| mix (ring 2, roll ±2, pitch ±3) | 79.9 % [77.3, 82.2] |

Climb (lateral offset of the commanded tip from the measured tip): 8.0-8.2 mm in every cell = joint 5 at its 50 mrad cap (0.05 rad x ~160 mm). In the sim joints 2-3 stay off their caps; on the rig they do not, which gives the real 35-40 mm.

### Deploy

- `insert.policy_file` + `insert.policy_sha256` in a run config: the chain loads that policy instead of the bundle's K-J (the bundle stays read-only). The file must be copied to the rig like the wrist nets.
- A policy whose `env_json` has `rel_pose_obs` gets its pose features relative to the hand-over pose on the robot too (`deploy/bimanual/insert.py`, event `insert_rel_pose_obs`). It refuses `insert.obs_recenter`.

### Acceptance (sim, before the rig), n = 1024 per cell

- ≥ 98 % from real-like hand-overs on the real-like plant.
- ≥ 95 % at ring 4 mm, and at roll 3 deg.
- Commanded tip climb ≤ 10 mm p95.
- Contact head reads "touching" before entry.

## Alignment

### Sim env (2026-11-07)

`blind_insertion/envs/align_task.py` (`AlignTask`, selected by `align_task: true` in the env JSON; `InsertionCfg.align_*` fields). Same physics, grasp and joint interface as the insertion env. Start: plug 12-24 mm out, ±5 mm across, ±8 mm up, ±3 deg (uniform). Goal: (0, 0, 6) mm, within 1 mm lateral, 1 mm gap, 1 deg, tip below 2 mm/s for 0.3 s. Plant: friction x0.5-2.0, gravity ±12 %, velocity noise as K2, encoder offsets ±1.5 deg.
- Action: plug-tip twist in the estimated port frame (port pose error 1 deg), through the believed tip Jacobian into the capped accumulated joint targets.
- Simulated port vision: 10 Hz, ~100 ms latency, slope ±5 %, run bias (0.3, 0.6, 0.4) mm and 0.5 deg, smooth position-dependent error (0.25, 0.74, 0.49) mm rms over 2-6 mm wavelengths, frame noise 0.1 mm / 0.3 deg, 10 % dropped frames.
- The actor sees the last frame, its age, the believed motion since the frame, joint angles relative to the start, velocities, target lead, previous action. No absolute FK.
- Success is judged on the vision-consistent error (truth + run bias): no method can remove a constant camera bias. True error is reported too.

Smoothness metrics (peak tip speed, path / start error, overshoot past 6 mm) were added after the first baseline run; `scripts/eval/align_compare.py` reports them.

### First baselines (n = 256 each, 30 s limit)

| Method | Aligned | Time p50 / p90 | True lateral at the end p50 / p95 |
|---|---|---|---|
| Step stage (1 mm steps, settle 0.3 s, 5 frames) | 247 (96.5 %) | 17.8 / 22.4 s | 0.83 / 1.61 mm |
| Continuous vision servo (gain 1.5/s, 8 mm/s max) | 239 (93.4 %) | 7.1 / 19.1 s | 0.77 / 1.66 mm |

The servo is 2.5x faster at the median but has a slow tail (p90 19.1 s), probably stick-slip near the goal where the commanded speed is small. Next: servo with a minimum speed / friction-aware lead, and the RL policy.

### Comparison with smoothness metrics (n = 256 each, `results/real_align/align_cmp/`)

| Method | Aligned [95 % CI] | Time p50 / p90 s | Peak tip speed p50 / p99 mm/s | Path / straight line p50 / p90 | Overshoot past 6 mm p95 / max |
|---|---|---|---|---|---|
| Step stage | 249 [94.5, 98.7] % | 17.3 / 22.5 | 25.6 / 43.5 | 3.4 / 4.6 | 1.0 / 1.9 mm |
| Servo, P only | 237 [88.7, 95.2] % | 6.7 / 19.8 | 23.4 / 45.0 | 2.7 / 5.2 | 1.4 / 2.1 mm |
| Servo + speed floor 1.5 mm/s + deadband 0.4 | 241 [90.6, 96.4] % | 6.8 / 18.2 | 23.3 / 45.0 | 2.7 / 4.7 | 1.4 / 2.0 mm |
| Servo, gain 2.5, 10 mm/s | 194 [70.2, 80.6] % | 7.6 / 20.8 | 27.0 / 46.8 | 3.2 / 8.9 | 1.5 / 2.4 mm |

**Learning (2026-11-07): no command law makes the motion smooth.** Every method commands at most 8-10 mm/s, yet the tip peaks at 23-27 mm/s (p50) and 43-47 mm/s (p99), the path is about 3x the straight line, and the plug passes the 6 mm goal by up to 2 mm. The sim has the real arm's behaviour (2026-11-03: 1 mm steps arrived at 54-78 mm/s). Speed here is from the believed FK; a true-speed check and a no-friction diagnostic are running, to confirm that joint stick-slip is the cause. If it is, the fix is below the command law: a friction-aware joint layer (the precision controller's feed-forward, or a lead that the policy controls), which the RL policy can learn through its target-lead observation.

### Vision-based done (2026-11-07, n = 256 each, `results/real_align/align_cmp3/`)

The stage stops when the fused vision estimate is within tolerance and the tip is still, as on the rig (`align_done_on` vision). The true error is read at that moment. Both with the chain Coulomb feed-forward.

| Method | Done [95 % CI] | Time p50 / p90 s | True lateral p50 / p95 mm | True gap error p95 mm | True angle p50 / p95 deg | Peak true speed p50 / p99 mm/s | Path ratio p50 / p90 | Overshoot p95 / max mm |
|---|---|---|---|---|---|---|---|---|
| Step + FF | 248 / 256 = 96.9 % [94.0, 98.4] | 16.2 / 22.3 | 0.88 / 2.04 | 1.35 | 0.84 / 1.58 | 19.2 / 38.4 | 2.33 / 3.28 | 1.18 / 1.83 |
| Servo + FF | 239 / 256 = 93.4 % [89.6, 95.8] | 6.1 / 15.7 | 0.81 / 1.91 | 1.24 | 0.81 / 1.57 | 16.0 / 37.5 | 1.63 / 2.93 | 1.30 / 1.69 |

- The servo is 2.7x faster at the median, with a straighter path and similar accuracy. 16 servo episodes did not finish in 30 s.
- The true lateral error at done is up to 2 mm (p95) for both. This is the vision model's position-dependent error, not the control. K2/K3 must tolerate it.
- These runs judged "still" from the one-step FK speed, which has about 5 mm/s of encoder-quantization noise. From now on the env uses the displacement over 5 steps (0.1 s, `align_still_window`). Runs before this change (including `A2-align-rl-s0`) have the noisy test.

### Precision controller in the sim (2026-11-07)

- `blind_insertion/envs/precision_ctrl_torch.py`: the real law, batched (integral, settle lock, friction feed-forward; no vibration watch). Parity with the NumPy law: max torque difference 0.0 N m over 20 x 600 random steps in float64 (`scripts/eval/align_ctrl_parity.py`). Float32 differs only at exact threshold ties.
- AlignTask option `ctrl_precision` (keys as `precision_ctrl.DEFAULTS`): the law's torque on the right arm, computed once per 20 ms control step from the sent target and the encoder reading, held over the substeps. Limits: the real law runs at 200 Hz, the sim at 50 Hz. The sim keeps i2rt's default-gravity error (±12 %); the real controller runs with the fitted gravity, so the sim is the harder case.
- Comparison step / servo x plain PD / Coulomb FF / controller: `scripts/eval/align_ctrl_matrix.sh`, `results/real_align/align_ctrl/`.

**Results (vision-done, 0.1 s stillness window, 30 s limit, n = 256 unless noted):**

| Method + joint layer | Done [95 % CI] | Time p50 / p90 s | True lateral p50 / p95 mm | Peak true speed p50 / p99 mm/s | Path ratio p50 / p90 | Overshoot p95 / max mm |
|---|---|---|---|---|---|---|
| Step, plain PD | 236 = 92.2 % [88.2, 94.9] | 18.2 / 24.3 | 0.86 / 2.08 | 20.6 / 39.0 | 2.36 / 3.12 | 1.14 / 1.62 |
| Step, Coulomb FF | 248 = 96.9 % [94.0, 98.4] | 16.7 / 22.7 | 0.91 / 1.99 | 19.0 / 37.8 | 2.33 / 3.05 | 1.29 / 2.13 |
| Step, controller | 232 = 90.6 % [86.4, 93.6] | 18.7 / 24.3 | 0.87 / 1.80 | 40.1 / 62.5 | 5.44 / 7.85 | 1.45 / 1.96 |
| Servo, plain PD | 252 = 98.4 % [96.1, 99.4] | 4.6 / 10.7 | 0.80 / 1.68 | 18.1 / 39.8 | 1.52 / 2.50 | 1.26 / 1.82 |
| **Servo, Coulomb FF** | **255 = 99.6 % [97.8, 99.9]** | **4.4 / 9.4** | 0.74 / 1.77 | **16.0** / 37.5 | **1.37** / 2.36 | 1.21 / 1.88 |
| Servo, controller | 94 = 36.7 % [31.1, 42.8] | 12.0 / 26.2 | 0.90 / 1.83 | 40.7 / 52.5 | 5.12 / 11.8 | 2.26 / 3.21 |
| Servo, controller without integral (n = 128) | 125 = 97.7 % [93.3, 99.2] | 4.3 / 11.7 | 0.80 / 1.80 | 19.0 / 36.6 | 1.47 / 2.37 | 1.18 / 1.92 |
| Servo, controller, integral 4x slower (i_tau 0.48, n = 128) | 91 = 71.1 % [62.7, 78.2] | 9.3 / 23.0 | 0.86 / 1.56 | 19.6 / 36.3 | 2.69 / 6.71 | 1.90 / 2.64 |

**Learnings:**
- **The continuous servo with the Coulomb feed-forward is the best method in the sim:** 99.6 % in 4.4 s median (9.4 s p90), against 16.7 s for today's step stage, with the straightest path. Its earlier slow tail came from the noisy one-step stillness test, not from the servo.
- **The precision controller's integral makes the sim motion worse** (peak speed x2, path x3-4, servo 36.7 %). Without the integral the controller equals the Coulomb FF. This does not match the rig, where the controller made the align stage the best so far (align_prec7). The likely cause is the sim, not the law: the sim runs the law at 50 Hz with its torque held 20 ms (real: 200 Hz), so the integral (kp / i_tau = 667 N m per rad s) rings; a 4x slower integral is better but still poor. Not verified: a 200 Hz sim law needs a per-substep hook in the runtime. Until then, the sim says nothing reliable about the controller's integral; do not use these rows against the real result.
- **RL (A2 at 11.8M steps, n = 128, same env and done test):** 11 done = 8.6 % [4.9, 14.7], 117 timeouts. Its motion is the smoothest of all (peak true speed p50 / p99 13.0 / 16.3 mm/s, path ratio 1.13 / 1.45), but it does not settle inside tolerance and passes the 6 mm goal by 2.1-2.8 mm (p95 / max). A2 trains with the old one-step stillness test (about 5 mm/s of quantization noise against a 2 mm/s limit), so it rarely sees a success; its training success fell from 46 % at 6.6M to 23 % at 11M (curriculum stage 0). Recommendation: restart it on the current env (`align_still_window` 5), keep the smoothness terms; compare at 60M.
- The remaining peak speed (p99 about 37 mm/s for every layer without the integral) and overshoot (about 1.2-1.9 mm past the 6 mm goal) are common to all methods. The true lateral error at done (p95 1.7-2.1 mm) is the vision model's position-dependent error, the same for every method.


### Bumpless switch-off of the controller (2026-11-07)

**Problem.** Leaving the controller's active stages drops its integral and feed-forward torque and switches i2rt from the fitted to the default gravity model in one tick. With the real hand-over integral (align_full8: joint 2 -1.14, joint 3 +0.75 N m) the commanded equilibrium of joint 2 jumps about 24 mrad (the 4-6 mm sag in the config note).

**Fix** (`deploy/bimanual/io_real.py`, config key `arm_control.<side>.bumpless_off`, off by default). The switch-off waits for the next send. That send reads the controller torque and i2rt's gravity torque at the measured pose before and after the model switch, switches the controller off, and commands target + (tau_ctrl + g_fit - g_default) / kp. The bias is held through insert and release. A switch-on or the next slow move clears it (moves start from the measured pose with their own trim). `io.hold()` (emergency hold) holds the measured pose without the bias.

**Check** (`deploy/checks/handover_bumpless.py`, real-I/O mock, stub gravity model with 0.8 / 0.5 N m differences on joints 2 / 3, the real hand-over integral): commanded equilibrium jump at a fixed pose 0.000 mrad on every joint with `bumpless_off`; without it, joint 2 jumps 24.3 mrad and joint 3 15.6 mrad. The existing real-I/O mock still passes. This checks the code path on a fake arm, not the real arm: the i2rt motor thread can run one cycle (1 ms) between the switch-off and the new command.

**Config:** `deploy/config/bimanual_align5_precision.json` = `bimanual_align4_precision` + `bumpless_off` + `align.full_allowed` true (default mode still align). Conventions check: FAILED none. **Untested on the arm.** First real use: align mode, then full.

### Runs

| Run | What | Status |
|---|---|---|
| `A1-align-rl-s0` | PPO from scratch on AlignTask, 2048 worlds, curriculum start x0.5 then x1.0, 60M steps | stopped at ~8M (env had the reset transient); `runs/A1-align-rl-s0_old_transient` |
| `A2-align-rl-s0` | as A1, env with the 0.5 s settle | training |

## K3 deploy

### 2026-11-08 (fork B)

**Chain path.** New config `deploy/config/bimanual_align5_k3.json` = `bimanual_align4` + `insert.policy_file` `runs/insert_K3.pt` + `insert.policy_sha256`. The file is a snapshot of K3 seed 0 at about 13M steps (sha256 `d22cb7ef4b92f35d`). TODO: replace with the final K3 checkpoint and its sha256.
- The conventions check passes (`FAILED: none`).
- A wrong sha256 stops `run_bimanual` at the start, with a message.
- Sim dry runs, full chain, seeds 7030-7032: 3 of 3 complete. The policy loads from `policy_file` (obs_dim 71). Each run logs `insert_rel_pose_obs` and `insert_vision_prior`. The prior equals the align stage's hand-over measurement (`align_handover` lat_mm, gap_mm) to 1e-7 mm.
- Seat detector: 3 of 3 `seated_detected` (1.26-2.12 s). Sim truth at the end: 1 of 3 seated; the other two stop at 5.12 and 5.36 mm, under the sim's 5.55 mm limit (known: the plug relaxes when the push stops; the sim fingers do not yield). n = 3: not a seat rate.

**Observation check (`--check-obs`).** `obs_max_abs_err` now covers dims 0:50 only. Segments are reported apart in `obs_seg_max_abs_err` (`deploy/bimanual/insert.py`). Seeds 7031, 7032:

| Segment | Max abs difference | Why |
|---|---|---|
| 0:50 (joints, velocities, actions, target errors) | 0.063 | expected: the K3 env adds its training velocity noise (`vel_obs_noise`, up to 0.04 rad/s sd) to its own observation. K-J dry run, same seed: 7e-10 on 0:50 and 0 on every segment |
| 50:56 (pose) | 0.23-0.25 | expected: the sim env's reference is its own reset pose, the chain's is the hand-over pose |
| 56:68 (twist, in-hand) | 0.068 | |
| 68:71 (vision prior) | 0.03-0.06 (0.3-0.6 mm) | expected: the sim env's own prior draw, not the align stage's measurement |

**Robustness in the sim** (`scripts/eval/k3_robust.py`, the 13M snapshot, real-like plant, real hand-over lead, n = 512 per row; `results/real_align/k3/robust_a.json`):

| Prior | Hand-over (gap 5.5, ring 0.5 mm) | Ring 3 mm |
|---|---|---|
| as trained | 95.5 % [93.3, 97.0] | 93.0 % [90.4, 94.9] |
| up +1 mm | 96.3 % [94.3, 97.6] | 92.0 % [89.3, 94.0] |
| up +2 mm | 94.1 % [91.8, 95.9] | 89.6 % [86.7, 92.0] |
| up +3 mm | 88.3 % [85.2, 90.8] | 73.4 % [69.4, 77.1] |
| across +2 mm | 95.5 % [93.3, 97.0] | 92.4 % [89.8, 94.4] |
| "on target" for every world | 96.5 % [94.5, 97.8] | **40.6 % [36.5, 44.9]** |

- K3 uses the prior to find an off-centre port. With a prior that always says "on target", a 3 mm offset seats 40.6 % (with the true prior 93.0 %).
- A prior bias up to 2 mm costs at most 3.4 points; 3 mm up costs 7 (hand-over) and 20 points (ring 3). The vision-to-net up difference on 2026-11-03 was 2-3 mm, cause not settled. **The up accuracy of port vision is the next measurement that matters for K3.**
- Contact head against the sim's labels (hand-over cell, first 150 steps): 95.9 % of edge points right; recall free 96.2, touching 97.6, inside 92.7 %.
- Encoder offsets ±3 deg (training ±1.5): hand-over 91.8 % [89.1, 93.9], ring 3 mm 86.9 % [83.7, 89.6] (n = 512 each), 3.7 and 6.1 points below. The relative pose removes the constant offset, but not all of it: the offset also bends the believed Jacobian and the believed motion directions. K3 is not immune, only less sensitive (K-J lost 75 points to the real absolute offsets).
- Coordinator's matrix for K3 seed 0 at 9M (n = 512, real-like plant): real-like hand-over 97.7 %, roll 3 deg 90.0 %, ring 4 mm 93.0 %, ring 5 mm 88.9 %, mix 79.9 % (`results/real_align/k2/k3s0_a.json`). K-J (n = 1024): 94.8, 65.0, 81.8, 62.2, 79.9 %. TODO: final checkpoints.

**Runbook:** [NEXT_RIG_20261109.md](NEXT_RIG_20261109.md).


## Real-to-sim replay and roll

### 2026-11-07: Real hand-over set

`scripts/real/real_handover_states.py` -> `results/real_align/replay/real_handovers.json`. 11 real runs reached the insertion, all with the bundle's K-J (one bundle, staged 2026-11-01). Operator truth: 4 seated (`ai3`, `ai4`, `align_full2`, `align_full9`), 7 not. Vision at the hand-over: the align stage's `align_handover` for the 5 align runs; offline port_vision on the last raw wrist frames before the insertion for 5 older runs. `ai3` has no good frame. Offline vision agrees with `align_handover` within 0.4 mm on 4 of the 5 align runs; `align_full2` differs by 2.6 mm in up (the press came after the last frame). Each row also has the run's angles (`rv_deg`), hand-over lead, in-hand estimate and FK pose features.

### 2026-11-07: Replay in the sim (n = 256 per run)

`scripts/eval/replay_real_handovers.py`. Plant: friction x1.3-2.0, gravity error 12 %, real velocity noise. True start = the run's vision + N(0, (0.4, 0.95, 0.63) mm); angles = the run's last `rv_deg` + N(0, 0.5 deg) (align runs) or the chain pool's angles (older runs). Lead and in-hand input = the run's logged values. K3: the prior = the run's vision exactly; the true start differs by the vision error, as on the robot. K3 = `P4-insert-k3-s0` at about 14M steps (snapshot `results/real_align/replay/k3s0_forkC.pt`, sha256 d22cb7ef); `align_full8` is replayed on plain PD (the sim has no controller).

| Run | Truth | K-J, real FK obs | K-J, sim's own FK obs | K3 |
|---|---|---|---|---|
| ai2 | no | 0.4 % [0.1, 2.2] | 1.2 % [0.4, 3.4] | 0.0 % [0.0, 1.5] |
| ai4 | **seated** | 0.4 % [0.1, 2.2] | 6.2 % [3.9, 9.9] | 28.5 % [23.3, 34.3] |
| ai5 | no | 7.8 % [5.1, 11.8] | 25.4 % [20.4, 31.1] | 14.8 % [11.0, 19.7] |
| full_A | no | 1.6 % [0.6, 3.9] | 11.3 % [8.0, 15.8] | 50.8 % [44.7, 56.8] |
| full_A2 | no | 5.5 % [3.3, 9.0] | 11.3 % [8.0, 15.8] | 64.5 % [58.4, 70.1] |
| align_full2 | **seated** | 0.4 % [0.1, 2.2] | 32.8 % [27.4, 38.8] | 89.8 % [85.5, 93.0] |
| align_full5 | no | 0.8 % [0.2, 2.8] | 28.9 % [23.7, 34.7] | 79.7 % [74.3, 84.2] |
| align_full8 (controller) | no | 1.2 % [0.4, 3.4] | 32.4 % [27.0, 38.4] | 90.6 % [86.4, 93.6] |
| align_full9 | **seated** | 0.4 % [0.1, 2.2] | 39.8 % [34.0, 45.9] | 89.8 % [85.5, 93.0] |
| align_full10 | no | 0.0 % [0.0, 1.5] | 35.5 % [29.9, 41.6] | 85.5 % [80.7, 89.3] |

Isolation, K-J, 5 align runs, n = 256 each: real FK obs without the real in-hand input, run angles: 0.0-2.0 %; same with the pool's angles: 4.3-13.3 %; sim FK obs with the real in-hand input and run angles (table): 28.9-39.8 %.

**Learnings.**
- The replay does not predict the real seats: the real seats and stalls get the same sim rate (K-J real obs: 0.4 % for the 3 seats and 0.0-7.8 % for the failures). The sim is too pessimistic for K-J with the real offsets (real plain-PD align runs: 2 of 4). So the replay is a ranking tool, not a calibrated predictor. Likely missing pieces: the true real grasp (the sim's true grasp is the nominal one, while the in-hand input says +4 mm), real compliance, and the real FK offsets being partly true geometry, not only error.
- All three replay variants rank the same way: the real FK pose offsets cost K-J the most, the run angles (roll about -1.5 to -2.2 deg) cost the next most.
- K3 is far less sensitive: 80-91 % on the 5 align hand-overs, and 51-65 % on the far-off older stops (A, A2: 5-9 mm off). Its failures are mostly lateral misses.
- The older runs without an align stage (5-16 mm off by offline vision) are hard for every policy, as expected.
- `grasp='real'` (the measured real grasp as the sim truth) crashes in `InsertionTask.__init__` (`grasp_info` is set only for `part1`), so the variant with a matched true grasp did not run. Open.

### 2026-11-07: Observation OOD (real obs against the sim's, Mahalanobis per channel group and time bin)

`scripts/eval/replay_obs_ood.py`, 4 plain-PD align runs against 512 sim worlds of the policy's own env (real-like plant, real hand-over lead). OOD = real distance above the sim's own p99 in all 4 runs.

| Channel | K-J | K3 |
|---|---|---|
| left joint target - measured (the press), obs[44:49] | **OOD, d 20-36 (sim p99 4)** | **OOD, same** |
| right / left joint angles | OOD, d 4-5 (p99 3) | OOD, same |
| right target lead | OOD at 0-0.2 s (d 8-14, p99 3) | OOD at all times |
| right joint velocity | OOD at 0.2-0.6 s | OOD at 0.2-0.6 s |
| TCP position / Euler | OOD (the absolute offsets) | in at the hand-over; OOD at 0.2-1.2 s (the real motion differs) |
| in-hand estimate | borderline (d 5-7, p99 5.3) | borderline |
| twist, previous action | partly OOD at 0.2-0.6 s | partly |

The left press lead is the largest gap and is the same for K3: real (j2, j3) = (+74..+79, -57..-64) mrad, sim dry runs (+101..+102, -71..-73) with almost no spread. The real hold latches a smaller lead (i2rt gravity, real friction). K3 training should randomise the left press lead (or mask it) and the joint-angle offsets. **Replay with the real left press lead in the observation only** (`--real-left-lead`, obs[44:49] = the run's value, the plant unchanged; n = 256 each): K3 on `align_full9` 0.0 % [0.0, 1.5] (89.8 % without), `align_full10` 0.0 % [0.0, 1.5] (85.5 % without); K-J with the sim's own FK obs 0.0 % on both (39.8 and 35.5 % without). Almost all lateral misses. Both policies use the left press lead as a cue for where the port is (in the sim, the scripted press lead follows the PC pose with almost no other spread).
- On the rig K-J saw this same lead and still seated 2 of 4, so the sim's response is too strong: read this as a sensitivity, not as a real rate.
- It is still a risk for K3 that costs nothing to remove: in the next fine-tune (K4), mask the left-arm channels (obs 6:11, 18:23, 44:49), or randomise the left press lead by about ±30 mrad and the left joint angles. The sensitivity test above is the acceptance check.

### 2026-11-07: Roll

The align stage's plug angles (`align.py PortVisionSource`) = the panel pose fitted in the image x a fixed plug-in-camera rotation (mount + the mean real grasp). So `rv_deg` sees the hand's roll against the panel, never a grasp rolled in the jaws. The real-grasp preset (`inhand_real` v2, n = 13 grasps) has 0.01 deg roll spread about the plug axis: roll in the hand was not measured, so it is unknown, not zero.

- **Vision part is precise** (`scripts/real/roll_vision_consistency.py`, 15 runs; R_cam from FK of the logged q_b x panel pose, so the panel orientation in the arm base should not change while the hand moves). Roll axis: frame-to-frame spread 0.21 deg (median of cycles); panel-in-base sd over a run 0.22 deg (median, range 0.02-0.68 deg while the hand turned 1-2 deg); across runs of a session sd 0.37 deg (11-03, n = 7) and 0.45 deg (11-06, n = 8), which also has the encoder and PC-pose spread. The reconstruction matches the logged `rv_deg` within 0.04-0.24 deg.
- **The measured roll is a steady offset:** rz -1.64 ± 0.36 deg (mean ± sd over 15 runs, range -1.1 to -2.3). Seats -1.29 and -1.57, stalls -1.36 and -2.26 (run means): no separation (n = 2 + 2).
- **Is the -1.6 deg real or a calibration bias?** Not resolved. A model-free check (`scripts/real/roll_image_check.py`: tilt of the shell's sleeve end against the panel's across axis in the same image, 16 runs x up to 40 frames) is too noisy: 1 deg sd per frame, and no relation to the logged rz across runs (correlation -0.20). Its session means differ by about 1.5 deg (11-03 -0.1, 11-06 -1.6 image deg), which may be the PC pose (perspective).
- **Recommendation: do not add roll to the align stage's corrected axes now.** The vision part (about 0.2-0.5 deg) would be good enough, but the absolute zero (mount roll + grasp roll) is unverified. If the -1.6 deg is a bias, a correction would create a 1.6 deg roll error. Sim cost of 2 deg roll for K-J: 94.5 % against 99.5 % (n = 1024); K2/K3 train with ±3 deg roll, so they should not need it (check K3 at roll ±3 in the matrix). The rig check that would settle it: a roll sweep at the hand-over (wrist ±1.5 deg about the plug axis, measure mode, no insertion), with a square-on photo of the plug's long edge against the port slot, or a calibration of the camera mount roll on the checkerboard (`precision_check.py --camera` setup).

## Servo deploy

### 2026-11-08: continuous vision servo in the real chain's align stage (sim and offline checks only; never run on the arm)

**What it is.** `align.servo` (a dict) in a run config switches the align stage from 1 mm steps to `ServoStage` (`deploy/bimanual/align.py`). Without the key, the step stage runs as before (measure mode always uses the step stage, so it never moves).
- **Every 20 ms tick:** fused estimate = the newest port-vision result + the believed FK tip motion since that frame's capture tick (port axes); plug-tip twist from the sim's `ServoPolicy` law (gain 1.5/s, speed floor 1.5 mm/s outside a 0.4 mm / 0.4 deg dead band, 8 mm/s and 4 deg/s caps, smoothing 0.2); joint increment by the believed tip Jacobian (damped least squares); target within 0.05 rad of the measured joints; the chain's Coulomb feed-forward on.
- **Funnel:** the gap goal stays at `gap_hold_mm` (9) until the lateral error is within `funnel_lat_mm` (2). Angle correction on the config's axes (rx, ry); roll is not corrected.
- **Vision off the loop:** `deploy/bimanual/vision_thread.py`. Default `ProcessWorker`: `port_vision.measure` in a child process, the image through shared memory, the result converted on the main side (`PortVisionSource.convert`, refactored out of `measure`, same numbers). A thread worker is available (`servo.vision_worker: "thread"`) but holds the interpreter lock (below).
- **Gates kept:** entry gap (12 mm), gripper change, distance from the stop, clearance floor, error growth, measurement against FK between frames, measurement scale against FK motion (regression over all frames; probes only when an axis did not move enough), angle limit (7 deg), second-source disagreement at done, time limit (30 s), back-off on failure (straight out, joint lead kept, feed-forward folded into the target and off at the end). **New:** no valid frame for 0.5 s -> hold still; for 3 s -> stop; tip faster than 60 mm/s over 0.1 s -> stop. **Done:** estimate within tolerance for `confirm_n` frames in a row and the tip still over 0.1 s (one-tick FK differences carry about 5 mm/s of encoder quantization).
- **Hand-over:** press back on over `repress_s` while the servo holds; only frames captured after the press ramp and a 0.3 s settle count (`confirm_n` + 1); then the lead is clipped into K-J's range, the feed-forward folded into the target and off, in-hand net refreshed, `align_handover` with `gap_mm` / `lat_mm` = the fused estimate (K4's vision prior).
- **Configs:** `bimanual_align6_servo.json` = `bimanual_align4` + servo, align mode only (`full_allowed` false); `bimanual_align6_servo_k4.json` = `bimanual_align5_k4` + servo (full chain with K4). Conventions check: FAILED: none on both.

**Sim dry runs** (`deploy/checks/align_servo_dryrun.py`; `results/real_align/servo_deploy/`; sim vision = truth + run bias (0.3, 0.6, 0.4) mm + pose error + frame noise, 10 % dropped, 100 ms latency):

| Runs | Aligned | Servo time | True error at done (across, up, gap - 6) | Tip speed p99 / max (0.1 s window) |
|---|---|---|---|---|
| align mode, seeds 7100-7103 | 4 of 4 | 5.1, 7.5, 3.4, 3.7 s | (-0.05, -0.24, 0.26), (-0.5, 0.95, 0.57), (0.23, 0.33, -0.84), (0.07, 0.38, 0.02) mm | 11.2-15.2 / 12.2-15.5 mm/s |
| align mode, vision latency 240 ms and 3.3 frames/s, seeds 7131-7132 | 2 of 2 | 3.5, 6.2 s | (-0.05, 0.0, 0.41), (-0.98, 0.79, 0.36) mm | 15.1, 11.2 / 15.3, 12.8 mm/s |
| full chain with K4, seeds 7111-7113 (7111 before the press fix) | 3 of 3 | 6.6, 3.5, 2.9 s + press 0.3, 5.4, 2.4 s | | 13.8-20.2 / 16.1-23.0 mm/s (press phase included) |
| full chain with K4, seeds 7121-7123 | 3 of 3 | 9.3, 4.6, 2.7 s + press 5.4, 7.6, 6.5 s | | 12.6-14.8 / 14.8-18.9 mm/s |
| step stage, same chain (regression, seed 7141) | 1 of 1 | 32.4 s, 24 cycles | | |

K4 after the servo hand-over: **5 of 6 truly seated** (sim truth at the end; seat at 1.04-1.8 s). The failure (seed 7123): the true hand-over was (-1.01, -1.27, 7.18) mm while the fused vision estimate said (-0.31, 0.15, 5.53): a 1.6 mm vision error (the sim's run bias draw), press abort with the plug on the case. n = 6 is a code-path check, not a rate.

**Vision worker timing** (`deploy/checks/align_servo_thread.py`, 10 s per backend, 50 Hz loop with per-tick torch work, mason CPU loaded by training):

| Worker | Loop period p50 / p99 / max | Ticks over 25 ms | Frames/s | Frame age at use p50 / p99 |
|---|---|---|---|---|
| none (baseline) | 20.0 / 20.05 / 20.05 ms | 0 of 500 | - | - |
| thread, fake 82 ms (sleep or numpy) | 20.0 / 20.05 / 20.05 ms | 0 | 9.9 | 140 / 180 ms |
| thread, real `port_vision.measure` | 20.0 / 34.2 / 44.1 ms | 16 of 495 (28 in a repeat) | 10.9 | 129 / 205 ms |
| **child process, real `port_vision.measure`** (default) | 20.0 / 20.1 / 23.1 ms | 0 of 500 | 6.0 | 240 / 360 ms |

port_vision holds Python's interpreter lock for part of its work, so in a thread it delays ticks; in a child process the loop keeps its period. On mason (loaded by training) one call took 108-267 ms (`docs/real/PORT_VISION.md`: 82 ms median on an idle CPU), so the rig should give more frames per second.

**Real frames through the robot-only path** (`deploy/checks/align_servo_realpath.py`, 12 align frames and 9 depth frames of 2026-11-06 `align_full9`): shell depth 119.469 mm (the run logged 119.467); child-process worker against the inline call: same ok flags, max difference 0.0 on position and angles. **Pass.**

**Other checks:** `deploy.checks.align_gates`: FAILED: none (step logic unchanged). The real-I/O mock does not cover the align stage; not run for it.

**Not verified:** anything on the arm. In particular: real stick-slip under a continuously moving target with the feed-forward on (the sim's joint friction model is a Coulomb band, not the measured breakaway behaviour); the camera latency constant (`cam_latency_s` 0.03, a guess) that maps a frame's arrival time to its capture tick; the child process start time on the rig (spawn, port_vision model load); frames per second on the rig CPU; the tip speed limit of 60 mm/s against real breakaways (2026-11-03 steps arrived at 54-78 mm/s: the stop may fire); the fused estimate's FK motion term with real link flex.

## Plug and port in one wrist image

2026-11-08, fork E. Operator request: measure the plug tip against the port in one right wrist image, with the measured reader dimensions. Offline only. No arm motion.

**The direct measurement already runs.** `deploy/bimanual/port_vision.measure` finds the plug shell and the port row in the same raw frame. It gives the tip against port 2 (across, up, gap) with no encoders and no in-hand estimate. The align stage (step and servo) uses this number, and K4's vision prior is this number. So the encoder FK error and the grasp position error do not enter the alignment target. The earlier note "the vision prior only measures the port" was wrong.

**Measured dimensions** (`data/real_yam/usbc_lab_20261027/measurements.csv`, calipers, operator): silver body 34.6 x 29.6 x 8.75 mm; black neck 43.4 mm long, 12.96 mm wide, 7.0 mm thick; metal plug protrusion from the neck edge 5.8 mm (three readings). The shell width and thickness are not measured: port_vision uses the USB-C specification (8.25 x 2.40 mm; effective rim width 8.15 mm from frames). Check: 5.8 mm of protrusion at 117 mm depth and 63 deg view angle projects to 30.1 px; port_vision's empirical constant is 32 px. They agree within 6 %.

**Visibility.** The shell and the port row are both visible in all 8 real hand-over frames (2026-11-06 prec4-7, full7-10; port_vision ok 8 of 8). Sheet: `results/real_align/relvis/sheet_handover.jpg` (`scripts/real/relvis_sheet.py`).

**The black neck is not a usable second depth scale.** It is black on a dark table. Edge step about 2 grey levels, SNR 1.3-2.2. Depth from the neck: SD 10-24 mm per run; from the shell width: 0.35-1.1 mm (8 runs, 20-42 frames each; `relvis_neck.py`, `neck.json`). Up stays the weak axis (held-out residual 0.73 mm, docs/real/PORT_VISION.md).

**Grasp error leak (geometric test, `relvis_geom.py`, `geom_sensitivity.json`).** Real camera, panel pose and tip of the 8 hand-overs; shell model from the measured protrusion and the spec shell; port_vision's arithmetic. Maximum error change over the 8 poses:

| Grasp error | across | up | gap (mm) |
|---|---|---|---|
| shift 3 mm (any axis) | 0.04 | 0.04 | 0.02 |
| pitch 5 deg | 0.00 | 0.26 | 0.04 |
| yaw 5 deg | 0.30 | 0.44 | 0.24 |
| roll 5 deg | 0.11 | 0.59 | 0.29 |

A port-only measurement with the tip from the in-hand estimate passes the in-hand error 1:1. Limits: this test does not include the image detector; a shift of more than about 10 mm takes the shell out of `PLUG_ROI` (refusal, not a wrong number). The model's constant offset against the empirical constants is (0.0, -0.2, +0.7) mm, the same on all 8 poses.

**Real frames: direct against "port + mount + in-hand"** (`relvis_compare.py`, `compare_inhand.json`). Same panel pose; the alternative takes the tip from the camera mount and the grasp model.
- The direct reading at rest (last 6 frames per run): SD 0.03-0.18 across, 0.08-0.35 up, 0.04-0.21 gap (mm).
- Direct minus alternative, nominal grasp, n = 8 runs: mean -2.6, -6.5, +1.2 mm; SD over runs 0.5, 1.7, 0.8 mm.
- With the run's in-hand net estimate, n = 3 runs: mean -2.6, -4.2, -2.4 mm; SD 0.06, 0.7, 0.2 mm.
- The direct reading led to real seats: full9 handed over at (-0.3, -0.6, 5.4) and seated. If the alternative were right, the plug would have been 2.6 mm across and 6 mm up off at that hand-over. So the direct path is the one to trust. The constant difference is the mount and grasp model; the run-to-run SD is grasp variation that only the image sees.

**Roll in the image is too noisy.** The plug angles still use a fixed mount and the mean grasp. I fitted the shell's sleeve-end edge against the panel axis in the same image (`relvis_roll.py`). Against the collect sweep's FK roll labels (one grasp, n = 88 frames, label range -3.0 to +3.3 deg): slope 0.76 [0.21, 1.34] (bootstrap 95 %), residual SD 3.9 deg per frame. On the 8 hand-overs the image roll reads +3.7 deg (SD over runs 2.0) against the align stage's -1.9 deg. The image roll cannot settle the open roll question. Do not correct roll from it.

**Deploy status.** No change needed: the direct measurement is already the align source and the prior. No new config. Not tested: port_vision on sim frames (no sim render of the real wrist view exists), and with a different grasp roll.

**For the rig session (no extra motion needed for the first two):**
1. Keep every align raw frame (the runs already log them). The server repeats `relvis_compare.py` on the new runs.
2. Operator check per run, side clip at the hand-over: is the plug in line with port 2? This is the only independent check of the direct reading.
3. To settle the roll: at the stop, turn joint 6 by +3 deg and -3 deg (one go each), 10 frames still at each. The image roll and the align stage's rv z must both change by 3 deg. This needs the operator's approval as a motion.

## Insertion retry, speed, backlash preload (2026-11-08)

Details: [RETRY_SPEED_20261108.md](RETRY_SPEED_20261108.md); runbook block D in [NEXT_RIG_20261109.md](NEXT_RIG_20261109.md).

- Retry loop (`insert.retry`, `bimanual_align7_retry`): back out to a 15 mm gap, align again with the servo, K4 again, at most 2 retries, operator prompt each time. Dry run with a 4 mm forced miss: the code path works; the second alignment then stopped under the press because the stall had turned the plug 2° in the hand. A stall can change the grasp; a retry does not fix that.
- Speed: on the rig the step alignment took 39 s of a 200 s run (`align_full9`); the servo is 4-5 s in the sim. Insertion limit 10 → 6 s per attempt (K4 sim seat p50 1.2-1.9 s).
- A faster servo (vmax 12) was best in AlignTask (100.0 % [98.5, 100], n = 256), but AlignTask has no press. In the full-chain dry runs it hunted under the press and hit the time limit in 3 of 6. Kept vmax 8. Higher gains hunt in both (10 Hz vision, about 0.1 s latency).
- Backlash preload (`servo.preload_mm` 1.0, `bimanual_align7_preload`): servo to 7 mm, then 1 mm toward the port. Dry runs 4/4 aligned and seated; tip moved 0.6-0.9 mm.
- Open: the dry-run release pulls a seated reader out in 5 of 9 seated runs, the reference config included.
- Correction: K4 is not memoryless (history 30 frames, 0.6 s). The RMA exploration tests an explicit estimator on top of that.
