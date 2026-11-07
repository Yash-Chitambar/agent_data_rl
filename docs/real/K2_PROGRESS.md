# K2 contact policy and alignment: progress log

Running log of the work after [CONTACT_ALIGN_PLAN_20261107.md](CONTACT_ALIGN_PLAN_20261107.md). Newest entries at the top of each section. Every number has its n; sim numbers are sim only.

## Operator decisions (2026-11-07)

- K2 spec: yes, including the climb penalty.
- Real-world learning: allowed in this form: run the real policy, learn from its failures, fix in the sim. No hundreds of manual demos.
- π0.5 / VLA: dropped for now.
- Alignment: explore both the continuous vision servo and an RL alignment policy.
- No rig until Monday 2026-11-09. Work on the policies until then.

## Learnings

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

### Runs

| Run | What | Status |
|---|---|---|
| `A1-align-rl-s0` | PPO from scratch on AlignTask, 2048 worlds, curriculum start x0.5 then x1.0, 60M steps | stopped at ~8M (env had the reset transient); `runs/A1-align-rl-s0_old_transient` |
| `A2-align-rl-s0` | as A1, env with the 0.5 s settle | training |
