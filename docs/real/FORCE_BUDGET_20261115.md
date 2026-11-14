# Force budget of the final push, and a small push plant (server task, 2026-11-15)

Task: use the right arm as its own force sensor. Estimate the force at the plug on every logged push and K4 insertion, and write the force budget of the push. Then fit a small parametric push plant to the logged pushes, and evaluate K4 and K7 in it.

**Nothing here has run on the arm.** Real numbers come from the logs of 2026-11-10, 11-11 and 11-13 (read-only). The sim numbers are sim only. With n this small, the tables give ranges, not means.

## Short answer

- The arm is a usable axial force sensor at the push pose. A free-space torque model (gravity, friction, inertia; linear) gives a held-out residual of **1.29-1.35 N p95 along the port axis** (by date, n = 6-12 runs per date). Static joint friction adds at most **±0.6-0.7 N**.
- **The arm delivers the commanded force at the plug.** Force pushes: 10 N commanded, 10.4 N estimated; 14 N commanded, 14.3 N estimated (n = 1 each). The 15 mm position pushes gave 7.4-9.8 N axial (n = 8).
- **More axial force did not give full depth.** The 4 fully-in runs had 9.0-9.7 N axial. The 14 N force push had 14.3 N and left a 2 mm band.
- **One logged signal separates the labels: the lateral force.** Fully in (n = 4): 5.4-5.9 N lateral (across -3.8..-4.7 N, down 3.4-4.2 N). Short pushes (n = 4): 1.8-3.3 N. This is fully confounded with the push type (all 4 fully-in runs used the position push). FK depth, compliance and the K4 contact force do not separate the labels.
- **Force budget.** The present force push stops at about **14-16 N**: the 4 mm advance watch (axial compliance 0.24-0.33 mm/N, n = 10) and the joint-1 torque cap of the push law (6 N m: 15.6-16.1 N, n = 12) bind at about the same force. 15, 20 and 25 N each need a looser watch or cap: **operator approval**.
- **Plant fit.** A 1-D plant (series compliance + Coulomb play + an in-only plug with a linear mating resistance) cuts the held-out trace error from 1.33 / 1.73 mm to 0.14 / 0.35 mm RMS (the two force pushes). **The mating force is not identifiable from FK:** the loss over the mating threshold R0 is flat (0.233-0.246 mm from 0 N to "never moves"). The compliance is identified: 0.13-0.24 mm/N plus 0-2 N of play (bootstrap 5-95 %).
- **Sim, fitted plant (n = 1024 per cell):** the compliance reproduces the held-out push advance (14 N: 3.2-4.6 mm sim, 4.03 mm rig). With a 10 N mating force K4's FK seat detector fires with the plug 1.4-2.6 mm short in 99.8 % of detections, and a 14 N axial push brings only 32 % of those worlds to full depth (jam, tilt). K7 seats 97.8 % by itself in that plant.
- **New config inside today's limits:** `bimanual_align10_forcevec` (force push with the fully-in force vector: axial 9.5 N + lateral [-4.2, -3.8] N). New code, off by default: `insert.final_push.lat_bias_n`, `BimanualCfg.right_series_stiffness`.

## 1. Free-space torque model of the right arm

Script: `scripts/real/force_budget.py fit` -> `results/real_align/force_budget/free_model.json`.

Data: `ticks.npz` of every non-dry run with ticks (25 runs; 102,660 contact-free ticks). Contact-free = stages hold, transport, approach, align (plug at least about 5 mm from the port) and home; not the first 5 ticks of a stage. `eff` is the measured motor torque. Believed angles = encoder + rig joint offsets.

Model per joint j (linear in its parameters, one least-squares fit with a weak pull to the Menagerie masses):

- gravity: 4 parameters (m, m·c) per body for link_2..link_6 (link_6 lumped with the gripper, camera and reader), shared by all joints;
- inertia: the Menagerie M·qdd + C·qd, one scale per joint;
- friction: Coulomb `Fc tanh(qd / 0.02)` + viscous + offset, per joint.

Force mapping: `F = a · tau`. Two maps at the reference push pose (median of the 14 logged seat poses):

- `tip`: a pure force at the plug tip, weighted least squares over the 6 joints (no moment). This is the main estimate.
- `wrench`: the full wrench at the TCP, `J^T [F, M] = tau` solved exactly (as `springback_timeline.F_eff`). The moment is free, so joint 4-6 noise enters with 4-4.5 N per N m.

Held out by date (train on the other two dates):

| Held-out date | Runs / ticks | Residual rms per joint, N m (j1..j6) | Axial, tip map: p50 / p95 \|F\| (N) | Axial, wrench map: p95 (N) | Lateral (across / up), tip: p95 (N) | Nominal Menagerie + offset: axial p95 tip / wrench (N) |
|---|---|---|---|---|---|---|
| 2026-11-10 | 12 / 61,176 | 0.21, 1.26, 0.73, 0.085, 0.076, 0.072 | 0.48 / 1.35 | 2.03 | 0.85 / 0.87 | 1.87 / 3.25 |
| 2026-11-11 | 7 / 28,569 | 0.21, 1.18, 0.64, 0.086, 0.090, 0.069 | 0.50 / 1.33 | 1.96 | 0.85 / 0.97 | 1.84 / 3.01 |
| 2026-11-13 | 6 / 12,915 | 0.21, 1.12, 0.58, 0.081, 0.083, 0.070 | 0.43 / 1.29 | 1.94 | 0.80 / 0.94 | 2.11 / 3.92 |

- Static ticks only (no joint above 0.01 rad/s): axial p95 1.12-1.28 N (tip map).
- Joint 2 has the largest residual (1.1-1.3 N m rms; fitted Coulomb 2.1 N m: stick-slip). At the push pose joint 2 hardly loads the port axis (0.03 N per N m, tip map), so this does not reach the axial estimate.
- Axial force per N m of joint torque at the push pose, tip map: j1 -2.65, j2 0.03, j3 0.07, j4 3.64, j5 -1.19, j6 -0.90.
- The research note's refutation test ("held-out residual above 2 N p95 along the port axis") is not met for the tip map (1.29-1.35 N). The wrench map is at the limit (1.94-2.03 N). A learned model was not tried: the linear model passes.
- Fitted Coulomb friction (N m): 0.14, 2.10, 1.28, 0.03, 0.02, 0.02. Fitted link_6 lumped mass 0.89 kg (Menagerie 0.81 kg). The gravity fit of 2026-11-06 found a 0.99 kg payload against the 0.5 kg in the driver: the driver under-compensates the hand, so a hand at rest on the plug presses it down.

## 2. Force at the plug on the logged insertions and pushes

Script: `scripts/real/force_budget.py contact` -> `contact.json`, traces `contact_traces.npz`. Model held out by date (the run's own date is not in its fit). Window: last 0.2 s of the push under load. "Friction band": the axial force range if every joint's static friction sits anywhere in ±Fc.

| Run | Label | Push | K4 max axial during the insertion (N) | Commanded at end (PD, N) | Estimated axial at end (N) | Across / up (N) | Advance at end (mm) | Load compliance (mm/N) |
|---|---|---|---|---|---|---|---|---|
| kjrc_1 | short | none | 4.2 | - | - | - | - | - |
| kjrc_2 | short | none | 4.8 | - | - | - | - | - |
| kjrc_3_push | short | position | 3.4 | 4.0 | 3.7 | -1.6 / -2.0 | 0.97 | 0.28 |
| k4_1 | short | position | 6.8 | 4.0 | 4.2 | -1.7 / -2.8 | 0.65 | 0.11 |
| k4_3 | full | position 15 mm | 7.2 | 8.9 | 9.1 | -3.8 / -4.2 | 2.16 | 0.29 |
| k4_4 | - | position 15 mm | 6.4 | 7.6 | 7.4 | -2.9 / -2.0 | 2.80 | 0.26 |
| k4_5 | full | position 15 mm | 6.5 | 9.1 | 9.1 | -4.1 / -3.4 | 2.58 | 0.33 |
| k4_6 | - | position 15 mm | 6.7 | 9.3 | 9.1 | -3.7 / -3.3 | 2.49 | 0.29 |
| demo_1 | short | skipped | 5.9 | - | - | - | - | - |
| demo_2 | - | position 15 mm | 5.7 | 9.6 | 9.8 | -4.3 / -3.8 | 2.35 | 0.30 |
| demo_3 | - | position 15 mm | 6.0 | 9.2 | 8.9 | -3.6 / -3.6 | 2.39 | 0.25 |
| demo_4 | full | position 15 mm | 6.7 | 10.4 | 9.7 | -4.7 / -3.6 | 2.07 | 0.24 |
| speed075_1 | - | skipped | 5.8 | - | - | - | - | - |
| fastmove_1 | full | position 15 mm | 7.4 | 9.1 | 9.0 | -4.1 / -3.9 | 2.02 | 0.29 |
| all9_full_1 | short | force 10 N | 5.2 | 10.0 | **10.4** | -0.1 / -2.1 | 2.61 | 0.29 |
| all9b_full_1 | short | force 14 N | 6.4 | 14.0 | **14.3** | -0.6 / -1.7 | 4.03 | 0.25 |

Friction band at the push pose: ±0.61-0.72 N (n = 14 seats). The wrench map and the local baseline (effort minus its seat-hold mean, as DEPTH_SPRINGBACK) agree with the tip map within 0.1-1.2 N.

- **Does the arm deliver 10 / 14 N at the plug?** Yes, within the model error: 10.4 and 14.3 N (±1.3 N model p95, ±0.6 N friction band; n = 1 each). Joint friction does not eat the push.
- K4 itself pushes 3.4-7.4 N during the insertion (n = 14). With the 0.22 mm/N compliance of section 4, K4's own load deflects the arm by 0.8-1.6 mm. So the encoder-FK seat check sees the hand, which is up to about 1.6 mm deeper than the plug. This explains part of the 2.5 mm "early seat" of FULL_DEPTH_20261113.
- Position push: the hand sinks 3.8-7.7 mm and moves 1.4-1.9 mm across while the PD target stays on the line (n = 8). The arm then presses the plug down by 2.0-4.2 N and across by 2.9-4.7 N. The force push holds the hand on its line with the lateral spring: down 1.7-2.1 N, across 0.1-0.6 N.

### Does a logged signal separate fully in from short?

Script: `force_budget.py labels` -> `labels.json`. Labels from DEPTH_SPRINGBACK section 4: fully in n = 4 (demo_4, k4_5, k4_3, fastmove_1), short n = 7 (4 with a push).

| Signal | Fully in (n = 4) | Short | Separates? |
|---|---|---|---|
| K4 max axial force during the insertion (N) | 6.5-7.4 | 3.4-6.8 (n = 7) | no (k4_1 inside) |
| K4 axial force at the seat (N) | 4.2-5.9 | 1.3-5.6 (n = 7) | no |
| Axial force at the push end (N) | 9.0-9.7 | 3.7-14.3 (n = 4) | no threshold (short both below and above) |
| Load compliance (mm/N) | 0.24-0.33 | 0.11-0.29 | no |
| Unload spring-back (mm/N) | 0.16-0.18 | 0.16-0.23 | no |
| Compliance, first / second half of the ramp (stiffness change) | 0.15-0.28 / 0.24-0.36 | 0.22-0.46 / 0.03-0.33 | no |
| Rest advance after the drop (mm) | 0.57-1.12 | -0.02-0.68 | no |
| **Lateral force at the push end (N)** | **5.4-5.9** | **1.8-3.3** | yes, but confounded |
| Across / down at the push end (N) | -3.8..-4.7 / 3.4-4.2 | -0.1..-1.7 / 1.7-2.8 | yes, but confounded |

Plain statement: **like the FK depth and the contact head, the axial force and the compliance do not separate the labels.** The lateral force does, at n = 4 + 4, but every fully-in run used the 15 mm position push at the 2026-11-10/11 PC spot, and the short pushes were the two short position pushes of 2026-11-10 and the two force pushes of 2026-11-13 (another PC spot). It cannot tell "the force vector matters" from "the position push matters" or "the spot matters". It is a hypothesis to test on the rig, not a detector.

## 3. Force budget of the force push (`bimanual_align10_all`, 14 N)

Per-run values at the logged seat poses (n = 12 pushes unless stated).

| Item | Setting | Data | Largest axial force before it stops the push |
|---|---|---|---|
| Axial compliance of the loop (arm, grasp, PC side), FK | - | 0.24-0.33 mm/N loading (n = 10 pushes >= 7 N); unload 0.16-0.24 mm/N | - |
| Advance watch `max_advance_mm` | 4.0 mm | 14 N push reached 4.03 mm (event 3.91 mm) | 12.1-16.7 N (4 mm / 0.24-0.33); force pushes 13.9-16.1 N (n = 2) |
| Push-law torque cap `tau_max_nm` (joint 1 binds) | [6, 6, 6, 2.5, 2, 2] N m | joint 1 0.37-0.38 N m per N; joint 5 0.09-0.10 N m per N | 15.6-16.1 N (j1); j5: 20-22 N |
| Guard `tau_cont_nm` (20 s total above) / `tau_peak_nm` (0.1 s) | 9 / 12 N m (j1-3); 3 / 5 (j4-6) | j1 PD at 14 N: 5.0 N m | j1 continuous: about 23.6 N; peak about 31 N; j5 continuous: about 30 N |
| Gripper watch `max_grip_change` | 0.02 | 0.0001-0.0202 per push, not monotonic in force (k4_4: 0.0202 at 7.4 N; all9b: 0.0097 at 14.3 N) | unknown; risk grows with force |
| Left press / PC (`max_left_mm` 2 mm) | press about 19 N down (est.) | left hand 0-0.25 mm at up to 14.3 N | at least 14.3 N; above that unknown (PC feet friction unmeasured) |
| Lateral spring cap `lat_f_max_n` / `max_lat_err_mm` | 8 N / 4 mm | off-line at most 2.4 mm (dry runs) | not a force limit on the axis |
| Press time `hold_max_s` | 60 s | press about 23 s on all9b_full_1 | 25 N adds about 7 s (ramp up and down at 3 N/s) |

**The largest force of the present push before a gate stops it: about 14-16 N** (advance watch and joint-1 cap together; n = 2 force pushes, n = 12 seat poses). The 14 N setting sits at the limit (4.03 mm against 4 mm).

Settings that could reach 15, 20, 25 N (compliance 0.24-0.33 mm/N, joint 1 0.38 N m per N, joint 5 0.10 N m per N):

| Target | `force_n` | `max_advance_mm` | `tau_max_nm` j1 / j5 | Guards | Approval |
|---|---|---|---|---|---|
| 14 N (today) | 14 | 4.0 (used 3.4-4.6 at 14 N) | 6 / 2 | inside | approved 2026-11-13 |
| 15 N | 15 | 5.0 (needs 3.6-5.0) | 6.0 (5.7 used, 5 % margin) -> 6.5 | inside | **yes** (force, watch) |
| 20 N | 20 | 7.0 (needs 4.8-6.6) | 8.0 / 2.5 | inside (j1 7.6 < 9) | **yes** (force, watch, cap) |
| 25 N | 25 | 9.0 (needs 6.0-8.3) | 10.0 / 3.0 | **j1 9.6 N m above `tau_cont` 9** (rated continuous of the DM4340); j5 2.5 of 3 | **yes**, and a guard change |

Inside today's limits no setting reaches 15 N: `force_n` is capped by the approval at 14 N, and at the upper compliance the 4 mm watch stops even 14 N. Before any approval for more force, two facts argue against it: (a) the fully-in runs had 9.0-9.7 N and the 14.3 N push stayed short; (b) the gripper and PC limits above 14 N are unknown. The scale reading of `NEXT_RIG_20261115` step 0 (force to push the plug home by hand) is the number that decides it. At 20-25 N the loop stores 5-8 mm of elastic deflection, which the slow unload must release.

## 4. Small parametric push plant (Part 2)

### Fit on the logs (1-D, quasi-static)

Script: `scripts/real/plant_fit.py` -> `plant_fit.json`, `plant_fit_traces.npz`; figure `push_traces.png` (`scripts/real/force_budget_fig.py`).

Input: the estimated plug force F(t) (section 2). Output: hand advance by encoder FK, from the push start to the release (load, drop, rest with the jaws closed). Parameters: series compliance c_s, Coulomb play f_h of the series element, mating resistance R(x) = R0 + s_R x of an in-only plug (the port holds a plug; by hand a plug 1-3 mm out stays put), travel limit x_max. Exhaustive grid (177,072 sets). Train: the 10 position pushes of 2026-11-10/11. Held out: the 2 force pushes of 2026-11-13.

| | Train RMS (mm, n = 10) | Held-out all9_full_1, 10 N (mm) | Held-out all9b_full_1, 14 N (mm) |
|---|---|---|---|
| Before: present sim (0.03 mm/N, no play, no mating resistance) | 1.02 | 1.33 | 1.73 |
| After: fitted plant | 0.23 | 0.14 | 0.35 |

Best: c_s 0.22 mm/N, f_h 1.5 N, R0 5 N, s_R 8 N/mm, x_max 0.5 mm. Bootstrap over the training pushes (200 resamples), 5 / 50 / 95 %: c_s 0.13 / 0.18 / 0.24 mm/N; f_h 0 / 1.5 / 2.0 N; R0 1 / 4 / 7 N; x_max 0.5 / 1.0 / 1.0 mm; mating force at the end of the travel 8.5 / 9.0 / 12.0 N.

**The bootstrap spread understates the uncertainty of the mating force.** The loss profile over R0 is flat: 0.233 mm at R0 5 N, 0.246 mm for R0 from 9 N to "never moves" (99 N). Within 10 % of the best loss, R0 spans 0-99 N and the plug travel at 14 N spans 0-3 mm. FK sees the hand, and the plug travel trades against the play of the series element. **Fitted mating force: not identified by these logs (0 N to more than 14 N all fit).** The compliance is identified.

### The plant in the sim

New option, off by default: `BimanualCfg.right_series_stiffness` (`blind_insertion/envs/bimanual_runtime.py`): per-joint stiffness between motor and link. The PD acts on the motor angle (link gain kp k / (kp + k)), and the encoder reads link + PD torque / k. Fitted value: joint 1 only, k = 640 N m/rad (0.22 mm/N along the port axis at the push pose). Under the force-push law the plug load stays equal to the command (checked by the fixed point; the law uses the nominal kp, as the rig).

Not in the sim: a friction-type mating force. An explicit force on the 10 g reader, held for 20 ms, cannot model Coulomb stick (stable explicit gains give tens of mm/s of creep). The sim keeps the spring-type `mate_force_n`: it pushes a short plug back out when the load drops, so "full depth" is read under load.

### K4 and K7 in the fitted plant (n = 1024 per cell)

Script: `scripts/eval/plant_push_eval.py` -> `plant_eval.json`. Cell real_ho, rig-spot layout, real right-arm lead (as `k7_depth.py`). Per world: K4/K7 insertion; the deploy seat2 detector on the **encoder FK** depth (true depth at the hand-over + FK advance; >= seat - 1 mm, < 6 mm/s, 0.3 s); 0.2 s hold; the 14 N force push of `bimanual_align10_all` (3 N/s, 1.5 s hold, lateral spring); slow unload to 0 N with the jaws closed. Full depth = true depth >= 5.3 mm (seat 5.8 mm). Only a gross reader slip (20 mm, 20 deg) ends an episode.

Seed 6500, 2026-11-14, GPU 4. "comp" = `right_series_stiffness` joint 1 640 N m/rad; "m" = spring mating force peak (N). Rates in % of n = 1024 worlds.

| Policy | Plant | Sim seat (K's own) | FK detector fired | Detector fired with the plug > 1 mm short | True depth at the detector p5-p95 (mm) | Pushed to 14 N | **Full depth under the 14 N load** | Full depth at 0 N, jaws closed (m0 only) | FK advance at 10 N / at the end, p5-p95 (mm) |
|---|---|---|---|---|---|---|---|---|---|
| K4 | rigid, m0 | 98.7 | 98.9 | 0.0 | 5.65-5.77 | 1002 | 97.8 | 91.6 | 0.12-0.75 / 0.62-1.73 |
| K4 | comp, m0 | 99.0 | 98.1 | 0.1 | 5.63-5.76 | 963 | 93.9 | 70.0 | 1.89-2.75 / 3.30-4.58 |
| K4 | comp, m10 | **20.0** | 93.1 | **99.8** | 3.16-4.41 | 738 | **23.2** | - | 1.81-5.75 / 3.59-7.79 |
| K4 | comp, m20 | 0.0 | 68.6 | 100 | 2.49-3.21 | 297 | 0.0 | - | 2.03-4.49 / 3.67-7.00 |
| K7 | rigid, m0 | 99.6 | 99.6 | 0.0 | 5.69-5.79 | 1015 | 98.9 | 95.7 | 0.11-0.72 / 0.61-1.65 |
| K7 | comp, m0 | 98.9 | 98.9 | 0.1 | 5.70-5.76 | 1010 | 97.9 | 93.3 | 1.82-2.60 / 3.19-4.49 |
| K7 | comp, m10 | **97.8** | 98.7 | 0.1 | 5.64-5.74 | 1002 | **56.1** | - | 2.38-6.49 / 3.94-8.38 |
| K7 | comp, m20 | 0.0 | 98.4 | 100 | 3.83-4.51 | 1002 | 0.0 | - | 1.98-4.99 / 3.62-7.51 |

Reading (sim only):

- **The fitted compliance reproduces the held-out real pushes in the full sim.** FK advance at 10 N 1.8-2.8 mm (rig 2.61 mm), at 14 N 3.2-4.6 mm (rig 4.03 mm). The rigid sim gives 0.1-0.7 and 0.6-1.7 mm.
- Compliance alone (no mating force) does not change the seat rates (K4 99.0, K7 98.9 %; rigid 98.7, 99.6 %).
- **Compliance + a 10 N mating force reproduces the rig's early seat for K4.** The FK detector fires with the plug 1.4-2.6 mm short in 99.8 % of K4 detections (the encoder sees the deflected hand, not the plug). Then the 14 N axial push gets only 32 % of the pushed worlds to full depth; the rest end as jam (644) or tilt (247). This matches the rig pattern (seat detected, push at 10-14 N, a 1-2 mm band).
- K7 (trained with a 0-10 N mating force) pushes through: 97.8 % sim seat, detector at 5.64-5.74 mm. The 56 % after the push is mostly a model artefact: in the 0.2 s measured-pose hold the spring pushes the seated plug back out and it tilts; a real port holds the plug. Read the K7 m10 push row as "not valid", not as a K7 failure.
- m20 (mating force above the 14 N push): 0 % by construction. The logs cannot exclude this case (section 4).
- Full depth at 0 N with the jaws closed falls with the compliance (K4 91.6 -> 70.0 %): with no port friction in the sim, the hand's stored deflection carries the plug back when the load ramps down. On the rig the port holds the plug; the sim cannot test the unload.
- Consequence for the rig: if the real mating force is near 10 N, K7 (or K6d) seats by itself where K4 stops early, and a pure axial push does not fix a short K4 seat. This agrees with the logs: more axial force (14 N) did not help.

## 5. What changed (all off by default or new files)

| File | Change |
|---|---|
| `scripts/real/force_budget.py` | new: free-space model (`fit`), force at the plug (`contact`), label check (`labels`) |
| `scripts/real/plant_fit.py`, `scripts/real/force_budget_fig.py` | new: 1-D plant fit, figure |
| `scripts/eval/plant_push_eval.py` | new: K4/K7 + FK seat detector + force push + unload in the fitted plant |
| `blind_insertion/envs/bimanual_runtime.py` | `BimanualCfg.right_series_stiffness` (default None: rigid, as before) |
| `deploy/bimanual/insert.py` | `final_push.lat_bias_n` (absent = [0, 0]: as before) |
| `deploy/config/bimanual_align10_forcevec.json` | new: `bimanual_align10_all` + `force_n` 9.5 + `lat_bias_n` [-4.2, -3.8] |
| `results/real_align/force_budget/` | JSON results, traces, figure. Tick cache (38 MB, regenerable) in `runs/force_budget/` |

`bimanual_align10_forcevec`: conventions check `FAILED: none`. Dry runs: SimIO, K4, `--speed 0.75`, GPU 4, 2026-11-14 (they check the code path, not a rate):

| Seed | Align | Push | Unload (jaws closed) | Seated after the release |
|---|---|---|---|---|
| 7041 | OK, 11 cycles, 7.9 s | 9.5 N reached, adv 0.60 mm, off line 2.00 mm, left 0.51 mm, no stop | 9.5 -> 0 N in 3.18 s, no stop | yes |
| 7042 | OK | 9.5 N, adv 0.37 mm, off line 2.40 mm, no stop | no stop | no (the known sim release effect) |
| 7043 | OK, 15 cycles, 10.1 s | 9.5 N, adv 0.58 mm, off line 2.20 mm, no stop | no stop | yes |
| 7044 | OK, 39 cycles, 28.9 s | 9.5 N, adv 0.27 mm, off line 2.24 mm, no stop | no stop | yes |
| 7045 | OK, 9 cycles, 4.1 s | not reached: K4 insertion guard stop (`guard_bound`, hand 8.4 mm down at step 175), 20 mm back-off | - | - |

With the lateral bias on, the hand stayed 2.0-2.4 mm off its line (watch 4 mm; n = 4). The sim arm is stiff, so the advance is 0.3-0.6 mm (rig: about 2.6 mm at 10 N). Logs: `data/real_runs/2026-11-14/full_*_dry_forcevec_*` in the worktree.

## 6. Limits

- n is tiny: 2 force pushes, 10 position pushes, 4 fully-in labels. Every label result is confounded with push type, date and PC spot.
- The force estimate assumes the free-space model holds in contact. Joint stick at a static pose can hide up to ±0.6-0.7 N (friction band); the model p95 is 1.3 N.
- The tip map assumes a pure force at the plug tip. A moment at the port (plug tilt) moves the estimate.
- The sim compliance sits in joint 1 only; the real compliance is somewhere in the loop (joints, links, grasp, PC). Lateral compliance is not fitted.
- The mating force is not identified. The sim mating force is a spring, not friction.

## 7. Rig runbook addition (after `NEXT_RIG_20261115.md` steps 2 and 3)

No extra approval: `bimanual_align10_forcevec` applies less axial force than the approved 14 N push and a total force inside what the approved position push applied (10.5-11.4 N, n = 4). The operator may still want to see the vector first.

Use it only when step 2 (force push, 14 N) leaves a band and step 3 (position push) gives no band. Then this run tells whether the force vector alone does it (a force-controlled push with known force), or whether the position push's hand motion is needed.

```bash
C=deploy/config/bimanual_align10_forcevec.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C | tail -1        # FAILED: none
.venv/bin/python -m deploy.run_bimanual --dry-run --stage full --align-mode full --seed 7043 --config $C --name dry_forcevec
#   prints "final push (force 9.5 of 9.5 N) ...", "unload: load 9.5 -> 0.0 N ..."
.venv/bin/python -m deploy.run_bimanual --stage full --align-mode full --speed 0.75 --demo --config $C --name fvec_1   # then fvec_2
```

- Pass per run: push reaches 9.5 N with no watch stop; off-line under 4 mm; operator: no band, reader straight.
- Stop: any watch stop twice; the reader turns in the jaws; the PC moves; anything looks wrong.
- Read-out: no band on 2 of 2 -> the force vector is the cause; keep the force push with the bias. Band on 2 of 2 -> the hand motion of the position push matters; keep the position push.
- Server, after the session: `PYTHONPATH=. python scripts/real/force_budget.py all` on the new run folders (add them to `DATES`/`LABEL`) gives the force at the plug and the vector per push.

10-minute check that would settle the budget (needs the rig, no motion of the arm): the step-0 scale reading. If the plug needs more than about 14 N to go home by hand, no setting inside today's limits can seat it, and the 15-25 N rows above go to the operator with that number.
