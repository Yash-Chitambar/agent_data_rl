# Measured mating force in the sim: depth, seat detector, push and push angle (server task, 2026-11-19/20)

Task: put the operator's kitchen-scale insertion force into the sim. Redo the force-budget cells of `FORCE_BUDGET_20261115.md` with it. Add the push-angle study (operator question, 2026-11-19) and the angle dependence of the resistance (operator measurement, 2026-11-20).

**Nothing here has run on the arm.** All rates are sim only, with the fitted compliance plant of FORCE_BUDGET (joint 1 series stiffness 640 N m/rad = 0.22 mm/N along the port axis). Rates are % of n worlds with the Wilson 95 % interval. MuJoCo Warp is not bitwise deterministic: compare distributions.

Real measurements used (by hand, PC on a kitchen scale, port 2):

| Date | Push | First contact | Fully in |
|---|---|---|---|
| 2026-11-19 | straight, along the port axis | about 200 g (2.0 N) | about 650 g (6.4 N) |
| 2026-11-20 | about 5 deg off the port axis | about 200 g (2.0 N) | about 1100 g (10.8 N) |

A hand push can overshoot, so the peaks may be high. The extraction force is not measured.

## Short answer

- **More push force is not needed.** The resistance at full depth is 6.4 N straight and about 10.8 N at 5 deg. The robot delivers 10.4 / 14.3 N at the plug in the force pushes and 7.4-9.8 N in the 15 mm position push (FORCE_BUDGET, n = 1 / 1 / 8). In the sim the 14 N push reaches full depth on 98.3-98.6 % of the K7 worlds.
- **The 1-2 mm band comes from the policy's early stop plus a push that does not finish, not from too little force.** With the measured profile K4 stalls short in the sim (66.6 % seat, n = 3072) and its FK seat detector fires with the plug 1-2.6 mm short (16.7 % of detections > 1 mm short). The push then finishes only part of these: the 4 mm advance watch stops the 14 N push (the arm's own deflection is 3.1 mm at 14 N, so a plug 1 mm short trips the watch), and the 15 mm position push gives only 8-11 N. K4 at the +30 % profile (8.3 N): 28-60 % of the pushed plugs end 1-2 mm short.
- **K7 fixes the depth by itself.** K7 (trained with a 0-10 N mating force) seats 98.5 % (n = 3072), the detector never fires > 1 mm short (0.0 %, 1 of 3028), and every push ends at full depth with no band (0 of about 1000 per push type). At +/-30 % of the profile K7 stays at 98.3-99.1 % full at rest (n = 1024 per push). This is a strong extra reason to run K7 at the next session.
- **Push angle (operator question).** Two different angles. (a) The push **axis** (where the force points): the chain's ZED axis is within 1 deg on the rig (PUSH_ANGLE), and in the sim a 5 deg yaw or pitch error changes nothing (K7 force14: 98.8 / 98.6 % full against 98.3 % at 0 deg, n = 1024 each). (b) The **gripper** angle against the seated plug: with the 2026-11-20 reading (6.4 N straight, 10.8 N at 5 deg) in the model, K4's 3-10 deg hand turn raises the resistance to 9-13 N. K4 then seats only 13.6 % by itself, and the 14 N push leaves 1-2 mm short on 43 % (n = 1024). `rot_ref 'handover'` (<= 3 deg re-squaring) helps K4 from 25.5 to 32.9 % full; it is not a fix. K7 turns the hand less (2.0 deg p50) and stays at 98.1-98.6 % full with every push.
- **New model option (sim only, off by default):** `InsertionCfg.mate_model = 'friction'` (Coulomb friction along the port axis, does not push a still plug out), `mate_onset_frac` (force at the engagement start), `mate_angle_gain_per_deg` (angle dependence).
- **New deploy option (off by default) and config:** `insert.final_push.axis_source: 'plug'` and `deploy/config/bimanual_align13_wide_k7_plugaxis.json` (conventions `FAILED: none`; dry runs seeds 7043, 7044). **Not recommended for the rig:** the sim shows no gain (K7 98.2 %, n = 1024) and the push axis is not the problem. It is there if the operator wants a push along the plug.
- **No new push or seat setting for the rig.** Inside today's limits nothing in the push helps K4 enough; more force does not help (the advance watch and the jam stop it, not the force). The settings that would help K4 (advance watch above 4 mm) loosen a watch: not proposed (K7 is the better change).

## 1. The model

### Profile

`InsertionCfg.mate_force_n = [peak, peak]`, `mate_start_m` 2 mm (engagement start), `mate_onset_frac` 0.31: zero above 2 mm of depth, 2.0 N at 2 mm, linear to 6.4 N at the seat (5.8 mm). The +/-30 % cells scale the whole profile (4.5 N and 8.3 N peak, same onset fraction).

Where the engagement starts: the scale reads "first contact" as one value, not as a depth. The sim keeps the 2 mm start of `FULL_DEPTH_20261113.md` (the plug shell is in the receptacle opening and meets the contact springs; above that the plug is in the chamfer, where the mjfit geometry gives the contact). This is an assumption; a start at 0.5 mm or 3.5 mm is not tested here (limit).

### Friction, not a spring

The old `mate_force_n` is an explicit spring: it pushes a short plug back out when the load drops. A real port holds a plug. New: `mate_model 'friction'`. Each control step, the plug's free-joint translational dof along the world axis nearest the port axis gets a MuJoCo `frictionloss` equal to the profile force at the live depth (0 outside the opening). MuJoCo solves it with the contacts, so it resists motion both ways and gives no force on a still plug at zero load.

Calibration of the stick (the reason FORCE_BUDGET did not do it): MuJoCo friction is soft. Under a load F below the limit the plug creeps at v = F R / b, with R = (1 - d)/d x invweight0 and b = 2 / (d x timeconst). The 10 g plug's own invweight0 (about 100) gives a damper: creep about 25 mm/s at 5 N. A first try with that default had no effect (K4 seat with a 30 N limit: same as with no mating force, n = 96). With `mate_friction_invweight` 0.5 (the held plug plus the hand, about 2 kg), `mate_friction_solimp` d = 0.99 and `mate_friction_solref` 2 ms, the creep is about 0.03 mm/s at 5 N. Check with a 30 N limit (n = 96): K4 stops at 2.2-3.0 mm, no world seats, no push gets in.

Limits of the model:
- The friction reaction goes to the world, not to the PC (the left-hand press and the PC feet do not feel the 2-6 N).
- One world axis (the port yaw is within +/-1.5 deg of it: cos 0.9997).
- Stick is stiff but not exact (creep about 0.03 mm/s at 5 N).
- The 2-mm engagement start and the linear profile are assumptions.

### Angle dependence (section 5)

`mate_angle_gain_per_deg` 0.14 (x1.69 at 5 deg: 10.8 / 6.4), linear from 0 deg (assumption), held flat above `mate_angle_max_deg` 7.5 (x2.05). theta = angle between the port axis and the reader axis that the jaws hold: the TCP rotation x the reader-in-TCP rotation at the end of the warm-up. It is the "gripper angle" of the hypothesis (a hand that turns against the seated plug, as the operator's rigid hand at 5 deg). The sim K4 turns the reader in the jaws by 7.6 deg p50 between the hand-over and the push (K7: 2.8 deg), close to the 2-7 deg hand turn of the real K4 (PUSH_ANGLE_20261119, n = 16).

## 2. Evaluation

Script: `scripts/eval/mating_force_eval.py` (built on `plant_push_eval.py`). Tables: `scripts/eval/mating_force_table.py`. Results: `results/real_align/mating_force/*.json`. Cell real_ho (rig-spot layout v2, gap 5.5 mm, ring 0.5 mm, real right-arm lead), seed 6600, GPU 3, 2026-11-19/20.

Per world: K4 or K7 insertion; the deploy seat2 detector on the encoder FK depth (true depth at the reset + FK advance; depth >= seat - 1 mm and < 6 mm/s for 0.3 s); 0.2 s measured-pose hold; then one push per world group (n = 1024 per group, same start distribution):

| Group | Law (deploy) |
|---|---|
| force10 / force14 | `final_push_force`: axial load ramp 3 N/s to 10 / 14 N, 1.5 s hold ("still" end 0.15 mm in 0.6 s), lateral spring 3 N/mm (cap 8 N), rotation spring 6 N m/rad (cap 0.5 N m), joint torques scaled under tau_max; then the unload ramp to 0 N with the jaws closed, 0.2 s hold |
| pos15 | `final_push`: target 15 mm into the port at 3 mm/s (smoothstep), orientation kept, 0.5 s hold, then 0.3 s measured-pose hold |
| all | the 4 mm advance watch on the FK hand advance stops the push (then the 0.3 s measured hold, as on the rig) |

Full depth = true plug depth >= 5.3 mm (0.5 mm short of the seat). "At rest" = after the unload / drop, jaws closed, 0 N. With the friction model the rest depth is valid (the old spring pushed short plugs out).

## 3. Results at the measured profile and +/-30 % (straight push axis, no angle dependence)

### Insertion and seat detector

| Policy | Peak (N) | n | Sim seat | FK detector fired | Fired with the plug > 1 mm short | > 0.5 mm short | True depth at the detector p5-p95 (mm) |
|---|---|---|---|---|---|---|---|
| K4 | 4.5 | 2048 | 95.8 [94.8, 96.6] | 98.7 [98.1, 99.1] | 2.2 [1.7, 3.0] | 4.3 [3.5, 5.3] | 5.38-5.75 |
| K4 | **6.4** | 3072 | **66.6 [64.9, 68.2]** | 98.7 [98.3, 99.1] | **16.7 [15.4, 18.0]** | 34.5 [32.9, 36.3] | 4.19-5.72 |
| K4 | 8.3 | 2048 | 19.6 [17.9, 21.3] | 98.4 [97.8, 98.9] | 67.6 [65.5, 69.6] | 89.1 [87.6, 90.4] | 3.70-5.55 |
| K7 | 4.5 | 2048 | 98.9 [98.4, 99.3] | 98.9 [98.4, 99.3] | 0.1 [0.0, 0.3] | 0.1 [0.0, 0.4] | 5.70-5.77 |
| K7 | **6.4** | 3072 | **98.5 [98.0, 98.9]** | 98.6 [98.1, 98.9] | **0.0 [0.0, 0.2]** | 0.1 [0.0, 0.2] | 5.70-5.77 |
| K7 | 8.3 | 2048 | 98.3 [97.7, 98.8] | 98.6 [98.0, 99.1] | 0.0 [0.0, 0.2] | 0.2 [0.1, 0.5] | 5.69-5.76 |

For comparison, FORCE_BUDGET (spring model, n = 1024): K4 at 10 N 20.0 % seat, detector > 1 mm short 99.8 %; K7 97.8 %.

### Push outcome (n = 1024 per cell)

| Policy | Peak (N) | Push | Watch stop (of pushed) | Full under load | **Full at rest** | Rest 0.5-1 mm short | **Rest 1-2 mm short** | Rest > 2 mm short | FK advance at the end p5-p95 (mm) | Commanded axial at the end p5-p95 (N) |
|---|---|---|---|---|---|---|---|---|---|---|
| K4 | 4.5 | force14 | 3.3 | 96.1 | 94.1 [92.5, 95.4] | 1.0 | 0.9 | 0.2 | 3.06-3.85 | 13.95-14.07 |
| K4 | 4.5 | pos15 | 0.0 | 97.1 | 97.1 [95.9, 97.9] | 0.7 | 0.5 | 0.0 | 1.29-1.94 | 8.20-10.88 |
| K4 | 6.4 | force10 | 1.1 | 78.4 | 77.9 [75.3, 80.4] | 11.1 | 8.9 [7.3, 10.8] | 0.6 | 2.01-3.29 | 9.96-10.07 |
| K4 | 6.4 | force14 | 18.6 | 86.0 | 84.4 [82.0, 86.5] | 7.9 | 4.5 [3.3, 5.9] | 0.1 | 3.05-4.09 | 13.00-14.02 |
| K4 | 6.4 | pos15 | 0.2 | 71.5 | 70.9 [68.0, 73.6] | 18.1 | 9.7 [8.0, 11.7] | 0.2 | 1.31-2.12 | 8.06-11.33 |
| K4 | 8.3 | force14 | 46.5 | 45.6 | 44.5 [41.5, 47.6] | 23.5 | 28.3 [25.6, 31.3] | 1.4 | 3.06-4.12 | 11.77-14.00 |
| K4 | 8.3 | pos15 | 0.2 | 13.0 | 12.1 [10.2, 14.2] | 25.2 | 59.7 [56.6, 62.7] | 2.7 | 1.29-2.08 | 7.77-11.09 |
| K7 | 4.5 | force14 | 2.5 | 98.6 | 98.6 [97.7, 99.2] | 0.0 | 0.0 | 0.0 | 3.06-3.83 | 13.94-14.07 |
| K7 | 4.5 | pos15 | 0.3 | 99.1 | 99.1 [98.3, 99.5] | 0.0 | 0.1 | 0.0 | 1.61-2.21 | 9.38-12.14 |
| K7 | 6.4 | force10 | 0.0 | 98.2 | 98.2 [97.2, 98.9] | 0.0 | 0.0 | 0.0 | 2.00-2.51 | 9.98-10.07 |
| K7 | 6.4 | force14 | 1.1 | 98.6 | 98.3 [97.4, 99.0] | 0.0 | 0.0 | 0.0 | 2.97-3.70 | 13.94-14.07 |
| K7 | 6.4 | pos15 | 0.2 | 98.6 | 98.6 [97.7, 99.2] | 0.0 | 0.0 | 0.0 | 1.53-2.11 | 9.36-12.13 |
| K7 | 8.3 | force14 | 0.7 | 98.5 | 98.4 [97.5, 99.0] | 0.0 | 0.0 | 0.0 | 2.94-3.64 | 13.94-14.06 |
| K7 | 8.3 | pos15 | 0.3 | 98.3 | 98.3 [97.4, 99.0] | 0.0 | 0.0 | 0.0 | 1.50-2.04 | 9.20-12.12 |

"Rest short" columns are % of the worlds that reached the rest hold; "Full" columns are % of all 1024. The rest of the 1024 had no detected seat (no push) or a gross slip.

Reading (sim only):
- **The sim reproduces the real pattern with K4 and the measured profile:** seat detected, push to 10-14 N, a 0.5-2 mm band on 12-28 % of the pushed plugs (K4 6.4 N: force14 12.4 %, force10 20.0 %, pos15 27.8 % of the rest worlds 0.5-2 mm short). The real logs: 4 fully in, 7 short (DEPTH_SPRINGBACK).
- **Cause 1, the early stop (detector).** K4 stops where its own push (3.4-7.4 N real, FORCE_BUDGET) meets the rising resistance. The arm deflects 0.22 mm/N, so the FK depth reads the hand 1-1.6 mm deeper than the plug, and seat2 (seat - 1 mm) fires. 34.5 % of the K4 detections at 6.4 N have the plug > 0.5 mm short, 16.7 % > 1 mm.
- **Cause 2, the push does not finish.** The short plugs are not tilted (rest short with plug tilt > 3 deg: 0.0-0.1 %; tilt in the port at rest p50 1.2-2.0 deg). Two push mechanisms leave them short:
  - Force push: the 4 mm advance watch. At 14 N the arm deflects 3.0-4.1 mm by itself (sim; rig 4.03 mm, FORCE_BUDGET), so a plug that still has to travel 1-2 mm trips the watch (K4 8.3 N: 46.5 % watch stops). At 10 N the deflection is 2.0-3.3 mm: fewer stops (1.1 %), but less force margin.
  - Position push: the force is only 8-11 N and the push moves the hand a fixed 15 mm. Against 6.4-8.3 N plus the port contact friction, 18-60 % stay 1-2 mm short.
- **Cause 3, the unload, is not a cause in this model.** Full under load and full at rest differ by at most 2.0 points in every cell: a plug that is home under load stays home after the unload (the port friction holds it; the jaws stay closed). The old spring model could not test this. The jaw opening and the retreat are not modelled.
- **K7 removes cause 1** (detector at 5.70-5.77 mm), so causes 2 do not matter: every K7 push ends at full depth.

## 4. Push-axis angle sweep (K7, force14, profile 6.4 N, no angle dependence)

The push frame (force direction, lateral line, advance watch) is the true port frame turned by the angle about the port y axis (yaw) or x axis (pitch); the sign alternates per world. The hand orientation is unchanged (rotation spring to the push-start orientation). This is an error of the port estimate, the quantity the chain controls.

| Push axis error | n | Watch stop | Full at rest | +angle / -angle (n = 512 each) | Rest 0.5-2 mm short | FK advance at the end p5-p95 (mm) |
|---|---|---|---|---|---|---|
| 0 deg (section 3) | 1024 | 1.1 | 98.3 [97.4, 99.0] | - | 0.0 | 2.97-3.70 |
| yaw 5 deg | 1024 | 4.9 | 98.8 [98.0, 99.3] | 98.8 / 98.8 | 0.0 | 2.77-3.97 |
| pitch 5 deg | 1024 | 1.4 | 98.6 [97.7, 99.2] | 98.6 / 98.6 | 0.0 | 2.96-3.78 |

- **Tolerance on the push axis: at least 5 deg in yaw and in pitch** (K7, 14 N force push, sim, n = 1024 per cell). The only effect is more advance-watch stops at yaw 5 deg (4.9 %, the hand slides 0.4 mm more), and these plugs were already home.
- The real push vector is within 1 deg of the port axis (PUSH_ANGLE, n = 16), so the push axis is not the problem.
- Not run: 1, 2, 3 deg (two runs were lost to GPU memory; at 5 deg there is no effect, so the smaller angles give no new information), the position push and K4 under the axis sweep (time). Limit.

## 5. Angle-dependent resistance: gripper angle, rot_ref 'handover', axis_source 'plug'

Profile 6.4 N x (1 + 0.14 theta), theta = held-reader angle against the port axis (section 1), cap 7.5 deg. Groups: the three pushes on the true port axis; force14 with `rot_ref 'handover'` (max 3 deg, the other worker's option); force14 with the 'plug' axis as deploy computes it (`plugfk0.5`: FK x the reader-in-TCP rotation at the hand-over, 0.5 deg estimate noise). n = 1024 per group.

### K7 (n = 5120, seed 6600)

Sim seat 98.2 [97.8, 98.5] %; detector > 1 mm short 0.1 % (3 of 5056); true depth at the detector 5.69-5.76 mm.

| Push | Watch stop | Full at rest | Rest 0.5-2 mm short | theta at the push start p5-p50-p95 (deg) | theta at rest p50 / p95 (deg) | Full at rest, theta at push 5-7.5 deg |
|---|---|---|---|---|---|---|
| force10 | 0.0 | 98.1 [97.1, 98.8] | 0.0 | 0.8-1.9-4.9 | 2.8 / 6.5 | 41 of 42 |
| force14 | 0.4 | 98.4 [97.5, 99.0] | 0.0 | 0.8-2.1-5.4 | 3.0 / 6.9 | 64 of 65 |
| pos15 | 0.2 | 98.6 [97.7, 99.2] | 0.2 | 0.8-2.0-5.2 | 4.0 / 6.9 | 53 of 53 |
| force14 + rot_ref handover 3 deg | 0.3 | 98.2 [97.2, 98.9] | 0.0 | 0.8-2.1-5.3 | 2.9 / 6.4 | 57 of 57 |
| force14 + axis 'plug' (FK x in-hand) | 0.1 | 98.2 [97.2, 98.9] | 0.0 | 0.8-2.0-5.4 | 3.0 / 6.8 | 54 of 55 |

K7 at theta up to 7.5 deg: the resistance is at most about 12.9 N at the seat (x2.05; 10.8 N at 5 deg). K7 reaches the seat before the push in 98 % of the worlds, so the push only has to hold it. No push type, `rot_ref` or 'plug' axis changes the K7 outcome (all within 98.1-98.6 %).

### K4 (n = 3072, seed 6600)

Sim seat **13.6 [12.4, 14.8] %** (straight-only profile: 66.6 %); detector fired 98.1 %, of which **73.2 % with the plug > 1 mm short** (90.5 % > 0.5 mm); true depth at the detector 3.60-5.55 mm. theta at the push start p5-p50-p95 2.9-6.6-9.8 deg (K7: 0.8-2.0-5.3 deg; real K4 gripper 1.8-7.3 deg, PUSH_ANGLE).

| Push | Watch stop (of pushed) | Full at rest | Rest 0.5-1 mm short | **Rest 1-2 mm short** | Rest > 2 mm short | Rest depth p5-p50-p95 (mm) | Commanded axial p5-p95 (N) |
|---|---|---|---|---|---|---|---|
| force14 | 27.6 | 25.5 [22.9, 28.2] | 25.3 | 43.2 [40.1, 46.4] | 3.4 | 3.86-4.86-5.72 | 13.13-14.00 |
| pos15 | 0.1 | 7.7 [6.2, 9.5] | 19.2 | 63.6 [60.5, 66.5] | 9.0 | 3.69-4.40-5.49 | 7.97-11.13 |
| force14 + rot_ref handover 3 deg | 36.1 | **32.9 [30.1, 35.9]** | 22.9 | 36.6 [33.6, 39.8] | 3.7 | 3.90-5.01-5.74 | 12.58-14.03 |

Full at rest by theta at the push start (force14): 2-3 deg 23 of 28; 3-5 deg 79 of 170 (46 %); 5-7.5 deg 97 of 397 (24 %); > 7.5 deg 50 of 315 (16 %).

Reading (sim only; the angle law is an assumption from one 5-deg reading):
- **With the angle dependence, the K4 hand turn makes the resistance 9-13 N, above what K4 pushes during the insertion and close to what the pushes give.** K4 stops early on most worlds and the push leaves a 1-2 mm band on 37-64 % of them. This is the rig pattern, stronger.
- **The gripper angle matters through the resistance, not through the push axis.** A push axis 5 deg off changes nothing (section 4). A held reader 5-7.5 deg off the axis halves the rate to full depth.
- **`rot_ref 'handover'` helps K4 a little:** 32.9 % against 25.5 % full at rest (Wilson intervals do not overlap; n = 1024 each). It turns the hand back by at most 3 deg, so a hand at 6-10 deg stays at 3-7 deg. It does not fix K4's early stop. It also gives more watch stops (36.1 % against 27.6 %).
- **K7 is the fix:** it turns the hand less (theta p50 2.0 deg against 6.6 deg) and pushes through the resistance itself. With the angle dependence K7 stays at 98.1-98.6 % full for every push.
- The position push is the weakest push here (8-11 N against up to 13 N).

### Resistance against what the robot delivers

Model resistance at the seat (6.4 N x (1 + 0.14 theta)) at the real gripper angles of PUSH_ANGLE (1.8-7.3 deg at the push, n = 16), against the real force at the plug (FORCE_BUDGET):

| Gripper angle (deg) | Resistance at the seat (N) | K4's own push in the insertion, 3.4-7.4 N | Position push 15 mm, 7.4-9.8 N | Force push 10 N (10.4 N) | Force push 14 N (14.3 N) |
|---|---|---|---|---|---|
| 0 (by hand, straight) | 6.4 | partly | yes | yes | yes |
| 1.8 | 8.0 | no | partly | yes | yes |
| 3 | 9.1 | no | partly | yes | yes |
| 5 (by hand, 2026-11-20) | 10.8 | no | no | no | yes |
| 7.3 | 12.9 | no | no | no | yes (1.4 N margin) |

- The 14 N force push is above the resistance at every logged angle; the 10 N push only below about 4 deg; the position push only below about 2-3 deg. A force margin is necessary, not sufficient: the advance watch (4 mm, of which about 3.1 mm is the arm's own deflection at 14 N) and the port contact friction still stop a short plug (section 3).
- K4's own push (3.4-7.4 N) is below the resistance at every angle above about 1 deg. This is why K4 stops short and the FK detector fires early. K7 was trained against a 0-10 N mating force and pushes through.

## 6. How the chain sets the push direction today

- Both pushes take the port frame `chain.port` from the ZED capture (`capture_pc`): x, y and **yaw** from the ZED net (v1 today; zed_v5 in the `align12/13_wide` configs, which also refine x, y with the case-edge fit, `yaw_source` 'gate'), and height, pitch and roll from the scene's reference PC (`scene.port_frame`: a flat PC on its feet). Port vision (wrist) does not update `chain.port`.
- Force push (`final_push_force`): the load is a commanded vector, -z of that port frame, plus a lateral spring that holds the hand on the start line in the port x, y axes. The hand orientation is held by a rotation spring (6 N m/rad, cap 0.5 N m) at the orientation the policy left at the seat. It is not the tool axis.
- Position push (`final_push`): the target moves 15 mm along -z of the port frame, the orientation is kept (IK).
- Real logs (PUSH_ANGLE_20261119, n = 16): the push vector is within 0.2-1.0 deg of the port-vision axis in yaw and 0.0-0.2 deg in pitch; the gripper is 1.8-7.3 deg off (K4 turns the hand up to about 7 deg pitch, 3 deg yaw, 5 deg roll before the seat).

New, off by default: `insert.final_push.axis_source` (`deploy/bimanual/insert.py`, `push_frame`). 'port' (absent) = as before. 'plug' = the plug frame at the push start: the believed TCP FK rotation x the in-hand estimate (`chain.inhand_est6`, rotation vector on the nominal grasp). If it is more than `axis_max_dev_deg` (default 3) from the port estimate, the chain keeps the port estimate. Event `insert_push_axis` logs the deviation. It acts on both push modes. It does not change any force, gate or limit.

Interaction with `rot_ref 'handover'` (the other worker's option, `bimanual_align13_wide_k7_pushangle.json`): `axis_source` sets the force direction and the lateral line (frozen at the push start); `rot_ref` sets the reference of the rotation spring. They are independent code paths and can be on together. Together they are not a consistent pair: with `rot_ref` the hand turns back by up to 3 deg under load, while a 'plug' axis was computed from the hand before that turn. Test them one at a time.

## 7. What changed (all off by default or new files)

| File | Change |
|---|---|
| `blind_insertion/envs/insertion_task.py` | `InsertionCfg.mate_onset_frac` (0: as before), `mate_model` ('spring': as before; 'friction'), `mate_friction_solref/solimp/invweight` (used only by 'friction'), `mate_angle_gain_per_deg` (0: off), `mate_angle_max_deg` |
| `scripts/eval/mating_force_eval.py`, `scripts/eval/mating_force_table.py` | new: K4/K7 + FK seat detector + force10 / force14 / pos15 per world group, push-axis offsets, 'plug' / 'plugfk' axis, 'rotref' groups |
| `deploy/bimanual/insert.py` | `push_frame`, `final_push.axis_source` / `axis_max_dev_deg` (absent: 'port', as before) |
| `deploy/config/bimanual_align13_wide_k7_plugaxis.json` | new: `bimanual_align13_wide_k7_fp_shadow` + `axis_source` 'plug' |
| `results/real_align/mating_force/` | JSON per run, logs |

Not changed: `deploy/bimanual/io_sim.py` (the dry run's `SIM_MATE_FORCE_N` is still the spring model).

`bimanual_align13_wide_k7_plugaxis.json`: conventions check `FAILED: none` (parent `bimanual_align13_wide_k7_fp_shadow`: `FAILED: none`). Dry runs (SimIO, K7, GPU 3, 2026-11-20), which check the code path, not a rate:

| Seed | Result |
|---|---|
| 7043 | edge fit ok, wide_motion ok, K7 seat at 1.84 s; `final push axis: plug axis 0.64 deg from the port estimate (across -0.26, up +0.59); using the plug axis`; 14 N reached, no stop; unload 14 -> 0 N in 3.5 s; true seated after the release |
| 7044 | align stage: step limit at the hand-over, back-off, no insertion (upstream of the push) |
| 7041 | GuardStop: ZED PC estimate rejected (upstream) |
| 7045 | GuardStop: wide_motion, no hold candidate (upstream) |

One seed reached the push; the upstream stops come from the parent config's stages, not from the new option.

## 8. Limits

- Sim only. The compliance sits in joint 1 only; lateral compliance is not fitted.
- The profile shape (engagement at 2 mm, linear) and the angle law (linear, flat above 7.5 deg, on the held-reader angle) are assumptions from three scale readings by hand.
- The grasp in the sim lets the reader turn in the jaws (K4 7.6 deg p50 between the hand-over and the push). The real grasp is not measured for this (PUSH_ANGLE: not measurable at rest).
- One seed, one cell (real_ho); n = 1024 per push group.
- The `still` end and the watch stops are modelled as on the rig; the gripper, left-hand and off-line watches are not.

## 9. Rig runbook

No new config is needed for the next session. Run the planned K7 config `bimanual_align13_wide_k7_fp_shadow.json` (NEXT_RIG_20261117; K7 needs approval A4). The data here are one more reason to give A4.

Ten-minute hand checks (no robot), to make the model better:
1. Scale, by hand, straight: the force at three depths (first contact, half way, home). Write down where in the stroke the force starts (mm from the panel, calipers on the reader neck).
2. Scale, by hand, at about 2 deg and 8 deg off the axis (a protractor card on the panel). This tests the linear angle law (sim: 1.28x at 2 deg, 2.05x from 7.5 deg).
3. Extraction: pull the plugged reader straight out with the scale under the PC (hook or tape), read the force. A value near 2-6 N confirms the friction model (no push-out).

On the robot (only if the operator wants it, after K7 runs):
- `rot_ref 'handover'` (`bimanual_align13_wide_k7_pushangle.json`, the other worker's config, needs approval): in the sim it does not change K7 (98.2 % against 98.4 %, n = 1024). Test it on the rig only if K7 still leaves a band, and then only with K7.
- `axis_source 'plug'` (`bimanual_align13_wide_k7_plugaxis.json`): no sim gain; do not spend rig time on it.

```bash
C=deploy/config/bimanual_align13_wide_k7_plugaxis.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C | tail -1        # FAILED: none
.venv/bin/python -m deploy.run_bimanual --dry-run --stage full --align-mode full --seed 7043 --config $C --name dry_plugaxis
#   prints "final push axis: plug axis 0.64 deg from the port estimate ...; using the plug axis"
```

- Pass per run: K7 seats; the push reaches 14 N with no watch stop; the printed plug axis is < 3 deg from the port estimate; operator: no band, reader straight.
- Stop: the plug axis line says "using the port axis" twice (the in-hand estimate disagrees); any watch stop twice; the reader turns in the jaws; the PC moves.
- Read-out for the session: a band after a K7 seat would contradict the sim (K7 0 of about 1000 per push, every cell). Then log the hand angle at the push (PUSH_ANGLE script) and send the run to the server.

Server, after the session: `scripts/eval/mating_force_eval.py` with the measured profile from the hand checks (`--onset`, `--conds comp_f<peak>`, `--env_more '{"mate_angle_gain_per_deg": g, "mate_start_m": s}'`).
