# Contact and alignment policy: analysis of the plain-PD insertions, plan, next rig session (2026-11-07)

Branch `real/graph-align`. Answers the operator's brief after session 2026-11-06. **Nothing here has run on the arm.** `bimanual.json`, all older configs and all weights are unchanged. New configs: `bimanual_align4.json`, `bimanual_align4_kjrc.json`, `bimanual_align4_precision.json`.

## Short answer

1. **Why the plug stalls.** K-J's observation uses three position features from absolute encoder FK: the TCP position and its Euler angles relative to a fixed nominal pose (`obs[50:56]`). On the rig these read 9-11, 29-31, 4.5-6.6 mm and -4.1, -3.4, 0 deg off nominal at every hand-over (n = 4). That is about 3 sd outside the sim's hand-over spread (sd 6, 11, 6.5 mm; 2.5, 1.0, 2.1 deg). Put the same offsets into the sim observation and K-J drops from 96.6 % to 21.8 % [19.4, 24.4] on a real-like plant, n = 1024. The plant alone (real friction, gravity error, the real hand-over lead) costs only 3-4 points. So the main cause found is the **absolute-FK observation**, not the plant and not the hand-over.
2. **Why the target climbs.** Joint 5's target lead sits at the 50 mrad cap from 0.3 s in every run, real and sim. It is K-J's normal push. In the sim the commanded tip rises 14-16 mm; on the rig 37-40 mm, from extra lead on joints 2-3. In the stalls the policy acts as if the plug is in: its contact head says "inside" from t = 0, before any contact. In the sim the same head is 96 % right.
3. **Fix to test now, no retraining:** `insert.obs_recenter` sets those six features to the sim's hand-over mean at the hand-over and holds the shift. The policy then sees only motion relative to the hand-over. Sim, real-like plant, real offsets: 21.8 % without it, **89.4 % [87.3, 91.1]** with it (n = 1024 each). Config `bimanual_align4_kjrc.json`.
4. **Fix to train (K2):** K-J with relative position features, wider angle and plant randomisation, an anti-climb term, and later the controller in the plant. Spec below. It needs your yes on the open questions at the end.
5. **Next rig session:** controller alignment with the 18 mm approach (align mode only), then K-J with the re-centre on plain PD, a side camera on every insertion. Commands and stop rules below.

## 1. The four plain-PD insertions against the sim

Real data: `scripts/real/kj_real_ref.py` (tip and target against the hand-over, port axes) and `scripts/real/kj_contact_view.py` (contact head replayed on the logged observations). Sim: `scripts/eval/kj_real4.py`, `kj_obs_shift.py`, `kj_contact_sim.py`; results in `results/real_align/kj_*.json`.

**Real traces (FK, port axes, relative to the hand-over):**

| Run | Outcome | Gap mm | Lateral mm | FK depth at 0.6 / 1.0 s | Target up max |
|---|---|---|---|---|---|
| 11-03 align_full2 | seated 1.86 s | 5.82 | 0.38, -0.62 | 2.0 / 6.7 | 39.6 |
| 11-03 align_full5 | stalled, guard 3.74 s | 5.27 | 0.35, 0.36 | 1.9 / 1.9 | 40.3 |
| 11-06 align_full9 | seated 2.2 s | 5.45 | -0.28, -0.28 | 3.1 / 7.4 | 37.7 |
| 11-06 align_full10 | stalled, guard 4.3 s | 5.38 | 0.26, 0.29 | 1.9 / 2.9 | 39.1 |

- All four reach about 2 mm FK depth by 0.5-0.6 s. The seats then go on to 7 mm by 1.0 s. The stalls stay at 2-3 mm.
- The seats read 8.8-9.4 mm FK depth at the end; the sim seat is 5.4-5.7 mm. The arm reads about 3.5 mm too deep under the push. This is flex the motor-side encoders cannot see (the known limit). So 2-3 mm FK depth in a stall is probably the plug at the port mouth or on the case.
- Gap, lateral offset and applied pitch turn do not separate seats from stalls. Vision roll at the hand-over: -1.5, -1.7 deg (seats), -1.9, -2.2 deg (stalls). n = 2 each: a hint only.
- Joint 5's lead is at the 50 mrad cap from 0.3 s in all four, and in the sim dry runs. The extra climb on the rig comes from joints 2-3: lead -30 to -50 and +20 to +30 mrad, against about ±10 in the sim.

**Sim from the same hand-over (gap 5.5, ring 0.5 mm, angles from the chain pool), n = 1024 per row:**

| Condition | Plant: training | Plant: real-like (friction x1.3-2.0, gravity error 12 %) |
|---|---|---|
| sim hand-over lead | 99.3 % [98.6, 99.7] | 97.3 % [96.1, 98.1] |
| real hand-over lead | 99.5 % [98.9, 99.8] | 96.7 % [95.4, 97.6] |
| + ring 3 mm | 97.4 % [96.2, 98.2] | 91.3 % [89.4, 92.9] |
| gap 2.5 mm | 99.0 % [98.2, 99.5] | 94.0 % [92.4, 95.3] |
| gap 9 mm | 99.4 % [98.7, 99.7] | 96.6 % [95.3, 97.5] |
| roll ±2 deg | 94.5 % [93.0, 95.8] | |
| roll ±4 deg | **56.3 % [53.3, 59.4]** | |
| pitch ±3 deg | 99.4 % [98.7, 99.7] | |
| **real TCP + Euler offsets in the observation** | **35.0 % [32.1, 37.9]** | **21.8 % [19.4, 24.4]** |
| real in-hand estimate offset only | 99.6 % [99.0, 99.8] | 97.4 % [96.2, 98.2] |
| real offsets, then re-centred (the fix) | | **89.4 % [87.3, 91.1]** |

Real: 2 of 4 seated, 95 % interval [15, 85] %.

**What separates a seat from a stall.** In the sim with the real offsets, the failures are mostly `lateral_miss` and `grasp_slip`: the policy pushes in the wrong place or too hard, because it believes it is elsewhere. On the rig, the same belief shows in the contact head (next section). From the four real runs alone nothing separates them. The sim says the policy's own belief decides, and that belief is wrong on the rig. Roll above 2 deg is the second risk: the align stage turns pitch and yaw only.

## 2. The operator's ideas

| Idea | What we know now | Sim plan | Expected effect, with n |
|---|---|---|---|
| **Contact points (Bronars et al. head)** | Done. Real runs: the head reads "inside" from t = 0, before contact, and flips inside/touching in the stalls. Sim: 96.2 % of edge points right (recall free 98.7, touching 95.9, inside 92.9 %; n = 256 rollouts x 150 steps x 64 points). Figure: `scripts/real/kj_contact_view.py`. | Use it as a check on every K2 eval and every real run: the head must read "touching" before entry on the rig. | Measured, see left. |
| **Closer is better** | Not in the sim: gap 1-14 mm all 98.4-99.6 % (n = 1024 each, `start_map_gap.json`); gap 2.5 is not better than 5.5 on a real-like plant (94.0 against 96.7 %). Lateral and roll matter: ring 4 mm 94.6 %, ring 5 mm 81.2 %, ring 6 mm 57.5 %; roll 4 deg 56.3 %. | None needed. Keep 6 ±1 mm; spend the effort on lateral and roll. | No gain expected from a closer stop. |
| **Wrist camera + blind policy (prior on where the port is)** | Port vision gives plug-to-port across/up/gap/angles from one image, about 10 Hz, independent of encoder error. | K2 gets the port-vision relative pose at the hand-over (held, noise and bias as port vision's held-out error) instead of the absolute TCP features. Test at ring 3-6 mm and roll 0-4 deg against K-J. n = 1024 per cell. | Planned. Acceptance: ≥ 98 % at ring 5 mm and roll 3 deg (K-J: 81.2 % at ring 5 mm; roll 3 deg not measured, 94.5 % at 2 deg and 56.3 % at 4 deg). No number before it runs. |
| **RL for alignment (postdoc)** | The align stage works but is slow (22-43 s) and jerky: steps against stick-slip on joints 2-3. The controller made it 21.8 s with steps at 0.94-1.09 of the ask (n = 1). | Compare three methods on the same sim plant (stick-slip from the measured friction, latency, 10 Hz vision with its noise and dropouts, encoder offsets): (a) today's step-and-measure stage, (b) a continuous vision servo (vision on its own thread, smooth 50 Hz target), (c) an RL policy, input = port-vision relative pose + joints, output = plug velocity. n = 1024 episodes each; `eval_align_batch.py` has the baseline. | Planned. Pass: hand-over in ≤ 10 s, overshoot ≤ 0.5 mm p95, error ≤ 0.5 mm p95. RL goes ahead only if it beats (b). |
| **VLA (π0.5) / RLT** | Read the paper (Xu et al., Physical Intelligence, "RL Token"). Base model π0.6. Needs 1-10 h of teleop demos per task, 400-1000 real episodes with an operator's success label, about 15 min to 5 h of robot data. Gains: up to 3x faster on the critical phase; Ethernet success was already high, screw 20 to 65 %. It does not fix a sensor; it learns on the real plant. | Not π0.5 now: demos break the no-real-demo rule and the task is one port. **The useful part is the recipe:** a small actor-critic, trained on the robot, that edits a base policy's action chunk and is pulled toward it (BC term). Our base would be K-J or K2. Sim rehearsal: treat the offset sim (21.8 %) as "real", allow 400 episodes, measure recovery. n = 1024 eval. | Planned, after K2. HIL-SERL (Luo et al.) got 100 % on USB insertion after 2.5 h of real training on a Franka, but with demos and interventions. |

## 3. Contact policy (K2)

Fine-tune from K-J, same action scheme, plain PD first.

1. **Observation:** TCP position and Euler relative to the hand-over pose (zero at the hand-over), not to a fixed nominal. Optional: port-vision pose at the hand-over (idea above). Velocity noise matched to the real telemetry: 0.005-0.04 rad/s rms at rest on joints 4-6 (sim: 0).
2. **Starts:** gap 4-8 mm, ring up to 4 mm, roll ±3 deg, pitch and yaw ±4 deg, real hand-over lead.
3. **Plant:** friction x0.5-2.0, gravity error ±12 %, encoder offsets ±1.5 deg (as now). Backlash and compliance stay out until the rig measures them.
4. **Anti-climb:** a reward penalty on joint target lead above 20 mrad, and the climb checked: commanded tip rise ≤ 10 mm p95 (K-J: 14-16 mm in the sim).
5. **Controller, later:** the precision law (`blind_insertion/envs/precision_ctrl.py`) as a batched torch plant, 50 % of worlds, with the hand-over target set from the measured pose (no plain-PD lead).

Sim acceptance before the rig, n = 1024 each: ≥ 98 % from real-like hand-overs on the real-like plant; ≥ 95 % at ring 4 mm and at roll 3 deg; climb as in 4; contact head "touching" before entry.

## 4. Alignment

- **Now:** `bimanual_align4_precision.json` = the controller config that aligned best (`f12`, `align_prec7`) + `align3`'s approach. The servo ends at 18 mm with 2 iterations, so its wind-up (4.4-8.5 mm uncommanded moves) happens far from the case. Entry gate 12 mm; back-off keeps the creep lead.
- **Blocked:** the controller cannot hand over to plain-PD K-J yet. Leaving the controller drops its integral torque in one tick: a jump at the plug. `bimanual_align4_precision.json` refuses `--align-mode full` (`align.full_allowed` false). Needed: a bumpless switch that folds the controller torque into the PD target over 0.2 s, with a mock check. Or K2 trained on the controller.
- **Next:** the continuous vision servo, then RL if the sim says it is better (section 2).
- **Roll:** the stage corrects pitch and yaw only. Roll above 2 deg costs K-J 5 points, 4 deg costs 43. Roll correction needs the vision roll checked first (camera roll bias, `results/real/inhand_real/camera_roll.json`).

## 5. Next rig session

Pull `real/graph-align`. Clamp the bases; `estop.py` open; one go per motion command; nothing moves without the operator's go. Run `C1=deploy/config/bimanual_align4_precision.json`, `C2=deploy/config/bimanual_align4_kjrc.json`.

### 0. No-motion checks

```bash
for c in bimanual_align4 bimanual_align4_kjrc bimanual_align4_precision; do .venv/bin/python -m deploy.checks.bimanual_conventions --config deploy/config/$c.json | tail -1; done   # FAILED: none, three times
.venv/bin/python -m pytest -q tests/test_precision_ctrl.py
.venv/bin/python -m deploy.checks.bimanual_realio_mock
.venv/bin/python -m deploy.run_bimanual --stage full --config $C1 --align-mode full --name refuse_check --dry-run   # must refuse: "does not allow align mode full"
.venv/bin/python -m deploy.run_bimanual --dry-run --stage approach_insert --seed 7020 --config $C2 --align-mode full --name dry_kjrc   # completes; events.jsonl has insert_obs_recenter
```

### Side camera (for every run in B)

Phone on a fixed support, square to the port row, lens at port height, 40-60 cm, 2x zoom, 60 fps if possible. Port 2 and the whole plug in frame. Start recording before the insertion starts; say the run name into it.

### A. Alignment on the controller, align mode, 2 picks

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C1 --align-mode align --name a4p_1
.venv/bin/python scripts/real/approach_jump.py data/real_runs/$(date +%F)/full_*a4p_1
.venv/bin/python scripts/real/align_log.py data/real_runs/$(date +%F)/full_*a4p_1
```

Then the same with `a4p_2`. **Pass:** largest uncommanded move ≤ 3 mm; stage reaches the hand-over; steps 0.85-1.15 of the ask (p10-p90); stage ≤ 35 s (it starts 12 mm further out than prec7).

### B. K-J with the re-centre, plain PD, full chain, up to 6 picks

Only after A passes, or after A fails in the approach only (B runs on plain PD).

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C2 --align-mode full --name kjrc_1
.venv/bin/python scripts/real/kj_real_ref.py data/real_runs/$(date +%F)/full_*kjrc_1 --png kjrc_1.png
.venv/bin/python scripts/real/kj_contact_view.py data/real_runs/$(date +%F)/full_*kjrc_1 --png kjrc_1_contact.png
```

Then `kjrc_2` to `kjrc_6`. The operator confirms each seat by hand and eye after the release. **Record per run:** seated (operator), FK depth, target up max, contact head before entry. **Expectation:** no rate from n = 6 can prove the fix. 6 of 6 has an interval of [61, 100] %; 2 of 4 had [15, 85] %. What would count as support: the head reads "touching" before entry, and no stall with the target climbing above 30 mm.

### Stop rules

**Stop at once (Enter on the estop)** if any of these happens:
- the plug is about to hit the case outside the port row;
- a jump is visible;
- a base or the PC moves;
- the reader falls.

**Do not continue, and report** if any of these happens:

| Check | Limit |
|---|---|
| Uncommanded approach move | above 3 mm |
| Joint 2 or 3 error in the approach | above 25 mrad |
| Entry gap | stage stops below 12 mm, or above 25 mm |
| Grasp | refused twice in a row |
| Gripper reading in the stage | changes by more than 0.02 (existing gate) |
| B | 2 stalls in a row, or any seat the operator cannot confirm |
| `insert_obs_recenter` event | missing from a B run |

Send the logs, the side videos and the PNGs. Then wait for the server.

## Questions for the operator (please confirm before the training work)

1. **K2 spec (section 3):** relative position features, anti-climb penalty, the wider starts. Yes, or changes?
2. **Real-world RL later (the RLT recipe on top of K2):** hundreds of rig episodes, an operator's success label on each, no demos. Is that allowed?
3. **VLA:** drop π0.5 for now?
4. **Alignment:** continuous vision servo first, RL only if it is better in the sim. Yes?
