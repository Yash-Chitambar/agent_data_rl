# Precision joint control and gravity system ID on duchamp3

Session 2026-11-06, right arm only. Branch `real/graph-align`. Motivation: the align stage was jittery and imprecise (session 2026-11-03: 1 mm steps arrived as breakaways at 54-78 mm/s; the approach jumped 4.8-6.9 mm). The athena3 cell's controller (github.com/XinyueZhuXY/yam-control d592e8e, `yam_controller.py`) has the features we lack. We took its **features**, not its values: the values come from this rig's own system ID.

## Result (right arm, encoder FK, one run per row)

| Run | Hand error median / max | Moved joint end error, median | Settled within 1 mrad | Settle lock engaged |
|---|---|---|---|---|
| Plain i2rt PD, default gravity (`20261106_141135`) | 6.1 / 17.8 mm | | 0 / 12 | |
| Controller, default values, default gravity (`20261106_141135`) | 1.0 / 1.5 mm | | 6 / 12 | 0 / 12 |
| Plain PD, fitted gravity (`20261106_150932`) | 4.2 / 6.5 mm | 10.3 mrad | 1 / 12 | |
| **Controller, fitted gravity, measured friction, bands at the encoder step (`20261106_150932`)** | **0.13 / 0.27 mm** | **0.33 mrad** | **12 / 12** | **12 / 12** |

Steps: joints 1-3, +-30 and +-3 mrad (3 mrad is about the align stage's 1 mm step), hold 3 s. Settling 0.4-2.5 s, slowest on joint 2 after 30 mrad. Limits:
- **Encoder FK only.** Link flex and gear backlash are not in it. athena3's 0.43 mm median was camera-seen. A camera check is still to do.
- The two controller runs used different test poses (`right_work`, then `right_test`, joint 2 gravity load 9.0 and 4.9 N m).
- One run each; the left arm is not done.

Without the fit and the band change the controller hunted on joints 2-3 (3-8 mrad overshoot, lock never engaged): the encoder step is about 0.38 mrad, larger than the old 0.1 mrad dead band and close to the 0.5 mrad lock band; joint 2's integral also hit its 3 N m cap.

## Gravity fit (right arm)

`data/real_yam/gravity_id/20261106_150421_right/fit.json`, from three collect runs (43 poses with both approaches, 8 held out, the 2 contact holds of pose 20 dropped). Model: i2rt's yam + linear_4310 description, payload = body `gripper`.

| Model | Payload kg | Held-out rms, joints 1-6 (N m) |
|---|---|---|
| i2rt as we ran it (0.5 kg, factors 1, 1.1, 1.1, 1.2, 1, 1) | 0.50 | 0.06, **0.82, 0.61**, 0.15, 0.07, 0.03 |
| payload fitted, factors 1 | 0.95 | 0.06, 0.20, 0.25, 0.06, 0.04, 0.04 |
| **full: payload and factors on joints 2-5** | **0.99** | 0.06, **0.13, 0.19**, 0.04, 0.03, 0.04 |

Factors 1.02, 0.97, 1.04, 1.18 (joints 2-5; joints 1 and 6 carry no gravity and stay 1). Joint 5's factor is the least certain (0.94-1.18 between subsets). Static friction (half the difference between approaches from above and below), median: 0.14, 0.45, 1.09, 0.07, 0.09, 0.09 N m.

## Code

| File | What |
|---|---|
| `blind_insertion/envs/precision_ctrl.py` | The law, pure NumPy (real and sim): integral (dead band, freeze, cap, pause while the target moves), settle lock, friction feed-forward, vibration watch. Every feature off by default. |
| `scripts/real/precision_arm.py` | 200 Hz thread per arm: writes the law's torque into i2rt's command `torques` field (no i2rt patch); `gravity_fit` reads a fit (ours or athena3's format) |
| `scripts/real/safety.py` | `connect(gravity=)`: fitted payload mass, factors, centre of mass; `SafeArm.ctrl`: re-applies the controller torque after every send |
| `scripts/real/deploy.py` | `RealArm(gravity=, precision=)`; no second integral (settle trim) when the controller runs |
| `deploy/bimanual/io_real.py` | Run config `arm_control.<side>.{gravity, precision}`; logs `ctrl_<side>` every tick; `precision_freeze(side)` |
| `scripts/real/arm_clearance.py` | Two-arm distance on the bimanual scene model; contact guard |
| `scripts/real/gravity_id.py` | `plan` / `collect` / `fit` |
| `scripts/real/precision_check.py` | Steps per joint, plain PD against the controller |
| `deploy/config/gravity_poses_right.json`, `precision_poses.json` | Planned poses; test poses |
| `tests/test_precision_ctrl.py`, `deploy/checks/bimanual_realio_mock.py` | Offline checks |

Settings used for the good row:

```bash
.venv/bin/python scripts/real/precision_check.py --side right --joints 1,2,3 --deltas 0.03,0.003 --modes pd,precision \
  --gravity data/real_yam/gravity_id/20261106_150421_right/fit.json --features integral,settle_lock,friction_ff \
  --friction 0.138,0.447,1.087,0.069,0.091,0.087 --set deadband=0.0004 --set latch_band=0.001 --set unlatch_band=0.002
```

## Safety: the other arm (incident 2026-11-06)

The first gravity collect drove the right hand onto the limp left arm at pose 20: joint 1 blocked by 37-52 mrad at 3-4 N m, operator stop. The planner had checked the table only, and the left arm's limp pose was unknown. The bimanual model with the left arm assumed at zero gave 32.6 mm at that pose, so an assumed pose is not good enough. Now:

1. **The other arm is measured.** `collect` and `precision_check` connect it first (held where it lies; its gripper opens and closes once) and check every pose, approach point and path against its real joints: 80 mm.
2. **Planning** solves joint 1 (no gravity load) so every pose keeps 100 mm from the left arm at rest and at limp variations (joint 1 +-0.25, joints 4-5 +-0.5 rad), with the hand inside the region visited in logged runs.
3. **Contact guard**, every tick while moving and holding:
   - Joint 1 off target by more than 12 mrad or above 1.2 N m holding.
   - Joint 1 off target by more than 30 mrad or above 2.0 N m moving.
   - Any joint past 80 mrad, or 150 mrad on joints 5-6 (kp 10: they lag up to 82 mrad in moves).
   - Normal values: joint 1 within 5.5 mrad and 0.6 N m holding, 16 mrad and 1.1 N m moving at 0.15 rad/s.
   - On a trip the arm holds where it is and the script stops.
4. Moves at 0.15 rad/s (the guard's limits come from moves at that speed).

## Ready to run, not yet run (2026-11-06)

- **Left arm gravity ID.** `gravity_id.py plan --side left` -> `deploy/config/gravity_poses_left.json`: 32 poses from the left arm's logged configurations, hand box from its logged hand positions (x 0.15-0.38, y 0.31-0.77, z up to 0.37 m), own side y >= 0.148 m, at least 139 mm from the right arm at rest and at its limp variations. `collect --side left` connects and holds the right arm and checks against its measured joints. Fake-arm run: 32 of 32 poses, 64 holds.
- **Camera check.** `precision_check.py --camera`: the arm goes to the test pose and holds; the operator slides the checkerboard (calib.py's, 15 mm squares) flat under the wrist camera (it looks at the table about 15 cm below it: for `right_test`, x 0.06, y 0.26 m in the right base frame); after a steady detection and "hands clear", every hold ends with `--cam_frames` frames. Per step: vision against the FK of the target and of the measured joints, relative to the last hold at the test pose; and vision's return to the test pose (no model). `--camera_plan` plans a view pose from the ZED's board pose instead (for the board where the test pose does not see it). Synthetic-camera run: pipeline consistent (0.06-0.39 mm with 0.1 mm noise).
- **The chain on the controller.** `deploy/config/bimanual_align2_precision.json` = `bimanual_align2` + `arm_control.right` (fitted gravity, controller values of the good row above, integral frozen in `5 insert` and `6 release` via `chain._precision_stage`). Left arm unchanged. Checks: conventions and gates `FAILED: none`, sim dry run completes (the sim does not run the controller), mock real I/O ok. First real use: align mode, no insertion.

## Incident: the controller in the pick (2026-11-06, run `full_155102_align_prec2`)

The first chain config ran the controller and the fitted gravity from connect on, so also in the pick. The pick was tuned on plain PD with i2rt's default gravity and pushes its target below the table on purpose (the hand stops on the reader and the table). With joint 2 holding a 1.84 N m integral from the hover and the friction feed-forward, the hand came down earlier and lower than in the good 2026-11-03 pick; the floor guard lifted the target by up to 13 mm (0.2 mm before), the gripper touched the table, and the table check stopped the run (link 5 at 30 mm). Fix: `arm_control.<side>.active_stages`. The controller torque and the fitted gravity run only in the listed stages (now transport, approach, align). Every other stage runs exactly as before: controller torque zero and integral reset, i2rt's default gravity (`precision_arm.use_gravity`), the chain's Coulomb feed-forward. The arm connects in that old behaviour. Checks: `tests/test_precision_ctrl.py` (the gravity switch against i2rt's own model) and the real-I/O mock (starts off, Coulomb feed-forward when off, on only in the listed stages). The same run's first try (`full_154548_align_prec1`) crashed in software: the stage hook read the block's `_about` note as an arm.

## First chain run with the controller (`full_160705_align_prec4`, align only): worse motion

Pick unchanged (no floor lifts, the controller off), stage `aligned` (-0.14, +0.04, 5.74 mm; rx -0.7 deg). But the motion got worse than plain PD on 2026-11-03:

| | 2026-11-03 (plain PD) | align_prec4 (controller in transport, approach, align) |
|---|---|---|
| Largest uncommanded 0.5 s move in the approach | 4.4-6.2 mm | 16.9 mm |
| Align-stage tip speed p99 / max | 27-32 / 54-86 mm/s | 61 / 141 mm/s |
| Stage ticks above 20 mm/s | 35-55 | 109 |

Cause (approach trace): the servo wound joint 2 up to 27-30 mrad behind its target (as on 2026-11-03); at that error the integral (kp / 0.12 s) rose at ~20 N m/s to 2.9 N m, the joint broke free and overshot 12-21 mrad, the integral reversed to -3 N m: a 15-20 mm hand swing. `precision_check` never showed it: its smooth steps never left a joint far behind. Next: `bimanual_align2_precision_f6.json`, the integral only under 6 mrad (`freeze`), larger errors left to the springs and the friction feed-forward. Toy stick-slip joint: hunting after a lagging ramp 41 -> 6 direction changes, 3 mrad steps unchanged. Not yet run on the arm.

## The controller in the align stage, then in contact (2026-11-06, `align_prec6`, `align_prec7`, `align_full7`, `align_full8`)

**Why prec4/prec5 did not help:** the align stage's creep law puts the target ahead of the goal and changes it every tick. The controller's friction feed-forward and settle lock act only on a still target: in prec5 the feed-forward ran 1 % of the stage time and the lock 0 %, and the integral pushed toward the creep offset, not the goal. With the creep offsets off (`lead_max_rad` 0) the feed-forward ran 80-82 %.

| Run (config) | Steps moved / asked, p10 / median / p90 | Stage s | Result |
|---|---|---|---|
| Nov 3 align2 (plain PD, creep) | 0.86 / 0.98 / 1.39 | 32.5 | aligned |
| prec6 (creep off, window 6 mrad) | -0.33 / 0.90 / 1.04 | 27.6 | stalled: joint 2 stuck 7-8 mrad, outside the window |
| **prec7 (creep off, window 12 mrad, `bimanual_align2_precision_f12.json`)** | **0.94 / 1.01 / 1.09** | **21.8** | **aligned (+0.10, -0.70, 5.49 mm; rx -0.9 deg)** |
| full8 (as prec7 + on in contact, integral frozen) | 0.90 / 1.01 / 1.18 | 21.4 | hand-over 0.15, -0.09 mm, gap 5.77 mm (best so far) |

Approach: largest uncommanded move 8.5 mm in prec7 (plain PD 4.4-6.2 mm): the approach servo still winds up and the 12 mrad window reaches into it.

**The insertion policy and the controller do not fit (full8).** In the policy's warm-up (zero actions, first 0.2 s) the plug rose 0.8, 1.8, 2.3 mm; on Nov 3 (plain PD) it stayed within 0.1 mm. Then the policy's actions took it to +7.7 mm at 0.30 s and the guard bound stopped it at +9.7 mm (0.34 s). Cause: K-J trained on plain PD, where the joint target sits above the measured joint (gravity, friction); the hand-over sets that offset on purpose (`handover_lead_rad`, joint 5 14.4 mrad, +2.9 mm at the plug). Plain PD holds still against it; the controller drives the arm onto the target, so the offset becomes a 2.9 mm move up, and every policy action is followed more fully than in training (the known climb, at once). To use the controller in the insertion: model it in the sim and fine-tune or re-evaluate K-J on it, or change how the hand-over sets the target. Until then the insertion runs on plain PD.

`align_full7` stopped at the hand-over on the vision-net check (4.49 mm in up, limit 3.5; earlier runs 0.2-3.0 mm; shell depth 122.0 mm, the deepest so far). `full8` used `--align-set disagree_mm=[3.0,5.0,2.5]` for the run.

Fixed on the way: the stage hook did not run on the chain's slow moves (home stayed on, frozen, and stopped 24 mrad short); `move_to` uses the old settle trim when the controller is off or frozen.

## Open

- **Joint 6 resistance at gravity pose 22** (joint 4 -0.95, joint 5 about 0, joint 6 1.18-1.26): twice it held 83 mrad off at 0.8 N m and could not follow. At the same joint 6 angle at other poses it held within 10 mrad. Nothing visible held it; no self-contact in the model. Cause unknown (camera cable? motor friction at that angle?). Watch the wrist there.
- `right_work` (the seated run's hand-over pose, raised 40 mm) is 76.7 mm from the measured left arm at rest: the check refuses it. `right_test` replaces it.
- Left arm: system ID, poses and checks with the arms' roles swapped.
- Camera-checked accuracy; the controller in the bimanual chain (`arm_control`) and the align stage's small steps; the sim model of the controller (server).
