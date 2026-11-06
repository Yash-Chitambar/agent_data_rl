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

## Open

- **Joint 6 resistance at gravity pose 22** (joint 4 -0.95, joint 5 about 0, joint 6 1.18-1.26): twice it held 83 mrad off at 0.8 N m and could not follow. At the same joint 6 angle at other poses it held within 10 mrad. Nothing visible held it; no self-contact in the model. Cause unknown (camera cable? motor friction at that angle?). Watch the wrist there.
- `right_work` (the seated run's hand-over pose, raised 40 mm) is 76.7 mm from the measured left arm at rest: the check refuses it. `right_test` replaces it.
- Left arm: system ID, poses and checks with the arms' roles swapped.
- Camera-checked accuracy; the controller in the bimanual chain (`arm_control`) and the align stage's small steps; the sim model of the controller (server).
