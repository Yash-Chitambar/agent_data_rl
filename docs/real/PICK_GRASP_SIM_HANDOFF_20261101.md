# Real pick grasp failure: what the sim must represent (2026-11-01, duchamp3)

First real session of the bimanual chain (`real/bimanual-deploy`). Full log of every run, change and failure:
[SESSION_20261101_duchamp3.md](SESSION_20261101_duchamp3.md). Run folders: `data/real_runs/2026-11-01/`
(gitignored; rsync from the rig, see the session log).

## Result in one paragraph

Approach + insertion works on the real rig with the reader placed in the jaws by hand (`approach_insert_171123_ai4`:
inserted). The servo pick reaches the reader every time (ZED reader estimate, hover, three wrist-servo waypoints),
but the grasp is marginal: of 8 closes inside the pick (pick4-pick10 repeat) **2 held the reader** (pick6, pick10); pick2 also caught it in its post-stop recovery close (no press). pick1-3 stopped earlier on SafeArm clearances (fixed, below). The cause
is contact mechanics the sim does not have: the sim sweep below grasps 100 % at every tilt and opening we tried on
the robot. Fix the sim until it fails the way the robot fails, then retrain / re-tune the pick, then test `full`.

## What the robot does (measured)

| Quantity | Sim (dry runs, same code) | Real | Source |
|---|---|---|---|
| Pick target depth at the close | tips ~11-13 mm below the table plane (tip floor) | same plan: 16-31 mm of commanded press removed by the relief | `ticks.npz` target_b, `close_press_relief` events |
| Measured fingertips at the close | -2.5 mm (penetrate the table) | **+0.7 .. +4.3 mm** (stop on the table) | FlexChecker heights of `q` |
| Hand orientation error while pressing (1 s before the close) | 0.4-0.9 deg | **3.3-3.7 deg** about one tool axis, every pressed run (pick4-pick11) | FK of `q` vs `target_b` |
| Residual after the relief v3 orientation fix | n/a | 1.0-1.3 deg, growing during the close | pick10, pick10 repeat |
| Reader in the side grasp, i2rt reading | sim finger 19.0 mm | 0.30-0.33 (ai1-ai5, pick6, pick10); 10-28 map said 0.254 | `grip` column |
| Grasp vs approach opening | 23-26 mm all succeed | 24 mm caught (with relief), 25 / 26 mm closed empty (pads over the reader) | pick3-pick7 |
| Hand during an unpressed close | stays | **rises**: the squeezing pads push the hand up and slide over the reader's top edges | pick9 + operator |

Operator observations (verbatim in the session log): "when the gripper hits the table it slightly rotates,
offsetting it from being parallel with the table / USB-C; there is no check or correction"; "while grasping the
gripper moves up, it should be even"; "make the grasp speed slower".

## Sim sweep: the sim does not show the failure

`scripts/policy/pick_feedback_sweep.py` (wraps `scripts/policy/pick_expert_eval.py`, patches
`reader_grasp.TILT_DEG`), oracle servo, encoder-frame planning, 512 episodes each, seed 1000:
tilt 50 / 55 / 60 deg x opening 23-26 mm: **512/512** every cell; 65 deg: 511/512 every cell
(`results/usbc/policy/pick/feedback_sweep_20261101_oracle/sweep.json`). The run with the real servo nets crashed in
`pick_expert_eval.py` (`'BimanualRuntime' object has no attribute 'rc'` at `refit_bvh`; logs in
`results/usbc/policy/pick/feedback_sweep_20261101_nets/`).

## Sim work (server, `sim/fidelity` then `main`)

1. **Fingertip / table contact:** the real tips stop on the table; the sim tips sink ~2.5 mm. Stiffer table contact
   (solref / margin for the FlexPoint tips) so the pads sit at the real height relative to the reader.
2. **Wrist compliance under the press:** with kp 80/80/80/40/10/10 (kd 5/5/5/1.5x3) the real hand turns ~3.5 deg
   when the tips press the table; the sim ~0.5 deg. Check the joint 4-6 controller and friction model (the real
   DM4310 wrist is weak) and the tip-table friction (the tips stick and the wrist rolls).
3. **Pad-reader contact:** the pads ride over the reader's top edges when the hand is tilted or rises. Reader body
   edge radius, pad friction and pad shape; make the sweep above fail at 25-26 mm and at a 2-4 deg tilt.
4. **Acceptance test:** re-run `pick_feedback_sweep.py` (oracle and real nets) with the plan's press and
   with relief v3 (below); the sim should reproduce: unrelieved press -> mostly empty closes; relief v3 -> mostly
   held; 26 mm worse than 24 mm.
5. **Then:** decide whether the pick expert itself should stop pressing (close at a measured height with the planned
   orientation, slow close), retrain the servo / in-hand nets on that grasp, and update `G.TILT_DEG` only if the
   sim then prefers another tilt.

## Real-side workaround now (config `deploy/config/bimanual_pick2.json`, uncommitted code went in with this doc)

- `gripper.right.g_reader` 0.254 -> 0.32 (side grasp); descent opening `right_approach_open_m` 0.024 (reading 0.437).
- `pick.close_press_relief` (`deploy/bimanual/chain.py::_close_press_relief`): at the close switch hold the measured
  position 3 mm down with the PLANNED orientation, jaws open 0.3 s, close ramped over 1 s, dwell 0.3 s, blend into
  the lift over 0.5 s; `pick.episode_s` 26.
- SafeArm body clearances checked on the commanded (pressed) target: gripper 3 mm, link_6 15 mm, link_5 30 mm.
  The real clearance was ~2.5 cm; restore 10 / 20 / 40 mm once the checks use the measured pose or the tip floor.

## Other sim-side items from the session

- **Seated detector:** FK advance since the hand-over is not a reliable seat signal on the robot (the servo's 6 mm
  stop gap varies: advances of 5.9 / 11 / 12.2 / 17.9 mm at stall in ai2-ai5, two false "seated"). Needs a contact
  or force cue. Config `bimanual_seat2.json` (stricter thresholds) did not fix it.
- **Pre-insert FK lateral** reads 13-32 mm on good and bad approaches alike: not usable as a guard.
- **Release:** the right jaws open only to the sim's 24 mm finger limit and drag the seated reader on retreat (ai4);
  `insert.release_open_m` (full open) exists in `chain.py`, used only by `bimanual_seat2.json`.
- **ZED PC net bias:** +3 .. +5 mm in y on every capture frame (two placements, ZED edge-fit ground truth).
- **Press force estimate** reads 70-78 % of the commanded 15 / 25 N.
- **Hold swing speed:** `HoldCfg.move_s` (1.5 s) now scales with `--speed`; real left joint 4 ran 2.5x the sim speed.
