# Physical rig TODO (2026-10-30)

What the sim still needs from the real YAM rig (duchamp3), in priority order. Items marked
**sim side** need no rig time. Background: `results/usbc/bimanual_20261029/README.md`
(identified arm dynamics, fingertip fit, runtime speed).

## 1. Gripper opening sweep (today, ~5 min, jaws only) — DONE 2026-10-30

**Result:** calipers 105.25 mm lens-to-tip on both arms; fit held to it: held-out IoU left 0.89,
right 0.81; maps monotonic, <= 0.4 mm RMSE; arms agree to ~2 mm. Applied to the wrist camera
(default). See `results/usbc/bimanual_20261029/README.md` (Gripper sweep).

**Why.** The sim fingertips sit 1.5-3.5 mm off the real ones in the wrist images
(`results/usbc/bimanual_20261029/gripper_tips/`). No FlexPoint CAD is available, so we fit the
tips to real wrist images. Recorded data has only open + one closed frame for the left arm and
open frames only for the right. A sweep through intermediate openings, on both arms, (a) fits
the tip offset on each arm with a closed frame, (b) gives the gripper encoder -> opening map,
and (c) holds out the intermediate levels to check (a).

**Moves:** gripper jaws only. Arms stay at folded rest and are held by the safety PD.

On duchamp3:

```bash
git fetch && git checkout codex/bimanual-calibrated && git pull
# 1. Bases clamped. Start the rig console in its own terminal (space = STOP):
.venv/bin/python scripts/real/rigctl.py
# 2. Both arms at folded rest, jaws empty, nothing between or near the fingers.
#    Normal room lighting (as in the bimanual session); nothing yellow in the lower wrist view.
# 3. Sweep both arms (one at a time; press Enter when each asks):
.venv/bin/python scripts/real/capture_gripper_sweep.py --side both --out data/gripper_sweep/sweep_20261030
# 4. Send the folder back (~100 MB of raw PNGs, too big for git):
rsync -a data/gripper_sweep/sweep_20261030 mason:/home/yashchitambar/peg_test_bimanual/data/gripper_sweep/
```

Check before sending: `sweep.json` says `"result": "completed"` and has 66 rows (11 levels x 3
frames x 2 arms). The script refuses a non-empty output folder.

**Sim side, after the data arrives:**

```bash
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/real2sim/fit_gripper_tips.py --sweep data/gripper_sweep/sweep_20261030
```

Accept the offsets if both arms reach held-out IoU >= 0.8 and the map is monotonic with
<= 0.5 mm RMSE; then the runtime uses them via
`BimanualCfg(tip_offset='results/usbc/gripper_sweep/sweep_20261030/fit.json')`.
If the offset still disagrees between arms by more than ~2 mm, suspect the hand-eye
calibration rather than the tips, and redo `scripts/real/calib.py` for that wrist.

## 2. PC-on-table friction: randomized instead (no rig time)

Not measurable with the tools at hand. The runtime samples the PC-feet friction per world on
every reset, mu U(0.3, 0.9) (`BimanualCfg.pc_friction_range`): hard plastic to rubber feet on a
smooth top. The feet have contact priority, so this value alone sets the PC-table pair.

## 3. Joint backlash (~25 min per arm, small arm motions) — DONE 2026-10-30

**Result:** <= ~2 arcmin, mostly below camera noise (`data/backlash/README.md`). Not modelled
in the bimanual runtime (sub-noise next to stick-slip and encoder-offset noise).

**Why.** Phase C showed backlash of 1-5 arcmin decides sim-to-real success; it is not measured
on these arms. The encoders sit on the motor side and cannot see it.

**Moves:** the arm goes to a pose where its wrist camera sees the board (same planner and
safety checks as `calib.py find`), then steps one joint at a time by 3 mrad stairs, at most
+/-9 mrad, at 0.05 rad/s, and returns to rest.

```bash
# Board (9 x 8 squares, 15 mm) flat on the table ~0.3 m in front of the arm, TAPED DOWN.
# Rig console running; bases clamped.
.venv/bin/python scripts/real/measure_backlash.py run --side right --out data/backlash/right_20261030 --hint 0.30 0.10
.venv/bin/python scripts/real/measure_backlash.py run --side left  --out data/backlash/left_20261030  --hint 0.30 -0.10
.venv/bin/python scripts/real/measure_backlash.py analyze data/backlash/right_20261030
.venv/bin/python scripts/real/measure_backlash.py analyze data/backlash/left_20261030
git add data/backlash && git commit -m "Backlash measurement, both arms" && git push   # small JSON + PNG
```

`--hint` is the rough board centre (m) in that arm's base frame; adjust to where the board is.
`analyze` prints per-joint backlash (arcmin) and per-stair camera noise (~0.3 arcmin expected)
and writes `backlash.png` (hysteresis loops). Joints 2-3 are gravity-preloaded, so expect a
smaller loop there. If the board is lost mid-joint, the script skips the rest of that joint.

## 4. Already measured: sim side only (no rig time)

These exist in the single-arm env (`EnvCfg`) and must be ported to the bimanual runtime:

- [ ] Action latency: measured ~8 ms (0.4 control step); randomize 0-1 step.
- [ ] Encoder-offset noise: +/-1.5 deg on joints 2-5 (`EnvCfg.enc_offset_deg`).
- [x] Backlash: measured <= ~2 arcmin; not modelled (below other noise; item 3).
- [x] PC-table friction randomization, mu 0.3-0.9 (item 2).
- [ ] Rigid insertion contact with randomized plug-port friction and pad friction/stiffness.

## 5. Optional, after the new grasp is chosen

10-20 grasps of the reader with the new grasp, holding still 1 s after each (wrist cameras +
ZED). Gives the in-hand reader pose spread for the reset distribution. Until then the sim uses
a generous in-hand pose randomization.

## 6. Ground-truth captures for the ZED estimator and the grasp (~60-75 min, arms move)

Why: the ZED PC-pose net has been checked on one real PC placement against a ±5 mm hand-marked landmark,
and in sim it is only valid at one arm pose (FINDINGS 2026-11-01, rounds 5-7). These captures give
measured PC poses under every arm pose the task uses. Rig console running; bases clamped; `estop.py` open.

**A. Mark and measure the PC spots (10 min)**
1. Mark 4 PC spots with tape at the front-left case corner and along the left edge: the nominal
   (port 2 at x 0.22, y 0.36 m in the right-arm base frame, yaw 0), then +1 cm x, +1.5 cm y and
   -1 cm x / +5 deg yaw.
2. Measure each spot in the right-arm base frame: run `scripts/real/snapshot.py point --target <x> <y> --tag spotN`
   with the ZED view, or measure with a steel ruler from the right-arm base plate edge (resolution ±1 mm, write
   the two offsets and the yaw from the corner pair). Optional, better: tape a 40 mm AprilTag (tag36h11) flat on
   the PC lid centre; `scripts/real/calib.py` already finds tags in the ZED frame (± ~1 mm).
3. Record per spot in `data/zed_gt/spots.json`: port-2 x, y (m), yaw (deg), method, uncertainty.

**B. ZED frames per spot and arm pose (8 frames per spot, ~5 min per spot)**
For each spot (PC on its tape marks, reader as listed):
```bash
O=data/zed_gt/spotN
.venv/bin/python scripts/real/snapshot.py pose --out $O --pose folded  --tag folded_reader      # both arms folded (rigctl rest)
.venv/bin/python scripts/real/snapshot.py pose --out $O --pose capture --tag capture_reader     # the ZED net's capture pose, grippers open
.venv/bin/python scripts/real/snapshot.py pose --out $O --pose parked --side left --tag parked_reader   # Part 1 park (left)
# remove the reader from the table, repeat the three lines with tags *_noreader
```
- Reader "in the strip": flat, 2-6 cm in front of the PC's port panel, plug toward the PC (Part 1 strip).
- Grippers fully open in every frame (the right gripper half-open is a separate known bias: 3 mm).
- `pose` moves the arms with the safety checks, logs the joint readings in `<tag>.json`, takes the ZED pair
  and both wrist frames, then returns the arms to where they started.
- Pressing (left fingertips on the PC top, 2 cm in front of the case centre, 25 N) needs the real hold script
  (not written yet): skip, or take it later with the hold.

**C. Grasp pull/slip test and in-hand pose (Part 1, ~20 min)**
1. Grasp the reader as in Part 1 (55° tilt, fingers on the side faces near the body front, 12 N squeeze):
   place it in the open jaws by hand, then
   `scripts/real/measure_grasp.py --session data/real_yam/usbc_lab_20261027 --name grasp55_NN --hold-seconds 10`.
2. In-hand pose: with the reader held still, `snapshot.py capture --out data/grasp55 --tag graspNN` (right
   wrist + ZED). Repeat 10-20 grasps (re-place the reader each time, ±5 mm / ±10° placement).
3. Pull/slip: with the reader held and the arm still, pull the reader slowly along its axis (plug direction)
   with a spring scale on a thread around the neck; record the force when it slips (or 15 N max). Five pulls,
   also five pulls across (vertical). Write the forces to `data/grasp55/pull.csv`.

**D. After the session**
Commit `data/zed_gt` (JSON + PNG, small) and `data/grasp55`. Sim side: score the ZED nets on every frame
against the spot ground truth (`scripts/perception/check_real_zed.py`, extend for `data/zed_gt`), and fit the
in-hand spread from the wrist frames.
