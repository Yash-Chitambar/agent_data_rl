# Physical rig TODO (2026-10-30)

What the sim still needs from the real YAM rig (duchamp3), in priority order. Items marked
**sim side** need no rig time. Background: `results/usbc/bimanual_20261029/README.md`
(identified arm dynamics, fingertip fit, runtime speed).

## 1. Gripper opening sweep (today, ~5 min, jaws only)

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
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/tools/fit_gripper_tips.py --sweep data/gripper_sweep/sweep_20261030
```

Accept the offsets if both arms reach held-out IoU >= 0.8 and the map is monotonic with
<= 0.5 mm RMSE; then the runtime uses them via
`BimanualCfg(tip_offset='results/usbc/gripper_sweep/sweep_20261030/fit.json')`.
If the offset still disagrees between arms by more than ~2 mm, suspect the hand-eye
calibration rather than the tips, and redo `scripts/real/calib.py` for that wrist.

## 2. PC-on-table friction (~10 min, no robot motion)

**Why.** The PC is a free 284 g body; insertion pushes on it. The sim friction is a guess.

1. PC on the table at its task position, on its own feet, nothing else touching it.
2. Tie a thread around the PC near its feet height; pull horizontally with a spring/luggage
   scale (or a kitchen scale used as a force gauge), slowly.
3. Record the peak force just before it slides (static) and the steady force while it slides
   (kinetic). 5 pulls toward the robots, 5 sideways.
4. Note whether it slides or tips/rocks first, and photograph the feet.

Write the 20 numbers + notes into `data/real_yam/pc_friction_20261030.csv`
(`direction, trial, static_N, kinetic_N, behaviour`) and commit it (small file).
mu = force / (0.284 kg x 9.81).

## 3. Joint backlash (later; script to come)

**Why.** Phase C showed backlash of 1-5 arcmin decides sim-to-real success; it is not measured
on these arms. The encoders sit on the motor side and cannot see it.

**Method (needs a script I will write when you are ready):** fix the calibration board on the
table, point one wrist camera at it, step one joint back and forth by small amounts and compare
the board motion in the image (link side) with the encoder (motor side). The dead band at each
reversal is the backlash. Needs: board fixed in place, ~20 min per arm.

## 4. Already measured: sim side only (no rig time)

These exist in the single-arm env (`EnvCfg`) and must be ported to the bimanual runtime:

- [ ] Action latency: measured ~8 ms (0.4 control step); randomize 0-1 step.
- [ ] Encoder-offset noise: +/-1.5 deg on joints 2-5 (`EnvCfg.enc_offset_deg`).
- [ ] Backlash model (`builder.add_backlash`) with a 1-5 arcmin range until item 3 is measured.
- [ ] PC-table friction randomization (range from item 2; until then mu 0.3-0.8).
- [ ] Rigid insertion contact with randomized plug-port friction and pad friction/stiffness.

## 5. Optional, after the new grasp is chosen

10-20 grasps of the reader with the new grasp, holding still 1 s after each (wrist cameras +
ZED). Gives the in-hand reader pose spread for the reset distribution. Until then the sim uses
a generous in-hand pose randomization.
