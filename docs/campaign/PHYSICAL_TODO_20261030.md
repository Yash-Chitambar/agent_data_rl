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
