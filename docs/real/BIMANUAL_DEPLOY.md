# Bimanual USB-C chain on duchamp3: first rig test

Runner: `deploy/run_bimanual.py` (code in `deploy/bimanual/`). It runs the sim chain (99.56-99.66 % in sim,
FINDINGS 2026-11-01) on the real arms, staged. The dry run (`--dry-run`) runs the same code on one eval_chain sim
world. Every run writes `data/real_runs/<date>/<stage>_<time>[_name]/` (format: `deploy/bimanual/runlog.py`).

| Stage | What moves |
|---|---|
| `capture` | Both arms go to the ZED capture pose with the grippers open. The ZED frame gives the PC estimate. The left arm parks. The ZED frame gives the reader estimate (one re-take). Then both arms go home. |
| `hold` | `capture`, then the left arm parks and does the guarded press on the PC top, 2 cm in front of the case centre (`--hold-force`, `--hold-seconds`). It unloads and goes home. |
| `approach_insert` | `capture`, then the right arm goes to the post-pick lift pose with the jaws open 24 mm. You put the reader in and it closes. Then: hold, transport, wrist-servo approach, in-hand estimate, torque hand-over, K-J insertion, seated detector or 10 s limit, open, retreat, unload, home. |
| `pick` | `capture`, then the servo-descent pick and the lift. You take the reader and it goes home. |
| `full` | `capture`, pick, then the rest as in `approach_insert`. |

## Safety rules (in addition to CLAUDE.md)

- Clamp both bases. Run `scripts/real/estop.py` in a second terminal: Enter there holds the arms.
- The runner connects only through `scripts/real/safety.py` (PD hold from connect). Every command goes through SafeArm:
  joint box, FlexPoint table clearance (tips -15 mm, bodies 40 mm), 0.1 rad step cap and 0.4 rad deviation limit.
- Runner guards (`deploy/bimanual/guards.py`, values in `deploy/config/bimanual.json` `safety`):
  - PD torque estimate kp (target - q) + feed-forward. Stop above 12 N m (joints 1-3) or 5 N m (4-6) for 0.1 s.
    Stop after 20 s in total above 9 / 3 N m.
  - Left press: 60 s at most.
  - Joint speed: 2.5 rad/s at most.
  - TCP box of both arms.
  - `--max-steps`.
- **Software stop:** press Enter in the runner terminal while a stage moves, or press Enter in estop.py.
  Both arms then hold their measured pose (zero PD error: no press force, no push).
  The runner then asks: `u` unload the press, `h` home, `r` both, `x` stay.
- At the end the arms hold powered until you type `SUPPORTED`. Support the arms first.
- Torque at 25 N: the left joint 2 needs 8.2 N m (sim plan: dq 0.103 rad x kp 80). This is near the ~9 N m rating.
  Use `--hold-force 15` first (about 5 N m).
- The prompts before hold/transport/approach/insert are one prompt, before the press starts. The press does not wait
  for you. `--pause-in-press` adds prompts during the press. A watchdog then holds both arms (press off) at 60 s.

## Night before (mason; done 2026-11-01)

```bash
cd /home/yashchitambar/wt/real-deploy        # branch real/bimanual-deploy
export PYTHONPATH=$PWD MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2
.venv/bin/python -m deploy.bimanual.bundle stage --tar     # deploy/bundle/ + deploy/bundle.tar (140 MB, MANIFEST.json)
.venv/bin/python -m deploy.bimanual.bundle copy-cmd
```

## Morning on duchamp3

### 0. Code, bundle and environment

```bash
# on mason: copy the bundle (user runs this; <rig-repo> = the real/duchamp3 checkout on duchamp3)
rsync -av --progress /home/yashchitambar/wt/real-deploy/deploy/bundle/ <user>@duchamp3:<rig-repo>/deploy/bundle/
# on duchamp3
cd <rig-repo> && git fetch origin && git checkout real/duchamp3 && git pull   # (or real/bimanual-deploy)
export PYTHONPATH=$PWD MUJOCO_GL=egl
bash scripts/real/setup_real.sh                      # only if i2rt / pyzed are missing (uv sync removes them)
.venv/bin/python -m deploy.bimanual.bundle verify    # sha256 of every file
.venv/bin/python -m deploy.bimanual.bundle link      # runs/perception/*.pt, runs/chain/ -> the bundle
.venv/bin/python scripts/real/rig.py                 # both CAN adapters up
```

### 1. Offline checks on the rig (no motion, ~10 min)

```bash
.venv/bin/python -m deploy.checks.bimanual_conventions --out data/real_runs/checks/conventions.json   # sim == real path
.venv/bin/python -m deploy.checks.bimanual_nets --real --out data/real_runs/checks/nets.json          # shapes, latency
.venv/bin/python -m deploy.checks.bimanual_timing --out data/real_runs/checks/timing.json             # TIMING GATE
.venv/bin/python scripts/real/handeye.py selftest
.venv/bin/python -m deploy.run_bimanual --dry-run --stage full --seed 7000                            # whole chain in sim
```

- **Timing gate.** `bimanual_timing` must show p99 compute below ~15 ms per tick in pick, transport and insert.
  The tick is 20 ms. On loaded mason it was pick 12.6 / 24 ms, transport 7 / 32 ms, insert 4.3 / 8.7 ms (p50 / p99).
  Approach has single slow ticks at each servo waypoint (IK). The arm holds the last target meanwhile.
  If the gate fails, use `--device cpu --net-device cuda:0` (the default) and close other GPU users.
  A slow tick only makes the scripted motion slower (the loop does not catch up).
- `bimanual_nets --real` puts a real capture-pose frame (2026-10-25, no PC at the bimanual spot) through the ZED net.
  Expected result: gate REJECT, presence about 0.

### 2. Capture protocol: validate the ZED nets before task motion (PHYSICAL_TODO section 6, ~45 min)

Rig console running, bases clamped, estop.py open.

1. Mark 4 PC spots: nominal (port 2 at x 0.22, y 0.36 m in the right base frame, yaw 0), +1 cm x, +1.5 cm y,
   and -1 cm x / +5 deg. Measure each one. Write the results to `data/zed_gt/spots.json` as
   `{"spot1": {"port_xy": [x, y], "yaw_deg": d, "method": "...", "unc_mm": 1}}`.
2. For each spot (PC on its marks, reader in the strip):
   ```bash
   O=data/zed_gt/spot1
   .venv/bin/python scripts/real/snapshot.py pose --out $O --pose folded  --tag folded_reader
   .venv/bin/python scripts/real/snapshot.py pose --out $O --pose capture --tag capture_reader
   .venv/bin/python scripts/real/snapshot.py pose --out $O --pose parked --side left --tag parked_reader
   # take the reader away, then run the three lines again with tags *_noreader
   ```
3. Score the nets (no motion):
   ```bash
   .venv/bin/python -m deploy.checks.bimanual_real_frames data/zed_gt/spot* --spots data/zed_gt/spots.json --out data/zed_gt/score.json
   ```
   - Expected: `capture_*` rows have `pc_gate` true and `pc_err_xy_mm` within a few mm (sim p99 4 mm).
   - The `folded` / `parked` rows must be REJECTED by the arm gate.
   - `capture_noreader` must have `reader_gate` false.
   - **Stop here** if the capture-pose PC error is > 5 mm or the gate rejects good frames. The pick and the
     approach then start from a bad port estimate.

### 3. Optional: wrist-camera hand-eye (~30 min per arm; recommended for the right arm before `pick`)

The wrist-camera mount error is the accuracy floor of the pick servo. The sim randomizes it up to 5.1 mm / 1.2 deg
(right) and the image does not show it. `scripts/real/calib.py` already collects board views and solves AX = XB.
`scripts/real/handeye.py` refits the mount with the spec's joint offsets fixed and reports the residual.
`--write` stores the result in the spec. Do not move the board during the session.

```bash
.venv/bin/python scripts/real/calib.py find --side right                 # MOVES: finds the board (on the table, ~30 cm in front)
.venv/bin/python scripts/real/calib.py collect --side right --n 40       # MOVES
.venv/bin/python scripts/real/calib.py solve --side right --run data/calib/run_NNN
.venv/bin/python scripts/real/handeye.py apply --side right --run data/calib/run_NNN          # read the report
.venv/bin/python scripts/real/handeye.py apply --side right --run data/calib/run_NNN --write  # if it is good
```

- Good: `holdout_mean_px` <= ~3 px and `fold_spread_mm` <= 1 mm. The 2026-10-25 calibration had 2.7 px hold-out.
- `change_vs_spec_mm/deg` is the mount change against the spec. If it is beyond the DR range (`within_dr` false),
  the servo nets were not trained for this camera. Do not run `pick` / `full`; tell the sim side.
- `--write` updates `blind_insertion/configs/real_cameras_duchamp3.json` (`wrist.<side>.pose_in_link6`,
  `wrist.sim.pose_in_link6` for the right camera, `calibration_version`) and keeps a backup.
  Every run's meta.json records `calibration.calibration_version` and the wrist poses.
  The runner itself does not use the mount at run time (the nets do). Commit the spec.

### 4. Stages, in order

Use `--name` and `--notes` on every run. Use `--pc-spot '<json of the spot>'` when the PC is on a measured spot.

```bash
R=".venv/bin/python -m deploy.run_bimanual"
# (a) captures only (arms go to the capture pose and back)
$R --stage capture --name capA --pc-spot '{"port_xy": [0.22, 0.36], "yaw_deg": 0}'
# (b) left hold alone: 15 N first, then 25 N
$R --stage hold --hold-force 15 --hold-seconds 5 --name hold15
$R --stage hold --hold-force 25 --hold-seconds 5 --name hold25
# (c) approach + insertion, reader placed in the jaws by hand
$R --stage approach_insert --speed 0.5 --name ai1
# (d) pick alone
$R --stage pick --speed 0.5 --name pick1
# (e) full chain
$R --stage full --speed 0.5 --name full1
```

What to watch and when to abort (Enter in the runner terminal):

- **capture.** The printed PC estimate should be within ~5 mm of the spot and the gate must be True.
  If it is rejected, the runner offers one re-take; do not continue on a rejected estimate.
  The reader estimate should be on the reader.
- **hold.** The left fingertips come down on the PC top, 2 cm in front of the case centre, at 1 cm/s.
  They should latch within ~2.5 cm. The PC must not slide.
  Watch `hold.npz` `Fn_est` (printed at the end: about the commanded force).
  Abort if the hand goes for the edge of the case or the PC moves.
- **approach_insert.** The hand reaches the look pose 3 cm in front of the port, then servos to 6 mm.
  The printed in-hand estimate should be within ~2 mm / 2 deg of 0 for a nominal placement.
  The insertion should seat within ~1-2 s. The runner stops the push when the seated detector fires.
  The plug advances past the 6 mm gap by >= 4.8 mm; the speed is held < 5 mm/s for 0.2 s.
  Abort if the plug hits the panel off the port (lateral miss) or the PC moves.
- **pick.** Hover above the reader, then three servo frames on the way down (re-hover if the hover correction is
  > 8 mm). The fingers close on the side faces, then it lifts 8 cm. Abort if the fingers come down on the reader's top.
- **full.** As above. Between pick and hold there is one prompt; after that it runs through.

Speed: `--speed 0.5` halves the scripted motions (pick, transport, approach ramps, retreat). The sim used 1.0.
Policies run at 50 Hz regardless.

### 5. After each run

The runner asks for the outcome (s/p/f/a) and free-text failure labels. Both go to meta.json.
Commit nothing large from `data/real_runs` (gitignored). Sync the logs back to mason:

```bash
# on mason, repo root (rsync --update; nothing local is deleted)
rsync -az --update --info=stats1 <user>@duchamp3:<rig-repo>/data/real_runs/ data/real_runs/
# replay a run in the sim (tracking of every sent command; friction x latency grid)
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/real2sim/replay_real_run.py data/real_runs/<date>/<run> --friction 0.5 1.0 1.5 --latency 0 1
# real frames + FK labels -> datasets and net errors
.venv/bin/python scripts/real2sim/export_real_dataset.py data/real_runs/<date>/* --out data/real_datasets/<date>
```

## Open items the operator must decide or measure

- **Gripper squeeze force is not calibrated.** The real close target is i2rt 0.0; the sim squeezes 12 N.
  The reader band (`g_empty` 0.015 / `g_reader` 0.254) was measured on 2026-10-28 for the top-down grasp.
  Re-check it with the 55 deg side grasp (`deploy.checks.gripper_cal`). The left gripper map is uncalibrated
  (open/close only).
- **Hold force.** Is 25 N acceptable for the left joint 2 (~8.2 N m)? Start at 15 N. The policy was trained with 25 N.
- **Joint offsets.** The sim chain is trained with +/-1.5 deg offset DR. A fresh `calib.py solve` reports fitted
  offsets: if they differ by > 1 deg from the spec, update the spec offsets before the pick.
- **Camera preprocessing.** It is the established `real_policy_image` / `real_top_policy_image`. The FOV matches the
  sim within 0.01 deg (`bimanual_conventions`). The look (colour, exposure) on today's light is checked only
  through the step-2 captures.
- **Friction feed-forward.** The FF in SafeArm.command was lost in a merge and is restored on this branch.
  The i2rt attributes it writes (`_command_lock`, `_commands.torques`) could not be checked on mason (no i2rt).
  If they are missing, the first approach tick raises.
