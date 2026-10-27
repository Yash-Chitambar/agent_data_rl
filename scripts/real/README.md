# Real YAM rig tools (duchamp3)

Two YAM arms on CAN, a webcam on each wrist, a ZED 2 above the table. Results and decisions are in
`docs/FINDINGS.md` (2026-10-24 to 26 entries); open real-vs-sim gaps in `docs/REAL_RIG_GAPS.md`. The camera
spec (intrinsics, mounts, ZED pose, joint offsets, base-to-base pose) is
`blind_insertion/configs/real_cameras_duchamp3.json`.

## Before anything moves

1. Clamp both arm bases to the table (an unclamped base nearly tipped at ~0.58 m reach).
2. Start the rig console: `.venv/bin/python scripts/real/rigctl.py` (space = STOP, c = clear, o/f arms
   on/off, r/l one arm, h home, v camera viewer, q parks the arms and quits). `estop.py` is the minimal
   alternative (Enter = STOP, c + Enter = clear).
3. Every script connects arms through `safety.connect` (PD hold from the first instant) and moves them
   through `safety.SafeArm` (joint box, table clearance, step and deviation caps).

## Tools

| Script | What | Moves? |
|---|---|---|
| `setup_real.sh` | install the real-arm deps (i2rt, python-can, pyrealsense2, pyzed) into `.venv` | - |
| `setup_can_autostart.sh` | one-time sudo: udev rule bringing the CAN adapters up at boot / plug-in | - |
| `rig.py` | rig config: arm -> CAN interface by adapter serial, wrist camera devices, joint offsets | - |
| `safety.py` | hold-mode connect, `SafeArm`, `close_robot` (no i2rt shutdown race) | - |
| `estop.py` | software stop terminal (stop file `/tmp/yam_estop`) | - |
| `rigctl.py` | rig console: stop, arms on/off (park at rest, then limp), camera viewer | yes (to rest) |
| `camview.py` | live browser view of all cameras (viser; tailnet + optional public link) | - |
| `camshare.py`, `armshare.py` | handoff of cameras / arms between the console + viewer and scripts | - |
| `calib.py` | camera calibration: `zed`, `find`, `collect`, `solve` (wrist intrinsics, hand-eye, joint offsets) | yes |
| `verify_cameras.py` | all three cameras on one board at a new spot, nothing fitted: the calibration check | yes |
| `fit_zed.py` | refit the ZED pose from several `verify_cameras.py` runs | - |
| `snapshot.py` | scene reference images for the sim (`capture`, or `point` both wrists at a table point) | `point` |
| `deploy.py` | 50 Hz deployment loop (sim action scheme, gains, offsets, CAN-level timing); `check` offline | `run` |
| `yam_latency_fit.py` | fit delay, gain and friction from `deploy.py` step logs | - |

## Common runs

```bash
.venv/bin/python scripts/real/deploy.py check                                   # offline, no arm
.venv/bin/python scripts/real/deploy.py run --side left --source hold --seconds 60 --infer --name left_hold_60s
.venv/bin/python scripts/real/deploy.py run --side right --source step --joint 3 --amp 0.04 --reps 10 \
    --start 0.0 1.2 1.0 -0.6 0.0 0.0 --name steps_joint3
.venv/bin/python scripts/real/deploy.py run --side right --source policy --policy runs/<run>/last.pt \
    --env_file blind_insertion/configs/<its config>.json --seconds 20 --name <name>
.venv/bin/python scripts/real/verify_cameras.py --out data/calib/verify_<name>  # board on the table first
.venv/bin/python scripts/real/fit_zed.py data/calib/zedfit/pos1 data/calib/zedfit/pos2 data/calib/zedfit/pos3 --write
```

Logs: `data/real_yam/` (step and hold tests, timing), `data/calib/` (calibration runs, gitignored),
`data/sim_ref/` (scene reference images), `results/real/` (fits and checks).

The same calibration, console and viewer tools for any YAM cell (rig file per cell) are in the
`yam-control` fork: github.com/Yash-Chitambar/yam-control, `calibration/`.
