# Real YAM rig tools (duchamp3)

Two YAM arms on CAN, a webcam on each wrist, a ZED 2 above the table. Results and decisions are in
`docs/FINDINGS.md` (2026-10-24 to 26 entries); open real-vs-sim gaps in `docs/REAL_RIG_GAPS.md`. The camera
spec (intrinsics, mounts, ZED pose, joint offsets, base-to-base pose) is
`blind_insertion/configs/real_cameras_duchamp3.json`.

## Next bimanual measurement session

See [BIMANUAL_MEASUREMENT_PLAN.md](../../docs/BIMANUAL_MEASUREMENT_PLAN.md) for the
prioritized remaining measurements, verified existing commands, tests needing a runner
extension, and the data handoff format. It supplements the historical gap table.

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
| `measure_manual.py` | create a session; record poses, masses, dimensions, forces and timed events | - |
| `measure_cameras.py` | capture both wrists and the ZED with frame timing and raw images | - |
| `measure_motion.py` | bounded friction trajectories and coordinated two-arm timing | `single`/`bimanual` |

## Bimanual measurement session

The detailed protocol is in [BIMANUAL_MEASUREMENT_PLAN.md](../../docs/BIMANUAL_MEASUREMENT_PLAN.md).
These new scripts exclude the gripper measurements. All positions passed to `measure_motion.py`
are **raw encoder joint radians**. A motion command starts only with `--run`; `plan` and
`measure_manual.py` are offline. The motion runner expects both tested arms at folded rest
(joints 2 and 3 within 0.05 rad of zero) and returns them there before torque off. Keep the
console open, both bases clamped, and the reader and PC away from free-space paths. Review
each start, excursion and return path and the other arm's clearance before running.

```bash
SESSION=data/real_yam/bimanual_session01
.venv/bin/python scripts/real/measure_manual.py init --out "$SESSION" \
  --session-id bimanual_session01 --operator "$USER" --mounts 'bases clamped' \
  --surface 'desk surface; describe condition'

# Manual instrument readings: SI units and honest uncertainty. Use new trial IDs for repeats.
.venv/bin/python scripts/real/measure_manual.py measure --session "$SESSION" \
  --item reader --quantity mass --value 0.03 --unit kg --uncertainty 0.001 \
  --method scale --trial-id scale_01
.venv/bin/python scripts/real/measure_manual.py pose --session "$SESSION" \
  --item target_port --trial-id survey_01 --parent right_arm_base --child port_2 \
  --pos 0.4 0.0 0.1 --quat 1 0 0 0 --method board_survey --uncertainty 0.001
.venv/bin/python scripts/real/measure_manual.py force --session "$SESSION" \
  --kind insert --item reader_port --trial-id insert_01 --time 0.5 \
  --depth 0.002 --force 1.2 --direction insertion --powered no

# Capture settings and every arrival time; writes images for every frame by default.
.venv/bin/python scripts/real/measure_cameras.py --session "$SESSION" \
  --name stationary_01 --seconds 20 --save-every 5

# In a separate terminal: .venv/bin/python scripts/real/rigctl.py
.venv/bin/python scripts/real/measure_motion.py plan --mode triangle \
  --amp 0.02 --rate 0.01 --reps 5
.venv/bin/python scripts/real/measure_motion.py single --session "$SESSION" \
  --name left_j3_triangle_01 --side left --mode triangle --joint 3 \
  --amp 0.02 --rate 0.01 --reps 5 --run
.venv/bin/python scripts/real/measure_motion.py bimanual --session "$SESSION" \
  --name both_hold_01 --mode hold --seconds 30 --infer --run
```

The numerical manual readings above only illustrate command syntax; replace them with
instrument readings. `measure_manual.py event --now` can mark a manual LED/trigger event;
use a hardware trigger timestamp if absolute camera timing matters. `measure_motion.py`
supports `ramp`, `triangle`, and `dwell` on either arm; use `--start` (or `--start-left` /
`--start-right`) only with a locally checked clear pose. Bimanual motion also requires
`--moving-side` and `--motion-reviewed`. The runner owns both arms in one process and
captures all three cameras by default; use `--no-cameras` for an arm-only timing run.

Outputs are within the session's `motion/` and `cameras/` folders. Motion `steps.csv`
contains 50 Hz command/observation rows and CAN send timestamps; `approach.csv` records
approach targets and feedback, including partial approaches, and `encoder_<side>.csv`
contains native feedback arrivals. The driver effort and temperature arrays are logged
when available, but effort is **not calibrated output torque**. Wrist exposure timestamps
are unavailable. ZED image timestamps and host monotonic time are separate clocks.
`policy_ready_ns` measures camera preprocessing completion; policy consumption time is
unavailable in this capture tool. Wrist frames have no hardware frame IDs, so exact
hardware drop counts cannot be recovered; `local_frame_id` is the capture sequence and
`exact_repeat_of_previous` flags identical images. Sampled ZED rectified-left and wrist
frames include matching 320×320 policy crops by default. Camera workers report readiness
before the capture window; sampled PNG writes are buffered and a writer backlog or failure
is reported as an incomplete capture. Actual force requires a gauge.
The software stop holds
the arms; the operator clears it in the console before the script can return them to rest.

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
