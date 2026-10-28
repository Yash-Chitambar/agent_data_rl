# Task H on the real right YAM: ACT pick -> handoff -> blind insertion -> release

One 50 Hz loop (`runner.py`) that runs the sim pipeline (`blind_insertion/pipeline/pick_insert.py`)
on the real arm. The handoff gate, ACT temporal ensembling and the seat detector are one module
(`blind_insertion/pipeline/handoff.py`) used by both, so sim and real cannot drift. `scripts/real/`
keeps the low-level tools (safety, estop, rig, calib, step tests); this folder is the task.

| File | What |
|---|---|
| `config/task_h.json` | every number the runner uses; the sim-derived ones are checked against the sim |
| `models.py`, `models/MANIFEST.json` | stage the evaluated checkpoints into `models/` (gitignored), with sha256 |
| `runner.py` | the state machine; `sim` = full dry run on the sim, `real` = the arm |
| `stages.py` | ACT stage, insertion stage, release (mirror ActPick / Inserter / Release) |
| `conventions.py` | joint frames, gripper map, FlexPoint kinematics |
| `backends.py` | `RealBackend` (scripts/real/deploy.RealArm + cameras) and `SimBackend` (one PickEnv world) |
| `cameras.py` | ZED top + right wrist webcam capture threads, frame age, policy crops |
| `safety_task.py` | FlexChecker: safety.Checker on the FlexPoint YAM, model frame, task clearances |
| `checks/` | offline and rig checks, below |

## Conventions (checked by `checks/conventions.py`)

- **Joints.** motor = what i2rt reads. model = motor + `rig.joint_offsets("right")` (0.73, 1.33,
  -1.11 deg on joints 2, 4, 5). ACT state and targets are model frame: send `target - offsets`.
  Insertion observation is model frame; its target is `q_motor + 0.01 a` (offsets cancel).
- **Gripper.** sim +1 open / -1 close (constant force) -> i2rt 1 / `gripper.close`. The finger reading
  (i2rt 0..1) maps to the sim's mean finger slide in m (open 48 mm, on the reader ~19 mm, empty ~-6 mm).
  Uncalibrated until `checks/gripper_cal.py`.
- **Kinematics.** FlexPoint gripper (the one on the arms). `scripts/real/deploy.py`'s ObsBuilder loads
  the linear gripper (different tcp): not used for this task.
- **Insertion policy.** scale 0.01, history 10, clip 6 (from its run config, staged next to it).

## Before the first run (in this order)

1. On mason: `python -m deploy.models stage --act <chosen ACT> --insert runs/H-reader-tilt-uncomp-s1/last.pt`;
   copy `deploy/models/` to duchamp3; there `python -m deploy.models verify`.
2. `python deploy/checks/handoff_equiv.py` (shared module = the logic the sim evals used).
3. `python -m deploy.checks.conventions` (GPU) and `python -m deploy.runner sim --episodes 16 --name dry`
   (full dry run through the same runner code; also replays every target through SafeArm's rules).
4. **Timing gate.** `python -m deploy.checks.timing` on duchamp3: ACT p99 must be < 12 ms (the tick
   is 20 ms, the loop takes ~4 ms). On a busy mason V100 it is 22 ms (model) / 41 ms (step). If the
   4090 is too slow, ACT moves to a thread at a lower query rate, which needs a sim eval first.
5. `python -m deploy.checks.timing --cameras`: ZED and wrist fps, frame age at use. The demos assume
   30 fps and 30-100 ms true age (logged arrival age ~0-70 ms).
6. `python -m deploy.checks.gripper_cal`: gripper map; set `gripper.close` for a 9-15 N squeeze.
7. Place the PC and reader (below). Clamp both arm bases. Start `scripts/real/estop.py`.
8. `python -m deploy.runner real --name try1`. Logs: `data/task_h/<name>/` (every tick: joints,
   phase, targets, ACT latency, frame age, insertion actions) + `summary.json`.

## Scene placement (right arm base frame, table top z = 0, +x forward, +y toward the left arm)

| Thing | Position | Note |
|---|---|---|
| USB-C port (2nd from left) | x 321 mm, y 237 mm, 25 mm above the table | 399 mm from the base, 36 deg toward the left arm |
| PC centre | x 370 mm, y 215 mm, yaw 2.3 deg | rear panel faces the arm, at x = 321 mm |
| Reader centre | x 204 mm, y 221 mm | flat, plug toward +x (the PC) |

Trained ranges: port +-2 cm x, +-4 cm y, PC yaw +-10 deg; reader +-3 / +-5 cm, yaw +-20 deg. The
handoff gate's pre-insert point (`handoff.p_nom`) assumes the nominal PC: place it within ~1 cm.

## Open items

- **ACT on real images**: sim-only, closed with domain randomization if it fails (no real demos).
- **Release on the real arm**: see FINDINGS 2026-10-28; the insertion policy keeps pushing after the
  plug seats (never trained past success), so the release trigger matters.
- **Squeeze force** and the gripper map: measured on the rig (step 6).
- **Safety clearances**: provisional (`safety.*`); the fingers must reach the table for the grasp.
