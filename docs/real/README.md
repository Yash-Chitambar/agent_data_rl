# Real rig (duchamp3): branch `real/duchamp3`

This branch is what runs on the YAM PC. It is `main` (the same simulator, scene and configs)
plus the rig-only code. Update it with `git merge main`; fix shared code on `main` first
([branches](../operations/BRANCHES.md)).

| Path | What |
|---|---|
| `scripts/real/` | Safety layer (`safety.py`, connect arms only through it), rig console `rigctl.py`, `estop.py`, camera calibration `calib.py`, snapshots, measurement tools (`measure_backlash.py`, `capture_gripper_sweep.py`), single-arm deployment loop `deploy.py`. See [scripts/real/README](/scripts/real/README.md). |
| `deploy/` | Task runners and offline/rig checks: single-arm task H (ACT pick → PPO insert → release), guarded bimanual contact commissioning. See [deploy/README](/deploy/README.md), [DUCHAMP3_COMMISSIONING](/deploy/DUCHAMP3_COMMISSIONING.md), [CONTACT_COMMISSIONING](/deploy/CONTACT_COMMISSIONING.md). |
| `docs/real/` | Rig session docs ([LEFT_CONTACT_COMMISSIONING](LEFT_CONTACT_COMMISSIONING.md)). |
| `docs/campaign/PHYSICAL_TODO_20261030.md` | What the sim still needs from the rig. |

Rules on the rig: clamp the arm bases before any motion; run `scripts/real/estop.py` (or
`rigctl.py`) first; connect arms only through `scripts/real/safety.py`. Real-arm Python
dependencies: `bash scripts/real/setup_real.sh` (not in `uv.lock`).

Data: small results (calibration JSON, measurement summaries) are committed here and merged
into `main` when the sim uses them; large raw captures go to a `data/<name>` branch.

The single-arm task-H deploy path uses `pc_bench.build_scene(table_z=REAL_TABLE_Z)`. The bimanual
USB-C chain runner is `deploy/run_bimanual.py` (`deploy/bimanual/`). The first rig test is described in
[BIMANUAL_DEPLOY](BIMANUAL_DEPLOY.md).
- [SESSION_20261102_duchamp3.md](SESSION_20261102_duchamp3.md): batch A result, why the insertion fails, real wrist data and its label caveat, server work list.
- [PICK_GRASP_V3.md](PICK_GRASP_V3.md): new right-arm pick, real test batches A-D, sim results and their limits.
- [PORT_VISION.md](PORT_VISION.md): classical port and plug measurement from the wrist camera (`deploy/bimanual/port_vision.py`), results on real frames, the label offset.
- [CONTACT_ALIGN_PLAN_20261107.md](CONTACT_ALIGN_PLAN_20261107.md): why the plug stalled on the case face; the K2/K3/K4 contact-policy plan.
- [K2_PROGRESS.md](K2_PROGRESS.md): progress log for the contact policies (K-J, K3, K4, K5w), the alignment stage, the servo, the retry and the RMA study.
- [ALIGN_STAGE.md](ALIGN_STAGE.md), [PRECISION_CONTROL.md](PRECISION_CONTROL.md): the step alignment before the hand-over, and the precision controller.
- [RETRY_SPEED_20261108.md](RETRY_SPEED_20261108.md): insertion retry, servo speed sweep, backlash preload (sim only, not run on the rig).
- [NEXT_RIG_20261109.md](NEXT_RIG_20261109.md): the runbook for 2026-11-10 (done; status at the top).
- [SESSION_20261106_duchamp3.md](SESSION_20261106_duchamp3.md), [SESSION_20261110_duchamp3.md](SESSION_20261110_duchamp3.md), [SESSION_20261111_duchamp3.md](SESSION_20261111_duchamp3.md): rig sessions; 11-10/11 is the K4 milestone (10 of 10 seated at the port).
- [SERVER_TASKS_20261112.md](SERVER_TASKS_20261112.md): server work after the milestone (data review, full-depth push, faster alignment, placement and grasp robustness).
- [SESSION_20261101_duchamp3.md](SESSION_20261101_duchamp3.md), [SESSION_20261103_duchamp3.md](SESSION_20261103_duchamp3.md): first bimanual session; first live align run (stopped on the approach jump).
- [PICK_GRASP_SIM_HANDOFF_20261101.md](PICK_GRASP_SIM_HANDOFF_20261101.md), [GRASP_FIDELITY.md](GRASP_FIDELITY.md), [PICK_BATCH_RUNBOOK.md](PICK_BATCH_RUNBOOK.md): the real pick failure, the sim grasp model that fixed it, and the pick batch runbook.
- [INHAND_REAL.md](INHAND_REAL.md), [WRIST_NETS_REAL.md](WRIST_NETS_REAL.md): real in-hand pose and reader look in the sim; wrist nets on real frames (r10).
- [NEXT_RIG_SESSION.md](NEXT_RIG_SESSION.md), [ANSWERS_20261103.md](ANSWERS_20261103.md): older runbooks (align stage, 2026-11-03/06); kept for history.
- [DATA_REVIEW_20261113.md](DATA_REVIEW_20261113.md), [FULL_DEPTH_20261113.md](FULL_DEPTH_20261113.md), [ALIGN_FAST_20261113.md](ALIGN_FAST_20261113.md), [ROBUSTNESS_PLAN_20261113.md](ROBUSTNESS_PLAN_20261113.md): server work after the milestone (real-run review, full insertion depth, fast alignment, placement robustness + wrist-net bias + approach jump).
- [NEXT_RIG_20261114.md](NEXT_RIG_20261114.md): **the current rig runbook** (one change per block, then `bimanual_align9_all`).
