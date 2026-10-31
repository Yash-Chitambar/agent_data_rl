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

The single-arm task-H deploy path uses `pc_bench.build_scene(table_z=REAL_TABLE_Z)`; the
bimanual policy deploy runner does not exist yet.
