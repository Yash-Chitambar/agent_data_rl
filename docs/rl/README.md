# RL: bimanual runtime, policy interface, training

## Runtime

`blind_insertion/envs/bimanual_runtime.py` (`BimanualRuntime`, `BimanualCfg`): both arms, free
PC and reader in MuJoCo Warp; 50 Hz control, 20 substeps replayed from a CUDA graph
(~28k env-steps/s at 2048 worlds, ~44k at 8192 on a V100; check `buffer_usage()`).
Checks: `scripts/checks/check_bimanual_runtime.py` (control, resets, randomization, hold drift).

Per-world randomization, resampled on reset: arm friction and gravity scale, PC-feet friction,
PC pose and reader pose (below). Still to add: action latency, encoder offsets, contact DR
([CURRENT](../campaign/CURRENT.md)).

## Task layout

`scripts/task/explore_pc_placement.py` searched 25,688 PC poses for both arms' reach (right arm
inserts along the full approach line with ±8° in-hand tilt; left arm braces or presses the
PC; no collisions; joint margin). Output `results/usbc/bimanual_20261029/placement/report.json`,
read by `BimanualCfg.task_layout`:

- PC port nominal (0.22, 0.36) m, yaw 0, right-arm base frame; per episode ±3 cm, ±10°
  (the nominal maximizes the worst score over that box). The legacy `PORT_XY` is infeasible.
- Reader: every (2 cm cell, 20° yaw bin) where a top-down pick works, clear of every PC pose,
  on the table and off the arms: 0.20 m², yaw ±90°, including in front of the PC.

## Policy interface

[BIMANUAL_POLICY_INTERFACE](BIMANUAL_POLICY_INTERFACE.md): 14 actions (per arm 6 joint increments
+ gripper), accumulated joint targets capped at 0.1 rad from the encoders; observations are
encoders, velocities, previous actions and target errors (+ optional RGB); object poses are
privileged (critic only). `blind_insertion/envs/bimanual_task.py`.

## Training

PPO and the TCN actor-critic with the contact decoder come from the single-arm work
(`blind_insertion/train.py`, `train_visual.py`, `policies/`). A bimanual training entry point
is the next step. Validation status of grasp, carry, insertion and GPU parity:
[BIMANUAL_TRAINING_HANDOFF](BIMANUAL_TRAINING_HANDOFF.md). Server runs: [SERVER](../operations/SERVER.md).
