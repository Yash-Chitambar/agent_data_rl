# Current campaign (updated 2026-10-31)

Task: right YAM picks the SanDisk USB-C SD reader and inserts it into port 2 of the GMKtec
mini PC; left YAM holds the PC. Recipe and question: [README](../../README.md),
[research direction](../research/RESEARCH_DIRECTION.md). Dated detail: [FINDINGS](FINDINGS.md).

## Plan

- **Long term:** a vision + proprioception bimanual policy, trained only in simulation,
  that inserts on the real rig zero-shot and recovers from missed contact; then the same
  recipe on new tasks.
- **Medium term:** a privileged-state PPO teacher that solves pick → brace → insert across
  the randomized layout in sim; then a student with real-available observations (encoders,
  wrist/ZED RGB) and sim cameras at least as bad as real.
- **Short term (this week):** finish the sim-side randomization, define reward and resets,
  and show with a short PPO run that the task is solvable in the bimanual runtime.

## Done: simulator matched to the rig

| Item | Result | Where |
|---|---|---|
| Arm dynamics | Driver gravity feedforward + per-joint gravity scale + Coulomb friction identified per arm; held-out tool error left 8.5 → 4.4 mm; GPU replay 4.9–15.6 → 1.6–4.4 mm | [real2sim](../real2sim/README.md#arm-dynamics) |
| Friction creep | Stiff frictionloss impedance (no sub-threshold creep) | same |
| Runtime speed | CUDA-graph substeps: 334 → 28,486 env-steps/s at 2048 worlds | [rl](../rl/README.md) |
| Fingertips / wrist cameras | Gripper sweep fit held to calipers (105.25 mm lens-to-tip, both arms); held-out tip IoU left 0.70 → 0.89, right 0.59 → 0.81; applied to the camera pose | [real2sim](../real2sim/README.md#cameras-and-fingertips) |
| Background | Wall, raceway, plugs, power strip, cables fitted to 21 real frames (edges 0.6–1.9 px at 320 px) | [real2sim](../real2sim/README.md#background) |
| Backlash | Measured ≤ ~2 arcmin on both arms (below camera noise): not modelled | `data/backlash/README.md` |
| PC friction | Unmeasurable on the rig: μ U(0.3, 0.9) per world | `BimanualCfg.pc_friction_range` |
| Task layout | Reachability search: nominal port (0.22, 0.36) m, yaw 0, ±3 cm / ±10° per episode; reader anywhere pickable and clear of the PC (0.20 m², yaw ±90°) | [rl](../rl/README.md#task-layout) |
| Grasp creep | `impratio=100`: stationary 20 s hold 30.3° → 0.3° | [handoff](../rl/BIMANUAL_TRAINING_HANDOFF.md) |
| Nominal insertion | CPU scripted insertion with privileged feedback seats (5.78/5.8 mm, 0.016 mm lateral); GPU seated hold matches CPU | same |

## Open (sim side), in order

1. **Action latency** (measured ~8 ms ≈ 0.4 control step → randomize 0–1 step) and
   **encoder-offset noise** (±1.5° joints 2–5) in the bimanual runtime (exist in the single-arm `EnvCfg`).
2. **Contact randomization:** plug–port friction, pad friction and stiffness (rigid contact model).
3. **Reward and reset** for the privileged teacher; GPU end-to-end pick → carry → insert with
   feedback; left-arm bracing has not been demonstrated.
4. Short PPO run: the task is solvable in sim at the randomized layout.
5. Known model limits, kept explicit: stick-slip (static > kinetic friction, 6–12 mrad slips)
   not modelled; gripper squeeze force and pad compliance unmeasured; sim arm links render
   lighter than the real black arms.

## Open (rig side): `real/duchamp3`

- Move the PC so port 2 is at (0.22, 0.36) m in the right-arm base frame, square to the table.
- New grasp for the reader (to be chosen); optional 10–20 grasp captures for the in-hand pose spread.
- Remaining checklist: [PHYSICAL_TODO](PHYSICAL_TODO_20261030.md).

## Gates (proposed, not yet agreed)

| Gate | Pass when |
|---|---|
| G1 sim solvable | privileged teacher ≥ 80 % seated insertion over the randomized layout (n ≥ 512) |
| G2 robust teacher | ≥ 80 % with latency, encoder offsets and contact DR on; no non-finite worlds |
| G3 student | real-available observations, ≥ 70 % in sim with camera noise ≥ real |
| G4 real | first real seated insertion; then a rate over ≥ 20 trials, failures logged |
