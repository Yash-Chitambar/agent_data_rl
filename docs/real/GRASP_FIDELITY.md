# Realistic reader-grasp model (`grasp_fidelity`)

Status: v2. Roll threshold, press rotation and hold rate match the real runs; the hand's rise does not (see Limits). Off by default: the baseline sim, Part 2 and Part 4
do not change unless the flag is set. Spec: [PICK_GRASP_SIM_HANDOFF_20261101.md](PICK_GRASP_SIM_HANDOFF_20261101.md).
Code: `blind_insertion/envs/grasp_fidelity.py`. Preset: `blind_insertion/configs/grasp_fidelity_v1.json`.

## Enable

Set `BimanualCfg.grasp_fidelity` to a preset name (`"v2"`), a JSON path or a dict.

Expert pick, reproducing the real press and roll (oracle servo; for real nets replace `--servo oracle` by the nets below):

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/policy/pick_expert_eval.py \
  --episodes 512 --nworld 256 --seed 1000 --believed_frame --enc_offset_deg 0.25 --finger_open 0.024 \
  --servo oracle --servo_fracs .5 .8 .92 --settle_max_s 3 --lift_v .05 \
  --grasp_fidelity v2 --tip_floor_mm -13 --press_mm 16 [--servo_roll_deg=-4.5] [--relief v3] \
  --out results/usbc/policy/pick/grasp_fidelity/x
# real nets: --servo runs/perception/reader_servo_hover.pt --servo_near runs/perception/reader_servo_near.pt \
#            --servo_near_rot runs/perception/reader_servo_near6.pt --rehover_mm 8
```

`--servo_roll_deg` fixes the roll bias on all worlds; without it each world draws one real close's target pose (n = 11).
Sweep: `scripts/policy/pick_feedback_sweep.py --tilts 55 --open_mm 24 26 --extra --grasp_fidelity v2 ...`.

Deploy dry run (same model through `deploy/bimanual/io_sim.py`; or `"sim": {"grasp_fidelity": "v2"}` in the config):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 \
  --config deploy/config/bimanual_pick2.json --grasp-fidelity v2 --name gf_check
```

Worktrees need `third_party` linked (`ln -s /home/yashchitambar/peg_test_bimanual/third_party third_party`).

## What `v2` changes

Failure source: `results/real/pick_20261101_analysis.md`. The servo target rolls the closing axis (-3.4..-5.6 deg in most
real closes), one finger stands on the table and the other finger's pads ride high.

| Field | Value | Basis |
|---|---|---|
| `servo_err` | one of 11 real [roll, tilt - 55] deg, + 0.3 deg noise, on every servo rotation | Real target poses. Perception error, injected, not caused by the sim nets |
| `reader_fillet_m` | 1-2 mm, four long edges | Radius not measured (assumed) |
| `reader_body_length_m` | 29.6 mm | Caliper. Vision nets trained on 41.3 mm: only a flag |
| `wrist_kp_scale` | 0.75-1.0 on joints 4-6 | 0.86 deg/N measured vs 0.74 from kp |
| `pad_friction` | 0.8-1.6 | Unmeasured |
| `pad_timeconst` | 2-6 ms | Unmeasured |

The table contact stays at the baseline: true sim tips do not sink (the -2.5 mm reading was encoder-offset DR through
believed FK). `tip_solref`, `tip_extend_m` and `tip_friction` exist but `v2` leaves them off; the fitted tip extension of
the first version had no support in the logs and is gone.

Eval flags that reproduce the real chain, not physics: `--tip_floor_mm -13` (chain floor), `--press_mm 16` (target
lowered at the end of the descent: on the robot the last servo correction causes this press), `--relief v3`
(`chain._close_press_relief`, including a position-mode ramped close through `BimanualRuntime.grip_target`).

## New per-episode columns

`held`, `final_finger_mm`, `press_*` (1 s before the close), `close_*` (incl. `close_roll_true_deg`, `close_tilt_deg`),
`high_pad_mm` (high pad above the low tip, fingers at 19 mm), `rise_mm`, `jump_mm` (largest 60 ms rise), `gf_*` (the draws).

## Results

`results/usbc/policy/pick/grasp_fidelity/`, summary `acceptance.json`. `held` = lifted > 60 mm. CI: Wilson 95 %.

Roll sweep, oracle servo, 24 mm, press 16 mm, n = 256 per roll (true roll = commanded - 0.4 deg):

| True roll at the close (deg) | -0.40 | -1.91 | -3.39 | -4.19 | -4.90 | -6.01 |
|---|---|---|---|---|---|---|
| Held of 256 | 249 | 247 | 233 | 127 | 15 | 0 |

Real: held up to 2.87 deg, lost at 3.2 deg, empty from 3.78 deg. High pad above the low tip: 7.4 mm at -3.39, 7.9-8.1 mm at
-4.19 (real held <= 7.15 mm, empty >= 7.76 mm).

Bootstrap of the real rolls, 24 mm unless stated, n = 512, seed 1000, Wilson 95 % CI (real: 2 of 8 closes):

| Run | Held | Success (strict) |
|---|---|---|
| Baseline physics, oracle, no press | 512 (99.3-100 %) | 512 |
| v2, oracle | 169 (29-37 %) | 164 |
| v2, oracle, 26 mm | 167 (29-37 %) | 165 |
| v2, oracle, relief v3 | 260 (47-55 %) | 257 |
| v2, real nets | 99 (16-23 %) | 92 |
| v2, real nets, relief v3 | 108 (18-25 %) | 100 |
| v2 without `servo_err` or press, real nets | 431 (81-87 %) | 430 |

Use the oracle rows as the reference. Nets + `servo_err` compound: true roll p5 reaches -14 deg, far beyond the real closes.
True roll with the sim nets alone: median 0.0, p5-p95 -1.9..+1.6 deg, n = 512; real target at the last waypoint -3.28 +/- 1.92 deg.
The 84 % in the last row is new: baseline physics gave 511/512 with the same nets. Causes not separated (body length and
look vs fillets and pad ranges). Its empty closes have high pad 9.2 mm.

Deploy dry run with v2, pick stage, seeds 7100-7107: relief config 3/8 completed, no-relief config 3/8 (others stop on the reader-present guard).

In-hand spread about the mean of successes, oracle: position p50 2.81 mm vs baseline 0.89, rotation p50 3.02 deg vs 0.19. Flags off, the compiled
model matches the pre-change commit in all 483 physics arrays; `bimanual_conventions` passes, and baseline oracle holds 512/512.

Open-loop replay of the real commands (`scripts/real2sim/replay_real_run.py --objects none`, baseline table, n = 2 runs, 1 s
before the close): rotation 3.58-3.55 deg about y, real 3.65-3.60; roll -4.22/-4.63 deg, real -4.20/-4.53; FK tip 3.56/3.88 mm, real 2.44/2.57.

Logged servo commands (`scripts/real2sim/servo_roll_bias.py`): target roll at the last waypoint: real -3.28 +/- 1.92 deg (n = 10),
sim dry runs +1.74 +/- 0.25 deg (n = 14, only two seeds). Over the three waypoints the real target moves +0.63, -1.14, -3.28 deg for one reader on one
table, sim +0.04, +0.57, +1.74. This supports a net error on real near frames. It does not prove it: no ground truth exists for real roll.

## Limits

- **26 mm is not worse than 24 mm in the sim** (167 vs 169). The real evidence is only pick4.
- **The sim hand does not rise.** Median rise close to stall ~0 mm when empty (real 4-7.8), held ~0 (real 1.2-2.8), and no 3-6 mm jump.
  In sim empty closes the low finger pushes the reader aside (median 13 mm); on the robot the fingers shut over the body. Bigger fillets
  (3-3.5 mm) and contact spheres at the tip end did not create a rise, and the tip spheres raised the hold rate at high roll.
- So **v2 cannot show a failure caused by the rise**. A strategy that scores well through press or hold-down on v2 may
  look better than it is: an unpressed close and a light hold-down are not tested for the rise the operator saw.
- Sim stall 19.0 mm is unchanged (reading 0.32; real 0.30-0.305).
- Roll is injected: the sim nets do not produce it. Nets + `servo_err` is pessimistic (true roll median -7.7 deg vs real about -4.2).
- Oracle true roll runs ~1 deg deeper than the measured real hand (median -5.1 vs -4.2): rows are real targets, and the real hand lags them.
- `reader_fillet_m`, `pad_friction` and `pad_timeconst` are assumed ranges, and the outcome/roll ordering on the robot is post-hoc on 10 closes.
