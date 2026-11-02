# Realistic reader-grasp model (`grasp_fidelity`)

Status: first version, under validation. Numbers: end of this file. Off by default: the baseline sim, Part 2 and Part 4
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

Validation is running. Preliminary (stiff-tip test build), n = 256, oracle servo, 24 mm, press 16 mm: 33 % held with the
real roll draws; roll 0-1.5 deg nearly all held, 3-4 deg about half, beyond 4 deg almost none. Not yet reproduced: the
hand's rise during the close (real 4-7.8 mm when empty).
