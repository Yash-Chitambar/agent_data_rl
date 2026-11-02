# Realistic reader-grasp model (`grasp_fidelity`)

Status: first version, under validation. Numbers: end of this file. Off by default: the baseline sim, Part 2 and Part 4
do not change unless the flag is set. Spec: [PICK_GRASP_SIM_HANDOFF_20261101.md](PICK_GRASP_SIM_HANDOFF_20261101.md).
Code: `blind_insertion/envs/grasp_fidelity.py`. Preset: `blind_insertion/configs/grasp_fidelity_v1.json`.

## Enable

Set `BimanualCfg.grasp_fidelity` to a preset name (`"v1"`), a JSON path or a dict.

Expert pick, reproducing the real press (oracle servo; for real nets replace `--servo oracle` by the nets below):

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/policy/pick_expert_eval.py \
  --episodes 512 --nworld 256 --seed 1000 --believed_frame --enc_offset_deg 0.25 --finger_open 0.024 \
  --servo oracle --servo_fracs .5 .8 .92 --settle_max_s 3 --lift_v .05 \
  --grasp_fidelity v1 --tip_floor_mm -13 --press_mm 16 [--relief v3] --out results/usbc/policy/pick/grasp_fidelity/x
# real nets: --servo runs/perception/reader_servo_hover.pt --servo_near runs/perception/reader_servo_near.pt \
#            --servo_near_rot runs/perception/reader_servo_near6.pt --rehover_mm 8
```

Sweep: `scripts/policy/pick_feedback_sweep.py --tilts 55 --open_mm 24 26 --extra --grasp_fidelity v1 ...`.

Deploy dry run (same model through `deploy/bimanual/io_sim.py`; or `"sim": {"grasp_fidelity": "v1"}` in the config):

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 \
  --config deploy/config/bimanual_pick2.json --grasp-fidelity v1 --name gf_check
```

Worktrees need `third_party` linked (`ln -s /home/yashchitambar/peg_test_bimanual/third_party third_party`).

## What `v1` changes

| Field | Value | Basis |
|---|---|---|
| `tip_solref`, `tip_solimp` | 2 ms, stiff; tips take contact priority | Baseline 20 ms lets the tips sink |
| `tip_extend_m` | 2.6-7 mm along the approach (1.5-4 mm vertical at 55 deg) | Fitted to the blocked tip height, FK +2..+3 mm. Physical source unknown |
| `tip_friction` | 0.6-1.2 | Unmeasured |
| `wrist_kp_scale` | 0.7-1.0 on joints 4-6 | Unmeasured |
| `reader_fillet_m` | 1-2 mm, four long edges | Radius not measured (assumed) |
| `reader_body_length_m` | 29.6 mm | Caliper |
| `pad_friction` | 0.8-1.6 | Unmeasured |
| `pad_timeconst` | 2-6 ms | Unmeasured |

Eval flags that reproduce the real chain, not physics: `--tip_floor_mm -13` (chain floor), `--press_mm 16` (target
lowered at the end of the descent: on the robot the last servo correction causes this press), `--relief v3`
(`chain._close_press_relief`, including a position-mode ramped close through `BimanualRuntime.grip_target`).

## New per-episode columns

`held`, `final_finger_mm`, `press_*` (1 s before the close), `close_*`, `rise_mm`, `gf_*` (the draws).
