# Sim fidelity for dry runs: release jaw opening and PC panel colour (2026-11-16)

Task: two sim-only fixes for dry runs and sim evals. (1) The sim jaw opening at the release. (2) The PC panel colour in the sim ZED render, so the case-edge fit works in the dry run. Worker on mason, GPU 1, no rig access. **Nothing here changes what the real chain sends.** No existing config changed. No weights changed. Sim numbers are sim only. Every number has its n.

## Summary

- New SimIO settings (config block `sim`, read only by `deploy/bimanual/io_sim.SimIO`; RealIO never reads them):

| Key | Default (SimIO) | Off | What it does |
|---|---|---|---|
| `sim.release_open_m` | **0.027** | `0` or `null` | From the first `6 release` tick, the sim right finger joint limit is 27 mm per finger (24 mm before). Restored at the next episode reset. |
| `sim.edgefit_panel_rgb` | **[0.35, 0.35, 0.35]** | `null` | One extra top render at the capture, with the PC panel material darkened. Only the case-edge fit (`zed_edge_refine`) uses it. The ZED nets, the wrist cameras and the logged frames keep the default look. |
| `sim.edgefit_camera` | **`nominal`** | `dr` | The extra frame comes from the calibrated top camera (no visual-DR camera pose / intrinsics draw for it). |

- Runner flag (dry run only): `--sim-release-open-m X` sets `sim.release_open_m` (`0` = the old 24 mm dry run). The edge-fit keys have no flag; use a dry-run-only config copy with a `sim` block.
- The defaults live in SimIO, so every dry run and every eval script built on `deploy.run_bimanual --dry-run` uses them without a config change: `k8_release_dry.py`, `align_dry_eval.py`, `parallel_dry_ab.py`, `placement_grid.py`, `deploy/checks/bimanual_dryrun.py`.
- Release (n = 24 dry runs): with 27 mm the retreat moves the plug 0.000 mm in 8 of 8 released runs (24 mm: 0.007-0.020 mm in 3 of 9). Seat before the release does not change (the option acts only from the release). But on seeds 7041 / 7042 the remaining "not seated after release" is **not** the retreat drag: it is a plug that ends the push at the edge of the sim seat band (5.55 mm), or a 0.09-0.12 mm slide back while the jaws open. Details in section 1.
- Real jaw opening: the real jaws do **not** open wider than the sim. The chain commands i2rt 0.437 (24 mm) at the release; the real reading is 22.7 mm per finger after 0.4 s and 23.5 mm at most (n = 24 real releases). So 27 mm is a sim contact workaround, not a match to the real width.
- Panel: with the dark fit frame the edge fit passes its quality gate in 51 of 52 sim frames with the nominal scene look (default panel: 4 of 52) and its error against sim truth is 0.58 mm p50, 1.72 mm max (n = 95 passing frames of 128). The dark panel makes the ZED nets worse (v1 p50 0.49 -> 2.30 mm, zed_v5 0.64 -> 1.75 mm, n = 128), so the dark look is used **only** for the fit frame.
- Edge-fit dry runs (`bimanual_align10_all_edgefit.json`, no `truth_model`, n = 9): the fit gate passes on 8 of 9; fit error 0.34-1.28 mm (median 0.52 mm). The one reject is the designed case: v1 is 15.1 mm off at PC (251, 404) mm, 9 deg, and the shift gate (8 mm) stops the run.

## 1. Release jaw opening

### Why the sim jaws open to 24 mm

- The sim right finger joint has an upper limit of 24 mm per finger (`PickTaskCfg.finger_open` 0.024, `blind_insertion/envs/bimanual_pick.py`). It was set for the pick descent: the wrist nets saw 24 mm fingers in training.
- The chain commands `gripper.right_approach_open_m` (24 mm, i2rt 0.437) at the release (`chain.grip_cmd`). Only `bimanual_seat2.json` sets `insert.release_open_m` (47.5 mm). The real chain is unchanged by this task.
- Real logs (`ticks.npz`, right `grip` and `finger_m` = the gripper map of the reading), every real run with a release after the insertion, 2026-11-02..13, n = 24: at the end of the insertion the jaws read 17.2-18.7 mm; 0.4 s after the release command 22.6-22.8 mm (n = 19 with the open command; 5 runs kept the jaws closed: `keep_reader_unless_seated`); maximum in the release 23.4-23.6 mm.
- The sim jaws reach 24.9 mm (soft limit) at the release start, 26.4-27.6 mm with the new key. So the real jaws open *less* than the sim. The K8 drag (an end pad sphere catches the reader body) is a sim pad-model effect. 27 mm moves the sim pads clear of it. It does not copy a real width.

### Implementation

- `deploy/bimanual/chain.py` `RealChain.hstep`: at a stage change it calls `self.io.on_stage(label)` if the I/O has that method, and logs the returned record as event `sim_stage`. RealIO has no `on_stage`: no change on the rig (checked: `deploy.checks.bimanual_realio_mock` ok, `deploy.checks.zed_edgefit_realio` `FAILED: none`).
- `deploy/bimanual/io_sim.py` `SimIO.on_stage`: at the first `6 release` tick, the right finger upper limits (both finger joints, every world) go to `max(old limit, release_open_m)`. With the grasp-fidelity model (position-controlled jaws) the open target also goes up to `release_open_m`. `SimIO.reset` restores the limits.
- `deploy/run_bimanual.py`: `--sim-release-open-m`.
- Same point in time as K8's `k8_release_dry.py --release-open-m` (the first `6 release` tick). The final-push unload jaw opening (label `5 insert`) is not changed.

### A/B dry runs

`scripts/eval/sim_release_ab.py run` (each run = `k8_release_dry.py` truth trace around `deploy.run_bimanual --dry-run --stage full --align-mode full`), configs `bimanual_align9_all`, `bimanual_align10_all`, seeds 7041, 7042, 3 runs per seed and opening, n = 24 runs. Table `results/real_align/sim_release/ab.md`, rows `results/real_align/sim_release/rows/`, traces `runs/simrel/dry/`. "Released after a seat" = the insertion ended with `seated_detected` and the chain ran the release. "Seated" = the sim seat test (all plug points within 0.25 mm of full depth, tilt <= 2 deg).

| Config | Opening | Runs | Released after a seat | Seated at end of insert | Seated at release start | Seated after release | Not seated after release | Retreat depth loss (mm, per released run) |
|---|---|---|---|---|---|---|---|---|
| align10_all | 24 mm (old) | 6 | 5 | 3 | 3 | 3 | 2 | 0, 0, 0, 0, 0 |
| align10_all | 27 mm | 6 | 5 | 5 | 5 | 4 | 1 | 0, 0, 0, 0, 0 |
| align9_all | 24 mm (old) | 6 | 4 | 3 | 1 | 1 | 3 | 0, 0.011, 0.020, 0.007 |
| align9_all | 27 mm | 6 | 3 | 2 | 0 | 0 | 3 | 0, 0, 0 |

Per seed (released / not seated after release): align10 7041: 24 mm 2 / 1, 27 mm 2 / 1; align10 7042: 24 mm 3 / 1, 27 mm 3 / 0; align9 7041: 24 mm 1 / 1, 27 mm 0 / 0 (3 stops before the release: guard bound, press abort); align9 7042: 24 mm 3 / 2, 27 mm 3 / 3.

Reading:
- The retreat drag is gone at 27 mm (0.000 mm in 8 of 8 released runs). This agrees with K8's batched probe (0.1 % retreat loss, n = 898).
- Seat before the release: the option cannot act before the `6 release` tick. Seated at the end of the insertion: 6 of 9 released runs (24 mm), 7 of 8 (27 mm). The difference is run-to-run variation (MuJoCo Warp is not bitwise deterministic).
- The other "not seated after release" causes on these seeds:
  - align9_all 7042 (all 6 runs): the plug reaches 5.78 mm in the push, then springs back to 5.56-5.63 mm in the last 15 ticks before the release (push end), and the first jaw-opening tick takes another 0.02-0.04 mm. The plug ends at 5.54-5.58 mm, on the edge of the seat band. The retreat does not move it.
  - align10_all 7041: the plug slides back 0.09 mm (24 mm) and 0.12 mm (27 mm) in the 0.4 s jaw opening. This is the K8 "opening" effect (3 % of nominal plugs, n = 925), not the retreat.
  - align10_all 7042, 24 mm, one run: the plug ended the insertion at 5.57 mm (sim seat test false before the release).
- So for "not seated after release" on 7041 / 7042, read the trace columns (`depth_end_insert_mm`, `depth_after_open_mm`, `depth_after_retreat_mm`) before you blame the release.

## 2. PC panel colour in the sim ZED render

### What is wrong

- The sim panel face is the material `pc_rear_photo` (a flash photo of the rear panel, mean RGB 117-128 of 255). In the ZED view it renders near white (`results/placement/sim_panel/sim_panel_default_vs_dark.png`, left). The real panel is dark grey under the light lid: real frame 2026-11-13 `all9_full_1`, panel median 89-93 with the ports, 75-80 between them; lid 190 (`results/placement/sim_panel/real_zed_pc_crop_20261113.png`).
- The fit looks for a bright-inside / dark-outside step at each top edge. With a light panel the panel edge has no step, and the fit finds the bottom of the panel face.
- A second cause, found here: the visual DR draws a top-camera pose error of up to 8.6 mm / 1.2 deg and 1.5 % focal jitter per episode. The fit uses the fixed calibration, so this moves the fit by about 10 mm in the sim. The real fit agrees with the arm-derived port to 0.8 mm p50 (n = 41, `ZED_EDGEFIT_20261115.md`), so the real calibration error is far smaller than the sim DR.

### Fix

- `SimIO.edgefit_top()`: one extra render of the same state with the panel material at `sim.edgefit_panel_rgb` (it multiplies the photo texture; no geometry change) and the top camera at its calibrated pose (`sim.edgefit_camera: nominal`). The photometric draw of the policy frame is kept (no new exposure / gain draw).
- `RealChain._edge_refine` (sim branch only) uses this frame when SimIO gives one; the log field `edge_fit.frame_source` is then `sim_policy_upsampled_dark_panel`. The real branch is unchanged.
- The ZED nets keep the default frame because the dark panel makes them worse (next table).

### Sweep (sim frames, not the full chain)

`scripts/perception/sim_panel_edgefit.py`: n = 128 ZED frames (ApproachRenderer, pick layout PC range, arms at the capture pose, grippers open, visual DR on, seed 12). The same frames rendered with each panel RGB. The fit starts from zed_v5 on the default frame (as in the dry run) with the `bimanual_align10_all_edgefit.json` block, geometry `sim`. "Nominal scene" = frames where the DR did not give the PC case or the table a random colour or texture (n = 52). Results `results/placement/sim_panel/final_nomcam.json`, `final_drcam.json`, earlier sweeps `sweep_cap*.json`.

| Panel RGB | Top camera | Fit quality gate pass (all, n = 128) | Pass, nominal scene (n = 52) | Fit \|err\| of passing frames p50 / max (mm) | v1 \|err\| p50 / p90 (mm) | zed_v5 \|err\| p50 / p90 (mm) |
|---|---|---|---|---|---|---|
| 1.0 (default) | nominal | 4 | 4 | 0.75 / 1.46 | 0.49 / 1.31 | 0.64 / 1.86 |
| **0.35** | **nominal** | **95** | **51** | **0.58 / 1.72** | 2.30 / 3.25 | 1.75 / 2.70 |
| 1.0 (default) | DR | 4 | 2 | 7.37 / 8.34 | 0.81 / 1.65 | 0.69 / 2.04 |
| 0.35 | DR | 88 | 47 | 7.35 / 25.7 | 2.29 / 3.69 | 1.86 / 3.07 |

n = 64 sweep (nominal camera): panel 0.5 / 0.35 / 0.2 pass 40 / 51 / 53; v1 p50 1.35 / 2.31 / 3.12 mm. 0.35 is the darkest step that keeps the panel face visibly grey (real ratio panel / lid about 0.4-0.5).

- Most failing frames with the dark panel have a DR random light table or a random lid colour: the lid then has no contrast to the table (real table: black). This does not happen on the rig.
- Net check (v1 gate `zed_v1_gate_arm`, zed_v5 `zed_v5_gate`): worse on the dark panel (above). Gate pass v1 128 -> 122, zed_v5 128 -> 124 of 128. **Decision: the dark panel is a fit-only frame; the nets keep the default look.**

### Edge-fit dry runs without `truth_model`

`bimanual_align10_all_edgefit.json` (mode `use`), SimIO defaults (dark fit frame, nominal camera, release 27 mm). Seeds 7041-7045; forced PC poses with `placement_grid.py one --reader 0.131 0.448 92 --seed 7043`. n = 1 run per row. Summary `scripts/perception/sim_panel_dry_summary.py` -> `results/placement/sim_panel/dry_ef.md`; run folders `results/placement/sim_panel/dry/ef/`.

| Run | Fit gate | Fit err x, y (mm) | \|err\| (mm) | Yaw err (deg) | v1 err (mm) | zed_v5 err (mm) | Shift fit - v1 (mm) | Result |
|---|---|---|---|---|---|---|---|---|
| seed 7041 | ok | +0.86, -0.05 | 0.86 | -0.28 | 4.69 | 4.00 | 5.2 | insert guard bound |
| seed 7042 | ok | -0.33, -0.17 | 0.37 | +0.85 | 0.91 | 1.70 | 0.6 | seated, in after release |
| seed 7043 | ok | -1.20, -0.44 | 1.28 | +0.85 | 3.77 | 2.39 | 4.9 | seated, in after release |
| seed 7044 | ok | -0.48, -0.83 | 0.96 | +0.56 | 3.86 | 2.78 | 3.2 | align stop (step limit after the press) |
| seed 7045 | ok | -0.38, 0.00 | 0.38 | -0.20 | 2.50 | 3.00 | 2.4 | align stop (step limit after the press) |
| PC (231, 394) mm, 4 deg | ok | -0.56, -0.50 | 0.75 | +0.23 | 2.43 | 2.25 | 3.0 | seated, not in after release |
| PC (251, 394) mm, 4 deg | ok | -0.28, -0.19 | 0.34 | +0.55 | 2.17 | 2.25 | 2.0 | v1 gate rejects (today's gate) |
| PC (251, 404) mm, 9 deg | **reject**: \|fit - v1\| 14.8 > 8 mm | -0.31, -0.28 | 0.42 | +0.16 | 15.13 | 2.88 | 14.8 | stop at the gate (designed) |
| PC (211, 384) mm, -3 deg | ok | -0.37, +0.36 | 0.52 | -0.44 | 4.02 | 2.52 | 3.9 | seated, in after release |

- Fit error against sim truth, n = 9: 0.34-1.28 mm (median 0.52 mm); yaw -0.44..+0.85 deg. ZED_EDGEFIT reported 12.7-15.9 mm for the render fit with the old panel (n = 11).
- The stops after the capture (guard bound, align step limit) are dry-run variation of the later stages, not the fit. The chain uses the v1 yaw (`yaw_source: gate`).

## What changed

| File | Change |
|---|---|
| `deploy/bimanual/io_sim.py` | `SIM_RELEASE_OPEN_M`, `SIM_EDGEFIT_PANEL_RGB`, `SIM_EDGEFIT_CAMERA`; `panel_rgb()`, `nominal_camera()` context managers; `SimIO.on_stage`, `SimIO.edgefit_top`; limit restore in `reset`; open target in `send` for the position-controlled jaws after the widening. |
| `deploy/bimanual/chain.py` | `hstep`: optional `io.on_stage` hook (absent on RealIO). `_edge_refine` sim branch: uses `io.edgefit_top()` when given. |
| `deploy/run_bimanual.py` | `--sim-release-open-m` (dry run only). |
| `scripts/eval/sim_release_ab.py` (new) | Release A/B dry runs and table. |
| `scripts/perception/sim_panel_edgefit.py` (new) | Panel RGB sweep: edge fit and ZED nets on paired sim frames. |
| `scripts/perception/sim_panel_dry_summary.py` (new) | Edge-fit dry-run table. |

No new deploy config (nothing changes on the rig). Conventions check `FAILED: none` on `bimanual_align10_all.json` and `bimanual_align10_all_edgefit.json` (after the chain change). Mock rig path: `bimanual_realio_mock` ok, `zed_edgefit_realio` `FAILED: none`.

## Limits

- Sim only. The 27 mm opening hides a pad-model artefact; it is not a measured real width (the real jaws open to about 23.5 mm).
- Dry runs are n = 1-3 per seed and condition; they vary between reruns. Use K8's batched probe for rates.
- A dry run that checks a change to the release itself should use `--sim-release-open-m 0` and read the retreat column of the trace.
- The dark panel RGB (0.35) is fitted by eye and by the fit gate, not to a photometric match. The fit frame uses the calibrated top camera: the sim then does not test the fit against a camera calibration error. The real cross-check (fit vs arm port, n = 41) bounds that error at about 1 mm.
- The fit frame needs one more render per capture (about 0.1 s in the dry run).

## Rig runbook

Nothing to run on the rig. No config, policy, gain or limit changes; no approval needed. The rig chain sends the same commands as before (the hook and the fit frame exist only in SimIO).

Server (dry runs):

```bash
# defaults now: release 27 mm, dark-panel calibrated-camera fit frame
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=. python -m deploy.run_bimanual --dry-run --stage full --align-mode full \
    --seed 7041 --config deploy/config/bimanual_align10_all_edgefit.json --name x
# the old release (to check a change to the release itself)
... --sim-release-open-m 0
# old fit frame: a dry-run-only config copy with  "sim": {"edgefit_panel_rgb": null}  (or "edgefit_camera": "dr")
# release A/B and table
python scripts/eval/sim_release_ab.py run --configs bimanual_align10_all --seeds 7041 7042 --opens 0 0.027 --reps 3 --par 6
python scripts/eval/sim_release_ab.py table
# panel sweep and dry-run table
python scripts/perception/sim_panel_edgefit.py --n 128 --seed 12 --rgbs 1 0.35 --top-cam nominal --out results/placement/sim_panel/x.json
python scripts/perception/sim_panel_dry_summary.py results/placement/sim_panel/dry/ef --md results/placement/sim_panel/dry_ef.md
```

Pass (dry run): the event `sim_stage` (key `sim.release_open_m`) appears at the release; the retreat depth loss is 0; `edge_fit.frame_source` = `sim_policy_upsampled_dark_panel`, fit gate ok, \|fit - truth\| <= 2 mm. Stop and look: a fit error > 2 mm with the gate ok, or a depth loss in the retreat with 27 mm.

Parallel runs: give each dry run its own `--name` (runs started in the same second share a folder and corrupt `ticks.npz`).
