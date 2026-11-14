# ZED case-edge fit in the capture stage (2026-11-15)

Task: put the PC case-edge fit (`scripts/perception/fit_pc_edges.py`, `docs/real/ZED_MAP_20261114.md`) into the capture stage of the deploy chain as an optional PC pose estimator, OFF by default. Worker on mason, GPU 3, no rig access. **Nothing here has run on the arm.** No weights changed. No existing config changed. Sim numbers are sim only. Every number has its n.

## Summary

- New config block `zed_edge_refine` (absent = off; every existing config behaves as before). Code: `deploy/bimanual/zed_edgefit.py`, called from `RealChain.capture_pc` (`deploy/bimanual/chain.py`).
- Gate net = today's `chain_args.zed_gate` (v1, `zed_v1_gate_arm`): its pose, yaw and gate run as today, and its gate stays required. Start pose of the fit = **zed_v5** (`start_gate: zed_v5_gate`). The fit gives x, y. **Yaw = the v1 yaw (unchanged from today).**
- The fit is used only when its own gate passes. A failed fit gate fails the capture like a net reject: the run stops at the gate (the operator can retake). No silent fall-back to the net.
- Offline replay on every real capture frame with a raw HD1080 frame (n = 89, 2026-11-01..13, incl. the 6 mapping captures): the fit against the arm-derived port 2 (n = 41 runs) is **x -0.02 mm, y +0.69 mm (median), |dxy| max 1.12 mm**. v1 on the same runs: y -3.08 mm (to -7.64), zed_v5: y +4.50 mm (to +7.22). Fit gate (quality) passes 89 / 89. With the 8 mm shift bound the combined gate passes 84 / 89 (rejects: 3 frames at the 2026-11-13 spot with shift 8.0-8.1 mm, map_y_plus with 16.3 mm, map_yaw by the v1 gate as today).
- Two new configs: `deploy/config/bimanual_align10_all_edgefit_capture.json` (mode `shadow`: fit and log only, the chain is exactly `bimanual_align10_all.json`) and `deploy/config/bimanual_align10_all_edgefit.json` (mode `use`). Conventions check: `FAILED: none` on both.
- The sim cannot test the image fit faithfully: the sim PC panel face renders light grey (real: dark grey), so the panel edge is not found, and the dry run stops at the fit gate (as designed). A dry-run-only option (`dry_run_fit: truth_model`) exercises the downstream chain with a fit-quality pose. Dry runs: seeds 7041-7045 and three forced PC poses, with and without the option (n = 1 each). See "Sim dry runs".

## What changed

| File | Change |
|---|---|
| `deploy/bimanual/zed_edgefit.py` (new) | Block parsing and defaults (`enabled` false), the fit with the edge-set rule ('all', or 'noL' when the left edge has < 30 inliers), the fit gate, the sim canvas (320 px policy render back into its HD1080 crop), log record. |
| `deploy/bimanual/chain.py` | `RealChain.__init__`: parse the block, load the start net, log a `zed_edge_refine` event. `render()`: keeps the raw frames of the last render (`self.last_raw`). `capture_pc()`: calls `_edge_refine` when the block is on. Nothing changes when the block is absent. |
| `deploy/bimanual/netcache.py` | `preload()` also loads the start net when the block is on (nets load before the arms connect). |
| `deploy/checks/zed_edgefit_realio.py` (new) | Mock-hardware check of the rig code path (RealIO, raw frame). |
| `scripts/perception/zed_edgefit_replay.py`, `zed_edgefit_summary.py`, `zed_edgefit_dry_summary.py` (new) | Offline replay on real frames, its tables, the dry-run table. |
| `deploy/config/bimanual_align10_all_edgefit_capture.json`, `bimanual_align10_all_edgefit.json` (new) | The two configs. |

Not touched: `insert.py`, `align.py`, `port_vision.py`, `fit_pc_edges.py`, every existing config, all weights.

### Is the raw HD1080 frame available on the rig?

Yes. `deploy/cameras.ZedTop` grabs the rectified left HD1080 frame; `RealIO.frames()` returns it next to the policy image, and `RealChain.render()` already logged it as `frames/top/*_zed_pc.jpg` (JPEG) in every run since 2026-11-01. The chain now also passes the in-memory frame (uncompressed BGR) to the fit. The replay used the saved JPEGs; the rig fit sees the same frame without JPEG loss.

### The block (`zed_edge_refine`)

| Key | Value | Meaning |
|---|---|---|
| `enabled` | true | false or absent = off |
| `mode` | `shadow` / `use` | shadow: fit and log only; use: the chain takes the fit |
| `start_gate` | `zed_v5_gate` | start net (estimator key, `netcache.zed`) |
| `geom`, `geom_sim` | `real`, `sim` | fit geometry on the rig / in the dry run |
| `yaw_source` | `gate` | `gate` = v1 yaw (today), `start` = zed_v5, `fit` = the fit |
| `min_inliers` | 30 of `n_points` 40 | per edge, panel, right and back required; left used only when it has >= 30 |
| `max_rms_px` | 1.2 | inlier rms |
| `max_dxy_start_mm`, `max_dyaw_start_deg` | 10, 4 | fit against the start net |
| `max_shift_mm` | 8 | fit against the gate net = the shift of the approach start against today |
| `dry_run_fit` | `render` | dry run only; `truth_model` = sim truth + N(0, 0.5 mm / 0.5 deg) in place of the fit |

The `estimate` event (`est: zed_pc`) now carries: `port_xy`, `yaw_deg`, `ok` = the values the chain uses; `net` = the v1 estimate and gate (today's values); `start` = the zed_v5 estimate and its gate details; `edge_fit` = fit pose, edges used, per-edge inliers and rms, inlier rms, statistical sd, distances to the start and to the gate net, the reject reasons, the time; `est_source`, `edge_mode`. The runner prints one line per capture: `edge fit (noL): port (...) mm, ... -> ok` or `-> REJECTED (reasons)`.

## Decisions

### Start net: zed_v5

On the 89 replayed frames the fit converges to the same pose from the v1 start and from the zed_v5 start (max difference 0.03 mm). So the start does not change the result where both are within reach. I choose zed_v5 because:
- zed_v5 follows PC moves (pose-change error <= 1.2 mm, n = 5); its offset to the fit is near constant: |fit - zed_v5| max 6.6 mm (n = 89). v1 saturates near y 395 mm: |fit - v1| reaches 16.3 mm (map_y_plus). The fit converges from about +/-20 mm, so v1 runs out of margin first.
- A near-constant start offset makes `|fit - start| <= 10 mm` a useful second check against a fit locked on a wrong edge.

The zed_v5 gate is logged, not required: it rejects many good frames by ensemble disagreement (pass 79 / 89 here; 2 / 8 on 2026-11-11). The v1 gate (presence, OOD, arm envelope) stays required, exactly as today. So no gate becomes looser.

Cost: the three zed_v5 weight files are not in `deploy/bundle`; copy them to the rig (as for `bimanual_align8_zedv5.json`).

### Yaw: keep the v1 yaw

Replay data (n = 89 frames; arm n = 41 runs):

| | median | range |
|---|---|---|
| fit - v1 | -1.29 deg | -2.17..+0.20 |
| fit - zed_v5 | -1.81 deg | -2.58..-0.33 |
| v1 - arm (port vision) | -0.56 deg | -1.06..+0.04 |
| zed_v5 - arm | -0.09 deg | -0.79..+0.78 |
| fit - arm | -1.93 deg | -2.50..-1.34 |

- The fit reads about 1.3-1.9 deg below both nets and the arm. The arm yaw is not independent (the port-vision panel model frame came from a v1 yaw on 2026-11-02), so the absolute yaw is still open.
- Every seated run so far used the v1 yaw, and the align stage corrects the plug angle against port vision (the arm frame). A 1.5 deg change of the approach axis moves the plug 0.8 mm sideways at the 30 mm look gap.
- So the chain keeps the **v1 yaw** (`yaw_source: gate`): the only change against today is x, y. One change at a time.
- Yaw changes: on the yaw mapping pose (+7.4 deg by the fit) v1 read +5.6 deg, zed_v5 +7.2 deg. When the rig check settles the absolute yaw (square and tape, `NEXT_RIG_20261115.md` step), `yaw_source: fit` (with an offset, if needed) is the next candidate.

### The fit gate thresholds (from the real frames, n = 89)

- Inliers: panel min 32 / 40 (all frames, 2026-11-14 doc: n = 107), right and back 40 / 40. The left edge has < 30 inliers on 77 / 107 frames (the left gripper hides it): then 'noL'. 'noL' was used on 77 / 89 replay frames.
- Inlier rms: max 0.86 px in the replay (n = 89), 1.14 px over the 107 frames of the mapping doc. Gate 1.2 px.
- |fit - zed_v5|: max 6.6 mm. Gate 10 mm. |yaw fit - zed_v5|: max 2.6 deg. Gate 4 deg.
- |fit - v1| (shift against today): see Safety.
- Time: 0.29 s median, 1.03 s max (two fits when 'noL'; CPU, mason).

## Offline replay (real frames)

`CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/zed_edgefit_replay.py --out results/placement/zed_edgefit/replay.json`, then `zed_edgefit_summary.py`. Every real run with a raw capture frame (2026-11-01..13: n = 89, the 18 zed_gt snapshot frames have no logged arm angles and are not run). Nets on the logged policy image with the logged arm angles; the fit through `zed_edgefit.refine` (the chain's function) on the raw JPEG, block of `bimanual_align10_all_edgefit.json`.

| Group | n | v1 gate ok | zed_v5 gate ok | fit gate ok (quality + start) | combined (v1 gate + fit gate, shift 8 mm) | edges all / noL | rms max (px) | shift \|fit - v1\| p50 / max (mm) | \|fit - v5\| max (mm) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-11-01 | 21 | 21 | 20 | 21 | 21 | 3 / 18 | 0.67 | 1.5 / 5.9 | 5.2 |
| 2026-11-02 | 13 | 13 | 13 | 13 | 13 | 0 / 13 | 0.24 | 1.3 / 1.6 | 4.5 |
| 2026-11-03 | 9 | 9 | 9 | 9 | 9 | 0 / 9 | 0.33 | 3.7 / 4.0 | 4.9 |
| 2026-11-06 | 10 | 10 | 10 | 10 | 10 | 0 / 10 | 0.20 | 3.5 / 4.0 | 4.2 |
| 2026-11-10 | 16 | 16 | 15 | 16 | 16 | 0 / 16 | 0.27 | 4.3 / 6.0 | 5.0 |
| 2026-11-11 | 8 | 8 | 2 | 8 | 8 | 0 / 8 | 0.26 | 3.9 / 4.6 | 5.8 |
| 2026-11-13 full runs | 6 | 6 | 5 | 6 | 4 | 6 / 0 | 0.86 | 6.8 / 8.0 | 6.6 |
| 2026-11-13 mapping captures | 6 | 5 | 5 | 6 | 3 | 3 / 3 | 0.70 | 4.8 / 16.3 | 6.3 |
| **All** | **89** | **88** | **79** | **89** | **84** | 12 / 77 | 0.86 | | 6.6 |

Combined pass with other shift bounds: 10 mm 87 / 89, 12 mm 87 / 89 (left: map_y_plus 16.3 mm; map_yaw by the v1 gate).

Against the arm-derived port 2 (`zed_arm_port.py`, n = 41 runs 2026-11-03..13; median [min, max]):

| Estimate | dx (mm) | dy (mm) | \|dxy\| p50 / max (mm) |
|---|---|---|---|
| **fit (the x, y the chain uses)** | -0.02 [-0.94, +0.52] | **+0.69** [+0.23, +0.96] | **0.80 / 1.12** |
| fit from the v1 start | -0.02 [-0.94, +0.52] | +0.69 [+0.23, +0.97] | 0.80 / 1.11 |
| v1 (today) | +0.74 [-0.72, +2.14] | -3.08 [-7.64, -2.03] | 3.28 / 7.69 |
| zed_v5 | -1.33 [-2.38, -0.51] | +4.50 [+3.57, +7.22] | 4.69 / 7.54 |

The mapping captures have no arm truth; on them the fit equals the mapping doc's truth (same method), and the replay checks the gate and the convergence from zed_v5.

Mock rig path (`deploy/checks/zed_edgefit_realio.py`, real frame `2026-11-10/full_152717_k4_1`): RealIO with fake arms and cameras, RealChain.capture_pc. Shadow: port = v1, fit logged ok. Use: port = fit (230.3, 398.2) mm, yaw = v1, `port_frame` from the fit, ok = v1 gate and fit gate. A uniform frame (no PC): fit gate rejects, capture fails. `FAILED: none`. The existing `deploy.checks.bimanual_realio_mock` still passes.

## Sim dry runs

Commands (GPU 3, `PYTHONPATH=. MUJOCO_GL=egl`): `python -m deploy.run_bimanual --dry-run --stage full --align-mode full --seed <s> --config <cfg>`, and for forced poses `python scripts/eval/placement_grid.py one --pc <x y yaw> --reader 0.131 0.448 92 --seed 7043 --config <cfg>`. Run folders and `summary.json` (`zed_edgefit_dry_summary.py`): `results/placement/zed_edgefit/dry/`. base = `bimanual_align10_all.json`; ef = `bimanual_align10_all_edgefit.json` (render fit); tm = the dry-run-only copy with `dry_run_fit: truth_model` (`results/placement/zed_edgefit/dry/configs/edgefit_truth_model.json`). n = 1 run per row. Sim only.

### A. The image fit on the sim render (ef, mode use)

| Run | Camera | v1 err (mm) | Render fit err x, y (mm) | Fit gate | Result |
|---|---|---|---|---|---|
| seed 7043 | default visual DR | 3.8 | +1.1, +15.5 | rejects (panel 20/40, back 0/40, \|fit - v1\| 12.4 mm) | stop at the gate |
| seed 7043 | camera DR off (`--sim-phys visual_dr` cam_err_scale 0, no focal/principal/k1 jitter) | 3.3 | -19.9, -1.9 | rejects (back 0/40, distances) | stop at the gate |
| seed 7043, capture | camera DR off + `pc_real_look` | 4.1 | -16.6, -1.3 | rejects | (capture only) |
| PC (251, 394) mm, 14 deg | camera DR off | 2.2 | -1.0, +0.2 | rejects (panel 7/40) | stop at the gate (v1 gate also rejects) |
| tm runs (render fit logged) | default visual DR | 0.9-4.7 | -4.4..+5.9, **+12.7..+15.9** (n = 11) | - | - |

Reading: the sim PC front panel renders light grey (also with `pc_real_look`); on the rig it is dark grey. The strongest bright-to-dark step at the panel side is then the bottom of the front face, not the top edge, and the fit slides 13-16 mm (default DR) or about 20 mm. **The sim render is not good enough for this fit** (also 320 px upsampled 3.3x, and the camera-pose DR of up to 8.6 mm / 1.2 deg moves the image against the calibration). In every case the fit gate rejects and the run stops at the gate: no silent fall-back. With the camera DR off and the PC at 14 deg, the render fit was within 1 mm of the truth but still lacked panel inliers.

### B. Downstream chain with a fit-quality pose (tm) against today (base), default visual DR

| Seed / pose | Config | Used PC err (mm) | v1 err (mm) | Shift (mm) | First wrist net across, up (mm) | Align | Insert (true seated) |
|---|---|---|---|---|---|---|---|
| 7041 | base | 4.7 | 4.7 | - | -3.4, -9.3 | ok | seated (yes) |
| 7041 | tm | 0.8 | 4.7 | 4.2 | 0.0, -10.0 | ok | seated (yes) |
| 7042 | base | 0.9 | 0.9 | - | 2.1, -10.7 | ok | seated (yes) |
| 7042 | tm | 0.3 | 0.9 | 1.0 | 1.7, -10.5 | ok | seated (yes) |
| 7043 | base | 3.8 | 3.8 | - | -3.4, -6.6 | ok | detected (no) |
| 7043 | tm | 0.6 | 3.8 | 3.5 | -0.4, -7.7 | ok | seated (yes) |
| 7044 | base | 3.9 | 3.9 | - | 1.0, -0.5 | stop: step limit after the press | - |
| 7044 | tm | 0.3 | 3.9 | 4.2 | -2.8, -0.7 | stop: step limit after the press | - |
| 7045 | base | 2.5 | 2.5 | - | 7.0, -17.8 | ok | seated (yes) |
| 7045 | tm | 0.0 | 2.5 | 2.5 | 4.8, -17.4 | ok | seated (yes) |
| PC (231, 394) 4 deg, 7043, OOD x1.4 | base | 2.4 | 2.4 | - | -1.0, -5.5 | ok | seated (yes) |
| same | tm | 0.6 | 2.4 | 1.9 | -1.6, -5.2 | ok | detected (no) |
| PC (251, 394) 4 deg (+20 mm x), 7043, OOD x1.4 | base | 2.2 | 2.2 | - | 0.6, -3.4 | stop: step limit after the press | - |
| same | tm | 0.6 | 2.2 | 2.2 | -1.7, -3.7 | ok | detected (no) |
| PC (251, 404) 9 deg, 7043 | base | 15.1 | 15.1 | - | - | v1 gate rejects | - |
| same | tm | 0.6 | 15.1 | **15.3** | - | **fit gate rejects (shift > 8 mm)** | - |

- Seeds 7041-7045: base 3 / 5 true seated, tm 4 / 5 (n = 5 each; 7044 stops in both at the same align step). This is the same order; n is too small for a difference. The fit-quality pose does not break the downstream chain.
- The sim PC sits at (218, 362) mm, where v1 is accurate (err 0.9-4.7 mm). The real gain is at the real spot (v1 y error 3-8 mm), which the sim does not show.
- Shifted poses: today's v1 gate rejects the asked "+20 mm x, +10 deg" pose (Mahalanobis 28.1, presence 0.36) and also +20 mm x at the same yaw (Mahalanobis 23.0 > 21.3). So both configs stop there. To exercise the chain behind it, two poses ran with `--zed-ood-scale 1.4` (sim study only, same for both configs). The pose (251, 404) mm shows the shift gate: v1 is 15 mm off (sim saturation, as on the real rig), the fit-quality pose is 0.6 mm off, and the 8 mm shift bound stops the run. With the 12 mm proposal it would stop too.


## Safety

- The fit moves where the approach starts by **|fit - v1| <= 8 mm** (`max_shift_mm`), only in x, y (yaw unchanged). It moves the start toward the arm-measured port: today's starts were 2.0-7.7 mm from the arm port (n = 41), the fit's 0.2-1.1 mm.
- Why 8 mm is inside what the rig has done: the chain has started approaches 7.6 mm from the true port (2026-11-13, v1 error) and the wrist servo and the align stage absorbed it. A pose that passes both distance gates lies within 8 mm of v1 AND within 10 mm of zed_v5. v1 reads low in y (-2..-8 mm) and zed_v5 high (+3.6..+7.2 mm), so the allowed y region is about truth -6..+5 mm even if the fit is wrong: inside the 8 mm the chain has absorbed. In x both nets are within 2.4 mm of the arm, so the region is about +/-10 mm along the port axis in the worst case; the approach stops at an 18 mm gap, so that still leaves >= 8 mm before contact, and the wrist servo measures the gap at the look pose. A wrong fit must also pass the edge gates (three edges with >= 30 / 40 points on straight lines of the known size, rms <= 1.2 px).
- All other safety limits (SafeArm, FlexChecker, guards, align and insert gates) are unchanged. No limit is looser.
- At the 2026-11-13 spot the shift is 6.8-8.1 mm (n = 12 frames): 3 of those frames are just above 8 mm and stop at the gate. The 2026-11-15 plan moves the PC back to the 2026-11-10 spot (shift 4-6 mm).
- **Proposal (needs the operator's approval):** `max_shift_mm` 12. It passes the 2026-11-13 spot (max 8.1 mm) and keeps rejecting the map_y_plus case (16.3 mm). The worst-case argument above becomes y truth -8..+5 mm.
- GIL: the fit runs on the CPU in the rig process for 0.1-1.0 s while both arms hold at the capture pose. The fit is many short numpy / OpenCV / scipy calls; none of them is a long single C call like `torch.load` or `Camera.open` (`netcache.py`). Check the CAN watchdog on the first capture (below).

## Limits

- The image fit is validated on real frames only (n = 89). The sim render does not test it (panel colour, 3.3x upsampling, camera DR). The sim dry runs test the code path, the gate stop, and the downstream chain with a fit-quality pose.
- Absolute yaw open at 1-2 deg (unchanged; the chain keeps the v1 yaw).
- The 'real' geometry (103.5 x 103 mm) comes from the fit residuals; the calipers are still to do (`NEXT_RIG_20261115.md`). A different size moves the port estimate by about half the size error along that axis.
- The fit reads +0.7 mm in y above the arm (n = 41). No correction is applied (inside the +/-1.5 mm uncertainty).
- The fit needs three edges. If the left hand or a cable hides the panel, right or back edge, the gate stops the run.
- Today's v1 gate rejected the yaw mapping pose (10.3 deg). The fit does not change that.

## Rig runbook (the session after the next one)

Before: copy `runs/perception/zed_v5s0.pt`, `zed_v5s1.pt`, `zed_v5_gate.pt` from mason and check the sha256 (as in `ZED_V5_20261114.md`). Pull the branch with `deploy/bimanual/zed_edgefit.py`.

### Part 1: capture only, shadow (no approval needed: the chain is bimanual_align10_all.json)

At 3-4 PC poses, one capture each. Use a straightedge on the table and move the PC by measured amounts:
1. the 2026-11-10 spot (y about 398-399 mm),
2. +10 mm in y (away from the right arm),
3. +10 mm in x (away from the bases),
4. +5 deg yaw (optional; v1 may reject).

```bash
.venv/bin/python -m deploy.run_bimanual --stage capture --config deploy/config/bimanual_align10_all_edgefit_capture.json --name edgefit_cap_<pose>
```

Read the `edge fit (...)` line. Pass, per pose:
- the line says `-> ok` (edges, rms <= 1.2 px, |fit - zed_v5| <= 10 mm, |fit - v1| <= 8 mm; the last may fail on pose 2, note it);
- the fit time <= 1.5 s, and no motor or CAN watchdog error during the capture;
- the change of the fit between poses equals the measured move within 1.5 mm (and 1 deg for yaw).
Stop (no Part 2): `edge fit: FAILED` or REJECTED by edge counts or rms at the spot; any motor error at the capture; a fit change that misses the measured move by > 2 mm.

On mason after the session: `CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/zed_edgefit_replay.py --dates <date> --out results/placement/zed_edgefit/replay_<date>.json` and `zed_edgefit_summary.py`.

### Part 2: full runs, mode use (needs the operator's approval)

PC at the 2026-11-10 spot. Same commands as for `bimanual_align10_all.json`, with the new config:

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --align-mode full --config deploy/config/bimanual_align10_all_edgefit.json --name edgefit_full_<n>
```

- Before the approach: the printed estimate must be the fit (`ZED PC estimate` = the `edge fit` port) with `gate True`.
- Expect the first wrist-net reading at the look pose (`4_approach`, across) near the sim value (-3.1..-3.7 mm) instead of -0.6..+0.3 mm (2026-11-10..13): the ZED y error is gone.
- Pass: 3 runs reach the hand-over with align ok, no new stop type against `bimanual_align10_all.json`.
- Stop: any capture rejected twice at the spot; an align stop on lateral range; the plug touching the case before the align stage.
- After the session: `zed_arm_port.py` on the new runs; pass |fit - arm| <= 1.5 mm in x and y.

## Approvals needed

1. Part 2 (full runs with the fit's x, y): a perception change that moves the approach start by up to 8 mm against today.
2. Optional: `max_shift_mm` 8 -> 12 (only if the PC stays at the 2026-11-13 spot).
3. Copy the three zed_v5 weight files to the rig (not in `deploy/bundle`).

## Files

- `deploy/bimanual/zed_edgefit.py`, `deploy/bimanual/chain.py` (capture stage), `deploy/bimanual/netcache.py` (preload)
- `deploy/config/bimanual_align10_all_edgefit_capture.json`, `deploy/config/bimanual_align10_all_edgefit.json`
- `deploy/checks/zed_edgefit_realio.py`
- `scripts/perception/zed_edgefit_replay.py`, `zed_edgefit_summary.py`, `zed_edgefit_dry_summary.py`
- `results/placement/zed_edgefit/replay.json`, `results/placement/zed_edgefit/dry/` (run folders, `summary.json`, `configs/edgefit_truth_model.json` = dry-run-only copy)
