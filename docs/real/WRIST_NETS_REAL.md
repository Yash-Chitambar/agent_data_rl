# Wrist nets on real frames

Branch `policy/wrist-real`. Offline work: no robot motion. One collect run, one reader spot, one PC. Numbers: `results/real/wrist_nets/`.

## Result

**Root cause: the nets had not seen the real PC.** The appearance of the PC panel and lid alone brings the slope down to the real value. Preprocessing is correct. The wrong in-hand pose has a smaller, separate effect. The range of the fine net adds to it.

New nets pass the 0.8-1.2 bar on hold-out real frames, on all axes and gap bands. Their A/A2 height difference is 5.3 and 5.6 mm: the bar's "about 6 mm" has no stated tolerance, and vision reads 3.9 mm. Not run on the robot or in closed loop.

## Step 1: diagnosis

Same 150 logged `q_b`, one change at a time, slope against the same real label (`scripts/real/wrist_diag.py`, `diagnosis.json`). Slope, `wrist_v14w`, all gaps, n = 150:

| Frames | Across | Up | Gap |
|---|---|---|---|
| Sim, old reader, old PC: what the net trained on | 0.94 | 0.88 | 0.98 |
| Sim, same, training visual DR | 0.89 | 0.82 | 0.96 |
| Sim, real in-hand pose and reader look, old PC | 0.82 | 0.84 | 1.03 |
| Sim, old reader, **real PC look** | 0.61 | 0.39 | 0.61 |
| Sim, real in-hand pose and reader look, **real PC look** | 0.44 | 0.33 | 0.59 |
| **Real frames** | **0.50** | **0.32** | **0.59** |

Intervals are in `diagnosis.json`; for real frames, 95 % bootstrap: 0.45-0.55, 0.27-0.37, 0.56-0.62.

`wrist_v18ws`, gap under 10 mm, |x|, |y| under 6 mm, n = 46: sim 0.87, 0.89, 0.84; real 0.48, 0.18, 0.47.

Cause by cause:

1. **Preprocessing or geometry: not the cause.**
   - Channel order is right: swapping lowers the slope.
   - Scale 0.9-1.2 and shifts of 10 px do not bring the slope back. On `wrist_v14w`, all gaps: best across 0.55, up 0.40, gap 0.65, each at a different setting.
   - Logged policy image against a fresh `real_policy_image` of the raw frame: mean difference 0.29 grey levels.
   - With the fitted port and the measured in-hand pose, sim and real overlay: gripper, panel and plug fall on the same pixels by eye.
2. **In-hand pose: real, smaller.** The real plug sits about 16 px below the sim's in the 320 px image. Slope falls from 0.94, 0.88, 0.98 to 0.82, 0.84, 1.03. For the fine net, pasting the real reader on sim frames takes up from 0.89 to 0.43. It also shifts constants.
3. **Reader length: not a cause.** The 29.6 mm is the silver plate's width (`docs/real/INHAND_REAL.md`). The body is 41.26 mm. The brief's 29.6 mm body was wrong.
4. **Appearance: the main cause.** Real ports are bright metal rims on a mid-grey panel. In the sim they are dark holes on a light panel. Stickers and lid also differ. Textures taken from real frames, in the sim, reproduce the real slope within about 0.1 on most axes (the table above). Blur alone does not: sim blur of 2 px keeps 0.92, 0.85, 0.96.
5. **Range: real for the fine net, not the main cause.** `wrist_v18ws` on sim at these poses: 0.91, 0.99, 0.96 inside its training range (|x|, |y| under 3.5 mm, n = 21), 0.67, 0.71, 0.41 over all frames. Trained on lateral +/-3 mm, gap 3-10 mm; the sweep reaches +/-15 mm and 44 mm.
6. **Regression to the mean: a symptom.** A ridge probe on `wrist_v14w`'s last features, fitted on training blocks, reads the hold-out at 0.91, 0.74, 0.75. The signal is in the features.

**Geometry fit** (`scripts/real/wrist_geometry_fit.py`, gradient correlation of the rendered PC with the real frames; reader not used). Port against the anchored port, training blocks, n = 83: across -1.84 [-2.84, -1.84], up -3.51 [-4.01, -2.51], gap +6.99 mm. The first fit (48 frames over all blocks, including hold-out blocks) gave -2.34, -3.01, +6.99 and was used to place the textures. It does not use the labels except through the anchor. It agrees with the ZED port within about 2 mm. Fit plus in-hand preset gives a label offset of (-1.07, -3.13, -0.58) mm; vision gives (-0.54, -2.29, -1.67). Both say the first anchor was high.

**Real stop gap.** The 11-12 mm comes from the labels' constant. Vision reads about 8 mm and the new nets read 8.1-8.6 mm at A and A2.

## Step 2: harness

`scripts/real/wrist_net_real_eval.py`. The collect sweep is cut, in visiting order, into 10 contiguous blocks of 15. Blocks 1, 4, 7, 9 are held out (n = 60), chosen from their pose ranges before any training. One guard frame each side (7 frames used by neither side). Training n = 83. A, A2, ai4 never train.

Reports slope by gap band with bootstrap intervals (frames, blocks), A/A2 differences against FK and vision, residual SD after removing a constant. The block interval has only 4 blocks, so it is narrow by construction. The frame interval ignores that neighbours correlate. Treat both as optimistic.

## Step 3: training

| Net | Start | Sim range: lateral, gap, rotation | Real data |
|---|---|---|---|
| `wrist_r10w` | `wrist_v14w` | +/-18 mm, 2-46 mm, 6.5 deg | none |
| `wrist_r10f` | `wrist_v18ws` | +/-14 mm, 2-22 mm, 5 deg | none |
| `wrist_r10w_ft_abs`, `wrist_r10f_ft_abs` | the above | same, 40k renders | **primary**: across, up, gap absolute with vision's offset |
| `wrist_r10w_ft_rel`, `wrist_r10f_ft_rel` | the above | same | comparison: across, gap absolute, up relative |

Sim: 160k renders, 30 epochs, `--inhand-mode real --reader-look v1 --inhand-real-spread 2`, body 41.26 mm, real PC textures (`blind_insertion/assets/pc_real_look`, from training blocks only), blur up to 2.5 px. Preset v1 at spread 2 covers all 13 rows of `inhand_real_v2` but draws axes independently, so it loses the x-pitch coupling. Fine-tune: 4000 steps at 1e-4, batches of 48 sim, 32 collect, 32 frames of ai1, ai2, ai3, ai5 (n = 229, FK differences within a run). Steps and weights: inner validation blocks 2 and 5, then retrained on all training blocks. Primary nets were named before the hold-out was read for any fine-tuned net. The sim-only nets were read on the hold-out first; the fine-tune settings did not change after that. Commands and logs: `results/real/wrist_nets/training/`, `scripts/perception/finetune_wrist_real.py`.

### Hold-out collect, slope [95 % frame bootstrap]

| Net | Gap band | n | Across | Up | Gap | Residual SD, mm |
|---|---|---|---|---|---|---|
| `wrist_v18ws` | all | 60 | 0.38 [0.34, 0.43] | 0.14 [0.09, 0.19] | 0.22 [0.20, 0.25] | 3.9, 4.7, 11.8 |
| `wrist_v14w` | all | 60 | 0.61 [0.52, 0.69] | 0.25 [0.19, 0.31] | 0.62 [0.58, 0.66] | 2.8, 4.2, 6.2 |
| `wrist_r10w` (sim only) | all | 60 | 0.98 [0.92, 1.04] | 0.63 [0.51, 0.77] | 0.95 [0.93, 0.97] | 1.1, 3.4, 1.4 |
| **`wrist_r10f_ft_abs`** | 0-10 | 27 | 0.99 [0.97, 1.02] | 0.93 [0.90, 0.96] | 0.97 [0.88, 1.07] | 0.26, 0.61, 0.26 |
| | 10-25 | 8 | 0.99 [0.92, 1.07] | 0.96 [0.87, 1.07] | 0.94 [0.65, 1.14] | 0.25, 0.43, 0.75 |
| | 25+ | 25 | 0.95 [0.90, 0.99] | 0.94 [0.86, 1.06] | 0.96 [0.88, 1.06] | 0.81, 1.23, 1.19 |
| | all | 60 | 0.97 [0.95, 1.00] | 0.94 [0.90, 0.98] | 0.97 [0.96, 0.99] | 0.56, 0.92, 0.92 |
| **`wrist_r10w_ft_abs`** | 0-10 | 27 | 1.00 [0.97, 1.03] | 0.94 [0.91, 0.99] | 0.90 [0.81, 0.96] | 0.26, 0.68, 0.33 |
| | 10-25 | 8 | 0.95 [0.86, 1.05] | 1.09 [0.98, 1.15] | 1.00 [0.86, 1.12] | 0.34, 0.54, 0.43 |
| | 25+ | 25 | 0.97 [0.93, 1.01] | 0.98 [0.94, 1.03] | 0.97 [0.92, 1.04] | 0.56, 0.61, 0.82 |
| | all | 60 | 0.98 [0.95, 1.00] | 0.96 [0.93, 1.00] | 0.99 [0.98, 1.01] | 0.44, 0.65, 0.60 |
| `wrist_r10f_ft_rel` | all | 60 | 0.97 | 0.94 | 0.97 | 0.61, 0.88, 0.97 |
| `wrist_r10w_ft_rel` | all | 60 | 0.98 | 0.96 | 0.98 | 0.45, 0.60, 0.71 |

Mean error against offset labels, all hold-out: abs nets within 0.4 mm. This is not independent: the offset came from all collect frames, and the fine-tune trains towards it. **Rel nets read up 4.2-4.7 mm low.** So the sim alone sets absolute up about 4 mm low, cause not found. Absolute up in the abs nets comes from vision's offset, so its error (about 0.6 mm, a guess) passes straight through.

### A and A2 at their stops (n = 5, 3 frames)

A2 minus A, mm. Intervals are frame bootstraps and are too narrow to mean much.

| | Across | Up | Gap |
|---|---|---|---|
| FK (labels, these frames) | -4.62 | **+6.84** | -1.41 |
| Vision tip | -4.9 | **+3.9** | -0.4 |
| `wrist_v18ws` | -1.25 | +0.66 | -0.91 |
| `wrist_v14w` | -1.99 | -0.36 | -0.86 |
| `wrist_r10f_ft_abs` | -4.98 | **+5.29** [5.18, 5.37] | +0.18 |
| `wrist_r10w_ft_abs` | -4.50 | **+5.64** [5.59, 5.69] | +0.35 |
| `wrist_r10f_ft_rel` | -4.92 | +5.81 | +0.28 |
| `wrist_r10w_ft_rel` | -4.42 | +6.20 | +0.35 |

There is no independent truth between FK and vision here. The nets lie between them, nearer FK.

Tip at the stops, across, up, gap:

| | A | A2 |
|---|---|---|
| Vision | -0.3, +5.0, 8.4 | -5.2, +8.9, 8.0 |
| Label + collect offset | +0.84, +5.0, 10.34 | -3.79, +11.84, 8.93 |
| `wrist_r10f_ft_abs` | +1.30, +5.59, 8.37 | -3.69, +10.87, 8.55 |
| `wrist_r10w_ft_abs` | +1.09, +4.80, 8.14 | -3.41, +10.43, 8.49 |

Gap at A follows vision, not the label. Across stays 1.4-1.8 mm from vision at both stops, and follows the label plus the collect offset. A2 up is 1.5-2.0 mm above vision. So these nets may not follow all of the in-hand difference between grasps. This matters for insertion: the across disagreement is larger than the port clearance. Not resolved.

### Other runs, never trained on

Settled frames in the sweep's range, slope [95 %]:

| Net | Frames | n | Across | Up | Gap |
|---|---|---|---|---|---|
| `wrist_r10f_ft_abs` | ai4 | 38 | 0.99 [0.91, 1.03] | 1.08 [1.03, 1.15] | 0.99 [0.98, 1.00] |
| | A + A2 | 81 | 1.03 [1.01, 1.06] | 1.02 [1.00, 1.05] | 0.94 [0.93, 0.95] |
| `wrist_r10w_ft_abs` | ai4 | 38 | 0.96 [0.89, 0.98] | 1.03 [0.99, 1.07] | 1.01 [0.99, 1.03] |
| | A + A2 | 81 | 1.00 [0.97, 1.02] | 0.94 [0.93, 0.96] | 1.00 [0.99, 1.01] |

All frames, with transport and moving frames (out of range; `eval_*.json`), are lower: 0.77-0.94 across and up. I added the settled, in-range subset after seeing low values on all frames for the sim-only nets; both are reported. The ai4 last-frames estimate is about 1 mm from vision on up and under 0.6 mm on across and gap, but the frames are not exactly vision's stop frames.

### Still good in sim

Renders held out of training, DR, n = 2048, absolute error p50 / p99, across, up, gap mm: `r10f_ft_abs` 0.16 / 0.84, 0.42 / 2.19, 0.32 / 1.50. `r10w_ft_abs` 0.18 / 1.10, 0.46 / 2.51, 0.40 / 1.86. Slope 0.99-1.00.

Sim at the real poses (real look), n = 150: `r10f_ft_abs` 0.95, 0.91, 0.97; `r10w_ft_abs` 0.99, 0.93, 1.02. Old look: 0.93, 0.92, 0.86 and 0.98, 0.98, 0.90.

Round 9's p50 on its own narrower test: v18ws 0.06, 0.18, 0.10. The r10 sim errors are higher: wider range and fine-tuning pull. No closed-loop sim was run.

**Rotation (hold-out, n = 60):** slope rx, ry, rz 0.72, 0.93, 0.86 (`r10f_ft_abs`), 0.87, 0.97, 0.93 (`r10w_ft_abs`). Trained on differences only. Absolute rotation comes from the sim. Mean difference against labels: about +0.7, +1.8, -0.3 deg, with labels assuming the plug was straight at the pressed seat. Not resolved.

### What this does and does not show

Shows: on one run and one grasp, the nets read poses they did not train on, across the sweep's range, with about the right scale. On other grasps (A, A2) and another day (ai4), the scale holds too.

Does not show:

- Another reader spot, another pick grasp distribution, PC placement or lighting. Hold-out frames share the run, grasp and light of the training frames, and neighbouring blocks overlap in pose.
- ai4 shares a day with ai1, ai2, ai3, ai5, which trained on differences.
- Another PC. The textures make these nets specific to this unit.
- That the absolute tip is right: the offset is an estimate, and vision and net disagree by 1.5 mm across at the stops.
- That closed-loop behaviour improves. Nothing ran in a loop.
- `inhand_ih5`: not retrained or checked.

## Step 4: runtime

Config: `deploy/config/bimanual_pick3_wrist_r10.json`, the same as `bimanual_pick3.json` except `wrist_net` and `fine_net`. The chain's own `wrist_loader` loads both through those names (checked; `checks.json`). The robot run and the dry run have not been run with this file.

- **Input:** unchanged. 320 x 320 RGB from `real_policy_image`, area resize to 192. `PoseNet`, width 48, `wrist_mode` resize. No code change in the chain.
- **Output:** unchanged axes. 6 values, mm and deg; tip in the port frame. Meaning changed: it now reads the real plug.
- **Time:** one frame, resize plus net: 2.5 ms p50, 2.7 ms p99 on a V100; about 9.5 ms on CPU (this server, all threads). Same size as the round 9 nets (23.7 MB).
- **Weights:** `runs/perception_real/` on mason, gitignored. Copy to the rig. They are not in the bundle, so `bundle verify` does not cover them. sha256 prefixes: `wrist_r10f_ft_abs` 384e7b43b7e0e937, `wrist_r10w_ft_abs` 78064b5eeab8bbce.

For the graph agent, before use:

1. **`stop: 6.0` now means a real 6 mm or so.** The old nets stopped at a real 8 mm or more. Check this against your guards and the seat detector. I changed no limit.
2. **`--fine-below-mm` 7 is unchanged.** The fine net is good to 22 mm and the coarse net is close to it near the port: both are usable.
3. **Dry run:** the default sim look is the old PC. Slopes hold (0.93, 0.92, 0.86), constants not checked. Use `pc_real_look` and `inhand_real` for a fair sim.
4. **The nets have no refusal.** They always answer, even out of range.

**Alongside port vision, not instead of it.** Hold-out, frame by frame, vision detected 50 of 60:

| Net - vision | Across | Up | Gap |
|---|---|---|---|
| `r10f_ft_abs`, gap under 15 mm, n = 26: mean, SD mm | -0.07, 0.25 | -0.42, 0.71 | -0.16, 0.46 |
| `r10w_ft_abs`, gap under 15 mm, n = 26: mean, SD mm | -0.08, 0.26 | -0.81, 0.88 | -0.29, 0.46 |

Their errors against the labels correlate weakly (-0.27 to 0.32), so they are close to independent checks. The net is fast and always answers, so use it for the servo. Vision (82 ms) can refuse, and it sees the plug directly, so use it as the check before the hand-over: a disagreement well above these SDs means do not trust the pose. A threshold is a guess until it has run. Neither can replace the other yet. They disagree by 1.5 mm across at A and A2, and no truth separates them.

## Cheapest next data

A second collect run with another reader spot and a light-contact anchor would test the hold-out claim on a new grasp. It would also fix the net-against-vision question at A and A2.
