# Port vision: shell edge reference against a bright patch (2026-11-14)

Server work, logs only. **Nothing here ran on the arm.** Task: the align stop of `full_163924_all9_align_1` (2026-11-13 16:39, `SESSION_20261113_duchamp3.md`).

New, all off by default:

- `deploy/bimanual/port_vision.py`: `plug_ref()`, `ref_fix()`, `_edge_near()`, `detect_plug(und, ref=None)`, `measure(..., plug_ref=None, ref_tol_px=None)`. Without `plug_ref` the code path is the old one.
- `deploy/bimanual/align.py` (`PortVisionSource`): source key `shell_ref` (default false). On: `prepare` also stores the shell reference and passes it to every `measure` (step stage and servo worker).
- `deploy/config/bimanual_align10_pv.json` = `bimanual_align9_all.json` + `align.source.shell_ref: true`, `ref_tol_px: [1.5, 3.0]`. Only `align.source` differs. **No limit changed.**
- `scripts/real/port_vision_width_replay.py` -> `results/real_runs_review/port_vision_width.{json,csv}`.

## 1. Cause

### What the frames show (`full_163924_all9_align_1`, 21 step frames + 9 prepare frames)

| Cycle (tag) | Gap mm | Left edge px | Right edge px | End row px | Width px | Old result |
|---|---|---|---|---|---|---|
| prepare (9 frames) | 18.1 | 619.0-619.3 | 665.8-666.0 | 534.4-534.9 | 46.6-46.9 | ok |
| 1 (`align_000`, 3) | 18.1 | 619.1 | 665.9-666.0 | 534.7-534.9 | 46.8-46.9 | ok |
| 2 (`align_001`, 3) | 16.6 | 619.3-619.6 | 666.0 | 534.4-534.9 | 46.4-46.7 | ok |
| 3-7 (`align_002`-`006`, 15) | 12.6-12.8 | 619.1-619.6 | **673.0-674.6** (one frame: no width, 57.2 px) | 534.1-534.8 | **53.5-55.5** | rejected: depth 100.4-104.1 mm (gate 108-145) |

- The plug did not move in the jaws: left edge and end row stay within 0.5 px of the prepare frames in all 21 frames.
- The detector takes the strongest falling step right of the shell (`_edge`). In cycles 3-7 a light grey patch (grey 135-145, background 65-75, shell rim 165-175) fills the side rows next to the shell, cols 666-674. The rim-to-patch step is about 30 grey levels; the patch-to-background step is about 65. So the detector takes the outer edge of the patch: +8 px, width +7 px, depth -16 mm.
- The patch contrast (side rows, 2-8 px right of the reference edge, against 14-20 px right) is 53-61 grey levels in cycles 3-7, 4 in cycle 1 and 15 in cycle 2.

### What the patch is

- **Not the port lip.** Port 2 projects at v 447 px, 87 px above the shell end row (534 px).
- **Not on the shell.** It is out of focus, and it moves against the shell between cycles 2 and 3 (above the top right corner -> beside the right side). The shell is fixed in the image.
- **In the scene, at the case bottom edge**, about 6-9 mm right of port 2 (blob centroid back-projected on the table plane, 7 frames of this run). Its upper end sits on the dark row of the case bottom edge. A bright object or a glint at the case foot fits. On 2026-11-13 a white outline is also on the table in front of the case (not in 2026-11-10/11 frames). I cannot tell from the frames if the patch is tape, lint or a glint. **Operator: look at the case bottom edge 5-10 mm right of port 2.**

### Why this run only

1. **Day and hour.** In the same pose window (gap 11-15.5 mm, |up| <= 1.5 mm), the patch (> 20 grey levels) is on **0 of 378 frames** of 2026-11-10/11 (max 8) and on **20 of 47 frames** of 2026-11-13. On 2026-11-13 it shows in 16:39, 16:50 and 16:54 (max 53-66), weakly in 16:45 and 17:10 (18), and not at 17:01 (`depth_b_1`, 18 frames in the window, max 0). Light or scene change.
2. **Pose window.** The patch is next to the shell's side rows only at gap about 12-15 mm, when the case bottom edge passes the shell rows. `165005` crossed this window in one step (14.2 -> 11.0 mm): partial patch, width 46.2-46.8 px. `165400_all9_full_1` measured at 12.4-12.5 mm: 1 of 3 frames rejected (55.3 px), the cycle passed with 2 of 3.
3. **No motion on retry.** In `163924` the 3 mm gap step stopped at 12.7 mm (time-out). A rejected cycle is a retry without motion, so the same pose gave the same reject 5 times -> stop.

## 2. The rule (`port_vision.ref_fix`)

The plug is rigid in the jaws, and the camera is on the same wrist. So the shell edges stay at fixed pixels for the whole stage. Measured on all real align frames the old detector accepted (n = 4793): |left edge - reference| p99 1.07 px, |right edge - reference| p99 0.71 px, end row p99 7.1 px (larger end row moves come from the slip runs, below).

Reference: at the prepare stop, the median left edge, right edge, end row and width of the 9 frames (frames more than 1 px off the median width are left out; >= 3 needed).

A frame's edges change **only if all of these hold**:

1. Width differs from the reference by more than 1.5 px.
2. End row within 3 px of the reference.
3. Exactly one side edge within 1.5 px of its reference (the trusted edge); the other is more than 1.5 px off.
4. A true local step of the right sign, at least `MIN_EDGE` (10 grey levels), exists within 1.5 px of the other edge's reference.
5. The width with that step is within 1.5 px of the reference width.

Then that step is the edge. In every other case the old result stays, with its gates.

**Real grasp change against a patch.** A plug that moves in the jaws moves the whole shell in the image: both side edges (slide or turn), the end row (slide along or pitch), or both edges in opposite directions (depth). A patch moves one side edge outward only, and the true shell step stays at its old place. A rigid shell cannot give a one-sided widening with the other edge and the end row fixed. When the shell moved, rule 2 or 3 fails, no edge changes, and the old width gate (108-145 mm) and the grip change gate act as before.

Evidence that it does not hide a move:

- Self test (`--selftest`, the 15 failed frames of cycles 3-7): with the true reference 15/15 measured. With a reference as if the shell had moved (whole shell +-3 px or +2 px across, end row +4 px, width -3 or +3 px, right edge reference +3 px): **0 of 15 changed and 0 of 15 measured** in each of the 7 cases. `FAILED: none`.
- The 2026-11-03 slip runs (`align_full1`: grip gate, reader moving in the jaws; `align_full3` x2: reader lost): end row moved up to 12.9, 6.8 and 7.0 px, left edge up to 1.8 px. The rule did not fire on any of their frames (0 changes). Their old results are kept.
- Off by default: with `shell_ref` false, `PortVisionSource.kw` and `measure` are as before (robot code path on the failed frames: 6 of 21 measured, as on the rig).

## 3. Replay on every real align frame

All `full_*` folders of 2026-11-0* and 2026-11-1* with raw right-wrist align frames: **44 runs, 4824 frames.** Shell depth = the logged prepare depth (accepted in all 44). Old = `measure(raw, plug_z_mm=z)`, new = `measure(raw, plug_z_mm=z, plug_ref=ref)`.

| Frames | n | Both ok | Identical (max diff across/up/gap) | Both rejected | Recovered | Newly rejected |
|---|---|---|---|---|---|---|
| prepare (`align_depth_*`) | 396 | 396 | 396 (0 / 0 / 0 mm) | 0 | 0 | 0 |
| step cycles (`align_NNN_i`) | 3472 | 3443 | 3443 (0 / 0 / 0 mm) | 13 | 16 | 0 |
| servo (`align_servo_*`) | 247 | 247 | 247 (0 / 0 / 0 mm) | 0 | 0 | 0 |
| wrist net frames (`alignN_NNN`) | 687 | 685 | 685 (0 / 0 / 0 mm) | 2 | 0 | 0 |
| in-hand (`align_inhand`) | 22 | 22 | 22 (0 / 0 / 0 mm) | 0 | 0 | 0 |
| **all** | **4824** | **4793** | **4793 (0 / 0 / 0 mm)** | **15** | **16** | **0** |

- The rule fired on 16 frames, all recovered, all right edge: 15 in `163924` (cycles 3-7) and 1 in `165400_all9_full_1` (`align_003_2`, -> (0.09, -1.82, 12.48) mm; the other two frames of that cycle read (0.07, -2.02, 12.37) and (-0.02, -1.89, 12.51)). On every other frame the result is bit-identical.
- Still rejected by both (15): 12 frames "no shell side edges" in the two `align_full3` runs (2026-11-03, reader lost: no shell to see), 3 frames "panel pose failed / case edges missing" in `align_full10`. The rule does not touch them.
- Old replay against the logged values (code drift since the run): ok/reject agrees on 3470 of 3472 step frames; p99 difference (0.20, 0.58, 0.26) mm, max (0.75, 2.38, 1.52) mm (2026-11-03 runs, before later port vision changes).

### The failed run with the rule (cycles 3-7, median of 3 frames, mm)

Cycle 2 measured (0.93, -0.60, 16.63) and commanded (-0.96, +0.60, -3.0): aim (-0.03, 0.00, 13.63). The step timed out after 3.8 mm. FK tip change since cycle 2 (logged `tip_rel`): cycle 3 (-0.86, +1.12, -3.85), cycles 4-7 (-0.56..-0.65, +0.92..+0.94, -3.67..-3.82). FK prediction = cycle 2 measurement + FK change.

| Cycle | New measurement (across, up, gap) | FK prediction | New - FK |
|---|---|---|---|
| 3 | (-0.19, 0.34, 12.73) | (0.07, 0.52, 12.78) | (-0.26, -0.18, -0.05) |
| 4 | (0.21, 0.01, 12.66) | (0.37, 0.32, 12.96) | (-0.16, -0.31, -0.30) |
| 5 | (0.24, 0.14, 12.65) | (0.31, 0.34, 12.81) | (-0.07, -0.20, -0.16) |
| 6 | (0.17, 0.06, 12.64) | (0.28, 0.32, 12.81) | (-0.11, -0.26, -0.17) |
| 7 | (0.21, 0.08, 12.67) | (0.28, 0.32, 12.96) | (-0.07, -0.24, -0.29) |

All 15 frames measure (frame spread within a cycle <= 0.56 mm, gate 1.5). They follow FK within 0.31 mm. The step went 0.9 mm further in gap than asked (3.85 against 3.0); the vision sees the same. With the rule, cycle 3 would have gone on to the next step instead of a retry. The recovered right edge sits 0.3-0.8 px inside the reference (width 45.6-46.3 against 46.7 px): the patch lowers the step's outer side. That is 0.15-0.4 px on the plug centre, under 0.1 mm across.

## 4. Checks

- `deploy.checks.bimanual_conventions --config deploy/config/bimanual_align10_pv.json`: `FAILED: none`.
- `deploy.checks.align_gates`: `FAILED: none`. `deploy.checks.align_realframes` (default config): ok.
- `scripts/real/port_vision_width_replay.py --selftest`: `FAILED: none` (8 cases above).
- Robot code path (`PortVisionSource` with a fake camera on the 30 failed-run frames): `shell_ref` false 6 of 21 measured, true 21 of 21.
- Sim dry runs, `bimanual_align10_pv.json`, `--stage full --align-mode full`, n = 1 each: **seed 7043** aligned (15 cycles, 14.2 s), seat detected, seated after the release. **Seed 7044** aligned (12 cycles, 12.6 s), seat detected, not seated after the release. Reference `bimanual_align9_all` seed 7044 today: align stop "after the press: step limit" (35 cycles). MuJoCo Warp is not bitwise deterministic; the K7 doc reports the same seed spread. **The dry run uses `sim_source` (sim truth), not port vision. It does not exercise this change**: it only shows the config loads and runs. The real-frame evidence is the replay.

## 5. Limits

- One patch case, one day, 2 runs (16 frames). The rule is checked on 4824 frames from 44 runs, one PC placement, one camera.
- The rule only repairs one side edge. A patch on both sides, a patch that changes the end row, or a patch with no true shell step under it still gives the old reject (and the stop after 5).
- The reference is taken at the prepare stop. If the prepare frames already have the patch (gap about 12-15 mm at the stop), the reference is wrong; then rule 5 usually blocks a change (width check), and the old result stays. Not seen in the 44 runs (prepare at the 6 mm stop on 2026-11-03/06, at 16.8-19 mm since 2026-11-10).
- The servo path: the reference goes into `kw` before the vision worker starts (prepare runs first). Checked by code reading and on the 247 logged servo frames through `measure`, not in a live servo run.
- Tolerances 1.5 px (side edge) and 3 px (end row) come from the p99 of the edge spread (1.07 / 0.71 px; end row p50 0.33 px). They are not tuned on the failure.
- The patch's nature is not known. If it is an object at the case foot, remove it; the rule is a back-up, not a replacement.

## 6. Rig runbook (needs `bimanual_align10_pv.json`; no limit change, no approval needed for limits)

```bash
cd ~/peg_test   # rig checkout with this commit
.venv/bin/python -m deploy.checks.bimanual_conventions --config deploy/config/bimanual_align10_pv.json | tail -1   # FAILED: none
.venv/bin/python -m deploy.run_bimanual --stage full --align-mode align --config deploy/config/bimanual_align10_pv.json --name pv10_align_1 --stream
```

Before the run: look at the case bottom edge 5-10 mm right of port 2 (tape, lint, a bright spot) and note it. Do not remove it for run 1 if you want a direct test.

Check in the log:

- `align_prepare` has `shell_ref` with `w` 45.5-47.5 px and `n` >= 7. If `shell_ref` is null: the rule is off for this grasp (old behaviour); go on.
- Offline after the run: `PYTHONPATH=. .venv/bin/python scripts/real/port_vision_width_replay.py --days <date>` -> `results/real_runs_review/port_vision_width.json`: `newly_rejected` must be empty, `fix_fired` lists the frames the rule changed.

Pass: aligned in align mode; any changed frame follows FK within 0.5 mm (as above).
Stop and report to the server:

- The stage stops "no measurement (plug depth ... from its width)" 5 times with `shell_ref` on: send the run folder (both edges or the end row moved, or no step under the patch).
- A changed frame disagrees with FK by more than 1 mm, or the stage stops on "measurement vs FK".
- The grip change gate fires: a real grasp change; the rule must not have changed those frames (check `fix_fired`).
