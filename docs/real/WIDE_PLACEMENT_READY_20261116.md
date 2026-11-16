# Wide PC and reader placement: readiness and rig runbook (2026-11-16)

Task: is the chain ready for a full test of wider PC position / yaw (and reader) randomization at the next rig session? Worker on mason, GPUs 3 and 4, no rig access. **Nothing here has run on the arm.** No weights changed. No existing config changed. Sim numbers are sim only. Every number has its n.

Coordinates (right-base frame, as the rig logs them): PC = port 2 (x, y) in mm and yaw in deg (ZED convention; **+yaw = counter-clockwise seen from above**: the insertion direction turns from +x toward +y, the port face turns toward the right arm). x = away from the arm bases, +y = toward the left arm. Offsets "dx, dy, dyaw" are from the K7 box centre (231, 394) mm, 4 deg (the v1 reading of the 2026-11-10/11 spot; the edge fit puts that spot at about (231, 398.5) mm). Reader = plug tip (x, y) and yaw, offsets from (131, 448) mm, 92 deg.

## Answer

**Not ready as one wide test. Ready for a short, bounded step.** In the sim the combined chain passes every node in a box that is one-sided: the PC moved **away from the bases (+x), toward the left arm (+y) and turned counter-clockwise (+yaw)**. Toward the bases, toward the right arm and at negative yaw, the two arms come closer than on any real run, the right wrist roll reaches its joint limit, and the left wrist camera falls under its table clearance. These are geometry limits of the present hold and approach plans, not perception or policy limits. Perception (edge fit + zed_v5) and K7 cover the whole +/-25 mm, +/-15 deg box in the sim; nothing of the new perception path has run on the rig.

Sim safe box (every node passes, no clearance flag; n = 16 K7 full dry runs at 11 poses, 16 of 16 seated before the release): **dx 0..+25 mm, dy -12..+25 mm, dyaw 0..+15 deg**. Rig test box with margins: **dx 0..+15, dy -10..+15 mm, dyaw 0..+12 deg** (dy is capped by the edge fit: in the sim render it loses panel-edge inliers from dy +12..+25) (K7); with K4 only dyaw 0..+6 deg.

## Readiness per node

| Node | Ready for the wide box? | Evidence (n) | Missing |
|---|---|---|---|
| ZED PC pose: edge fit (use) + zed_v5 gate net + zed_v5 yaw | **Sim yes, rig not tested** | Capture dry runs with the real fit on the dark-panel render (`SIM_RELEASE_PANEL_20261116.md`), 125 poses (dx, dy in -25..+25 mm, dyaw -15..+15 deg): zed_v5 gate 125 / 125, fit gate 104 / 125, used pose error p50 0.67 / max 3.41 mm, fit yaw error p50 0.28 / max 1.73 deg. Fit rejects: dy +25 (panel edge 21-28 < 30 inliers, 15 poses), dy +12 with dyaw -15 or dx +12 (rms 1.28-1.38 px or 29 inliers, 6 poses). Real frames (replay, n = 89): wide capture gate passes 79 / 89; all 10 rejects are the zed_v5 two-net disagreement (2.86-4.24 mm, limit 2.82), 6 of them on 2026-11-11; \|fit - zed_v5\| <= 5.8 mm on all 89 (bound 8) | Rig capture at moved poses; zed_v5 weights on the rig |
| Why not v1 (today's gate net) | **No** | v1 saturates near y 395 mm: on the real map captures it read +1.3 of a +9.5 mm y move and -3.5 of -12.2 mm (fit). With v1 as the gate net and the 8 mm shift bound the chain rejects (251, 404) mm at 9 deg (v1 15.1 mm off; ZED_EDGEFIT dry run). Sim: v1 gate 148 / 216 in the step-2 box, yaw error to 13 deg (`ZED_V5_20261114.md`) | - |
| v1 yaw away from the spot | **No beyond about +5 deg** | Real map captures (changes against map_real, edge fit as truth): translations: v1 yaw within 0.0-0.6 deg of the fit (n = 4), zed_v5 within 0.1-1.0 deg; the +7.4 deg turn: v1 +5.6 deg (1.8 short), zed_v5 +7.2 deg. Against the arm at the spot (n = 41): v1 -0.56 deg, zed_v5 -0.09 deg (median). Sim: v1 yaw error p50 3.1 deg at +10, 7.4 at +15 (n = 8 each). So the wide configs take the zed_v5 yaw | Absolute yaw by tape (runbook) |
| Reader ZED net v2 | **Sim yes** | Full dry runs at reader dx -40..+40, dy -40..+40 mm, yaw 70-110 deg (n = 16): estimate error <= 3.0 mm, 15 of 16 seated before the release (1 align step limit at dy -40: the sim-noise class); reader yaw 110 deg passed 2 of 2 | Real frames at other reader spots (presence head never ran on the rig) |
| Pick | **Yes** | 80 of 81 grid runs held; 1 empty close at the corner (-25, +25, -15) with the reader moved (+20, -20, 100 deg) | - |
| Left hold / press reach | **Yes (kinematics)**, see clearances | Reach map: 210 / 210 poses of the rig box feasible, joint margin >= 0.51 rad right, >= 0.58 rad left (`results/placement/reach/pc_map.json`). Every grid run latched | - |
| Clearances and limits | **Only in the one-sided box** | Section "Clearances" | Approval items below |
| Approach | **Sim yes in the box** | Outside: 5 torque guard stops (right joint 2 12.9-13.4 N m, joint 4 5.3 N m) at dy +25 with dyaw -15: the arm models overlap there (link_4 to left link_2 to -36 mm) | Real wrist frames at new spots (wrist nets were tuned at one spot) |
| Align (port vision) | **Rig check** | Sim align uses `sim_source` (no port-vision render). Grid align stops are the known sim noise ("after the press: step limit", 7 of 67 K7 runs, also at the nominal pose class) | Port vision at new background and light |
| Insert: K7 | **Sim yes, never on the arm** | Batched sim (`K7_20261114.md`, n = 1024 per cell): every box cell >= 97.1 %. Grid: K7 seated before the release at every pose where the chain reached the insertion (39 of 39 runs that reached it). Replay of bad real hand-overs: K7 below K4 (full_A2 54.3 against 80.1 %, n = 256) | Rig runs; node-swap study (other worker) |
| Insert: K4 (A/B twin) | **Only dyaw <= +6 deg** | Batched: yaw +10 56.2 %, +15 2.5 % (n = 1024). Grid: press_abort at dyaw +15 (2 of 2) | - |
| Release | **Sim artefact mostly fixed** | Grid with the 27 mm sim release jaw (75 of 81 runs): 15 runs seated at the end of the insertion (true depth 5.44-5.65 mm) but not strictly seated after the release (other causes, `SIM_RELEASE_PANEL_20261116.md`). Real: 10 of 10 stayed in | - |

## Still missing before a full wide test

1. Rig evidence for the new perception path (edge fit in use, zed_v5 gate, zed_v5 yaw, reader_zed_v2 presence): part A of the runbook.
2. K7 on the arm (never run) and the K7-against-K4 question on bad hand-overs (node-swap worker).
3. A hold / press plan that keeps the arms apart toward the bases and at negative yaw, and a left press wrist pose that keeps the left wrist camera above 30 mm at +x / -y. Today these regions are blocked by geometry, and the sim dry run does not show it (the visual meshes do not collide in the sim; only the torque guard caught the worst corner).
4. Wrist nets and port vision at new backgrounds: only the rig shows them (collect the align frames of part B).
5. The edge fit at dy >= +12 mm: in the sim render the panel edge loses inliers (left gripper in view at the capture pose). On real frames the fit passed at the +9.5 mm map pose. Part A measures it.

## What changed

- New configs `bimanual_align11_wide_k7.json`, `bimanual_align11_wide_k4.json` (no existing config or code changed).
- New scripts `scripts/eval/wide_capture_grid.py`, `wide_clearance.py`, `wide_grid_summary.py`, `wide_safe_box.py` (analysis only, never on the rig).

## Configs

| Config | Content | Conventions |
|---|---|---|
| `deploy/config/bimanual_align11_wide_k7.json` | `bimanual_align10_all` + `zed_edge_refine` (mode use, start zed_v5, block of `bimanual_align10_all_edgefit`) + `chain_args.zed_gate: zed_v5_gate` (fit shift bound and yaw now against zed_v5) + `pick.reader_estimator: reader_zed_v2` + K7 (`runs/insert_K7.pt`, sha256 `9a6e23b2f225ccc1`) | FAILED: none |
| `deploy/config/bimanual_align11_wide_k4.json` | the same with K4 (`4d288f234ff1af10`): the A/B twin | FAILED: none |

No limit is looser than `bimanual_align10_all`. The zed_v5 gate (presence, OOD, two-net disagreement, v1's arm envelope) is a different gate, not a looser one; its OOD and disagreement thresholds are the trained ones. No separate capture config: `--stage capture` with the wide config moves nothing after the capture pose, so the fit is only printed and logged there (shadow by construction), with the exact gates of the full run.

Dry runs (`deploy.run_bimanual --dry-run --stage full --align-mode full`, sim):
- Before the panel fix: seed 7043, wide_k7, render fit: stops at the fit gate (light panel; expected). Truth-model copies (`results/placement/wide/configs/wide_*_tm.json`), seed 7043: K7 aligned 11.8 s, seat detected 1.06 s, true depth 5.60 mm; K4 aligned 9.0 s, seated 1.76 s.
- With the panel fix (real fit, configs as committed): 8 runs, seeds 7043 and 7045 (`results/placement/wide/realfit/`, `realfit_summary.json`). At the rig base B (dy +4): K7 2 of 2 and K4 2 of 2 seated before the release (K4 seed 7045 not strictly seated after it, depth 5.60 mm), fit error 0.57 and 0.76 mm, align 6.5-11.4 s. K7 at (+12, +4, +8): 2 of 2 seated, fit error 0.11 / 0.48 mm. K7 at (+12, +16, 0): 1 seated (fit 0.56 mm), 1 stopped at the fit gate (the stop works; same class as the capture-grid rejects at +y). No guard or safety stop.

## Sim placement grid (full chain)

`scripts/eval/placement_grid.py` with the truth-model copies (the fit replaced by sim truth + N(0, 0.5 mm / 0.5 deg); the real fit is checked by the capture grid above), speed 0.75, seeds 7043 (all) and 7045 (corners), reader at (131, 448) mm, 92 deg unless moved. 81 runs: K7 51 PC poses-runs + 16 reader runs, K4 14. Tables: `results/placement/wide/summary.json`, `summary.md`, `safe_box.json`.

First failing node per pose (K7; "ok" = seated before the release; "rel" = seated before, not strictly after the release; "al" = sim align noise; flags from the clearance check):

| dyaw | Poses that pass with no flag | Poses with a flag or a stop |
|---|---|---|
| +15 | (0, -25), (0, 0), (0, +25), (+25, 0), (+25, +25) | (-25, *): arm-arm in the hold 13-19 mm closer than any real run; (+25, -25): left wrist camera 2.3 / -3.7 mm table margin |
| +8, +10 | (0, 0) (K7 and K4) | - |
| 0 | (0, -12), (0, 0), (0, +12), (0, +25), (0, +30) rel, (+12, 0), (+25, +25) | (-12, 0), (-25, *): hold arm-arm -17..-28 mm; (0, -25): left camera margin 4.7 mm; (+25, 0): 3.2 mm; (+25, -25): -0.7 mm (SafeArm stop) |
| -8 | none | (0, 0): hold arm-arm -12 mm, transport 0 mm |
| -15 | none | every pose: hold / transport / home arm-arm, right joint 6 0.15-0.24 rad from the SafeArm box (at the Menagerie limit); dy +25: approach torque guard stops (5 runs); (+25, -25): left camera -2.1 / -8.5 mm |

K4 (seed 7043, n = 14): the same clearance pattern (it is the same motion); in addition press_abort at dyaw +15 (2 of 2).

## Clearances, reach and joint limits

`scripts/eval/wide_clearance.py` replays every dry run (ticks at 10 Hz, measured and target) on the rig checkers: the SafeArm joint box and the FlexChecker table clearances of `bimanual_align10_all` (motor frame, as on the rig), and the TwoArm model distances (all geoms of both arms, as in `PARALLEL_SCHEDULE_20261115.md`) arm-arm and arm-PC at the true sim angles. The teleported capture and home moves are replayed as straight joint paths from a real start pose. Reference: the same replay on 13 real runs of 2026-11-10/11/13 (`results/placement/wide/clearance_real.json`).

| Quantity | Real runs (n = 13) | Sim nominal (0, 0, 0) | Where it gets worse |
|---|---|---|---|
| Arm-arm in the hold (right finger vs left gripper) | -0.6..+14.6 mm | -6.5 mm | -dx: -17 (dx -12), -25..-38 (dx -25); -dyaw: -12 (-8), -15..-18 (-15); +dx makes it larger (+17 at dx +25) |
| Arm-arm in the transport | +11.4..+30.4 mm | +6.9 mm | -dx and -dyaw: to -31 mm |
| Arm-arm in the approach / home move | 42-50 / 47-70 mm | 48 / 65 mm | dyaw -15 with dy +25: -14..-36 / -1..-9 mm (overlap; the torque guard stopped these runs) |
| Right joint margin to the SafeArm box | 0.58-0.80 rad | 0.58 rad | dyaw -15: joint 6 (wrist roll) 0.15-0.24 rad, i.e. at the Menagerie joint limit |
| Left table margin (wrist_camera body, limit 30 mm) | 10.8-17.0 mm | 8-17 mm | +dx and -dy: 3.2 (+25, 0), 4.7 (0, -25), -0.7..-8.5 mm at (+25, -25): **SafeArm stops the left arm in the press there** |
| Left wrist camera to the PC (visual mesh) | -2.8..+1.3 mm | -11 mm (encoder DR) | +dx, -dy: 8-13 mm closer than nominal |

Notes: the model "touches" already on real runs (finger meshes at -0.6 mm in the hold, -6.3 mm in the insertion; camera mesh -2.8 mm); the real runs had no contact, so read the numbers as change against the proven runs. The sim distances include the encoder-offset DR draw (true angles), which shifts the nominal by up to 10 mm per seed.

**No pose in the recommended box needs a looser limit.** Poses that would need one (not proposed, avoid them): dx +25 with dy <= 0 (left `body_clearance.wrist_camera` 30 mm: margin -0.7..-8.5 mm at (+25, -25)); dyaw -15 (right joint 6 at its Menagerie limit). A different left press pose (wrist pitch) or right hold pose would open these regions; that is new motion code, not a limit change.

## Rig runbook (after the same-spot steps of `NEXT_RIG_20261115.md`)

Do this block only after `NEXT_RIG_20261115.md` step 2 (or 3) finished and the PC is at its marked base spot B (the 2026-11-10 spot after step 0, port about (231, 398.5) mm by the fit). Goal: 1 capture per pose, then 3-4 full runs. About 30 minutes.

### Approvals needed (before part B)

| Item | Change | Data |
|---|---|---|
| A1 | Edge-fit pose in use (approach starts where the fit says, up to 8 mm from the gate net) | `ZED_EDGEFIT_20261115.md`: fit vs arm +0.7 mm median, max 1.1 mm (n = 41) |
| A2 | zed_v5 as the gate net and its yaw | this doc, "Readiness" row 1-3 |
| A3 | reader_zed_v2 in the pick | sim <= 3.0 mm (n = 16); 40 real frames 2.1 mm median from v1 |
| A4 | K7 on the arm (new policy) | `K7_20261114.md` |
| A5 | PC moves off the proven spot (inside the rig box only) | this doc, safe box |

Part A (captures) needs no approval: no motion beyond the capture pose.

### Before

```bash
git pull
sha256sum runs/perception/zed_v5s0.pt runs/perception/zed_v5s1.pt runs/perception/zed_v5_gate.pt runs/perception/reader_zed_v2.pt runs/insert_K7.pt
# first 16: e486acdce74b045f 0adaf0b87fc62860 b937820a1fb248d0 e0baaa66df16661b 9a6e23b2f225ccc1 (copy from mason runs/ if missing)
for c in bimanual_align11_wide_k7 bimanual_align11_wide_k4; do
  echo "$c: $(.venv/bin/python -m deploy.checks.bimanual_conventions --config deploy/config/$c.json | tail -1)"; done   # FAILED: none, twice
.venv/bin/python -m deploy.run_bimanual --dry-run --stage full --align-mode full --seed 7043 --config deploy/config/bimanual_align11_wide_k7.json --name dry_wide_k7
```

### Tape marks (5 minutes, by hand)

- At B: tape an L at the left-front corner of the case and a straight tape line along the port panel face (the face with port 2), 150 mm long. Mark port 2 on that line.
- x = away from the arm bases, y = toward the left arm. Move the PC by its port-2 mark with a ruler (+/-1 mm).
- Yaw (+ = counter-clockwise seen from above): keep port 2 on its mark, turn the case, and measure the gap between the panel face and the B line 100 mm along the line from port 2: 8 deg = 14.1 mm, 10 deg = 17.6 mm, 12 deg = 21.3 mm.
- Write down each pose as measured (x, y mm from B; gap mm). The server needs it as the truth.

### Part A: capture only at every pose (about 1 minute each)

Poses in this order (from B, nearest first): **P0** B; **P1** x +10; **P2** y +10; **P3** yaw +8 (14.1 mm gap); **P4** x +10, y +10, yaw +8; **P5** x +15; **P6** x +15, yaw +12 (21.3 mm gap). B is about dy +4 in the box coordinates, so y +10 from B is dy +14 (the sim fit edge). Do not test toward the bases (-x), more than 10 mm toward the right arm (-y), more than 10 mm toward the left arm (+y), or clockwise (-yaw).

```bash
S='{"port_xy": [<x m>, <y m>], "yaw_deg": <d>, "method": "tape from B", "unc_mm": 1}'
.venv/bin/python -m deploy.run_bimanual --stage capture --config deploy/config/bimanual_align11_wide_k7.json --name wide_cap_P<n> --pc-spot "$S"
```

Read the lines `edge fit (...) -> ok/REJECTED`, `ZED PC estimate ... gate`, and the reader estimate.
- Pass per pose: gate True; `edge fit ... -> ok`; fit time <= 1.5 s; no motor or CAN error; the fit change from P0 equals the tape move within 2 mm and the yaw change within 1 deg; the reader estimate ok.
- At a reject, answer `r` (retake) once. Never `c`. A second reject is a result: write it down and go on to the next pose.
- Stop part A: a CAN or motor error at the capture; the fit misses the tape move by more than 3 mm at two poses (then no part B).
- Expected: the zed_v5 disagreement can reject (10 of 89 real frames; 6 of 8 on 2026-11-11). If it rejects at P0 twice, part B is not possible this session.

### Part B: full runs (needs A1-A5)

At most 4 runs. Each at a pose whose capture passed in part A.

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --align-mode full --speed 0.75 --demo --config deploy/config/bimanual_align11_wide_k7.json --name wide_k7_<pose>
```

1. **P0 (B), K7.** First K7 run on the arm and first run with the new perception at the proven spot.
2. **P4 (x +10, y +10, yaw +8), K7.**
3. **P6 (x +15, yaw +12), K7.** Only if 1 and 2 seated.
4. A/B, only if a K7 run failed in insertion: the same pose with `bimanual_align11_wide_k4.json` (K4), only at yaw <= +6 deg (P0, P1, P2).

- Pass per run: estimate = the edge fit, gate True; pick held; hold latched; aligned; seated; push and unload lines as in `NEXT_RIG_20261115.md`; operator band check (band mm, hangs / straight); no guard or SafeArm stop.
- Watch at the left press: the left wrist camera and the PC lid (the model puts them close; real runs 0-3 mm in the model). Stop on contact.
- Stop part B: any guard or SafeArm stop in the hold, transport or approach; a plug pulled out; the PC moves; two runs without a seat; K7 seats at P0 slower than 4 s or fails twice (then the A/B at P0 with K4).

### What the server needs back

The run folders (also the captures), the tape poses with their gaps, the operator band per full run, and any reject line.

## Limits

- Sim only; n = 1-2 dry runs per pose (a code-path and failure-mode check, not a rate). Rates of the insertion come from the batched K7 eval (n = 1024 per cell).
- The real image fit is checked on 89 real frames at 2-3 spots only; the sim capture grid uses the sim render with a dark panel and the nominal camera.
- The clearance model (TwoArm meshes) overlaps a little on proven real runs; the limits are relative to those runs.
- The safe box is one-sided because of the present hold and approach plans. The rig box is inside the sim box with 5-10 mm and 3 deg of margin.
- Absolute PC yaw is still open at 1-2 deg (fit -1.9 deg against the arm, n = 41); the tape gap gives the yaw change only.

## Files

- Configs: `deploy/config/bimanual_align11_wide_k7.json`, `deploy/config/bimanual_align11_wide_k4.json`
- Scripts: `scripts/eval/wide_capture_grid.py` (capture grid), `scripts/eval/wide_clearance.py` (limits and clearances), `scripts/eval/wide_grid_summary.py` (tables), `scripts/eval/wide_safe_box.py` (flags and the box)
- Results: `results/placement/wide/` (`grid/`, `capture/`, `realfit/`, `clearance*.json`, `summary.json`, `safe_box.json`, `configs/` = dry-run-only truth-model copies)

