# Push angle against the port axis, and the depth band (server task, 2026-11-19)

Task: measure from the real logs the angle between the robot's final push and the port axis. Test the operator's hypothesis (2026-11-19): "by hand I push directly parallel; the robot's gripper angle to the push is not perfectly perpendicular", as the cause of the 1-2 mm silver band.

**Nothing here ran on the arm.** Real numbers come from the logs of 2026-11-10, 11-11 and 11-13 (read-only). CPU only for the analysis; GPU 4 for three dry runs. n is small: read the per-run table, not a test statistic.

## Short answer

- **The operator's fact is right: the gripper is 1-7 deg off the port axis at the push.** At the push start the hand (plug as rigidly held) is pitched -5.6..+6.1 deg and yawed +1.1..+3.3 deg against the panel (n = 16 insertions). The align stage squares the plug to the panel within 1 deg (rx -0.98..-0.13, ry -0.06..+0.38 deg at the hand-over, n = 16). **K4 then turns the hand by 2-7 deg during the 1-2 s insertion**, and the push keeps that orientation.
- **The push vector itself is on the port axis.** The commanded push is horizontal at the v1 yaw: 0.2-1.0 deg from the port-vision axis in yaw, 0.0-0.2 deg in pitch (n = 16). The commanded motion is straight (target displacement along the axis within 0.1 mm per 15 mm).
- **The angle does not explain the band in these logs.** The 4 fully-in runs had the *largest* angles at the push end (4.3-7.3 deg total; pitch -6.9..-2.1, yaw +2.3..+3.8). The 4 short runs with a push had 1.8-5.9 deg. The 3 short runs with no push: 2.6-6.4 deg at the seat. No threshold separates full from short. The result is confounded with the push type and the date (all 4 fully-in runs: 15 mm position push, 2026-11-10/11 PC spot).
- **What differs is how the push moves the hand, not its start angle.** The 15 mm position push turns the hand by a further pitch -1.1..-2.1, yaw +1.1..+1.4 deg, and sweeps the (rigid-grasp) tip 2.6-3.1 mm across and 4.8-9.6 mm down while the hand advances 1.9-2.8 mm (n = 8). With the plug in the port, this is the reader turning in the jaws (or the jaws sliding on it) under 3-5 N of lateral load (FORCE_BUDGET). The force pushes hold the hand within 0.8 deg and 0.6 mm across (n = 2) and leave the plug short. This agrees with FORCE_BUDGET: the lateral load separates the labels, the axial force does not.
- **Wrist vision during and after the push is not a valid angle truth.** Port-vision panel pose on the 2 Hz stream frames disagrees with FK by a constant offset that stays after the load ends (release stage: roll -2.5..-5.5, yaw +0.8..+2.7 deg, pitch either about 0 or about -5 deg, n = 13 frames over 11 runs). An elastic arm deflection would vanish with the load. So it is a vision bias at close range with the port plugged. Observation only: the -5 deg pitch mode came on 3 of 4 full runs and 0 of 5 short runs with frames; it may depend on the plug depth (occlusion), not on the angle.
- **Fix, as a test (off by default, new config):** `insert.final_push.rot_ref: "handover"` turns the force push's rotation spring back toward the hand-over orientation (squared by the align stage), up to 3 deg, under the unchanged 0.5 N m moment cap. Config `bimanual_align13_wide_k7_pushangle.json`. It is a new hand motion: **operator approval needed.** First, a zero-risk hand test decides if the angle matters at all (runbook, step 1).

## 1. What I checked and how

Script: `scripts/real/push_angle.py` -> `results/real_align/push_angle/push_angle.json`. Runs: every real insertion with a seat on the three dates (n = 16; 12 with a final push: 10 position, 2 force). Labels: DEPTH_SPRINGBACK section 4 (full n = 4, short n = 7, "seated" with no depth word n = 5).

Port axis truth per run:

| Source | What | Used for |
|---|---|---|
| Port vision ("arm") | Panel PnP pose of every align frame before the hand-over (logged `panel_rv_deg`), hand-eye calibration, believed FK (encoder + rig joint offsets). Yaw and pitch of the panel normal. Per-run SD 0.06-0.29 deg yaw, 0.06-0.15 deg pitch (n = 28-85 frames per run). | The axis frame of every angle below |
| v1 ZED yaw | The `zed_pc` estimate of the run | The push direction the chain used |
| Edge fit | `results/placement/zed_edgefit/replay.json` | Cross-check: 1.55-2.48 deg below the arm (n = 15); absolute yaw stays open (ZED_EDGEFIT) |

Angles: rotation vectors in the port axes (x across, y up, z out of the port): rx = pitch, ry = yaw, rz = roll; deg.

- Plug against the panel at the hand-over: the align stage's last vision reading (`rv_fused`: panel PnP + camera mount + mean real grasp; per-grasp pitch SD 0.6 deg stays as an error).
- Plug against the panel later: that reading composed with the FK hand turn since it. This assumes a rigid grasp, so after the plug enters the port it is the **gripper's** angle against the port axis (the quantity of the hypothesis), not the plug's.
- Push motion: plug-tip FK displacement (rigid grasp) and the commanded (target) displacement, from the push start (seat + 0.2 s) to the end of the hold (position) or the unload start (force).
- Plug tilt in the jaws from wrist frames at rest: not measurable. The panel fit fails or is biased with the port plugged (section 3), and the reader-outline fit fails with the panel behind it (ALIGN_STAGE). The in-hand estimate at the insert start (`inhand6`) gives rx -1.98..+0.40, ry -0.76..+1.30 deg (n = 16) and does not separate the labels.

## 2. Results (n = 16 insertions)

Columns: push-vector error = v1 yaw minus port-vision yaw / port pitch (the push is horizontal); hand-over = plug against the panel at the hand-over (rx, ry); insertion turn = FK hand turn from the hand-over to the seat (rx, ry, rz); push start / push end = gripper against the panel (rx, ry); push turn = FK hand turn during the push (rx, ry); tip drift = rigid-grasp tip motion during the push, across, up / advance (mm).

| Run | Date | Label | Push | Push-vector error yaw / pitch | Hand-over rx, ry | Insertion turn rx, ry, rz | Push start rx, ry | Push turn rx, ry | Push end rx, ry | Tip drift across, up / in (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| `kjrc_1` | 11-10 | short | none | -0.6 / +0.1 | -0.3, +0.1 | +6.7, +1.3, +3.5 | +6.1, +2.0 | - | - | - |
| `kjrc_2` | 11-10 | short | none | -0.6 / -0.1 | -0.7, +0.1 | +4.1, +0.8, +3.7 | +3.2, +1.2 | - | - | - |
| `kjrc_3_push` | 11-10 | short | position 5 mm | -0.9 / -0.1 | -0.6, +0.2 | +3.7, +0.6, +2.8 | +2.6, +1.1 | -0.2, +0.3 | +2.4, +1.5 | -0.5, -1.7 / 1.0 |
| `k4_1` | 11-10 | short | position 5 mm | -0.4 / -0.0 | -0.9, +0.0 | -3.5, +1.8, +2.7 | -5.2, +2.0 | -0.3, +0.2 | -5.5, +2.2 | -1.1, -0.9 / 0.6 |
| `k4_3` | 11-10 | **full** | position 15 mm | -0.2 / +0.0 | -0.5, -0.1 | -4.5, +1.1, +2.2 | -5.6, +1.2 | -1.4, +1.1 | -6.9, +2.3 | -3.0, -5.4 / 2.0 |
| `k4_4` | 11-10 | seated | position 15 mm | -0.8 / -0.1 | -0.3, +0.2 | -0.1, +2.0, +4.4 | -0.8, +2.4 | -2.1, +1.4 | -2.9, +3.7 | -2.6, -9.6 / 2.8 |
| `k4_5` | 11-10 | **full** | position 15 mm | -0.6 / -0.1 | -0.1, +0.2 | -2.5, +1.6, +3.5 | -3.4, +1.9 | -1.3, +1.2 | -4.8, +3.0 | -2.7, -5.4 / 2.5 |
| `k4_6` | 11-10 | seated | position 15 mm | -0.7 / -0.0 | -0.3, +0.0 | +0.0, +2.3, +5.1 | -0.8, +2.6 | -1.2, +1.2 | -2.1, +3.9 | -2.8, -6.2 / 2.5 |
| `demo_1` | 11-11 | short | none (skipped) | -0.7 / +0.2 | -0.4, +0.2 | -1.1, +1.6, +4.0 | -1.9, +1.8 | - | - | - |
| `demo_2` | 11-11 | seated | position 15 mm | -0.2 / +0.2 | -0.6, +0.1 | -1.0, +1.0, +3.0 | -2.2, +1.4 | -1.1, +1.3 | -3.3, +2.6 | -2.9, -4.8 / 2.3 |
| `demo_3` | 11-11 | seated | position 15 mm | -0.5 / +0.0 | -1.0, +0.1 | +1.4, +1.9, +3.8 | -0.2, +2.3 | -1.1, +1.2 | -1.3, +3.5 | -3.0, -6.3 / 2.4 |
| `demo_4` | 11-11 | **full** | position 15 mm | -1.0 / -0.1 | -0.5, +0.4 | +0.0, +2.1, +4.3 | -1.0, +2.6 | -1.1, +1.2 | -2.1, +3.8 | -3.1, -5.5 / 2.1 |
| `speed075_1` | 11-11 | seated | none (skipped) | -0.7 / -0.0 | -0.5, +0.3 | -1.6, +3.0, +3.3 | -2.5, +3.3 | - | - | - |
| `fastmove_1` | 11-11 | **full** | position 15 mm | -0.7 / +0.0 | -0.6, -0.1 | -3.6, +1.8, +2.9 | -4.8, +1.8 | -1.4, +1.1 | -6.2, +3.0 | -2.9, -5.8 / 1.9 |
| `all9_full_1` | 11-13 | short | force 10 N | -0.4 / +0.1 | -0.7, +0.1 | -0.1, +1.3, +3.1 | -1.2, +1.5 | -0.1, -0.4 | -1.3, +1.2 | +0.6, -0.0 / 2.6 |
| `all9b_full_1` | 11-13 | short | force 14 N | -0.5 / -0.0 | -0.6, +0.2 | -0.8, +2.3, +4.3 | -1.7, +2.6 | +0.8, -0.6 | -0.9, +2.1 | +0.4, +3.2 / 3.9 |

(The run without a push: "push start" = seat + 0.2 s. `all9b_full_1` centred the line up by 2.6 mm before the load.)

By label:

| Quantity | Full (n = 4) | Short, pushed (n = 4) | Short, no push (n = 3) |
|---|---|---|---|
| Push-vector error, yaw (deg) | -1.0..-0.2 | -0.9..-0.4 | -0.7..-0.6 |
| Gripper at push start, rx / ry (deg) | -5.6..-1.0 / +1.2..+2.6 | -5.2..+2.6 / +1.1..+2.6 | -1.9..+6.1 / +1.2..+2.0 |
| Gripper at push end, total angle (deg) | **4.3-7.3** | 1.8-5.9 | (2.6-6.4 at the seat) |
| Hand turn in the push, total (deg) | 1.6-1.8 | 0.4-1.0 | - |
| Tip drift across / up in the push (mm) | -3.1..-2.7 / -5.8..-5.4 | -1.1..+0.6 / -1.7..+3.2 | - |

Reading:

- The push direction error (<= 1 deg against port vision; about 2.5 deg against the edge-fit yaw) is the same in both groups. It is not the cause.
- The gripper angle is real (1-7 deg) but larger on the fully-in runs. If a tilted gripper jammed the plug, the full runs would be the short ones. The data do not support the angle as the cause of the band at this n.
- The consistent signal is the position push: it turns the hand by about 1.7 deg and drives the hand 5-6 mm down and 3 mm across against the plug (3-5 N lateral, FORCE_BUDGET). All 4 fully-in runs had it. The two 5 mm position pushes (short) and both force pushes (short) did not. This is the same confound as FORCE_BUDGET: push type, date and PC spot change together.
- A plausible mechanism (not proven): the lateral load and the hand turn work the plug against the port walls (a wiggle) and overcome the last 1-2 mm of the latch, as a hand does. A stiff pure-axial push at a fixed angle does not.

## 3. Where the angle comes from

| Source | Size (n) | Corrected today? |
|---|---|---|
| PC yaw estimate (v1) used for the push direction | -1.0..-0.2 deg against port vision (16); align13 uses zed_v5: -0.8..+0.8 deg (ZED_EDGEFIT, n = 41) | No, but small. Absolute yaw open by about 2 deg (edge fit) |
| Port pitch | -0.14..+0.16 deg (16) | Not needed |
| Step alignment (`align.angle`) | leaves rx -0.98..-0.13, ry -0.06..+0.38 deg against the panel (16); rz (roll) -3.3..-1.7 deg, not corrected | Yes: rx, ry, by vision with the mean grasp |
| **Policy (K4) during the insertion** | **rx -4.5..+6.7, ry +0.6..+3.0, rz +2.2..+5.1 deg** from the hand-over to the seat (16). ry is always positive | No: the push holds the seat orientation |
| Position push (PD, gravity sag, contact) | rx -2.1..-1.1, ry +1.1..+1.4 deg (8 of the 15 mm pushes) | - |
| Force push | within 0.8 deg (2): the rotation spring holds the push-start orientation | - |
| Arm compliance not seen by the encoders | Not measurable here: the vision offset stays after the load ends (below), so it is not elastic | - |

Wrist vision during the push (stream frames, `port_vision.panel_pose`, emax <= 3.5 px), hand turn by vision minus FK turn, both since the last pre-insert frame:

| Segment | n frames | Median rx, ry, rz (deg) | p5..p95 rx |
|---|---|---|---|
| Insertion (before the seat) | 57 | -0.04, +0.12, -0.23 | -2.4..+0.3 |
| Push | 60 | -3.0, +1.6, -3.8 | -5.9..0.0 |
| Release (no load, jaws opening) | 13 | rx bimodal about 0 or about -5; ry +0.8..+2.7; rz -5.5..-2.5 | - |

The offset does not go away when the load ends, and the PC press hand moved at most 0.25 mm (FORCE_BUDGET). So the panel pose is biased when the plug sits in the port and the camera is close (114-117 mm, often one side edge only). I did not use it as truth.

Sim side: K7 is the rig policy now. Whether K7 turns the hand as K4 does on the rig is not known from real logs (no real K7 insertion yet). The dry runs below print the push-start turn.

## 4. The change (off by default, new config)

| File | Change |
|---|---|
| `scripts/real/push_angle.py` | new: the analysis of sections 1-3 |
| `results/real_align/push_angle/push_angle.json` | per-run results and the vision rows |
| `deploy/bimanual/insert.py` | `final_push_force(..., q_ref=None)`; `final_push.rot_ref` ('start' when absent: as before) and `rot_ref_max_deg` (default 3). `run_insertion` keeps the hand-over angles (`q_handover`) and passes them. New event `insert_push_rot_ref` (wanted and applied turn, port axes) |
| `deploy/config/bimanual_align13_wide_k7_pushangle.json` | new: `bimanual_align13_wide_k7_fp_shadow` + `final_push.rot_ref: "handover"`, `rot_ref_max_deg: 3.0` |

Law: the rotation spring's reference is R_ref(f) = exp(s w) R_start, s = f / force_n, w = the turn from the push-start orientation to the hand-over orientation, clipped to 3 deg. So the turn ramps up and down with the load. Moment = 6 N m/rad x error, capped at `rot_m_max_nm` 0.5 N m (unchanged). Force 14 N, gates, joint caps, watches: unchanged. Default (`rot_ref` absent): the code path is the same as before (reference = push-start orientation).

Why the hand-over orientation: it is the last orientation that a measurement squared to the panel (rx, ry within 1 deg). It does not fix roll (rz) and it does not undo a real in-hand turn of the reader during the insertion.

Checks: conventions `FAILED: none`. Dry runs (SimIO, GPU 4, 2026-11-19, `--stage full --align-mode full`, seeds 7041-7046, plus the parent config at 7043 and 7044; they check the code path, not a rate): section 5.

## 5. Dry runs

SimIO, K7, GPU 4, 2026-11-19, `--stage full --align-mode full`. They check the code path, not a rate (n = 1 per row; MuJoCo Warp is not deterministic, so the parent-config row at the same seed is a different episode). Logs: `data/real_runs/2026-11-19/full_*_dry_pushangle_*` in the worktree.

| Config | Seed | Reached the push? | Rotation reference turn (port x, y, z), deg | Push | Unload | Sim plug at the end: seated, rx (deg) |
|---|---|---|---|---|---|---|
| pushangle | 7041 | no: capture stop (ZED PC estimate rejected) | - | - | - | - |
| pushangle | 7042 | no: align stop after the press (step limit) | - | - | - | - |
| pushangle | 7043 | yes, seat at 1.94 s | +1.95, -0.23, -1.01 (not clipped) | 14.0 N, advance 0.63 mm, off line 3.29 mm, left 0.70 mm, no stop | 14 -> 0 N in 3.5 s, no stop | yes, rx +1.81 |
| pushangle | 7044 | yes, seat at 1.74 s | +1.43, -0.18, +0.28 (not clipped) | 14.0 N, advance 0.54 mm, off line 1.55 mm, left 0.61 mm, no stop | no stop | no after the release (known sim release effect), depth 5.53 mm, rx +1.81 |
| pushangle | 7045 | no: wide_motion clearance stop (no hold candidate) | - | - | - | - |
| pushangle | 7046 | no: align stop after the press (step limit) | - | - | - | - |
| parent (`fp_shadow`) | 7043 | yes, seat at 2.14 s | (reference = push start) | 10.4 of 14 N, **STOPPED: hand advanced 4.11 mm > 4.0** | - | no, depth 5.62 mm, rx -1.44 |
| parent (`fp_shadow`) | 7044 | no: align stop after the press (step limit) | - | - | - | - |

- The new path runs with no error or stop on 2 of 2 pushes; the turn was 1.4-2.0 deg (under the 3 deg clip). In the sim K7 turned the hand by about 1.5-2 deg before the push (n = 2), less than real K4 (2-7 deg, n = 16).
- The stops before the push (capture, align step limit, wide_motion clearance) are upstream of this change; the parent config also stopped at the align step limit (seed 7044).
- The sim has no latch and no real mating force: it cannot show whether the turn removes a band.

## 6. Limits

- n = 16 insertions, 4 fully-in labels; every label result is confounded with push type, date and PC spot. One PC spot per date.
- The gripper angle uses the mean real grasp for the hand-over reading (pitch SD 0.6 deg) and a rigid grasp after it. After the plug enters the port the reader can turn in the jaws; the numbers are the gripper's angle, not the plug's.
- The port axis frame is port vision. Its yaw is tied to a v1 yaw of 2026-11-02 through the panel model; the edge fit reads 1.6-2.5 deg lower. This shifts every yaw value by the same constant and changes no comparison between runs.
- Wrist vision during the push is biased (section 3). No direct measurement of the plug tilt in the port exists.
- The proposed change is not tested on any real push. The sim has no real mating force or latch; the dry runs only exercise the code.

## 7. Rig runbook

Step 1 (no robot, decides the hypothesis; operator by hand, kitchen scale as on 2026-11-19): push the reader into port 2 at about 5 deg off the axis (pitch, then yaw), with the same 6-10 N. 3 tries per angle.
- Fully in (no band) at 5 deg on 3 of 3: the angle alone does not leave a band. Do not run step 2 for the band; look at the lateral load / wiggle (FORCE_BUDGET `bimanual_align10_forcevec`).
- Band at 5 deg and none when straight: the angle matters. Run step 2.

Step 2 (needs the operator's approval of the new hand motion: a turn of at most 3 deg under the 14 N push, moment at most 0.5 N m):

```bash
C=deploy/config/bimanual_align13_wide_k7_pushangle.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C | tail -1        # FAILED: none
.venv/bin/python -m deploy.run_bimanual --stage full --align-mode full --config $C --name pushangle_1
#   prints "final push: rotation reference = hand-over orientation, turn [...] deg (port x, y, z; wanted [...])" before the push
```

3 runs. Per run report: the printed turn (applied, wanted), the `final push` and `unload` lines, the operator's band (mm) before and after the jaws open.
- Pass: no band on 3 of 3, no stop.
- Stop the block: any `STOPPED` line (advance, left hand, grip change, off line), a reader slip, or a wanted turn above 6 deg (the clip then hides most of the error; report it).
- Same band as `all9b_full_1` (about 2 mm) on 2 of 3: the angle is not the cause; keep the default config.
