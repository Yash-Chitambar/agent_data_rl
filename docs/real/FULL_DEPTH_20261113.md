# Full insertion depth (server task 1, 2026-11-13)

Task: [SERVER_TASKS_20261112.md](SERVER_TASKS_20261112.md) item 1. Data: the 2026-11-10/11 rig runs (`/home/yashchitambar/peg_test/data/real_runs/`), 12 insertions with logs (7 run folders of 2026-11-10, 16:10-17:05, are empty on mason).
Tools: `scripts/real/depth_analysis.py` -> `results/real_align/depth/real_depth.{json,png}`; wrist frames `results/real_align/depth/shell_end_frames.jpg`.
**Nothing here has run on the arm.** Sim numbers are sim only.

## 1. Is the sim port shorter than the real one?

**Answer: not in its geometry, but yes in the units the robot uses.**

- Geometry: the sim reader's plug stands 5.8 mm out of the neck (`builder.reader_variant`, exposure 5.8 mm; cavity 6.5 mm), so the sim seat depth is 5.8 mm (`seat_depth_m` 0.0058). The real plug stands 5.8 mm out of the neck (calipers, 3 readings, `measurements.csv`). They agree.
- Units on the rig: depth = FK advance of the hand along the port axis since the hand-over, minus the vision gap at the hand-over (seat-2). In these units the real port mouth is not at 0:
  - K4's contact head changes from "touching" to "inside" at FK depth +1.4 to +3.2 mm (first tick with inside share > 0.5; median 2.2 mm, n = 9 K4 insertions).
  - `speed075_1` stayed at 1.6 to 2.2 mm for 1 s with the head on "touching" (the plug on the port edge), then went in.
  - So the real port mouth is at about +1.5 to +2 mm: the hand-over gap is about 1.5 to 2 mm larger than the vision value (6.0 to 6.9 mm). Cause not settled (port-vision gap scale, or where the port plane is in the image).
- Fully in, in these units: **7.3 to 7.6 mm at rest** (operator "fully inserted" / "perfectly" / "really well": 7.56, 7.32, 7.36 mm, n = 3). 7.3 to 7.6 minus 1.5 to 2 = 5.3 to 6.1 mm of true travel: the 5.8 mm plug.
- The sim says "seated" at 5.8 mm, and the detectors accept from 4.8 mm (seat depth minus 1 mm). On the rig this is 2.5 mm before fully in. So the operator's idea is right in effect: for the policy and the detectors, the real port is about 1.5 to 1.8 mm "deeper" than the sim port.

Second difference: **the sim port has no mating force.** A real USB-C port resists the plug (contact springs and side latches; the spec allows 5 to 20 N to mate). The mjfit port has clearance and friction only. K4 was also trained with a force penalty above 10 N.

## 2. What the logs say about depth

"At rest" = no push force: after the 0.2 s measured-pose hold (K4), or after the push is dropped. Under load the FK depth reads 1.3 to 1.9 mm more (the arm, the fingers and the grasp are elastic), so depth under load is not plug travel.

| Run | Policy | Depth at the K4/K-J seat (load) | At rest after the policy | Push | At rest at the end | Hand up at the end | Operator |
|---|---|---|---|---|---|---|---|
| `demo_4` | K4 | 8.41 | 7.00 | 15 mm | **7.56** | -7.7 | fully inserted (best) |
| `k4_3` | K4 | 8.55 | 6.72 | 15 mm | **7.36** | -9.5 | seated, really well |
| `fastmove_1` | K4 | 8.65 | 6.72 | 15 mm | **7.32** | -7.9 | inserted perfectly |
| `demo_1` | K4 | 8.23 | 6.94 | skipped | 6.94 | -4.4 | seated, hangs slightly |
| `demo_2` | K4 | 7.46 | 5.82 | 15 mm | 6.69 | -7.5 | seated |
| `demo_3` (rerun) | K4 | 7.39 | 5.65 | 15 mm | 6.65 | -7.3 | seated |
| `k4_1` | K4 | 8.08 | 6.37 | 5 mm | 6.13 | -5.2 | seated, about 3 mm short |
| `kjrc_1` | K-J | 7.05 | 6.00 | none | 6.00 | 0.1 | seated, hangs |
| `speed075_1` | K4 | 6.82 | 5.90 | skipped | 5.90 | -4.5 | success |
| `kjrc_3_push` | K-J | 6.14 | 5.38 | 5 mm | 5.77 | -3.2 | seated, about 3 mm short |
| `kjrc_2` | K-J | 6.53 | 5.58 | none | 5.58 | 0.1 | seated, hangs, about 1 mm short |

(mm, TCP, seat-2 units; `k4_2` was pulled out and is not in the table.)

- **Depth at rest follows what the camera shows.** In the last wrist frame before the release (six runs of 2026-11-11, `shell_end_frames.jpg`), the visible silver band of the plug shell grows as the rest depth falls: `demo_4` (7.56) thinnest, then `fastmove_1` (7.32) and `demo_1` (6.94), `demo_2` (6.69), `demo_3` (6.65), `speed075_1` (5.90) thickest. By hand on the image: about 0.8 mm to 2.3 mm of band for a 1.66 mm difference in rest depth. A band stays visible on the best run: the plug stops in the port before the neck touches the case.
- **K4 alone stops 0.3 to 1.9 mm short of full** (rest 5.65 to 7.00 mm, n = 8 K4 seats; full 7.3 to 7.6).
- **The 15 mm push adds 0.56 to 1.00 mm at rest** (n = 5), and the three "fully in" runs are the three with the deepest K4 rest (6.72 to 7.00) plus a push. `demo_2` and `demo_3` had a push and stayed about 0.7 to 0.9 mm short: so the push did not reach a hard stop on those; it ran out of force.
- **Push force:** PD estimate 9.0 to 13.3 N peak along the port axis (n = 5). K4 at its seat: 4.5 to 11.3 N.
- **The push moves the hand down:** 5.5 to 6.3 mm more down during the push (hand at -7.3 to -9.5 mm at the end), with the plug in the port. The push target is a straight line, but the arm is a joint-space PD: when the axis is blocked, the hand gives way in the free direction. That is a side load on the port and it wastes push force.
- **The two skipped pushes** (`demo_1`, `speed075_1`): the hand was 3.5 and 3.3 mm low at the seat by FK (K4 did not pull up on these two). It is the hand, not a wrist-net reading. The 2.5 mm gate skipped the push and the reader stayed short or hanging.

### Check against the data review (T0, `DATA_REVIEW_20261113.md`)

Agreed: the spring-back of 1.5 to 1.9 mm when K4's lead is dropped; the push peak is at most 0.8 mm past K4's loaded depth; the hand goes down 7 to 9 mm; the skipped pushes are a real hand offset.
One difference: T0 says a hard stop and "more force needed" cannot be separated from the logs. Compared **at rest** instead of at the peak, the push adds 0.56 to 1.00 mm (n = 5), and the wrist frames confirm that pushed runs ended at different true depths (`demo_2`, `demo_3` show more shell than `demo_4`). So on `demo_2` and `demo_3` the plug was not at its stop: more travel was left. The limit is that n is small and the frames are read by hand.

## 3. Why the plug stops short

1. The seat detector stops K4 at 4.8 mm or more (rig units) with the axial speed under 6 mm/s for 0.3 s. That is up to 2.5 mm before fully in, while K4 still creeps in at about 3 mm/s (`demo_4`: 8.33 to 8.59 mm in the last 0.08 s). This early stop is what made the milestone reliable (it ends K4's search), so it stays.
2. K4 pushes with 5 to 11 N and was trained without a mating force, with a penalty above 10 N.
3. The 15 mm position push gives about the same axial force as K4 and a side load; it adds 0.6 to 1.0 mm at rest.
4. The push does not run when the hand is more than 2.5 mm off the axis at the seat (2 of 8 K4 seats).

## 4. Deploy change: force-mode final push

`deploy/bimanual/insert.py` `final_push_force`, used when `insert.final_push.mode` is `"force"` (default `"position"` = the 2026-11-10 push, unchanged).

- Each tick the right target = the **measured** joints + the joint step whose PD torque equals a chosen load at the hand: `kp dq = J^T [F, M]`.
- `F` = `force_n` along the port axis (ramp `ramp_n_s` 4 N/s x `--speed`, then hold) + a lateral spring (`lat_k_n_mm` 3 N/mm, at most 8 N) that keeps the hand on a line. `M` = a rotation spring on the start orientation.
- So the axial PD load is at most `force_n` (joint friction only lowers the real force), with no fixed 15 mm target, and the hand does not drift down.
- The joint torques of the load are scaled together to stay under `tau_max_nm` [6, 6, 6, 2.5, 2, 2] (safety: `tau_cont_nm` [9, 9, 9, 3, 3, 3]).
- Optional centring (`centre_max_mm`): before the force ramp, the line moves toward the port axis by the seat-2 lateral reading, at 2 mm/s. For the seats where the hand is 3 to 4 mm low.
- Stops (hold the measured pose): hand advance > 4 mm, left hand > 2 mm, a gripper reading change > 0.02 (all as before), and hand > 4 mm off its line (new). Early end: no advance of 0.15 mm in 0.6 s at full force.
- New report (both push modes, key `full_depth_mm`): event `insert_depth` and a printed line `depth at rest x.xx mm ... FULL / SHORT` (rest depth >= 7.2 mm). On the 11 logged insertions this rule gives FULL for the 3 "fully in" runs and SHORT for the other 8.
- Risk to know: a force source moves fast if the plug gives way. The plug has at most about 2 mm of travel left, and the advance stop is checked every 20 ms.

At the 2026-11-11 seat pose, 10 N along the axis needs joint 1 3.8 N m and joint 5 0.9 N m; 14 N needs 5.4 and 1.3 N m. The 15 mm position pushes used about 2.2 N m on joint 1 and 1.0 N m on joint 4: their PD estimate of 9 to 13 N "in" came with a down part and moments. So `force_n` 10 N is a more direct axial push than the old one, not the same load.

### Configs (new files; `bimanual_align5_k4_push15_seat6_fastmove.json` is the base and is unchanged)

| Config | Push | Gate | Needs approval |
|---|---|---|---|
| `bimanual_align8_depth_a.json` | force mode, 10 N, hold 1.5 s, no centring | depth >= 5, lateral <= 2.5 (as before) | the new push motion itself (go per run, as always) |
| `bimanual_align8_depth_b.json` | force mode, 14 N, centring up to 4.5 mm | depth >= 5, **lateral <= 4.5** | **yes: lateral gate 2.5 -> 4.5 mm with centring; force 10 -> 14 N** |
| `bimanual_align8_depth_k6d.json` | as `depth_a`, with the policy K6d (`runs/insert_K6d.pt`, section 5) | as `depth_a` | **yes: a new policy on the arm** (weights copied through the laptop like K4) |

Conventions check: `FAILED: none` for the three.

## 5. Sim change: mating force (default off)

- `InsertionCfg.mate_force_n` = (lo, hi) N, `mate_start_m` (2 mm): a force along the port axis between the plug and the PC (equal and opposite), zero up to 2 mm of depth, then linear to the per-episode peak at the seat depth. Only when the plug is in the opening. `None` (default) = off: K4's training and evals are unchanged, and no random draw is added.
- Dry run: env `SIM_MATE_FORCE_N=<N>` gives the same force in `SimIO` (off by default).
- It is a spring, not friction: a held 20 ms friction force is not stable in this sim. It asks for the right thing (push with at least the peak to reach the seat), but it pushes a released plug back out, so "seated after release" is not a valid number with it.

K4 and K6d with a mating force (`k2_matrix`, real-like plant; `results/real_align/depth/`). **K6d** = K4 fine-tuned for 10M steps with `mate_force_n` [0, 10] N and the force penalty from 15 N instead of 10 N (`runs/k6d_env.json`, run `P4-insert-k6d-s0`, checkpoint `ckpt_0010M.pt` = `runs/insert_K6d.pt`, sha256 `479c6b220b847d47`). K4's weights are unchanged.

| Mating force (n = 512 per cell) | K4 real_ho | K4 ring 3 mm | K6d real_ho | K6d ring 3 mm |
|---|---|---|---|---|
| 3 N | 99.4 % [98.3, 99.8] | | | |
| 6 N | 77.0 % [73.1, 80.4] | 74.0 % [70.1, 77.6] | **99.4 % [98.3, 99.8]** | **97.7 % [95.9, 98.7]** |
| 10 N | 0.0 % [0.0, 0.7] | | **90.0 % [87.1, 92.3]** | **89.3 % [86.3, 91.7]** |
| 14 N | 0.0 % [0.0, 0.7] | 0.0 % [0.0, 0.7] | not run | not run |

K4's failures with a mating force are grasp slips (21 to 23 % at 6 N, 92 % at 10 N: the reader moves more than 10 mm or 15 deg in the hand). K6d at 10 N: grasp slip 6 %, jam 2 to 3 %.

Acceptance cells without a mating force (n = 1024 per cell; K4 from `results/real_align/k2/k4s0_final.json`):

| Cell | K4 | K6d | K6d seat time p50 / p99 s |
|---|---|---|---|
| real_ho | 99.1 % | 98.7 % [97.8, 99.3] | 1.18 / 3.98 |
| ring 3 mm | 98.6 % | 98.8 % [98.0, 99.3] | 1.36 / 5.68 |
| ring 4 mm | 98.2 % | 97.7 % [96.5, 98.4] | 1.50 / 5.89 |
| ring 5 mm | 97.9 % | 95.6 % [94.2, 96.7] | 1.64 / 7.51 |
| roll 3 deg | 98.4 % | 97.8 % [96.7, 98.5] | 1.68 / 5.00 |
| roll 4 deg | 97.6 % | 96.5 % [95.2, 97.4] | 1.93 / 7.01 |
| pitch 4 deg | 97.5 % | 97.9 % [96.9, 98.7] | 1.72 / 6.18 |
| gap 8 mm | 96.5 % | 95.3 % [93.8, 96.4] | 1.36 / 4.76 |
| mix | 97.9 % | 97.8 % [96.7, 98.5] | 1.78 / 6.14 |

(The K6d `mix` cell ran with another seed than the K4 matrix; the others use the same seeds.)

- **In the sim, K4 does not stop short against a mating force. It fails:** with 6 N or more it keeps working the plug until the reader slips in the jaws. The sim grasp itself can hold 10 N: K6d seats 90 % at 10 N with the same grasp. So K4's way of pushing is the problem, and training with the force fixes it in the sim.
- **K6d costs little without the force:** within 1.2 points of K4 on 8 of 9 cells; ring 5 mm is 2.3 points lower (95.6 against 97.9 %).
- **Limits:** one seed, 10M steps (K4 had 30M), the mating force is a spring model with a guessed range, and the real force of this port is not measured. K6d is a candidate for a rig test, not a replacement for K4. The 12M-step end of the run (`runs/P4-insert-k6d-s0/last.pt`) is not evaluated.
- `k2_matrix.py` takes the policy's `env_json` as its base, so K6d evals without the force need `--env_extra '{"mate_force_n": null}'`.

## 6. Dry runs (SimIO, K4, 2026-11-13; they check the code path, not a rate)

| Config | Seed | Mating force | Seat | Final push | Hand off its line | Stops | True depth at the end |
|---|---|---|---|---|---|---|---|
| `depth_a` | 7041 | 0 | yes | skipped: depth 4.97 < 5.0 | | | 5.59 mm |
| `depth_a` | 7044 | 0 | yes | skipped: lateral 2.90 > 2.5 | | | 5.71 mm |
| `depth_a` | 7045 | 0 | yes | skipped: depth 4.89 < 5.0 | | | 5.68 mm |
| `depth_a` with the depth gate at 4.5 mm (scratch copy, sim only) | 7044 | 0 | yes | 10 N reached, advance 0.16 mm | 0.32 mm | none | seated after the release |
| same | 7047 | 0 | yes | 10 N reached, advance 0.52 mm | 0.57 mm | none | |
| same | 7048 | 0 | yes | 10 N reached, advance 0.42 mm | 0.23 mm | none | seated after the release |
| `depth_b` | 7044 | 0 | yes | 14 N reached, centred (0.61, -2.81) mm, advance 0.45 mm | 1.32 mm | none | 5.67 mm, seated after the release |
| `depth_b` | 7045 | 0 | yes | 14 N reached, centred (-0.35, 1.83) mm, advance 0.83 mm | 1.13 mm | none | 5.51 mm |
| `depth_b` | 7045 | 4 N spring | yes | 14 N reached, centred (-0.08, 1.20) mm, advance 3.88 mm | 0.98 mm | none | 2.43 mm after the drop (the spring pushes the plug back) |
| position push (base config) | 7044 | 4 N spring | yes | 15 mm target, advance 3.05 mm | | none | 2.35 mm after the drop |
| position push (base config) | 7045 | 4 N spring | yes | skipped: depth 4.93 < 5.0 | | | 2.71 mm |
| `depth_a` | 7044 | 4 N spring | align stop (sim) | | | | no insertion |

- The force push ran 6 times (3 with centring, 3 without; 3 more scratch seeds ended in the sim before a seat). It reached its force each time, kept the hand within 0.6 mm of its line without centring and within 1.3 mm while centring up to 2.8 mm, and no watch stopped it. Left hand 0.15 to 0.52 mm, gripper unchanged. On the rig, the position push moved the hand 5.5 to 6.3 mm off its line.
- K6d in the deploy path (`depth_k6d`): the runner loads it and checks its sha256; seeds 7044 and 7048 seated and the 10 N push ran (hand 0.20 and 0.21 mm off its line, true depth 5.71 and 5.59 mm).
- With the 6 N spring (`SIM_MATE_FORCE_N=6`) the dry run is not a valid depth test: the spring pushes the plug back out in the 0.2 s hold after the seat, before the push. It did show the advance stop: on K6d seeds 7047 and 7048 the push stopped itself at 3.99 and 4.10 mm of hand advance (limit 4 mm), at 6.6 and 7.4 N.
- The sim seat reads 4.9 to 6.7 mm in seat-2 units (the sim port mouth is at 0), so the 5.0 mm depth gate skips many sim pushes. On the rig the seat read 6.8 to 8.6 mm (n = 8), so the gate passes there.
- With the 4 N spring the force push advanced the hand 3.88 mm and the position push 3.05 mm (one run each: not a comparison).
- MuJoCo Warp is not bitwise deterministic: seed 7041 ended as `guard_bound` once and as a seat once today.

## 7. Runbook for the next rig session (depth)

Rules as in every session: bases clamped, `estop.py` open, nothing moves without the operator's go, one go per motion command. Same PC spot and reader spot as 2026-11-11. `.venv/bin/python`, `PYTHONPATH=.`.

**Step 0, by hand, no robot (5 minutes). These numbers decide everything else:**
1. Push the reader into port 2 by hand until it stops. Is a silver band still visible between the neck and the case? Measure it (calipers or feeler: about 0 to 1 mm). Does the reader hang or sit straight?
2. Pull it out 1, 2 and 3 mm and look: which of these is "hangs" and which is "about 3 mm short" by eye?
3. If a small scale or force gauge is at hand: the force to push the plug from 2 mm short to fully in, and the force to pull it out.

**Step 1, no motion:**
```bash
git pull   # branch real/graph-align
CA=deploy/config/bimanual_align8_depth_a.json; CB=deploy/config/bimanual_align8_depth_b.json
for c in $CA $CB; do .venv/bin/python -m deploy.checks.bimanual_conventions --config $c | tail -1; done      # FAILED: none, twice
.venv/bin/python -m deploy.run_bimanual --dry-run --stage approach_insert --seed 7044 --config $CA --align-mode full --name dry_depth_a   # prints "final push (force ..." and "depth at rest ..."
```

**Step 2, config A, 3 runs** (one go per run):
```bash
.venv/bin/python -m deploy.run_bimanual --stage full --config $CA --align-mode full --speed 0.75 --demo --name depth_a_1
```
Per run, report: the two printed lines (`final push (force ...)`, `depth at rest ... FULL/SHORT`), the operator's view (silver band visible? hangs?), and the last wrist frame before the release.
- Pass: the operator says fully in on 3 of 3, the hand stays within 2 mm of its line (`off line` in the printed line), no stop by a watch.
- Stop the block: a watch stops the push twice; the reader moves in the jaws (jaw change > 0.02); the PC moves; anything looks wrong to the operator.
- If the push is skipped by the lateral gate (hand more than 2.5 mm off the axis at the seat): note it; that case is config B.

**Step 3, config B, only with the operator's approval of its two changes** (lateral gate 2.5 -> 4.5 mm with centring; force 10 -> 14 N): 3 runs, same command with `$CB` and `--name depth_b_1`. Same report. Use B if A leaves the plug short or skips the push.

**Step 3b, K6d, only with the operator's approval of a new policy on the arm, and only after step 2:** copy `runs/insert_K6d.pt` (7.6 MB) from mason through the laptop as for K4 (`sha256sum runs/insert_K6d.pt | cut -c1-16` = `479c6b220b847d47`). Dry run with `deploy/config/bimanual_align8_depth_k6d.json` (prints the policy path and sha), then 3 runs, `--name depth_k6d_1`. Question: does K6d rest deeper than K4 before the push (K4: 5.65 to 7.00 mm)? Stop K6d at the first new failure mode (a stall on the case, a jaw change, the reader turning in the hand).

**Step 4:** the session notes get, per run: rest depth (printed), FULL/SHORT, operator verdict, push force reached, centring (mm). Send the run folders to mason. The server then sets `full_depth_mm` from more than 3 labelled runs.
