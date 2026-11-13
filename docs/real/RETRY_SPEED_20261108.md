# Insertion retry, faster alignment, backlash preload (2026-11-08)

> **Rig status (2026-11-13): not run on the arm.** Block D needed the servo alignment, and the servo failed its tip-speed check on the rig (2026-11-10, 2 of 2). The milestone config (`bimanual_align5_k4_push15_seat6`) uses the step alignment, K4, no retry and no preload. The retry code is off unless a config sets `insert.retry`. Insertion stalls did not happen with the milestone config (10 of 10 seated), so the retry is not a priority now. The servo speed sweep is input for [SERVER_TASKS_20261112.md](SERVER_TASKS_20261112.md) task 2.

Operator request 2026-11-08: add a retry loop to the insertion, make the alignment and the insertion faster, and add the backlash preload (idea 8).
Everything here is sim and dry-run only. **Nothing here has run on the arm.**

## 1. Retry loop (`insert.retry`)

Code: `deploy/run_bimanual.py` `Runner.retry_insertion`; tip move `deploy/bimanual/align.py` `move_tip_closed`; `run_insertion` now returns
`advance_mm` (TCP advance into the port since the hand-over, believed FK displacement).

After an attempt that did not seat (outcome `timeout`, `guard_bound` or `press_abort`):

1. Hold the measured pose for 0.2 s (no push).
2. Back the plug straight out along the port axis, jaws closed, left press on. Distance = `advance_mm` + `entry_gap_mm` (15) − hand-over gap, at most 35 mm, at 10 mm/s.
3. Operator prompt `retry i: align again and insert` (Enter = go, s = no retry). Config `retry.confirm` (default true).
4. The align stage again from that pose (servo, press off, re-press, new vision prior, new in-hand estimate). All align gates apply again.
5. The insertion policy again, `insert_s` per attempt.

At most `max_retries` (2) more attempts. No retry when:
- the align stage is off or not in full mode;
- the press clock (`safety.hold_max_s`, 60 s, counted over the whole run) has less than `min_press_left_s` (20 s) left;
- the operator answers `s`.

If the align stage stops during a retry, it backs the plug off itself, and the run ends without a seat (reader to the operator's hand).
Logs: events `insert_retry`, `retry_back_off`, `insert_retry_skip`, `insert_retry_result`; `results.insert_attempts` in the run result.

Why a retry helps: the failures we see (sim and rig) are lateral misses on the case face. Each retry measures the port again with fresh frames
and a fresh hand-over. Failures that come from a fixed cause (a large encoder offset, a bad grasp) repeat; failures from vision noise and the
left-arm lead do not. So the gain is between none and 1 − (1 − p)^3.

Limits: unchanged. The retry adds no new limit and loosens none. It adds press time; the existing 60 s press clock stops the run if it is used up.

## 2. Speed

Real time budget, run 2026-11-06 `align_full9` (seated), robot time without prompts:

| Stage | Time |
|---|---|
| capture | 53 s |
| pick | 20 s |
| hold | 9 s |
| transport | 9 s |
| approach (12 net steps) | 20 s |
| **align (step stage)** | **39 s** |
| **insertion** | **2.3 s** (time limit 10 s) |
| release | 9 s |
| home | 43 s |

The alignment is the slowest motion stage. The insertion is fast when it seats; a stall used the full 10 s.

Changes:
- **Alignment: servo instead of steps** (`align.servo`, already in `bimanual_align6_servo*`). Sim: 4.4 s median against 16.7 s for the step stage.
- **Faster servo settings**: see the sweep below.
- **Insertion time limit 10 s → 6 s** per attempt (`insert.insert_s`). 6 s is the sim training episode. K4 sim seat time: p50 1.2-1.9 s, p99 3.7-7.3 s over the nine cells (n = 1024 each). A late seat lost to the 6 s limit gets a retry.
- **Scripted motions (`--speed`)**: the runbook runs `--speed 0.5`. It is a limit: changing it needs the operator's approval. Proposal in the runbook: 0.75 after two clean runs.

Servo speed sweep (AlignTask, chain Coulomb FF, done on vision, n = 256 each, same starts; `scripts/eval/align_speed_sweep.sh`,
`results/real_align/align_speed/`):

| gain | vmax mm/s | other | aligned | 95 % | time p50 / p90 s | path ratio p50 / p90 |
|---|---|---|---|---|---|---|
| 1.5 | 8 | (today's config) | 99.6 % | [97.8, 99.9] | 4.3 / 7.7 | 1.40 / 2.00 |
| **1.5** | **12** | | **100.0 %** | **[98.5, 100]** | **4.1 / 6.9** | **1.35 / 1.93** |
| 2.5 | 8 | | 97.7 % | [95.0, 98.9] | 4.2 / 10.2 | 1.57 / 2.99 |
| 2.5 | 12 | | 96.5 % | [93.5, 98.1] | 4.0 / 9.8 | 1.54 / 3.27 |
| 2.5 | 12 | wmax 8 deg/s | 98.4 % | [96.1, 99.4] | 4.1 / 7.8 | 1.56 / 2.51 |
| 2.5 | 12 | wmax 8, angle gain 2 | 62.5 % | [56.4, 68.2] | 10.2 / 23.1 | 3.90 / 10.6 |
| 2.5 | 16 | | 99.2 % | [97.2, 99.8] | 4.1 / 9.5 | 1.57 / 2.91 |
| 3.5 | 12 | | 86.7 % | [82.0, 90.3] | 4.5 / 16.5 | 1.89 / 7.16 |

Learnings:
- A higher gain does not make the servo faster. With 10 Hz vision and about 0.1 s latency, it hunts: longer paths, longer tails, lower rates.
- MuJoCo Warp is not bitwise deterministic: the same setting (gain 1.5, vmax 8) gave p90 9.4 s in `align_ctrl/servo_ff` and 7.7 s here. Differences in p90 under about 2 s are noise at n = 256.
- A higher speed cap at gain 1.5 cuts the slow tail a little (p90 7.7 → 6.9 s; within the noise above) and keeps the path straight. The median is set by the settle and the confirm frames, not by the motion.
- AlignTask has no left press. In the full-chain dry runs (with the press), vmax 12 made the alignment under the press hunt: it hit its 9 s time limit in 3 of 6 runs (seeds 7300, 7303, 7305). With vmax 8, seed 7300 aligned (and seated); see the preload dry runs below for the other seeds.
- **Kept: gain 1.5, vmax 8 mm/s** (today's servo). The servo's gain is not where the time goes. The speed gain on the rig comes from the servo instead of the step stage, and from the 6 s insertion limit with retries.
- Real alignment time expected: about 5-10 s instead of 39 s (the step stage in `align_full9`). With the retry and the 6 s limit, a failed insertion costs about 6 s + 3 s back-off + the alignment, not 10 s and the end of the run.

## 3. Backlash preload (`align.servo.preload_mm`)

The encoders are on the motor side, so gear lash sits between the encoder and the hand. The wrist camera is after the lash, so the servo sees the
true plug position. The lash matters when control changes hands: at the hand-over the feed-forward turns off and the policy starts to push.
If the last servo motion went away from the port, the gears take up their lash on the first push, and the believed FK is off by the lash.

The preload: the servo aligns to a gap 1 mm larger (`align.target_mm` gap 7 mm), then the hand moves 1 mm straight toward the port (sin speed
profile, mean 2 mm/s), then settles. The last motion is then in the insertion direction. The fused estimate moves with the believed FK, so the
policy's vision prior keeps the 6 mm gap. Event `align_preload` (tip motion, estimate, sim truth).

Lateral lash is not preloaded: the insertion does not push in one fixed lateral direction.

Full-chain dry runs (SimIO, K4, 2026-11-08; n is small: they check the code path, not a rate):

| Config | Seeds | Aligned (incl. under the press) | Seated (FK detector) | Truly seated before release | After release |
|---|---|---|---|---|---|
| `align7_retry`, 4 mm test shift | 7200, 7201 | 2/2, retry 0/1 | 1/2 (the 7201 seat was false, see the runbook) | 0/2 | 0/2 |
| preload, vmax 12 | 7300-7305 | 3/6 | 3/3 aligned | 3/3 | 1/3 |
| preload, vmax 8 | 7300 (`--align-set`), 7303, 7305, 7306 | 4/4 | 4/4 | 4/4 | 2/4 |
| `align6_servo_k4` (reference) | 7300 | 1/1 | 1/1 | 1/1 | 0/1 |

- The preload moved the tip 0.61-0.93 mm toward the port (commanded 1.0; PD lag under the press). The hand-over gap by sim truth: 5.0-6.4 mm.
- **Open, not new: the dry-run release pulls a seated reader out** in 5 of 9 seated runs, the reference config included. It is in the release (`retreat_right`, jaw opening), not in the retry or the preload. On the rig, 11-06 `align_full9` stayed seated after its release. Watch the release on Monday; the operator confirms the seat after the release, not before.

## 4. Config

`deploy/config/bimanual_align7_retry.json` = `bimanual_align6_servo_k4` + `insert.retry` + `insert_s` 6 s.
`deploy/config/bimanual_align7_preload.json` = `bimanual_align7_retry` + preload (servo speed unchanged). See NEXT_RIG_20261109.md block D.
