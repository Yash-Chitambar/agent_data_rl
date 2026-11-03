# Pick grasp v3: batched real test of the grasp changes

Status: code, configs and dry runs are ready. Nothing here has run on the robot. **The real batches decide the grasp mechanics. Every percentage below comes from the sim.**

Why: [analysis v2](../../results/real/pick_20261101_analysis.md) (branch `sim/grasp-fidelity`). On the real runs:

1. The servo target rolls the closing axis -3.4..-5.6 deg (8 of 10 runs). Held closes: |roll| <= 2.87 deg. Empty: >= 3.78 deg. This is a post-hoc order on 10 closes with mixed configs.
2. The servo plan sits 3-13 mm below the height where the hand stops (`servo_frame` logs). The target then presses.
3. Step closes jump 3.1-6.1 mm in 60 ms when they ride over. Slow ramps caught 2 of 2 (n small).
4. pick11 stalled at 0.301 and lost the reader 2.26 s after the lift start.

## What the pick does

Code: `blind_insertion/pipeline/grasp_v3.py`, hooks in `deploy/bimanual/chain.py`. Same code on the real and the sim io. Config: `deploy/config/bimanual_pick3.json`, block `pick.grasp_v3`.

| Step | What happens | Config keys | Log event (`events.jsonl`, kind `grasp_v3`) |
|---|---|---|---|
| Level | Each servo rotation keeps tilt and yaw. Its roll goes to 0. | `level_roll`, `level_roll_max_deg` | `roll_fix` |
| Touch-down | At the last wrist frame the hand settles, then descends at 10 mm/s. The integral is frozen. Friction feed-forward is on. | `touchdown`, `td_speed_m_s`, `td_max_below_mm`, `td_friction_ff` | `td_start` |
| Detect | D = target fall since an anchor. The anchor resets when the hand follows. Contact at D >= 8 mm. | `td_drop_mm`, `td_follow_frac`, `td_eps_mm`, `td_step_mm_s` | `touchdown`, `touchdown_fallback` |
| On the reader? | Off by default: FK height has a run-to-run bias. Set a limit to skip the close when fingertip FK at the trigger is above it. `touchdown` logs `tip_fk_mm`. | `td_tip_max_mm` | `td_blocked` |
| Hold | Measured height + 0.5 mm, planned xy and orientation, bounded integral. | `td_lift_mm`, `hold_*` | `hold_settled` |
| Roll gate | Measured |roll| above 2.5 deg: no close, next attempt. | `roll_gate_deg` | `roll_gate` |
| Close | Hold-down 8 mm below the hold, in place before the jaws start. Target runs down at 0.2 /s, at most 0.12 below the reading, until the reading stalls. | `holddown_mm`, `holddown_speed_m_s`, `holddown_wait_s`, `gentle_close`, `gentle_rate`, `gentle_delta`, `stall_*`, `close_*` | `close_start`, `close_stall`, `close_fell_through` |
| Squeeze | Ramp to the same final target as before, dwell. | `squeeze_s`, `dwell_s` | `squeeze_start`, `squeeze_end` |
| Test lift | Lift 15 mm. Held = reading in 0.275-0.36. | `test_lift_mm`, `held_band` | `test_result` |
| Lift | Reading below the band for 5 ticks: lost, next attempt. | `lost_ticks` | `lost_in_lift` |
| Retry | Open, rise 40 mm, go to the hover, new wrist frames. At most 3 attempts. The last attempt closes from the levelled press. | `max_attempts`, `press_from_attempt`, `rise_mm`, `reopen_s` | `reopen`, `rise`, `retry`, `giveup` |
| Wrist gains | Optional: joints 4-6 stiffer from Hold to the test lift. | `wrist_kp`, `wrist_kd` (`null` = off), `wrist_ramp_s` | `wrist_gains` |

Per-tick data: `grasp.npz` (phase, attempt, D, target height, reading, grip target). Each run also logs one `pick_outcome` event: held, lost_in_lift, closed_empty (or roll_gate, on_reader), with batch and config. Force and speed limits did not change. Body clearances on the commanded pose: gripper 10, link_6 15, link_5 35 mm in the default and batch D (no press); batches A, B, C keep the pick2 values 3 / 15 / 30 mm, because they still press in the descent.

## Rules for the whole session

- Nothing moves without the operator's go. Arm bases stay clamped. `scripts/real/estop.py` stays open in a second terminal.
- About 10 tries per batch. Keep the reader in the same spot within a batch. Change one thing at a time.
- Stop adding changes when one clearly works: about 8 or more of 10 (today: 2 of 8). Ten tries can show a big jump. They cannot separate 80 % from 90 %. So the winner gets a longer run, then `full`.
- Compare openings only after a winner exists. Only the robot can choose the opening: the sim is not sensitive to it.
- Film each batch with a table-height camera (a phone is fine). The wrist camera cannot see the pads. Say the batch and try number at the start of each clip, or note which try each clip belongs to.
- Put the batch and try in `--notes`. At the end of each run the runner asks for the outcome: type what you saw.

## Step 0: no motion, on the rig

```bash
export PYTHONPATH=. ; C=deploy/config/bimanual_pick3.json
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C
.venv/bin/python -m deploy.checks.bimanual_timing --config $C        # p99 < 20 ms per stage, with nothing else running
.venv/bin/python -m deploy.checks.bimanual_realio_mock               # last line "ok": true
for b in A B C D; do .venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 --config $C --batch $b; done
.venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 --config $C --batch D --pick-test air
.venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 --config $C --batch D --pick-test touchdown
```

Each dry run must end with `completed True`.

## The batches

Run the same command about 10 times. Replace `n` with the try number.

| Batch | Changes | One command |
|---|---|---|
| A | Level roll only (step close from the plan's press) | `.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C --batch A --notes "A try n"` |
| B | A + slow close | `... --batch B --notes "B try n"` |
| C | B + hold-down (hold at the measured pose - 8 mm) | `... --batch C --notes "C try n"` |
| D air | Touch-down, jaws open, in air, about 30 mm above the plan (3 tries) | `... --batch D --pick-test air --notes "D air n"` |
| D no close | Touch-down on the table, lift, no grasp (3 tries) | `... --batch D --pick-test touchdown --notes "D td n"` |
| D | C with touch-down | `... --batch D --notes "D try n"` |

`...` is `.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C`. Each batch has one attempt per try, so each try is one count. A failed try stops the run with "pick failed": the jaws are open and the hand is up. Answer `h` to go home.

Pass for D air: `touchdown_fallback` and no `touchdown`; `d_max_mm` in the `test_rise` line stays below about 6.5. A `touchdown` in air is a false positive: stop and skip batch D, or raise `td_drop_mm`. The detector margin from the logs is only 1.31 mm and it is not validated on a slow descent.

Pass for D no close: `touchdown` with `d_mm` near 8, `hold_settled` with |`roll_deg`| below 1, and the reader does not move.

**After each batch:**

```bash
.venv/bin/python scripts/real/pick_batch_table.py data/real_runs/$(date +%F) --batch A
```

The table gives per try: outcome, roll at the close, rise from close to stall, grip readings. It ends with the held count and its CI. `lost_in_lift` counts as a failure.

**Then, with the winner W:**

```bash
... --batch W --grasp-set max_attempts=3 --notes "W long n"                    # longer run with retries, about 20 tries
... --batch W --open-mm 25 --notes "W open25 n"                                # opening comparison: 24, 25, 26 mm
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C --batch W --name full_W
```

Every change also has its own switch, for example `--grasp-set level_roll=false`, `gentle_close=false`, `holddown_mm=0`, `touchdown=false`. The config without `--batch` is the sim's best stack (level, touch-down, slow close, hold-down, 3 attempts). It is not the rig's first choice: the batches come first.

**Watch for:** both fingers level at the close; the jaws close slowly; the hand does not rise; the reader does not turn. Keep hands ready to take the reader: it can drop during the lift. Reading at the stall: expect 0.30-0.345. An empty close should read about 0.03: if it does not, the reading scale is wrong (pick3). Restart the arm connection.

## If it fails

| What you see | What to change |
|---|---|
| `touchdown` too early or in air | `td_drop_mm` up (the free-motion peak in the logs was 6.69 mm), or `td_speed_m_s` up to at most 0.015 |
| Touch-down presses too hard or turns the hand | `td_drop_mm` down, but stay above the air-test peak |
| Touch-down is unreliable | Stay with batch C, or `--grasp-set touchdown=false`. The roll fix stays on. |
| `td_blocked` on a good pose | Remove `td_tip_max_mm` (FK height has a run-to-run bias) |
| `roll_gate` often | Check `hold_settled.roll_deg`. Raise `roll_gate_deg` to at most 3. |
| SafetyStop on a body clearance in A, B, C or in attempt 3 of the default | The commanded press: `--grasp-set press_from_attempt=null`, or read the limit in the session notes before changing it |
| Roll is near 0 and the close is still empty | `gentle_rate` down, `holddown_mm` up, then `td_lift_mm` to 0. Film it: the rise tells which |
| Hand rises in the close | `holddown_mm` up (stiffness is only ~0.24 N/mm) |
| Jaws stall early in air | `gentle_delta` up |
| Jaws hit the reader on the way down | `gripper.right_approach_open_m` 0.024 -> 0.025 |
| Good hold reads as empty | `held_band`. Check the reading scale first: pick3 had a wrong scale, so restart the arm connection if an empty close does not read 0.03. |
| Hand shakes with the wrist gains on | Remove the two `--grasp-set` wrist overrides |

Stop at any time: Enter in the runner terminal, or `estop.py`.

## Results (sim only)

Script: `scripts/policy/grasp_v3_eval.py`, data in `results/real/grasp_v3/final/`. Physics: preset `grasp_fidelity` v2 (`docs/real/GRASP_FIDELITY.md`), first version. n = 512, seed 1000, 24 mm. Held = lifted > 60 mm. CI: Wilson 95 %. "First" = attempt 1. Roll = true roll at the close: p50 [p5, p95].

| Config | Servo | Injected roll | Held (CI %) | First | Roll (deg) |
|---|---|---|---|---|---|
| Baseline | oracle | real draws | 137 (23-31) | 137 | -5.04 [-6.8, 0.5] |
| Relief v3 | oracle | real draws | 229 (40-49) | 229 | -5.05 [-6.9, 0.5] |
| Touch-down only | oracle | real draws | 90 (15-21) | 90 | -4.43 [-5.9, 1.4] |
| Slow close + hold-down only | oracle | real draws | 173 (30-38) | 173 | -5.06 [-6.9, 0.5] |
| Retry only | oracle | real draws | 103 (17-24) | 91 | -5.07 [-6.8, 0.5] |
| **Level roll only (A)** | oracle | real draws | **484 (92-96)** | 484 | -0.66 [-1.3, -0.2] |
| Level + slow close + 3 mm hold-down | oracle | real draws | 496 (95-98) | 496 | -0.66 [-1.3, -0.2] |
| All, 3 attempts | oracle | real draws | 503 (97-99) | 502 | 0.02 [-0.3, 0.3] |
| All + wrist gains | oracle | real draws | 504 (97-99) | 502 | 0.05 [-0.3, 0.3] |
| All, 3 attempts | oracle | -3 / -4.5 / -5.6 deg | 502 / 506 / 502 | 500 / 503 / 501 | 0.04 [-0.2, 0.3] |
| Baseline | oracle | -4.5 deg | 7 (1-3) | 7 | -4.90 [-5.4, -4.5] |
| Baseline | real nets | none | 424 (79-86) | 424 | 0.00 [-1.7, 1.7] |
| Level roll only | real nets | none | 381 (70-78) | 381 | -0.25 [-1.3, 0.0] |
| Level, +-1.5 deg clamp | real nets | none | 390 (72-80) | 390 | 0.00 [-1.6, 1.2] |
| Level + tilt back to plan | real nets | none | 354 (65-73) | 354 | -0.40 [-0.9, 0.0] |
| All, 3 attempts | real nets | none | 336 (61-70) | 332 | -0.08 [-1.3, 0.2] |
| All, 3 attempts | real nets | real draws | 330 (60-68) | 329 | -0.06 [-1.2, 0.3] |

Reference, other harness (`GRASP_FIDELITY.md`): v2 nets + injection 99 (16-23); nets, no injection 431 (81-87).

All `All` rows also inject a plan 3-13 mm too low. Batches B, C and D as configured (one attempt, 8 mm hold-down) have no n = 512 row: they have dry runs only. Time to grasp: p50 about 17-18 s, first attempt, in every row. Largest PD press estimate near the table up to the end of the squeeze, p50 / p99, sim projection (use only to compare rows): baseline 24 / 45 N, level only 47 / 69 N, all 11 / 19 N.

**What these results show and do not show**

- Roll: v2 can show it, and it is the largest effect. With the real rolls the fix takes oracle held from 137 to 484. The real run shows this only as a post-hoc order on 10 closes.
- With unbiased nets, levelling costs about 8 points (424 to 381). The real nets are biased (real target -3.28 +- 1.92 deg against sim +1.74 +- 0.25). The robot decides which case applies: batch A.
- With the sim nets the full stack reaches 65 %, below the 95 % target. The main losses are net outliers on the new body: position across the jaws > 10 mm or tilt > 60 deg. The nets trained on the 41.3 mm body. The cause of the shortfall is not separated. Do not read the "All" nets rows as a real prediction.
- Hand rise: v2 does not reproduce it (median ~0 mm; real 1.2-2.8 held, 4-7.8 empty), nor the 3-6 mm ride-over jump. So **every gain from hold-down or slow close above proves nothing about rise**. That includes the 8 mm hold-down. Only batches B and C can answer this.
- Opening: v2 is not sensitive to 24 / 26 mm and its stall reads 0.32 (real 0.30-0.305). Only the robot can choose the opening.
- Wrist gains: no clear gain in the sim (503 against 504). The press error comes from joints 2-4 more than 5-6. Run-time gain changes have not run on the arm. So they are off by default. Switch: `--grasp-set 'wrist_kp=[40,30,30]' --grasp-set 'wrist_kd=[1.5,2,2]'`.
- Touch-down: the detector is not validated on a slow real descent. It fired at tips 3.7-4.2 mm on 6 of 10 real presses in replay, and never in free motion on 7 left-arm descents at 10 mm/s (`results/real/touch_replay.json`). The margin is 1.31 mm. The real free-motion peak comes from D air.

**What happens to the servo's rotation correction:** roll goes to 0. Yaw, tilt and position stay. The reader lies flat, so a level axis is known. Putting the tilt back too made results worse (354 against 381) and is off (`tilt_fix`). A +-1.5 deg clamp is worse on real rolls (oracle 412, 77-84 %, against 484). Centring (oracle, real draws, |y| across the jaws p50 / p95): baseline 1.28 / 2.8 mm, level only 1.15 / 2.8 mm, all 0.35 / 1.4 mm. With unbiased nets levelling leaves it unchanged (0.53 / 8.2 against 0.51 / 8.4 mm).

**Dry runs** (`bimanual_pick3.json`, baseline physics unless stated). Pick seeds 7000-7009: 10 of 10 complete, seed 7007 on attempt 3 (levelled press). Full seeds 7000-7012: 13 of 13 complete, 12 truly seated after the release; seed 7001 stopped early at 5.5 mm and the old config seated it. Batches A, B, C, D, D air, D no close, C at 26 mm: seed 7000 complete. v2: pick seeds 7100-7107 8 of 8 (old configs 3 of 8); full seeds 7000-7002: 1 of 3 truly seated (one early stop, one insertion timeout). In-hand spread grows on v2 and the approach nets saw the old body, so check the hand-over on the first `full`.

**Checks:** conventions and RealIO mock pass. Timing does not pass on this shared host, with either config: pick p99 28.7 ms (old config 23.9), transport 36 (29), approach 53 (48). The new pick adds about 5 ms at p99. The rig measured p99 < 12 ms on 2026-11-01. Run step 0 there: this is the deciding number.

**Open points for the rig:** SafeArm checks the commanded pose. `bimanual_pick3.json` without `--batch` keeps gripper 10 mm, so its attempt 3 (press) can stop on a clearance (pick3 stopped at 9 mm). `update_kp_kd` at run time and the gentle close force on the real gripper are untested: the i2rt source is not on this host.

## Real test result and sim correction (operator, 2026-11-02, duchamp3)

Batch A (level roll only): 10 of 10 held. `full` with batch A: 2 of 2 held through transport and approach, 0 of 2 inserted (the wrist servo stops too high and varies between runs). Details: [SESSION_20261102_duchamp3.md](SESSION_20261102_duchamp3.md).

**Sim correction, in-hand pose of the real grasp.** Operator report: the real grasp is consistent and good, but it does not match the sim. In the sim the reader lines up exactly with the bottom of the gripper. On the robot the butt of the reader angles slightly out of the gripper (less than 10 deg) and the reader still looks straight. The operator has images and video of this for the sim to reproduce. Correct the sim's in-hand pose and retrain the approach and in-hand nets on it.

**Operator request, port edge detection.** Add an alignment check from the wrist camera alongside the GaP policy: find the USB-C port's edges in the image, so the exact port position is known before insertion. On 2026-11-02 the wrist net reported about the same alignment while the hand moved 5-6 mm, so a second measure that does not depend on the net could catch this.
