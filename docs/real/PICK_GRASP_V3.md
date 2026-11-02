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
| Hand shakes with the wrist gains on | `wrist_kp` and `wrist_kd` to `null` |

Stop at any time: Enter in the runner terminal, or `estop.py`.

## Results

Pending the sweep (`scripts/policy/grasp_v3_eval.py`, `results/real/grasp_v3/`).
