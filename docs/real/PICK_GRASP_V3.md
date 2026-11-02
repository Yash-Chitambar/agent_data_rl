# Pick grasp v3: level the closing axis, touch down, close slowly, retry

Status: code, config and dry runs ready. Nothing here has run on the robot. Touch-down is not validated on a slow real descent.

Why: [analysis v2](../../results/real/pick_20261101_analysis.md) (branch `sim/grasp-fidelity`). On the real runs:

1. The servo target rolls the closing axis -3.4..-5.6 deg (8 of 10 runs). Held closes: |roll| <= 2.87 deg. Empty: >= 3.78 deg. This is a post-hoc order on 10 closes with mixed configs.
2. The servo plan sits 3-13 mm below the height where the hand stops (this work, from the `servo_frame` logs). The target then presses.
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
| Close | Hold-down 3 mm. Target runs down at 0.2 /s, at most 0.12 below the reading, until the reading stalls. | `holddown_mm`, `gentle_close`, `gentle_rate`, `gentle_delta`, `stall_*`, `close_*` | `close_start`, `close_stall`, `close_fell_through` |
| Squeeze | Ramp to the same final target as before, dwell. | `squeeze_s`, `dwell_s` | `squeeze_start`, `squeeze_end` |
| Test lift | Lift 15 mm. Held = reading in 0.275-0.36. | `test_lift_mm`, `held_band` | `test_result` |
| Lift | Reading below the band for 5 ticks: lost, next attempt. | `lost_ticks` | `lost_in_lift` |
| Retry | Open, rise 40 mm, go to the hover, new wrist frames. At most 3 attempts. | `max_attempts`, `rise_mm`, `reopen_s` | `reopen`, `rise`, `retry_rehover`, `giveup` |
| Wrist gains | Optional: joints 4-6 stiffer from Hold to the test lift. | `wrist_kp`, `wrist_kd` (`null` = off), `wrist_ramp_s` | `wrist_gains` |

Per-tick data: `grasp.npz` (phase, attempt, D, target height, reading, grip target). Force and speed limits did not change. Body clearances on the commanded pose are back up: gripper 10, link_6 15, link_5 35 mm.

## Rig commands, in this order

Set up first: bases clamped, `scripts/real/estop.py` in a second terminal, `export PYTHONPATH=.`. Use `C=deploy/config/bimanual_pick3.json`.

**0. No motion.**

```bash
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C
.venv/bin/python -m deploy.checks.bimanual_timing --config $C        # p99 < 20 ms per stage
.venv/bin/python -m deploy.checks.bimanual_realio_mock               # "ok": true at the end
.venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 --config $C
```

**1. Touch-down in air (3 runs).** Jaws stay open. The hand descends slowly about 30 mm above the plan.

```bash
.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C --pick-test air --name air1
```

- Pass: `touchdown_fallback`, no `touchdown`. The `test_rise` line prints `d_max_mm`. It must stay below about 6.5 mm.
- A `touchdown` in air is a false positive. Stop. Raise `td_drop_mm`, or set `touchdown` to `false`.

**2. Touch-down on the table (3 runs).** Reader in place. Jaws stay open.

```bash
.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C --pick-test touchdown --name td1
```

- Pass: `touchdown` with `d_mm` near 8, `hold_settled` with |`roll_deg`| below 1. The reader does not move.
- Listen and watch: the tips touch gently, then the hand lifts a little.

**3. One close, then ten.**

```bash
.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C --name pick3_1
```

- Watch: both fingers level, the jaws close slowly, the hand does not rise.
- Record for each run: reading at `close_stall` (expect 0.30-0.345), `test_result`, attempts.
- Keep hands ready to take the reader: it can drop during the lift. Go to ten runs only after two clean holds.

**4. Full chain.**

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C --name full3_1
```

## If it fails

| What you see | What to change |
|---|---|
| `touchdown` too early or in air | `td_drop_mm` up (the free-motion peak in the logs was 6.69 mm), or `td_speed_m_s` up to at most 0.015 |
| Touch-down presses too hard or turns the hand | `td_drop_mm` down, but stay above the air-test peak |
| Touch-down is unreliable | `touchdown: false`. The pick then uses the old descent and holds at the measured pose. The roll fix stays on. |
| `td_blocked` on a good pose | Remove `td_tip_max_mm` (FK height has a run-to-run bias) |
| `roll_gate` often | Check `hold_settled.roll_deg`. Raise `roll_gate_deg` to at most 3. |
| Roll is 0 and the close is still empty | `td_lift_mm` to 0, `holddown_mm` to 4, `gentle_rate` down |
| Hand rises in the close | `holddown_mm` up (stiffness is only ~0.24 N/mm) |
| Jaws stall early in air | `gentle_delta` up |
| Jaws hit the reader on the way down | `gripper.right_approach_open_m` 0.024 -> 0.025 |
| Good hold reads as empty | `held_band`. Check the reading scale first: pick3 had a wrong scale, so restart the arm connection if an empty close does not read 0.03. |
| Hand shakes with the wrist gains on | `wrist_kp` and `wrist_kd` to `null` |

Stop at any time: Enter in the runner terminal, or `estop.py`.

## Results

Pending the sweep (`scripts/policy/grasp_v3_eval.py`, `results/real/grasp_v3/`).
