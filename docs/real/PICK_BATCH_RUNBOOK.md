# Pick batch runbook, for the agent that guides the operator on duchamp3

Give this file to the local agent. Open it with: "Read `docs/real/PICK_BATCH_RUNBOOK.md` and guide me through it."

Goal: find which grasp change fixes the real reader pick, then test it in the full chain. Today the pick holds 2 of 8 closes. Details and config knobs: [PICK_GRASP_V3.md](PICK_GRASP_V3.md). Last session: [SESSION_20261101_duchamp3.md](SESSION_20261101_duchamp3.md).

## Rules for the agent

1. **Do not run a command that moves an arm until you show the exact command and the operator says "go".** One command, one go. A "go" for one try does not cover the next try.
2. Before the first motion, the operator confirms: both arm bases are clamped, and `scripts/real/estop.py` is open in a second terminal. Check again after any base or PC move.
3. Commands with `--dry-run`, the checks and the log tools do not move the robot. You can run them without asking.
4. Do not change configs, gains, force limits, speed limits or clearances without the operator's approval. Do not edit a limit to get past a stop: report the stop.
5. Change one thing at a time. Do not skip the order below without the operator's approval.
6. Report only what the logs and the operator show. Sim percentages do not predict these batches.
7. Commit or push only when the operator asks. Never force push.

**Stop now and report** if: the arm or a body hits something, a finger comes down on top of the reader, the arm shakes, a base moves, or a stop appears that this file does not explain.

## Setup (no motion)

```bash
cd ~/agent_data_rl
git status                       # commit or stash local changes first (ask the operator)
git fetch origin && git checkout real/bimanual-deploy && git pull --ff-only
export PYTHONPATH=. ; C=deploy/config/bimanual_pick3.json
.venv/bin/python -m deploy.bimanual.bundle verify                     # "bundle OK"
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C    # "FAILED: none"
.venv/bin/python -m deploy.checks.bimanual_realio_mock                # the LAST "ok" line is true (capture "ok": false is normal on mock frames)
.venv/bin/python -m deploy.checks.bimanual_timing --config $C         # p99 < 20 ms per stage, other jobs closed
for b in A B C D; do .venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 --config $C --batch $b; done
.venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 --config $C --batch D --pick-test air
.venv/bin/python -m deploy.run_bimanual --dry-run --stage pick --seed 7000 --config $C --batch D --pick-test touchdown
```

Each dry run must end with `completed True`. **If timing fails with other jobs closed, stop and report.** The new pick adds about 5 ms; the server cannot decide this, only the rig can.

## Scene

- PC and reader where they were on 2026-11-01: PC port about (0.23, 0.39) m, reader about (0.13, 0.45) m, yaw about 90 deg (right base frame). The run prints both estimates.
- Tape the reader's spot. Put it back on the tape before every try in a batch. If a failed try moved the reader, put it back and say so in the notes.
- Both gates must print True. A rejected gate means stop and check the scene: do not continue past it.
- Set up a table-height camera (a phone is fine) looking at the fingers and the reader. Start a clip for each try, or each batch, and say the batch and try number at the start.

## One try

Replace `n` with the try number and `X` with the batch.

```bash
.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C --batch X --notes "X try n"
```

During the try, the operator watches: both fingers level at the close, slow jaws (B, C, D), no hand rise, no reader rotation. Keep hands ready: the reader can drop during the lift.

The runner asks for the outcome at the end. Its words are for insertion; for a pick, type:

| Key | Meaning for a pick |
|---|---|
| `s` | held and lifted |
| `p` | caught, then lost during the lift |
| `f` | closed empty |
| `a` | aborted |

Then type what the operator saw (finger over the reader, reader turned, hand rose). If the pick fails, the run stops with "pick failed", the jaws open and the hand up: answer `h` to go home.

## The order

Run about 10 tries per batch, with the reader in the same spot. After each batch:

```bash
.venv/bin/python scripts/real/pick_batch_table.py data/real_runs/$(date +%F) --batch X
```

Show the operator the table. `lost_in_lift` counts as a failure. **Pass = about 8 or more of 10 held.** Ten tries show a big jump; they cannot separate 80 % from 90 %.

| Step | Batch | What changes | Then |
|---|---|---|---|
| 1 | `A` | Level roll only | Pass: go to step 7 with W = A. Otherwise step 2. |
| 2 | `B` | A + slow close | Pass: step 7 with W = B. Otherwise step 3. |
| 3 | `C` | B + hold-down (8 mm) | Pass: step 7 with W = C. Otherwise step 4. |
| 4 | `D --pick-test air` (3 tries) | Touch-down in air, jaws open, about 30 mm above the plan | Pass: `touchdown_fallback`, no `touchdown`, and `d_max_mm` in the `test_rise` line below about 6.5. **A `touchdown` in air is a false positive: do not run D.** Tell the operator and go to "If nothing passes". |
| 5 | `D --pick-test touchdown` (3 tries) | Touch-down on the table, lift, no grasp | Pass: `touchdown` with `d_mm` near 8, `hold_settled` with \|`roll_deg`\| below 1, reader not moved. Otherwise do not run D. |
| 6 | `D` | C with touch-down | Pass: step 7 with W = D. Otherwise "If nothing passes". |

Check after batch A, before going on: the table's roll at the close should be small (about 1 deg or less). If it is still near -3 to -5 deg, the roll fix is not working: stop and report. If roll is small and the closes are still empty, roll was not the whole cause: tell the operator, and go on to B.

**Step 7, the winner W:**

```bash
# longer run, about 20 tries, with retries on
.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C --batch W --grasp-set max_attempts=3 --notes "W long n"
# only if it holds up: opening comparison, about 5-10 tries each at 25 and 26 mm (24 mm is the batch result)
.venv/bin/python -m deploy.run_bimanual --stage pick --speed 0.5 --config $C --batch W --open-mm 25 --notes "W open25 n"
# then the full chain
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C --batch W --name full_W
```

For `full`: the hand-over after the new grasp is the main risk (a realistic sim seated only 1 of 3). The operator judges the seat by eye: the seated detector gave false positives on 2026-11-01, so type the outcome from what they see. Do not treat "seated_detected" as success. Watch the retreat: the jaws may drag the reader out.

## If nothing passes

Do not start tuning on your own. Show the tables for every batch. With the operator's approval, you can try one knob at a time from the "If it fails" table in [PICK_GRASP_V3.md](PICK_GRASP_V3.md). Fingertip pads and a reader cradle are last resorts: only the operator decides them.

## At the end

1. Run `pick_batch_table.py data/real_runs/$(date +%F)` for all batches.
2. Write `docs/real/SESSION_$(date +%Y%m%d)_duchamp3.md`: one row per run (time, command, outcome, what the operator saw), then the batch tables and any config change.
3. With the operator's approval, commit and push the notes. The run logs are gitignored.
4. The operator copies `data/real_runs/<date>/` and the videos to mason, into `~/peg_test/data/real_runs/<date>/`, so the server side can fit the hand rise and measure the servo nets' roll error.
