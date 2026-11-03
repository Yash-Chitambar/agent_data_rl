# Next rig session, draft: align stage, then the insertion

Give this file to the local agent. Open it with: "Read `docs/real/NEXT_RIG_SESSION.md` and guide me through it."

Goal: measure the plug and port from the wrist camera at the real stop, then bring the plug to the hand-over gap in slow steps, then hand over to K-J. What the stage does: [ALIGN_STAGE.md](ALIGN_STAGE.md). Last session: [SESSION_20261102_duchamp3.md](SESSION_20261102_duchamp3.md).

This draft is on branch `real/graph-align`. Other agents add their parts in the section "Additions from other agents" below. Merge those branches before the session.

## Rules for the agent

1. **Do not run a command that moves an arm until you show the exact command and the operator says "go".** One command, one go. A "go" for one try does not cover the next try.
2. Before the first motion, the operator confirms: both arm bases are clamped, and `scripts/real/estop.py` is open in a second terminal. Check again after any base or PC move.
3. Commands with `--dry-run`, the checks, the log tools and step 1 (no arm connected) do not move the robot. You can run them without asking.
4. Do not change configs, gains, force limits, speed limits or clearances without the operator's approval. Do not edit a limit to get past a stop: report the stop. The `align` keys in `deploy/config/bimanual_align1.json` are also settings: ask first.
5. One thing at a time. Do not skip a step without the operator's approval.
6. Report only what the logs and the operator show. Sim percentages do not predict these runs.
7. Commit or push only when the operator asks. Never force push.

**Stop now and report** if: the arm or a body hits something, the plug touches the case during the align stage, the arm shakes or jumps by more than about 5 mm, a base moves, the PC moves, or a stop appears that this file does not explain.

## Setup (no motion)

```bash
cd ~/agent_data_rl
git status                       # commit or stash local changes first (ask the operator)
git fetch origin && git checkout real/graph-align && git pull --ff-only
export PYTHONPATH=. ; C=deploy/config/bimanual_align1.json
.venv/bin/python -m deploy.bimanual.bundle verify                     # "bundle OK"
.venv/bin/python -m deploy.checks.bimanual_conventions --config $C    # "FAILED: none"
.venv/bin/python -m deploy.checks.align_gates                         # "FAILED: none"
.venv/bin/python -m deploy.checks.bimanual_realio_mock                # the LAST "ok" line is true
.venv/bin/python -m deploy.checks.bimanual_timing --config $C         # p99 < 20 ms per stage, other jobs closed
.venv/bin/python -m deploy.run_bimanual --dry-run --stage approach_insert --seed 7000 --config $C   # completed True
```

**If timing fails with other jobs closed, stop and report.** The server cannot decide timing: only the rig can.

## Scene

As batch A on 2026-11-02: PC port about (0.23, 0.39) m, reader on its taped spot. Put the reader back on the tape before every try. Both gates must print True.

Film each run from table height, side on, so that the plug and port row are in view. Start the clip before the command.

## Step 1. Camera only (no arm connected, nothing can move)

```bash
.venv/bin/python scripts/real/align_camera_check.py --n 30 --out data/real_runs/$(date +%F)/camera_check
```

Hold the PC so that its port row is in the wrist camera's view, at about 10-40 mm in front of the lens side of the plug position. This checks the camera, the frame format and the measurement time. Report `ok` of `n` and `ms_p50`.

Pass: the camera opens and the tool prints a result. A low `ok` count here does not stop the session (a hand-held PC is not the real view). Measurement time above about 300 ms: report it.

## Step 2. Measure at the real stop: the align stage does not move

The chain runs as usual to the approach stop. Then the stage takes 10 measurements with the arm still, prints them, backs out 15 mm with closed jaws, and asks for the reader. No insertion.

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C --align-mode measure --name measure1 --notes "measure at the stop, no align motion"
```

The operator takes the reader by hand at the end (the prompt says when).

After the run, show:

```bash
.venv/bin/python scripts/real/align_log.py data/real_runs/$(date +%F)/full_*measure1
```

Check: `align: shell depth` printed a number (no number: the grasp is refused, and the stage stops, which is correct), frames measured (`ok`), spread between measurements, and the values against what the operator sees. Expected at the stop, from port vision on the last runs: gap about 8-9 mm, up about +5 to +9 mm, across -5 to 0 mm. **Before step 3: the measurements must be steady (spread below about 1 mm: the arm does not move, and port vision noise at rest was under 0.12 mm offline), and at least half of the cycles must have a measurement.** This is also the first time port vision runs live. Compare the printed time per measurement: about 82 ms offline. If not, stop and report. Do not run step 3. Ask the port vision agent.

Repeat once with a new pick (`--name measure2`). The values should agree within about 1-2 mm, unless the operator saw a different grasp.

## Step 3. Align without insertion

The stage moves in steps of at most 1 mm. Near the end it stops about 6 mm in front of the port (measured). Then it backs out. No insertion. Probes move the plug 2 mm in each axis and back: this is normal.

The operator: stand at the estop. Look from the side during the stage, without touching the rig. **If the plug touches the case: Enter at once.** Watch for jumps; slow creeping is normal, and a single joint can jump.

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C --align-mode align --name align1 --notes "align, no insertion"
```

After the run:

```bash
.venv/bin/python scripts/real/align_log.py data/real_runs/$(date +%F)/full_*align1
```

Check and tell the operator:

- Outcome and reason. A stop is a good result if the reason is right. Do not change a gate to pass it.
- `slope` per axis. It must be 0.6-1.6. On `scale gate`: the camera does not follow the hand. Stop and report.
- `moved fraction`: how far each step went against the asked step. Many below 0.25 or above 3 means that the creep controller does not fit the real friction. Report it with `lead`.
- Does the operator agree by eye with the last line? A photograph or a clip frame of the plug at the end of the stage, before it backs out, helps. Pause with `--pause-in-press` is not needed: the stage holds only for its measurements, so use the clip.
- Time of the stage. Its limit is 90 s, with the press off.

**Pass:** 3 of 3 tries end `OK`, and the operator sees the plug in front of port 2, lined up, not touching. Then step 4. Otherwise stop and report the logs.

## Step 4. Full run

One try, then look. Only if step 3 passed.

```bash
.venv/bin/python -m deploy.run_bimanual --stage full --speed 0.5 --config $C --name align_full1 --notes "align + K-J"
```

The operator judges the seat by eye. The new seat detector (`insert.seat2`) has only been checked on two failed and two seated real runs. Type the outcome from what you see. Do not treat `seated_detected` as success.

If there is no seat, the run lifts straight out with closed jaws and asks you to take the reader. It does not open the jaws (2026-11-02 A: the time limit opened them and the reader fell). The outcomes are `press_abort`, `guard_bound`, `timeout`.

After each try:

```bash
.venv/bin/python scripts/real/align_log.py data/real_runs/$(date +%F)/full_*align_full1
```

Go on with `align_full2`, and so on, one go per try, for about 5 tries. Then stop and report. On a "seated" from the runner with the plug not in: stop and report at once, as this is a false seat.

## Additions from other agents

Keep each item short: what, why, the command. Merge order matters: put your branch name here.

- **Port vision (`real/port-vision`):** merged at d1eb4d0. Its own wish list for the session is in [PORT_VISION.md](PORT_VISION.md), section "What the next rig session should collect". Items 3 (grasp spread) and 4 (still noise) come free from step 2 repeated; items 1, 2, 5, 6 need an operator decision.
- **Wrist nets (`policy/wrist-real`):** _(open)_ whether a net can be a second source (`align.secondary`), and its real slope.
- **Sim grasp (`sim/grasp-real`):** _(open)_ in-hand preset; any change to K-J or the in-hand net.
- **Re-collect with a light-contact anchor:** _(open, needs a decision)_. The 2026-11-02 anchor was a full push, and the reader pitched in the jaws: the labels' height is not known within about 4.5 mm. A new `collect` run with the anchor at light contact (plug in by 1-2 mm, no push, `anchor 1.5`) would fix the constant for the port vision and the nets. Cost: about 20 min of the rig, with the operator's hand on the PC. Command, if it is wanted (one go):

  ```bash
  .venv/bin/python -m deploy.run_bimanual --stage collect --speed 0.5 --config deploy/config/bimanual_pick3.json --batch A --collect-seed 2 --name collect2
  ```

## At the end

1. Write `docs/real/SESSION_$(date +%Y%m%d)_duchamp3.md`: one row per run (time, command, outcome, what the operator saw), then the align tables and any config change.
2. With the operator's approval, commit and push the notes. The run logs are gitignored.
3. The operator copies `data/real_runs/<date>/` and the videos to mason, into `~/peg_test/data/real_runs/<date>/`.
