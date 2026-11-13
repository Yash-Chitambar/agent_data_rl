# Server tasks after the K4 milestone (2026-11-12)

For the server agent on mason. Branch `real/graph-align` (commit `09b1ed8` = milestone, this file after it).
Milestone: `docs/campaign/FINDINGS.md` top entry. Run details: `docs/real/SESSION_20261110_duchamp3.md`,
`docs/real/SESSION_20261111_duchamp3.md`. Real run logs on mason: `/home/yashchitambar/peg_test/data/real_runs/2026-11-11`
(complete) and `.../2026-11-10` (partial, still copying). Each run folder: `events.jsonl` (every stage event),
`ticks.npz` / `can_*.npz` (50 Hz joints and commands), `frames/` (ZED and wrist images), `live_demo.mp4`.
Best runs: `full_161705_demo_4` and `full_171905_fastmove_1` (operator: fully inserted).

## Rules

- No rig access from mason. Every change for the rig is a **new config file** (do not edit `deploy/config/bimanual.json`
  or existing configs) or code that is off by default, plus a sim dry run (`deploy.run_bimanual --dry-run`) and the
  conventions check (`FAILED: none`). The operator runs it on the rig.
- Do not change model weights in place; new checkpoints get new names.
- Report numbers from logs and sim only, with n. Results go in `docs/campaign/FINDINGS.md` (newest first).

## 0. Comb the real data first

Read every 2026-11-10/11 run (about 30 `full_*` runs, dry runs excluded) and write one table: stage times, pick
result, approach jump, align cycles / time / timeouts, hand-over error, K4 seat time and depth, push advance,
outcome, operator note (in the session docs). Then list what the data says, with numbers. Known so far:

| What | Data |
|---|---|
| Insertion | 10 of 10 seated with `bimanual_align5_k4_push15_seat6` when the plug reached the port; 10 of 15 starts end to end |
| Run time | 213.7 s (`--speed 0.5`) -> 164.3 s (0.75 + capture/home moves 0.4 rad/s); align is now the slowest stage |
| Align | 34-55 s, 14-22 cycles, from a gap of 17-19 mm to 6 mm |
| Wrist net vs vision | net "up" reads +0.7 to +4.4 mm above vision on 12 of 13 hand-overs; the 3.5 mm gate stopped 2 runs |
| Approach jump | over the 3 mm limit on 6 runs (a4p_1, kjrc_2, k4_3, demo_1, demo_2, fastmove_1: 3.6-5.0 mm; joint 2 error up to 33 mrad) |
| Pick | 1 empty close (`full_160801_demo_3`: hand settled 7 mm high); the rest held |

## 1. Push the plug all the way in

- Now: K4 seats at FK depth 6.8-8.6 mm, then `insert.final_push` (target 15 mm into the port at 4 mm/s x speed; stops if
  the right hand advances > 4 mm, the left > 2 mm, or a jaw reading changes) adds 2.0-2.7 mm. With a 5 mm push it added
  only 0.4-0.9 mm (about 2-3 N along the port axis).
- Operator: fully in on demo_4 (seat 8.38 + push 2.09 mm), fastmove_1 (8.60 + 2.04), k4_5 rerun "flush" (8.35 + 2.62);
  k4_1 about 3 mm short (8.02 + 0.38). Check whether total FK advance >= about 10.4 mm means fully in, on every run.
- The push was **skipped** twice by its lateral gate (seat lateral up -3.3 / -3.5 mm, gate 2.5): the reader hung slightly
  out on demo_1. The "up" there may be the same wrist-net bias (see 3).
- Questions: what force reaches full depth (USB-C mating force can be up to 20 N); can K4 itself push deeper (train with
  a deeper seat target or a final-depth reward); does the grip slip (k4_4 rerun: jaw reading closed by 0.020 during the
  push, the watch stopped it). Deliver: a push that reaches full depth on the logged cases, as a new config.

## 2. Faster camera measurements and alignment

Align step data (`align_step`, `align_measure` events), 2026-11-10/11 step-mode runs:
- **Steps take 75-85 % of the align time** (26-45 s of 34-55 s). **4-12 steps per run end at the 150-tick (3.0 s)
  timeout** with `reached: false`, often with `frac` (moved / asked) already 0.9-1.3. The reach test, not the arm, is
  slow. Check the criterion in `deploy/bimanual/align.py` (step completion), and the stiction on 1 mm steps.
- **Probe cycles:** the first 5-6 cycles move +/-2-3 mm across, up and gap and back (slope estimate, about 15-20 s).
  The measured slopes are the same on every run: across 0.94-1.10, up 0.76-0.98, gap 1.02-1.12 (n = 20). A fixed slope
  prior (with a check) could replace most of the probing.
- **Gap step 1 mm per cycle** from 18 to 6 mm: at least 12 cycles. Larger gap steps while far, or a closer approach stop
  (now 18 mm, `chain_args.stop`), cut cycles.
- **Each measurement takes about 0.8-1.1 s** (5 frames, wrist camera at about 24 frames/s): find where the time goes
  (settle wait, frame wait, net compute) and shorten it.
- The continuous servo (`align_mode servo`, 2026-11-10) aligned in 2.9-7.8 s but failed its sway check (tip speed p99
  24-25 mm/s, limit 15): an option if the sway is fixed in sim (lower gains near the port, filtering).
- Target: align under 15 s with the same hand-over accuracy (lateral <= 1 mm, gap 6 +/- 1 mm). Check in the sim dry
  run, and by replaying the logged real measurements where possible.

## 3. Policy robustness: placement and grasp

Every real run used one PC spot (port about (231, 394) mm, yaw 3-6 deg, right-base frame) and one reader strip
(x 121-142, y 438-459 mm, yaw 87-99 deg). The PC spot is outside the sim training box in y (nominal (220, 360) mm
+/- 10 / 15 mm, +/- 5 deg; insertion checked at +/- 3 cm / 10 deg) and still works, because each stage measures.
- Re-centre the sim layout on the real spot and widen it: PC +/- 2-3 cm, +/- 10-15 deg; reader across the strip and
  yaw 70-110 deg. Report where each stage fails (ZED estimate and gate, pick reach and grasp, hold, approach IK, align,
  K4) and retrain / adjust the weak ones. The operator will test PC +/- 1-2 cm and yaw 0-15 deg on the rig.
- Grasp: the reader must be picked well at other positions and yaws. Use the demo_3 empty close (hand 7 mm high at the
  close, fingers over the reader top) as a test case.
- Wrist net "up" bias (about +3 mm against vision, 12 of 13): fix it (retrain or calibrate the net on the real frames)
  or subtract it in the disagreement check, so the align gate stops only real disagreements.
- Approach jump (uncommanded move up to 5 mm at the align start, joint 2 error 27-33 mrad): find the cause in the
  ticks (wind-up at the 18 mm stop, `align_windup_at_entry`) and remove it.

## Order

0 (data review) first, then 1 and 2 (small changes, quick rig tests), then 3. Write each result in FINDINGS and a
short runbook for the next rig session (commands, new config names, stop criteria).
