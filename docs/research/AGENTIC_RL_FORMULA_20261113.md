# Agentic RL for fine-grained manipulation: the formula

2026-11-13, updated with the rig session of 2026-11-13 and the server work of 2026-11-14 and 2026-11-15. Review page (it shows the 2026-11-13 text, not this update): https://claude.ai/artifact/2H828Up4rhPgvAQGxJ2Vny

## Short answer

The formula: an agent builds the task as a graph of scripted stages, puts RL only in the node where contact decides the outcome below the sensing resolution, and uses every real run to correct the simulator, not to teach the policy. On our rig this gave 12 of 12 seated USB-C insertions when the plug reached the port (10 of 10 on 2026-11-10/11, 2 of 2 on 2026-11-13), with zero real demonstrations and no force sensor. All 12 are at one placement: one reader spot and two PC spots about 4 mm apart. The alignment now takes 10.1 to 13.9 s on the rig (n = 4); it took about 60 s. Depth is not solved: both runs of 2026-11-13 left a silver band of about 2 mm, and no signal from the arm verifies the depth. The server work of 2026-11-15 (nothing on the arm) shows that more axial force is unlikely to help, and that the sim result rests on a low port friction.

**The formula in eight lines.** It had six lines on 2026-11-13. The edits of 2026-11-15 and their evidence are in "Formula edits decided on 2026-11-15".

1. *Graph first.* The agent writes the task as stages with a measured gate between stages.
2. *Triage every node* with six measurements. A node whose result cannot be measured without the operator gets a measurement first. RL only where the tolerance is below what any sensor in the node can measure and contact decides the result. A VLA only where object variety is high and the tolerance is loose. Everything else is scripted, with classical or learned perception.
3. *Funnel the precision.* A direct camera measurement of tool against target brings the error to about 1 mm. A contact policy with a capture range of several millimetres does the rest.
4. *Train the contact policy in sim on a real-like plant*: relative observations, a vision prior, a contact head, history, masked inputs, a mating force. The plant is a small set of physical parameters fitted to the real logs. State which parameters the logs can identify.
5. *Learn from the rig by fixing the sim.* Each real failure is assigned to a node and a cause, by an intervention where one is possible. It is reproduced in sim as a failure mode, fixed in the cheapest class that the evidence allows, and gated by a dry run or a real-log replay before it returns to the rig. A learned fix needs the mode in sim. A rate claim needs the sim rate to agree with the rig rate.
6. *Optimise the graph's numbers and its schedule* on replayed real logs. For the schedule, change start times only, after a clearance check on the logged paths. Keep safety gates out of the optimiser's reach.
7. *Verify what ran.* The run log shows every config setting as executed. A new setting must change the logged behaviour.
8. *State the evidence.* Every number has its n. A rig comparison uses paired runs and a continuous outcome. A sim result is not a rig rate.

**Five conclusions.**

1. **The operator's view of RLT is correct.** RLT needs hours of teleoperated demonstrations per task, human success labels, optional human corrections and 400 to 1,000 real episodes. It reaches 65 % (from 20 %) on its sub-millimetre screw task. Its strong point is speed: about 3 times faster in the critical phase.
2. **No reviewed source combines the three things we have**: an agent-built graph, sub-millimetre real insertion, and zero real demonstrations without a force sensor. The agentic sources (TI-0.1, Moonlake, GaP) stop at coarse tasks or use a force sensor. TI-0.1 says its reconstructions are not yet fine enough for high precision.
3. **Only one of our nine nodes needs RL.** The triage rule gives RL for the insertion and scripted logic for the rest. The RL node has no failed start in the milestone count and takes about 1 % of the run time. At the milestone the alignment, a scripted node, had 3 of the 5 failed starts and 37 % of the time. New numbers in that node, set by hand from the logs, cut it to 10.1 to 13.9 s (n = 4).
4. **The gains came from the graph and from scripted perception, not from more RL or a VLA.** Measured on the rig on 2026-11-13: alignment about 60 s to 10.1 to 13.9 s (n = 4), approach uncommanded move 0.21 to 2.86 mm (n = 4), full run 117 and 122 s (n = 2; milestone 164 s). The one RL step that we called justified, a mating force in sim, is done (K6d, then K7). K7 passes its sim acceptance rule. It is below K4 on the replay of real hand-overs without the step alignment. It is not on the arm. The real insertion force is not measured, so the sim force range (0 to 10 N) is a guess. The arm logs cannot identify it (2026-11-15). The same day gave one more case for this conclusion: a perception retrain (zed_v6) is worse on real frames than the net it replaces, and the scripted edge fit stays the PC pose to use.
5. **Our claim is narrow.** n = 12 seated insertions, one reader spot, two PC spots about 4 mm apart, no true depth measurement. The fast chain has n = 4 alignments and n = 2 full runs. Fully in is an operator word on 4 runs, all with the older position push. The three tests that matter most are a depth measurement that does not need the operator, a locked set of placements, and an automatic extraction that allows n of 50 or more.

## What changed since 2026-11-13

Dated 2026-11-15. Sources: `docs/campaign/FINDINGS.md`, `docs/real/SESSION_20261113_duchamp3.md` and the server documents of 2026-11-14/15 on branch `real/graph-align`. The rig session of 2026-11-13 was short: 8 robot starts, 2 full insertions. The server work after it ran nothing on the arm.

**Confirmed on the rig.**

- *Fast alignment.* 10.1, 10.6, 13.1 and 13.9 s, 7 to 8 cycles (n = 4). The old alignment on the same day took 53.3 s and 19 cycles (n = 1). Our offline model said 14.3 s.
- *Approach.* The uncommanded move at the stop is 0.21, 0.24, 2.24 and 2.86 mm (n = 4). The old approach on the same day gave 5.88 and 2.32 mm (n = 2).
- *Run time.* 117 and 122 s (n = 2). The milestone run took 164 s.
- *Wrist-net offset.* No disagreement stop in 3 runs. The one run without the offset stopped at 3.56 mm against a limit of 3.5 mm.

**Failed on the rig.**

- *The force push did not solve depth.* The push reached 10 N and 14 N (n = 2). The operator saw a band of about 2 mm after both. More force did not change the result at rest.
- *The depth label is not valid.* It printed FULL at 7.33 mm for a run with a 2 mm band. No threshold on the arm-side depth separates the labels: fully in is 7.32 to 7.59 mm (n = 4) and one short run is at 7.33 mm. The new configs print NOT VERIFIED.
- *A new port-vision stop.* A light patch at the bottom edge of the case made a stronger edge than the right side of the plug shell. One alignment stopped on it.

**Found on the server (nothing on the arm).**

- *Spring-back.* The hand comes back 1.93 mm (10 N) and 3.37 mm (14 N) in the 0.3 s after the push, when the load goes to 0 in one tick. It is elastic deflection of the arm and the grasp: 0.25 to 0.29 mm/N on the way up. The position pushes of the runs that the operator called fully in spring back 1.39 to 1.53 mm too (n = 7 pushes). So the spring-back does not explain the band. The most likely reading is that the plug does not travel the last 2 mm at 10 to 14 N. Confidence is moderate: the arm-side depth sees the hand, not the plug.
- *K7 replaces K6d.* K7 is K4 fine-tuned for a PC box of ±30 mm and ±15° and a sim mating force of 0 to 10 N. It passes the acceptance rule (worst cell of the ±25 mm, ±15° box 97.1 %, 10 N mating force 99.1 %, n = 1,024 per cell). On real hand-overs without the step alignment it is below K4 (54.3 % against 80.1 % on one run, n = 256). After the step alignment the two are equal.
- *The PC estimate was wrong, and so was our statement about it.* The ZED net (v1) reads the port 8.1 mm low in y at the 2026-11-13 spot and 3 to 4 mm low at the 2026-11-03..11 spot. Its gate passed a mapping pose with a 16.3 mm error. An earlier statement, "v1 is within about 2 mm at the real spot", was wrong. A classical fit of the case edges is within 1.12 mm of the port that the arm measures (median x -0.02 mm, y +0.69 mm, n = 41 runs, 89 real frames replayed). The retrained net (zed_v5) follows moves within 1.2 mm (n = 5) but has a fixed offset of +4 to +6 mm in y on real frames.
- *Port-vision fix.* A shell reference from the prepare frames. Replay on 4,824 real align frames of 44 runs: 4,793 identical, 16 recovered, 0 newly rejected, 15 rejected by both.
- *Offline tests of two of our ideas.* Idea 2 is falsified. Idea 5 is weak. See the ideas table.

**What this did to the formula (as written on 2026-11-14).** The six lines stayed at that date. Two lessons were sharper. First, most of the gain since the milestone came from the cheapest fix classes: config numbers, scripted logic and classical perception. Second, the sim could not show two of the three new effects: the sim arm and grasp are about 10 times stiffer than the real ones, and the sim PC panel has the wrong grey level for the edge fit. A real-log replay tested each fix in place of the sim.

### Added on 2026-11-15: second server round

Sources: the entry "2026-11-15" of `docs/campaign/FINDINGS.md` and `docs/real/CONTACT_SENS_20261115.md`, `K8_20261115.md`, `FORCE_BUDGET_20261115.md`, `PERCEPTION_BIAS_20261115.md`, `PARALLEL_SCHEDULE_20261115.md` on branch `real/graph-align`. Nothing ran on the arm. No runbook step changed. Every new config is off by default and waits for the operator.

**Contact sensitivity (sim, n = 1,024 per cell).**

- *Port friction is the main sim risk.* K4 and K7 train with a port friction of 0.10 to 0.20. At 0.35 to 0.60, K4 falls from 94.6 % to 33.9 % and K7 from 98.7 % to 53.0 %. At 0.80 to 1.00 they seat 0.9 % and 7.1 %. The failure is a lateral miss: the plug does not slide into the opening.
- *The fit is second, the grasp third.* A tight plug (20 and 50 µm per side) costs K4 29 points and K7 7 points. A squeeze force of 0.6 times nominal costs about 10 points for each. The pad model is not a main risk.
- *K7 is less sensitive than K4* in every cell outside the training range.
- *The real seat times fit low friction.* The 14 real K4 insertions have 13 seat detections and a median seat time of 1.42 s. The sim fits them at a port friction of 0.20 or less (Kolmogorov-Smirnov p = 0.40 to 0.46, n = 512 per sim bin) and does not fit them above 0.20 (p ≤ 0.006). This is a statement about the sim model. It is not a measured friction coefficient: the sim port has sharp rigid edges and no lead-in.
- *Idea 1 is falsified.* A 2 mm alignment stop gives K4 94.1 % and 90.9 % (two prior models) and K7 98.9 % and 96.8 %; the pass limit was 98 %. A replay of 23 real alignment stages shows no gain: the stop comes at the same cycle in 19 stages and 0.2 to 1.2 s earlier in 4. The lateral tolerance does not set the stage time. The 1 mm stop stays.

**K8, a fine-tune of K7 for friction and fit (sim).**

- *No variant passes.* Three variants (K8a, K8b, K8c) seat 91.8 to 93.8 % at a port friction of 0.35 to 0.60 (K7 53.0 %, n = 1,024). All three fail the cell with an 8 mm start gap: 49.6 to 72.3 % against 97.1 % for K7 (n = 1,024). No checkpoint and no config came out.
- *A trade-off.* In the K8b checkpoints the 8 mm cell falls as the high-friction cell rises: at 15M steps 97.1 % and 80.9 %, at 30M steps 72.3 % and 93.8 % (n = 1,024 each).
- *The K7 gap stays.* On the real hand-over `full_A2` without the step alignment, K8b seats 55.5 %, K7 54.3 % and K4 80.1 % (n = 256 each).
- *A sim release artefact.* "Seat detected, not seated after the release" in the dry runs is not a property of the policy. A pad sphere of the open sim jaws touches the reader during the retreat and pulls the plug out. With the sim jaws at 24 mm this happens in 257 of 906 worlds (28 %). With the jaws at 27 mm it happens in 1 of 898 (0.1 %). No real run shows it. Confidence is moderate: the real side has no plug measurement.

**Force budget (real logs, then sim).**

- *The arm is a usable axial force sensor.* A linear free-space torque model reads the force along the port axis with a residual of 1.29 to 1.35 N (95th percentile, held out by date, 6 to 12 runs per date).
- *The arm delivers the commanded force.* 10 N gave 10.4 N and 14 N gave 14.3 N (n = 1 each).
- *More axial force is unlikely to help.* The 4 runs that the operator called fully in had 9.0 to 9.7 N at the end of the push. The 14 N push left the band (n = 1). The present push stops at about 14 to 16 N (advance watch and joint-1 torque cap).
- *One signal separates the labels, and it is confounded.* The lateral force at the end of the push is 5.4 to 5.9 N on the fully-in runs (n = 4) and 1.8 to 3.3 N on the short pushes (n = 4). All fully-in runs used the position push at the older PC spot. So this is a hypothesis for the rig, not a detector.
- *A small plant fits the pushes.* A 1-D plant (series compliance, play, a mating resistance) is fitted on 10 position pushes. On the 2 held-out force pushes the trace error falls from 1.33 and 1.73 mm to 0.14 and 0.35 mm RMS. The compliance is identified: 0.13 to 0.24 mm/N (bootstrap 5 to 95 %).
- *The mating force is not identifiable from the encoders.* The fit error is flat (0.233 to 0.246 mm) from 0 N to a plug that never moves. The encoders see the hand, not the plug.
- *The seat detector fires early in the fitted plant.* With the fitted compliance and a 10 N mating force in sim, K4's detector fires with the plug 1.4 to 2.6 mm short in 99.8 % of its detections (n = 1,024 worlds). The 14 N push then brings 32 % of the pushed worlds to full depth (n = 738). This is the rig pattern. It holds only if the real mating force is near 10 N, which is not measured.

**Perception.**

- *The zed_v5 offset has a cause.* zed_v5 reads y +4.0 to +4.4 mm on real frames (n = 89). It uses the yellow pad of the left gripper as a landmark, and it trained with that gripper fully open. A sim twin with closed fingers gives +4.59 mm; the training opening gives -0.32 mm (n = 89 each).
- *The retrain is worse.* zed_v6 trains with random finger openings. It has no offset in the sim twins (at most 0.34 mm, n = 89). On real frames it reads x -7.7 mm and y +7.2 mm against the port that the arm measures (n = 41 runs). zed_v5 reads x -1.3 mm and y +4.5 mm on the same runs. zed_v6 is not for the rig.
- *A learned fix failed where a scripted fix worked.* Two retrains of the PC net (zed_v5, zed_v6) did not pass the real-frame check. The classical edge fit is within 1.12 mm (n = 41 runs). The sim twin did not show the zed_v6 error; only the real-frame replay did.
- *The wrist-net up bias follows the grasp.* It follows the height of the plug in the jaws (Spearman +0.65, n = 40 runs; +0.40 on the 25 hand-over runs). A correction from that height leaves a residual of -0.54 to +1.19 mm in a replay of 25 real hand-overs, against -2.79 to +0.96 mm with the fixed offset. It is off by default and needs approval.

**Parallel schedule (clearance replay on real logs, dry runs).**

- *About 12 s saved per run*, of 117 to 122 s. Both arms move to the capture pose at the same time: the arms stay 116.4 mm apart or more (n = 32 real runs). Both arms go home with the right arm 3 s after the left: never closer than the serial home on 31 of 31 runs. A planned-clearance rule of 50 mm lets 4 of 7 homes run in parallel.
- *The left swing during the pick fails clearance*: 5 to 27 mm between the fingers (n = 29). The serial swing of today already passes at 9.5 to 27 mm.
- *30 paired dry runs* (serial against parallel, the same seeds): no new stop. The dry run tests the code path, not the timing. One operator go for a two-arm move needs approval.

**What this does to the formula.** Four of the six lines change (2, 4, 5 and 6) and two lines are new (7 and 8). The decisions and their evidence are in the next section. Three lessons:

1. The 12 of 12 can rest on a contact value that the sim has by guess. The sim says so itself: K4 seats 33.9 % at a friction of 0.35 to 0.60. The real seat times say that the real port acts like low friction in this model. One measurement by hand (insertion and extraction force) is still missing.
2. The depth problem is a measurement problem. Axial force, compliance, arm-side depth and the contact head do not separate fully in from short (n = 4 fully in, n = 4 to 7 short). The logs cannot give the mating force.
3. A learned fix failed again where a scripted fix worked (zed_v6 against the edge fit), and a retrain for robustness (K8) paid with another cell. Both were found before the rig, by a real-frame replay and by a fixed acceptance rule.

## Formula edits decided on 2026-11-15

Seven edits were proposed in `LOOKBACK_IDEAS_3_8_20261114.md`, `AGENTIC_RL_IDEAS_20261114.md` and `READINGS_20261114.md`. The limit is eight lines. "Own data" means our logs, our sim or our project record. "Literature" means the sources of the two companion pages; they were read through a summarising fetch tool.

| Edit | Decision | Own data | Literature |
| --- | --- | --- | --- |
| Every config setting appears in the run log | **Adopted, line 7** | **Yes.** The fix for the approach jump was written on 2026-11-03 as two config keys. The code never passed them. All 37 real approaches from 2026-11-03 to 2026-11-11 ran the old 11 iterations. The cause was put on the controller for 10 days. The conventions check passed each time | None used |
| "Reproduce" split into mode and rate | **Adopted, line 5** | **Yes.** The sim showed the K-J stall as a mode: 21.8 % (n = 1,024) and 0 % (n = 256 x 2). The rig rate was 2 of 4. With a 10 N spring the sim gives K4 0 %, and the rig K4 seats. A mode was enough to choose the K4 fix. No sim rate predicted a rig rate. The K8 friction fit is the first rate-level check (seat-time distribution, n = 14 real) | None used |
| A sixth triage measurement, result observability | **Adopted, line 2** | **Yes.** No arm-side signal separates fully in from short: arm-side depth (n = 4 and 1), contact head (rho 0.37, n = 8), axial force and compliance (n = 4 and 4 to 7). The mating force is not identifiable from the encoders. The final push has had no triage class since 2026-11-13 for this reason | Kress-Gazit et al. and ASPIRE name success detection as the first need |
| A small parametric plant, not a learned residual | **Adopted, line 4**, with a limit | **In part.** The parametric plant works at our data size: held-out error 1.33 and 1.73 mm to 0.14 and 0.35 mm RMS (n = 2 pushes, fitted on 10), and the sim with it reproduces the push advance (14 N: 3.2 to 4.6 mm, rig 4.03 mm). We did not fit a learned residual, so we have no comparison. Our data add the limit in the line: one parameter (mating force) is not identifiable | Dao and Fern (six parameters beat expressive models on about 4 minutes of data); ASAP (over 400 clips for a learned model). The "not a learned residual" part rests on these only |
| Attribution by intervention | **Adopted in a weak form, line 5** ("where one is possible") | **In part.** Two causes of 2026-11-15 came from interventions, not from reading: the zed_v5 offset (finger sweep and image swap on 89 sim twins) and the release artefact (jaws at 27 mm: 28 % to 0.1 %, n = 906 and 898). Log reading gave a wrong first cause at least 5 times (see the loop table). The node-swap replay itself has not run, so its accuracy on our failures is not known | Who&When (reading the log finds the decisive step in 14.2 % of cases); GTP-FA (a fix on the wrong stage lowers success by 28 to 55 points) |
| An evidence rule for small n | **Adopted in a short form, line 8** | **In part.** The depth label rested on 3 runs and failed on the next run. The lateral force separates the labels at n = 4 and 4 and is fully confounded. K4 (10 of 10) and the re-centre (3 of 3) are not different at this n, and this document still said that the retrain was needed. The paired, sequential test is not in the line: we have not run it | STEP (a 36-point gap needs about 20 trials; an 8-point gap is not decided in 100); N-SCORE (continuous, paired outcomes need fewer trials). The pairing rests on these only |
| Line 6 grows to "numbers and schedule" | **Adopted, line 6** | **Offline only.** Clearance replay on 29 to 32 real runs and 30 paired dry runs: about 12 s saved, no new stop. The same replay rejected one overlap (the left swing). Not on the arm | ScheduleGen (reviewed in `READINGS_20261114.md`) |

**Not adopted now.**

- *A force funnel beside the position funnel* (ideas page, edit 2). Our data now argue against the aim: the fully-in runs had 9.0 to 9.7 N and 14.3 N left the band. The budget is done as an analysis. It does not become a line before the force is measured by hand.
- *A sequential stop rule, a claims ledger, a second agent on the raw logs, a read-only evaluation set* (ideas page, edits 5 and 6). All have literature support only. None has run here. They stay as experiments.
- *"A sim result counts toward a rig claim by the measured sim-real correlation"* (ideas page, edit 6). We have no such correlation. Line 8 has the weaker form that our data support: a sim result is not a rig rate.
- *"Try the no-training fix for the same cause first, at equal n"* (look-back, change 2). It is now a rule in "The order of fix classes", not a line. Support: one case (the re-centre, 3 of 3).

**What the eight lines still do not say.** They give no rule for a retrain that trades one cell for another (K8). The fixed acceptance rule caught it. They also give no rule for the dry-run gate when the sim has a known artefact (the release). Today the answer is a sim option for dry runs, which is a proposal only.

## The loop

The formula is a loop with seven steps. Real runs are used as measurements of the simulator, never as demonstrations.

```
1 Build the graph --> 2 Gate in sim --> 3 Run on the rig --> 4 Read the logs
  (agent writes        (dry run,          (operator            (events, ticks,
   the stages)          conventions)       says go)             frames)
                           ^                                        |
                           |                                        v
                     7 Fix it in sim <-- 6 Pick the fix class <-- 5 Find the node
                     (retrain or          (one of five) [*]        (which stage,
                      rewrite)                                      what cause) [*]

Fix classes: config, scripted logic, sim fidelity, RL retrain, perception retrain.
[*] A person still checks these steps today. They are the steps to automate.
```

The run goes clockwise. A fix never goes straight to the rig: it returns through the gate (step 2).

| Step | What happens | Done by today | To automate |
| --- | --- | --- | --- |
| 1 Build the graph | The agent writes the stages, the gates between them and the stop rules, each as code with a config file | Agent, with the operator's task description | Node triage (below) run by the agent on every new node |
| 2 Gate in sim | Dry run of the whole graph through the real code path; a conventions check (frames, units, signs, weights hash); sim acceptance cells for a policy (n = 1,024 each); replays of real logs | Agent | Mostly automatic. Added on 2026-11-14/15: a gate replay (35 gates, 37 real runs), a port-vision replay (4,824 real frames), an edge-fit replay (89 real frames), a clearance replay (29 to 32 real runs). Where the sim cannot show the effect, the replay is the only gate. Two gaps: no check that each new config setting changes the executed behaviour (formula line 7; a dropped setting ran on 37 real approaches), and the dry run cannot test the release (a sim artefact, 28 % of worlds, n = 906) |
| 3 Run on the rig | One go per motion from the operator; every stage logs events, 50 Hz joints and frames | Operator | Stays manual for safety. Add: the operator records the true result (depth, fully in) for every run |
| 4 Read the logs | One table per session: stage times, gate values, outcome | Agent (`review_runs.py`), started by a person | Run after every session without a request |
| 5 Find the node | The failing or slow stage and its cause, with the evidence | Agent proposes, person confirms. At least five causes or claims were wrong at first: the late ticks, the wrist-net bias, "v1 is within about 2 mm", the 7.2 mm depth threshold, and the approach jump (put on the controller for 10 days; the cause was a setting that the code did not pass). Data corrected each one. Five different main causes were named for the K-J stall from 2026-11-02 to 2026-11-08 | An intervention in place of reading, where one is possible: change one thing in the replay or the sim twin and see whether the outcome changes. Two causes of 2026-11-15 came this way (zed_v5 landmark, release artefact). Failures are in four different log events today; one outcome event per run is needed |
| 6 Pick the fix class | Config, scripted logic, sim fidelity, RL retrain or perception retrain | Person, mostly | A written rule: try the cheapest class that the evidence allows; a learned fix (RL or perception retrain) only when the sim reproduces the failure mode; a real-frame or real-log replay before the rig |
| 7 Fix it in sim | Change the node or the sim, retrain if needed, new config file, never an edit in place | Agent | Already automatic |

**The order of fix classes.** Cost rises from left to right: config (minutes), scripted logic (hours), perception retrain (hours, needs real frames), sim fidelity (a day, needs a real measurement), RL retrain (a day of GPU). In our 2026-11-10/11 data, most fixes were the two cheapest classes: the reach test, the fixed slope prior, the seat check speed, the move speed. An earlier text said here that RL retrain was needed three times (K2 to K4). The look-back of 2026-11-14 does not support this. K2 and K3 did not go to the rig. One retrain line reached the rig, as K4, after the sim showed the failure mode. A config fix for the same cause (the observation re-centre, no training) seated 3 of 3 on the rig; K4 seated 10 of 10. These two rates are not different at this n, so the record does not show that the retrain was necessary. The rule from this case: try the no-training fix for the same cause on the rig first, and compare it with the retrain at equal n. The 2026-11-13 data repeat the pattern of cheap fixes. The approach jump came from a number that the code never passed (6 iterations at the stop in place of 4 and 2); a scripted fix removed it. The port-vision stop got a scripted rule. The PC estimate is the clearest case: the net retrain (zed_v5) did not pass the real-frame check, and a classical edge fit that takes 0.1 s on the CPU is within 1.12 mm (n = 41 runs). Classical perception belongs in the scripted class, before a perception retrain. The second retrain (zed_v6, 2026-11-15) confirms it: no offset in the sim twins (n = 89), and x -7.7 mm, y +7.2 mm on real frames (n = 41 runs). Four learning efforts started without a sim reproduction of a real failure (K4c, RL alignment, zed_v5, K6d and K7). None has a rig success so far; two are dead and two are pending.

## Where we stand

On 2026-11-10 and 2026-11-11 the real rig seated the plug 10 times out of 10 when the plug reached the port, and 10 of 15 starts finished end to end. On 2026-11-13 it seated 2 of 2 more with a faster chain. The insertion policy had zero real demonstrations. All numbers in this section are from our own logs: `docs/real/DATA_REVIEW_20261113.md`, `FULL_DEPTH_20261113.md`, `ALIGN_FAST_20261113.md` and the session files for the milestone; `SESSION_20261113_duchamp3.md`, `DEPTH_SPRINGBACK_20261114.md` and `docs/campaign/FINDINGS.md` on `real/graph-align` for the update.

**The task.** Two YAM arms (6 joints each, no force or tactile sensor, wrist RGB cameras, one fixed ZED stereo camera). The right arm picks a USB-C SD card reader from the table and inserts it into a port of a mini PC. The left arm presses on the PC to hold it. The plug is 5.8 mm long and the clearance is under 0.1 mm.

**The terms.**

- *Graph*: the task as a chain of stages (nodes) with a check between stages (a gate). The agent harness writes and edits this graph.
- *K4*: the insertion policy. A neural network trained with reinforcement learning (PPO) in simulation only. It reads 0.6 s of joint history, the hand pose relative to the hand-over pose, and one camera measurement of the plug against the port (the *vision prior*). It has a *contact head*: an extra output that says free, touching or inside.
- *Hand-over*: the moment one node gives control to the next. Here: the alignment node stops 6 mm in front of the port and K4 takes over.
- *Seated*: the seat detector saw the plug at depth and at rest. *Fully in*: the operator saw the plug at its stop. By hand (2026-11-13): fully in shows no silver band; a plug 1 to 2 mm out hangs.
- *Arm-side depth*: the hand advance from the encoders minus the camera gap at the hand-over. It sees the hand, not the plug.
- *Dry run*: the whole graph runs in the simulator through the real code path before any real motion.

**The result, with its limits.**

| What | Number | Limit |
| --- | --- | --- |
| Insertion when the plug reached the port | 12 of 12 seated (Wilson 95 % interval 76 to 100 %; the sum is ours): 10 of 10 on 2026-11-10/11, 2 of 2 on 2026-11-13 | n = 12; one reader spot; two PC spots about 4 mm apart in y (edge fit, ±1.5 mm) |
| End to end | Milestone: 10 of 15 starts. 2026-11-13: 2 of 3 full-mode starts | Milestone stops: 3 at the alignment hand-over, 1 empty pick, 1 software end; each rerun seated. 2026-11-13: 1 stop at the hand-over with an old config (3.56 mm against 3.5 mm). Also that day: 2 starts with no motion (camera port, viewer label) and 1 port-vision stop in align-only mode |
| K4 time to depth | under 1 s on 8 of 9 logged insertions; seat detected at 1.1 to 1.7 s (one at 3.9 s); 1.12 and 1.26 s on 2026-11-13 (n = 2) |  |
| Depth | **Not verified.** Fully in by the operator on 4 runs, all with the 15 mm position push. Short on 7 runs. The force push (10 N and 14 N, n = 2) left a band of about 2 mm each time | No true depth sensor. No threshold on the arm-side depth separates the labels (fully in 7.32 to 7.59 mm, n = 4; one short run 7.33 mm). The earlier "0.3 to 1.9 mm short" (n = 8) is in these units and is not a measured plug depth |
| Alignment time | 10.1, 10.6, 13.1, 13.9 s (n = 4, fast alignment, 2026-11-13). Before: median 59.7 s, range 42 to 70 s (n = 14) | Steps still end at the 2 s timeout (7 of 23 steps in the 4 runs) |
| Hand-over accuracy | across and up within 0.8 mm, gap 5.6 to 6.9 mm (n = 14). Fast alignment: within 0.67 mm, gap 5.2 to 6.8 mm (n = 4) |  |
| Approach | uncommanded move at the stop 0.21 to 2.86 mm (n = 4). Before: over 3 mm on 7 of 18 approaches | The pass limit is 3 mm; one run is at 2.86 mm |
| Run time | 213.7 s, then 164.3 s with faster moves, then 117 and 122 s (n = 2) with the fast alignment | Human teleoperation would be much faster |

**How the agent learned from the rig.** No real demonstration was collected. Each real failure was traced in the logs to a cause, the cause was put into the simulator or the graph, and the policy was retrained or the node rewritten. Three examples: (1) the real arm reads 0.4 to 1.3 degrees off on joints 2 to 5, so absolute pose inputs failed on the rig; K4 uses only relative pose. (2) The earlier policy read the left arm's press error as a hidden port locator; with the real values it fell from 89.8 % to 0 % in a replay of two real hand-overs (n = 256 each); K4 masks all left-arm inputs. (3) The sim port did not resist the plug; a mating force is now in the sim (K6d, then K7, sim only). We do not know that the real port resists with a force in the trained range: the real insertion force is not measured.

**What failed or was wrong.** (1) We expected a force push along the port axis to reach full depth. It did not (n = 2). (2) We read the spring-back after the push as the cause of the shortfall. The same spring-back is on the runs that the operator called fully in. (3) The depth label rested on 3 labelled runs and failed on the next run. (4) We wrote that the PC net was within about 2 mm at the real spot. It is 3 to 8 mm low in y. The alignment absorbed the error on every run, so no run showed it. (5) K7 meets every sim acceptance cell and is still worse than K4 on bad real hand-overs. The acceptance rule did not contain the replay.

**What K4 does not do.** A probe of K4's memory shows that it does not find the port from contact. Its position error after contact stays at the camera error (across 0.43 to 0.47 mm, up 0.93 to 0.96 mm, n = 1,024, sim). It goes where the camera measurement points, and its search tolerance does the rest. With the camera error three times larger, sim success falls from 99.1 % to 81.8 %. So the precision of the whole chain today comes from the camera alignment, not from touch.

## Paper-by-paper review

No source reviewed here shows an agent-built pipeline that reaches sub-millimetre precision on a real robot with zero real demonstrations. The sources split in two groups: agentic real-to-sim loops that stop at coarse tasks, and precise-task methods that need real demonstrations, human corrections or a force sensor. Every number below has its status in the Sources section.

### RLT (Physical Intelligence): RL Token

Paper: [RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/abs/2604.23073), Xu et al., 24 April 2026 (v2 30 April). The paper was read; the blog page returned an error (HTTP 429) on every attempt.

- **Idea.** A vision-language-action model (VLA: one large network from camera images and a text command to robot actions) does the whole task. For the hard part, a small actor-critic network learns online on the real robot. It reads one compressed token of the VLA's internal state (the RL token) plus joint state, and it edits the VLA's proposed action chunk. A penalty keeps it near the VLA's action.
- **Agent and human.** There is no LLM agent. A human gives teleoperated demonstrations, labels each episode as success or failure, may take over to correct, and chooses when control passes from the VLA to the RL part during training. The paper says the method does "require additional human intervention during training to provide reward signals, intervention corrections, and switching".
- **Data.** Teleoperated demonstrations per task to fine-tune the VLA first: the HTML shows a garbled "11-1010 hours"; an alphaXiv overview reads it as 1 to 10 hours. Then 400 to 1,000 real episodes, "approximately 15 minutes to 5 hours of actual robot data". No simulation.
- **Policy class.** Frozen π0.6 VLA plus a small TD3-style actor and critic. 50 Hz, RL chunks of 10 steps.
- **Precision and results.** Screw installation (M3, "sub-millimeter alignment"): success "from 20% to 65%" on the critical phase. Zip tie ("millimeter precision"): full-task success improves by 60 %. Ethernet: same success as the base policy, about 3 times faster in the critical phase, faster than the teleoperated demonstrations. Charger: "centimeter-level alignment", about 3 times faster. Evaluation: 50 episodes per task in the critical phase.
- **Hardware and sensing.** Bimanual arms with cameras; no force sensor is described.
- **Not shown.** Any result without demonstrations. Success above 65 % on the sub-millimetre task. Generalisation to new part positions as a measured rate. An automatic reward. Per-task success tables in the text (several numbers are in figures only).

**The operator's view checked.** Correct. RLT rests on real demonstrations (hours per task), human success labels, optional human corrections and a human-chosen hand-over. Its own contribution, the online RL part, needs minutes to hours.

### TI-0.1 (Tutor Intelligence)

Report: [TI-0.1 Technical Report](https://research.tutorintelligence.com/ti0-1), 9 November 2026. Read in full through a summarising fetch tool.

- **Idea.** A method called TRACE: reconstruct the real cell in simulation from recorded images, commands and robot state; train a policy with privileged state by PPO in Isaac Lab; distil it into a camera-based diffusion policy (TI-0.1); run it on hardware; feed the hardware runs back into the reconstruction.
- **Agent and human.** LLM agents (the report names GPT-6 Astra and Claude Opus 5.5) compare real and rendered views, propose scene changes, write each task as a hybrid MDP (discrete modes with guard conditions, the same thing as our graph of stages with gates) and write the reward, "with minimal human supervision". Humans give "a few" demonstrations to seed a reconstruction, plus CAD and calibration. The report does not say that the hardware feedback is automatic.
- **Data.** A few demonstrations per reconstruction (count not given). Fleet data from a 100-robot facility for lighting and materials.
- **Policy class.** Privileged PPO teacher, then an action-chunking diffusion policy from cameras.
- **Precision and results.** Lego pick and place: 99.8 % in simulation on held-out configurations. No real success rate and no n are given. Arm replay error 9.9 mm RMS over 6.4 s; after timing calibration under about 0.5 mm on straight-line trials.
- **The most useful finding for us.** A policy that picked bricks by their studs scored 84.5 % with the first contact model and 2.6 % after the pad stiffness, travel and friction were revised (2,048 episodes per condition, frozen policy, sim). A body-grasp policy went from 92.8 % to 96.0 %. A sim success can rest on contact physics that the real gripper does not have.
- **Not shown.** Any real success rate. Any precise task. The report says: "High-precision tasks demand finer geometry and contact fidelity than our current reconstructions provide."

### Moonlake: simulation-enabled scaling

Blog: [A Simulation-enabled Scaling Paradigm for Robot Learning](https://moonlakeai.com/blog/simulation-enabled-scaling-paradigm), October 2026. Read in full through a summarising fetch tool. The robot is a bimanual YAM, the same arm as ours.

- **Idea.** Three uses of a calibrated simulation: evaluate policies in sim instead of on the robot, diagnose failures with an agent, and improve policies in sim.
- **Agent and human.** An agent diagnoses failures in sim rollouts (example: a policy knocks neighbouring bottles off the table in dense layouts). Humans give a few teleoperated trajectories to calibrate camera, lighting, mass and friction, and 50 to 400 teleoperated episodes per task for most policies.
- **Data.** π0.5 fine-tuned on about 50 to 400 teleoperated episodes. One variant uses no real data beyond one calibration trajectory: a code policy with privileged state makes sim demonstrations, and π0.5 is fine-tuned on them.
- **Results.** Sim against real ranking of 15 policies: Pearson r = 0.93. RL in 384 parallel sim environments on the sort-from-tote task (nuts, bolts, bearings; starting from 50 teleoperated episodes): about +38 points in sim and +39 points on the real robot; grasp success 16 % to 48 %; 28.0 s to 17.8 s per correct sort. Zero-shot distillation: about 26 to 78 % task progress on bottles and 12 to 44 % on mugs over 40 real layouts, one run each. Iteration cost: about 5.5 days real-only against about 20 hours with sim.
- **Stated lesson.** "Physics, compared to visuals, have outsized impacts towards achieving high sim-real correlation at a policy level."
- **Not shown.** Any millimetre-level task. Absolute success is low (grasp success 48 % at best). Small n per configuration. The simulation stack is proprietary.

### Final-millimetre methods that use real data

All of these train on the real robot. "Interventions" means a person takes over by teleoperation during training.

| Paper | Idea and policy class | Real data and human role | Precision result (task, n) | Does not show |
| --- | --- | --- | --- | --- |
| [HIL-SERL](https://arxiv.org/abs/2410.21845), Luo et al., Oct 2024 | Off-policy RL from images on the real robot, with a learned success classifier as reward | 20 to 30 demonstrations per task, plus interventions; 1 to 2.5 h training (timing belt 6 h) | RAM insertion 29 to 100 %, USB grasp-insertion 26 to 100 %, SSD assembly 79 to 100 % (behaviour cloning to HIL-SERL, 100 trials each) | Tolerances in mm; any result without a person in the loop |
| [SERL](https://arxiv.org/abs/2401.16013), Luo et al., Jan 2024 | The same family, earlier | 20 demonstrations per task; 20 to 105 min training | PCB insertion, cable routing, relocation: 100 of 100 each | Tolerances |
| [π\*0.6 / RECAP](https://arxiv.org/abs/2511.14759), Physical Intelligence, Nov 2025 | A VLA improved with its own experience through an advantage-conditioned policy | Demonstrations, autonomous trials and interventions (box assembly: 600 autonomous + 360 intervention trials per iteration); human success labels and resets | "More than doubles task throughput and roughly halves the task failure rate" on the hardest tasks | No insertion task |
| [VLA-Precision](https://arxiv.org/abs/2609.04355), Su et al., Sep 2026 | Online RL of π0.5 (LoRA) | 60 to 120 demonstrations per task plus interventions; 45.8 min per task | Nine chemistry tasks, rubber stopper ±0.5 mm, tube rack ±0.25 mm: 177 of 180 held-out trials | A zero-demonstration variant |
| [ForceRFT](https://arxiv.org/abs/2609.22840), Wang et al., Sep 2026 | Force-guided residual RL on a frozen SmolVLA | 50 demonstrations per task, corrections, 30 min online | Plug insertion 7 to 25 of 30; ring on peg 10 to 27 of 30 | Tolerances; needs a force signal |
| [From Reach to Insert](https://arxiv.org/abs/2605.04649), Meng et al., May 2026 | Imitation for the reach, then human-in-the-loop RL with tactile sensing | 100 demonstrations per geometry; interventions; about 30 min | 0.05 mm clearance: imitation 0 of 75, imitation + RL 39 of 75, full method 50 of 75. 0.25 mm: 70 of 75 | The paper's text and table disagree on some rates |
| [Learning Where it Matters](https://arxiv.org/abs/2607.15982), Hausdörfer et al., Jul 2026 | One demonstration, then autonomous local data collection and offline RL | 1 kinesthetic demonstration; 300 to 500 autonomous episodes (2 to 2.5 h) | Average 96 % (50 rollouts per task) against fine-tuned π0.5 20 %; HIL-SERL 10 % and 6 % on two of the tasks | Needs a state-based reward |
| [PLD](https://arxiv.org/abs/2511.00091), Xiao et al., Oct 2025 | Residual RL on a VLA makes data, then the VLA is distilled on it | 200 teleoperated trajectories; up to 2 h (Franka), up to 8 h per sub-task (YAM GPU insertion) | Franka peg insertion 30 of 30 for all methods (not a hard test) | Tolerances |
| [ConRFT](https://arxiv.org/abs/2502.05450), Chen et al., Feb 2025 | Offline + online RL fine-tuning of Octo | 20 to 30 demonstrations, interventions, 45 to 90 min | 96.3 % average over 8 tasks, 20 trials each | Insertion-specific numbers |
| [DSRL](https://arxiv.org/abs/2506.15799), Wagenmaker et al., Jun 2025 | RL in the noise space of a diffusion policy | No interventions; 40 to 80 real episodes | π0 toaster task 5 to 18 of 20 | No real insertion |
| [ResFiT](https://arxiv.org/abs/2509.19301), Ankile et al., Sep 2025 | Off-policy residual RL on a behaviour-cloned base | About 900 to 1,000 demonstrations; 15 to 76 min online | Two humanoid tasks: 14 to 64 %, 23 to 64 % | No insertion |
| [ResiP](https://arxiv.org/abs/2407.16677), Ankile et al., Jul 2024 | Residual PPO on a diffusion policy, in sim | 50 sim demonstrations; 10 to 40 real ones for transfer | Sim peg-in-hole 5 to 99 % (1,024 episodes); real one\_leg 50 to 60 % against 20 to 30 % (10 trials) | Zero real demonstrations |
| [TRANSIC](https://arxiv.org/abs/2405.10315), Jiang et al., May 2024 | Sim policy plus a residual learned from human corrections on the real robot | 20 to 100 corrected trajectories per skill (Insert: 90 trajectories, 489 corrections) | Insert 45 % against 0 % direct transfer (20 trials) | Tolerance in mm |

### Sim-to-real insertion with zero real demonstrations

This is our family. Clearance is the free space between plug and hole. Ours is under 0.1 mm.

| Paper | Sensing | Clearance | Real result (n) | Does not show |
| --- | --- | --- | --- | --- |
| [Blind Precise Assembly](https://arxiv.org/abs/2610.23849), Bronars et al., Oct 2026 (the paper this repo reproduces) | Joint encoders only | 1.11 to 1.49 mm per side; one tighter peg at 0.5 mm | 173 of 237 (73 %), eight printed pegs, start in contact near the hole | A full chain; tight clearance on real hardware |
| [IndustReal](https://arxiv.org/abs/2305.17110), Tang et al., May 2023 | Wrist depth camera, joints; no force sensor | 0.5 to 0.6 mm | Insert 100 of 120; pick-place-insert 89 % of 120 | Below 0.5 mm |
| [AutoMate](https://arxiv.org/abs/2407.08028), Tang et al., Jul 2024 | Wrist camera for pose; no force sensor | 0.5 to 1.0 mm diametral | Specialists 86.5 % (200 trials, 20 assemblies) | Below 0.5 mm |
| [FORGE](https://arxiv.org/abs/2408.04587), Noseworthy et al., Aug 2024 | Joint-torque force estimate | 0.5 mm diametral; position error up to 5 mm | Peg 0.84, gear 0.98, M16 nut 0.69 (45 trials each) | Works without a force estimate |
| [TacSL](https://arxiv.org/abs/2408.06506), Akinola et al., Aug 2024 | GelSight tactile fingers | Not verified | Peg insertion 67 of 81 | Clearance |
| [SRSA](https://arxiv.org/abs/2503.04538), Guo et al., Mar 2025 | Not stated in the part read | AutoMate parts | 90 % against AutoMate 54 % (5 assemblies x 10) |  |
| [SPARR](https://arxiv.org/abs/2602.23253), Guo et al., Feb 2026 | Wrist RGB + force/torque | NIST gear 0.005 to 0.014 mm | 95 to 100 % on 10 AutoMate tasks (20 episodes each) against 65 to 80 % zero-shot. Uses about 0.5 h of real RL per task, no human demonstrations | Without a force sensor |
| [NVIDIA blog](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/), Oct 2026 | Force/torque | Cable connectors | "Success rates of 90-95%" against a factory target of 99.5 %; about 40 s per cable | n, tolerances |
| [InsertAnything](https://arxiv.org/abs/2609.24511), Ma et al., Sep 2026 | Fingertip force + joints, no vision | Trained at 2, 0.5, 0.1, 0.02 mm | 0.02 mm: 20 of 20 with force, 15 and 10 of 20 with pose only. 152 of 160 on eight unseen connectors | Without force sensing |
| [PACE](https://arxiv.org/abs/2609.07534), Wang et al., Sep 2026 | Two wrist RGB, wrist force/torque, joints | About 0.1 mm radial | Peg insert 96.7 % (30 trials) | Without force sensing |
| [Rebar insertion](https://arxiv.org/abs/2609.20477), Sun et al., Sep 2026 | RGB student | 1.4 mm | 137 of 150 | Tight clearance |
| [Success Guided Sampling](https://arxiv.org/abs/2610.12465), Zhang et al., Oct 2026 | RGB + joints, no force sensor | Not reported | Gear mesh 47 of 50; rod in hole 30 of 49; nut and bolt 18 of 48 | Clearances |
| [AlphaNIST](https://arxiv.org/abs/2610.13354), Zhang et al., Oct 2026 | Point cloud + joints | NIST board incl. USB-A, RJ45 | 98.2 % training success, simulation only | Any real result |
| [Touch2Insert](https://arxiv.org/abs/2603.03627), Yajima et al., Mar 2026 | Tactile pose estimate + stiffness control, not RL | Real connectors | USB-C 65 %, Lightning 100 %, audio jack 95 % (20 trials each) |  |

**What the two tables say together.** (1) Every method that reaches about 0.1 mm or tighter on real hardware has a force or tactile signal, or real RL with a person. (2) Without force sensing, most published zero-demonstration results stop at about 0.5 mm clearance. An earlier text said all. The second literature pass (`AGENTIC_RL_IDEAS_20261114.md`) found one at about 0.1 mm: Marougkas et al. ([arXiv:2505.11858](https://arxiv.org/abs/2505.11858)), with an RGB-D pose tracker and no force sensor described, report a cylinder at 10 of 10 and a triangular plug at 3 of 10. (3) We found no published zero-demonstration RL result on real USB-C. Our 12 of 12 seated at under 0.1 mm with encoders and one RGB camera is unusual, and it is also small (n = 12, one placement, depth not verified).

### Agentic harnesses, real-to-sim loops and graph tuning

| Paper | What the agent does | Who decides code against learned policy | Result (n, sim or real) | Does not show |
| --- | --- | --- | --- | --- |
| [GaP: Graph-as-Policy](https://arxiv.org/abs/2607.05369), Chen et al., Jul 2026 (the harness idea we use) | Agents build a typed graph of skill nodes; LLM edits are kept or rejected on 100 fresh sim instances | No stated rule; a node can call a VLA. No RL | Real USB-C insertion 121 of 130 on a UR5 with **force/torque feedback** and force thresholds (insert below 10 N, confirm at 30 N), about 30 s per insertion; port pose varied over 3 distances and 3 angles | RL in a node; insertion without a force sensor. It says reliability "is not yet at industry levels" |
| [ENPIRE](https://arxiv.org/abs/2606.19980), Xiao et al., Jun 2026 | Coding agents build reward, reset and safety on real YAM arms, then choose among heuristics, code policies, behaviour cloning and RL | The agent, case by case | Pin insertion time from over 1.5 h to about 40 min with eight agents. A "99% success rate" is stated without task or n | A traceable success table; two of three agents failed real Push-T |
| [RPG](https://arxiv.org/abs/2610.02204), Wang et al., Oct 2026 | An LLM agent refines a skill library in MuJoCo with a YAM model; no weight updates | All code | Sim 28.6 to 95.0 % over 15 rounds, 22 tasks. Real 30 of 30 on three coarse tasks | Precise insertion (one qualitative clip) |
| [RoboSimStudio](https://arxiv.org/abs/2610.18835), Chen et al., Oct 2026 | Agents build simulations from a phone video, on MuJoCo Warp | Not applicable | 18 simulations including USB-C "with sub-millimeter clearance"; no insertion success rate | Sim-to-real numbers |
| [Agentic RSR](https://arxiv.org/abs/2610.10479), Li et al., Oct 2026 | An agent reconstructs the scene in MuJoCo and writes a program policy | All code | 18 tasks: sim 10 of 18, real 8 of 18, one run each | Precision |
| [EmbodiedSWE](https://arxiv.org/abs/2609.27308), You et al., Sep 2026 | Agents write `solve(env)` programs in Isaac Lab; solutions become VLA training data | All code, then a VLA | Real lamp disassembly stages 100 / 80 / 30 / 20 % (10 trials) | The later, finer stages fail |
| [CaP-X](https://arxiv.org/abs/2603.22435), Fu et al., Mar 2026 | Benchmark of coding agents; RL on the coding model | All code | Real Franka: lift 84 %, stack 76 % (n = 25) | Contact-rich tasks |
| [Eureka](https://arxiv.org/abs/2310.12931) and [DrEureka](https://arxiv.org/abs/2406.01967), Ma et al., 2023 and 2024 | An LLM writes reward code (Eureka) and randomisation ranges (DrEureka) | Not applicable | Eureka beats expert rewards on 83 % of 29 sim tasks. DrEureka on a real quadruped: 1.83 against 1.32 m/s | Manipulation with tight tolerance |
| [RoboGen](https://arxiv.org/abs/2311.01455), Wang et al., Nov 2023 | An LLM proposes tasks, splits them, and picks the learner per sub-task | The LLM: RL for "contact rich" sub-tasks, motion planning elsewhere | Sim only: average success 0.774 on 69 tasks; RL alone "completely fails for most tasks" | Real results; a measured rule |
| [Plan-Seq-Learn](https://arxiv.org/abs/2405.01534), Dalal et al., May 2024 | An LLM gives (target region, end condition) pairs; a motion planner reaches; RL acts near contact | Fixed by design: plan in free space, learn at contact | Sim: NutAssembly 0.96. Without the end-condition check: 69 % against 100 % | Real robot |
| [ManipGen](https://arxiv.org/abs/2410.22332), Dalal et al., Oct 2024 | A VLM plans; learned local policies handle contact | Fixed by design | Real 76 % zero-shot over 5 tasks x 10 trials | Insertion |
| [GUAPO](https://arxiv.org/abs/2005.10872), Lee et al., May 2020 | Model-based control outside the pose estimator's uncertainty region, RL inside it | **A measured rule: the perception uncertainty region** | Real peg insertion 93 % (30 trials); pure model-based 0 %, pure RL 0 %, residual 0 % | Needs 90 min of real RL |
| [Residual RL](https://arxiv.org/abs/1812.03201), Johannink et al., Dec 2018 | Hand-made controller plus a learned residual | Fixed | Real block insertion, misaligned: 2 of 20 to 15 of 20, under 3 h |  |
| [Behaviour-tree parameters](https://arxiv.org/abs/2109.13050), Mayr et al., Sep 2021 | CMA-ES tunes thresholds and skill parameters of a behaviour tree in a randomised digital twin | Not applicable: the structure is fixed, only numbers are learned | Real peg-in-hole (15 trials per policy): no search about 20 %, random parameters 53 %, learned in sim median 100 % | Tight clearance |
| [Skill-based multi-objective RL](https://arxiv.org/abs/2203.10033), Mayr et al., Mar 2022 | A planner builds the tree, Bayesian optimisation tunes 3 to 4 parameters in sim | Not applicable | Peg task (1.5 mm): learned 96 %, random 41 %, six human operators 52.2 %; sim or real not clear from the text |  |
| [Sequential Dexterity](https://arxiv.org/abs/2309.00987), Chen et al., Sep 2023 | A learned transition feasibility function says whether the next policy can succeed from this state; used to fine-tune the previous policy and to choose the stage at run time | Not applicable | Real 12 of 20 and 10 of 20 against 3 of 15 and 5 of 18 for the baseline |  |
| [RialTo](https://arxiv.org/abs/2403.03949), Torne et al., Mar 2024 | Real-to-sim-to-real: scan the scene, RL in the twin with a demo loss, distil | People | About 15 real demonstrations; real 91 % against 25 % for behaviour cloning under pose randomisation | Centimetre-level tasks only |
| [ASID](https://arxiv.org/abs/2404.12308), Memmel et al., Apr 2024 | An exploration policy is trained in sim to excite the unknown parameters; one real episode identifies them | Not applicable | Real rod balancing 6 of 9 against 0 of 9 for domain randomisation | Insertion |
| [PolaRiS](https://arxiv.org/abs/2512.16881), [SimFoundry](https://arxiv.org/abs/2606.28276), [GAUGE](https://arxiv.org/abs/2608.05948) | Sim as an evaluation tool; physical fidelity benchmarks | Not applicable | Sim-real policy ranking r = 0.9 (PolaRiS), 0.911 (SimFoundry). GAUGE: "no engine dominates" | No insertion task |

**What this table says.** (1) Agent harnesses today are graphs of code, sometimes with a VLA in a node. None trains an RL policy for a precise node, except ENPIRE, which gives no traceable numbers. (2) The only harness with a real USB-C number (GaP) uses a force sensor and force thresholds. (3) The choice between code and learned policy is made by design or by an LLM's opinion. Only GUAPO uses a measured rule, and it is the right kind of rule: learn where perception is uncertain. (4) Tuning the numbers of a fixed graph in a randomised simulation works on real hardware (Mayr: about 20 % to 100 %, small n).

## Comparison with our pipeline, stage by stage

We are ahead on precision without demonstrations or a force sensor. We are behind on depth, on evaluation size, on automatic scene building and on generality. On alignment speed the gap is smaller than on 2026-11-13.

| Stage | Ours | Where others are better (evidence) | Where we are better (evidence) |
| --- | --- | --- | --- |
| Graph building | The agent writes stages, gates and stop rules as code and config. The triage of nodes was done by people and by trial | TI-0.1: agents write the hybrid MDP and the reward for each new task "with minimal human supervision". GaP: LLM edits to the graph are accepted or rejected on 100 fresh sim instances. ENPIRE: the agent also builds reward, reset and safety | Our graph has a learned RL node for the precise part and gates with measured limits. GaP has no RL; TI-0.1 has no precise node |
| Sim fidelity loop | Hand-led identification: arm friction and gravity (tool error 8.5 to 4.4 mm), fingertips, camera pose, then failure-driven fixes (encoder offsets, left-arm lead, mating force). Not in the default sim: the compliance of the arm and grasp (sim 0.02 to 0.04 mm/N, rig 0.19 to 0.24 mm/N on the way down; since 2026-11-15 an option that is off by default, fitted at 0.22 mm/N in joint 1), a friction-type mating force, a measured port friction, joint stick-slip, the real PC size (top face 102.5 x 97 mm in the sim, 103.5 x 103 mm real) and panel grey level | TI-0.1 and RoboSimStudio: agents reconstruct the scene from recordings or a phone video. Moonlake: calibration in about 1 hour from a few trajectories, and a measured sim-real ranking (r = 0.93, 15 policies). ASID: an exploration policy built to identify parameters from one real episode | Our fidelity is driven by the failing node, at the scale the node needs (0.1 mm contact, joint stick-slip). TI-0.1 says its reconstructions are not fine enough for high precision |
| Sim as an evaluator | Acceptance cells (n = 1,024) and a replay of real hand-overs. The replay "ranks policies; it does not predict real rates". The two disagree for K7: it passes every acceptance cell and is below K4 on 3 of 5 real hand-overs without the step alignment (n = 256 each) | Moonlake, PolaRiS, SimFoundry: sim score against real score measured over many policies (r = 0.90 to 0.93) | Ours has replays of real starts, real gates and real frames, which they do not report |
| Perception | ZED estimate for the pick; wrist nets; a classical port-vision measure of plug against port in one image. Wrist net "up" bias +0.5 to +3.6 mm (n = 5 runs, 2026-11-13). The ZED PC net (v1) is 3 to 8 mm low in y on real frames, and its gate passed a 16.3 mm error. A classical case-edge fit is within 1.12 mm of the arm-derived port (n = 41 runs); it is not on the arm yet | VLA methods need no hand-built perception. IndustReal and AutoMate report a full perception-to-insert chain over 120 to 200 trials | The direct plug-against-port measure removes encoder and grasp error from the target (hand-over within 0.8 mm, n = 14). It also absorbed the 3 to 8 mm PC error on every run |
| Alignment | Step-and-measure with the camera: 10.1 to 13.9 s on the rig (n = 4). Before: median 59.7 s | RLT: the whole critical phase takes seconds and is faster than teleoperation. HIL-SERL: USB grasp-insertion cycle 6.7 s. GaP: about 30 s per USB-C insertion in total, at n = 130 | Ours needs no real training time. The speed gap to GaP is closed for this part at n = 4; the gap to RLT and HIL-SERL stays |
| Insertion policy | K4: sim RL, zero real demonstrations, no force sensor; 12 of 12 seated, under 0.1 mm clearance. Depth not verified: fully in by the operator on 4 runs, a band of about 2 mm on both runs of 2026-11-13 | HIL-SERL: 100 of 100 on RAM and USB. VLA-Precision: 177 of 180 at ±0.25 to 0.5 mm. InsertAnything: 20 of 20 at 0.02 mm with fingertip force. SPARR: 95 to 100 % with 0.5 h of real RL and force/torque. All have much larger n and full depth | Zero demonstrations, zero real RL time, no force or tactile sensor. RLT needs hours of demonstrations and reaches 65 % on its sub-millimetre task. Most published zero-demonstration results without force stop near 0.5 mm clearance; one reaches about 0.1 mm for a cylinder (Marougkas et al., 10 of 10) |
| Verification on the rig | Operator go per motion, gates, dry run, weights hash, full logs. 15 starts at the milestone and 8 on 2026-11-13, one placement. No true depth measure. A gate replay on 37 real runs: no safety gate stopped a fine run or missed a failure | Moonlake: locked, hashed test layouts projected on the table, five difficulty tiers. HIL-SERL: 100 trials per task. SPARR, InsertAnything: 20 trials per condition over many parts | Our gates stop unsafe hand-overs before contact. The price is false stops: every stop of a fine run came from four quality gates |

**The honest summary.** Our evidence is one task, one placement and n = 12, with the depth not verified. Their evidence is wider and larger. What is new in ours is the combination: an agent-built graph, a sim-trained RL node at under 0.1 mm, no real demonstrations, no force sensor.

## What to take from each paper

Nine items, ordered by gain for cost. The expected gains are estimates from our logs, not measured results. The Status line in the first and fifth row gives what the rig showed on 2026-11-13.

| From | What to take | Where in our graph | Expected gain | Cost | Test |
| --- | --- | --- | --- | --- | --- |
| Mayr et al. | Tune the numbers of the fixed graph in sim with an optimiser (CMA-ES or Bayesian optimisation), scored on replayed real logs | Alignment (step sizes, reach test, timeouts, frames), final push (force, gates), hand-over gates | Was: alignment 59.7 s to about 15 s. **Status: reached without an optimiser.** Numbers set by hand from the step replay gave 10.1 to 13.9 s on the rig (n = 4). What is left for an optimiser: the step timeouts (7 of 23 steps) and the false stops | 2 to 3 days of code; CPU only | Dry runs plus a rig session: wall time and hand-over error on 10 runs |
| Moonlake | A locked, hashed set of test placements, and a measured sim-against-real score per node | Verification; the gate in sim | We learn whether our sim acceptance cells predict the rig. Today they do not predict rates | 1 day to define; 1 rig session of 20 to 30 runs | Rank correlation between sim cell score and real outcome per placement |
| TI-0.1 | The contact-model sensitivity test: freeze a policy, change the pad stiffness, travel and friction, and report the drop | Gate in sim for K4, K7 and the pick | Catches a policy that rests on sim-only contact (their case fell from 84.5 % to 2.6 %) | Half a day; GPU eval only | K4 and K7 across pad models, friction ranges, fit and contact constants, n = 1,024 per cell. Pass: no cell drops more than 5 points. **Status: run on 2026-11-15, failed.** Port friction 0.35 to 0.60: K4 94.6 % to 33.9 %, K7 98.7 % to 53.0 %. The pad model costs 3 to 8 points. The real seat times fit a sim friction of 0.20 or less (n = 14 real insertions) |
| GUAPO | The triage rule: learn only inside the region where perception is uncertain; hand over at its edge | Node triage; the hand-over gap (6 mm today) | A principled hand-over point instead of a tuned one | 1 day analysis | Sweep the hand-over gap in sim (3, 6, 9, 12 mm) with the real camera error; pick the largest gap that keeps 98 % |
| GaP and InsertAnything | A force threshold for "fully in" (GaP confirms depth at 30 N). We have no sensor, so use the joint-torque estimate with friction removed | Final push; seat detector | Was: full depth. **Status: tested on the rig, failed.** The push reached 10 N and 14 N by the estimate (n = 2) and a band of about 2 mm stayed. We do not know the force that the real port needs; GaP inserts below 10 N on its own connector and confirms at 30 N. The force estimate itself is good: 10.4 N and 14.3 N for 10 N and 14 N commanded (n = 1 each), and the fully-in runs had 9.0 to 9.7 N (n = 4) | 1 rig session, done | Next: the insertion and extraction force by hand on a scale (no robot), then a position-push against force-push comparison |
| Sequential Dexterity | A transition feasibility function: from the hand-over state, will K4 seat? Learned in sim, checked on real hand-overs | The alignment-to-K4 gate; the retry decision | Replaces the hand-tuned 3.5 mm disagreement gate (3 false stops in 27 hand-overs in the gate replay). **Status: less urgent.** A fixed offset on the wrist net gives 0 false stops in 27 in the replay and none in 3 rig runs. The margin is small: the raw bias was +0.5 to +3.6 mm over 5 runs | 2 days; sim data exists (4,096 saved insertion starts) | Offline on the logged hand-overs: it must pass every one that seated |
| ASID | Identification moves chosen to excite one parameter: a slow joint 2 and 3 sweep for stick-slip, a press against the port for mating force and for the compliance of the arm and grasp | Sim fidelity step | A stick-slip model (the cause of the servo sway) and a compliance value (0.25 to 0.29 mm/N on the rig, n = 2 force pushes; the plant fit of 2026-11-15 on 10 position pushes gives 0.13 to 0.24 mm/N). A press does not give the mating force: the encoders cannot identify it. The approach jump is no longer in this row: its main cause was a number that the code did not pass, and a scripted fix removed it (0.21 to 2.86 mm, n = 4) | 1 rig hour + 2 days modelling | The sim reproduces the spring-back (1.9 to 3.4 mm) and the servo sway |
| SPARR | A small real-world residual on the sim policy, with no human demonstrations (0.5 h per task in their case) | K4 node, only if sim fixes stall | They report 65 to 80 % zero-shot to 95 to 100 % | High for us: needs an automatic reset and a success signal; no force sensor | Only after the depth and placement tests. See the later section |
| RLT | Chunked actions and a penalty toward a reference action for any real fine-tuning; the idea that RL should also make the node fast | K4 real fine-tune, if done | Speed: their critical phase is about 3 times faster | Same as SPARR | Same |

**What not to take.** A VLA for the whole task (RLT, π\*0.6): it needs hours of demonstrations per task, which the operator has ruled out. Human interventions during training (HIL-SERL, ConRFT, VLA-Precision): the same reason. Agent-built scene reconstruction (TI-0.1, RoboSimStudio): our scene is already built, and their fidelity is not yet enough for our node.

## The fine-grained node

A precision mechanism is needed at every point where the task tolerance is smaller than the error of the stage before it. In our graph that is two points: the camera alignment and the K4 insertion. Together they form a funnel, and each stage must end inside the capture range of the next.

**The funnel, with our numbers.**

| Stage | Error at its end | Capture range of the next stage |
| --- | --- | --- |
| Open-loop arm motion (transport, approach) | 4.4 mm held-out tool error, plus an uncommanded move of 0.21 to 2.86 mm (n = 4; up to 5.9 mm before the fix), plus encoder offsets of 0.4 to 1.3 degrees, plus a PC estimate 3 to 8 mm low in y | Alignment accepts a start gap up to 12 mm |
| Camera alignment (plug and port in one wrist image) | across and up within 0.8 mm, gap 6 ± 1 mm (n = 14, real; the same for the fast alignment, n = 4) | K4 in sim: 98.2 % from a 4 mm ring, 97.9 % from 5 mm (n = 1,024) |
| K4 insertion | seated, under 0.1 mm clearance; depth not verified | The final push needs the hand within 2.5 mm of the axis (4.5 mm with centring) |
| Final push | position push: fully in on 4 runs by the operator. Force push: a band of about 2 mm on 2 of 2 | Release |

**The general mechanism.** Four parts, in this order.

1. *A relative measurement in the task frame.* Measure the tool against the target directly, in one image, so that arm calibration and grasp error cancel. Without it, the earlier policy (absolute pose inputs) scored 21.8 % in a sim test with the real offsets.
2. *A hand-over inside the capture range.* The capture range of the contact policy must be several times the measurement error. Ours: camera error sigma about (0.4, 0.95) mm against a capture ring of 4 to 5 mm.
3. *A contact policy trained in simulation on a real-like plant.* The seven ingredients are below.
4. *An independent check and a force-limited finish.* A seat detector that does not use the policy's own belief, then a push with a force limit along the port axis.

**The seven ingredients of the contact policy without real demonstrations.**

| Ingredient | What it is | Our evidence (sim unless marked) |
| --- | --- | --- |
| Relative observations | Pose relative to the hand-over pose, never absolute | Removes the constant encoder offset. At ±3 degrees offsets (twice the training range) the loss is 3.7 to 6.1 points |
| A vision prior | One camera measurement of plug against port, held during the insertion | Blind relative policy from a 4 mm ring: 39.1 %. With the prior: 93.0 % (n = 512), later 98.2 % |
| A contact head | An extra output trained on sim contact labels: free, touching, inside | 95.9 % of edge points right. On the rig it is our only contact sensor: it showed the real port mouth at +1.5 to 2 mm |
| History | 0.6 s of observations through a small temporal network | Needed for the push and the stall. It does not locate the port (probe R² about 0) |
| A real-like plant | Measured latency, joint friction, encoder offsets, and the real hand-over state as the start distribution | A replay of 5 real hand-overs is an acceptance test: 91.8 to 96.5 % (n = 256 each) |
| Masked inputs | Remove every input that the sim gets wrong | The left-arm lead: 0 % with the real values, 95.7 to 97.7 % after masking |
| A mating force | A force that resists the plug in the port, random per episode | K4 at 10 N: 0 %. K6d trained with 0 to 10 N: 90.0 % (n = 512). K7 (also the wide PC box): 99.1 % (n = 1,024). Not on the rig. The real force is not measured, so we do not know that 0 to 10 N covers it. The arm logs cannot identify it (fit error flat from 0 N to a plug that never moves). K8 added friction and fit ranges to K7 and lost the 8 mm start cell (49.6 to 72.3 % against 97.1 %, n = 1,024): no candidate |

**Compared with RLT.** The structure is the same: a general policy brings the tool near, a small RL policy does the critical phase, and the RL policy starts from a reference. The source of the reference differs. In RLT it is a VLA trained on hours of teleoperation, and the RL runs on the real robot with a person labelling. In ours it is geometry plus one camera measurement, and the RL runs in sim. RLT's side has no contact sim gap and becomes fast. Our side needs no human time per task, but it carries the sim gap. On 2026-11-13 we wrote that our depth shortfall is exactly the mating force that the sim did not have. That is not shown. What the data show: 10 to 14 N by our estimate do not remove the band (n = 2), the sim arm and grasp are about 10 times stiffer than the real ones, and the real insertion force is unknown. Since 2026-11-15: the fully-in runs had less axial force (9.0 to 9.7 N, n = 4) than the 14 N push that stayed short.

**Where a small amount of real RL could help, and its cost.** Two places: the last millimetres of depth, and speed. The published costs are 400 to 1,000 episodes (RLT) or about 0.5 hours with a force/torque sensor (SPARR). On our rig a real RL loop needs three things that do not exist yet: (1) an automatic reset, which means a scripted extraction of the plug; (2) an automatic success signal, and the 2026-11-13 data show that the arm-side depth cannot give it; (3) a force limit; the new torque model reads the axial force to 1.29 to 1.35 N (95th percentile, held out by date), so this one is nearer. Our estimate, not a measurement: with a 20 s insert-and-extract cycle, 400 episodes take about 2.2 hours of rig time and 400 mating cycles on one port. The cheaper first steps are a measured real insertion force and a depth signal. The mating force in sim (K7) is trained and waits for both.

## Node triage

One node of our nine needs RL: the insertion. Every other node is better as scripted logic with classical or learned perception, and its numbers should be tuned by an optimiser. The rule below gives this answer from measurements, and the agent can apply it to a new graph.

**Six measurements per node.** All come from the run logs or from one sim evaluation. The sixth was added on 2026-11-15.

| Measurement | Definition | How to get it |
| --- | --- | --- |
| Tolerance ratio R | Tolerance the node must reach, divided by the error of the best direct measurement available inside the node | Gate limits and the measured sensor error |
| Contact share | Does contact below the sensing resolution decide the outcome? (yes or no) | Logs: is the node's result set while the parts touch? |
| Variety | Number of object types, scenes or instructions the node must handle | The task description |
| Sim reproducibility | Does the sim reproduce the node's real failure, and does the node's sim score survive a change of contact model? | Real-log replay; contact-model sensitivity test |
| Priority | Share of failed starts and share of run time | Session table |
| Result observability | Can the node's result be measured without the operator, at the tolerance? (yes or no) | Is there a logged signal that separates the operator's labels? Test it on the labelled runs |

**The rule, applied in this order.**

0. The result is not observable: build a measurement for the node first. No other class is chosen before it.
1. Variety is high and the tolerance is loose (above about 5 mm): a VLA or another learned general policy.
2. R is 1 or more: the error can be measured, so measure it and correct it. Classical vision with scripted steps or a servo.
3. R is under 1 and contact decides the outcome: RL. In sim if the sim reproduces the failure mode; with a real residual if it does not.
4. Everything else: scripted, with its numbers tuned by an optimiser.
5. Priority orders the work. It does not choose the class.

This is GUAPO's rule made measurable (learn inside the region where perception is uncertain) plus the sim check from TI-0.1. A limit from the look-back of 2026-11-14: the measurements choose the node class. They did not choose the fix class in any of 13 past failures; the cause analysis did. Only sim reproducibility changed a decision.

**Result observability today.** Yes for seven nodes (pick held, press latched, hand-over error, seat detected: all logged). No for the final push: axial force, compliance, arm-side depth and the contact head do not separate fully in from short (n = 4 fully in, n = 4 to 7 short). In part for the K4 insertion: the seat is detected, the plug depth is not; in a fitted plant with a 10 N mating force the detector fires with the plug 1.4 to 2.6 mm short (sim, 99.8 % of detections, n = 1,024). Sim reproducibility of the K4 node is also weaker than we wrote: its sim score does not survive a change of port friction (33.9 % at 0.35 to 0.60).

**The worked table.** Time is from the fastest milestone run (164.3 s, `fastmove_1`). Failures are the 5 of 15 milestone starts that did not finish. The numbers of 2026-11-13 are in the evidence and action columns. No class changed.

| Node | Time share | Failed starts | Real evidence | R | Contact decides? | Class by the rule | Action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Capture moves + ZED estimate | 15 % (24.1 s) | 0 (one camera fault, not counted) | Reader estimate good enough for the pick servo on 19 picks. The PC estimate (v1 net) is 3 to 8 mm low in y on real frames; the alignment absorbed it | above 3 at the spot (the alignment accepts 12 mm); not away from it (16.3 mm error at one mapping pose) | no | Scripted + perception | Classical case-edge fit for the PC x and y (within 1.12 mm, n = 41 runs, offline), the net as its start and gate. Not on the arm. No further net retrain: zed_v6 is worse on real frames (x -7.7 mm, y +7.2 mm, n = 41 runs) |
| Servo pick | 13 % (20.9 s) | 1 | 18 held of 19 logged picks. The empty close: hand 7 mm high at the close (-17.7 mm against -23.1 to -25.1 mm) | about 1 to 3 | partly (fingertips press on the table) | Classical vision + scripted | Add a height check before the close (scripted). No RL |
| Left hold | 5 % (8.1 s) | 0 | Latched on 18 of 18; press 18 to 22 N for a 25 N command | above 3 | no | Scripted | None |
| Transport | 4 % (6.0 s) | 1 (software end) | Fixed time | above 3 | no | Scripted | None |
| Approach to an 18 mm gap | 8 % (12.8 s) | 0 | Milestone: uncommanded jump over 3 mm on 7 of 18. Cause: 6 servo iterations at the stop in place of 4 and 2, with joints 2 and 3 sticking. 2026-11-13 with the fix: 0.21 to 2.86 mm (n = 4) | above 3 (target is a 12 mm window) | no | Scripted | Done with scripted logic. No RL |
| Wrist-camera alignment | 37 % (61.6 s) | 3 | Milestone: ends within 0.8 mm (n = 14); 123 of 229 steps time out with the hand at the goal; false stops by a 3.5 mm gate against a net bias of about +2.6 mm. 2026-11-13: 10.1 to 13.9 s (n = 4), about 9 to 12 % of a 117 to 122 s run; no disagreement stop in 3 runs with the net offset; one new stop from a light patch in port vision | about 1 to 2 (1 mm tolerance, 0.25 to 0.74 mm pose residual) | no | Classical vision + scripted, tuned | Numbers tuned by hand from the replay: done. Port-vision shell reference: validated on 4,824 real frames, not on the arm. RL alignment reached 53 % in sim training against 99.6 % for the servo: do not use RL here |
| K4 insertion | under 2 % (1.1 to 1.7 s) | 0 of 12 | Seated 12 of 12; depth not verified | under 0.2 (0.1 mm clearance, about 0.4 to 1 mm camera error) | yes | **RL in sim** | Keep K4. K7 (mating force, wide PC box) is trained; it goes on the arm only after its replay gap to K4 is explained. K8 (friction and fit ranges) gave no candidate. Open risk: the port friction (sim only, see the 2026-11-15 block) |
| Final push | 4 % (about 6 s) | 0, but skipped on 2 of 8 | Position push: adds 0.6 to 1.0 mm at rest (n = 5), moves the hand 5.5 to 6.3 mm off its line; the 4 fully-in runs used it. Force push (10 N, 14 N; n = 2): 2.6 and 3.9 mm under the load, 0.71 and 0.54 mm at rest, band about 2 mm | under 1 (depth is not measured) | yes | Not decided. The node has no measurement to triage with | The scripted force push failed. Before any RL: measure the real insertion and extraction force and get a depth signal. Then compare the position push and the force push on the rig. A lateral force is a hypothesis from the logs (5.4 to 5.9 N on the fully-in runs, 1.8 to 3.3 N on the short pushes, n = 4 each, confounded with push type and PC spot) |
| Release + home | 14 % (23.2 s) | 0 | Fixed time | above 3 | no | Scripted | None |

**Reading the table.** The class column and the priority column point at different nodes. The RL node (K4) has no failures and takes 1 % of the time. At the milestone the node with most failures and most time (alignment) was a scripted node with badly tuned numbers. Tuning it worked: the 2026-11-13 runs confirm that the gain was in the graph, not in more RL. The priority has now moved to the final push. That node falls under rule 0: nothing in it measures the depth. So the next step there is a measurement, not a policy. The force budget of 2026-11-15 agrees: more axial force is unlikely to help, and the logs cannot give the mating force.

## Optimising the graph itself

On 2026-11-13 we estimated that optimisation of the graph's numbers can remove about 45 s of the 164 s run and up to 3 of the 5 failed starts. The rig then gave 117 and 122 s (n = 2) with numbers set by hand from the replays. It needed replayed real logs, not RL and not yet an optimiser. RL over the graph (which stage next, retry or stop) is a later step, and it starts as a simple classifier.

**How much there was to gain, and what the rig gave.** The first two items were estimates on 2026-11-13. The measured values follow each one.

- Time: the alignment was 61.6 s of 164.3 s. Our offline model on 15 real runs gave 14.3 s (range 11.9 to 18.4) with a new reach test, no probe cycles, larger gap steps and 3 frames. Eight sim dry runs aligned 8 of 8 at 16.0 s median. **Measured on the arm: 10.1 to 13.9 s (n = 4).** The model was a little slow, and inside its range for 2 of the 4 runs.
- Failures: of the 5 failed starts, 2 are a gate limit set 0.01 and 0.04 mm too tight against a known bias, 1 is a post-press budget, 1 is a pick height that a check can catch, 1 is software. Three of five are numbers. **Measured on the arm:** with the net offset, no disagreement stop in 3 runs; the run without it stopped at 3.56 mm. The count is too small for a rate.
- Published support: Mayr et al. tuned behaviour-tree numbers in a randomised twin and went from about 20 % (no search) and 53 % (random numbers) to a median of 100 % on a real peg-in-hole (15 trials per policy).

**What can be optimised, and with what.**

| What | Examples in our graph | Method | Where it runs |
| --- | --- | --- | --- |
| Continuous stage numbers | Step size, reach test, timeout, frames per measurement, settle time, push force, move speeds | Black-box optimiser (CMA-ES or Bayesian optimisation) over 10 to 20 numbers; score = time + a large penalty per failure | Sim with real-log replay; then 10 confirming rig runs |
| Hand-over conditions | The 6 mm hand-over gap; the 3.5 mm source-disagreement gate; the 2.5 mm push gate | A transition feasibility function: the chance that the next node succeeds from this state, learned in sim and checked on logged real hand-overs | Sim (4,096 saved insertion starts) + the 14 logged hand-overs |
| Discrete choices | Step or servo alignment; position or force push; K4 or K7 | A/B on the rig with a continuous outcome (time, depth in mm) | Rig |
| Retry and stage choice | After a stall: push on, back off and align again, pick again, or stop | A rule first. Later a small graph-level policy over the gate values (an options or semi-MDP policy) | Sim with failures injected at the real rates |
| Stage start times (the schedule; added 2026-11-15) | Both arms to the capture pose at once; both arms home with the right arm 3 s after the left | A table of what each stage needs and which arm it moves, then a clearance check. Start times only: no speed, path or safety gate changes | Offline on logged joint paths, then paired dry runs, then 3 rig runs after approval |

**Five kinds of real-log replay.** All five exist now.

1. *Step replay* (exists): every logged alignment step is run again through a new reach test and timing model. It gave the 14.3 s estimate.
2. *Hand-over replay* (exists): K4 starts in sim from each logged real hand-over state. It ranks policies. It does not predict real rates.
3. *Gate replay* (exists since 2026-11-14): every gate limit is evaluated on every logged run. 35 gates on 37 real runs. The output is a false-stop and a missed-stop count per gate. No safety gate stopped a fine run or missed a failure. Every stop of a fine run came from four quality gates: the 3.5 mm disagreement gate (3 false stops in 27 hand-overs), the push lateral gate read at one tick (2 false skips in 13 seats), the scale gate (1) and the push gripper watch (1).
4. *Frame replay* (exists since 2026-11-14): a perception change runs again on every saved real frame. Port vision: 4,824 align frames of 44 runs, 0 newly rejected. PC edge fit: 89 capture frames. This is the gate for a perception fix that the sim cannot render. It rejected zed_v6, which the sim twins passed (2026-11-15).
5. *Path replay* (exists since 2026-11-15): the logged joint paths of both arms are shifted in time and checked for clearance. 29 to 32 real runs per overlap. It passed two overlaps (capture: 116.4 mm or more, n = 32; home with a 3 s lead: never closer than serial, 31 of 31) and rejected one (the left swing during the pick: 5 to 27 mm, n = 29). Expected saving about 12 s per run. Not on the arm.

**Limits that the optimiser must respect.**

- *The sim must contain the effect the number depends on.* The alignment numbers depend on joint stick-slip, which the sim does not model. An optimiser will find numbers that work only in the sim. TI-0.1 saw the same with a learned policy (84.5 % to 2.6 %). So: score every candidate on a set of friction and contact models, and on the step replay. Two new cases from 2026-11-14/15: the sim arm and grasp give 0.02 to 0.04 mm/N where the rig gives 0.19 to 0.24 mm/N, so a dry run cannot test an unload change; and the sim PC panel is light grey, so the edge fit locks 13 to 20 mm off in the sim and works on real frames.
- *The rig cannot resolve success rates.* To tell 70 % from 90 % with the usual 5 % error and 80 % power takes about 59 runs per option. One session gives 15 to 20 runs. So rig tests must use continuous outcomes: seconds, millimetres of depth, millimetres of hand-over error.
- *A sim score can rest on a guessed contact value.* K4 seats 94.6 % at the training port friction and 33.9 % at 0.35 to 0.60 (n = 1,024). Robustness can cost another cell: K8 gained the friction cell and lost the 8 mm start cell. So a candidate is scored against a fixed rule that was written before its results, on all cells.
- *The dry run has a known artefact at the release.* The sim jaws at 24 mm pull the plug out in 28 % of worlds (n = 906); at 27 mm in 0.1 % (n = 898). A dry-run option for 27 mm is proposed. Until then "not seated after the release" in a dry run is not evidence about a policy.
- *A schedule dependency includes what the cameras and nets need*, not only geometry. zed_v5 used the left gripper as a landmark, so the pose and opening of the other arm are inputs of the PC estimate.
- *Safety gates are not tunable by the optimiser.* Each gate is marked as safety or quality. The optimiser may move quality gates. A safety gate changes only with the operator's approval, as today.

**When RL over the graph is worth it.** A graph-level policy sees a few numbers (gate values, attempt count) and picks among a few actions. It is worth training only when a rule is not enough, which means when the best action depends on several gate values together. Today the retry has one known trap: in a sim dry run, a stall turned the plug 2 degrees in the hand, and the second alignment then failed. A rule covers this: after a stall, measure the in-hand pose again before the retry. We do not have enough real failures (5) to train or to test more than a rule.

## Do we need more RL or VLA parts?

Mostly no. The one RL step in sim that we called justified (the mating force) is done as K7; it waits for a real force measurement and for its replay gap. Everything else is no or later.

| Candidate | Answer | Reason | Evidence from our logs | Evidence from the papers |
| --- | --- | --- | --- | --- |
| A VLA for the coarse stages (capture, pick, transport, approach) | **No** for this task. Later, if the task set grows to many objects | The coarse stages are not where we fail, and a VLA needs demonstrations | Coarse stages caused 2 of 15 failed starts (1 pick, 1 software). Pick held 18 of 19 | Moonlake on the same YAM arm: π0.5 with 50 teleoperated episodes plus sim RL reached 48 % grasp success. RLT needs hours of demonstrations per task. The operator ruled out manual demonstrations |
| RL residual on the approach | **No** | The approach error was a scripted fault plus joint stick-slip. A scripted fix removed most of it. The alignment node absorbs the rest | Milestone: jump over 3 mm on 7 of 18 approaches, none of them a failed start. With the fix: 0.21 to 2.86 mm (n = 4). RL alignment reached 53 % training success in sim against 99.6 % for the servo | GUAPO: residual RL alone scored 0 % on its peg task; switching by uncertainty scored 93 % |
| RL for the pick | **No** now. Decide again after the placement test | One failure, with a measurable cause that a check catches | Hand 7 mm high at the one empty close. In sim the servo descent scored 99.61 % (n = 2,048) and a learned pick (ACT) 84.9 % | IndustReal reports pick 158 of 160 |
| RL in sim with a mating force (K6d, then K7) | **Done in sim. Not on the arm yet** | We do not know that the depth shortfall is a mating force: 10 to 14 N from the scripted push left the band (n = 2), and the real force is not measured | K4 fails at 10 N in sim (0 %); K6d seats 90.0 % (n = 512); K7 99.1 % (n = 1,024). K7 is below K4 on real hand-overs without the step alignment (54.3 against 80.1 % on one run, n = 256) and equal after it. Dry runs with K7: 4 of 4 seat detected, 2 of 4 not seated after the release; the same seeds with K4 give the same result, and the cause is a sim artefact (2026-11-15). K8 (three variants) fixes the friction cell (91.8 to 93.8 %, K7 53.0 %) and fails the 8 mm start cell (49.6 to 72.3 %, K7 97.1 %, n = 1,024): no candidate | InsertAnything and GaP both finish against a force threshold, with a force sensor |
| Real-world RL fine-tuning of K4 | **Later.** The first condition is now met (the force push stops short), the others are not | It needs an automatic reset, a true depth signal and a force limit. The first two do not exist; the force estimate is now good to about 1.3 N. The depth signal is the hard one: no arm-side threshold separates fully in from short | No real K4 failure to learn from: 12 of 12 seated | SPARR: 65 to 80 % to 95 to 100 % with 0.5 h of real RL, but with a force/torque sensor. RLT: 400 to 1,000 episodes with a person labelling |
| A learned hand-over check (feasibility function) | **Later** (was: yes, now) | A fixed net offset removed the false stops for now. A learned check is the next step if they come back | Gate replay: 3 false stops in 27 hand-overs at the 3.5 mm limit; 0 in 27 with the offset. Rig: no disagreement stop in 3 runs. The raw bias varied +0.5 to +3.6 mm (n = 5 runs), so the fixed offset has little margin | Sequential Dexterity: real 12 of 20 against 3 of 15 for the baseline |
| A learned graph-level policy for retries and stage choice | **Later** | Too few real failures to train or test it; a rule covers the known case | 5 failed starts, 4 different causes | Plan-Seq-Learn: the stage end condition matters (100 % against 69 % without it), but theirs is written by an LLM, not learned |

## New ideas

Eight ideas of our own. Each has a test and a result that would prove it wrong. Status on 2026-11-15: two are falsified (1 and 2), one is weak (5), two had a look-back test that did not falsify them and did not prove them (3 and 8), one is done in its new form with a stated limit (4), and two wait for the rig (6 and 7). Ten more ideas are in `AGENTIC_RL_IDEAS_20261114.md`; two of them ran on 2026-11-15 (the small plant and the force budget), and their results are in the 2026-11-15 block above.

| # | Idea | Experiment | Cost | Falsified if | Status (2026-11-15) |
| --- | --- | --- | --- | --- | --- |
| 1 | **Size the funnel from the capture map.** The alignment tolerance (1 mm per axis) was set by hand. K4's vision prior is the hand-over measurement itself, so a looser stop does not make the prior worse. Stop the alignment as soon as the measured offset is inside K4's capture range with margin (about 2 mm) | Sim: K4 from a 2 mm ring with the real prior error, n = 1,024. Rig: 10 runs at a 2 mm tolerance | Half a day sim, part of one rig session | Sim rate under 98 %, or 2 or more of 10 real runs fail to seat, or the seat time doubles | **Falsified (sim and real replay, 2026-11-15).** 2 mm ring: K4 94.1 % and 90.9 %, K7 98.9 % and 96.8 % (two prior models, n = 1,024 each); the limit was 98 %. The seat time rises by 10 to 13 %, so that part of the rule holds. Replay of 23 real alignment stages: the same stop cycle in 19, 0.2 to 1.2 s saved in 4. The 10 rig runs are not needed. The 1 mm stop stays |
| 2 | **The contact head as a depth sensor.** The head's change from "touching" to "inside" marks the real port mouth (+1.4 to +3.2 mm, n = 9). Measure depth from that mark, not from the camera gap | Offline on the logged K4 insertions: does depth-from-mouth order the six 2026-11-11 runs by their visible plug band better than the present depth? | 1 day, no rig | Rank agreement with the band is no better than the present rest depth | **Falsified.** The flip depth spreads over 0.71 to 3.05 mm (n = 8 K4 seats). Depth from it agrees poorly with the operator (rho 0.37, n = 8). The present rest depth ranks the six hand-read frames perfectly (rho 1.00, n = 6) |
| 3 | **The reproduce-first gate.** RL retrain is allowed as a fix class only after the sim reproduces the real failure. Otherwise the class must be sim fidelity or scripted logic | Look back at the three retrains (K2, K3, K4) and at the stick-slip case. Did the sim show the failure before each retrain? | 1 day, no rig | A past retrain helped on the rig although the sim did not reproduce the failure | **Look-back done (2026-11-14): not falsified, weak test.** The premise was wrong: K2 and K3 did not go to the rig, so the record has one retrain line with a rig result (K4), not three. The sim showed its failure mode first. Four learning efforts without a reproduction have no rig success (two dead, two pending). The config fix for the same cause seated 3 of 3 against 10 of 10 for K4. "Reproduces" is now split into mode and rate (formula line 5). zed_v6 and K8 (2026-11-15) are two more learned fixes that did not pass |
| 4 | **A plant correction from real logs, in place of real RL.** Was: fit a small correction to the sim's joint response, with the spring-back as an example of a plant error. That framing was wrong. The spring-back is elastic deflection of the arm and grasp when the load drops (0.25 to 0.29 mm/N; rig 1.93 and 3.37 mm, n = 2). It is real behaviour that the stiff sim lacks. It is on the fully-in runs too, so it is not the cause of the short depth. New form: add one compliance value and stick-slip to the sim from the logs, before any learned residual | Fit the compliance on the 2 force pushes and 7 position pushes; hold out 3. Then evaluate K4 and K7 in the corrected plant | 2 to 3 days | The corrected sim does not reproduce the held-out spring-back within 30 %, or the policies trained in it score lower on the hand-over replay | **Done in the new form (2026-11-15), with a limit.** A 1-D plant fitted on 10 position pushes; held out: the 2 force pushes (not the 3 pushes of the plan). Held-out trace error 1.33 and 1.73 mm to 0.14 and 0.35 mm RMS. The sim with the compliance gives 3.2 to 4.6 mm of advance at 14 N (rig 4.03 mm). The source does not state the spring-back error against the 30 % limit. The mating force is not identifiable. Stick-slip is not added. The policy part of the falsifier (hand-over replay of a policy trained in the plant) has not run |
| 5 | **A camera depth label.** The visible silver band of the plug in the last wrist frame follows the depth (about 0.8 to 2.3 mm of band for 1.66 mm of depth, read by hand on 6 runs). A detector for the band is a true depth signal with no new hardware | Hand-place the plug at 0, 1, 2, 3 mm short (no robot), 10 images each, and fit the detector | 1 to 2 days | Band error above 0.3 mm against calipers | **Weak so far.** The band follows the rest depth for K4 (rho 0.86, n = 8) but does not separate fully in from short, and it fails across two policies (rho 0.30, n = 11). It needs a frame at rest. The hand-placed calibration is not done. By hand (2026-11-13): fully in shows no band, so a band detector has a clear zero |
| 6 | **Automatic extraction for a large n.** A scripted extraction (GaP uses a 2 Hz wiggle with a lateral force limit) lets the rig insert, extract and insert again without a person. The aim is statistics: n = 50 or more per placement in one session. It is also the reset that any later real RL needs | Rig: 20 insert-extract cycles at the known placement | 1 rig session | The grasp turns more than 2 degrees per cycle, or a cycle takes over 40 s, or the operator sees port wear | Not run. Needs the rig |
| 7 | **A tuned servo before any RL alignment.** The servo aligned in 2.9 to 7.8 s on the rig and failed only the sway limit. The sway is joint stick-slip. Put stick-slip in the sim, tune the servo gains with the optimiser, and train RL alignment only if that fails | Sim with a stick-slip model fitted to a real joint sweep; optimiser on 4 to 6 gains; then 5 rig runs in align-only mode | 1 rig hour + 3 days | The tuned servo still exceeds the sway limit (tip speed p99 15 mm/s) on the rig | Not run. Less urgent: the step alignment now takes 10.1 to 13.9 s (n = 4) |
| 8 | **The agent writes the triage table itself.** After every session the agent computes the triage measurements per node and proposes a class and a fix class | Blind test on three past sessions (2026-11-01, 11-02, 11-10): compare the agent's fix class with the one that worked | 2 days | Agreement under 50 %, or the agent proposes RL where a config change solved it | **Look-back done (2026-11-14): not falsified, not proven.** Agreement 11 of 13 failures (85 %); RL proposed in 0 cases. Only 4 of the 13 proposals were blind (3 of 4 agree). The rule "propose scripted or perception, never RL" also scores 11 of 13, so the test does not separate the agent from that prior. The measurements did not choose the fix class in any case. Next: a fresh agent with the raw logs only |

## Ranked next experiments

Re-ranked on 2026-11-15, after the second server round. The operator is away from the rig. So items 1 to 7 need no rig and come first. Items 8 to 16 wait for the rig; their order is the order to run them in.

**Closed since 2026-11-13.**

- *Gate replay*: done, 35 gates on 37 real runs. The learned feasibility check is not built; it is less urgent.
- *Depth from the contact head and from the plug band*: done. The first is falsified, the second is weak. No depth signal came out of it.
- *Depth session on the rig*: run in part. The force push failed its pass rule (fully in on 0 of 2). K6d was not run and K7 replaces it.
- *Fast alignment on the rig*: passed with n = 4, not the planned 10. Wall time 10.1 to 13.9 s against a 20 s limit, hand-over within 1 mm. One new stop appeared (the port-vision light patch).
- *Contact-model sensitivity of K4 and K7* (was item 1): done on 2026-11-15, n = 1,024 per cell. It failed its pass rule: port friction 0.35 to 0.60 costs K4 61 points and K7 46 points. The real seat times fit a sim friction of 0.20 or less (n = 14).
- *Funnel sizing* (idea 1): falsified in sim and on 23 replayed real stages. No rig runs.
- *K8, a friction- and fit-robust K7*: three variants, none passes (8 mm start cell 49.6 to 72.3 %, n = 1,024). No checkpoint.
- *Compliance in the sim* (the first half of the old item 3; idea 4): done as an option that is off by default. Held-out push error 0.14 and 0.35 mm RMS (n = 2). The real PC size and panel grey level exist as test patches only.
- *Force budget and the arm as a force sensor*: done. Axial residual 1.29 to 1.35 N (95th percentile). The push limit is 14 to 16 N. The mating force is not identifiable.
- *Cause of the zed_v5 offset and of the wrist-net bias*: found. zed_v6 is rejected (n = 41 runs). The wrist-net correction waits for approval.
- *The sim release effect*: explained as an artefact (28 % at 24 mm jaws, 0.1 % at 27 mm, n = 906 and 898).
- *Parallel schedule, offline part*: done. About 12 s. The rig part is item 10.
- *The reproduce-first look-back and the triage look-back* (ideas 3 and 8; was item 5): done. Neither is falsified; neither is proven.

**No rig needed.**

1. **Explain K7 against K4 on bad hand-overs, by a node swap** (sim replay, 3 days; idea 1 of the ideas page). K7 is below K4 on 3 of 5 real hand-overs without the step alignment, and K8 did not close the gap (`full_A2`: 55.5, 54.3 and 80.1 %, n = 256 each). Swap one node output at a time in the replay. Score the method on the 5 milestone failures with known causes. Pass: it names the known node on 4 of 5. This also tests the "by an intervention" words of formula line 5, which have support from two cases only. K7 does not go on the arm before this.
2. **The executed-config check and one outcome event per run** (1 day; formula line 7). Every setting of a config must appear in the run log as executed, and a new setting must change the logged behaviour in the dry run. Test: the check must flag the two approach keys on the logs of 2026-11-03 to 2026-11-11 (37 approaches). Add one `run_outcome` event with the failing node and a field for the operator's result: today an agent that counts `stopped` events finds 3 of the 7 failed starts of 2026-11-10.
3. **A depth measurement that does not need the operator** (sim 1 day, offline 1 to 2 days). This is rule 0 of the triage for the final push. Two parts: a wrist-camera pose after the release that sees the plug from the side (search in sim; pass: 5 px per mm or more within reach), and the band detector on a frame at rest. The calibration by hand waits for the rig.
4. **The plan for paired rig comparisons** (1 day, offline; formula line 8). From the logged outcomes (rest depth of 10 position pushes and 2 force pushes, alignment times): how many pairs does the depth A/B need, and which comparisons can one session not decide? If the depth A/B needs more than 20 pairs, item 3 comes before it.
5. **The dry-run release option in the sim IO** (under 1 day). Jaws to 27 mm at the release, as a config option that is off by default. Then the gate in step 2 of the loop can test a release. Pass: the 30 seeds of the parallel dry runs show "not seated after the release" only with a logged cause.
6. **A blind triage test with a fresh agent** (idea 8 again; 2 days). The agent gets only the raw logs of one session. Include the sessions of 2026-11-03 and 2026-11-06, where the cheap prior was wrong.
7. **Lower priority.** The cause of the 8 mm loss in K8 (only if the measured port force points to high friction). The parameter optimiser on the step replay and the gate replay (the step timeouts, 7 of 23 steps, and the small margin of the net offset). The real PC size and panel grey level as sim defaults.

**Waiting for the rig.**

8. **By hand, no robot** (`NEXT_RIG_20261115.md` step 0; 5 to 10 minutes). The insertion force and the extraction force on a kitchen scale, calipers on the PC case, a look at the light patch. The two forces are the most useful numbers for the depth problem and for the friction risk. If the plug needs more than about 14 N by hand, no setting inside the present limits can seat it. A high extraction force with a low insertion force points to spring retention with low friction.
9. **Depth A/B on the rig** (the same runbook): 2 runs with the slow unload, then 2 runs with the 15 mm position push, with the operator's band before and after the jaws open. If the force push leaves a band and the position push does not: 2 runs with the lateral force vector (9.5 N axial plus about 5.7 N lateral, config `bimanual_align10_forcevec`). The coordinator counts the lateral push as a new motion that needs approval.
10. **Parallel capture and home** (3 runs, after the operator approves one go for a two-arm move). Pass: mean saving of 12 s or more with no new stop.
11. **The PC edge fit on the arm**: first in shadow mode (log only), then in use. It needs approval. No new net: zed_v6 is not for the rig.
12. **The wrist-net grasp correction** (config `bimanual_align10_all_netgrasp`, needs approval): first a check with the reader re-seated 2 to 3 mm higher and lower in the jaws.
13. **K7 on the arm**: 3 runs at the current spot. Only after items 1 and 9.
14. **Placement test with a locked set**: for example 9 placements, 2 runs each, fixed before the session; the first box with today's nets is PC x ±10 mm, y -20 to +10 mm, yaw 0 to +6 degrees (`docs/real/ROBUSTNESS_PLAN_20261113.md` on `real/graph-align`). It needs item 11, because the v1 net saturates in y, and K7 for more yaw (sim K4 seats 58.6 % at +14 degrees, n = 256). This closes the largest gap in the claim. Report per-stage failures.
15. **Automatic extraction** (idea 6), then n = 50 at one placement. This turns "12 of 12" into a rate.
16. **Stick-slip identification and the tuned servo** (idea 7). **A real residual on K4** stays last: only after items 3 and 15 give a depth signal and a reset.

## Sources

Every external number in this document comes from a page fetched on 2026-11-13. All pages were read through a fetch tool that summarises the page, so quotes are short and a number in a figure may be missing. The three given sources, the RLT paper and GaP were read by the author of this page; the other papers were read by three research sub-agents with the same rule. Check a number against the PDF before it goes into a paper. "Read in full" means the full-text page was fetched; for long pages the tool read the first 100,000 characters, so appendices may not be covered.

**The three given sources.**

| Source | Status | Note |
| --- | --- | --- |
| [TI-0.1 Technical Report](https://research.tutorintelligence.com/ti0-1), 9 Nov 2026 | Read in full (two passes) | Its reference list (36 entries) was read; the entry behind the given anchor is Chi et al. 2023, Diffusion Policy ([arXiv:2303.04137](https://arxiv.org/abs/2303.04137)), which was not fetched itself |
| [Moonlake blog](https://moonlakeai.com/blog/simulation-enabled-scaling-paradigm), Oct 2026 | Read in full | Per-size numbers of the zero-shot chart could not be read from the figure; only ranges are cited. The text says 12 fine-tuned policies and the chart 15 |
| [RLT blog page](https://www.pi.website/research/rlt) | **Could not verify** | HTTP 429 on every attempt (five). All RLT claims here come from the paper |
| [RLT paper, arXiv:2604.23073](https://arxiv.org/abs/2604.23073) | Read in full (HTML) | Demonstration hours are garbled in the HTML ("11-1010"); "1 to 10 hours" is from an [alphaXiv overview](https://www.alphaxiv.org/overview/2604.23073), not confirmed in the PDF. Per-task success and speed are mostly in figures. Whether the +40 % and +60 % full-task gains are absolute points is not stated |

**Read in full.** [HIL-SERL](https://arxiv.org/abs/2410.21845), [SERL](https://arxiv.org/abs/2401.16013), [π\*0.6](https://arxiv.org/abs/2511.14759), [ResiP](https://arxiv.org/abs/2407.16677), [PLD](https://arxiv.org/abs/2511.00091), [ConRFT](https://arxiv.org/abs/2502.05450), [DSRL](https://arxiv.org/abs/2506.15799), [ResFiT](https://arxiv.org/abs/2509.19301), [VLA-Precision](https://arxiv.org/abs/2609.04355), [ForceRFT](https://arxiv.org/abs/2609.22840), [From Reach to Insert](https://arxiv.org/abs/2605.04649), [Learning Where it Matters](https://arxiv.org/abs/2607.15982), [IndustReal](https://arxiv.org/abs/2305.17110), [AutoMate](https://arxiv.org/abs/2407.08028), [FORGE](https://arxiv.org/abs/2408.04587), [TacSL](https://arxiv.org/abs/2408.06506), [TRANSIC](https://arxiv.org/abs/2405.10315), [RialTo](https://arxiv.org/abs/2403.03949), [SRSA](https://arxiv.org/abs/2503.04538), [SPARR](https://arxiv.org/abs/2602.23253), [NVIDIA blog](https://developer.nvidia.com/blog/the-machines-that-make-the-machines/), [Blind Precise Assembly](https://arxiv.org/abs/2610.23849), [ASID](https://arxiv.org/abs/2404.12308), [RoboSimStudio](https://arxiv.org/abs/2610.18835), [GAUGE](https://arxiv.org/abs/2608.05948), [RPG](https://arxiv.org/abs/2610.02204), [Success Guided Sampling](https://arxiv.org/abs/2610.12465), [AlphaNIST](https://arxiv.org/abs/2610.13354), [PolaRiS](https://arxiv.org/abs/2512.16881), [InsertAnything](https://arxiv.org/abs/2609.24511), [PACE](https://arxiv.org/abs/2609.07534), [Rebar insertion](https://arxiv.org/abs/2609.20477), [Touch2Insert](https://arxiv.org/abs/2603.03627), [Eureka](https://arxiv.org/abs/2310.12931), [DrEureka](https://arxiv.org/abs/2406.01967), [RoboGen](https://arxiv.org/abs/2311.01455), [Plan-Seq-Learn](https://arxiv.org/abs/2405.01534), [ManipGen](https://arxiv.org/abs/2410.22332), [GUAPO](https://arxiv.org/abs/2005.10872), [Residual RL](https://arxiv.org/abs/1812.03201), [Mayr 2021](https://arxiv.org/abs/2109.13050), [Mayr 2022](https://arxiv.org/abs/2203.10033), [GaP](https://arxiv.org/abs/2607.05369), [ENPIRE](https://arxiv.org/abs/2606.19980), [EmbodiedSWE](https://arxiv.org/abs/2609.27308), [CaP-X](https://arxiv.org/abs/2603.22435), [Agentic RSR](https://arxiv.org/abs/2610.10479), [Sequential Dexterity](https://arxiv.org/abs/2309.00987).

**Read in part.** [SimFoundry](https://arxiv.org/abs/2606.28276) (first half).

**Known gaps inside the sources that were read.**

- ENPIRE: the "99% success rate" has no task, n or agent attached in the text. Not used as evidence.
- Mayr 2022: the text does not say whether the 96 / 41 / 52.2 % rates were measured on the real robot.
- TacSL: the clearance could not be read (the HTML shows "55 mm").
- TRANSIC: the 0 % direct-transfer value comes from an appendix table with garbled headers.
- AutoMate, From Reach to Insert and ManipGen each have a mismatch between text and table. The table values are cited.
- VLA-Precision: the intervention rate is ambiguous in the text and is not cited.
- Blind Precise Assembly: the local PDF could not be opened on this machine (no PDF tool); the numbers come from the arXiv page.

**Dropped because they could not be verified.** A ResearchGate paper on residual policy adaptation for VLA assembly (68.4 to 95.1 %); RLT's median episode lengths; the full-text numbers of LEAGUE, BOSS, GenSim, BeBOP and SimEX (abstracts only); any source on bandits or options-level RL for graph parameters. No paper first published in November 2026 was found other than TI-0.1, and no zero-demonstration RL result on real USB-C.

**Our own numbers.** Milestone numbers: from branch `research/agentic-rl`: `docs/real/DATA_REVIEW_20261113.md`, `FULL_DEPTH_20261113.md`, `ALIGN_FAST_20261113.md`, `K2_PROGRESS.md`, the session files of 2026-11-10 and 2026-11-11, and `docs/campaign/FINDINGS.md`. Update numbers: from branch `real/graph-align`: `docs/campaign/FINDINGS.md` (entry of 2026-11-14), `docs/real/SESSION_20261113_duchamp3.md`, `DEPTH_SPRINGBACK_20261114.md`, `PORT_VISION_WIDTH_20261114.md`, `ZED_MAP_20261114.md`, `ZED_EDGEFIT_20261115.md`, `K7_20261114.md`, `GATE_REPLAY_20261114.md`, `NEXT_RIG_20261115.md`. Three numbers in the update are our own arithmetic on those files: the sum 12 of 12 with its Wilson interval, the 7 of 23 step timeouts, and the alignment's share of the run time (9 to 12 %). The review page still has the 2026-11-13 text.

**Numbers of the second round (2026-11-15).** From branch `real/graph-align`: `docs/campaign/FINDINGS.md` (entry of 2026-11-15), `docs/real/CONTACT_SENS_20261115.md`, `K8_20261115.md`, `FORCE_BUDGET_20261115.md`, `PERCEPTION_BIAS_20261115.md`, `PARALLEL_SCHEDULE_20261115.md`. Look-back numbers: `LOOKBACK_IDEAS_3_8_20261114.md` on this branch. The literature behind the formula edits (STEP, N-SCORE, Who&When, GTP-FA, Dao and Fern, ASAP, Kress-Gazit et al., ASPIRE, Marougkas et al., ScheduleGen) is in `AGENTIC_RL_IDEAS_20261114.md` and `READINGS_20261114.md` with its fetch status; it was not read again for this update. Our own arithmetic in this round: the point losses of K4 and K7 (61 and 46), and "3 of 7 failed starts" from the look-back text.

**Differences between our sources that this update did not resolve.**

- *Seat counts.* This document counts 12 of 12 seated when the plug reached the port. `K8_20261115.md` counts 14 real K4 insertions with 13 seat detections (`k4_2` ended on the guard bound), and the look-back gives 5 of 6 before the seat-speed change. The run sets differ. Conclusion 3 ("no failed start" for the RL node) holds for the milestone count only.
- *The force-vector config.* `FORCE_BUDGET_20261115.md` says that it needs no extra approval. `FINDINGS.md` says that the coordinator counts it as a new motion that needs approval. This document follows `FINDINGS.md`.
- *The 14 N result in the fitted plant.* `FINDINGS.md` gives 32 %; the table of `FORCE_BUDGET_20261115.md` gives 23.2 % of all worlds. Both are the same result: 32 % of the 738 pushed worlds.
- *Compliance.* 0.25 to 0.29 mm/N (2 force pushes, 2026-11-14), 0.24 to 0.33 mm/N (10 pushes under load) and 0.13 to 0.24 mm/N (plant fit, with 0 to 2 N of play) are three definitions on overlapping data.
- *The zed_v5 offset.* The 2026-11-14 text gives +4 to +6 mm in y; `PERCEPTION_BIAS_20261115.md` gives +4.0 to +4.4 mm (n = 89 frames) and +4.5 mm against the arm port (n = 41 runs).
- *Run time.* 117 and 122 s count from the log open. 109.8 and 114.3 s (`READINGS_20261114.md`) count from the capture start. `PARALLEL_SCHEDULE_20261115.md` resolved this one.
- *Clearance replay.* `FINDINGS.md` says 33 real runs; the tables of `PARALLEL_SCHEDULE_20261115.md` have n = 29 to 32 per overlap.
