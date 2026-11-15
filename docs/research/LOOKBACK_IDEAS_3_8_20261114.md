# Look-back tests of ideas 3 and 8 on the project history (2026-11-14)

Offline only: documents, git history and event logs. No rig, no training, no web.
Ideas and rules: `AGENTIC_RL_FORMULA_20261113.md`, sections "The loop", "Node triage", "New ideas".

Sources. Paths without a prefix are in the worktree `/home/yashchitambar/wt/graph-align` (branch `real/graph-align`).
Short names: `K2P` = `docs/real/K2_PROGRESS.md`; `CAP` = `docs/real/CONTACT_ALIGN_PLAN_20261107.md`;
`S<mmdd>` = `docs/real/SESSION_2026<mmdd>_duchamp3.md`; `FIND` = `docs/campaign/FINDINGS.md`;
`ROB` = `docs/real/ROBUSTNESS_PLAN_20261113.md`; `FD` = `docs/real/FULL_DEPTH_20261113.md`;
`DSB` = `docs/real/DEPTH_SPRINGBACK_20261114.md`; `AF` = `docs/real/ALIGN_FAST_20261113.md`.
Commits are short hashes of that branch. Logs: `/home/yashchitambar/peg_test/data/real_runs/<date>/*/events.jsonl`.

## Short answer

- **Idea 3 (reproduce-first gate): not falsified, but the test is weak.** Only one retrain line reached the rig
  (K-J to K2 to K3 to K4, used as K4). The sim reproduced its real failure before the retrain. No retrain helped on
  the rig without a reproduction, because no such retrain reached the rig. The record has three learning efforts that
  started without a reproduction. None of them has a rig success: RL alignment (dropped), `zed_v5` (failed its
  real-frame check), K6d and K7 (pending).
- **A new point against the retrain itself.** The cheap fix for the same cause (`insert.obs_recenter`, no training)
  seated 3 of 3 on the rig. K4 seated 10 of 10. These two rates are not different at this n. The record does not show
  that the retrain was necessary.
- **Idea 8 (agent writes the triage table): not falsified, and not proven.** Agreement was 11 of 13 failures with a
  rig-confirmed outcome (85 %). I proposed RL in 0 cases. But only 4 of the 13 proposals were blind, and the session
  documents already contain a diagnosis. The five measurements did not select the fix class in any case; the cause
  analysis did.

## Test A: idea 3, the reproduce-first gate

### Correction to the premise

K2 and K3 did not go to the rig. They are steps in one line of fine-tunes: K-J -> K2 (relative pose) -> K3 (+ camera
prior) -> K4 (+ left-arm inputs masked). Only K-J, K-J with the re-centre, and K4 ran on the arm
(K2P "K3 still reads the left press lead ... so the rig candidate is K4, not K3"; S1110 has only `kjrc_*` and `k4_*` runs).
So the look-back has one retrain with a rig result, not three.

### Table

| Case | Real failure that started it | Did the sim reproduce it before the fix? (evidence, n) | Fix class | Rig result (n) |
|---|---|---|---|---|
| K-J, first deploy | None: first policy. Rig: 1 of 4 inserted with a hand-placed reader (S1101, ai2-ai5), 0 of 2 from real picks (S1102) | Not applicable | RL (first training) | 1 of 6 (S1101, S1102) |
| K-J stalls, first look (11-03) | Plug stalls on the case, target climbs (S1102 "climbs in every run"; S1103 `align_full5`) | **No.** K-J on the real grasp in sim: 94.1 % (commit `58b23cc`, "no retraining run"). A dry-run failure (0 of 12) got another cause: plug turned by the net (`docs/real/ALIGN_STAGE.md` line 3, 27) | Scripted (align stage, seat check 2) + perception (r10 nets, port vision). No retrain | K-J 2 seats of 4 good hand-overs (S1106 "Result" 1; CAP section 1) |
| K-J re-centre (`obs_recenter`) | Same stalls: 2 of 4 (S1103, S1106) | **Yes.** Real FK offsets in the sim observation: 96.6 % -> 21.8 % [19.4, 24.4], n = 1,024; re-centred 89.4 % (CAP short answer 1, 3; commit `ca7e28a`) | Config / scripted (no training) | 3 of 3 seated: `kjrc_1`, `kjrc_2`, `kjrc_3_push` (S1110) |
| K2 (relative pose) | Same stalls | Yes, same evidence as above | RL retrain | **Not run on the rig.** Sim: ring 4 mm 18 to 39 % against K-J 81.8 % (K2P, 2026-11-07 and 11-08 tables) |
| K3 (+ camera prior) | None from the rig. A sim result: K2 cannot find an offset port blind (K2P 2026-11-07) | Not applicable (sim-only motive) | RL retrain | **Not run on the rig.** Sim replay with the real left lead: 0.0 %, n = 256 x 2 (K2P 2026-11-08 "fork C") |
| K4 (+ left inputs masked) | The same K-J stalls, plus the replay result for K3 | **Yes, in part.** Replay of real hand-overs: K3 and K-J 0.0 % with the real left lead, K4 91.8 to 96.5 % (n = 256 each; K2P; commits `86a8497`, `a6a35b6`, `c10d9f6`). But the sim is too strong: K-J seated 2 of 4 on the rig with that lead (K2P: "a sensitivity, not a rate") | RL retrain | 10 of 10 seated with the seat6 config; 5 of 6 before the seat-speed change (FIND 2026-11-11; S1110) |
| K4c (stronger climb penalty) | Real target climb 35 to 40 mm (CAP section 1) | **No.** Sim climb is 8.0 to 8.2 mm (joint 5 at its cap); the real climb came from joints 2-3 (K2P 2026-11-08) | RL retrain | Not run. Sim: climb did not change (K2P; commit `d6028ba`) |
| K6d, K7 (mating force) | Seated but 0.3 to 2 mm short (S1110, S1113 "band about 2 mm", n = 2) | **No measurement.** The sim got a mating spring with a guessed 0 to 10 N; "the real force of this port is not measured" (FD lines 19, 120; `docs/real/K7_20261114.md` line 75). With 10 N the sim says K4 seats 0 %; on the rig K4 seats and rests short | RL retrain | **Pending.** Not on the arm |
| Pick, closes empty | 2 of 8 closes held (S1101) | **No, then yes.** Baseline sim 512 of 512 at every tilt and opening (S1101 18:55). After the `grasp_fidelity` v2 model: oracle 169 of 512 held (commit `ac541b4`; `docs/real/GRASP_FIDELITY.md`) | Sim fidelity, then scripted (level roll) | 10 of 10 batch A, 13 of 13 with full and collect (S1102) |
| Wrist nets r10 | Net slope 0.34, 0.17, 0.35 on 150 real frames; 0 of 2 inserted (S1102) | **Yes.** Real PC textures in the sim: slope 0.94, 0.88, 0.98 -> 0.44, 0.33, 0.59; real 0.50, 0.32, 0.59 (FIND 2026-11-03; commit `2216461`) | Perception retrain + real fine-tune | Slope 0.97, 0.94, 0.97 on 60 hold-out frames (FIND 2026-11-03). In use as second source; a +2.6 mm up bias stays (S1111, n = 13) |
| `zed_v5` PC net | Placement limits found in sim sweeps (ROB) | **No.** No measured PC pose at the real spot when it trained (`docs/real/ZED_V5_20261114.md` via FIND 2026-11-13) | Perception retrain | Failed the real-frame check: 7.4 mm from v1, n = 44; later both nets found wrong (FIND 2026-11-14) |
| Stick-slip: servo sway | Tip speed p99 23.8 and 24.8 mm/s, limit 15: fail 2 of 2 (S1110) | **No.** Sim joint friction has static = kinetic; dry run p99 13.1 mm/s (AF line 118). A separate stick-slip model reproduced it afterwards: p99 32.3 mm/s (AF table) | Scripted (lead cut), found in the new model | Not run on the rig. Model only: p99 17.6 mm/s, still over 15 (AF) |
| Stick-slip: RL alignment (A2, A3) | Jerky align steps, 54 to 78 mm/s (S1103) | **No.** Same sim without stick-slip | RL (new policy) | Not run. 8.6 % (A2, n = 128), 53 % training (A3) against 99.6 % for the servo (K2P) |
| Stick-slip: approach jump | Uncommanded move over 3 mm on 7 of 18 approaches (`docs/real/DATA_REVIEW_20261113.md`) | **No.** "Only the rig can confirm (the sim joints do not stick)" (FIND 2026-11-13) | Scripted + config (`approach_nojump`) | 0.21, 0.24, 2.24, 2.86 mm, n = 4; old approach the same day 5.88, 2.32 mm, n = 2 (FIND 2026-11-14) |
| Single-arm Task H insertion (10-26/27) | Old policy 0.4 % on the real-arm config in sim (FIND 2026-10-26 evening) | Sim-only motive | Sim fidelity (measured friction) + RL retrain | **Unknown.** These documents give no rig rate for it |

### Verdict

**Not falsified.** The falsifier is "a past retrain helped on the rig although the sim did not reproduce the failure".
The record has no such case.

What the record does show:

1. **The gate was already in use, without a name.** On 2026-11-03 the sim did not reproduce the stall (94.1 %) and
   nobody retrained. On 2026-11-07 the sim reproduced it (21.8 %, n = 1,024) and the retrain started the same day
   (commits `ca7e28a` 05:28, `e52fe40` 06:14).
2. **Every learning effort without a reproduction has no rig success so far** (n = 4: K4c, RL alignment, `zed_v5`,
   K6d/K7; two are dead, two are pending). This supports the gate. It is not a proof: none of them had a rig trial as
   a policy.
3. **The reproduction was not quantitative.** The sim said 21.8 % (offsets) and 0.0 % (left lead) for K-J. The rig
   gave 2 of 4. The replay "gives seats and stalls the same rate" (K2P). So "the sim reproduces the failure" meant
   "the sim shows a mechanism that fits", not "the sim predicts the rate". The gate needs this defined.
4. **The retrain is not shown to be necessary.** Same cause, cheaper class: `obs_recenter` seated 3 of 3. K4 seated
   10 of 10. K-J plain against K4: 2 of 4 against 10 of 10, Fisher one-sided p = 0.066. Re-centre against K4: no
   difference at n = 3 and 10. The formula document says "RL retrain was needed three times (K2 to K4)". The record
   supports "one retrain reached the rig, and a config fix for the same cause also worked".
5. **A fix can work without a reproduction if it is not RL.** `approach_nojump` worked on the rig (n = 4) and the sim
   cannot show the jump. So the gate must apply to learned parts only, as idea 3 says.
6. **K6d and K7 do not pass the gate today.** Their failure (short by about 2 mm) is in the sim only through a guessed
   force. The 10 N and 14 N force pushes did not seat the plug on the rig (2 of 2 short, S1113), which a 0 to 10 N
   spring does not explain. By idea 3 the next class is sim fidelity: measure the mating force first.

## Test B: idea 8, the agent writes the triage table

### Method and leakage

Order of reading: formula document (task instruction), then S1101 + its logs, then S1102, then S1110 and S1113
together, then the logs of all four, then the later documents.

| Session | Blind to later documents? | Leak |
|---|---|---|
| 2026-11-01 | Yes. Proposals written before S1102 was open | The memory index line "Real pick grasp fix: software fixes first". The formula table (pick = scripted, no RL) |
| 2026-11-02 | Yes for later documents | The session document has its own "Server work" list (the answer). The formula text says K2 to K4 came later |
| 2026-11-10 | **No** | S1113 was read in the same step. The formula triage table is built from this session |
| 2026-11-13 | Yes for later documents | The formula document's 2026-11-14 update. Most outcomes are still pending on the rig |

A second limit: a session document is not raw data. The rig agent and the operator wrote causes into it. A real test
of idea 8 needs an agent that sees only `events.jsonl`, `ticks.npz` and the frames.

### Measurements from the logs (my own count)

Stage wall time includes operator prompts. The pick has no stage event of its own; it is inside "capture".

| Session | Runs with events (dry runs excluded) | Time share by stage (mean s, n) | Failures in the events |
|---|---|---|---|
| 11-01 | 22 (+1 empty log) | capture + pick 48 % (82 s, 21); home 16 %; release 12 %; approach 6 % (40.5 s, 5); insert under 1 % (4.5 s, 4) | 13 `stopped`: 3 SafetyStop (pick), 6 empty closes, 1 no reader, 1 estop (hold), 1 operator abort (approach), 1 press time (release). `seated_detected` 3 of 4 insertions; operator: 1 inserted |
| 11-02 | 13 | capture + pick 42 % (107 s, 13); collect 35 % (one run); approach 4 % (41.7 s, 3); insert 1 % (9.6 s, 2) | 0 `stopped`. Pick held 13 of 13. Insert: 1 timeout, 1 `seated_detected` (false by the operator) |
| 11-10 | 18 (+1 empty log) | capture + pick 47 % (114 s, 18); align 18 % (51.6 s, 15); home 16 %; approach 7 % (19.7 s, 15); insert 1 % (7.0 s, 9) | 3 `stopped` (1 TypeError, 2 camera). Pick held 16 of 16. Align: 12 ok, 2 not ok. Insert: 8 `seated_detected`, 1 `guard_bound`. Push ran on 6 |
| 11-13 | 13 (+1 empty log; 6 are capture-only) | capture + pick 55 % (56 s, 12); align 11 % (23.4 s, 6; ok runs median 11.9 s, n = 4); insert 1 % (7.3 s, 2) | 0 `stopped`. Align: 4 ok, 2 not ok. Insert: 2 `seated_detected`, both short by the operator |

What I could compute of the five measurements: priority (time share, failed starts) from the logs; tolerance ratio R
and contact share only as rough values from the session text; variety is 1 in every node; sim reproducibility only
where the session document states it (pick: no, S1101 18:55; servo sway: no, S1110; depth: no, S1113 dry-run note).

### Proposals against outcomes

"In-session" failures (fixed and confirmed inside the same session document) are not counted: GIL stalls, hold swing
speed, SafeArm limits (11-01); TypeError, seat speed 3 -> 6 mm/s (11-10); camera path, viewer label (11-13).

| # | Session | Failure (n) | R; contact decides; sim reproduces | My proposal | What worked later (n, source) | Agree | Blind |
|---|---|---|---|---|---|---|---|
| 1 | 11-01 | Pick closes empty, 6 of 8 | about 1; partly; no | Scripted (measure the hand tilt, correct it), sim fidelity second. No RL | Scripted: level roll, 10 of 10 (S1102). Found through the sim fidelity model | Yes | Yes |
| 2 | 11-01 | False or early seat, 3 of 3 detections (the one true insertion ended by timeout) | under 1 (gap not measured); no; not tested | Scripted + a measured gap | Scripted: seat check 2 on the vision gap (commit `20bac6b`); 10 of 10 seats detected with seat6 (FIND 11-11) | Yes | Yes |
| 3 | 11-01 | Plug stops short or on the case, 3 of 4 | under 1; no (before contact); not tested | Perception + a scripted alignment stage. No RL | Port vision + align stage (commits `659b1b9`, `29f1b12`); hand-overs within 0.8 mm (formula table, n = 14) | Yes | Yes |
| 4 | 11-02 | Nets report a third of the motion; 0 of 2 inserted | under 1; no; not tested | Perception retrain + classical port vision | Both (FIND 11-03; `docs/real/PORT_VISION.md`) | Yes | Document gives the answer |
| 5 | 11-02 | False seat at 5.76 mm, 1 of 2 | as row 2 | Scripted | As row 2 | Yes | Partly (same as row 2) |
| 6 | 11-02 | Policy climbs past the port, 2 of 2 | under 1; yes; not tested | Perception first; RL retrain only after a sim reproduction | **Miss.** Good hand-overs did not stop it (K-J 2 of 4, S1106). Fix: observation re-centre (config, 3 of 3) or K4 retrain (10 of 10) | **No** | Yes |
| 7 | 11-02 | Label height unknown within 4.5 mm | - | Perception (classical offset from the frames) | Port-vision label offset (commit `dd0c645`) | Yes | Document gives the answer |
| 8 | 11-02 | Sim in-hand pose differs from the rig | - | Sim fidelity | Sim fidelity: `inhand_real` preset (commits `50a1667`, `9f64668`) | Yes | Document gives the answer |
| 9 | 11-10 | Approach jump over 3 mm, 3 of 15 approaches | above 3; no; no | Scripted (deadband at the stop) + sim fidelity | Scripted + config plumbing: `approach_nojump`, n = 4 (FIND 11-14) | Yes | **No** |
| 10 | 11-10 | Seated but 1 to 3 mm short, all seats | under 1; yes; no | Scripted force push, then RL with a mating force | **Miss.** Force push 10 N and 14 N: 0 of 2 fully in (S1113). The 15 mm position push gave 3 runs "fully in" (DSB). Open | **No** | **No** (the formula table proposes the same) |
| 11 | 11-10 | "Sources disagree" stop, 1 (3.54 against 3.5 mm) | 1 to 2; no; - | Config (bias offset or gate) | Config: `bias_mm` offset, 0 stops in 3 runs; the run without it stopped (FIND 11-14) | Yes | **No** |
| 12 | 11-10 | Post-press budget ran out after a 2.8 mm hand jump, 1 | - | Config (budget) + the jump fix | No repeat in 4 runs with fast align + no-jump (S1113) | Yes (weak) | **No** |
| 13 | 11-10 | Alignment slow, 49 to 55 s | 1 to 2; no; - | Config + optimiser | Scripted + config (`fastalign`): 10.1 to 13.9 s, n = 4 (FIND 11-14) | Yes | **No** |

**Agreement: 11 of 13 (85 %).** By blindness: blind 3 of 4 (rows 1, 2, 3, 6); document-led 4 of 4 (rows 4, 5, 7, 8);
not blind 4 of 5 (rows 9 to 13).

Pending or unknown (not counted):

| Session | Failure | My proposal | Status |
|---|---|---|---|
| 11-10 | Servo sway, fail 2 of 2 | Config (gains) + sim fidelity | Lead cut in a stick-slip model, p99 17.6 mm/s, still over the limit; not on the rig (AF) |
| 11-10 | Push stopped by the gripper watch, 1 | Config (limit) | Counted as a false stop of a quality gate (FIND 11-13 gate replay); no rig change yet |
| 11-10, 11-13 | Left wrist camera drops off USB, 3 starts | Hardware; scripted (camera optional) | Config (device path) each time; the fault came back (S1113) |
| 11-13 | Port vision takes the wrong shell edge, 1 align stop | Perception (classical) | Shell reference: 16 frames recovered of 4,824 in replay; not on the rig (FIND 11-14) |
| 11-13 | Band of 2 mm after 10 N and 14 N, 2 of 2 | Scripted (keep the load while the jaws open); sim fidelity | **My fix was rejected offline:** the sim hand jumps 1.8 to 2.5 mm at 3 N (FIND 11-14). Slow unload and a position-push A/B wait for the rig |
| 11-13 | Label FULL with a 2 mm band, 1 | Perception (a real depth signal) | Label turned off (config). No FK threshold separates the runs (DSB section 4) |
| 11-13 | ZED PC estimate moves less than the PC; yaw gate rejects | Perception retrain | **Class right, kind wrong.** The retrain (`zed_v5`) is 4 to 6 mm off. A classical case-edge fit is within 1.12 mm on 41 runs, offline (FIND 11-14) |

### RL against config or scripted

- I proposed RL where a config or scripted change solved it: **0 cases.**
- The reverse (I did not propose RL, and an RL retrain was the fix that went to the rig): **1 case**, row 6. Even
  there the config fix (re-centre) seated 3 of 3.
- Row 10: the rule's class is "RL, or scripted with a force estimate". The scripted option failed 0 of 2. RL is
  pending and does not pass the reproduce-first gate yet.

### Verdict

**Not falsified** (agreement 85 %, limit 50 %; no RL proposed where config solved it). **Not proven**, for three
reasons:

1. Only 4 proposals were blind, and one of the 4 was the miss.
2. The five measurements did not choose any fix class. They choose the node class (RL node or scripted node). The fix
   class came each time from one question: what is the cause, and is it in a sensor, a number, the code or the
   contact? Only "sim reproducibility" changed a decision.
3. The agreement measures a cheap prior: "propose scripted or perception, never RL" scores 11 of 13 here without
   any measurement. The test needs cases where RL is the right answer to separate the agent from that prior.

## Patterns that the formula document does not have

1. **A correct fix that never ran.** The 2026-11-03 answer named the cause of the approach jump (the servo integrates
   at the stop, 6 iterations) and wrote the fix as config keys `iters` 4, `final_iters` 2
   (`docs/real/ANSWERS_20261103.md` answer 1; commit `cee9510`). `chain_argv` never passed the keys. All 37 real
   approaches from 2026-11-03 to 11-11 ran the old 11 iterations (ROB Part C item 1). The jump was then put on the
   controller and on joint wind-up (S1106, S1110) for 10 days. The conventions check passed every time. Missing gate:
   show that each new config key changes the executed behaviour.
2. **Fixes that were turned off again.**
   - `bimanual_seat2` (11-01 17:06): "operator: go back to the original config" after ai5 (S1101).
   - Press relief v1 to v3 (11-01): 2 of 8 held; replaced by the level roll (S1102).
   - Plug angle turn in the align stage (commit `565d076`, 11-03 12:39): off on the same day, "the hand does not turn
     at all" (`ANSWERS_20261103.md` answer 5), after a false scale stop and grip loss with larger turns (S1103).
   - Precision controller in the chain: best align stage on 11-06, then a hand sag at the switch-off on 11-10; all
     later runs on plain PD (S1106, S1110).
   - Climb penalty (K2 spec, K4c): no effect on the climb (K2P 2026-11-08).
   - Force push 10 N and 14 N: no gain over the position push (S1113); "load on while the jaws open" dropped (FIND 11-14).
   - Retry loop, backlash preload, servo alignment: written, never run on the rig (`docs/real/RETRY_SPEED_20261108.md` line 3).
3. **Claims that data corrected later.**
   - "The late ticks come from the streaming": wrong; port vision runs in the control thread (`DATA_REVIEW_20261113.md` item 13).
   - "The sim has the real arm's behaviour" for the align speed peaks (K2P 2026-11-07): the peaks were a reset
     artefact (K2P, same day), and the sim has no stick-slip (AF line 118).
   - "Align takes 34 to 55 s": 59.7 s median wall time (FIND 11-13).
   - "v1 is within about 2 mm at the real spot": wrong, 8.1 mm low (FIND 11-14).
   - "Fully in from 7.2 mm" (3 labelled runs): a run at 7.33 mm had a 2 mm band (S1113; DSB section 4).
   - "The left wrist camera is dead": it came back 14 minutes later (S1110).
   - Operator on the same push class: "about 3 mm short" (`k4_1`), then "flush" (`k4_5`) (S1110). The true depth was not measured in either.
   - Main cause of the K-J stall: nets (11-02), plug turned by the net (11-03), the policy's search (11-06), absolute
     FK features (11-07), the left press lead (11-08). Five days from the first sight (S1102) to a sim reproduction (`ca7e28a`).
4. **The sim reproduction over-predicts.** K-J: sim 21.8 % and 0.0 % against 2 of 4 real. The mating spring: K4 0 %
   at 10 N in sim against 10 of 10 detected seats real. A reproduction that is too strong still chose a working fix
   once (K4). It can also justify a retrain that is not needed (K4 against the re-centre).
5. **The dry-run gate has a known false failure.** "Seat detected, not seated after the release" appears in dry runs
   of K4, K7 and the align9/align10 configs (K7: 2 of 4 runs; K3: 2 of 3 runs; K2P line 213; FIND 11-13, 11-14) and is accepted as "the
   known sim release effect". K4 passed to the rig with it and seated 10 of 10. The gate in step 2 cannot test the
   release at all.
6. **Real-only code paths escape the dry run.** Three software stops on the arm came from paths that the dry run does
   not reach: `log_event` TypeError (S1110), `KeyError: 'mm'` with `--demo` (S1113), three controller bugs (S1106).
7. **Failures are not in one place in the logs.** A failed start is a `stopped` event, or `align_result.ok` false, or
   `insert_result.outcome`, or `pick_outcome`. Align stops do not write `stopped`. The operator's true result is in
   the session document only. On 11-10 the events show 3 `stopped`; the session has 7 failed starts (1 software stop, 2 camera, 1 guard
   pull-out, 2 align stops, 1 process end). An agent that counts `stopped` events finds under half of them.
8. **A sim fidelity model can find the cause without becoming the training sim.** `grasp_fidelity` v2 reproduced the
   pick hold rate (169 of 512) but "the rise is not reproduced" (commit `ac541b4`). It was enough to point at the roll.
   No policy trained in it.
9. **Quality gates at the edge of the normal spread cost more starts than any policy.** 3.5 mm disagreement gate: 3
   false stops in 27 hand-overs; 2.5 mm push gate: 2 false skips in 13 seats (FIND 11-13 gate replay). The formula
   document has the result; it does not have the rule "set a gate from the measured spread, not by hand".
10. **Repository mistake.** Commit `fcf5237` added 34.4k raw files (543 MB); removed in `965f1c7`, still in the
    history (FIND 11-13).

## Changes that these tests suggest for the formula

1. Idea 3: define "reproduces" as two levels. Level 1: the sim shows the same failure mode after a measured change
   (enough to choose the class). Level 2: the sim rate on replayed real hand-overs is inside the real interval
   (needed before a rate claim). K4 had level 1 only.
2. Idea 3: add "try the no-training fix for the same cause on the rig first, in the same session as the retrain"
   and compare them at equal n. The 11-10 session did this by accident (3 against 6 runs) and then dropped the cheap arm.
3. Idea 8: run the blind test again with a fresh agent that gets only the raw logs of one session, and add the
   2026-11-03 and 11-06 sessions (the K-J stalls), where the right class was not the cheap prior.
4. Loop step 2: add an effective-config check (pattern 1) and a real-path mock for every new event or label (pattern 6).
5. Loop step 4: one `run_outcome` event per run with the failing node and the operator's result (pattern 7).
