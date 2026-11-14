# Agentic RL for fine-grained manipulation: new ideas from a second literature pass

2026-11-14. Companion to `AGENTIC_RL_FORMULA_20261113.md`. This page adds sources that are not in that document's source list, ten new ideas, proposed edits to the six-line formula, and three experiments for the server. Nothing here has run. No rig access and no sim run were used.

## Short answer

- The largest gap in the formula is not a robot method. It is the agent's evidence rule. The formula says how to fix a node. It does not say when a claim is proven, how a failure is assigned to a node, or how a real result at n = 10 is compared with a sim result at n = 1,024.
- The literature gives three usable rules. (1) Assign a failure by an intervention, not by reading the logs. (2) Compare options on the rig with paired, sequential tests on continuous outcomes. (3) Fit a small parametric plant to the logs; do not fit an expressive residual to a few runs.
- For the depth problem, two sources show that an arm without a force sensor can estimate external force to about 1 N from its own motor signals. Our own logs then give a hard limit: the present push cannot reach more than about 14 to 16 N before its advance watch stops it.
- One statement in the formula document needs a correction. A published zero-demonstration result without a force sensor exists at about 0.1 mm clearance (Marougkas et al., cylinder 10 of 10, triangular plug 3 of 10). The earlier statement said that such results stop near 0.5 mm.

## How this page was made

Each source below was fetched on 2026-11-14 through a fetch tool that summarises the page. For pages longer than 100,000 characters the tool read only the first 100,000. A number that is only in a figure can be missing. The status of each source is in the table at the end. Sources that could not be fetched are marked "not verified" and are not used as evidence. No source is cited from memory.

Terms used here. *Node*: one stage of the task graph. *Hand-over*: the state at which one node gives control to the next. *Replay*: a sim run that starts from a logged real state. *e-value*: a number that measures evidence against a null hypothesis and that stays valid when the test stops early.

## What the new sources report

### Evaluation with small n

| Source | What it reports | Use for us |
| --- | --- | --- |
| [STEP](https://arxiv.org/abs/2503.10966), Snyder et al., Mar 2025 | A sequential test for two policies with binary outcomes. It keeps the false-positive rate at a set level and can stop early. Real robots: a 56 % against 92 % gap was decided in about 20 to 23 trials (batch: 50). A 40 % against 82 % gap in 14 to 18 trials. A 68 % against 76 % gap was not decided in 100 trials; the authors estimate about 500. A batch test that is repeated after each trial breaks the error control | A rig session of 15 to 20 runs can decide only large gaps. Small gaps in success rate cannot be decided on our rig |
| [N-SCORE](https://arxiv.org/abs/2603.13616), Snyder et al., Mar 2026 | The same idea for graded or continuous outcomes, with paired trials. Abstract: "up to 70% reduction in evaluation burden" against batch methods. Hardware, 50 trials per policy nominal: 286 trials in total against 376 for STEP on binary success. "competing policies can be separated more quickly when using fine-grained task progress than binary success" | Use millimetres and seconds as the outcome, in pairs, with a stop rule set before the session |
| [SureSim](https://arxiv.org/abs/2510.04354), Badithela et al., Oct 2025 | Paired real and sim trials on the same start conditions correct the sim's bias (prediction-powered inference). 60 paired real trials plus 700 to 2,100 sim trials. Interval width 0.16 against 0.187 real-only. It saves 20 to 25 % of hardware trials at a sim-real correlation of about 0.59. At a correlation of about -0.05 the sim gives no gain | The sim helps a real estimate only when the per-run sim prediction correlates with the per-run real outcome. This is measurable on our logs |
| [Robot Learning as an Empirical Science](https://arxiv.org/abs/2409.09491), Kress-Gazit et al., Sep 2024 | Best practices: define success before the test; the person who built the behaviour does not run the evaluation; blind, interleaved A/B runs; report counts; report failure modes. Three authors disagreed on the success of one episode | "Fully in" needs a written definition and a measurement. Interleave A and B in one session |
| [RoboGate](https://arxiv.org/abs/2603.22126), Kim, Mar 2026 | Sim only. 20,000 uniform samples, then 10,000 samples in the bins with 30 to 70 % success. A logistic risk model: AUC 0.754 to 0.780. A closed-form failure boundary. No real validation | Sample the sim near the failure boundary, not on a fixed grid. Weak source (one author, sim only, a small AUC gain) |

### Failure attribution and agent self-checks

| Source | What it reports | Use for us |
| --- | --- | --- |
| [Who&When](https://arxiv.org/abs/2505.00212), Zhang et al., May 2025 | 184 annotated failures from 127 LLM multi-agent systems. An LLM that reads the full log names the responsible agent in 53.5 % of cases and the decisive step in 14.2 %. Reasoning models do not reliably help. Accuracy falls with log length. The methods are more reliable in aggregate than per case | An LLM that reads logs is a weak judge of the decisive step. Our loop step 5 ("find the node") depends on exactly this |
| [GTP-FA](https://arxiv.org/abs/2606.03385), Xu et al., Jun 2026 | A two-stage grasp-then-plan pipeline. A failure is assigned to the grasp or to the planner from repeated trials with one stage held fixed. Real Franka, five tasks, 50 trials each: 11.2 % to 76.8 %. Without attribution, updates to one stage alone lowered success (PPO -28.1 and -38.7 points; SAC -54.6 and -54.5 points). The labels are pseudo-labels, "not intended as ground-truth causal annotations" | A fix sent to the wrong node makes the system worse. Hold one node fixed and repeat |
| [MAGMA-GEN](https://arxiv.org/abs/2609.20056), Bernat et al., Sep 2026 (abstract only) | A diagnosis of an ambiguous failure is kept only if "re-execution from the same state under matched conditions improves downstream progress" | The acceptance test for a diagnosis is a re-run from the same state |
| [Hidden Pitfalls of AI Scientist Systems](https://arxiv.org/abs/2509.08713), Luo et al., Sep 2025 | Two AI scientist systems were tested for four faults. Post-hoc selection: with test results inverted, one system picked the top test candidate 31.5 % of the time against 82.0 % in the control. An LLM auditor found faults with 55 % accuracy from the paper alone and 82 % with logs and code | The agent's summary is not enough to audit the agent. The auditor needs raw logs and code |
| [POPPER](https://arxiv.org/abs/2502.09858), Huang et al., Feb 2025 | Agents test a hypothesis by falsification tests. Each p-value becomes an e-value; the product must reach 1/alpha. Type-I error 0.103 at alpha 0.1; ReAct-style baselines 0.078 to 0.145, a Fisher combination 0.311. The design agent sees only the data schema. Of 128 failed logs, 35.9 % were misread p-values | A written falsification test before the data is read, and a fixed rule to combine tests |
| [ASPIRE](https://arxiv.org/abs/2607.00272), Lu et al., Jun 2026 | Coding agents write robot programs and keep repairs in a skill library. A fresh verifier agent must reproduce the success before an entry is kept. Real bimanual YAM, 20 trials per task: lift can 13 to 19 of 20, drawer 0 to 11 of 20, with a skill from sim. No insertion task. Stated limit: skills are "verified once at admission and not re-validated", so entries can become stale. Real use needs "robust success detection, safe reset" | Every stored fact needs a second agent that reproduces it, and a repeat check later |
| [Agent-driven RL research](https://arxiv.org/abs/2603.27416), Khandelwal and Gupta, Mar 2026 | A coding agent ran more than 70 PPO experiments in 14 waves on a quadruped in Isaac Lab. Sim only. A person chose each phase's topic. Lesson: "total reward is not directly comparable across configurations with different weight scales" | The comparison metric must be fixed across experiments. No new method |
| [ASIA](https://arxiv.org/abs/2605.10480), Piga and Forgione, May 2026 | A coding agent searches model classes for system identification under a fixed, read-only evaluation pipeline. Nanodrone: position error 2.1 against 2.4 (prior method). The authors flag "implicit test leakage": the agent saw test scores in its logs | The evaluation code and the held-out set must be read-only for the agent |

### Real-to-sim identification from little data

| Source | What it reports | Use for us |
| --- | --- | --- |
| [Simulator adaptation by distribution matching](https://arxiv.org/abs/2604.11090), Dao and Fern, Apr 2026 | A Wasserstein distance between sim and real distributions of joint positions, velocities and actions. No time alignment, no motion capture. 64 trajectories of 4 s (about 4.3 minutes) on a real Go2 quadruped. With noise, a six-parameter friction and armature model was identified with 18.75 % mean error. The expressive action-delta and residual actuator networks gave "substantially worse finetuning performance" in the parameter-shift case | Fit a few physical parameters by distribution matching. An expressive residual on a few runs is the wrong tool |
| [Few-shot neural differentiable simulator](https://arxiv.org/abs/2603.06218), Huang et al., Mar 2026 | CMA-ES identifies MuJoCo contact parameters (solref, solimp, friction) from 33 real trajectories of about 20 frames. Held-out trajectory error 1.14 to 0.73 (14 test trajectories). Task: one cube pushed into another | MuJoCo contact parameters can be fitted from tens of short real traces with CMA-ES. Our engine and our data size |
| [ASAP](https://arxiv.org/abs/2502.01143), He et al., Feb 2025 | A learned delta-action model aligns the sim with real humanoid rollouts. The full 23-joint model needs over 400 clips; the authors trained only a 4-joint ankle model on 100 clips. Two robots broke during collection | A learned residual plant needs hundreds of clips. We have about 20 insertions |
| [AdaptSim](https://arxiv.org/abs/2302.04903), Ren et al., Feb 2023 (abstract only) | Adapt the sim parameter distribution for task performance, not for trajectory match. Claim: "an irreducible gap between simulation and reality" | The target of identification is the node's outcome, not every trace |
| [Contact-Aware Neural Dynamics](https://arxiv.org/abs/2601.12796), Jing et al., Jan 2026 | Argues that a low-dimensional correction "is not sufficient for contact-rich manipulation". It uses tactile sensors, and it does not test explicit identification | The opposite view. It is argued, not measured, and it needs tactile data |
| [Guided multi-fidelity BO](https://arxiv.org/abs/2509.17952), Nobar et al., Sep 2025 | A Gaussian process corrects the digital twin from each real run. The optimiser queries the twin when it is accurate and the hardware when it is not. Two gains of a motor drive: 5 real experiments against 9 for plain BO (20 repeats) | A cost-aware choice between sim and rig. Only two parameters |
| [Real-is-Sim](https://arxiv.org/abs/2504.03597), Abou-Chakra et al., Apr 2025 | A sim that is corrected from cameras while the real robot runs. PushT only. Sim and real success curves agree (60 real measurements per point); no correlation value | Not for contact at 0.1 mm |

### Force and contact from motor signals or a camera

| Source | What it reports | Use for us |
| --- | --- | --- |
| [FACTR 2 (NEXT)](https://arxiv.org/abs/2606.12406), Oh et al., Jun 2026 | A small recurrent network learns the free-space motor torque from joint position, velocity and tracking-error history. External torque = measured motor torque minus the prediction. Data: about 10 minutes of contact-free motion; training about 1 minute on one GPU. Franka in contact: 0.547 N m mean joint error against 1.471 (disturbance observer) and 4.395 (FILIC). AgileX Piper in free space: 0.018 N m against 0.248. The YAM is used for teleoperation in this work. No error in newtons. No separate test of gearbox friction | A learned free-space model can replace our friction model in the force estimate. Our logs already hold many minutes of contact-free motion |
| [Minimalist Compliance Control](https://arxiv.org/abs/2603.00913), Shi et al., Mar 2026 | External wrench from motor current, a motor model and the Jacobian. ARX X5 arm with quasi-direct-drive motors: 1.05 ± 1.60 N mean error against an ATI sensor. Stiction and backlash are not modelled | About 1 N is reachable on an arm of our class. The error spread (1.6 N) is large against a 5 N decision |
| [NeuralActuator](https://people.csail.mit.edu/frankzydou/projects/NeuralActuator/index.html), Dou et al., RSS 2026 | A Transformer on 9 frames of commands, joint state and motor telemetry gives an external force and a contact probability. Low-cost servo arms (OpenManipulator-X, SO-101). Force error 0.23 N (force-sensor test set). About 95 minutes of data | A contact-probability output from motor signals alone. More data than NEXT |
| [Visual force for compliant grippers](https://arxiv.org/abs/2605.00307), Zuo et al., May 2026 (abstract only) | Grasp force from wrist RGB-D images of a deforming soft finger and an inverse finite-element model: 0.23 N RMSE in the load phase | Needs depth images and soft fingers. Not for our rig |

### Insertion from sim without demonstrations or a force sensor

| Source | Sensing | Clearance | Real result (n) | Does not show |
| --- | --- | --- | --- | --- |
| [Marougkas et al.](https://arxiv.org/abs/2505.11858), May 2025 | RGB-D pose tracker (M3T, 30 Hz), position control, no force sensor described | About 2, 1 and 0.1 mm; IndustReal plugs 0.5 to 0.6 mm | Cylinder 10 of 10 at each level. Triangular plug 6, 6 and 3 of 10. Chargers 10 of 10; HAN connector 9 of 10 | Tight clearance with a shape that needs rotation; needs CAD models and a depth camera |
| [Park et al.](https://arxiv.org/abs/2509.13731), Sep 2025 | Two wrist RGB cameras, segmentation masks as the common sim and real image; no force sensor described | Flat cable; success = both corners within 1 mm | 60 to 90 % over eight settings, 10 trials each | Clearance in mm; n is small |
| [Dynamic Compliance Tuning](https://arxiv.org/abs/2311.07499), Zhang et al., Nov 2023 | Wrist force/torque sensor, no camera | 0.05 and 0.02 mm; one part at -0.21 mm | 7 to 10 of 10 per task | Works without a force sensor |
| [Foresight Residual RL](https://arxiv.org/abs/2607.16506), Liu et al., Jul 2026 (abstract only) | Sim only (Isaac Gym), nut tightening in three phases | Not given | Full task 85.6 % against 54.5 %, with per-phase success unchanged | Any real result; n |

**What these tables add to the earlier review.** (1) The earlier statement "without force sensing, the published zero-demonstration results stop at about 0.5 mm clearance" is too strong: Marougkas et al. report 10 of 10 at about 0.1 mm for a cylinder, and 3 of 10 for a triangular plug. The corrected statement: at about 0.1 mm without force sensing, the published results hold only for a shape that needs no rotation. (2) No new source inserts a real USB-C plug without a force sensor. (3) No source combines an agent loop with a statistical evidence rule on a real robot.

## Ten new ideas

None of these is in the earlier eight ideas or the nine "take from each paper" items. Where an idea touches an earlier one, the difference is stated. Ideas 1 to 4 are about how the agent works.

### 1. Assign a failure by a node swap in the replay, not by reading the logs

- **Idea.** For each failed or short run, the agent replays the run in sim from each logged hand-over and replaces one node's logged output with a nominal value, one node at a time. The node whose replacement changes the outcome gets the failure.
- **Source.** GTP-FA, MAGMA-GEN, Who&When.
- **Why it can work here.** We already have the hand-over replay and the gate replay. Who&When shows that log reading finds the decisive step in 14.2 % of cases. Our own record has two wrong first causes (the late ticks, the wrist-net bias). GTP-FA shows that a fix sent to the wrong stage lowers success by 28 to 55 points.
- **Experiment (offline and sim).** Use the 5 failed starts of 2026-11-10/11, which have known causes (2 gate limits, 1 post-press budget, 1 pick height, 1 software), and the three hand-overs where K7 is below K4 (ai4, ai5, full_A2). For each run, swap in turn: the ZED estimate, the pick pose, the hand-over pose, the vision prior, the policy, the gate limits. n = 256 worlds per swap. Compare the node named by the swap with the known cause. Compare also with a fresh agent that only reads the logs.
- **Cost.** 3 days.
- **Proven wrong if.** The swap names the known node on fewer than 4 of the 5 failed starts, or it is no better than the log-reading agent.
- **Limit.** A cause that the sim does not contain (joint stick-slip) gives "no swap changes the outcome". That result is still useful: it sends the fix to the sim-fidelity class.

### 2. A sequential, paired evidence rule for every rig comparison

- **Idea.** The agent writes each rig comparison before the session: the two configs, one continuous outcome (rest depth in mm, seat time, alignment time), an interleaved order, and a stop rule from a sequential test. The agent may not report "A is better than B" without the test result.
- **Source.** STEP, N-SCORE, Kress-Gazit et al.
- **Why it can work here.** A session gives 15 to 20 runs. STEP decided a 36-point gap in about 20 trials and could not decide an 8-point gap in 100. So success rates are out of reach, and the formula document already says to use continuous outcomes. The missing parts are the pairing, the fixed order and the stop rule. The depth A/B for the next session (`bimanual_align10_all` against `bimanual_align10_pospush`) is the first case.
- **Experiment (offline).** Take the per-run outcomes in the logs (rest depth for 7 position pushes and 2 force pushes; alignment time for old and fast alignment). Simulate the sequential test on resampled outcomes. Output: the number of pairs that each planned comparison needs, and the comparisons that one session cannot decide.
- **Cost.** 1 day.
- **Proven wrong if.** The planned depth A/B needs more than 20 pairs at the observed spread, with no outcome that needs fewer. Then the rule gives no decision in one session, and a better measurement (idea 9) must come first.

### 3. A claims ledger with a re-run command, a blind auditor and a repeat check

- **Idea.** Each quantitative claim in the notebook becomes one row: the claim, n, the command that reproduces it from the logs, and the result that would refute it, written before the analysis. A second agent that sees only the raw logs and the code re-runs each row; a scheduled job repeats the rows when the logs or the code change.
- **Source.** Hidden Pitfalls of AI Scientist Systems, POPPER, ASPIRE, ASIA.
- **Why it can work here.** The pitfalls paper measured 55 % audit accuracy from the write-up against 82 % with logs and code. ASPIRE names stale entries as its own limit. Our notebook has at least one statement that was later found wrong ("v1 is within about 2 mm at the real spot") and the working memory file holds "verified" facts with no check attached.
- **Experiment (offline).** Take 30 numeric claims from `FINDINGS.md` (2026-11-10 to 11-14) and from the formula document. Write the re-run command for each. Give a fresh agent only the logs and the commands. Count: reproduced, changed, not reproducible.
- **Cost.** 2 days.
- **Proven wrong if.** At least 29 of 30 reproduce and no changed value alters a decision. Then the ledger costs more than it saves.
- **Difference from the earlier idea 8.** Idea 8 asks whether the agent can choose the fix class. This idea checks whether the agent's stated facts are true.

### 4. A paired sim twin for every real run

- **Idea.** For every real run, the agent stores the sim replay prediction of the same outcome from the same hand-over (seat time, rest depth, seat rate over 256 worlds). The correlation between prediction and real outcome, per node, becomes the number that says how much a sim result is worth on the rig.
- **Source.** SureSim.
- **Why it can work here.** The formula document says the replay "ranks policies; it does not predict real rates". SureSim gives the test for that sentence: sim trials tighten a real estimate only when the paired correlation is well above zero (0.59 helped; -0.05 did not). We have 37 runs with logs and 10 replayed hand-overs.
- **Experiment (offline and sim).** Replay every logged hand-over (K4, n = 256). Correlate sim seat time and sim rest depth with the real values (n = 8 to 10 insertions with a rest depth). Report the correlation with its interval.
- **Cost.** 1 to 2 days.
- **Proven wrong if.** The correlation interval includes zero for every outcome. Then sim acceptance cells say nothing about the rig for this node, and every K7 claim needs rig runs. This is a likely result at n = 10; the experiment then tells us how many paired runs make the question decidable.
- **Difference from the earlier Moonlake item.** That item compares sim cells with real outcomes per placement on a future session. This one pairs each past real run with its own replay and needs no rig.

### 5. Identify a small contact and compliance model by distribution matching on the logs

- **Idea.** Add a few physical parameters to the sim plant (an axial series stiffness for arm and grasp, a mating-force profile over depth, port friction, the contact solver constants) and fit them with CMA-ES so that the sim's insertion and push traces match the logged ones in distribution. Retrain or re-evaluate K4 and K7 in the fitted plant.
- **Source.** Dao and Fern; Few-shot neural differentiable simulator; ASAP as the counter-example.
- **Why it can work here.** The spring-back analysis already measured the main parameter: 0.25 to 0.29 mm/N under load, so 4 to 5 N/mm, against a PD stiffness of 0.38 N/mm. The sim has no such spring, so the sim hand does not come back and the sim "depth" means something else. Huang et al. fitted MuJoCo contact constants from 33 short traces. Dao and Fern found that a six-parameter model beat expressive residual models on about 4 minutes of data.
- **Experiment (sim and logs).** Fit on the 2026-11-10/11 insertions and position pushes. Hold out the two force pushes of 2026-11-13. Check: predicted advance under load (2.60 and 4.06 mm) and spring-back (1.93 and 3.37 mm). Then report K4 and K7 rest depth in the fitted plant against the real 5.65 to 7.00 mm.
- **Cost.** 3 to 4 days.
- **Proven wrong if.** The held-out spring-back error is above 0.5 mm, or three CMA-ES seeds give parameter sets that differ by more than a factor of 2 with the same cost (not identifiable), or the sim rest-depth distribution does not move toward the real one.
- **Difference from the earlier idea 4.** Idea 4 fits "a small correction to the sim's joint response". The sources say to keep the correction parametric and physical at our data size. This idea replaces idea 4's learned residual.

### 6. The arm as its own force sensor, and a force budget for the push

- **Idea.** Train a small network on the contact-free parts of the existing 50 Hz logs to predict the free-space joint torque, and take the difference to the measured torque as the external force. Combine it with the stiffness of idea 5 to state, for each push, the force at the plug and the plug travel (hand advance minus force divided by stiffness).
- **Source.** FACTR 2 (NEXT), Minimalist Compliance Control, NeuralActuator.
- **Why it can work here.** NEXT needs about 10 minutes of contact-free motion; our logs hold the capture, transport and home moves of more than 37 runs. Our present estimate uses a friction model and "is wrong by several newtons". Two sources report about 1 N on arms of this class.
- **A limit that our logs already show.** At 0.25 to 0.29 mm/N, the 4 mm advance watch of the force push allows at most about 14 to 16 N. The 14 N push used 4.06 mm. Vendor data sheets found in the search give a USB Type-C insertion force of 5 to 20 N (not verified against the USB-IF text). If the real port needs more than about 15 N, the present push cannot seat the plug at any force setting. The load curve was linear with no step, which agrees with a plug that did not move.
- **Experiment (offline).** Train on contact-free segments, hold out by date. Report the residual per joint and, through the Jacobian, in newtons along the port axis. Check in contact against the PD force at rest (10.0 and 13.9 N on the two force pushes; the left press at a 25 N command).
- **Cost.** 2 days. A true test needs 10 minutes on the rig with a scale.
- **Proven wrong if.** The held-out free-space residual is above 2 N along the port axis (95th percentile). Joint stick-slip of 0.9 to 1.1 N m on joints 2 and 3 can cause this.
- **Difference from the earlier GaP item.** That item uses the existing torque estimate with a friction model for a force threshold. This idea replaces the friction model with a learned free-space model and adds the travel estimate and the force budget.

### 7. Choose the rig placements from the sim failure boundary

- **Idea.** Sample the sim uniformly over placement and plant parameters, then sample again in the region of 30 to 70 % success, and fit a risk model. Put the locked rig placements on that model: some inside, some near the boundary.
- **Source.** RoboGate; STEP for the reason.
- **Why it can work here.** K7 passes fixed cells (worst 97.1 %) but is below K4 on three bad hand-overs, and the cause is open. A fixed grid cannot show where the boundary is. A rig run near a predicted boundary tests the sim; a run far inside it tests little.
- **Experiment (sim).** K4 and K7, about 30,000 episodes each over PC x, y, yaw, hand-over offset (across, up, gap), roll, vision-noise scale and mating force. Fit a logistic model. Check it on the 10 real hand-over replays.
- **Cost.** 1 to 2 days, one GPU.
- **Proven wrong if.** The risk model's AUC on held-out sim episodes is under 0.75, or it does not order the 10 replay rates (Spearman under 0.6).
- **Rating note.** The source is weak. The idea stands on our own need, not on its evidence.

### 8. A conformal band on the insertion traces, calibrated on successes only

- **Idea.** Compute one scalar per tick during the insertion (for example the tracking residual or the policy's contact-head entropy), and build a time-varying band from successful sim episodes. A real run that leaves the band is flagged.
- **Source.** FAIL-Detect; E-valuator for the sequential form.
- **Why it can work here.** We have successes and no K4 failure on the rig, which is the case FAIL-Detect is built for. The band has a second use: the share of real successes outside a sim-calibrated band is a per-node measure of the sim gap, with the tick and the signal where the gap appears.
- **Experiment (offline).** Calibrate on 1,024 sim successes. Test on sim failures (detection before the seat time) and on the logged real insertions.
- **Cost.** 2 days.
- **Proven wrong if.** Balanced accuracy on sim failures is under 70 %, or the band flags more than 2 of the 10 real seated runs.
- **Rating: low to medium.** FAIL-Detect reports about 72 % balanced accuracy on hardware. We cannot test missed failures on the rig with zero real failures.

### 9. An inspection node with a view chosen for depth

- **Idea.** After the release, the right arm moves its wrist camera to a pose that sees the reader and the case edge from the side, and a scripted measure reads the visible plug length. The agent chooses the pose in sim to maximise pixels per millimetre of depth within reach and focus limits.
- **Source.** Kress-Gazit et al. (a clear success definition); ASPIRE's stated need for "robust success detection". No source gives the method.
- **Why it can work here.** The present wrist view moves about 0.8 px per mm along the port axis, so it cannot see depth. The band detector of the earlier idea 5 used that view and did not separate fully in from short. A side view at close range can give several pixels per 0.3 mm.
- **Experiment (sim, then rig).** Sim: search wrist poses after the release; report px per mm, reach, and collision margin for both wrist cameras. Rig: hand-placed plug at 0, 1, 2 and 3 mm short, 10 images each.
- **Cost.** 1 day sim; 1 rig hour; 1 day for the measure.
- **Proven wrong if.** No reachable pose gives 5 px per mm or more in sim, or the real error against calipers is above 0.3 mm.
- **Difference from the earlier idea 5.** A new view and a new node, not a detector on the existing frame.

### 10. Train the insertion policy on the outcome after the push and the release

- **Idea.** Keep the scripted final push and release in the training loop and multiply the insertion reward by the result after the release (seated and at depth). The policy then learns to end in a state from which the later nodes succeed.
- **Source.** Foresight Residual RL (abstract only: 54.5 to 85.6 % full task, per-phase success unchanged).
- **Why it can work here.** In dry runs the plug was not seated after the release on 2 of 4 seeds, with K4 and with K7, although the seat was detected.
- **Experiment (sim).** Fine-tune K7 for 30M steps with the foresight reward. Compare "seated after the release" over 64 dry-run seeds, and the K7 acceptance cells.
- **Cost.** 2 to 3 days.
- **Proven wrong if.** The after-release rate does not rise by 10 points at n = 64, or an acceptance cell falls by more than 2 points.
- **Rating: low.** The real rig kept the plug in on 10 of 10 seated runs, so the sim release effect is probably a sim fault. A policy trained to avoid a sim fault is the TI-0.1 trap. Find the cause of the sim release effect first.

## Rating of the ideas

| Rank | Idea | Rating | Reason |
| --- | --- | --- | --- |
| 1 | 5 Small contact and compliance model | High | It attacks the depth problem and the sim-fidelity problem with data we have. Two sources support the method at our data size |
| 2 | 6 Arm as force sensor, force budget | High | The force budget is already a finding from our logs. The estimator is cheap to test offline |
| 3 | 1 Node-swap attribution | High | A test with known answers exists (5 failed starts). It changes loop step 5 |
| 4 | 2 Sequential paired rule | High for its cost | One day. It decides what the next rig session can prove |
| 5 | 7 Placements from the boundary | Medium | Useful for the placement session; the source is weak |
| 6 | 4 Paired sim twin | Medium | Probably undecided at n = 10, but it turns a sentence into a measured number |
| 7 | 3 Claims ledger | Medium | Process cost each day; value depends on how many claims fail |
| 8 | 9 Inspection node | Medium | Solves the depth signal if the view exists; needs the rig to confirm |
| 9 | 8 Conformal band | Low to medium | Modest published accuracy; no real failures to test it |
| 10 | 10 Foresight reward | Low | Risk of training on a sim fault |

**Considered and not proposed.**

- *An LLM as the numerical optimiser of the graph's numbers* ([SAS-Prompt](https://arxiv.org/abs/2504.20459), [Trace](https://arxiv.org/abs/2406.16218)). SAS-Prompt tuned 8 parameters of a table-tennis controller from traces; on an 8-dimensional test function it lost to Adam (340.87 against 268.93). It gives no gain over CMA-ES for our 10 to 20 numbers, and it adds the post-hoc selection risk that the pitfalls paper measured. Use a plain optimiser.
- *Multi-fidelity Bayesian optimisation between sim and rig* (Nobar et al.). 5 against 9 real experiments for two motor gains. It becomes useful only when the rig is available and when the sim contains the effect. Later.
- *A learned delta-action plant* (ASAP). Over 400 clips for a full model. Not at our data size.
- *Real-world residual RL now.* No new source lowers its cost without a force sensor. The earlier answer ("later") stands.

## What should change in the formula

Proposed edits to the six lines. Each has its reason.

1. **Line 2 (triage): add a sixth measurement, "result observability".** Question: can the node's result be measured without the operator, at the tolerance? A node with no such measure gets a verification node before any other work. Reason: the depth problem is a measurement problem; Kress-Gazit et al. and ASPIRE both name success detection as the first need.
2. **Line 3 (funnel): add a force funnel beside the position funnel.** The push must be able to reach the mating force inside its travel limit, given the measured stiffness of arm and grasp. Reason: our own numbers (4 mm watch, 0.25 to 0.29 mm/N) cap the push near 15 N.
3. **Line 4 (real-like plant): say "a small parametric plant, fitted by distribution matching on the logs".** Do not fit an expressive residual to tens of runs. Reason: Dao and Fern; Huang et al.; ASAP's data need.
4. **Line 5 (learn from the rig): replace "traced to a node and a cause" with "assigned to a node by an intervention in the replay".** A diagnosis is accepted only when a re-run from the same state with the proposed fix changes the outcome. Reason: Who&When (14.2 % step accuracy by reading), GTP-FA (wrong-node fixes lower success), MAGMA-GEN.
5. **Line 6 (optimise): keep it, and add "the optimiser and the agent cannot write to the evaluation code or the held-out logs".** Reason: ASIA's implicit test leakage; the post-hoc selection result of the pitfalls paper.
6. **Add line 7, the evidence rule.** "Every claim has n, a re-run command and a refuting result written first. Rig comparisons are paired, interleaved and sequential on a continuous outcome. A sim result counts toward a rig claim only by the measured sim-real correlation of that node. A second agent with the raw logs repeats every claim." Reason: STEP, N-SCORE, SureSim, POPPER, the pitfalls paper.
7. **Correct the review sentence on clearance** as given in the short answer (Marougkas et al.).

## Top three experiments for the server (no rig)

1. **Fit the small contact and compliance plant on the logs (idea 5).** Start: add the axial series stiffness and a mating-force profile to the insertion env behind a config key, default off; write the distribution-matching cost on insertion and push traces; run CMA-ES with three seeds; hold out the two force pushes of 2026-11-13. Output: fitted parameters with their spread, held-out spring-back error, K4 and K7 rest depth in the fitted plant. 3 to 4 days, one GPU. This decides whether the sim can say anything about full depth.
2. **Free-space torque model and the force budget (idea 6).** Start: extract contact-free segments from all logged runs; train the predictor; report the residual in newtons along the port axis on held-out dates; recompute the force at the plug for every logged push; write the force budget table (stiffness, travel limit, largest reachable force). 2 days, CPU or one GPU. Output for the runbook: the push settings that can reach 20 N, and one 10-minute rig check with a scale.
3. **Node-swap attribution on the failures with known causes, with the sequential-test plan as a by-product (ideas 1 and 2).** Start: script the swap replay for the 5 failed starts and the three K7 hand-overs; score against the known causes and against a log-reading agent; in the same pass, compute from the logged outcomes how many pairs each planned rig comparison needs. 3 to 4 days, one GPU. Output: a measured accuracy for loop step 5, an answer to "why is K7 below K4 on bad hand-overs", and the list of comparisons that the next session can decide.

Next in line: the boundary map for K7 (idea 7), 1 to 2 days on one GPU, before the placement session.

## Sources

All pages were fetched on 2026-11-14 through a summarising fetch tool. "Full" means the full-text page was fetched and fitted in the tool's 100,000-character window. "Part" means the page was longer and only the first 100,000 characters were read. "Abstract" means only the arXiv abstract page was read. Check a number against the PDF before it goes into a paper.

| Source | URL | Fetched | Status |
| --- | --- | --- | --- |
| STEP, Snyder et al. 2025 | https://arxiv.org/html/2503.10966v1 | 2026-11-14 | Part (100k of 126k characters) |
| N-SCORE, Snyder et al. 2026 | https://arxiv.org/html/2603.13616 | 2026-11-14 | Part (100k of 180k). The search result said RSS 2026; the page does not name a venue |
| SureSim, Badithela et al. 2025 | https://arxiv.org/html/2510.04354v1 | 2026-11-14 | Part (100k of 106k) |
| Robot Learning as an Empirical Science, Kress-Gazit et al. 2024 | https://arxiv.org/html/2409.09491v1 | 2026-11-14 | Full |
| RoboGate, Kim 2026 | https://arxiv.org/html/2603.22126v1 | 2026-11-14 | Full. Its VLA numbers come from a modelled pipeline and are not used |
| Who&When, Zhang et al. 2025 | https://arxiv.org/html/2505.00212 | 2026-11-14 | Full. 53.5 % and 14.2 % are the abstract's best values; the GPT-4o table values are lower |
| GTP-FA, Xu et al. 2026 | https://arxiv.org/html/2606.03385 | 2026-11-14 | Part (100k of 178k). Sim n not seen. Attribution accuracy is not reported as a number |
| MAGMA-GEN, Bernat et al. 2026 | https://arxiv.org/abs/2609.20056 | 2026-11-14 | Abstract only. No numbers seen |
| Hidden Pitfalls of AI Scientist Systems, Luo et al. 2025 | https://arxiv.org/html/2509.08713v2 | 2026-11-14 | Part (100k of 119k) |
| POPPER, Huang et al. 2025 | https://arxiv.org/html/2502.09858 | 2026-11-14 | Part (100k of 156k) |
| ASPIRE, Lu et al. 2026 | https://arxiv.org/abs/2607.00272 and https://arxiv.org/html/2607.00272v2 | 2026-11-14 | Abstract, and part of the HTML (100k of 120k). Verifier trial counts are not stated |
| Agent-driven RL research, Khandelwal and Gupta 2026 | https://arxiv.org/html/2603.27416v1 | 2026-11-14 | Full. The abstract and the body disagree on the number of repeat runs |
| ASIA, Piga and Forgione 2026 | https://arxiv.org/html/2605.10480v1 | 2026-11-14 | Full |
| E-valuator, Sadhuka et al. 2025 | https://arxiv.org/html/2512.03109v1 | 2026-11-14 | Part (100k of 113k). Named only as the sequential form for idea 8 |
| Simulator adaptation by distribution matching, Dao and Fern 2026 | https://arxiv.org/html/2604.11090 | 2026-11-14 | Full. Finetuning results of the expressive models are in plots only |
| Few-shot neural differentiable simulator, Huang et al. 2026 | https://arxiv.org/html/2603.06218v2 | 2026-11-14 | Full. Brax numbers are in a figure only and are not cited |
| ASAP, He et al. 2025 | https://arxiv.org/html/2502.01143v3 | 2026-11-14 | Part (100k of 111k). The conclusion's "52.7 %" is not supported by the tables seen and is not cited |
| AdaptSim, Ren et al. 2023 | https://arxiv.org/abs/2302.04903 | 2026-11-14 | Abstract only. The number of real rollouts was not seen |
| Contact-Aware Neural Dynamics, Jing et al. 2026 | https://arxiv.org/html/2601.12796v1 | 2026-11-14 | Full. Real data amount and n are not stated |
| Guided multi-fidelity BO, Nobar et al. 2025 | https://arxiv.org/html/2509.17952v1 | 2026-11-14 | Full. The HTML shows "99 and 77" experiments; 9 and 7 match the stated gains |
| Real-is-Sim, Abou-Chakra et al. 2025 | https://arxiv.org/html/2504.03597v1 | 2026-11-14 | Full |
| FACTR 2 (NEXT), Oh et al. 2026 | https://arxiv.org/html/2606.12406 | 2026-11-14 | Full. No force error in newtons. The role of the YAM is teleoperation |
| Minimalist Compliance Control, Shi et al. 2026 | https://arxiv.org/html/2603.00913 | 2026-11-14 | Full |
| NeuralActuator, Dou et al. RSS 2026 | https://people.csail.mit.edu/frankzydou/projects/NeuralActuator/index.html | 2026-11-14 | Project page only; the paper PDF was not read |
| Visual force for compliant grippers, Zuo et al. 2026 | https://arxiv.org/abs/2605.00307 | 2026-11-14 | Abstract only |
| Marougkas et al. 2025 | https://arxiv.org/html/2505.11858v1 | 2026-11-14 | Full. "No force sensor" means that none is described; the comparison counts with IndustReal are in a figure |
| Park et al. 2025 (flat cable) | https://arxiv.org/html/2509.13731v1 | 2026-11-14 | Full. "No force sensor" means that none is described |
| Dynamic Compliance Tuning, Zhang et al. 2023 | https://arxiv.org/html/2311.07499v3 | 2026-11-14 | Full |
| Foresight Residual RL, Liu et al. 2026 | https://arxiv.org/abs/2607.16506 | 2026-11-14 | Abstract only. n not seen |
| FAIL-Detect, Xu et al. 2025 | https://arxiv.org/html/2503.08558v3 | 2026-11-14 | Part (100k of 112k). The number of calibration rollouts was not seen. The search result said RSS 2025; the page does not name a venue |
| SAS-Prompt, Ben Amor et al. 2025 | https://arxiv.org/html/2504.20459v1 | 2026-11-14 | Full |
| Trace / OPTO, Cheng et al. 2024 | https://arxiv.org/abs/2406.16218 | 2026-11-14 | Abstract only. The robot experiment was not read |
| SafeOpt, Berkenkamp et al. 2016 | https://arxiv.org/abs/1602.04450 | 2026-11-14 | Abstract only. Not used as evidence |
| ReBound (reset-free driving), Honda et al. 2026 | https://arxiv.org/abs/2604.07672 | 2026-11-14 | Abstract only. A search result said that residual learning from sim failed on the real car; the abstract does not say this. Not used |
| USB Type-C insertion force 5 to 20 N | https://cjt.com/ps/USB_TYPE-C_SPEC_CJ-SPEC.pdf ; https://www.mouser.lt/pdfDocs/susb013.pdf | 2026-11-14 | **Not verified.** The first PDF has no readable text; the second timed out; the Wikipedia page read gave no force value. The range comes from search-result snippets of vendor data sheets only |

**Found in the search and not fetched; not used as evidence.** Guardian (arXiv:2512.01946), FIPER (arXiv:2510.09459), Sentinel, Dream2Fix (arXiv:2603.13528), FATE-VLA (arXiv:2606.02307), RoboFail (arXiv:2412.02818), Prim-LAfD (arXiv:2212.00955), ShapeForce (arXiv:2511.19955), "Forces for free" (Science Robotics 2025), an embodied-AI agent benchmark (arXiv:2601.21570), AutoResearch-RL (arXiv:2603.07300), BayesContact (arXiv:2607.16123), Neural Posterior Domain Randomization, BayesSim (arXiv:1906.01728), a multi-fidelity vehicle-controller paper (arXiv:2506.08719).

**Search gaps.** No source was found for: automatic reset or autonomous practice for insertion without a force sensor (2025 to 2026); depth or seat estimation of a connector from RGB alone; identification of arm and grasp compliance for a low-cost arm; an agent loop for robotics with a statistical evidence rule. No paper first published in November 2026 was found.

**Our own numbers.** From `docs/campaign/FINDINGS.md` (entries of 2026-11-11 to 2026-11-14), `docs/real/K7_20261114.md`, `docs/real/DEPTH_SPRINGBACK_20261114.md` and `docs/real/ZED_EDGEFIT_20261115.md` on `real/graph-align`, and from `AGENTIC_RL_FORMULA_20261113.md`. The force budget in idea 6 (about 14 to 16 N) is this page's arithmetic from the logged 0.25 to 0.29 mm/N and the 4 mm advance watch; it has not been checked on the rig.
