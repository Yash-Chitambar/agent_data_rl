# Measuring the real YAM's latency and backlash (for Gate C)

**Why.** Phase C (`results/C/C-matrix.png`) shows that two robot properties decide whether a
sim-trained policy works on the arm:

- **Latency.** A policy trained without latency gets 16–24% when 1–2 control steps (20–40 ms)
  of latency are present. A policy trained with 1 step gets 99% at 1 step but only 33% with no
  latency. A fixed latency in training must match the robot to within one 20 ms step.
- **Backlash.** A policy that never felt backlash gets 43–47% at 1–5 arcmin. Training with any
  level from 1 to 5 arcmin covers all three levels.

Sensor noise and encoder quantization do not matter. These tests give the two numbers. They also
give the real gains and the encoder noise at no extra cost. The tests take about 1–2 hours.

## What the sim assumes (so the numbers mean the same thing)

| Property | Sim model | Code |
|---|---|---|
| Control | 50 Hz (20 ms steps). Each step: `q_target = q_measured + 0.04 * action`, held for 20 ms | `envs/peg_env.py` `step()` |
| Joint servo | PD per joint, `kp = 80, 80, 80, 30, 30, 30` N·m/rad, `kd = 6, 6, 6, 2, 2, 2` N·m·s/rad; full gravity compensation | `assets/builder.py` (`"yam"`) |
| Latency `k` | The action applied now was computed `k` steps (k x 20 ms) earlier. `k` is an integer; 0 means "the command acts the instant the encoders are read" | `EnvCfg.latency` |
| Sub-step latency | `latency_ms` more: for that many 1 ms substeps the arm keeps the previous step's targets. Total delay = 20 ms x `latency` + `latency_ms` | `EnvCfg.latency_ms` |
| Backlash | A free play of ±(arcmin/2) between each motor and its link. The encoder reads the motor side only | `builder.add_backlash`, `EnvCfg.backlash_arcmin` |

So **latency** = (age of the encoder reading when the policy uses it) + (time to compute the
action) + (time until the motor starts to move). **Backlash** = the link motion that the encoder
does not see.

## Safety and setup

- The arm is in free space: no gripper load and no fixture near it. Keep the e-stop in reach.
- Start from the lab's insertion pose (the arm above the table, as in deployment). Record that
  pose; the script simulates from the same pose.
- Use the **same software stack as deployment**: the same driver, the same 50 Hz loop thread and
  the same PC. Latency comes from the stack, so a separate test script with its own timing
  measures the wrong thing.
- Use the sim gains (table above) in MIT mode with gravity compensation on. If deployment
  uses other gains, run the tests with the deployment gains and write them down: the gains matter
  too (C matrix: 20% success at 4x the training gains).
- Steps are 0.04 rad (2.3°), one joint at a time. That is one full policy action.

## Test 1: loop timing (10 min, no motion)

Run the deployment control loop with zero action (hold the pose) for 60 s. Log per control step:

| Column | Meaning |
|---|---|
| `t_obs` | Host time (s, monotonic clock) when the encoder reading that the policy uses **arrived** from the motor |
| `t_act` | Host time when the action was ready (after inference, or after a dummy model of the same size) |
| `t_send` | Host time when the command was written to the bus |

Save as `data/real_yam/loop_timing.csv`. The loop part of the latency is `t_send - t_obs`. The
spread between steps (jitter) matters as much as the mean.

The same 60 s of encoder data gives the encoder noise. Also save it as
`data/real_yam/hold_60s.csv`, in the step-test format below.

## Test 2: joint step response (30 min)

For each joint 1–6, in the deployment loop:

1. Hold for 1 s.
2. Command `+0.04 rad` on that joint only, then hold for 1 s.
3. Command back (`-0.04 rad`), then hold for 1 s.
4. Repeat 10 times.

Log **every encoder reading** the driver gives, not only one per control step. Aim for at
least 250 Hz; write down the real rate. Use one CSV per joint, `data/real_yam/steps_joint<J>.csv`:

| Column | Meaning |
|---|---|
| `t` | Host time (s, monotonic) when this encoder reading arrived |
| `q_cmd_1` … `q_cmd_6` | Target position last **sent** for each joint (rad) |
| `q_1` … `q_6` | Measured position for each joint (rad) |
| `t_cmd` (optional, recommended) | Host time when the current `q_cmd` was sent |

Then run:

```bash
.venv/bin/python scripts/real_yam_latency.py fit 'data/real_yam/steps_joint*.csv' --out results/C/real_latency.json
```

The script finds each commanded step and simulates the same step from the same pose with the sim
model. It then fits two numbers per joint: the **delay** (ms) and a **gain scale** (real stiffness
/ sim stiffness). `scripts/real_yam_latency.py selftest` checks the fit on fake data with a known
answer (23 ms, 0.7x).

The delay here covers everything from the send to the first encoder motion: bus, motor driver,
encoder read. It does not include the loop part from Test 1.

## Test 3: backlash (30–60 min)

The encoder must stay still while the link moves. Two methods:

**Method A (preferred): dial indicator on the link.** You need a dial indicator with 0.01 mm
resolution (0.001 mm is better) on a magnetic base fixed to the table.

1. Hold the pose with the motors on (deployment gains, or stiffer).
2. For joint J, put the indicator tip on the link just after joint J. It must point tangent to
   joint J's rotation, at a measured distance `r` (mm) from J's axis. A point close to the next
   joint avoids most of the play of the other joints.
3. Push the link gently by hand toward the indicator, then pull it away. Use only enough force to
   reverse it (about 1–3 N). Read the indicator at each end. Log the encoders at the same time.
4. Do this 5 times. Then repeat at a second force (for example 1 N and 3 N with a push-pull
   gauge). Backlash is the play extrapolated to zero force; the rest is compliance.
5. Do joints 1–6. Then do joint 1–3 again at a second pose.

Compute the play per joint:

```
theta_rad = (dx_indicator_mm - r_mm * dq_encoder_rad) / r_mm
backlash_arcmin = theta_rad * 3437.75
```

For scale: 1 arcmin at r = 100 mm is 0.029 mm, and 5 arcmin is 0.15 mm.

**Method B (no indicator): clamp the link, move the motor.** This works only if the encoder is
on the motor side of the gearbox. Check the motor datasheet first: with an output-side encoder,
this method shows almost no play even when play exists.

1. Clamp the link rigidly to the table.
2. Command the motor slowly back and forth, ±0.5° at 0.1°/s, with a low torque limit.
3. Log the encoder and the motor current.
4. The backlash is the width of the flat zone, where the encoder moves but the current does not
   rise.

Record the results in `data/real_yam/backlash.csv`:

| joint | pose | r_mm | force_N | dx_mm | dq_rad | backlash_arcmin | method |
|---|---|---|---|---|---|---|---|

## What I do with the numbers

Total latency per joint: `L = mean(t_send - t_obs) + delay_ms`. In control steps: `k = L / 20 ms`.

| Measurement | Training setting for D/E |
|---|---|
| `k` within ±0.25 of an integer **and** jitter (std) < 5 ms | Fixed `latency = round(k)` |
| `k` between integers (for example 0.5) | Fixed sub-step delay: `latency = floor(k)`, `latency_ms` = the rest (1 ms resolution) |
| Jitter ≥ 5 ms | Randomize `latency_ms` per world over the measured range (`noise_rand: {"latency_ms": [...]}`) |
| Backlash ≤ 5 arcmin on all joints | `backlash_arcmin` randomized over 0 to the max measured (C shows any level covers 1–5) |
| Backlash > 5 arcmin | Outside what C tested: I add a C run at the measured level before D |
| Gain scale outside 0.7–1.4 | The sim gains are wrong: I change the YAM gains in the builder and retrain (the gain column of the C matrix shows the risk) |
| Encoder noise (std of the 60 s hold) | Sets `q_noise`; C shows 1e-4 to 5e-4 rad is harmless |

## What to send back

Put these in `data/real_yam/` (small CSVs; commit them) or send them to me:

- `loop_timing.csv`, `hold_60s.csv`, `steps_joint1.csv` … `steps_joint6.csv`
- `backlash.csv`
- Notes: gains used, motor model per joint, encoder rate, PC and driver version, poses used, and
  anything odd (missed bus frames, warnings).
