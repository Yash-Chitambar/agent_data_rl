# Research direction

## Question

How do we use the agent **GaP harness + RL in simulation** to train a **real** policy that
is aware of contact, on robots without force sensing (YAM)?

## Hypothesis

A policy trained only in simulation transfers to a contact-rich real task when the
simulator matches the real robot on the three axes the key-box study exposed
([slides](keybox_sim2real.pptx)): **motion** (the arm lands where the real one lands),
**perception** (sim cameras are at least as bad as the real ones) and **physics** (contact
priors from real runs, demos and facts a person states). Proprioception carries the contact
signal: the single-arm reproduction showed that joint signals alone suffice for precise
insertion in sim (below). The working claim is unproven on hardware.

## Pipeline

```
real geometry, motion, images ──> calibrated simulator (MuJoCo Warp, GPU-batched)
        ^                                   │
        │                                   v
real tests, failures,            privileged PPO teacher (true poses, contact labels)
language facts                              │
        ^                                   v
        └──── zero-shot deploy <── student: encoders + RGB, sim cameras >= real noise
```

Each stage has a gate in [CURRENT](../campaign/CURRENT.md). Keep three kinds of evidence
separate: simulated skill, causal match to real measurements, real deployment results.

## Task

Right YAM picks a SanDisk USB-C SD reader and inserts it into port 2 of a GMKtec mini PC;
left YAM holds the PC (brace the rear face or press the top). The PC rests freely on the
table. Clearance is sub-0.1 mm; there is no force or tactile sensing. Plan of record for
the measurement-driven transfer: [USB_C_SIM2REAL_PLAN](USB_C_SIM2REAL_PLAN.md); what the
key-box lessons mean here: [KEYBOX_LESSONS_FOR_USBC](KEYBOX_LESSONS_FOR_USBC.md).

## Foundation: single-arm blind insertion (complete)

Reproduction of Bronars et al., *Blind Precise Assembly* ([PDF](bronars_blind_precise_assembly_2610.23849.pdf)),
then extension to the YAM and USB-C. Highlights (all in sim; detail in
[FINDINGS](../campaign/FINDINGS.md), plan in [archive](../archive/RESEARCH_PLAN_single_arm.md)):

- Blind (joints only) peg insertion 99.9 % vs paper 99.4 % (2,048 unseen geometries); visual 99.95 %.
- On the YAM, latency and backlash decide transfer; policies trained with per-world
  randomized latency, noise, quantization and backlash hold 96–99 % on every such test.
- USB-C (20–100 µm clearance) 91.5 % with a clearance curriculum and sphere pads.
- Single-arm task H (fixed PC): scripted expert, ACT pick + PPO insertion, 85.5 % inserted
  end to end in sim. Superseded by the bimanual task with a free PC.

## Experiments that answer the question

1. **Calibrated vs nominal + DR** at equal compute: does matching motion/contact beat broad
   randomization for real success?
2. **Which gap matters:** ablate dynamics identification, camera/fingertip alignment and
   contact priors one at a time (sim score and real score).
3. **Contact awareness:** remove the contact-label auxiliary loss and the target-error
   observation; measure recovery from missed insertions in sim and on the rig.
4. **Recipe reuse:** the next task's real-data cost given this calibrated cell.
