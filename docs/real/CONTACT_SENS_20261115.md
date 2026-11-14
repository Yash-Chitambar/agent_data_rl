# Contact-model sensitivity and funnel sizing of K4 and K7 (2026-11-15)

Sim only. Nothing here ran on the arm. No config, limit or weight changed. No new config: test 2 does not support a looser alignment tolerance (section 2).

Policies: K4 (`runs/insert_K4.pt`, on the arm) and K7 (`runs/insert_K7.pt`, wide-placement candidate, `K7_20261114.md`).

## How the tests ran

- Start: cell `real_ho` of `k2_matrix.py` (gap 5.5 mm, lateral ring 0.5 mm, angles from the chain pool), real right-arm lead, real-like plant (joint friction x1.3-2.0, gravity error 12 %), no mating force, training-only DR off. Place `real` = rig spot (layout v2, PC +/-5 mm, 4 +/- 1.5 deg). Place `box` = step-2 box (PC +/-25 mm, 4 +/- 15 deg), uniform per world.
- n = 1024 worlds per cell, first episode of each world. Several contact variants share one batch (per-world parameters, set after the reset). Seeds: test 1 8100, test 2 8300.
- Probe per world (`scripts/eval/contact_sens.py`), deploy-like:
  - **Seat**: the sim seat test held 0.2 s (the usual success). **Seat time**: its time.
  - Seat detector on the true depth (depth >= 4.8 mm, axial speed < 6 mm/s for 0.3 s). When it fires, the policy stops (action 0, target = measured pose), as deploy does. **Full depth**: depth >= 5.55 mm (seat 5.8 - 0.25) after a 0.2 s rest.
  - **Release probe**: then the right jaws open, the right arm holds, the left press target goes to its measured pose; depth 0.5 s later.
  - **Failure class**: class of the episode end (`failure_class`). New class `full_no_seat_test`: fully in at rest, but the aperture-polygon seat test failed (contact penetration), not an insertion failure.
- The policy stops at the detector here. In `k2_matrix.py` it pushes on until the seat test holds. So K4's base seat rate is 94.6 % here against 97.4 % there (more `grasp_slip` after the stop). K7 is the same in both (98.7 against 98.9 %). Compare cells within this doc.
- MuJoCo Warp is not bitwise deterministic. The 95 % CI half-width at 95 % is about 1.4 points.

## 1. Contact-model sensitivity

Full table (every cell, CI, seat time p50/p99, failure classes): `results/real_align/contact_sens/test1.md` (`.json`).

Training randomisation of both policies (`DEFAULT_CONTACT_DR`): port mu U(0.10, 0.20), port solref U(3, 4) ms, damping ratio 1, port solimp (0.99, 0.999); pad mu 1.2, pad solref 2 ms, squeeze x1.0, pads 4x4 spheres r 1.2 mm, plug at nominal fit (45 / 80 um per side). **Only the `base` cells are inside the training randomisation. Every other cell is outside.**

### Summary, place `real` (n = 1024 per cell)

| Variant | K4 seat % | K4 full % | K7 seat % | K7 full % | Main failure |
|---|---|---|---|---|---|
| base (training) | 94.6 | 89.5 | 98.7 | 92.0 | - |
| port mu 0.05-0.08 | 98.4 | 91.1 | 98.3 | 90.3 | - |
| **port mu 0.35-0.60** | **33.9** | **34.6** | **53.0** | **52.7** | lateral_miss (583 / 408) |
| **port mu 0.80-1.00** | **0.9** | **0.9** | **7.1** | **6.7** | lateral_miss |
| port solref 2 ms | 94.1 | 87.3 | 98.5 | 91.3 | - |
| port solref 8 ms | 76.5 | 93.8 | 88.2 | 93.1 | full_no_seat_test (penetration) |
| port damping 0.5 | 92.8 | 87.8 | 98.5 | 92.4 | - |
| port solimp (0.90, 0.95) | 54.3 | 95.4 | 51.1 | 93.5 | full_no_seat_test (penetration) |
| pad mu 0.8 | 96.0 | 91.2 | 99.0 | 92.3 | - |
| pad mu 2.0 | 88.6 | 83.2 | 94.9 | 88.8 | grasp_slip, lateral_miss |
| pad solref 6 ms | 86.9 | 83.6 | 95.6 | 89.8 | lateral_miss, grasp_slip |
| squeeze x0.6 | 83.8 | 81.6 | 88.9 | 84.5 | lateral_miss, grasp_slip |
| squeeze x1.4 | 95.7 | 90.9 | 99.0 | 93.6 | - |
| pads 2x2 r 1.2 mm | 89.3 | 85.5 | 93.4 | 87.8 | lateral_miss |
| pads 2x2, port mu 0.35-0.60 | 28.3 | 28.9 | 49.9 | 49.5 | lateral_miss |
| pads 6x6 r 0.8 mm | 96.6 | 90.8 | 99.7 | 93.1 | - |
| pads 6x6, port mu 0.35-0.60 | 34.7 | 34.9 | 53.0 | 52.5 | lateral_miss |
| **tight plug (20 / 50 um)** | **65.7** | **64.5** | 92.1 | 88.8 | grasp_slip (285 / 66) |
| tight plug, port mu 0.35-0.60 | 8.7 | 8.6 | 34.9 | 34.6 | lateral_miss, grasp_slip |
| loose plug (90 / 110 um) | 45.5 | 84.9 | 42.2 | 88.0 | full_no_seat_test |
| loose plug, port mu 0.35-0.60 | 18.6 | 30.9 | 23.8 | 48.7 | lateral_miss |

K7 in the box (n = 1024 per cell) follows K7 at the real spot within 0-4 points in every variant (base 98.7 %, port mu 0.35-0.60 55.6 %, 0.80-1.00 8.7 %, squeeze x0.6 88.5 %, tight 90.0 %). K4 in the box: base 74.3 %, port mu 0.35-0.60 23.1 % (the box is outside K4's training).

Seat time: port mu 0.35-0.60 doubles the median seat time (K4 1.32 -> 2.12 s, K7 1.10 -> 2.04 s). The other variants change it by 0.2 s or less, except the tight plug for K4 (2.38 s).

### Findings

1. **The result rests on the port friction.** Port mu 0.35-0.60 halves K7 (98.7 -> 53.0 %) and cuts K4 to a third (94.6 -> 33.9 %). At 0.8-1.0 both fail (0.9 / 7.1 %). The failure is `lateral_miss`: the plug does not slide into the opening. Lower friction (0.05-0.08) changes nothing. The training range (0.10-0.20) is low for plated metal on metal (dry steel on steel is about 0.4-0.8; no measurement on the rig). The 10 of 10 real seats on 2026-11-11 show that the real port does not act like sim mu 0.35+ in this model, but they do not give the value.
2. Second: the fit. A tight spec plug (20 / 50 um) costs K4 29 points (grasp_slip: the plug sticks and the reader turns in the jaws), K7 7 points. Tight + high friction: 8.7 / 34.9 %.
3. Third: the grasp. Squeeze x0.6 costs 10-11 points (both). Pad mu 2.0, pad solref 6 ms and 2x2 pads cost 3-8 points. 6x6 small pads are equal to the base. The pad model is not a main risk; the squeeze force is.
4. Port stiffness and damping (solref 2 ms, damping 0.5) change nothing. Softer port contact (solref 8 ms, solimp 0.90/0.95) and the loose plug only break the sim seat test (penetration past the 25 um aperture tolerance): full depth stays at 85-95 %. The seat test, not the policy, is sensitive there.
5. **K7 is less sensitive than K4** in every out-of-training cell (port mu 0.35-0.60: 53.0 against 33.9 %; tight plug: 92.1 against 65.7 %; squeeze x0.6: 88.9 against 83.8 %). K7 has the same sensitivity order as K4. K7 in the box is as robust as K7 at the real spot.

### The release effect ("not seated after release")

Of the worlds fully in at rest, the share still fully in 0.5 s after the jaws open (`results/real_align/contact_sens/release.md`, `scripts/eval/contact_sens_release.py`):

| Condition | K4 | K7 |
|---|---|---|
| base, real spot | 96.5 % (884/916) | 97.8 % (921/942) |
| tight plug | 99.1 % (654/660) | 99.2 % (902/909) |
| loose plug | 86.0 % (747/869) | 87.3 % (787/901) |
| port mu 0.35-0.60 | 98.6 % (349/354) | 99.6 % (537/539) |
| port solimp soft | 98.9 % | 98.7 % |
| pads, squeeze, pad mu | 95.3-97.0 % | 97.9-99.0 % |

- The depth change at the release is small: p50 0.01-0.03 mm, p99 0.1-0.25 mm. The probe sees the plug move back a little. It does not see a pull-out.
- The loss scales with the fit: tight 99 %, nominal 96-98 %, loose 86-87 %. Higher port friction keeps the plug in. Pads and squeeze do not change it.
- So the probe points to the plug sliding back in a loose fit with low port friction when the hand load goes. The dry-run effect on seeds 7041/7042/7044/7045 may be the same mechanism made larger by the retreat (the probe holds the arm, it does not retreat). Not proven: a dry run with a tight plug or port mu 0.3 on those seeds would test it.

## 2. Funnel sizing: 1 mm or 2 mm alignment tolerance

Script `scripts/eval/funnel_rings.py`. Table: `results/real_align/contact_sens/test2.md`. Two prior modes per ring:
- `truth`: true start on the ring, prior = truth + N(0, (0.4, 0.95, 0.63) mm) (as in training);
- `real`: the vision measurement is on the ring (where the alignment stopped) and is the prior; the true start = measurement + N(0, (0.4, 0.95, 0.63) mm) (model of `replay_real_handovers.py`).

Seat % (seat time p50 s), n = 1024 per cell:

| Ring mm | K4 truth | K4 real | K7 truth | K7 real |
|---|---|---|---|---|
| 0.5 | 95.1 (1.28) | 93.4 (1.34) | 99.2 (1.08) | 97.8 (1.20) |
| 1.0 | 94.8 (1.30) | 93.0 (1.36) | 98.9 (1.10) | 97.8 (1.22) |
| 1.5 | 93.8 (1.30) | 90.7 (1.44) | 98.7 (1.16) | 97.2 (1.28) |
| 2.0 | 94.1 (1.36) | 90.9 (1.48) | 98.9 (1.22) | 96.8 (1.32) |
| 2.5 | 93.8 (1.42) | 92.4 (1.50) | 98.8 (1.26) | 96.2 (1.36) |
| 3.0 | 92.3 (1.48) | 90.9 (1.62) | 98.2 (1.34) | 96.2 (1.42) |

Full depth is 86-95 % in every cell, with no trend over the ring.

Offline replay of 23 real alignment stages (2026-11-10/11/13, `scripts/eval/align_tol_replay.py` -> `results/real_align/contact_sens/align_tol_replay.json`): against the same replay with the 1 mm rule, a 2 mm lateral tolerance stops at the same cycle in 19 runs and 1-2 cycles earlier in 4 (0.2-1.2 s saved). At the real hand-over the lateral error was 0.9 mm or less in all 23 runs. The gap target, the angle tolerance and the slope probes set the stage length, not the lateral tolerance.

### Verdict: falsified, keep 1 mm

- The rule: falsified if the 2 mm ring is under 98 % or the seat time doubles. Seat time does not double (+10-13 %). But the 2 mm ring is under 98 % for K4 in both modes (94.1 / 90.9 %) and for K7 in the real-prior mode (96.8 %).
- The 98 % bar is also missed at 0.5 mm by K4 (both modes) and K7 real (97.8 %) with this strict probe. Relative to the 0.5 mm ring, 2 mm costs K4 1.0 / 2.5 points and K7 0.3 / 1.0 points; a "2 mm per axis" stop allows the corner (2.8 mm, ring 3): K4 -2.8 / -2.5, K7 -1.0 / -1.6 points.
- The real logs show no gain: the lateral tolerance does not bind on the rig.
- So no new config. `bimanual_align10_all` keeps `align.tol_mm` [1, 1, 1]. If K7 goes on the arm, a 1.5 mm tolerance would cost under 1 point in the sim, but it would save almost no time on the rig.

## What changed

New files only: `scripts/eval/contact_sens.py`, `funnel_rings.py`, `contact_sens_all.sh`, `contact_sens_table.py`, `contact_sens_release.py`, `align_tol_replay.py`; results in `results/real_align/contact_sens/` (tables `test1`, `test2`, `release`, `align_tol_replay.json`, per-process summaries in `raw/`). Per-world arrays and logs: `runs/contact_sens/` (gitignored). The scene patches (pad spheres, plug outline, per-world port solimp) are monkeypatches inside the eval process; no library code changed.

## Limits

- Sim only. The contact values are guesses; no cell is a measurement.
- The probe stops the policy at the depth detector, as deploy does. Absolute rates differ from `k2_matrix.py` for K4 (section "How the tests ran").
- The seat test fails under soft port contact and the loose plug (penetration). Use full depth for those cells.
- The release probe holds the arm still; it does not model the retreat.
- Box pads were not tested: the contact randomisation and the grasp layout need the pad spheres.

## Proposed next steps (server, sim only)

1. Train a K7 variant with port mu U(0.10, 0.60) and the spec fit range (20-90 um), then re-run this grid. It needs no rig time.
2. Dry runs on seeds 7041/7042/7044/7045 with a tight plug and port mu 0.3 (scratch configs) to test the release mechanism.

## Rig runbook

Nothing to run on the rig from this doc. No config is new; no approval is asked. For the operator, one optional measurement helps the sim: **insertion and extraction force of the reader in port 2** (kitchen scale, step 0 of `NEXT_RIG_20261115.md`). A high extraction force with a low insertion force points to low port friction with spring retention; send both numbers.

Re-run the sim (server):

```bash
scripts/eval/contact_sens_all.sh t1_k7 1      # lanes: t1_k4 t1_k7 t1_k7box t1_k4box t2_k4 t2_k7; about 10 min per 4096-world job on a V100
python scripts/eval/contact_sens_table.py && python scripts/eval/contact_sens_release.py
python scripts/eval/align_tol_replay.py --out results/real_align/contact_sens/align_tol_replay.json
```

Pass for a new policy on this grid: base >= 98 % and port mu 0.35-0.60 >= 90 % seat (real spot, n = 1024). Stop: base below the policy it replaces.
