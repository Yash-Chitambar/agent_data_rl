# Session 2026-11-03, duchamp3: first live align run stopped on a jump in the approach

Branch `real/graph-align` (`c545340`), config `deploy/config/bimanual_align2.json`, guide [NEXT_RIG_SESSION.md](NEXT_RIG_SESSION.md). Run logs: `data/real_runs/2026-11-03/` (gitignored). Video: `IMG_8788.MOV`.

## Status

Stopped in step 2 after one run, by the guide's stop condition "jumps by more than about 5 mm". `measure2`, step 3 and step 4 did not run. Limits and configs did not change.

## Runs

| Time | Run | Command | Outcome | Operator saw |
|---|---|---|---|---|
| Before 14:48 | setup checks | as in the guide | all pass; timing p99 at most 11.7 ms per stage | |
| Before 14:48 | `camera_check` | step 1, empty jaws | camera opens, 0 of 30 (no plug in view, expected) | |
| 14:48 | `full_144822_measure1` | `--stage full --speed 0.5 --config $C --align-mode measure --name measure1` | pick held, stage measured, backed out, no contact in the log | "very close to usbc port and aligned, then started moving away and down"; at the stop the plug was lower than the port, tilt "slightly down but basically flat" |

## Measurements at the stop (10 cycles, 50 of 50 frames, arm still)

| Source | Across mm | Up mm | Gap mm | rx, ry, rz deg |
|---|---|---|---|---|
| Port vision | -0.49 | -4.96 | 4.59 | -4.77, -1.18, -1.50 |
| `wrist_r10f_ft_abs` | -1.13 | -3.01 | 6.00 | -0.77, +0.10, +0.74 |

Spread at most 0.35 mm per axis, SD at most 0.08 mm. Time per frame: median 85 ms, one first frame 455 ms. Shell depth 121.08 mm, accepted. The port is the one jogged into on 2026-11-02: a `collect` frame at about the same offset shows the same view.

Plug against the panel in the old-stop offline numbers: rx -3.2 to -3.4, ry -1.8 to -2.4, rz -1.1 to -1.3 deg.

## The jump, late in the approach

Port axes, relative to the stop. Encoder FK with joint offsets. Ticks are 20 ms.

| Tick | Across | Up | Gap axis | Commanded up | Joint 2, 3 target - measured mrad | Vision across, up, gap | Net across, up, gap |
|---|---|---|---|---|---|---|---|
| 2740 | +1.9 | +3.6 | +0.1 | +8.5 | +15, +21 | +1.2, -2.0, 4.3 | +0.94, +1.09, 5.96 |
| 2764 | +1.8 | +3.7 | +0.1 | +8.0 | +18, +23 | | |
| 2776 | -1.9 | +6.9 | -0.5 | +7.7 | +20, +11 | | |
| 2800 | -0.5 | 0.0 | +0.1 | +7.5 | +2, +11 | | |
| 2820 | -0.8 | +0.2 | 0.0 | +7.6 | +2, +10 | -1.4, -5.1, 4.4 | -1.90, -2.74, 5.97 |

- Joint 3 lets go first (tip up 3.2 mm, sideways 3.7 mm in 0.24 s). Then joint 2 lets go (tip down 6.9 mm in about 0.4 s). The commanded tip moves 0.5 mm in this time.
- Before it, the arm stayed at about the same place from tick 2339 to 2764 while the commanded height fell from +16.3 to +8.0 mm: stuck in friction while the servo moved the target. Joint 2 error went from -25 to +18 mrad.
- The net read 2.4-3.1 mm above vision in up on these frames, so it still asked to go down when vision read 2 mm low.
- Vision followed FK through the approach within about 1.5 mm in up, on 12 offline frames: this is the first evidence that it tracks the live motion.
- Largest 0.5 s move in the last 300 approach ticks of 2026-11-02: A 3.0, A2 4.1 mm. The 0.5 s window here gives 6.9 mm or more.
- Video: the drop is visible between about 23 and 24 s.
- No contact in the log: the gap axis stayed within 0.5 mm. The commanded tip was 2.2 mm deeper than the measured tip, so the same jump could close the gap by about that much.

## Back-off is not straight

Measured: out 16.4 mm, down 6.6 mm, across 2.4 mm. Commanded: straight out (up constant).

- First 0.5 s: 4.6 mm down. The stage sets the target to the measured pose and so removes the 34 mrad holding lead on joint 5 (kp 10). The wrist sags until the error rebuilds to 15-21 mrad.
- Then joints 2 and 3 never move: errors reach only +4 and +10 mrad, inside the friction band. The wrist joints make the whole motion.
- Lowest plug tip in the back-off 20 mm above the tabletop. No clearance problem in this run.

## Other notes

- **Tilt not settled.** Vision says rx -4.77 deg, the net -0.77, the operator "basically flat". In the video the body's edges slope about 0.5-2.5 deg, tip down, but the view is oblique and hand-held, with no level reference, so this does not separate them. Step 3 turns the plug by the vision value. It needs a square-on side view first.
- **Grip reading.** 0.324 at the squeeze, 0.316 from the approach on. All 13 picks on 2026-11-02 read 0.308-0.315. Inside the held band; no gate fired. One run.
- **Log artefact.** In measure mode `rot_applied`, `fused` angle and `turned` grow without motion (6.75 deg at cycle 10), and `slope` means nothing. Use the raw `rv_deg`. If `measure_n` were higher this would stop the stage at `max_deg`.
- **Jaws at the end.** They open at the prompt "Hold it, then Enter opens the gripper", as written. The guide's "closed jaws" covers the back-off.

## Questions for the server

1. How should the approach end so that the joints do not store this error and let go near the case? The creep controller runs only from the align stage on.
2. Is a stop 5 mm low, with a gap near 4.6 mm, an acceptable entry for the align stage?
3. Why does the net read about 3 mm above vision on these frames? Which one is right, given the label constant?
4. Should the back-off keep the wrist lead?
