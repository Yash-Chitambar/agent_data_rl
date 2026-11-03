# Classical port and plug measurement from the right wrist camera

`deploy/bimanual/port_vision.py`. No net, no encoders and no in-hand estimate at run time. One raw 1280x1024 frame gives the plug tip relative to port 2. Offline only: it never ran in a live loop.

```python
from deploy.bimanual.port_vision import measure, plug_depth
m = measure(raw_bgr)                                  # side='right' only
m = measure(raw_bgr, plug_z_mm=plug_depth(frames))    # better: one shell depth per grasp
# m.ok, m.reason, m.conf, m.port_px, m.plug_px (undistorted px), m.across_mm, m.up_mm, m.gap_mm, m.sigma_mm
```

The interface is as in the brief. `plug_z_mm` is an optional addition. Axes match `p_mm` of the manifest: x across, y up, z gap. Zero is the centre of the port 2 mouth. Time: 82 ms median, 118 ms p95 (n = 633, one call, this server's CPU), so about 10 Hz.

## Method

1. **Undistort** with the calibrated intrinsics.
2. **Shell.** The plug is near a fixed place in the image. The detector finds the bright shell over the dark sleeve in a window: shell end row, two rim edges, width. All shell rows give one width.
3. **Panel pose.** Bright blobs match a stored row pattern (ports 1, 2, the small slot, USB-A, RJ45, HDMI, lower USB-A). The pattern gives only the identity and a coarse pose, and 3 matched features are enough, so port 2 can be hidden behind the plug. The pose comes from PnP on the **case edges**: chrome band, bottom edge, both side silhouettes, plus slots that fit within 2.5 px. Edges were needed: PnP on the blobs alone had a pitch error of 2-5 deg SD against FK; with the edges 0.2-0.3 deg (yaw about 1 deg).
4. **Tip.** A ray through the shell at a depth from the shell width, then a rotation into the panel frame.

**Pixels to mm.** Two scales. The panel scale comes from the case and port geometry in `deploy/config/port_vision_model.json` (about 97 mm between the side edges). The plug scale comes from the shell width with an effective width of 8.15 mm (spec shell: 8.25 mm), 47.5 px at about 117 mm. **Gap** is the tip's distance from the panel plane: the difference of these two depths and the image rows.

**Why depth matters.** The camera looks about 63 deg off the panel normal. One image row mixes up and gap (camera y axis = -0.45 up + 0.89 gap). Depth carries most of "up" (camera z axis = -0.89 up - 0.45 gap). So up is the weak axis. A per-frame width error of 0.2 px moves up by about 0.5 mm, which is why `plug_z_mm` helps.

**The model.** Panel geometry is triangulated from the tune blocks with FK as the known camera motion. So FK sets the panel's scale once, offline; it is not used at run time. Two constants come from frames with the plug in the port mouth (not in the labeled set): the port centre in the model frame (the bright blob is the lower lip, about 0.5 mm below the centre) and the effective shell width.

**Fail safe.** ok=False with a reason: frame size, no shell, width out of range, no pattern, missing case edges, panel fit over 5 px, rotation over 25 deg from nominal. `port_px` or `plug_px` can be filled on a failed call. The 5 px gate came from the tune blocks: residuals did not grow up to 5 px and detection rose.

## Results on real frames

Split, contiguous blocks only: the collect sweep cut into 6 blocks of 25 samples in visiting order. Blocks 0, 2, 4 tune the model and gates. Blocks 1, 3, 5 and all other runs report. Slope: least-squares output per mm of label, same definition as `summary.json`. Residual: SD of (output - label). False: ok=True and over 2 mm across or 3 mm up or gap from the label after removing that set's constant. All data: `results/real/port_vision/metrics.json`, `eval_frames_full.csv`.

**Held-out collect blocks, one depth per grasp (`plug_z_mm` = this run's median):**

| Gap band (mm) | n | Detected | Rate (Wilson 95 %) | False | Slope across, up, gap | Residual SD across, up, gap (mm) |
|---|---|---|---|---|---|---|
| 0-5 | 15 | 12 | 0.80 (0.55-0.93) | 0 | 0.98, 0.98, 1.00 | 0.38, 0.72, 0.39 |
| 5-10 | 28 | 28 | 1.00 (0.88-1.00) | 0 | 1.01, 1.00, 1.08 | 0.18, 0.74, 0.49 |
| 10-15 | 3 | 3 | 1.00 (0.44-1.00) | 0 | n too small | 0.47, 0.70, 0.71 |
| 15-25 | 8 | 8 | 1.00 (0.68-1.00) | 1 | 1.06, 0.92, 1.17 | 0.69, 1.15, 0.88 |
| 25+ | 21 | 19 | 0.90 (0.71-0.97) | 1 | 1.08, 1.00, 1.01 | 1.12, 0.82, 0.82 |

Pooled held out, gap under 15 mm, n = 43: slope 1.00, 0.98, 1.04; residual 0.29, 0.73, 0.47 mm. The nets on the same axes: `wrist_v18ws` 0.34, 0.17, 0.35; `wrist_v14w` 0.50, 0.32, 0.59 (other frame set).

**Held out, depth from each frame's own width (the default call):**

| Gap band (mm) | n | Detected | False | Slope across, up, gap | Residual SD across, up, gap (mm) |
|---|---|---|---|---|---|
| 0-5 | 15 | 12 | 3 | 0.99, 1.10, 1.39 | 0.38, 1.49, 0.72 |
| 5-10 | 28 | 28 | 0 | 1.01, 1.15, 1.13 | 0.18, 1.08, 0.58 |
| 10-15 | 3 | 3 | 0 | n too small | 0.61, 0.80, 0.42 |
| 15-25 | 8 | 8 | 0 | 1.05, 1.02, 1.14 | 0.72, 1.10, 0.67 |
| 25+ | 21 | 19 | 3 | 1.08, 1.02, 1.02 | 1.15, 0.92, 0.79 |

Use `plug_z_mm` near the port. The false rate of the default call at 0-5 mm is 3 of 12.

**Tune blocks (not a test), one depth per grasp:** 0-5: n 10, det 8, residual 0.27, 0.53, 0.22. 5-10: n 35, det 34, residual 0.25, 0.50, 0.32.

**Other runs (report only).** Detection, all frames: same_spot 77 of 80, self 41 of 42. Unlabeled: ai2 69 of 75, ai5 28 of 37, ai1 0 of 86, ai3 0 of 40. In ai1 the plug is rolled about 90 deg (edge on). In ai3 the shell is above the window. Both are hand-placed grasps; refusal is the wanted result in ai1 and a limit in ai3. Picked grasps (A, A2, collect) stay in the window.

Slopes inside a run (settled frames, one depth): A n 5: 0.86, 0.99, 1.01. A2 n 7: 1.11, 0.95, 1.04. The n is small. ai4 n 7 has almost no range. Slopes pooled over A and A2 are lower on up (0.79) because the two runs differ in hand, see below.

**Pick-run frames as negatives:** 0 of 123 ok=True (Wilson 95 % 0-3 %). These frames have no port row or the plug is elsewhere; no shell passes. This is a test of the shell gate, not of the row gate under a real plug. No frame set has a real plug with a wrong panel.

**Hand check:** `results/real/port_vision/sheet_success.jpg` (24 tiles) and `sheet_fail.jpg` (24 tiles): in all 24 success tiles the port dot is on port 2 and the tip dot on the shell tip. No wrong port was seen. The check is by eye.

## Acceptance: A and A2 at the stop

Stop frames, n = 6 (A), n = 4 (A2), one depth per run. Noise inside a stop is under 0.12 mm.

| A2 - A | Across | Up | Gap |
|---|---|---|---|
| FK tip (labels) | -4.58 | +6.80 | -1.38 |
| Vision plug tip | **-4.93** | **+3.88** | -0.39 |
| Camera height FK / vision | | 6.27 / 5.27 | |
| Vision minus FK tip (the plug in the hand) | -0.35 | -2.92 | +0.99 |

- **Sideways: passed** (-4.93 against -4.58).
- **Height: not reached as written.** The vision tip reports 3.9 mm, not 6. The camera against the panel moved 5.3 mm by vision against 6.3 mm by FK. The rest is a visible in-hand difference: in A2 the shell end sits 12.7 px lower in the frame (544.1 against 531.4) and the reader's rear edge 12.3 px lower (832.6 against 820.3). The finger positions in the frame do not change (by eye). So the two grasps were not the same, against the in-hand net's +3.69 and +3.84. I have no independent truth for the A2 grasp. The 1 mm camera difference is not explained: vision minus FK camera height is +0.57 mm in A and -0.44 mm in A2, near the 0.45 mm seen in the collect windows, but both errors add here.

Tip at the stop by vision, across, up, gap (mm): **A -0.3, +5.0, 8.4. A2 -5.2, +8.9, 8.0.** ai4 +4.3, -2.4, 9.9. The session note said too high: confirmed. It said a real gap of 11-12 mm: vision reads about 8 mm, because of the constant label offset below.

Note that camera across against FK is not useful (-4.95 against -1.65): yaw and across trade off in the pose, and the tip is much nearer to the panel than the camera. The tip across is not affected in the same way.

## Label offset (`results/real/port_vision/label_offset.json`)

**Answer for the collect run: true = manifest + (-0.5, -2.3, -1.7) mm, across, up, gap. y range -1.6 to -3.0 mm. This is an estimate, not a proof.** The manifest is not changed. Uncertainty of about 0.6 mm on y is a judgement from the spread below, not a measured interval.

Evidence:

1. Vision against the sweep labels, n = 134: median -0.54, -2.29, -1.67.
2. Plug in the port mouth, two moments outside the sweep. The mouth is visible as a dark notch. On those frames the tip's true height is about 0 and its depth is known, so FK label there plus the image-measured in-hand change gives the offset: first entry (n = 8) -1.97, -2.10, -2.36; second entry (n = 3) -1.09, -2.75, -0.97.
3. Sensitivity: moving the mouth row by 2 px or the mouth plane by 0.5 mm changes y by under 0.03 mm, because those errors shift the port and the tip along nearly the same ray. It does change gap (up to 0.4 mm). A 0.25 mm error in the plug top at the mouth moves y by 0.25 mm. Using only the first or only the second entry to set the constants gives the range on y.

Items 1 and 2 are **not independent**: the constants in item 1 come from the same mouth frames. Only 11 frames at two moments carry the absolute value.

**Which seated anchor is likelier right?** Neither. The first is much closer. If the recheck were right the offset on y would be +4.49; the data say about -2.3, which has the opposite sign. Confidence is high that the recheck reading is wrong as an anchor. It is moderate on the value -2.3.

What the frames show at the seats:

- **The first anchor reads high by about 2.3 mm**, as the session note guessed. The plug moved in the hand when pushed: free shell end at v 535.6 before the first seat, 530.5 after it (width 46.7 to 47.5 px), for the whole sweep. The reader's rear edge moved 824.2 to 825.3.
- **The recheck reading came in a transient.** The stable second seat (t +640 to +677 s after the first anchor) reads label y -0.35 to -0.99, not -4.49. The arm then pressed (target 13 mm past the plug), and the logged hand reached y -2.15 (no ticks in between), and fell further to -4.49 in the 0.5 s around the anchor key. Vision agrees that the camera really went low (FK and vision within 0.1 to 0.8 mm on height). In-hand: shell end 525 at the second seat, 518 at the release, rear edge 829.4. Grip reading 0.28 to 0.269. The first anchor had the same kind of step (y +1.49 to 0, z -6.59 to -5.80 at the key).
- **The PC moved a little during the jogs.** Vision minus FK across is -4 mm before the first seat and about 0 in the sweep, and -1.3 to -1.65 before the second seat. Yaw can take part of this, so read it as a hint only. Height agrees within 0.45 mm in all of these windows. The sweep itself looks stable (case edge fit residuals 0.1 to 0.2 mm on the tune blocks).
- Under load FK and vision differ by 0.9 to 1.3 mm in depth along the panel normal, the two seats alike. So the arm or PC gives under the push. Free, at the sweep's start and before the second seat, they agree within 0.05 to 0.35 mm. So a pressed pose should not anchor labels.

Other runs: `same_spot` A -1.62, -2.34, -3.63; A2 -1.96, -5.27, -2.60; ai4 (own seat) -0.65, -3.89, -1.41. A and A2 borrow the collect anchor, so their value also holds that run's grasp and PC. ai4's own seated anchor also reads high, in the same direction: this fits the pressed-anchor story, n = 1 run.

## Limits

- One PC, one session's lighting, right camera only. No other PC model or lighting tested. New PC: refit `port_vision_model.json`.
- Tested offline on logged frames. Never in a closed loop. Under motion blur the gates are untested as gates: frames marked moving are counted in detection but not in slopes.
- The absolute constants come from 11 frames at two moments, and the offset range on y is about 1.4 mm. Trust differences (slope, residual) more than absolute up. A port entry probably needs a few tenths of a mm (guess from the USB-C clearances and chamfer). Up residual near the port is 0.7 mm per frame with `plug_z_mm`: average several frames.
- The tip uses "plug about parallel to the panel axes", and the fixed tip-to-middle depth of a level plug. A plug pitched in the hand breaks both by an amount not measured here. The operator reports under 10 deg at the butt.
- Plug tilt (rx, ry, rz) is not an output.
- Fully seated or deeply entered: the shell hides and the call fails. That is by design: use before insertion.
- Sim-free: no sim frames were tested.
- Tune and held-out blocks come from one sweep, so they share the grasp and PC. The other runs are the only real generalisation test, and they have a lower n of settled frames.

## What the next rig session should collect

No motion was run for this work. To test it:

1. **Light-contact anchor.** Plug in the mouth by 1-2 mm with no push, log the frame and FK, three times, with re-grasp. Predicted: the offset of the sweep labels falls to near 0, and the mouth reading agrees with vision within 0.5 mm.
2. **Direct truth for up.** A shim or feeler gauge under a free plug at the mouth, or a ZED view of the plug from the side, at 5 heights.
3. **Grasp spread.** 5 picks, frames at a fixed hand pose at the stop, to measure the shell-end row and width spread: this tests the A/A2 in-hand finding directly. Add a side photo of each grasp.
4. **Live loop.** Log `measure` at the approach stop for 10 s still (noise) and then during slow moves of 1, 2, 5 mm/s (blur gates), with no contact.
5. **Other lighting and PC placement.** Lights changed, PC moved and yawed 5 deg: repeat the fixed-pose test.
6. **PC stability.** Log the PC position from the ZED during any push. Do not trust a pressed pose as an anchor.

## Reproduce

```bash
PY=/home/yashchitambar/peg_test_bimanual/.venv/bin/python; export PYTHONPATH=$PWD
$PY scripts/real/port_vision_fit.py model --iters 3   # tune blocks only; the port centre then falls back to PORT_CENTER_MM
$PY scripts/real/port_vision_fit.py eval
$PY scripts/real/port_vision_report.py
$PY scripts/real/port_vision_accept.py
$PY scripts/real/port_vision_sheet.py
$PY -m pytest -q tests/test_port_vision.py
```
