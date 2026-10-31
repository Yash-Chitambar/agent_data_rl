# USB-C PC image handoff — duchamp3, 2026-10-27

This note accompanies the real-camera captures for the SanDisk USB-C SD reader task. Use it when importing the scene to mason or when comparing the PC and port with MuJoCo. The images document this setup; they do not by themselves establish connector-level metrology.

## Capture bundle

All paths below are relative to the repository root.

| File | View and use |
|---|---|
| `data/sim_ref/usbc_lab_20261027_port/pc_overhead_zed_left.png` | Fresh rectified ZED left image captured at 16:56 PDT with both arms at rest. Use this image to estimate the PC footprint position and yaw on the table. |
| `data/sim_ref/usbc_lab_20261027_port/pc_overhead_zed_right.png` | Synchronized ZED right stereo image from the same capture. Preserve it with the left image; the current plan uses RGB only, so do not treat stereo as validated metric depth. |
| `data/sim_ref/usbc_lab_20261027_port/pc_overhead.json` | Camera calibration reference and capture metadata. No robot joint readings are included because this was a stationary capture. |
| `data/sim_ref/usbc_lab_20261027_port/reader_pickup_start_zed_left.png` | Fresh no-motion overhead frame of the PC and reader staged flat for a representative pickup, plug directed toward the PC. |
| `data/sim_ref/usbc_lab_20261027_port/reader_pickup_start_annotated.png` and `reader_pickup_start_projection.json` | Manual image landmarks and ray-plane estimates for this one staged scene. Reader body center is provisionally `(0.178, 0.144) m`, plug-axis yaw about `11 deg`; use a broad ±10 mm / ±10 deg initialization range. Pixel points, assumed feature heights and method are recorded in JSON. This is for scene initialization only; the reader body corners do not support a reliable footprint fit, and the approximate PC panel point does not identify USB-C #2. |
| `data/sim_ref/usbc_lab_20261027_port/reader_pickup_start.json` | Active camera calibration reference for the fresh pickup-start capture. |
| `data/sim_ref/usbc_lab_20261027_port/pc_overhead_wrist_right.png` and `pc_overhead_wrist_left.png` | Wrist views at the arms' rest pose, captured with the overhead frame. |
| `data/sim_ref/usbc_lab_20261027_port/centered_wrist_right.png` | Close wrist view taken with the right arm aimed at `[0.31, 0.30, 0.0] m` in the right-base frame. It centers the PC but sees the panel from above and at too great a distance for port measurement. It was captured before the arm returned to its recorded start pose. |
| `data/sim_ref/usbc_lab_20261027_port/centered_wrist_right_policy.png` | The corresponding undistorted, resized policy crop. Use for RGB preprocessing parity checks, not geometric measurement. |
| `data/sim_ref/usbc_lab_20261027_port/pointed_zed_left.png`, `pointed_wrist_right.png`, `pointed_wrist_left.png`, `pointed.json` | Earlier pointed reference set at the same approximate table target. Useful for camera-view comparisons; the wrist images do not show the panel closely enough to locate the USB-C connector. |
| `results/real/usbc_overlay_20261027_aligned/` | Initial real/sim side-by-side and blended renders. The PC was placed at port XY `[0.310, 0.303] m` as a provisional scene placement. This is not a surveyed port location. See its README for comparison limitations. |

The capture script uses the active camera spec `blind_insertion/configs/real_cameras_duchamp3.json`. The October 27 fresh camera/base candidate was not adopted. Retain the active calibration and its few-millimetre relative-pose uncertainty; see `data/real_yam/usbc_lab_20261027/README.md` for independent-position checks and why the candidate was rejected.

## How to use the overhead image in simulation

1. Use the rectified-left ZED image and its calibrated intrinsics and extrinsics from `pc_overhead.json` / the active camera spec. Respect the image size and coordinate convention recorded in the metadata.
2. Mark the four visible top-footprint corners (or fit the rounded rectangular footprint), then project image rays onto the table plane using the calibrated camera pose. Account for the PC body's measured height/feet when choosing visible edges; top-surface points are above the table plane.
3. Fit a planar rectangle using the reported approximate footprint dimensions: 102.5 mm across the port panel and 97.0 mm front-to-back. Estimate table XY and yaw in the right-arm base frame. Save the picked image points, mapping code, residuals and uncertainty alongside the derived pose so the fit can be reproduced.
4. Set only the *per-scene PC pose* in a comparison render and show the real and simulated overhead views with the same crop. Do not change port-relative-to-case coordinates to compensate for a global placement or yaw mismatch.
5. Keep the PC free to slide/rotate in the eventual physics model. A fitted pose for this one capture initializes the comparison; it is not a fixed-fixture assumption for task physics.

For the fresh reader-start image, `reader_pickup_start_projection.json` is a reproducible but coarse first estimate. Its assumed surface heights and hand-selected pixels dominate uncertainty; do not tune reader dimensions or grasp geometry from this projection. Confirm the pickup target with a real robot pose that passes the measurement runner's start-pose attainment check before collecting dynamics.

The overhead frame makes PC orientation visibly different from the simulator's default zero-yaw placement. Its corners are blurred/rounded and partly occluded by perspective, so report an uncertainty range and do not present a pixel-picked pose as precision survey data. Use the separate tape/caliper dimensions and the independent base-frame uncertainty in the lab README.

## Port geometry: pending evidence

The current front-panel geometry and USB-C port arrangement in `blind_insertion/scenes/pc_bench.py` are photo estimates. The existing wrist images show that this panel does not match the real PC well enough to infer the target opening center. Do not enlarge the receptacle or move its center to make insertion succeed.

Before adjusting port-relative geometry, collect a close-up that shows the entire front panel, both case edges and the second USB-C port. A ruler/caliper reference should lie in the plane of the panel if possible. Prefer a front-facing right-wrist image for appearance and pixel correspondences; take a perpendicular side view or use calipers for depth/recess dimensions. Record which opening is USB-C #2 as viewed from the arms. Save the raw frame and metadata, then derive the port center relative to the measured case edges with an explicit pixel-to-length scale and uncertainty.

The close-up is not yet in this bundle. The PC mass/dimensions currently recorded in `data/real_yam/usbc_lab_20261027/measurements.csv` are single operator readings with repeatability/resolution unrecorded: mass 0.284 kg, width 0.1025 m, depth 0.097 m (orientation assumed), body height 0.0412 m, total height with feet 0.0445 m, four 5 mm pads. The operator reports reader mass 0.017 kg and caliper measurements: silver body 34.6 mm wide, 29.6 mm long and 8.75 mm thick; black neck 43.4 mm long, 12.96 mm wide and 7 mm thick; metal plug protrusion 5.8 mm (three readings). Other repeats and instrument resolution are unrecorded; the operator requested no further repetitions. Keep values labeled by source and uncertainty; do not infer force, compliance, or connector insertion depth from images.

## Server-side deliverable

On mason, create a versioned comparison profile containing the fitted scene PC pose, dimension sources/ranges, camera-spec hash, raw-image checksums, annotated corner pixels, and fitting method. Render the scene at that pose, compare case-edge residuals in overhead and wrist views, and separately update the measured port-relative geometry only after the panel close-up arrives. Preserve the current profile/checkpoint for reproduction. Do not treat image appearance agreement as proof of contact or insertion physics; validate free-body settling, translation, rotation and tipping separately using the recorded mass and support configuration.
