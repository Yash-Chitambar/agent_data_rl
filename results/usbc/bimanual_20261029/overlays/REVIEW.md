# Overlay review

- `zed_overlay.png`, `wrist_right_overlay.png`, `wrist_left_overlay.png`: equal-weight real/render blends using capture-calibrated cameras and host-arrival-matched encoder poses. No image registration shifts are applied after rendering. PC placement is initialized from a manual top-image landmark, not independently surveyed.
- `*_comparison.png`: real left, simulation right. Three camera captures have different arrival times; these are stationary-pose comparisons, not exposure-synchronized frames.
- `rear_photo_geometry_overlay.png`: original photo and projected case/port geometry. Case corners and port centers were selected on this same image; this is a registration diagnostic, not independent validation.
- `target_port_registration.png`: green contour is the preserved 8.34 x 2.56 mm physical receptacle aperture. It lies inside the apparent photographed opening. Do not claim exact port size/alignment from this view or enlarge physical geometry to match a manually rectified photo. Rounded-case boundary selection, planar assumptions and true inner/outer receptacle boundaries need checking.

Visible unresolved discrepancies: gripper pads and opening, mounting-plate footprint/edge, wall/raceway/tool/power-strip positions, robot materials and lighting. PC placement was fitted, so agreement there does not validate camera or base transforms independently. Reader is absent in this capture and therefore has no real/sim dimensional overlay here.

Status: diagnostic images generated; visual fidelity gate NOT passed.
