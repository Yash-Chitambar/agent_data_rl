# Physical YAM scene, 2026-10-25 (Pacific)

Pulled main (`dc535aa`) and integrated `origin/real-yam-setup` (`e7b843e`) on
`codex/physical-yam-scene`. No robot commands or hardware connections were made.

## What is used now

`pc_bench.build_scene()` and `PickCfg()` default to the physical rig profile.
Both YAMs use I2RT's **flexible_4310 / FlexPoint** yellow soft-tip and carriage
meshes, pinned to `120c3c81400171174604e503943f8d1ebc891058`. Assets, source link
and MIT license are in `blind_insertion/assets/flexpoint/`. The old linear
fingers are removed, not merely painted yellow. The original YAM arm meshes
and kinematics remain. The upstream moving-finger inertias and TCP are used.
The housing/link-6 inertia is retained pending a payload weighing.

The CAD is available publicly, but the user has no separately supplied CAD of
this particular hardware. Exact revision correspondence is unverified. Collision
pads are a rigid box/sphere approximation; the compliant yellow body is NOT a
validated deformable model. Surrounding finger-body contact is not represented
by a calibrated collision decomposition. Friction and force-displacement tests
are still needed before calling the contact model physically accurate.

## Frames and images

The right arm base is the world frame, with the tabletop at z=0. The measured
left-base position is (-0.01011, 0.60771, -0.00081) m and its measured quaternion
is applied from `real_cameras_duchamp3.json`. Both wrist camera poses and lenses
come from that file. The ZED rectified-left optical pose is
(-0.21015, 0.35358, 0.87333) m. OpenCV axes are converted to MuJoCo axes by a
180-degree rotation about camera x. Snapshot replay adds each arm's measured
encoder offsets exactly once. Camera positions agree with the stored snapshot
to <5 micrometres numerically; this is NOT an estimate of hardware accuracy.

New pick-policy images are **320 x 320 RGB**:

- Wrist: capture 1280 x 1024, undistort with that arm's lens coefficients, crop
  around its principal point (952 pixels right, 932 left), resize, BGR to RGB.
  Use `real_camera.real_policy_image`. Incorrect raw resolution now raises.
- Top: capture ZED **HD1080 rectified LEFT**, crop a 1046-pixel square around
  (975,557), resize, BGR to RGB. Use `physical_rig.real_top_policy_image`.
- Sim uses each calibrated fx/fy in the matching crop sensor window. Do not
  undistort the already-undistorted sim image. Raw camera images in the supplied
  photos and the processed square policy images are different representations.
- RGB-only wrist cameras are represented as RGB. No wrist depth is synthesized
  and claimed as a real modality. Exposure, white balance, rolling shutter,
  compression and measured frame age remain uncalibrated. No exact frame delay
  is invented from nominal frame rate. Existing image randomization remains.

## Scene estimates and dimensions

PC port center (0.321370,0.236917,0.025) m and reader body center XY
(0.203937,0.220908) m were estimated by calibrated ZED ray/height intersection
at hand-selected pixels (1088,571) and (1111,691), respectively, in
`pointed_zed_left.png`. These are approximate object placements, not surveyed
poses; yaw remains the previous nominal orientation. Back wall, table extent,
clutter and webcam housing are visual/photo estimates. Lab materials and lighting
remain simplified. The four caliper measurements and original USB-C plug mesh
are unchanged; `check_reader_dimensions.py` verifies them.

## Training and compatibility

`configs/task_h_reader_real.json` now selects `gripper: flexpoint`, the calibrated
RGB wrist camera and the updated fixture range. Its imported latency, per-joint
friction and encoder-offset randomization remain. PickEnv now also samples the
measured per-joint friction ranges rather than the old uniform low friction, and
brackets command delay at 0/1 control step. A measured camera-frame latency is
still missing. This turn did not train or deploy a policy.

Legacy model reproduction: `build_scene(physical=False)` selects the old arm
mount and camera/gripper profile. For an old visual checkpoint, use
`PickCfg(physical_rig=False, cam_res=(320,240))` AND restore its saved object
positions and randomization settings; the new scene is not a drop-in replacement
for old ACT checkpoints. Retrain/re-evaluate the pick and insertion stages for
new gripper geometry, TCP, placements and image preprocessing.

## Reproduce checks

Check shared GPU capacity first; pin an idle GPU.

```
PYTHONPATH=. .venv/bin/python scripts/tools/check_reader_dimensions.py
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/tools/render_physical_rig.py
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/tools/check_pc_bench_warp.py
.venv/bin/python scripts/server/jobs/make_manifests.py --check
```

Side-by-side images are in `results/H/physical_rig/`: real on the left, simulated
on the right, both with the same calibrated policy crop. Visual similarity is
not a sim-to-real policy success claim.

## Validation outcome

Camera pose/crop assertions passed for both wrists; wrong-resolution real input
is rejected. The original reader mesh/dimension check and 171 manifest commands
passed. A randomized GPU step/render smoke returned finite states and both
320x320 RGB frames. A four-world, 12-second existing-expert diagnostic stayed
finite but achieved **0/4 completed handoffs and 0/4 held readers at the end**
(phases 4,5,4,4). See `pick_smoke.json`. This small diagnostic is a failure to
validate the old expert on the new contact/dynamics model, not a policy benchmark.
The scene is implemented; contact calibration and expert adaptation remain
necessary before generating successful training demonstrations at scale.
