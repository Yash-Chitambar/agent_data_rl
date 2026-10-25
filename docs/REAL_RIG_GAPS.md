# Real rig (duchamp3) vs sim: open gaps and how to close them

The gaps between the real YAM rig and the sim, found during the 2026-10-25 setup (`docs/FINDINGS.md`,
2026-10-25 entry, "Open"). Each gap has a test, a pass criterion and the change it causes in the sim
or on the rig. Latency and backlash have their own full protocol: `docs/REAL_YAM_LATENCY_BACKLASH.md`.

Update the status column when a gap closes. Record the result in `docs/FINDINGS.md` as well.

| # | Gap | Impact | Status | Needs |
|---|---|---|---|---|
| 0 | No deployment loop | Blocks 1-3 | **done** 2026-10-25 (`scripts/real/deploy.py`, offline check passed) | - |
| 1 | Latency and backlash not measured | High (Phase C: one 20 ms step off fails) | Test 1 done: loop 4.0 ms (std 1.6); step test and backlash open | Loop; dial indicator for backlash |
| 2 | Gains: real 80/80/80/40/10/10, sim 80/80/80/30/30/30 | High (C gain sweep) | real runs sim kp; kd 5 not 6 on joints 1-3 (DM limit); gain fit open | Loop |
| 3 | Joint offsets not applied | High (0.4-1.3°, mm at the gripper) | applied in deploy.py; hardware checks open | Loop |
| 4 | Real fixture and part poses not measured | Medium | open | Real fixture on the table |
| 5 | Finger pads: real yellow 3D-printed, sim Menagerie fingertips | Medium (contact, camera view) | open | Pad CAD or calipers |
| 6 | Wrist payload: sim 109 g D405 bracket, real webcam + cable | Low | open | Kitchen scale |
| 7 | Camera auto exposure / white balance, frame age unknown | Visual policies only | open | Loop + LED or flash |
| 8 | ZED not in the sim | Top-camera policies only | open | Code only |
| - | CAN interfaces do not come up after reboot | Setup | open | sudo, once |

## Order

1. Build the deployment loop (gap 0). Offline, no hardware.
2. Decide the gains (gap 2). This must come before the step test, because the step test measures
   the gains in use.
3. Test 1 (loop timing, no motion), then Test 2 (joint steps) from the latency protocol. About 40 min
   at the arm.
4. Weigh the payload (gap 6). Measure the pads (gap 5). Lock the cameras (gap 7). These need no motion.
5. Place the real fixture and locate it with the ZED (gap 4).
6. Backlash (Test 3), when a dial indicator is available.
7. ZED in the sim (gap 8), only when a policy needs the top camera.

## 0. Deployment loop

`scripts/real/deploy.py`. One loop, used for the tests and for the policies:

- 50 Hz, the sim's control rate. In each step: read the encoders, add the joint offsets
  (`q_model = q_measured + rig.joint_offsets(side)`), compute the action, send
  `q_target = q_meas + 0.04 * action` through `safety.SafeArm.command`. Remove the offsets before
  sending (the motors use their own zero).
- Set the gains at start with i2rt `update_kp_kd`, from a flag. Default: the sim gains.
- Action sources: a policy checkpoint, `hold` (zero action) or `step` (the joint step test).
- Log per step `t_obs`, `t_act`, `t_send` and every encoder reading, in the CSV format of
  `docs/REAL_YAM_LATENCY_BACKLASH.md`.
- **Offline check:** run the loop against the sim model in place of the arm. It must give the same
  trajectory as `PegEnv` for the same actions.

## 1. Latency and backlash

Follow `docs/REAL_YAM_LATENCY_BACKLASH.md`: Test 1 (loop timing), Test 2 (joint steps, fitted by
`scripts/real/yam_latency_fit.py`), Test 3 (backlash with a dial indicator). The table at the end
of that file gives the change in training for each result.

Note: i2rt runs its own motor thread and re-sends the last command between our 50 Hz steps. So the
encoder timestamps must come from that thread (the time each reading arrived), not from our loop.

## 2. Gains

**Recommended: run the real arm at the sim gains** (kp 80/80/80/30/30/30, kd 6/6/6/2/2/2). The
policies trained with these, and the real wrist (kp 10) is too soft: it lagged commands by up to
0.19 rad during calibration.

- **Test:** Test 2 at the new gains. The fit gives a gain scale per joint (real stiffness / sim
  stiffness).
- **Pass:** gain scale 0.7-1.4 on all joints, no oscillation, hold drift < 0.01 rad in 5 s.
- **Risk:** a stiffer wrist can oscillate. Start with 0.01 rad steps and the e-stop in reach, then
  go to 0.04 rad.
- **If it fails** (gain scale outside 0.7-1.4, or the wrist oscillates): fix the sim's YAM gains in
  `assets/builder.py` to the measured values and retrain. Also widen `arm_kp_scale`/`arm_kd_scale`
  randomization to cover the measured range.

## 3. Joint offsets

- **Apply:** in the deployment loop (section 0). Also apply them in `calib.py` and any FK tool.
- **Test A (camera):** with the offsets applied, the wrist camera's board corners against FK.
  Pass: hold-out error close to the calibration result (right 2.7 px, left 2.2 px; without offsets
  8.0 / 3.0 px).
- **Test B (touch):** tape a cross on the table at a known point in the base frame. Locate it with
  the ZED. Move the fingertip 5 mm above it, at 3 different arm poses. Measure the miss with
  calipers. Pass: < 2 mm at all poses, and smaller with the offsets than without.
- **Sim:** randomize the joint offsets in training, about ±1.5° on joints 2-5. This covers the
  error that remains and the drift after a motor re-zero. (`EnvCfg` has no field for this yet.)
- **Re-measure** after any motor work or re-zero: `calib.py collect` + `solve` (~20 min per arm).

## 4. Fixture and part poses

- **Test:** place the real fixture (the mini PC for Task H, or the peg fixture) where it will be
  for the runs. Locate it in the right base frame with the ZED (`top.extrinsics_cam_in_right_base`
  in the spec): either an ArUco tag on the fixture, or 3+ touched points with the fingertip.
  Do the same for the part's start pose.
- **Pass:** the real pose is inside the sim's sampling range (`EnvCfg.fix_x/fix_y/fix_z`,
  `fix_tilt_deg`, or the Task H config), with margin, after conversion to the sim world frame.
- **If not:** move the fixture, or shift the sim range and retrain.
- Write the poses into the spec as a new `scene` section.

## 5. Finger pads

- **Measure:** get the pad CAD (STL) from whoever printed them, or measure them with calipers:
  length, width, thickness, the mount offset on each finger, and the surface (smooth PLA or a
  rubber layer). Take a photo from the wrist camera with the gripper open and closed.
- **Sim:** add the pads to the YAM model as meshes or boxes on `tip_left`/`tip_right`, with a
  friction that matches the material (measure it with a slide test on a tilted board if needed).
- **Pass:** the sim wrist image of the pads overlays the real photo (the camera is calibrated, so
  this is a direct check). The sim gripper width at closed matches the real one to 1 mm.

## 6. Wrist payload

- **Measure:** take the webcam, its bracket and the cable to the first tie point off the arm, and
  weigh them. Measure the centre of mass position on the bracket (balance it on an edge).
- **Sim:** replace the D405 bracket mass (109 g) in the model with the measured values.
  Also update `safety.EE_MASS` if the gripper total changes (i2rt gravity compensation uses it).
- **Pass:** at a hold pose with the arm out, the real joint 2-4 torques match the sim gravity torques
  to ~10%. The loop logs torques, so this is a 10 s hold test.

## 7. Wrist camera image

- **Lock** exposure, gain and white balance with `v4l2-ctl` (`auto_exposure`, `white_balance_automatic`)
  at the start of every run. Check with `v4l2-ctl --list-ctrls`. Put the values in the spec.
- **Frame age:** show a blinking LED (or a timer on a phone screen) to the wrist camera. Log
  the host time of each frame and compare it with the time of the LED change. Measure at 30 and
  60 fps. The frame age adds to the policy's latency for visual policies.
- **Sim:** set `cam_rand` color and light randomization to cover the real images (compare
  histograms of real and sim frames at the same pose). Add the frame age to the visual latency.
- **Pass:** real frame age measured, and stable to ±1 frame. Real image statistics inside the
  sim's randomized range.

## 8. ZED in the sim

- Only needed for a policy that uses the top camera.
- **Sim:** add a fixed camera in the world at `top.extrinsics_cam_in_right_base` (OpenCV ->
  MuJoCo: 180° about x, as for the wrist camera), with fovy from `top.intrinsics_rectified`
  (HD720: 69.06°). The real side uses the rectified left image.
- **Pass:** board corners from a sim render at a known board pose match the real ZED image to a
  few pixels (the same check as for the wrist cameras).

## CAN after reboot

The CAN interfaces are down after a reboot (`ip link set canN up type can bitrate 1000000` needs
sudo). Add a systemd-networkd `.network` file or a udev rule per adapter serial (see `rig.py`).
One-time sudo setup.
