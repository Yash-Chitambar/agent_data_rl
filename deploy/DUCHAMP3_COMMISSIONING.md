# Duchamp3 commissioning, 2026-10-27

Model manifest verified. ACT timing on RTX 4090: p99 2.73 ms, max 2.81 ms, with synthetic images. Camera run after selecting ZED serial 20120598: top 29.8 fps, wrist 35.7 fps, host-arrival-to-use p99 32.86 ms and 30.71 ms; exposure time and clock alignment are not measured. Shared handoff and conventions checks passed. Camera intrinsics/spec file differs in overhead extrinsics from the active lab spec; review before geometric overlays.

Gripper calibration with operator supporting reader: normalized open 1.000, closed empty 0.038, closed on reader 0.3371, close command 0.0. A single calibration maps empty to 5.9 mm and reader to 19.0 mm against the checkpoint's assumed map. The current grasp threshold is 5.0 mm, so empty falsely passes. Force, stable retention and a robust threshold are not measured. No policy trial should proceed until this is resolved and validated.

The original real runner calls close after a pick or error, disabling torque while the arm may be raised and the reader held. The original frame capture similarly closes after reaching the raised ready pose. Both real entry points are disabled pending a reviewed powered-hold, retrieval and recovery workflow. The pick timeout also originally transitioned to insertion even if the gate failed; it must be fixed before enabling real execution.

The FlexChecker rejected the recorded folded-rest starting pose (link_2 at -36 mm against +40 mm clearance); the cause and a collision-checked path must be resolved without arbitrarily weakening safety. Actual joint-3 errors previously exceeded 0.02 rad on two approach trials. The gripper driver reported a CAN socket shutdown race after a successful calibration; the cleanup path now stops and joins the motor thread first. Calibration on connect still moves the empty gripper automatically.

No real policy rollout or ready-pose image capture has been performed. Current camera and gripper commissioning changes are local research modifications, not a validated robot deployment package.
