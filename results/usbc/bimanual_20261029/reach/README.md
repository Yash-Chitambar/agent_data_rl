# Nominal pick audit

The legacy +5.4 mm TCP offset above the reader body centre misses the reader
with the current FlexPoint pad geometry. The fingers close to their limits
while the reader remains on the table.

The new `pc_bench.reader_grasp_offset` derives the reader-frame TCP offset from
the lowest right-gripper contact-sphere row. For this model its vertical offset
is -3.225 mm. Pad geometry, friction, squeeze force, and reader mass are unchanged.
This proposal places the lowest contact row at the reader body's mid-thickness.

All four nominal Cartesian waypoints have negligible numerical IK residual and
no static unintended obstacle penetration in the audit. This does not validate
the paths between waypoints.

The CPU native-gain rollout, with the current right-joint-2 friction correction,
lifts the reader approximately 70 mm and keeps its body centre approximately
1.3 mm from the TCP during lift/carry. The reader is never attached to the arm.
The legacy rollout does not lift it. Saved joint states and trace CSVs support
the before/after render.

This is one simulated grasp, not hardware validation, a successful insertion,
or a bimanual bracing policy. Final pre-insertion position still has tracking
and in-hand offset error. Dynamic contact forces, other reset poses, and GPU
behavior need validation. Left arm remains at the recorded reference pose.

Reproduce from the repository root:

```bash
PYTHONPATH=. .venv/bin/python scripts/tools/audit_bimanual_reach.py --legacy-grasp
PYTHONPATH=. .venv/bin/python scripts/tools/audit_bimanual_pick.py --legacy-grasp
PYTHONPATH=. .venv/bin/python scripts/tools/audit_bimanual_reach.py
PYTHONPATH=. .venv/bin/python scripts/tools/audit_bimanual_pick.py
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/tools/render_bimanual_pick_audit.py
```

The original single-arm expert's explicit grasp offset is unchanged for
checkpoint compatibility. New bimanual scripted trajectories should use the
geometry-derived proposal, then validate it under the relevant contact model.


## Alignment follow-up and the new identified dynamics

The audit now supports `--inhand-align`, which derives the pre-insertion TCP
target from the observed rigid transform of the held reader instead of assuming
the nominal grasp transform. Under the earlier ideal-gravity plant this reduces
the final hold's lateral error from 8.78 to 3.39 mm, and tilt from 6.55 to 3.87°.

`--feedback-align` additionally uses privileged reader/PC poses to correct joint
targets at 50 Hz. Commands change by at most 0.004 rad/update and stay within
0.1 rad of measured model angles and the joint limits. This is a diagnostic
expert, not a deployable camera policy. With gain 0.08 and six seconds of alignment,
the earlier plant holds approximately 0.387 mm lateral error, 6.015 mm gap and
0.337° tilt. The higher-gain 0.4 / three-second attempt oscillates more; its
outputs are retained as `feedback_fast_*`.

The shared branch subsequently added identified dynamics in commit 8b71aa4.
`--identified-dynamics` applies its nominal per-arm friction/impedance and
driver-gravity residual, held for 20 ms, with randomization disabled.
The profile hash is recorded. **This run lifts but loses the reader during carry.**
It has zero gripper contacts at the end of pre-insertion and its reader-to-TCP
distance reaches 146 mm during alignment. Consequently its apparent final
port error is not an alignment-controller accuracy result.
`retained_after_carry` now distinguishes this failure from a successful lift.

Reproduce this unresolved test with:

```bash
PYTHONPATH=. .venv/bin/python scripts/tools/audit_bimanual_pick.py --feedback-align --identified-dynamics
```

Next work must address carry clearance/tracking and retained contact under the
identified dynamics before proceeding to insertion or bimanual bracing. The
older ideal-gravity pick/alignment results do not validate the new runtime default.
