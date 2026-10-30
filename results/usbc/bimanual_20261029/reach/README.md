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
