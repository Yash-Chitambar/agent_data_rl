# Bimanual policy interface and validation status

The new interface is `blind_insertion.envs.bimanual_task.BimanualTask`.
It is an experimental training interface; successful physical pick, brace and
insertion rollouts have not yet been demonstrated. Existing single-arm training
configurations and processes in other worktrees are not changed.

## Actions and controller state

The 14 normalized actions are right joints 1–6, right gripper, left joints 1–6,
left gripper. Arm actions are clipped to [-1,1] and scaled by 0.04 rad per control
step. The default is now `action_mode="accumulate"`:

1. Add the increment to the previous joint target.
2. Limit the target to within 0.1 rad of the quantized encoder observation,
   converted once into model coordinates.
3. Clip to the joint limits and send the resulting absolute joint targets.

A zero arm action retains the target, subject to the tracking-error cap. It does
not actively cancel a retained position error. Reverse actions reduce it. Each
reset initializes only the selected worlds' targets from their encoder readings.
Gripper commands retain the runtime's open/close convention; they are not
integrated. Gripper force and normalized hardware opening remain approximate.

The 0.1 rad cap follows the earlier accumulated-action convention recorded in
`docs/FINDINGS.md` (October 26). It is a controller design bound, not a calibrated
friction parameter or a standalone hardware safety guarantee. This patch does
not modify a hardware deployment loop.

With native elbow kp=80, the previous relative action window of 0.04 rad limited
the proportional contribution to 3.2 N·m. The archived left-elbow trials bracket
breakaway between 3.2 and 4.8 N·m. Retaining a bounded target allows the controller
to build sufficient error rather than continually re-centering its target.
`action_mode="relative"` remains available for controlled comparisons.

## Observations and compatibility

Actor observations include 12 encoder angles, 12 joint velocities, 14 previous
actions, and **12 target errors** (right then left), plus optional calibrated RGB
and camera ages. Target error is commanded model angle minus the encoder-derived
model angle; it exposes controller state and does not leak object state.
Object poses remain privileged critic information.

This changes the observation contract and default action semantics. Start a new
policy/configuration or explicitly adapt a checkpoint; do not silently load an
older relative-action checkpoint. Deploying a resulting policy requires the
same accumulated target state, cap, offsets, encoder convention and reset behavior.

## Checks and remaining physical limitations

`scripts/tools/check_bimanual_target_accumulation.py` uses a synthetic one-joint
MuJoCo load with 4 N·m friction and kp=80. Under the same action sequence,
accumulation produces more than four times the displacement of relative targets.
It also checks bounded error, independent arms, joint limits and reversal.
This is a command-authority stress test, **not a fit of elbow dynamics**.

The regularized friction constraint still creeps below its nominal threshold.
The test reports this explicitly; the measured robot's sticking requires separate
validation. Left elbow state-dependent friction, payload/gravity residuals,
gripper force/compliance, PC support friction, and full task reach/contact
rollouts remain unresolved. Geometry checks and synthetic seating do not prove
electrical connection or successful hardware transfer.

The GPU episode check additionally verifies target-error observations and
selective target reset, alongside termination, truncation and seating duration.
