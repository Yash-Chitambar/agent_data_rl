# Left-arm and bilateral PC-contact session

Prepared offline; no robot/camera has been connected for this preparation. Keep last night's working policy checkout and controller configuration intact. This package is on the separate lab-handoff branch, not automatically deployed to the working policy checkout.

## Prepare without hardware

From this checkout, using the existing rig virtualenv:

```bash
/home/yashc/agent_data_rl/.venv/bin/python scripts/real/prepare_bimanual_session.py prepare --out /home/yashc/left_bimanual_session
```

This creates a manifest, fit/validation trial queue and contact-event schema. It refuses overwrite. Record the current WORKING policy's commit, dirty diff, weights hashes, controller profile and calibration before collection; the manifest's preparation commit is not that working baseline.

## Order when the operator returns

1. Confirm device/arm ownership, empty jaws, stop console and folded rest. Revalidate current shared base/camera transforms using independent board positions. No blind promotion of the earlier calibration candidate.
2. Camera-only readiness and short capture; check all three views, exact policy crops, file integrity and timing. A common visible event estimates relative timing; exposure clocks remain unaligned unless measured.
3. Left stationary connection check, then left joints 1–6 at folded rest. Use `measure_motion.py plan` before the supervised `single --side left --cameras` execution. Five repeats per condition; preserve complete held-out poses/trials. Do not run a batch of unreviewed motions.
4. Review left task-pose trajectories and collect holds, steps, ramps and reversals at those poses, empty and with task-relevant payload. Do not copy right-arm integral or friction settings. The old right arm also lacks complete task-pose coverage.
5. Commission left gripper: open/empty/object readbacks, real retained grasp, slip and release. The old measure_grasp.py is right-specific; it is NOT a left-arm-ready tool. Review/parameterize its ownership, stop and supported-shutdown path before left use.
6. PC contact on BOTH left and right: record approach without contact, first contact, support establishment, stationary support, PC displacement/rotation, withdrawal and natural slip. Identify the PC surface and contacting finger/gripper/wrist/link. Review each path; no scripted blind pushes or force targets.
7. Commission both-arm stationary recording first, then one moving arm with the other stationary. Finally record the intended support-and-insertion interaction. Per-arm collision checks do NOT establish inter-arm clearance; this requires reviewed geometry/path checks and supervised trials.

## Data and synchronization

`measure_motion.py bimanual` logs both arms in one host-clock control loop. `--cameras` on a single-arm trial and the default bimanual camera path start three-camera capture before motor connection using an explicit readiness marker. Collector startup/shutdown/fault handling still requires supervised commissioning. Camera source failure and early recording termination must be tested. Default 300-second capture limit bounds the recording; do not allow task duration plus approach/return to outlast it.

Native encoder arrivals, requested/executed targets and available CAN send times are distinct. Gripper commands/readbacks are not squeeze-force measurements. Preserve raw encoder and model coordinates separately. Save camera frame IDs and arrival timestamps. Host software alignment is not hardware exposure synchronization. Existing camera logs do not contain actual policy consumption IDs: action-labeled policy collection needs a tick-to-consumed-frame log from the working policy runner, rather than relabeling these collector frames as consumed.

After collection:

```bash
/home/yashc/agent_data_rl/.venv/bin/python scripts/real/prepare_bimanual_session.py timeline --session /home/yashc/left_bimanual_session --out /home/yashc/left_bimanual_timeline.jsonl
```

This merges CSV records on host monotonic timestamps and retains original fields. It does not certify stream completeness, clock alignment or training readiness. Camera hardware timestamps stay in their original clock domain.

Use `contact_events.csv` for both arms. Event timestamps must identify actual event/annotation timing; retrospective annotations must say so. Keep support conditions `free`, `left_arm_supported`, `right_arm_supported`, `both_arms_supported` separate. Leave force empty without an identified instrument. Contact stiffness, normal force and pad compliance remain unresolved with only calipers and a scale.

## Outstanding before live session

Review and test arm ownership, camera loss, stop during approach/contact, write failure, early camera termination, partial-log retention and supported shutdown. Existing measurement runner return behavior is not yet certified for loaded/contact trials. Do not use a normal automatic return while a gripper holds the PC. Freeze the actual working controller profile independently for each arm; do not import experimental deployment-branch gains/trim silently.

## Stationary left/right grasp collector

`measure_grasp_stationary.py --side left --session SESSION --name left_reader_001 --rig-code-root /home/yashc/agent_data_rl` uses the explicitly selected working rig's safety/ownership code. It accepts either side and sends no arm trajectories. Connect with empty jaws; it records open and empty-closed readings, waits for the operator to place/support the reader, records object-held readback and operator retention confirmation, then waits for retrieval before opening. On interruption while an object may be held it retains power until explicit SECURED confirmation. Only normalized readings and qualitative retention are measured. Syntax checked; full hardware commissioning is pending its first supervised trial. No PC force or compliance accuracy is implied.

## Shared evidence export

After collection, `scripts/tools/export_bimanual_profile.py --session LEFT_SESSION --right-session RIGHT_SESSION --camera-spec CAMERA_JSON --out NEW_PROFILE_JSON` exports per-arm controller evidence, left grasp trials, mounting geometry, retained left-base transform and source checksums. It refuses overwrite. The exporter does not fit parameters or fill missing PC–arm contact values. Use a new output version after new measurements. Current local snapshot: `/home/yashc/left_bimanual_session/bimanual_evidence_v1.json`.

The mounting platform is one shared plate under both bases, 23 mm above the PC tabletop. Its footprint remains to be surveyed; retain both collision surfaces rather than lowering a single table geom and discarding the platform. An overhead marker survey is a coarse geometric reference, not force or compliance measurement.
