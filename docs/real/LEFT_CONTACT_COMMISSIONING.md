# Left PC-contact commissioning, 2026-10-28

Status: incomplete. Both arms were closed with torque zero after operator-supported shutdown of interactive_002. No confirmed contact or lateral slide was collected. Do not label these episodes contact demonstrations.

Raw evidence: /home/yashc/left_bimanual_session/contact/interactive_001 and interactive_002; their camera runs have the same names under cameras/. Session code/contact_interactive_002 contains exact pre-trial source hashes. Later source changes require a new source snapshot.

interactive_001 failed because the latest saved PNG was still being written. The reader now accepts only a complete frame within two seconds, examining up to four recent files; otherwise it fails. Stale or missing recording remains a stop condition.

interactive_002 stopped when the bright-lid segmentation area changed by more than 15 percent. The review image shows the hand shadow altering segmentation. The last accepted apparent displacement was approximately 0.2 mm; position after tracking became uncertain is unknown. Do not relax the area gate to proceed. The label-template diagnostic replay had minimum correlation 0.394 and a boundary match; it is not enabled for motion and does not measure yaw.

Repeated 1-mm requested Cartesian steps did not produce equal physical steps. Recorded endpoint model-FK error reached approximately 10.8 mm. This is encoder/FK evidence, not independently measured tool displacement. A new pre-jog gate refuses further lower/back/slide commands when model-FK attainment error exceeds 3 mm. This threshold is a conservative commissioning choice, not measured accuracy. Disengagement retains its original swept collision checks and is exempt from the attainment gate so a recovery can still be requested. All live feedback, geometry, camera, stop and timeout checks remain active.

Next prerequisites:

1. Arms folded, torque off. Add two separated, contrasting visual references near the front/port-panel corners of the PC lid, away from the rear contact shadow. Record a camera-only reference. Validate translation and rotation tracking on held-out images and known displacements before enabling it for contact.
2. Compare native controller behavior with a bounded settling-trim candidate at an already checked high free-space pose. Keep PC contact out of this trial. Record nominal goal, trimmed command, executed command and native feedback separately. All biased commands require swept geometry checks and bounded bias/rate/time. Right-arm trim success does not establish left-arm validity. Do not enable integral accumulation during contact searching.
3. Require attainment and external visual verification before resuming low approach. Do not silently enlarge clearance, penetration, tracking-error or PC-displacement limits to finish a trial.
4. Only after the operator labels physical contact may the one-way 0.5-mm slide increments begin, at most 5 mm total. Mechanical forces remain unmeasured.

Offline checks: tests/test_contact_frames.py, test_contact_tracking.py, test_contact_steps.py, test_contact_geometry.py and test_recovery.py (11 passed). Offline checks are not hardware acceptance.

## Lid-marker camera-only checks

Session camera runs pc_stickers_002 and pc_stickers_003 record two yellow lid markers before/after an unmeasured manual displacement. Initial S>=90 segmentation failed under pale highlights. Offline candidate S>=50 (H15..45, V>=100) was fitted on raw_000150 from both runs. The post-startup neighboring frames, excluding those fit images, were detected in 47/47 cases: stationary translation component ranges below 0.22 mm and yaw ranges below 0.49 degrees. These are short-term repeatability observations, not absolute accuracy. The estimated marker-midpoint shift was [3.72,-52.42] mm and relative yaw 5.41 degrees. The manual displacement was not measured, so this cannot establish metric accuracy. The camera-only ROI was expanded for offline inspection because manual movement exceeded the original ROI; no live displacement limits were expanded. PC position has changed; old contact plans must not be replayed without fresh localization and geometric planning.

Report: /home/yashc/left_bimanual_session/contact/pc_marker_replay_v2.json. Earlier failures remain preserved. The marker module remains disconnected from live contact control. Dynamic shadow/occlusion and measured displacement checks remain prerequisites. Next physical task is to clear the PC and reader from the swept work area for a separate high-pose left-controller test; do not proceed directly to contact.

## Free-space controller results

Three completed runs returned to the recorded folded pose and closed both controllers with torque zero. Camera runs left_free_settle_001..003 cover both wrists and overhead. Reports and native feedback are under /home/yashc/left_bimanual_session/contact/free_settle_001..003; compact aggregate free_settle_comparison_001.json preserves source hashes. Each exact controller source is archived under session/code/free_settle_00N.

The native gain profile is kp [80,80,80,40,10,10], kd [5,5,5,1.5,1.5,1.5]. Explicit added feedforward was disabled; native driver compensation remained active. Candidate trim ki=.3, bias cap=.025 rad, rate=.003 rad/s, deadband=.002 rad. Baseline hold was 3 seconds, frozen-bias hold 2 seconds. The correction was evaluated only in free space with the PC removed. A conservative old PC exclusion box remained in the geometric guard.

| Run | Pose | Trim duration | Baseline final model-FK error | Frozen hold maximum error, final second | 3-mm gate |
|---|---|---|---|---|---|
| 001 | high | 12 s | 9.60 mm | 2.94 mm | pass |
| 002 | lower | 12 s | 5.68 mm | 3.74 mm | fail |
| 003 | lower | 25 s | 5.83 mm | 4.72 mm | fail |

Longer settling did not resolve the lower-pose failure. Joint-2 reversal/sticking and coupled joint errors remain model-fitting targets. The controller candidate is NOT adopted for contact, and the 3-mm gate is unchanged. These are individual trials with model-FK/encoder measurements; no external tool-position or force accuracy is established. Raw top-level controller text in the first logs describes the baseline; the nested settle_candidate, phase names, nominal targets, biases and actual executed commands identify the candidate integral phases explicitly. Preserve the raw records unchanged.

Next: review the adopted right-policy controller and compensation separately from the native controller used here; develop and validate a left-specific profile in free space before contact. Do not transplant right-arm friction values or increase tolerances to pass. Contact PC displacement/sliding and both-arm bracing data are still outstanding. No immediate physical setup change is required; leave objects outside the swept workspace and arms at rest.
