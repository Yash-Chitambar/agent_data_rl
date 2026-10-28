# Task H physics audit: single-arm insertion first

2026-10-27. Scope: right arm inserts the SanDisk USB-C SD reader into the second
USB-C port on the GMKtec PC. A later bimanual model may have the left arm hold the
PC. This audit distinguishes measured inputs, provisional inputs and model limits.
It does not claim that the physical simulator has been validated.

## Existing measured inputs

- Wrist and ZED camera intrinsics and poses, the left-to-right base transform, and
  arm joint offsets are in `blind_insertion/configs/real_cameras_duchamp3.json`.
  The physical pickup scene applies them in `scenes/physical_rig.py`.
- `data/real_yam/` and `results/real/` contain single-arm step/hold responses and
  CAN/encoder timing. The elbows show roughly 3-5 N m breakaway and about 1 N m
  sliding resistance; see `docs/FINDINGS.md`. The insertion config samples one
  MuJoCo `frictionloss` value per joint, which cannot by itself reproduce distinct
  static and sliding values or history-dependent re-sticking.
- Reader overall length, body width/thickness and neck width are measured by
  calipers. Other dimensions listed below are estimates.

## Physics ledger

| Sim item | Current implementation | Evidence needed before changing it | Acceptance check |
|---|---|---|---|
| Port and PC pose | `scenes/pc_bench.py` uses nominal port XY (0.321370, 0.236917) m, 25 mm port height, and 30 mm from the panel's left edge. PC case boxes are photo estimates. | Survey port center/axis, panel, PC feet and table plane in the right-base frame. Record independent views and uncertainty. | Held-out pose check and plug-tip alignment; keep the original USB-C plug dimensions. |
| Reader collision and inertia | `assets/builder.py` uses a 27 g nominal reader; body length 42 mm and neck thickness 4.5 mm are photo estimates. | Weigh the reader; caliper remaining dimensions; locate the mass center if practical. | Object dimensions/mass match measurements; resting pose and inertial response match held-out observations. |
| PC support | `assets/builder.py` places the fixture on a prescribed mocap body. It is effectively rigidly positioned during insertion, including in `PegEnv` and the physical pickup scene. | State whether the PC is clamped, taped or free; measure translation/rotation under gentle known loads. | If the PC moves materially, replace the mocap fixture with a dynamic or compliant support model before fitting insertion contact. |
| Reader/port contact | `PegEnv` samples plate friction 0.5-2.0 and contact time constant 3.5-7.1 ms; `builder.py` sets the fixture's contact geometry and solver parameters. No real contact trace has been fitted. | Manual supported force-versus-depth and withdrawal traces at centered and small known misalignments, with PC motion recorded. | Match onset, peak, seating depth, withdrawal and misalignment trend on withheld trials without changing plug size to force success. |
| Reader and PC on table | Table friction is nominal in `builder.py`; PC support is not dynamic. | Starting/sliding force, normal load and direction on the actual surface; note tipping versus translation. | Match object displacement and rest behavior at task speeds. |
| Arm servo and friction | `task_h_reader_real_acc.json` uses accumulated targets, per-joint friction ranges, gain scales and 0/1-step latency. `PickEnv` now uses the same joint-3 friction upper bound of 4 N m. | New ramp, triangle and dwell/reversal trials at held-out task poses; normal payload, warm and cold. | Match onset, stop, reversal, final error and velocity, not just a single step threshold. Test held-out poses. |
| Output compliance and lost motion | Randomized backlash is included; output-side backlash and flex have not been measured. | Optional dial-indicator or external visual displacement during small load reversals. Encoder readings alone are insufficient. | Match output/tool motion separately from encoder motion. |
| Wrist payload | Robot model still contains nominal camera/bracket inertial properties. | Mass and center of mass of webcam, bracket and supported wiring, plus held-pose bias. | Match held-pose and small-step response without using friction to hide gravity errors. |
| Gripper and in-hand pose | `PickEnv` uses the FlexPoint visual/physics model and assumed closing force, pad contact and speed. Dedicated gripper identification remains deferred. | At minimum, observe reader pose relative to the wrist after a normal grasp and its variability. Detailed pad tests remain a separate decision. | Plug-tip pose before insertion matches reality; no unexplained slip or bounce in held-out pickup trials. |
| Camera timing and appearance | Physical pickup uses calibrated top/wrist poses. Insertion `PegEnv` uses the right wrist spec. Image timing, exposure and lighting are only approximated. | `measure_cameras.py` frames/timestamps, a timed visible trigger, exposure settings, and paired stationary/moving views. | Rendered and real policy crops, blur, occlusion and frame age agree over held-out poses. |

## Sequence for importing measurements

1. Establish a known single-arm setup: PC support, table plane, port frame and
   reader mass/geometry. Record uncertainty and held-out trials.
2. Fit arm response using measured commands and native encoder arrivals. Keep the
   existing 50 Hz controller semantics; compare against held-out ramp/reversal data.
3. Fit contact only after geometry and PC support are fixed. MuJoCo's `solref`,
   `solimp` and friction interact, so tune against multiple force/depth/offset
   conditions rather than one successful insertion.
4. Compare paired real/sim trajectories and policy images through pickup, approach,
   initial contact and seating. Use the residuals to decide whether more measurements
   or a different model are needed.
5. Add the left arm and a PC support/contact model when starting bimanual trials.
   `scenes/pc_bench.py` currently parks the left arm as a visual, noncolliding body;
   measurements alone cannot make it hold the PC.

Collection commands and schemas are in `scripts/real/README.md`. Keep measured,
estimated and randomized parameters separate; a successful sim rollout alone is
not evidence that its forces or geometry are correct.
