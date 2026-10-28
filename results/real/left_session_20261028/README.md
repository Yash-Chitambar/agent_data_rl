# Left-arm session, 2026-10-28

Raw artifacts: /home/yashc/left_bimanual_session on duchamp3. This compact report is not the full synchronized recording.

Completed: three-camera reference captures; five-second left stationary check; left-wrist verification at two board positions (both image-consistency PASS); six folded-rest joint trials, five cycles each, 7,800 logged steps and zero unsent commands; one operator-confirmed stationary reader retention after two unsuccessful placements. Reader readback 0.33615 versus empty 0.04864 in the successful trial. No grip force measured.

Elevated acquisition failed the unchanged 0.020-rad attainment requirement with joint-3 error -0.04425 rad. Zero response cycles were collected there; the failed approach and native feedback remain diagnostic data. Checked recovery completed with torque off. No gain, friction or calibration refit was adopted.

PC survey and operator mounting measurement reveal a shared raised plate under the robot bases, 0.023 m above the lower PC tabletop. The scene must contain both surfaces. The plate footprint and absolute base-origin relation require verification before contact motion. Lower-table z=-0.023 m is a provisional physical reference, not an adopted live-safety modification. Board planes are approximately -0.021 m; camera agreement is not absolute metrology.

PC–arm contacts on both arms, loaded/elevated response, grasp distributions and complete bimanual support trials remain pending. Host-clock coverage of camera and controller logs is verified for the folded sweep; hardware exposures are not synchronized, and these are not policy-consumption logs.

Reference correction: operator confirmed prior ruler gaps 4.5/5.1/4.9 were centimetres. The raw file mislabeled them gap_mm; it was preserved. Corrected model-frame table heights are -27.88/-33.74/-30.16 mm, median -30.16 mm. The model base mesh bottom is -10 mm, so the 23-mm physical mounting height alone must not be interpreted as model table z. The earlier -23-mm assumption is superseded for model-frame planning. Table observed range is not a calibrated confidence interval.

Non-contact left PC survey completed using native calibration-controller gains and the checked 55-degree view. Joint observations and all three camera streams were saved; selected wrist image arrival is 87.4 ms from the middle joint observation, with unknown exposure timing. Arm returned to rest and torque off. No PC-arm contact trial has run. Support role (brace on table versus grasp) remains to be selected before contact planning.
