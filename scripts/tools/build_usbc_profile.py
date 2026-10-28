"""Build a provisional rig profile from the recorded duchamp3 measurements.

This exports evidence for review; it does not change simulator defaults or certify physics.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, default=ROOT / 'data/real_yam/usbc_lab_20261027')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    measurements = a.session / 'measurements.csv'
    run = a.session / 'motion/right_pickup_mid_empty_hold_001/run.json'
    camera = ROOT / 'blind_insertion/configs/real_cameras_duchamp3.json'
    with measurements.open(newline='') as f:
        rows = list(csv.DictReader(f))
    by_id = {row['trial_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise SystemExit('duplicate measurement trial IDs')

    def recorded(trial, expected_unit):
        r = by_id[trial]
        if r['unit'] != expected_unit or not r['value']:
            raise ValueError(f'{trial}: missing value or unexpected unit')
        return {'value': float(r['value']), 'unit': r['unit'],
                'uncertainty': float(r['uncertainty']) if r['uncertainty'] else None,
                'uncertainty_reason': None if r['uncertainty'] else 'not reported',
                'source_trial': trial, 'method': r['method'], 'notes': r['notes'],
                'validation_status': 'operator_reported'}

    pc = {k: recorded(v, unit) for k, v, unit in (
        ('mass', 'pc_mass_01', 'kg'), ('width', 'pc_width_01', 'm'),
        ('depth', 'pc_depth_front_back_01', 'm'), ('body_height', 'pc_height_without_feet_01', 'm'),
        ('height_with_feet', 'pc_height_with_feet_01', 'm'), ('pad_count', 'pc_feet_count_01', 'count'),
        ('pad_diameter', 'pc_pad_diameter_01', 'm'))}
    reader = {k: recorded(v, unit) for k, v, unit in (
        ('mass', 'reader_mass_01', 'kg'), ('body_width', 'reader_body_width_01', 'm'),
        ('body_length', 'reader_body_length_01', 'm'), ('body_thickness', 'reader_body_thickness_01', 'm'),
        ('neck_length', 'reader_neck_length_01', 'm'), ('neck_width', 'reader_neck_width_01', 'm'),
        ('neck_thickness', 'reader_neck_thickness_01', 'm'), ('plug_protrusion', 'reader_plug_protrusion_01', 'm'))}
    for trial in ('reader_plug_protrusion_02', 'reader_plug_protrusion_03'):
        recorded(trial, 'm')
    control = json.loads(run.read_text())['arm']['right']
    profile = {
        'schema': 'usbc-rig-profile-v0', 'profile_id': 'duchamp3-usbc-provisional-20261027',
        'validation_status': 'provisional_not_for_training_or_deployment',
        'frame': 'right_arm_base', 'quaternion_order': 'wxyz',
        'task': {'target': 'SanDisk USB-C SD reader to GMKtec second USB-C port',
                 'support_mode': 'free', 'support_validation': 'manual contact motion not yet characterized'},
        'pc': {'measurements': pc,
               'pad_height_derived': {'value': pc['height_with_feet']['value'] - pc['body_height']['value'],
                                      'unit': 'm', 'uncertainty': None,
                                      'source_trials': ['pc_height_with_feet_01', 'pc_height_without_feet_01'],
                                      'validation_status': 'derived_from_operator_report'},
               'pad_locations': {'value': 'near_bottom_corners', 'validation_status': 'operator_assumption',
                                 'inset_m': None},
               'mass_distribution_and_inertia': None, 'port_2_pose': None},
        'reader': {'measurements': reader,
                   'plug_protrusion_repeat_trials': ['reader_plug_protrusion_01','reader_plug_protrusion_02','reader_plug_protrusion_03'],
                   'legacy_plug_axial_length': {'value': 0.00665, 'unit': 'm',
                       'source': 'blind_insertion/assets/builder.py:USBC_PLUG_LEN',
                       'validation_status': 'legacy_unverified'},
                   'plug_length_note': 'Reported exposed protrusion is 0.0058 m; compare definitions and seating geometry before changing legacy plug dimensions.',
                   'neck_flexibility': None, 'inertia': None},
        'controller': {'source_run': str(run.relative_to(ROOT)), 'gains_and_compensation': control['gains'],
                       'joint_offset_rad': control['joint_offset_rad'],
                       'validation_status': 'logged_baseline_not_fitted', 'measured_torque': None},
        'camera': {'spec_path': str(camera.relative_to(ROOT)), 'spec_sha256': sha(camera),
                   'policy_crop_resolution': [320, 320],
                   'policy_consumption_timestamps': None,
                   'validation_status': 'calibration_retained_with_few_mm_relative_pose_uncertainty'},
        'unresolved': ['port_2 pose and recess', 'PC feet insets and material', 'PC/reader inertias',
                       'contact force and compliance', 'reader neck flexibility',
                       'actual policy-consumed frame IDs', 'held-out controller validation'],
        'source_sha256': {'measurements.csv': sha(measurements), 'motion_run.json': sha(run),
                          'camera_spec': sha(camera)},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(profile, f, indent=2, allow_nan=False)
        f.write('\n')
    print(a.out)


if __name__ == '__main__':
    main()
