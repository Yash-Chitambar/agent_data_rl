"""Offline session preparation and timeline export. Never imports robot/camera drivers."""
import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def prepare(out):
    out=Path(out)
    if out.exists(): raise ValueError(f'refusing overwrite: {out}')
    root=Path(__file__).resolve().parents[2]
    spec=root/'blind_insertion/configs/real_cameras_duchamp3.json'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    trials=[]
    for joint in range(1,7):
        trials.append(dict(id=f'left_rest_j{joint}_fit',kind='controller_response',side='left',joint=joint,
                           split='fit',repeats=5,status='pending',amplitude_rad=.01))
    for side in ('left','right'):
        for condition in ('approach_no_contact','establish_contact','support_hold','release','slip_or_pc_motion'):
            trials.append(dict(id=f'pc_contact_{side}_{condition}',kind='pc_arm_contact',side=side,
                split='development',status='pending',contact_condition=condition))
    for name in ('left_task_pose_fit','left_task_pose_heldout','left_loaded_fit','left_loaded_heldout',
                 'left_gripper_empty_reader_pc','both_arms_stationary_rgb','bimanual_support_insert'):
        trials.append(dict(id=name,status='pending',split='validation' if 'heldout' in name else 'development'))
    out.mkdir(parents=True)
    manifest=dict(schema='bimanual_session_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,calibration_sha256=hashlib.sha256(spec.read_bytes()).hexdigest(),
        world_frame='right_arm_base',length_unit='m',angle_unit='rad',quaternion_order='wxyz',
        clock='host CLOCK_MONOTONIC; camera exposure clocks not aligned',
        support_conditions=['free','left_arm_supported','right_arm_supported','both_arms_supported'],
        commissioning_status='offline_prepared_only',working_policy_baseline='must capture current working checkout before collection',
        trials=trials)
    (out/'session.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fields=['trial_id','host_monotonic_ns','arm','pc_contact_region','arm_contact_region','event',
            'support_condition','pc_translation_m','pc_rotation_rad','grasp_slip_observed',
            'force_n','force_instrument','notes']
    with (out/'contact_events.csv').open('w',newline='') as f:csv.writer(f).writerow(fields)
    (out/'trial_queue.json').write_text(json.dumps(trials,indent=2)+'\n')
    (out/'README.md').write_text('Prepared offline. No hardware readiness gate has passed.\n'
        'Keep both-arm observations/actions and all three cameras on the same host monotonic clock.\n'
        'Host timestamps provide software alignment, not synchronized exposures.\n'
        'Reserve complete poses/trials for validation before fitting.\n'
        'Leave force blank unless measured with an identified force instrument.\n'
        'Contact includes either gripper, fingers, wrist or arm link; identify both contacting surfaces.\n')


def timeline(session,out):
    session=Path(session);out=Path(out)
    if out.exists():raise ValueError(f'refusing overwrite: {out}')
    events=[];streams=set()
    for p in sorted(session.rglob('*.csv')):
        if p.resolve()==out.resolve():continue
        with p.open(newline='') as f:
            for row in csv.DictReader(f):
                field=next((k for k in ('host_monotonic_ns','host_receive_ns','t_tick','t_obs') if row.get(k)),None)
                if field is None:continue
                stamp=int(row[field]) if field.endswith('_ns') else int(float(row[field])*1e9)
                if stamp<0:raise ValueError(f'negative timestamp: {p}')
                stream=str(p.relative_to(session));streams.add(stream)
                events.append(dict(host_monotonic_ns=stamp,source=stream,timestamp_field=field,record=row))
    events.sort(key=lambda e:e['host_monotonic_ns'])
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:
        for e in events:f.write(json.dumps(e)+'\n')
    return dict(events=len(events),streams=sorted(streams),
        limitation='Offline host-clock alignment only; does not establish exposure sync or policy consumption')


def main():
    ap=argparse.ArgumentParser(description=__doc__);sp=ap.add_subparsers(dest='cmd',required=True)
    p=sp.add_parser('prepare');p.add_argument('--out',required=True)
    p=sp.add_parser('timeline');p.add_argument('--session',required=True);p.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.cmd=='prepare':prepare(a.out);print(a.out)
    else:print(json.dumps(timeline(a.session,a.out),indent=2))

if __name__=='__main__':main()
