"""Export measured left/right evidence for Mason; never opens hardware."""
import argparse
import hashlib
import json
from pathlib import Path


def export(session, right_session, camera_spec, out):
    session,right_session,camera_spec,out=map(Path,(session,right_session,camera_spec,out))
    if out.exists():raise ValueError(f'refusing overwrite: {out}')
    sources={}
    def read(p):
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    geometry=read(session/'support_geometry_profile.json')
    spec=read(camera_spec)
    controllers={}
    for side,base,pattern in [('left',session,'left_rest_j*_fit/run.json'),('right',right_session,'right_rest_empty_j*_triangle_001/run.json')]:
        runs=[]
        for p in sorted((base/'motion').glob(pattern)):
            x=read(p);arm=x['arm'][side]
            runs.append({'source':str(p),'joint':x['args']['joint'],'gains':arm['gains'],
                         'joint_offsets_rad':arm['joint_offset_rad'],'steps':x['steps'],
                         'unsent_steps':x['unsent_steps'],'fit_status':'observed_not_fitted'})
        controllers[side]={'trials':runs,'complete_task_pose_identification':False}
    grasp=[]
    for p in sorted((session/'grasp').glob('left_reader_*.json')):
        x=read(p);grasp.append({'source':str(p),'readings':x['readings'],
                               'force_measured':False,'operator_notes':x.get('operator_notes')})
    result={'schema':'bimanual_evidence_profile_v1','status':'provisional; not deployable or physics-validated',
        'world_frame':'right_arm_base','quaternion_order':'wxyz','units':'SI',
        'support_geometry':geometry,'left_base_in_right_base':spec['arms']['left_base_in_right_base'],
        'controllers':controllers,'left_grasp_trials':grasp,
        'right_grasp':'retain existing right-session measurements; do not copy left readings',
        'support_modes':['free','left_arm_supported','right_arm_supported','both_arms_supported'],
        'contact_parameters':{'pc_left_arm':None,'pc_right_arm':None,'pc_table':None},
        'unresolved':['mounting plate footprint','loaded/elevated held-out responses',
                      'both PC-arm contact datasets','contact stiffness/force unmeasured',
                      'actual policy-consumed camera frames','working policy controller provenance'],
        'sources_sha256':sources}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('session','right-session','camera-spec','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();x=export(a.session,a.right_session,a.camera_spec,a.out)
    print(json.dumps({'output':a.out,'left_trials':len(x['controllers']['left']['trials']),
                      'right_trials':len(x['controllers']['right']['trials']),'status':x['status']},indent=2))

if __name__=='__main__':main()
