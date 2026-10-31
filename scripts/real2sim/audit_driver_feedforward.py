"""Compare the pinned offline driver model with captured computed feedforward.

The captured field is asynchronous and not measured motor torque.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--bundle',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();m=mujoco.MjModel.from_xml_path(str(a.model));d=mujoco.MjData(m)
m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.jnt_limited[:]=0;m.opt.gravity[:]=[0,0,-9.81]
reports={}
for run in sorted((a.bundle/'motion').glob('left_rest_j*_fit')):
    info=json.loads((run/'run.json').read_text())['arm']['left']['gains']
    factors=np.array(info['gravity_comp_factor'])[:6]
    friction=np.array(info['coulomb_friction'])[:6]
    with (run/'steps.csv').open() as f:rows=list(csv.DictReader(f))
    residual=[];gravity_residual=[];magnitude_residual=[]
    for r in rows:
        if not r['computed_motor_feedforward_driver']:continue
        d.qpos[:6]=json.loads(r['q']);d.qvel[:]=0;d.qacc[:]=0
        mujoco.mj_inverse(m,d)
        gravity=d.qfrc_inverse[:6]*factors
        recorded=np.asarray(json.loads(r['computed_motor_feedforward_driver']))[:6]
        residual.append(gravity+friction*np.sign(json.loads(r['qd']))-recorded)
        gravity_residual.append(gravity-recorded)
        magnitude_residual.append(np.abs(recorded-gravity)-friction)
    x,y=np.asarray(residual),np.asarray(gravity_residual)
    reports[run.name]={'samples':len(x),'reconstruction_rmse_nm':np.sqrt(np.mean(x*x,axis=0)).tolist(),
                       'friction_magnitude_residual_rmse_nm':np.sqrt(np.mean(np.asarray(magnitude_residual)**2,axis=0)).tolist(),
                       'gravity_only_rmse_nm':np.sqrt(np.mean(y*y,axis=0)).tolist(),
                       'input_sha256':hashlib.sha256((run/'steps.csv').read_bytes()).hexdigest()}
report={'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
        'scope':'last computed feedforward field vs reconstruction; not torque accuracy',
        'limitations':['asynchronous recorded command and observed q/qd','driver source revision unconfirmed','folded-rest poses only'],
        'runs':reports}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(reports,indent=2))
