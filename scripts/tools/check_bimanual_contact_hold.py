"""CPU/GPU hold parity initialized from actual nominal insertion rollout states."""
import json
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
from blind_insertion.scenes import physical_rig as R

out=Path('results/usbc/bimanual_20261029/reach')
prefix='insert_settled_clearance_identified_feedback_nominal_pick'
phases=('hold','seat_hold')
e=BimanualRuntime(BimanualCfg(nworld=2,friction_scale_range=(1.,1.),
    gravity_scale_jitter=0.,pc_friction_range=None))
m=e.mjm
cpu=[mujoco.MjData(m) for _ in phases]
with np.load(out/(prefix+'_states.npz')) as qs, np.load(out/(prefix+'_dynamics.npz')) as ds:
    for i,phase in enumerate(phases):
        cpu[i].qpos[:]=qs[phase];cpu[i].qvel[:]=ds[phase+'_qvel']
        cpu[i].ctrl[:]=ds[phase+'_ctrl'];mujoco.mj_forward(m,cpu[i])
        e.q[i]=torch.as_tensor(cpu[i].qpos,device=e.device,dtype=e.q.dtype)
        e.v[i]=torch.as_tensor(cpu[i].qvel,device=e.device,dtype=e.q.dtype)
with wp.ScopedDevice(e.cfg.device):mjw.forward(e.m,e.d)
action=torch.zeros((2,14),device=e.device)
for side,offset in (('right',0),('left',7)):
    q,v,act,grip,limits=e.indices[side]
    action[:,offset:offset+6]=torch.tensor(np.stack([d.ctrl[act] for d in cpu]),device=e.device)
    action[:,offset+6]=-1 if side=='right' else 1
driver=mujoco.MjModel.from_xml_path('results/usbc/bimanual_20261029/driver_model/driver_gravity.xml')
dd=mujoco.MjData(driver);cal=R.load();trace=[]
for step in range(50):
    for d in cpu:
        for side in ('right','left'):
            q,v=e.indices[side][:2];params=e.profile['arms'][side]
            dd.qpos[:6]=d.qpos[q]-np.array(cal['arms'][side]['joint_offsets_rad'])
            mujoco.mj_kinematics(driver,dd);mujoco.mj_comPos(driver,dd)
            mujoco.mj_rne(driver,dd,0,dd.qfrc_bias)
            d.qfrc_applied[v]=np.array(params['driver_gravity_comp_factor'])*dd.qfrc_bias[:6]-np.array(params['gravity_scale'])*d.qfrc_gravcomp[v]
        for _ in range(20):mujoco.mj_step(m,d)
    e.step(action)
    metrics={k:v.detach().cpu().tolist() for k,v in e.insertion_metrics().items()}
    trace.append({'time_s':(step+1)*.02,'gpu_metrics':metrics,
                  'max_qpos_difference':np.max(abs(e.q.cpu().numpy()-np.stack([d.qpos for d in cpu])),axis=1).tolist()})
cpu_metric=BimanualRuntime.__new__(BimanualRuntime)
cpu_metric.lay=e.lay;cpu_metric.pc_qadr=e.pc_qadr;cpu_metric.device=torch.device('cpu')
cpu_metric.q=torch.tensor(np.stack([d.qpos for d in cpu]),dtype=torch.float64)
report={'scope':'One second open-loop hold from saved aligned and seated states; not GPU pickup/insertion rollout',
        'comparison_note':'5e-3 maximum raw qpos difference is a coarse numerical guard, not a physical alignment tolerance; use insertion metrics for seating',
        'phases':phases,'impratio':float(m.opt.impratio),'randomization':False,
        'final_cpu_metrics':{k:v.tolist() for k,v in cpu_metric.insertion_metrics().items()},
        'final_gpu_metrics':trace[-1]['gpu_metrics'],'buffer_usage':e.buffer_usage(),'trace':trace}
(out/'contact_hold_gpu_parity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='trace'},indent=2))
assert report['final_cpu_metrics']['geometric_seated']==[False,True]
assert torch.isfinite(e.q).all()
assert max(max(row['max_qpos_difference']) for row in trace)<.005, 'CPU/GPU state divergence'
assert all(row['gpu_metrics']['geometric_seated'][1] for row in trace), 'GPU lost seated hold'
