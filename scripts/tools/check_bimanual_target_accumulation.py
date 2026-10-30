"""Controller stress test, not identification of the robot's friction dynamics.

Use a one-joint MuJoCo load above the relative-action torque ceiling, to check
that the bimanual target generator can escape it without unbounded windup.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blind_insertion.envs.bimanual_task import BimanualTask, BimanualTaskCfg

limits=torch.tensor([[-2.,2.]]*6)
traces={}
for mode in ('relative','accumulate'):
    t=BimanualTask.__new__(BimanualTask)
    t.cfg=BimanualTaskCfg(action_mode=mode)
    t.joint_target=torch.zeros((1,2,6))
    t.runtime=SimpleNamespace(indices={s:(None,None,None,None,limits) for s in ('right','left')})
    m=mujoco.MjModel.from_xml_string("""<mujoco>
      <option timestep="0.001" gravity="0 0 0"/>
      <worldbody><body><joint name="j" axis="0 0 1" frictionloss="4" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size=".02" mass="1"/></body></worldbody>
      <actuator><position joint="j" kp="80" kv="5"/></actuator>
      </mujoco>""")
    d=mujoco.MjData(m);trace=[]
    for k in range(50):
        measured=torch.zeros((1,2,6));measured[0,1,2]=float(d.qpos[0])
        a=torch.zeros((1,14));a[0,9]=1 if k<10 else 0
        target=t._update_joint_targets(a,measured)
        error=float(target[0,1,2]-measured[0,1,2])
        assert abs(error)<=t.cfg.target_cap_rad+1e-7
        assert torch.count_nonzero(target[:,0])==0
        d.ctrl[0]=float(target[0,1,2])
        for _ in range(20):mujoco.mj_step(m,d)
        trace.append([d.time,d.qpos[0],d.ctrl[0],error])
    traces[mode]=np.asarray(trace)
# MuJoCo's regularized friction permits creep; compare displacement, not perfect sticking.
assert traces['accumulate'][-1,1]>4*abs(traces['relative'][-1,1])
assert traces['accumulate'][-1,1]>.04
# Saturation at limits and reversal must release the old target immediately.
t.joint_target[:]=1.99
measured=torch.full_like(t.joint_target,1.99)
a=torch.ones((1,14))
t._update_joint_targets(a,measured)
assert float(t.joint_target.max())<=2.
a[:]=-1
t._update_joint_targets(a,measured)
assert float(t.joint_target.max())<1.99
out=Path('results/usbc/bimanual_20261029/control');out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(8,4))
for mode,x in traces.items():
    ax.plot(x[:,0],np.rad2deg(x[:,1]),label=mode+' position')
    ax.plot(x[:,0],np.rad2deg(x[:,2]),'--',label=mode+' target',alpha=.65)
ax.set(xlabel='Time (s)',ylabel='Angle (degrees)',title='Synthetic 4 N·m load: target accumulation stress test')
ax.legend();fig.tight_layout();fig.savefig(out/'target_accumulation.png',dpi=140)
report={'scope':'synthetic single-joint stress test, not calibrated elbow dynamics',
        'limitation':'regularized MuJoCo friction creeps below the nominal threshold; not a stick-slip model',
        'load_friction_nm':4.,'kp':80.,'relative_torque_ceiling_nm':3.2,
        'accumulated_target_cap_rad':.1,
        'final_position_rad':{mode:float(x[-1,1]) for mode,x in traces.items()},
        'checks':['movement under load above relative torque ceiling','bounded target error','arm independence','joint-limit clipping','reversal releases retained target']}
(out/'target_accumulation.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
