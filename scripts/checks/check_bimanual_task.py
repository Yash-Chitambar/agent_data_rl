"""GPU episode-boundary checks; synthetic seating is not a solved physical rollout."""
import json
from pathlib import Path
import torch
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.envs.bimanual_task import BimanualTaskCfg,BimanualTask

t=BimanualTask(BimanualTaskCfg(physics=BimanualCfg(nworld=2,substeps=2),episode_s=.006,success_hold_s=.004))
t.reset();zero=torch.zeros((2,14),device=t.device);zero[:,[6,13]]=1
for _ in range(3):obs,reward,term,trunc,info=t.step(zero)
assert trunc.all() and not term.any() and not reward.any()
assert 'privileged' not in obs and 'privileged' in info
assert obs['target_error_rad'].shape==(2,12)
torch.testing.assert_close(obs['target_error_rad'],(t.joint_target-t._measured_model_angles()).flatten(1))
try:t.step(zero)
except RuntimeError:pass
else:raise AssertionError('terminal episode advanced')
t.reset();e=t.runtime;f,p=e.pc_qadr,e.lay.peg_qadr
# Plant two exact seated states for state-machine checks, without pretending a policy inserted them.
e.q[:,p:p+7]=e.q[:,f:f+7]
from blind_insertion.envs import kinematics as K
rot=K.quat_to_mat(e.q[:,f+3:f+7]);local=torch.tensor([0.,0.,-float(e.lay.seating_depth[0])],device=t.device)
e.q[:,p:p+3]+=(rot@local)
r,term,trunc,info=t._transition();assert not term.any() and not r.any()
# Break seating in world 1: hold must restart there.
e.q[1,p]+=.02
r,term,trunc,info=t._transition();assert term.tolist()==[True,False] and r.tolist()==[1.,0.]
assert t.seated_steps.tolist()==[2,0]
t.joint_target[1,1,2]+=.07
saved_target=t.joint_target[1].clone()
saved=e.q[1].clone();t.reset(torch.tensor([0],device=t.device));assert torch.equal(e.q[1],saved)
assert t.steps.tolist()==[0,2] and not t.done.any()
assert torch.equal(t.joint_target[1],saved_target)
torch.testing.assert_close(t.joint_target[0],t._measured_model_angles()[0])
report={'checks':['timeout distinct from termination','no auto-reset of terminal observation','reject advance after done','success requires continuous seating','broken seating resets hold','selective episode reset','privileged state separated','target error observed','selective target reset'],'scope':'synthetic episode-boundary validation; not a physically successful insertion rollout'}
Path('results/usbc/bimanual_20261029/task_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
