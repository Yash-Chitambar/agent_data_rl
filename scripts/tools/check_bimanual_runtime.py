"""Bilateral control, selective reset and moving-port checks on two GPU worlds."""
import json
from pathlib import Path
import torch
from blind_insertion.envs.bimanual_runtime import BimanualCfg,BimanualRuntime

e=BimanualRuntime(BimanualCfg(substeps=2,right_fric_comp={'scale':.5,'ramp':.02}))
assert e.mjm.dof_frictionloss[e.indices['right'][1][1]]==1.0
assert e.mjm.dof_frictionloss[e.indices['left'][1][1]]==.01
a=torch.zeros((2,14),device=e.device)
for side,offset in [('right',0),('left',7)]:
 q,v,act,grip,limits=e.indices[side];a[:,offset:offset+6]=e.q[:,q];a[:,offset+6]=1
 a[:,offset]+=.002
p0=e.plug_in_port()[0].clone()
e.q[1,e.pc_qadr]+=.01
assert not torch.allclose(e.plug_in_port()[0][1],p0[1])
assert torch.allclose(e.plug_in_port()[0][0],p0[0])
e.reset(torch.tensor([1],device=e.device))
assert torch.allclose(e.q[1],e.initial_q)
for _ in range(3):e.step(a)
assert torch.isfinite(e.q).all()
for side,offset in [('right',0),('left',7)]:
 q,v,act,grip,limits=e.indices[side]
 assert torch.allclose(e.ctrl[:,act],a[:,offset:offset+6])
 assert (e.ctrl[:,grip]==4).all()
assert (e.force[:,e.indices['left'][1]]==0).all()
assert e.force[:,e.indices['right'][1]].abs().max()<=.75
saved=e.q[1].clone();e.reset(torch.tensor([0],device=e.device))
assert torch.equal(e.q[1],saved)
try:e.step(torch.zeros((2,7),device=e.device))
except ValueError:pass
else:raise AssertionError('legacy 7D action accepted')
r={'worlds':2,'action_dimension':14,'checks':['both arms controlled','moving port observed','selective reset','finite stepping','right-only optional compensation','legacy action rejected'],'scope':'physics runtime smoke; task reward, sensor and reset distribution integration unfinished'}
Path('results/usbc/bimanual_20261029/runtime_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
