"""Bilateral control, selective reset and moving-port checks on two GPU worlds."""
import json
from pathlib import Path
import torch
from blind_insertion.envs.bimanual_runtime import BimanualCfg,BimanualRuntime

e=BimanualRuntime(BimanualCfg(substeps=2,right_fric_comp={'scale':.5,'ramp':.02},dynamics_profile=None))
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
# Identified real-arm dynamics (default): per-world friction, driver gravity residual on both arms.
d=BimanualRuntime(BimanualCfg(nworld=4,substeps=2))
prof=d.profile
for k,side in enumerate(('right','left')):
 v=d.indices[side][1];base=torch.tensor(prof['arms'][side]['coulomb_nm'],device=d.device)
 f=d.frictionloss[:,v];lo,hi=d.cfg.friction_scale_range
 assert ((f>=base*lo-1e-6)&(f<=base*hi+1e-6)).all() and (f[:,1:3].std(0)>0).all()
 assert torch.allclose(torch.tensor(d.mjm.dof_solref[v],device=d.device,dtype=torch.float32),torch.tensor(prof['friction_solref'],device=d.device).expand(6,2))
 res=d.gravity_residual(k)
 assert torch.isfinite(res).all() and res[:,1:3].abs().max()>0 and res[:,0].abs().max()<.05
mu=d.geom_friction[:,d.pc_feet,0];lo,hi=d.cfg.pc_friction_range
assert ((mu>=lo)&(mu<=hi)).all() and (mu.std(0)>0).all() and (mu==mu[:,:1]).all()
before=d.frictionloss.clone();mu_before=mu.clone();d.reset(torch.tensor([2],device=d.device))
assert torch.equal(mu_before[[0,1,3]],d.geom_friction[[0,1,3]][:,d.pc_feet,0]) and not torch.equal(mu_before[2],d.geom_friction[2,d.pc_feet,0])
assert torch.equal(before[[0,1,3]],d.frictionloss[[0,1,3]]) and not torch.equal(before[2],d.frictionloss[2])
hold=torch.zeros((4,14),device=d.device)
for side,offset in [('right',0),('left',7)]:hold[:,offset:offset+6]=d.q[:,d.indices[side][0]];hold[:,offset+6]=1
q0=d.q.clone()
for _ in range(50):d.step(hold)
drift=max((d.q[:,d.indices[s][0]]-q0[:,d.indices[s][0]]).abs().max().item() for s in ('right','left'))
assert torch.isfinite(d.q).all() and drift<.05
r={'worlds':2,'action_dimension':14,'checks':['both arms controlled','moving port observed','selective reset','finite stepping','right-only optional compensation','legacy action rejected',
   'identified dynamics: per-world friction within range and resampled only on reset','stiff friction impedance','PC-feet friction per world in range, resampled only on reset','finite driver gravity residual (none on joint 1)',f'reference hold under identified dynamics: max joint drift {drift:.4f} rad in 0.1 s'],'scope':'physics runtime smoke; task reward, sensor and reset distribution integration unfinished'}
Path('results/usbc/bimanual_20261029/runtime_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
