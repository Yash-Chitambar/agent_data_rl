"""Four-world diagnostic of the existing expert on the physical rig; reports, not success assertions."""
import json,torch
from pathlib import Path
from blind_insertion.envs.pick_env import PickEnv, PickCfg, Expert
c=PickCfg(nworld=4,render=False,seed=12,rand={})
e=PickEnv(c); ex=Expert(e); ex.reset()
for _ in range(e.T):
 a,done=ex.act();e.step(a)
 assert bool(torch.isfinite(e.t_qpos).all())
r={k:v.cpu().tolist() for k,v in ex.result().items()}
r['phase']=ex.phase.cpu().tolist();r['finite_state']=True
print(r);Path('results/H/physical_rig/pick_smoke.json').write_text(json.dumps(r,indent=2)+'\n')
