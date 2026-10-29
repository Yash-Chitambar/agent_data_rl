"""Pure CPU checks of geometric seating, including false-success cases."""
import json
from pathlib import Path
import torch
from blind_insertion.scenes import pc_bench as S
from blind_insertion.envs.bimanual_runtime import BimanualRuntime

profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m,lay=S.build_scene(detailed=False,bimanual=True,support_mode='free',profile=profile,measured_support=True)
e=BimanualRuntime.__new__(BimanualRuntime);e.lay=lay;e.device=torch.device('cpu')
e.pc_qadr=int(m.joint('pc_free_joint').qposadr[0])
e.q=torch.tensor(m.qpos0,dtype=torch.float32).repeat(5,1)
f,p=e.pc_qadr,lay.peg_qadr;depth=float(lay.depth[0])
e.q[:,f:f+7]=torch.tensor([0.,0.,0.,1.,0.,0.,0.])
e.q[:,p:p+7]=torch.tensor([0.,0.,-depth,1.,0.,0.,0.])
e.q[1,p]=.005 # outside aperture
e.q[2,p+2]=.001 # in front
e.q[3,p+2]=-depth-.01 # through back of socket
# Move both bodies together: seating must be invariant to world translation.
e.q[4,p:p+3]+=torch.tensor([.1,.2,.3]);e.q[4,f:f+3]+=torch.tensor([.1,.2,.3])
r=e.insertion_metrics()
assert r['geometric_seated'].tolist()==[True,False,False,False,True]
out={'geometric_seated':r['geometric_seated'].tolist(),'socket_depth_m':depth,'checks':['aligned seated','lateral miss rejected','front gap rejected','over-insertion rejected','joint world translation invariant'],'scope':'geometry only; not physical insertion or electrical connection'}
Path('results/usbc/bimanual_20261029/insertion_metric_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
