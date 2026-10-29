"""Check calibrated three-camera GPU output and actor/privileged state separation."""
import json
from pathlib import Path
import torch
from PIL import Image
from blind_insertion.envs.bimanual_runtime import BimanualCfg,BimanualRuntime

e=BimanualRuntime(BimanualCfg(nworld=1,render_rgb=True,camera_resolution=320))
out=Path('results/usbc/bimanual_20261029/runtime_rgb');out.mkdir(parents=True,exist_ok=True)
frames=e.render()
assert set(frames)=={'top','wrist_cam','left_wrist_cam'}
for name,frame in frames.items():
 assert frame.shape==(1,320,320,3) and frame.dtype==torch.uint8
 assert frame.float().std()>5,'empty or uniform camera image'
 Image.fromarray(frame[0].cpu().numpy()).save(out/(name+'.png'))
obs=e.proprioception();assert obs['encoder_rad'].shape==(1,12)
assert set(obs)=={'encoder_rad','encoder_velocity_rad_s'}
q,v,act,grip,lim=e.indices['right'];expected=e.q[:,q]-e.encoder_offsets[0]
assert torch.allclose(obs['encoder_rad'][:,:6],expected)
before={k:v.clone() for k,v in frames.items()}
e.q[:,e.indices['right'][0][0]]+=.1
later=e.render()
assert not torch.equal(before['wrist_cam'],later['wrist_cam'])
assert all(torch.equal(before[k],frames[k]) for k in frames),'returned frames alias render buffer'
report={'resolution':[320,320],'cameras':list(frames),'rgb_only':True,'checks':['nonempty RGB','camera follows moving arm','frame ownership','encoder offsets applied once','object poses excluded from proprioception'],'limitations':['instantaneous frames; exposure latency not modeled','gripper encoder mapping not identified']}
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
