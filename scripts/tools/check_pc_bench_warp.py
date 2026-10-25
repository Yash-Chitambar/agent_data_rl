"""One-world stepping and two-camera smoke check for the detailed bench."""
import json
from pathlib import Path
import torch
from PIL import Image
from blind_insertion.envs.pick_env import PickEnv, PickCfg
c=PickCfg(nworld=1,cam_res=(320,240)); c.rand=dict(cam_pos=0.01, cam_rot_deg=2.0, light=True, colors=True)
e=PickEnv(c)
for _ in range(3):
 e.step(torch.cat([e.q_ready[None],torch.ones((1,1),device=e.device)*4],1))
frames=e.render()
for name,im in frames.items():
 Image.fromarray(im[0].cpu().numpy()).save('results/H/scene_review/training_'+name+'.png')
r={'finite_qpos':bool(torch.isfinite(e.t_qpos).all()),'frames':{k:list(v.shape) for k,v in frames.items()},'textures':c.use_textures}
Path('results/H/scene_review/warp_check.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
assert r['finite_qpos']
