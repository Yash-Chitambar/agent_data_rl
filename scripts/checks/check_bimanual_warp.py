"""CPU/Warp short-horizon parity on the measured free-PC bimanual scene."""
import json
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
from blind_insertion.scenes import pc_bench as S, physical_rig as R

profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m,lay=S.build_scene(detailed=True,bimanual=True,support_mode='free',profile=profile,
                    measured_support=True,controller_profile='native_free_settle_20261028')
d=mujoco.MjData(m);R.reference_pose(m,d,lay);S.place_pc(m,d);S.place_reader(m,d,lay)
for prefix in ('','left_'):
 for i in range(1,7):d.ctrl[m.actuator(prefix+f'joint{i}').id]=d.qpos[m.joint(prefix+f'joint{i}').qposadr][0]
mujoco.mj_forward(m,d)
wp.init()
with wp.ScopedDevice('cuda:0'):
 wm=mjw.put_model(m);wd=mjw.put_data(m,d,nworld=2,nconmax=512,njmax=3000)
 for _ in range(50):
  mujoco.mj_step(m,d);mjw.step(wm,wd)
 wp.synchronize()
 q=wd.qpos.numpy();v=wd.qvel.numpy()
 error=np.max(np.abs(q-d.qpos),axis=1)
 report={'worlds':2,'steps':50,'timestep_s':m.opt.timestep,'max_qpos_difference':error.tolist(),
         'finite':bool(np.isfinite(q).all() and np.isfinite(v).all()),
         'scope':'short hold with free PC/table contacts; not full task or hardware validation'}
 Path('results/usbc/bimanual_20261029/warp_parity.json').write_text(json.dumps(report,indent=2)+'\n')
 print(report)
 assert report['finite']
 assert error.max()<1e-3, 'CPU/Warp short-horizon qpos differs by >=1e-3'
