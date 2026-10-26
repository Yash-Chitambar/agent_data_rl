"""Replay the recorded arm pose with measured camera calibration, no hardware access."""
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from blind_insertion.scenes import pc_bench as S, physical_rig as R
from blind_insertion.assets.real_camera import real_policy_image
import cv2
out=Path('results/H/physical_rig');out.mkdir(parents=True,exist_ok=True)
m,lay=S.build_scene();d=mujoco.MjData(m);R.reference_pose(m,d,lay)
S.place_pc(m,d);S.place_reader(m,d,lay);mujoco.mj_forward(m,d)
r=mujoco.Renderer(m,height=640,width=640)
ref=Path('data/sim_ref/2026-10-25_scene')
checks={}
snapshot=json.loads((ref/'pointed.json').read_text())
for name,side in [('top',None),('wrist_cam','right'),('left_wrist_cam','left')]:
 r.update_scene(d,camera=name);sim=r.render();Image.fromarray(sim).save(out/(name+'.png'))
 if side:
  real=real_policy_image(cv2.imread(str(ref/f'pointed_wrist_{side}.png')),R.load(),640,side)
  cid=m.camera(name).id; expected=snapshot['wrist_cam_in_right_base'][side]['pos']
  checks[side+'_camera_position_error_m']=float(np.linalg.norm(d.cam_xpos[cid]-expected))
 else:real=R.real_top_policy_image(cv2.imread(str(ref/'pointed_zed_left.png')),640)
 pair=Image.new('RGB',(1280,640));pair.paste(Image.fromarray(real),(0,0));pair.paste(Image.fromarray(sim),(640,0));pair.save(out/(name+'_comparison.png'))
r.close()
checks['both_flexpoint_meshes']=bool(sum(m.geom(i).name.startswith('') and m.geom_dataid[i]>=0 and 'soft_tips' in m.mesh(m.geom_dataid[i]).name for i in range(m.ngeom))==4)
checks['target_port_xy']=list(S.PORT_XY);checks['reader_xy']=list(S.READER_XY)
checks['camera_check_passed']=all(checks[k]<.0001 for k in ('right_camera_position_error_m','left_camera_position_error_m'))
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
assert checks['camera_check_passed'] and checks['both_flexpoint_meshes']
