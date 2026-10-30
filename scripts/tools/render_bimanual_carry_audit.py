"""Compare failed and settled grasps using saved identified-dynamics rollout states."""
import json
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from blind_insertion.scenes import pc_bench as S
out=Path('results/usbc/bimanual_20261029/reach')
profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m,l=S.build_scene(detailed=True,bimanual=True,support_mode='free',profile=profile,measured_support=True)
d=mujoco.MjData(m);renderer=mujoco.Renderer(m,640,640)
panel=Image.new('RGB',(1920,680));draw=ImageDraw.Draw(panel)
cases=[
 ('clearance_identified_feedback_nominal_pick_states.npz','first_loss','Before: reader escapes the shallow grasp'),
 ('settled_clearance_identified_feedback_nominal_pick_states.npz','carry_high','After joint settling: reader retained during carry'),
 ('settled_clearance_identified_feedback_nominal_pick_states.npz','hold','After descent: held 6 mm ahead of target port')]
try:
 for i,(file,phase,title) in enumerate(cases):
  with np.load(out/file) as states:d.qpos[:]=states[phase]
  mujoco.mj_forward(m,d)
  cam=mujoco.MjvCamera();cam.lookat[:]=[.25,.23,.015];cam.distance=.32;cam.azimuth=140;cam.elevation=-25
  renderer.update_scene(d,camera=cam);panel.paste(Image.fromarray(renderer.render().copy()),(640*i,40))
  draw.text((640*i+10,12),title,fill='white')
finally:renderer.close()
panel.save(out/'identified_carry_comparison.png')
