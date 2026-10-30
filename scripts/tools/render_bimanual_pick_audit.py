"""Render saved CPU rollout states for a before/after grasp review."""
import json
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw
from blind_insertion.scenes import pc_bench as S
out=Path('results/usbc/bimanual_20261029/reach')
profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m,l=S.build_scene(detailed=True,bimanual=True,support_mode='free',profile=profile,measured_support=True,controller_profile='native_free_settle_20261028')
d=mujoco.MjData(m);r=mujoco.Renderer(m,640,640)
panel=Image.new('RGB',(1920,680));draw=ImageDraw.Draw(panel)
try:
 for i,(file,phase,label) in enumerate([
  ('legacy_nominal_pick_states.npz','close','OLD: pads above reader'),
  ('nominal_pick_states.npz','close','NEW: pad-derived grasp height'),
  ('nominal_pick_states.npz','lift','NEW: reader lifted under contact physics')]):
  with np.load(out/file) as states:d.qpos[:]=states[phase]
  mujoco.mj_forward(m,d)
  cam=mujoco.MjvCamera();cam.lookat[:]=[.204,.221,.015];cam.distance=.29;cam.azimuth=140;cam.elevation=-25
  r.update_scene(d,camera=cam);frame=r.render().copy()
  panel.paste(Image.fromarray(frame),(i*640,40));draw.text((i*640+10,12),label,fill='white')
finally:r.close()
panel.save(out/'grasp_comparison.png')
