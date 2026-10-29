"""Render actual scene PC/reader closeups and save the dimensional audit."""
import json
from pathlib import Path
import mujoco,numpy as np
from PIL import Image
from blind_insertion.scenes import pc_bench as S,pc_rear
from blind_insertion.assets import builder as B
p=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m,l=S.build_scene(detailed=True,bimanual=True,support_mode='free',profile=p,measured_support=True)
d=mujoco.MjData(m);S.place_pc(m,d);S.place_reader(m,d,l);mujoco.mj_forward(m,d)
out=Path('results/usbc/bimanual_20261029/object_audit');out.mkdir(parents=True,exist_ok=True)
r=mujoco.Renderer(m,height=480,width=640)
try:
 for name,look,distance,az,el in [('rear_panel',d.xpos[m.body('fixture').id]+[0,-.020,.002],.20,0,0),
                                ('reader',d.xipos[m.body('peg').id],.16,90,-80)]:
  cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.lookat[:]=look;cam.distance=distance;cam.azimuth=az;cam.elevation=el
  saved=d.qpos[l.peg_qadr:l.peg_qadr+3].copy()
  if name=='rear_panel':d.qpos[l.peg_qadr:l.peg_qadr+3]=[0,0,-10]
  mujoco.mj_forward(m,d)
  r.update_scene(d,camera=cam);Image.fromarray(r.render().copy()).save(out/f'{name}.png')
  d.qpos[l.peg_qadr:l.peg_qadr+3]=saved;mujoco.mj_forward(m,d)
finally:r.close()
v=l.variants[0];total=max(c[2]+h[2] for c,h in v.peg_boxes)
assert np.isclose(total,.09046)
assert v.peg_length==B.USBC_PLUG_LEN
assert v.hole_w==B.USBC_REC[0] and v.hole_h==B.USBC_REC[1]
report={'pc':p['pc']['measurements'],'rear_panel':pc_rear.layout(p),
 'reader':{'total_length_m':total,'housing_length_m':v.peg_boxes[0][1][2]*2,'outer_width_m':B.READER_BODY[0],
 'outer_thickness_m':B.READER_BODY[1],'silver_length_m':.0296,'silver_width_m':.0346,'neck_length_m':v.handle_height,
 'mass_kg':v.peg_mass,'plug_geometry_preserved_m':[ *B.USBC_PLUG,B.USBC_PLUG_LEN],
 'exposed_plug_measured_m':.0058,'plug_length_difference_m':B.USBC_PLUG_LEN-.0058,
 'note':'Original head length retained as requested. Housing split derived to preserve total; 0.85 mm plug exposure difference remains unresolved, not silently treated as measured housing.'},
 'limitations':['photo-derived port coordinates are not exact metrology','silver width/thickness are not outer black housing dimensions','original connector geometry retained; exposed length differs from measurement']}
(out/'dimensions.json').write_text(json.dumps(report,indent=2)+'\n')
print('total reader length',total,'target',report['rear_panel']['target_from_left'],report['rear_panel']['port_height'])
