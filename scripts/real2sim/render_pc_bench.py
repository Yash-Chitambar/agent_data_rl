"""Render photo-referenced task H scene and verify unchanged physical model."""
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from blind_insertion.scenes import pc_bench as S

out=Path('results/H/scene_review'); out.mkdir(parents=True,exist_ok=True)
m,lay=S.build_scene()
d=mujoco.MjData(m)
d.qpos[lay.arm_qadr]=S.LEFT_HOME  # folded review pose; model reset/home is unchanged
S.place_pc(m,d); S.place_reader(m,d,lay)
mujoco.mj_forward(m,d)
m.vis.global_.offwidth=1600; m.vis.global_.offheight=1200
r=mujoco.Renderer(m,height=1200,width=1600)
views=[('workcell',(.33,.31,.15),1.50,0,-40),
       ('task',(.40,.015,.025),.48,40,-48),
       ('rear_panel',(.48,-.02,.025),.26,25,-25),
       ('reader',(.28,0,.006),.18,25,-38)]
for name,target,dist,az,el in views:
    cam=mujoco.MjvCamera(); cam.lookat[:]=target;cam.distance=dist;cam.azimuth=az;cam.elevation=el
    r.update_scene(d,camera=cam);Image.fromarray(r.render()).save(out/(name+'.png'))
r.close()
base,bl=S.build_scene(detailed=False)
# Compare actual robot, part, fixture mass/inertia and all collision geoms by name.
checks={}
for name in ['peg','fixture','arm']:
    a=m.body(name);b=base.body(name)
    checks[name+'_mass']=bool(np.allclose(a.mass,b.mass))
    checks[name+'_inertia']=bool(np.allclose(a.inertia,b.inertia))
for field in ['geom_pos','geom_size','geom_contype','geom_conaffinity','geom_friction','geom_solref']:
    checks[field]=all(np.allclose(getattr(m,field)[m.geom(base.geom(i).name).id],getattr(base,field)[i])
                      for i in range(base.ngeom) if base.geom_contype[i] or base.geom_conaffinity[i])
checks['arm_base_pose']=bool(np.allclose(m.body('arm').pos,base.body('arm').pos))
checks['left_base_pose']=bool(np.allclose(m.body('left_arm').pos,base.body('left_arm').pos))
checks['all_passed']=all(checks.values())
(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
assert checks['all_passed']
