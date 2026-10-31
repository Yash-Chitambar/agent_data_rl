"""CPU contact probes for restored arm collisions. No hardware access."""
import json
from pathlib import Path
import mujoco
import numpy as np
from blind_insertion.scenes import pc_bench as S,physical_rig as R
from blind_insertion.scenes.bimanual_collision import apply
profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m,l=S.build_scene(detailed=False,bimanual=True,support_mode='free',profile=profile,measured_support=True)
d=mujoco.MjData(m);R.reference_pose(m,d,l);S.place_pc(m,d);S.place_reader(m,d,l);mujoco.mj_forward(m,d)
report=apply(m)
for prefix,own,other in [('',32,64),('left_',64,32)]:
 for finger in ('left','right'):
  gid=m.geom(prefix+finger+'_pad_s00').id
  assert m.geom_contype[gid]&own
  assert m.geom_conaffinity[gid]&other
  assert not m.geom_conaffinity[gid]&own
# Put the free PC COM on an authored forearm capsule, not merely inside its broad-phase box.
cap=report['right'][0]['geom_id'];pc=m.body('fixture').id
j=m.joint('pc_free_joint');qa=int(j.qposadr[0]);rot=d.xmat[pc].reshape(3,3).copy()
d.qpos[qa:qa+3]=d.geom_xpos[cap]-rot@m.body_ipos[pc]
mujoco.mj_forward(m,d)
right_geoms=set(np.flatnonzero((m.geom_contype&32)!=0).tolist())
def pc_arm_contacts():
    return sum(1 for c in d.contact if (m.geom_bodyid[c.geom1]==pc and c.geom2 in right_geoms) or
               (m.geom_bodyid[c.geom2]==pc and c.geom1 in right_geoms))
report['pc_arm_contact_count']=pc_arm_contacts();assert report['pc_arm_contact_count']>0
# Negative control: disable all right-arm shapes and rebuild body masks.
mask=(m.geom_contype&32)!=0
m.geom_contype[mask]=0;m.geom_conaffinity[mask]=0
m.body_contype[:]=0;m.body_conaffinity[:]=0
for g in range(m.ngeom):
 m.body_contype[m.geom_bodyid[g]]|=m.geom_contype[g];m.body_conaffinity[m.geom_bodyid[g]]|=m.geom_conaffinity[g]
mujoco.mj_forward(m,d)
assert pc_arm_contacts()==0
# Separate fresh model: overlap matching arm configurations to probe inter-arm collision.
m,l=S.build_scene(detailed=False,bimanual=True,support_mode='free',profile=profile,measured_support=True)
d=mujoco.MjData(m);R.reference_pose(m,d,l)
m.body_pos[m.body('left_arm').id]=m.body_pos[m.body('arm').id]
m.body_quat[m.body('left_arm').id]=m.body_quat[m.body('arm').id]
for i in range(1,7):d.qpos[m.joint(f'left_joint{i}').qposadr]=d.qpos[m.joint(f'joint{i}').qposadr]
mujoco.mj_forward(m,d)
report['cross_arm_contact_count']=sum(1 for c in d.contact if
 ((m.geom_contype[c.geom1]&32 and m.geom_contype[c.geom2]&64) or
  (m.geom_contype[c.geom1]&64 and m.geom_contype[c.geom2]&32)))
assert report['cross_arm_contact_count']>0
report['status']='contact generation passed; proxies not validated against hardware'
out=Path('results/usbc/bimanual_20261029/collision_checks.json');out.write_text(json.dumps(report,indent=2)+'\n')
print(report)
