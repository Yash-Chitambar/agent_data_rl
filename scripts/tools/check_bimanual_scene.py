"""CPU construction/regression checks; not a hardware-fidelity certificate."""
import json
from pathlib import Path
import mujoco
import numpy as np
from blind_insertion.scenes import pc_bench as S, physical_rig as R

profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
z=json.loads(Path('results/real/left_session_20261028/corrected_table_probe.json').read_text())['table_z_model_m_median']
m,l=S.build_scene(detailed=False,support_mode='free',profile=profile,bimanual=True,table_top=z)
d=mujoco.MjData(m)
R.reference_pose(m,d,l)
S.place_reader(m,d,l,table_top=z)
S.place_pc(m,d,table_top=z)
assert m.nu==16
assert np.isclose(m.body('fixture').mass[0],.284)
assert m.body('fixture').mocapid[0]==-1
assert m.joint('pc_free_joint').bodyid[0]==m.body('fixture').id
assert np.isclose(m.geom('table').pos[2]+m.geom('table').size[2],z)
for prefix in ('','left_'):
    for i in range(1,7):
        a=m.actuator(prefix+f'joint{i}').id
        d.ctrl[a]=d.qpos[m.joint(prefix+f'joint{i}').qposadr][0]
        assert m.actuator_trnid[a,0]==m.joint(prefix+f'joint{i}').id
        if i<4:assert m.actuator_biasprm[a,2]==-5
    pads=[g for g in range(m.ngeom) if m.geom(g).name.startswith((prefix+'left_pad_s',prefix+'right_pad_s'))]
    assert len(pads)==32
    assert all(m.geom_contype[g]!=0 for g in pads)
mujoco.mj_forward(m,d)
pc_start=d.xpos[m.body('fixture').id].copy()
for _ in range(100):mujoco.mj_step(m,d)
assert np.isfinite(d.qpos).all()
assert np.linalg.norm(d.xpos[m.body('fixture').id]-pc_start)<.01
assert any(m.body(int(m.geom_bodyid[c.geom1])).name=='fixture' or
           m.body(int(m.geom_bodyid[c.geom2])).name=='fixture' for c in d.contact)
legacy,_=S.build_scene(detailed=False)
assert legacy.nu==8
from blind_insertion.scenes.measured_support import dimensions
support=dimensions()
measured,_=S.build_scene(detailed=False,support_mode='free',profile=profile,
                         bimanual=True,measured_support=True)
table=measured.geom('table');plate=measured.geom('shared_mounting_plate')
assert np.isclose(table.pos[0]+table.size[0]-support['base_max_m'][0],.480)
assert np.isclose(plate.pos[2]+plate.size[2]-table.pos[2]-table.size[2],.023)
assert plate.contype[0] and plate.conaffinity[0]
assert np.allclose(measured.body_pos,m.body_pos)
assert np.allclose(measured.body_quat,m.body_quat)
assert np.allclose(measured.cam_pos,m.cam_pos)
assert np.allclose(measured.cam_quat,m.cam_quat)
report={'checks':'construction, table placement, bilateral actuation/pads, short settling, legacy default',
        'table_z_m':z,'pc_mass_kg':.284,'actuators':m.nu,'status':'passed',
        'measured_support':support,
        'not_validated':['hardware dynamics','plate footprint','contact forces','GPU parity','image overlays']}
out=Path('results/usbc/bimanual_20261029');out.mkdir(parents=True,exist_ok=True)
(out/'scene_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
