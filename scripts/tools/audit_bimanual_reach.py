"""CPU IK/contact audit of nominal pick waypoints; not a successful dynamic rollout."""
import argparse,json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from blind_insertion.scenes import pc_bench as S,physical_rig as R

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--legacy-grasp',action='store_true');args=parser.parse_args()
prefix='legacy_' if args.legacy_grasp else ''
profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m,l=S.build_scene(detailed=False,bimanual=True,support_mode='free',profile=profile,measured_support=True)
d=mujoco.MjData(m);R.reference_pose(m,d,l);S.place_pc(m,d);S.place_reader(m,d,l);mujoco.mj_forward(m,d)
ids=[m.joint(f'joint{i}').id for i in range(1,7)]
qa=m.jnt_qposadr[ids];limits=m.jnt_range[ids];initial=d.qpos[qa].copy()
tcp=l.tcp_site;pa=l.peg_qadr;fa=int(m.joint('pc_free_joint').qposadr[0])
def rotation(q):
    r=np.zeros(9);mujoco.mju_quat2Mat(r,q);return r.reshape(3,3)
reader_R=rotation(d.qpos[pa+3:pa+7]);port_R=rotation(d.qpos[fa+3:fa+7])
offset=np.array([0,.0054,l.grasp_z[0]]) if args.legacy_grasp else S.reader_grasp_offset(m,d,l)
grasp=d.qpos[pa:pa+3]+reader_R@offset
pre=d.qpos[fa:fa+3]+port_R@(offset+[0,0,.006])
rot=reader_R@np.asarray(S.R_TOP_TCP)
waypoints=[('approach',grasp+[0,0,.08],rot),('grasp',grasp,rot),
           ('lift',grasp+[0,0,.08],rot),('preinsert',pre,port_R@np.asarray(S.R_TOP_TCP))]
rng=np.random.default_rng(93);report={'grasp_offset_reader_m':offset.tolist(),'scope':'nominal pick waypoints; IK and static contact only; no grasp or dynamic success','waypoints':{}}
seed=initial
for name,target,target_R in waypoints:
    def residual(q):
        d.qpos[qa]=q;mujoco.mj_forward(m,d)
        return np.r_[d.site_xpos[tcp]-target,.1*Rotation.from_matrix(d.site_xmat[tcp].reshape(3,3)@target_R.T).as_rotvec()]
    best=None
    for start in [seed,initial,*rng.uniform(limits[:,0]+1e-6,limits[:,1]-1e-6,(8,6))]:
        fit=least_squares(residual,np.clip(start,limits[:,0]+1e-7,limits[:,1]-1e-7),bounds=(limits[:,0],limits[:,1]),max_nfev=200)
        if best is None or np.linalg.norm(fit.fun)<np.linalg.norm(best.fun):best=fit
        if np.linalg.norm(best.fun)<1e-5:break
    residual(best.x);seed=best.x
    contacts=[]
    for c in d.contact:
        g1,g2=int(c.geom1),int(c.geom2)
        if c.dist>=-1e-5:continue
        right1=bool(m.geom_contype[g1]&32);right2=bool(m.geom_contype[g2]&32)
        if right1==right2:continue
        other=g2 if right1 else g1
        if other in l.peg_slots:continue
        contacts.append({'geom1':m.geom(g1).name,'geom2':m.geom(g2).name,'penetration_mm':float(-c.dist*1000)})
    report['waypoints'][name]={'target_xyz_m':target.tolist(),'q_model_rad':best.x.tolist(),
       'position_error_mm':float(np.linalg.norm(best.fun[:3])*1000),
       'orientation_error_deg':float(np.rad2deg(np.linalg.norm(best.fun[3:])/.1)),
       'unintended_contacts':contacts}
out=Path('results/usbc/bimanual_20261029/reach');out.mkdir(parents=True,exist_ok=True)
(out/(prefix+'nominal_waypoints.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
