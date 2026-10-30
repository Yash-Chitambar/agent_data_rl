"""Nominal CPU pick rollout using audited IK waypoints, without attaching the reader."""
import argparse,csv,json
from pathlib import Path
import mujoco,numpy as np
from blind_insertion.scenes import pc_bench as S,physical_rig as R
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--legacy-grasp',action='store_true');args=parser.parse_args()
output_prefix='legacy_' if args.legacy_grasp else ''
profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
points=json.loads((Path('results/usbc/bimanual_20261029/reach')/(output_prefix+'nominal_waypoints.json')).read_text())['waypoints']
m,l=S.build_scene(detailed=False,bimanual=True,support_mode='free',profile=profile,measured_support=True,controller_profile='native_free_settle_20261028')
m.dof_frictionloss[m.joint('joint2').dofadr]=1.
d=mujoco.MjData(m);R.reference_pose(m,d,l);S.place_pc(m,d);S.place_reader(m,d,l)
for prefix in ('','left_'):
    for i in range(1,7):d.ctrl[m.actuator(prefix+f'joint{i}').id]=d.qpos[m.joint(prefix+f'joint{i}').qposadr][0]
    for side in ('left','right'):d.ctrl[m.actuator(prefix+'grip_'+side).id]=4.
qa=[int(m.joint(f'joint{i}').qposadr[0]) for i in range(1,7)]
acts=[m.actuator(f'joint{i}').id for i in range(1,7)]
grips=[m.actuator('grip_'+s).id for s in ('left','right')]
mujoco.mj_forward(m,d);initial=d.qpos[l.peg_qadr:l.peg_qadr+3].copy()
rows=[];phases={};states={}
sequence=[('approach',2.,False),('grasp',2.,False),('close',1.5,True),('lift',2.,True),('preinsert',2.,True),('hold',1.,True)]
last=d.qpos[qa].copy()
for name,duration,closed in sequence:
    target=np.array(points[{'close':'grasp','hold':'preinsert'}.get(name,name)]['q_model_rad'])
    n=int(round(duration/m.opt.timestep))
    for k in range(n):
        t=(k+1)/n;blend=t*t*(3-2*t)
        d.ctrl[acts]=last+(target-last)*blend;d.ctrl[grips]=-12. if closed else 4.
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all():raise RuntimeError('nonfinite rollout')
        if k%20==0:
            mujoco.mj_forward(m,d)
            pr=d.qpos[l.peg_qadr:l.peg_qadr+3];qr=d.qpos[l.peg_qadr+3:l.peg_qadr+7]
            rot=np.zeros(9);mujoco.mju_quat2Mat(rot,qr)
            center=pr+rot.reshape(3,3)@np.array([0,0,l.grasp_z[0]])
            error=np.linalg.norm(center-d.site_xpos[l.tcp_site])
            rows.append([d.time,name,*pr,error,float(np.max(abs(d.qpos[qa]-d.ctrl[acts])))])
    last=target
    states[name]=d.qpos.copy()
    contacts=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'distance_m':float(c.dist)} for c in d.contact if any(m.geom(g).name.startswith(('left_pad_s','right_pad_s')) or m.geom(g).name in ('left_tip_hull','right_tip_hull') for g in (c.geom1,c.geom2))]
    phases[name]={'finger_q_m':[float(d.qpos[m.joint(s+'_finger').qposadr][0]) for s in ('left','right')],'gripper_contacts':contacts,'reader_tip_xyz_m':d.qpos[l.peg_qadr:l.peg_qadr+3].tolist(),
                  'reader_center_to_tcp_m':float(rows[-1][-2]),'max_joint_error_rad':float(rows[-1][-1])}
out=Path('results/usbc/bimanual_20261029/reach')
np.savez_compressed(out/(output_prefix+'nominal_pick_states.npz'),**states)
with (out/(output_prefix+'nominal_pick.csv')).open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['time_s','phase','reader_x','reader_y','reader_z','center_to_tcp_m','max_joint_error_rad']);w.writerows(rows)
lift=phases['lift'];held=lift['reader_tip_xyz_m'][2]>initial[2]+.04 and lift['reader_center_to_tcp_m']<.02
report={'scope':'CPU scripted pick in nominal simulator; no hardware validation or bimanual bracing',
        'reader_attached_artificially':False,'lifted_and_near_tcp':bool(held),'phases':phases,
        'limitations':['scripted joint interpolation is not collision-aware planning','default approximate pad friction and squeeze','ideal gravity except right joint2 friction','single fixed reset']}
(out/(output_prefix+'nominal_pick.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
