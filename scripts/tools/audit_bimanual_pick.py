"""Nominal CPU pick rollout using audited IK waypoints, without attaching the reader."""
import argparse,csv,json,hashlib
from pathlib import Path
import mujoco,numpy as np
from blind_insertion.scenes import pc_bench as S,physical_rig as R
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--insert',action='store_true');parser.add_argument('--diagnostic-impratio',type=float,default=None);parser.add_argument('--diagnostic-noslip',action='store_true');parser.add_argument('--passive-hold',action='store_true');parser.add_argument('--legacy-grasp',action='store_true');parser.add_argument('--inhand-align',action='store_true');parser.add_argument('--feedback-align',action='store_true');parser.add_argument('--settle-grasp',action='store_true');parser.add_argument('--clearance-carry',action='store_true');parser.add_argument('--identified-dynamics',action='store_true');parser.add_argument('--alignment-gain',type=float,default=.08);parser.add_argument('--align-seconds',type=float,default=6.);args=parser.parse_args()
if not 0<args.alignment_gain<=1 or not 0<args.align_seconds<=30:raise ValueError('invalid alignment controller parameters')
if args.insert and not args.clearance_carry:raise ValueError('insertion requires clearance carry')
if args.passive_hold and (args.feedback_align or args.inhand_align or args.clearance_carry):raise ValueError('passive hold cannot use alignment control')
args.feedback_align=args.feedback_align or args.clearance_carry
args.inhand_align=args.inhand_align or args.feedback_align
waypoint_prefix='legacy_' if args.legacy_grasp else ''
output_prefix=('insert_' if args.insert else '')+(f'impratio{args.diagnostic_impratio:g}_' if args.diagnostic_impratio is not None else '')+('noslip_' if args.diagnostic_noslip else '')+('passive_' if args.passive_hold else '')+('settled_' if args.settle_grasp else '')+('clearance_' if args.clearance_carry else '')+('identified_' if args.identified_dynamics else '')+('feedback_' if args.feedback_align else 'inhand_' if args.inhand_align else '')+waypoint_prefix
profile=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
points=json.loads((Path('results/usbc/bimanual_20261029/reach')/(waypoint_prefix+'nominal_waypoints.json')).read_text())['waypoints']
m,l=S.build_scene(detailed=False,bimanual=True,support_mode='free',profile=profile,measured_support=True,controller_profile='native_free_settle_20261028')
if args.diagnostic_impratio is not None:
    if not 1<=args.diagnostic_impratio<=1000:raise ValueError('impratio outside diagnostic range')
    m.opt.impratio=args.diagnostic_impratio
if args.diagnostic_noslip:m.opt.noslip_iterations=20
m.dof_frictionloss[m.joint('joint2').dofadr]=1.
dyn=None
if args.identified_dynamics:
    dyn_path=Path('results/usbc/bimanual_20261029/dynamics_id/profile.json')
    dyn=json.loads(dyn_path.read_text())
    driver=mujoco.MjModel.from_xml_path('results/usbc/bimanual_20261029/driver_model/driver_gravity.xml')
    driver.jnt_limited[:]=0
    dd=mujoco.MjData(driver)
    indices={}
    for side,pfx in (('right',''),('left','left_')):
        qidx=[int(m.joint(pfx+f'joint{i}').qposadr[0]) for i in range(1,7)]
        vidx=[int(m.joint(pfx+f'joint{i}').dofadr[0]) for i in range(1,7)]
        indices[side]=(qidx,vidx)
        m.dof_frictionloss[vidx]=dyn['arms'][side]['coulomb_nm']
        m.dof_solref[vidx]=dyn['friction_solref'];m.dof_solimp[vidx]=dyn['friction_solimp']
    calibration=R.load()
d=mujoco.MjData(m);R.reference_pose(m,d,l);S.place_pc(m,d);S.place_reader(m,d,l)
for prefix in ('','left_'):
    for i in range(1,7):d.ctrl[m.actuator(prefix+f'joint{i}').id]=d.qpos[m.joint(prefix+f'joint{i}').qposadr][0]
    for side in ('left','right'):d.ctrl[m.actuator(prefix+'grip_'+side).id]=4.
qa=[int(m.joint(f'joint{i}').qposadr[0]) for i in range(1,7)]
acts=[m.actuator(f'joint{i}').id for i in range(1,7)]
grips=[m.actuator('grip_'+s).id for s in ('left','right')]
mujoco.mj_forward(m,d);initial=d.qpos[l.peg_qadr:l.peg_qadr+3].copy()
rows=[];phases={};states={};contact_trace=[];first_loss=None
seating_trace=[]
sequence=[('approach',2.,False),('grasp',2.,False),('close',1.5,True),('lift',2.,True),('preinsert',2.,True),('hold',1.,True)]
if args.passive_hold:sequence[-2:]=[('preinsert',20.,True),('hold',1.,True)]
if args.feedback_align:sequence.insert(-1,('align',args.align_seconds,True))
if args.clearance_carry:
    index=next(i for i,x in enumerate(sequence) if x[0]=='preinsert')
    sequence[index:index+1]=[('carry_high',3.,True),('settle_high',2.,True),('preinsert',6.,True)]
if args.settle_grasp:sequence.insert(2,('settle_grasp',3.,False))
if args.insert:sequence.extend([('insert',8.,True),('seat_hold',1.,True)])
last=d.qpos[qa].copy()
for name,duration,closed in sequence:
    target=np.array(points[{'close':'grasp','settle_grasp':'grasp','hold':'preinsert','align':'preinsert','carry_high':'preinsert','settle_high':'preinsert','insert':'preinsert','seat_hold':'preinsert'}.get(name,name)]['q_model_rad'])
    if args.inhand_align and (name=='carry_high' or (name=='preinsert' and not args.clearance_carry)):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        mujoco.mj_forward(m,d)
        def mat(q):
            r=np.zeros(9);mujoco.mju_quat2Mat(r,q);return r.reshape(3,3)
        tcp_R=d.site_xmat[l.tcp_site].reshape(3,3).copy();tcp_p=d.site_xpos[l.tcp_site].copy()
        reader_R=mat(d.qpos[l.peg_qadr+3:l.peg_qadr+7])
        reader_p=d.qpos[l.peg_qadr:l.peg_qadr+3].copy()
        relative_R=tcp_R.T@reader_R;relative_p=tcp_R.T@(reader_p-tcp_p)
        fa=int(m.joint('pc_free_joint').qposadr[0]);port_R=mat(d.qpos[fa+3:fa+7])
        target_R=port_R@relative_R.T
        target_p=d.qpos[fa:fa+3]+port_R@np.array([0,0,.006])-target_R@relative_p
        if name=='carry_high':target_p[2]+=.04
        scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos
        def residual(q):
            scratch.qpos[qa]=q;mujoco.mj_forward(m,scratch)
            return np.r_[scratch.site_xpos[l.tcp_site]-target_p,.1*Rotation.from_matrix(scratch.site_xmat[l.tcp_site].reshape(3,3)@target_R.T).as_rotvec()]
        limits=np.array([m.joint(f'joint{i}').range for i in range(1,7)])
        fit=least_squares(residual,d.qpos[qa],bounds=(limits[:,0],limits[:,1]),max_nfev=250)
        if np.linalg.norm(fit.fun)>1e-4:raise RuntimeError('in-hand alignment IK failed')
        target=fit.x
    elif args.inhand_align and name in ('hold','align','settle_high','preinsert','insert','seat_hold'):
        target=last.copy()
    if args.settle_grasp and name=='close':target=last.copy()
    if args.passive_hold and name in ('preinsert','hold'):target=last.copy()
    feedback_target=last.copy()
    descent_start=d.qpos[l.peg_qadr:l.peg_qadr+3].copy()
    feedback_phase=args.feedback_align and (name in ('align','hold','insert','seat_hold') or (args.clearance_carry and name in ('settle_high','preinsert')))
    n=int(round(duration/m.opt.timestep))
    for k in range(n):
        t=(k+1)/n;blend=t*t*(3-2*t)
        d.ctrl[acts]=last+(target-last)*blend;d.ctrl[grips]=-12. if closed else 4.
        if name=='settle_grasp':
            if k%20==0:
                feedback_target+=np.clip(.08*(target-d.qpos[qa]),-.004,.004)
                feedback_target=np.clip(feedback_target,d.qpos[qa]-.1,d.qpos[qa]+.1)
                limits=np.array([m.joint(f'joint{i}').range for i in range(1,7)])
                feedback_target=np.clip(feedback_target,limits[:,0],limits[:,1])
            d.ctrl[acts]=feedback_target
        if feedback_phase:
            if k%20==0:
                mujoco.mj_forward(m,d)
                fa=int(m.joint('pc_free_joint').qposadr[0])
                port_R=mat(d.qpos[fa+3:fa+7]);reader_R=mat(d.qpos[l.peg_qadr+3:l.peg_qadr+7])
                tip=d.qpos[l.peg_qadr:l.peg_qadr+3].copy()
                desired=d.qpos[fa:fa+3]+port_R@np.array([0,0,.006])
                if name in ('insert','seat_hold'):
                    gap=.006+(-float(l.seating_depth[0])-.006)*blend if name=='insert' else -float(l.seating_depth[0])
                    desired=d.qpos[fa:fa+3]+port_R@np.array([0,0,gap])
                if name=='settle_high':desired[2]+=.04
                if name=='preinsert':desired=descent_start+(desired-descent_start)*blend
                error=np.r_[desired-tip,.1*Rotation.from_matrix(port_R@reader_R.T).as_rotvec()]
                jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv))
                mujoco.mj_jac(m,d,jp,jr,tip,int(m.site_bodyid[l.tcp_site]))
                dofs=[int(m.joint(f'joint{i}').dofadr[0]) for i in range(1,7)]
                J=np.vstack([jp[:,dofs],.1*jr[:,dofs]])
                delta=J.T@np.linalg.solve(J@J.T+np.eye(6)*1e-5,error)
                feedback_target+=np.clip(args.alignment_gain*delta,-.004,.004)
                feedback_target=np.clip(feedback_target,d.qpos[qa]-.1,d.qpos[qa]+.1)
                feedback_target=np.clip(feedback_target,limits[:,0],limits[:,1])
            d.ctrl[acts]=feedback_target
        if dyn is not None and k%20==0:
            for side,(qidx,vidx) in indices.items():
                params=dyn['arms'][side]
                dd.qpos[:6]=d.qpos[qidx]-np.asarray(calibration['arms'][side]['joint_offsets_rad'])
                mujoco.mj_kinematics(driver,dd);mujoco.mj_comPos(driver,dd)
                mujoco.mj_rne(driver,dd,0,dd.qfrc_bias)
                d.qfrc_applied[vidx]=np.asarray(params['driver_gravity_comp_factor'])*dd.qfrc_bias[:6]-np.asarray(params['gravity_scale'])*d.qfrc_gravcomp[vidx]
        mujoco.mj_step(m,d)
        if not np.isfinite(d.qpos).all():raise RuntimeError('nonfinite rollout')
        if k%20==0:
            mujoco.mj_forward(m,d)
            pr=d.qpos[l.peg_qadr:l.peg_qadr+3];qr=d.qpos[l.peg_qadr+3:l.peg_qadr+7]
            rot=np.zeros(9);mujoco.mju_quat2Mat(rot,qr)
            center=pr+rot.reshape(3,3)@np.array([0,0,l.grasp_z[0]])
            error=np.linalg.norm(center-d.site_xpos[l.tcp_site])
            rows.append([d.time,name,*pr,error,float(np.max(abs(d.qpos[qa]-d.ctrl[acts])))])
            if args.insert:
                port=m.body('fixture').id;port_R=d.xmat[port].reshape(3,3)
                rel_R=port_R.T@rot.reshape(3,3)
                rel_p=port_R.T@(pr-d.xpos[port])
                edge_points=np.column_stack([l.edge_points[0],np.zeros(len(l.edge_points[0]))])
                pts=edge_points@rel_R.T+rel_p
                polygon=np.asarray(l.hole_polys[0]);edges=np.roll(polygon,-1,axis=0)-polygon
                delta=pts[:,None,:2]-polygon[None]
                cross=edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]
                inside=bool((cross>=-1e-9).all())
                depth=float(l.seating_depth[0])
                tilt_deg=float(np.rad2deg(np.arccos(np.clip(rel_R[2,2],-1,1))))
                seated=inside and bool((abs(pts[:,2]+depth)<=.00025).all()) and tilt_deg<=2.
                seating_trace.append({'time_s':float(d.time),'phase':name,'geometric_seated':seated,'tip_depth_m':float(-rel_p[2])})
            reader_contacts=[]
            for ci,c in enumerate(d.contact):
                if c.geom1 not in l.peg_slots and c.geom2 not in l.peg_slots:continue
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                reader_contacts.append({'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'normal_N':float(force[0]),'distance_m':float(c.dist)})
            contact_trace.append({'time_s':float(d.time),'phase':name,'center_to_tcp_m':float(error),'reader_contacts':reader_contacts})
            if name not in ('approach','grasp','settle_grasp','close') and error>.02 and first_loss is None:
                first_loss={'time_s':float(d.time),'phase':name,'recent_contacts':contact_trace[-16:]}
                states['first_loss']=d.qpos.copy()
    last=feedback_target if feedback_phase or name=='settle_grasp' else target
    states[name]=d.qpos.copy()
    contacts=[{'a':m.geom(c.geom1).name,'b':m.geom(c.geom2).name,'distance_m':float(c.dist)} for c in d.contact if any(m.geom(g).name.startswith(('left_pad_s','right_pad_s')) or m.geom(g).name in ('left_tip_hull','right_tip_hull') for g in (c.geom1,c.geom2))]
    mujoco.mj_forward(m,d)
    port=m.body('fixture').id
    port_R=d.xmat[port].reshape(3,3);reader_R=np.zeros(9)
    mujoco.mju_quat2Mat(reader_R,d.qpos[l.peg_qadr+3:l.peg_qadr+7])
    tip_in_port=port_R.T@(d.qpos[l.peg_qadr:l.peg_qadr+3]-d.xpos[port])
    tilt=float(np.rad2deg(np.arccos(np.clip((port_R.T@reader_R.reshape(3,3))[2,2],-1,1))))
    phase_contacts=[c for entry in contact_trace if entry['phase']==name for c in entry['reader_contacts']]
    phases[name]={'pc_position_m':d.xpos[port].tolist(),'min_reader_contact_distance_m':min((c['distance_m'] for c in phase_contacts),default=0.),'peak_reader_contact_normal_N':max((c['normal_N'] for c in phase_contacts),default=0.),'tip_in_port_m':tip_in_port.tolist(),'tilt_deg':tilt,'finger_q_m':[float(d.qpos[m.joint(s+'_finger').qposadr][0]) for s in ('left','right')],'gripper_contacts':contacts,'reader_tip_xyz_m':d.qpos[l.peg_qadr:l.peg_qadr+3].tolist(),
                  'reader_center_to_tcp_m':float(rows[-1][-2]),'max_joint_error_rad':float(rows[-1][-1])}
out=Path('results/usbc/bimanual_20261029/reach')
np.savez_compressed(out/(output_prefix+'nominal_pick_states.npz'),**states)
with (out/(output_prefix+'nominal_pick.csv')).open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['time_s','phase','reader_x','reader_y','reader_z','center_to_tcp_m','max_joint_error_rad']);w.writerows(rows)
lift=phases['lift'];held=lift['reader_tip_xyz_m'][2]>initial[2]+.04 and lift['reader_center_to_tcp_m']<.02
longest_seated_s=0.;run_start=None
for sample in seating_trace:
    if sample['geometric_seated']:
        if run_start is None:run_start=sample['time_s']
        longest_seated_s=max(longest_seated_s,sample['time_s']-run_start)
    else:run_start=None
report={'longest_sampled_seated_hold_s':longest_seated_s,'geometric_seating_hold_pass':bool(args.insert and longest_seated_s>=.2 and first_loss is None),'scope':'CPU scripted pick in nominal simulator; no hardware validation or bimanual bracing',
        'insertion_attempt':args.insert,'diagnostic_impratio':float(m.opt.impratio),'diagnostic_noslip_iterations':int(m.opt.noslip_iterations),'passive_hold_after_lift':args.passive_hold,'reader_attached_artificially':False,'first_grasp_loss':first_loss,'clearance_carry':args.clearance_carry,'settle_grasp':args.settle_grasp,'retained_after_carry':all(phases[n]['reader_center_to_tcp_m']<.02 for n in ('preinsert','hold')),'dynamics_profile_sha256':hashlib.sha256(dyn_path.read_bytes()).hexdigest() if dyn else None,'privileged_inhand_alignment':args.inhand_align,'privileged_feedback_alignment':args.feedback_align,'alignment_gain':args.alignment_gain,'align_seconds':args.align_seconds,'lifted_and_near_tcp':bool(held),'phases':phases,
        'limitations':['scripted joint interpolation is not collision-aware planning','default approximate pad friction and squeeze','nominal identified residual model without randomization' if dyn else 'ideal gravity except right joint2 friction','single fixed reset']}
if args.insert:(out/(output_prefix+'seating_trace.json')).write_text(json.dumps(seating_trace,indent=2)+'\n')
(out/(output_prefix+'nominal_pick.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
