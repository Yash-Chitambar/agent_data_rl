"""Supervised no-contact approach commissioning. No contact/slide commands.

Cameras must already be recording. The dedicated geometry guard does not alter
calibration or policy safety defaults. Failure holds power for supported shutdown.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from deploy import conventions as C
from deploy.backends import real_loop
from deploy.contact_geometry import ContactGeometry
from deploy.safety_task import FlexChecker
from deploy.recovery import supported_shutdown


class Pair:
    def __init__(self, arms):self.arms=arms
    def hold(self):
        for arm in self.arms.values():arm.hold()
    def close(self):
        for arm in self.arms.values():arm.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--ready-file',type=Path,required=True)
    p.add_argument('--camera-run',required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if not a.run:raise SystemExit('No hardware opened; --run required for supervised approach')
    import safety,estop,calib
    if estop.stopped():raise SystemExit('software stop is set')
    plan=json.loads(a.plan.read_text())
    if not plan.get('offline_geometric_checks_passed'):raise SystemExit('offline plan failed')
    ready=json.loads(a.ready_file.read_text())
    if not 0<=time.monotonic_ns()-ready['host_monotonic_ns']<30e9:
        raise SystemExit('camera readiness marker is stale')
    out=a.session/'contact'/a.name
    out.mkdir(parents=True,exist_ok=False)
    profile=json.loads((a.session/'support_geometry_profile.json').read_text())
    pc=json.loads((a.session/'standoff_pc_check.json').read_text())
    import cv2
    from deploy.checks.placement import TopCam,find_blob
    folder=a.session/'cameras'/a.camera_run/'zed'
    images=sorted(folder.glob('raw_*.png'))
    if not images:raise SystemExit('no current saved overhead frame; no arms connected')
    source_image=images[-1]
    hit=find_blob(cv2.imread(str(source_image)),TopCam(),np.array(pc['pc_center_xy_m']),
                  profile['pc']['top_z_model_estimate_m'],.12,.1025*.097)
    if hit is None:raise SystemExit('PC not localized; no arms connected')
    actual_center,yaw,_=hit
    if np.linalg.norm(actual_center-np.array(pc['pc_center_xy_m']))>.01:
        raise SystemExit('PC moved >10 mm; replan before connecting')
    geo=ContactGeometry(C.rig.SPEC,actual_center,yaw,profile['table_plane_z_m'])
    offsets={side:C.joint_offsets(side) for side in ('left','right')}
    check=FlexChecker(offsets['left'],clearance=.04,tip_clearance=.002,
        body_clearance={'gripper':.015,'wrist_camera':.03,'link_6':.02},table_z=0.)
    lateral=calib.Mover.__new__(calib.Mover);lateral.side='left';lateral.check=calib.calib_checker(calib.PATH_CLEARANCE)
    D=real_loop();D.FRIC_COMP=None
    arms={};rest={};rows=[];result='incomplete';ok=False
    meta={'camera_run':a.camera_run,'pc_source_image':str(source_image),
          'pc_center_xy_m':actual_center.tolist(),'pc_yaw_deg_mod90':float(yaw),
          'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'calibration_sha256':hashlib.sha256(C.rig.SPEC.read_bytes()).hexdigest(),
          'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'contact_commanded':False,
          'controller':'native gain profile; no integral or explicit feedforward',
          'geometry_floor_z_m':0.,'table_z_measured_m':profile['table_plane_z_m']}
    def observations():
        return {side:arm.read() for side,arm in arms.items()}
    def guard(poses):
        why=check.violation(poses['left'])
        if why:raise safety.SafetyStop(why)
        if not lateral.lateral_ok(poses['left']):raise safety.SafetyStop('native lateral/reach guard')
        geo.set_encoders(poses,{'left':.006,'right':.048})
        distances=geo.distances()
        if distances['minimum_interarm_distance_m']<.05:raise safety.SafetyStop('interarm gap <50 mm')
        if distances['minimum_pc_distance_m']['left']<.015:raise safety.SafetyStop('precontact PC gap <15 mm')
        return distances
    def camera_health():
        # Acquisition and disk timestamps are not exposure timestamps. This only
        # rejects a missing/stalled capture; it is not a visual safety sensor.
        folder=a.session/'cameras'/a.camera_run
        if (folder/'capture.json').exists():raise safety.SafetyStop('camera capture already ended')
        for cam in ('zed','wrist_left','wrist_right'):
            images=list((folder/cam).glob('raw_*.png'))
            if not images or time.time()-max(v.stat().st_mtime for v in images)>2:
                raise safety.SafetyStop('camera output stalled: '+cam)
    def move(goal,phase):
        current=observations();q0=current['left'][0][:6].copy();park=current['right'][0][:6].copy()
        for alpha in np.linspace(0,1,201):guard({'left':q0*(1-alpha)+goal*alpha,'right':park})
        count=max(1,int(np.ceil(np.max(np.abs(goal-q0))/.05/.02*1.5)))
        deadline=time.monotonic();last_stamp={};last_change={side:deadline for side in arms}
        for k in range(count+50):
            if estop.stopped():raise safety.SafetyStop('software stop')
            if k % 5 == 0:camera_health()
            obs=observations();poses={s:v[0][:6].copy() for s,v in obs.items()}
            distances=guard(poses)
            for side,v in obs.items():
                stamp=v[2]
                if stamp!=last_stamp.get(side):last_change[side]=time.monotonic();last_stamp[side]=stamp
                if time.monotonic()-last_change[side]>.1:raise safety.SafetyStop('feedback stale: '+side)
            alpha=min(1.,(k+1)/count);blend=alpha*alpha*(3-2*alpha)
            target=q0*(1-blend)+goal*blend
            guard({'left':target,'right':poses['right']})
            if np.any(np.abs(target-poses['left'])>np.array([.15,.15,.15,.2,.3,.3])):
                raise safety.SafetyStop('native tracking guard')
            arms['left'].send(np.r_[target,.02],len(rows))
            rows.append({'host_monotonic_ns':time.monotonic_ns(),'phase':phase,
                'requested_left_encoder':target.tolist(),'executed_left_encoder':arms['left'].safe.last_safe.tolist(),
                'observations':{s:{'q_encoder':v[0].tolist(),'qd_encoder':v[1].tolist(),
                    'observation_time':v[2],'time_exact':v[3]} for s,v in obs.items()},
                'geometry_distances':distances})
            deadline+=.02;time.sleep(max(0,deadline-time.monotonic()))
    try:
        for side in ('right','left'):
            arms[side]=D.RealArm(side,np.array([80,80,80,40,10,10.]),np.array([5,5,5,1.5,1.5,1.5]))
            q=arms[side].read()[0];rest[side]=q[:6].copy()
            if np.max(np.abs(q[1:3]))>.05:raise safety.SafetyStop(side+' not at folded rest')
        meta['controllers']={s:v.info() for s,v in arms.items()}
        arms['left'].safe=safety.SafeArm(arms['left'].robot,check)
        q=arms['left'].read()[0];q[6]=.02;arms['left'].safe.command(q)
        time.sleep(1.5)
        if arms['left'].read()[0][6]>.12:raise safety.SafetyStop('empty gripper did not close')
        for step in plan['phases']:move(np.array(step['target_q_encoder']),step['phase'])
        print('No-contact descent complete; withdrawing via high waypoint.',flush=True)
        move(np.array(plan['phases'][0]['target_q_encoder']),'retreat_high')
        move(rest['left'],'return_rest')
        result='completed';ok=True
    except BaseException as exc:
        result=repr(exc);estop.STOP_FILE.write_text('stop: contact approach commissioning failed\n')
        print(result,flush=True)
    finally:
        if arms:Pair(arms).hold()
        meta.update(result=result,rows=rows)
        (out/'trial.json').write_text(json.dumps(meta,indent=1)+'\n')
        for side,arm in arms.items():
            (out/(side+'_native_feedback.json')).write_text(json.dumps(list(arm.timing.enc),default=lambda v:np.asarray(v).tolist())+'\n')
        if arms:
            if ok:Pair(arms).close()
            else:supported_shutdown(Pair(arms))
    print(result,out,flush=True)

if __name__=='__main__':main()
