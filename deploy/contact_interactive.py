"""Operator-paced PC contact commissioning, called only after a checked approach.

No automatic search for contact. Target steps and actual motion are distinct.
"""
import select
import sys
import time
from pathlib import Path
import numpy as np
import torch
from deploy import conventions as C
from deploy.contact_steps import ContactSteps
from deploy.contact_frames import latest_complete_frame
from deploy.contact_tracking import require_attained


def interact(arms,move,guard,camera_health,geo,profile,pc,meta,rows,args,offsets,set_margin):
    import estop,safety
    import cv2
    from deploy.checks.placement import TopCam,find_blob
    robot=C.Robot();base=geo.calibration['arms']['left_base_in_right_base']
    import mujoco
    rotation=np.zeros(9);q=np.array(base['quat']);mujoco.mju_quat2Mat(rotation,q/np.linalg.norm(q));rotation=rotation.reshape(3,3)
    translation=np.array(base['pos'])
    last=arms['left'].safe.last_safe[:6].copy()
    local,orientation=robot.fk(last+offsets['left']);requested=rotation@local.numpy()+translation
    seed=last+offsets['left'];normal=np.array([np.cos(np.deg2rad(pc['yaw_deg_mod90'])),np.sin(np.deg2rad(pc['yaw_deg_mod90'])),0.])
    tangent=np.array([normal[1],-normal[0],0.])
    normal_travel=0.;contact=False;slider=None;reference=np.array(meta['pc_center_xy_m']);last_frame=None
    area_reference=None;last_observation=None
    meta['operator_events']=[];meta['pc_observations']=[]
    def observe_pc():
        nonlocal last_frame,area_reference,last_observation
        try:image,frame=latest_complete_frame(args.session/'cameras'/args.camera_run/'zed')
        except RuntimeError as exc:raise safety.SafetyStop(str(exc)) from exc
        if image==last_frame:return
        hit=find_blob(frame,TopCam(),reference,profile['pc']['top_z_model_estimate_m'],.12,.1025*.097)
        if hit is None:raise safety.SafetyStop('PC tracking lost; operator assessment required')
        xy,yaw,contour=hit;area=cv2.contourArea(contour)
        if area_reference is None:area_reference=area
        if not .85*area_reference<=area<=1.15*area_reference:
            raise safety.SafetyStop('PC appearance/occlusion changed; tracking uncertain')
        displacement=float(np.linalg.norm(xy-reference))
        last_observation={'image':str(image),'host_processing_ns':time.monotonic_ns(),'xy_m':xy.tolist(),
                          'delta_xy_m':(xy-reference).tolist(),'displacement_m':displacement,'yaw_deg_mod90':float(yaw)}
        meta['pc_observations'].append(last_observation);last_frame=image
        if displacement>.006:raise safety.SafetyStop('PC displacement exceeds initial 6-mm commissioning bound')
        geo.m.geom_pos[geo.pc,:2]=xy
        angle=np.deg2rad(yaw)/2;geo.m.geom_quat[geo.pc]=[np.cos(angle),0,0,np.sin(angle)]
    def actual_height():
        measured=arms['left'].read()[0][:6]+offsets['left']
        return float((rotation@robot.fk(measured)[0].numpy()+translation)[2])
    def command(xyz,phase):
        nonlocal seed,requested
        if phase!='disengage':
            measured=arms['left'].read()[0][:6]+offsets['left']
            actual=rotation@robot.fk(measured)[0].numpy()+translation
            try:require_attained(actual,requested)
            except ValueError as exc:
                print('No jog executed:',str(exc),'Use retreat; controller characterization required.',flush=True)
                return False
        pos=rotation.T@(xyz-translation)
        qq,pe,re=robot.chain.ik(torch.tensor(pos[None],dtype=torch.float32),orientation[None],
             torch.tensor(seed[None],dtype=torch.float32),iters=300,q_rest=robot.home[None])
        if pe.item()>.001 or re.item()>.03:raise safety.SafetyStop('jog IK failed')
        observe_pc()
        try:move(qq[0].numpy()-offsets['left'],phase)
        except ValueError as exc:
            print('No jog executed:',str(exc),flush=True);return False
        seed=qq[0].numpy();requested=xyz.copy()
        return True
    print('At free-space standoff. Commands: lower (1 mm), back (0.5 mm), contact, slide (0.5 mm), retreat.',flush=True)
    print('Only request lower/back while visually watching the gap. Space in rig console stops.',flush=True)
    set_margin(0.)
    deadline=time.monotonic()+60;next_pc=0.;next_log=0.
    while True:
        if estop.stopped():raise safety.SafetyStop('software stop during contact commissioning')
        if time.monotonic()>deadline:raise safety.SafetyStop('operator command timeout')
        obs={side:arm.read() for side,arm in arms.items()}
        for side,v in obs.items():
            age=time.monotonic()-v[2]
            if not np.isfinite(age) or not 0<=age<=.1:raise safety.SafetyStop('stale contact feedback: '+side)
        distances=guard({side:v[0][:6] for side,v in obs.items()})
        if time.monotonic()>next_pc:
            camera_health();observe_pc();next_pc=time.monotonic()+.5
        if time.monotonic()>next_log:
            rows.append({'host_monotonic_ns':time.monotonic_ns(),'phase':'contact_wait' if contact else 'precontact_wait',
                'pc_observation':last_observation,'geometry_distances':distances,'observations':{side:{'q_encoder':v[0].tolist(),'qd_encoder':v[1].tolist(),
                    'observation_time':v[2],'time_exact':v[3]} for side,v in obs.items()},'left_first_can_send_host_s':None})
            next_log=time.monotonic()+.02
        ready,_,_=select.select([sys.stdin],[],[],.02)
        if not ready:continue
        line=sys.stdin.readline()
        if not line:raise safety.SafetyStop('operator terminal closed')
        action=line.strip().lower();deadline=time.monotonic()+60
        meta['operator_events'].append({'host_monotonic_ns':time.monotonic_ns(),'command':action})
        if action=='lower':
            if contact or requested[2]-.001<.0119:
                print('Lower refused: contact already marked or height bound reached.',flush=True);continue
            xyz=requested.copy();xyz[2]-=.001;command(xyz,'lower_clear')
        elif action=='back':
            if contact or actual_height()>profile['pc']['top_z_model_estimate_m']+.004 or normal_travel+.0005>.025:
                print('Back refused: lower to contact height first, or normal travel limit/contact reached.',flush=True);continue
            meta['contact_attempted']=True
            if command(requested-normal*.0005,'normal_contact_search'):normal_travel+=.0005
        elif action=='contact':
            if contact or actual_height()>profile['pc']['top_z_model_estimate_m']+.004:
                print('Contact mark refused: wrong phase/height.',flush=True);continue
            contact=True;meta['contact_commanded']=True;set_margin(-.001)
            slider=ContactSteps(tuple(requested),tuple(tangent));slider.arm(True,True)
            print('Operator contact recorded. Slide is one-way, maximum 5 mm total.',flush=True)
        elif action=='slide':
            if not contact:print('Slide refused: first confirm actual contact.',flush=True);continue
            target=slider.propose(operator_next=True)
            if target is None:print('Slide travel limit reached; disengage.',flush=True);continue
            if command(np.array(target),'one_way_slide'):slider.executed(target)
        elif action=='retreat':
            # First separate normal to rear face; never reset by dragging back laterally.
            if not command(requested+normal*.015,'disengage'):continue
            if slider is not None:slider.disengaged(True)
            print('Normal disengagement executed; returning via high waypoint.',flush=True)
            return
        else:
            print('No motion: expected lower/back/contact/slide/retreat.',flush=True);continue
        actual=arms['left'].read()[0][:6]+offsets['left'];tcp=rotation@robot.fk(actual)[0].numpy()+translation
        print('requested TCP mm',np.round(requested*1000,1).tolist(),'measured-model TCP mm',np.round(tcp*1000,1).tolist(),flush=True)
