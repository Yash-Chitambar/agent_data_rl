"""Measured duchamp3 transforms and matching square RGB policy views."""
from pathlib import Path
import json
import mujoco
import numpy as np

SPEC = Path(__file__).resolve().parents[1] / 'configs/real_cameras_duchamp3.json'

def load():
    return json.loads(SPEC.read_text())

def camera_pose(pose):
    q=np.asarray(pose['quat'],float);q/=np.linalg.norm(q)
    out=np.zeros(4);mujoco.mju_mulQuat(out,q,np.array([0.,1,0,0]))
    return np.asarray(pose['pos'],float),out

def apply(m):
    s=load()
    left=m.body('left_arm').id
    pose=s['arms']['left_base_in_right_base']
    m.body_pos[left]=pose['pos'];m.body_quat[left]=np.array(pose['quat'])/np.linalg.norm(pose['quat'])
    for side,prefix in [('right',''),('left','left_')]:
        cam=m.camera(prefix+'wrist_cam').id;w=s['wrist'][side]
        m.cam_bodyid[cam]=m.body(prefix+'link_6').id
        m.cam_pos[cam],m.cam_quat[cam]=camera_pose(w['pose_in_link6'])
        crop=w['policy_crop_px'];K=w['intrinsics']
        m.cam_sensorsize[cam]=[crop,crop];m.cam_resolution[cam]=[crop,crop]
        m.cam_intrinsic[cam]=[K['fx'],K['fy'],0,0]
        m.cam_fovy[cam]=w['fovy_deg_for_crop']
    cam=m.camera('top').id;t=s['top'];K=t['intrinsics_rectified']['HD1080']
    m.cam_pos[cam],m.cam_quat[cam]=camera_pose(t['extrinsics_cam_in_right_base'])
    w,h=K['resolution'];cx,cy=round(K['cx']),round(K['cy'])
    crop=2*min(cx,w-cx,cy,h-cy)
    m.cam_sensorsize[cam]=[crop,crop];m.cam_resolution[cam]=[crop,crop]
    m.cam_intrinsic[cam]=[K['fx'],K['fy'],0,0]
    m.cam_fovy[cam]=np.degrees(2*np.arctan(crop/2/K['fy']))

def reference_pose(m,d,lay):
    """Recorded encoder positions plus fitted offsets, never double applied to sim commands."""
    s=load();p=json.loads((SPEC.parents[2]/'data/sim_ref/2026-10-25_scene/pointed.json').read_text())
    for side,prefix in [('right',''),('left','left_')]:
        q=np.array(p['joints_measured'][side][:6])+s['arms'][side]['joint_offsets_rad']
        for i,value in enumerate(q,1):d.qpos[m.joint(prefix+f'joint{i}').qposadr]=value
        for finger in ('left','right'):d.qpos[m.joint(prefix+finger+'_finger').qposadr]=.0475


def real_top_policy_image(frame, res, mode='HD1080'):
    """Rectified ZED LEFT BGR -> principal-centered square RGB, identical crop to apply()."""
    import cv2
    if mode != 'HD1080':raise ValueError('Physical rig policy camera uses HD1080 calibration')
    K=load()['top']['intrinsics_rectified'][mode];w,h=K['resolution']
    if frame.shape[:2] != (h,w):raise ValueError('ZED frame resolution does not match calibration')
    cx,cy=round(K['cx']),round(K['cy']);half=min(cx,w-cx,cy,h-cy)
    return cv2.cvtColor(cv2.resize(frame[cy-half:cy+half,cx-half:cx+half],(res,res),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)


def decorate(spec):
    """Backdrop fit to reference ZED; estimates, not surveyed geometry."""
    for g in spec.geoms:
        n=g.name
        if n.startswith('detail_'):
            if any(k in n for k in ('wall','raceway','pegboard','hole','outlet','socket','bit','hammer')):
                g.pos[0]-=.30
            if any(k in n for k in ('pegboard','hole')):
                g.pos[2]+=.25
            elif any(k in n for k in ('bit','hammer')):
                g.pos[2]+=.10
            if any(k in n for k in ('power_','power_slot','switch')):
                g.pos[0]-=.14
    table=spec.geom('table')
    table.pos[:2]=[.05,.31];table.size[:2]=[.45,.85]

    # Generic webcam housing at calibrated optical pose; housing dimensions are approximate.
    calibration=load()
    for g in spec.geoms:
        if 'd405_wrist' in g.meshname:
            g.rgba=[0,0,0,0]
    for side,prefix in [('right',''),('left','left_')]:
        pose=calibration['wrist'][side]['pose_in_link6']
        body=next(b for b in spec.bodies if b.name==prefix+'link_6')
        frame=body.add_body(name=prefix+'calibrated_webcam_visual',pos=pose['pos'],quat=pose['quat'])
        frame.add_geom(name=prefix+'webcam_housing',type=mujoco.mjtGeom.mjGEOM_BOX,
                       pos=[0,0,-.006],size=[.015,.012,.006],rgba=[.08,.08,.08,1],
                       contype=0,conaffinity=0,mass=0)
        frame.add_geom(name=prefix+'webcam_lens',type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       pos=[0,0,-.001],size=[.004,.001,0],rgba=[.02,.025,.035,1],
                       contype=0,conaffinity=0,mass=0)
