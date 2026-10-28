"""Offline swept checks for a case-relative contact approach. Never moves hardware."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from deploy import conventions as C
from deploy.contact_geometry import ContactGeometry
from deploy.safety_task import FlexChecker


def preview(session, out):
    session,out=Path(session),Path(out)
    if out.exists():raise ValueError('refusing overwrite')
    profile=json.loads((session/'support_geometry_profile.json').read_text())
    pc=json.loads((session/'standoff_pc_check.json').read_text())
    observed=json.loads((session/'left_standoff_observation_003.json').read_text())
    calibration=json.loads(C.rig.SPEC.read_text())
    b=calibration['arms']['left_base_in_right_base'];rot=np.zeros(9)
    mujoco.mju_quat2Mat(rot,np.array(b['quat'])/np.linalg.norm(b['quat']));rot=rot.reshape(3,3)
    offset=C.joint_offsets('left');robot=C.Robot()
    q=np.mean([v['q_encoder'][:6] for v in observed['observations']],axis=0)
    _,orient=robot.fk(q+offset)
    # Preserve actual observed hand orientation rather than invent a new contact frame.
    angle=np.deg2rad(-30.)
    ry=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    world_orientation=ry@(robot.R_base@robot.R_top).numpy()
    geom=ContactGeometry(C.rig.SPEC,pc['pc_center_xy_m'],pc['yaw_deg_mod90'],profile['table_plane_z_m'])
    check=FlexChecker(offset,clearance=.04,tip_clearance=.002,
        body_clearance={'gripper':.015,'wrist_camera':.03,'link_6':.02},table_z=0.)
    # Conservative floor remains above both the table and mounting plate. No
    # global safety settings are edited by this contact-specific preview.
    cx,cy=pc['pc_center_xy_m'];rear=cx+.097/2
    positions=[('above_beyond',[rear+.035,cy,.12]),
               ('descend_clear',[rear+.035,cy,.04])]
    rows=[];accepted=True
    for phase,world in positions:
        pos=rot.T@(np.array(world)-np.array(b['pos']))
        qq,pe,re=robot.chain.ik(torch.tensor(pos[None],dtype=torch.float32),
            torch.tensor((rot.T@world_orientation)[None],dtype=torch.float32),
            torch.tensor((q+offset)[None],dtype=torch.float32),iters=400,q_rest=robot.home[None])
        target=qq[0].numpy()-offset
        failures=[];minimum_interarm=1.;minimum_pc=1.
        if pe.item()>.001 or re.item()>.03:failures.append('IK accuracy')
        for alpha in np.linspace(0,1,201):
            sample=q*(1-alpha)+target*alpha
            why=check.violation(sample)
            if why:failures.append(why);break
            geom.set_encoders({'left':sample,'right':[0,0,0,-.1,0,0]}, {'left':.006,'right':.048})
            distances=geom.distances();minimum_interarm=min(minimum_interarm,distances['minimum_interarm_distance_m'])
            minimum_pc=min(minimum_pc,distances['minimum_pc_distance_m']['left'])
            if minimum_interarm<.05:failures.append('interarm clearance');break
            if minimum_pc<.015:failures.append('precontact PC clearance');break
        rows.append(dict(phase=phase,target_world_m=world,target_q_encoder=target.tolist(),
            minimum_interarm_distance_m=minimum_interarm,minimum_pc_distance_m=minimum_pc,
            failures=failures,passed=not failures))
        if failures:accepted=False;break
        q=target
    result={'schema':'contact_approach_preview_v1','accepted_for_hardware':False,
        'offline_geometric_checks_passed':accepted,'phases':rows,
        'limitations':['No hardware connection','Right arm assumed parked; must verify live',
            'Closed left-finger CAD opening 6 mm is an estimate; not a calibrated opening',
            'Case box and camera localization remain approximate',
            'Requires native reach/goal checks and live controller health checks before motion',
            'No force limit claimed; no contact or sliding authorized by this preview']}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(preview(a.session,a.out),indent=1))
if __name__=='__main__':main()
