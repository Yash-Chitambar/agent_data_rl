"""CPU-only geometry preview for both real arms and a conservative PC box.

No robot/camera connection. Unlike the historical visual-only left arm, both
arms retain collision geometry. This is not calibrated contact dynamics.
"""
import json
from pathlib import Path
import mujoco
import numpy as np
from blind_insertion.assets import builder as B


class ContactGeometry:
    def __init__(self, spec_path, pc_center, pc_yaw_deg, table_z, pc_size=(.097,.1025,.0445)):
        self.calibration=json.loads(Path(spec_path).read_text())
        root,_=B.load_robot('yam',gripper='flexpoint')
        left,_=B.load_robot('yam',gripper='flexpoint')
        for source in (root,left):
            for key in list(source.keys):source.delete(key)
            for actuator in list(source.actuators):source.delete(actuator)
        pose=self.calibration['arms']['left_base_in_right_base']
        frame=root.worldbody.add_frame(pos=pose['pos'],quat=pose['quat'])
        frame.attach_body(left.body('arm'),'left_','')
        size=np.asarray(pc_size,float)
        quat=[np.cos(np.deg2rad(pc_yaw_deg)/2),0,0,np.sin(np.deg2rad(pc_yaw_deg)/2)]
        root.worldbody.add_geom(name='measured_pc_proxy',type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size/2,pos=[*pc_center,table_z+size[2]/2],quat=quat,
            contype=1,conaffinity=7,rgba=[.6,.6,.6,1])
        self.m=root.compile();self.d=mujoco.MjData(self.m)
        self.pc=self.m.geom('measured_pc_proxy').id
        self.arm_geoms={};self.tip_geoms={}
        for side,prefix in [('right',''),('left','left_')]:
            ids=[];tips=[]
            for g in range(self.m.ngeom):
                if g==self.pc:continue
                bid=int(self.m.geom_bodyid[g])
                if bid==0:continue
                ancestor=bid
                while self.m.body_parentid[ancestor]!=0:
                    ancestor=int(self.m.body_parentid[ancestor])
                root_name=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_BODY,ancestor)
                if (root_name=='left_arm') != (side=='left'):continue
                body=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_BODY,bid) or ''
                name=mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
                # Builder disables arm collision flags for training. Explicitly use
                # its link capsules, camera/housing boxes and fingertip hulls here.
                if int(self.m.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),int(mujoco.mjtGeom.mjGEOM_BOX)) and not name.endswith('_tip_hull'):
                    continue
                self.m.geom_contype[g]=4;self.m.geom_conaffinity[g]=7
                ids.append(g)
                if body in (prefix+'left_finger',prefix+'right_finger'):tips.append(g)
            self.arm_geoms[side]=ids;self.tip_geoms[side]=tips

    def set_encoders(self, poses, finger_slide=.006):
        for side,prefix in [('right',''),('left','left_')]:
            q=np.asarray(poses[side],float)+np.asarray(self.calibration['arms'][side]['joint_offsets_rad'])
            for i,v in enumerate(q,1):self.d.qpos[self.m.joint(prefix+f'joint{i}').qposadr]=v
            for name in ('left_finger','right_finger'):
                self.d.qpos[self.m.joint(prefix+name).qposadr]=finger_slide[side] if isinstance(finger_slide,dict) else finger_slide
        mujoco.mj_forward(self.m,self.d)

    def distances(self):
        def distance(a,b):return float(mujoco.mj_geomDistance(self.m,self.d,a,b,1.,None))
        cross=min((distance(a,b) for a in self.arm_geoms['left'] for b in self.arm_geoms['right']),default=1.)
        pc={side:min((distance(g,self.pc) for g in ids),default=1.) for side,ids in self.arm_geoms.items()}
        non_tip={side:min((distance(g,self.pc) for g in ids if g not in self.tip_geoms[side]),default=1.) for side,ids in self.arm_geoms.items()}
        return {'minimum_interarm_distance_m':cross,'minimum_pc_distance_m':pc,'minimum_non_tip_pc_distance_m':non_tip}
