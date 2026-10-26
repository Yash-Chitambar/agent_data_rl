"""I2RT flexible_4310 geometry with rigid contact-pad approximation.

Upstream source/license in flexpoint/. No soft-body/material calibration is implied.
Keep the existing link_6/base inertia (it already includes the gripper housing);
use upstream moving-finger inertias. Payload still requires a physical weighing.
"""
from pathlib import Path
import mujoco
import numpy as np

ASSET = Path(__file__).with_suffix('') / 'flexible_4310.xml'


def install(spec):
    for key in list(spec.keys): spec.delete(key)
    source = mujoco.MjSpec.from_file(str(ASSET))
    model = source.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Remove the old fingers and all references before installing the upstream pair.
    for e in list(spec.equalities):
        if e.name1 in ('left_finger', 'right_finger'):
            spec.delete(e)
    for e in list(spec.excludes):
        if any(n in ('left_finger','right_finger','left_pad','right_pad') for n in (e.bodyname1,e.bodyname2)):
            spec.delete(e)
    spec.delete(spec.actuator('gripper'))
    for name in ('left_finger','right_finger'):
        spec.delete(spec.body(name))
    for g in list(spec.geoms):
        if g.name == 'gripper_housing' or g.meshname == 'gripper':
            spec.delete(g)
    for name in ('tcp_site','grasp_site'):
        spec.delete(spec.site(name))
    for side, old in [('left','linear_module'),('right','linear_module_2')]:
        body = source.body(old)
        bid = model.body(old).id
        R = data.xmat[bid].reshape(3,3)
        sign = 1 if side == 'left' else -1
        center = np.array([sign*.0017, .0007, .134])
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, R.T.flatten())
        body.add_geom(name=side+'_pad', type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=R.T@(center-data.xpos[bid]), quat=quat,
                      size=[.0017,.0072,.0082], rgba=[.08,.08,.08,1],
                      contype=4, conaffinity=3, condim=4, priority=1,
                      friction=[1,.005,.0005],solref=[.01,1],mass=0)
        body.name=side+'_finger'
        joint=next(iter(body.joints)); joint.name=side+'_finger'
        joint.range=[-.005,.048]; joint.actfrcrange=[-100,100]
        joint.armature=.1; joint.damping=[200,0,0]
        for g in body.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                g.contype=g.conaffinity=0
                if g.meshname == 'linear_module':g.rgba=[.12,.12,.12,1]
    # Body names changed but equality names refer to the original joints.
    for e in source.equalities:
        e.name1='left_finger';e.name2='right_finger'
    for e in list(source.excludes):source.delete(e)
    for site in source.sites:site.group=4
    root=source.body('gripper');root.mass=0;root.inertia=[0,0,0]
    root.add_geom(name='gripper_housing',type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0,0,-.04],size=[.03,.05,.02],group=3,contype=4,conaffinity=3,mass=0)
    # Keep the established TCP orientation; tip midpoint comes from upstream CAD.
    source.site('grasp_site').quat=[0,1,0,0]
    spec.body('link_6').add_frame().attach_body(root,'','')
    a=spec.add_actuator(name='gripper',target='left_finger',trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.gainprm[0]=1
