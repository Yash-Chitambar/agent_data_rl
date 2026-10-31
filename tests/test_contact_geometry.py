import numpy as np
from deploy.contact_geometry import ContactGeometry


def test_both_arms_have_link_geometry_and_detect_overlap():
    s=ContactGeometry('blind_insertion/configs/real_cameras_duchamp3.json',[.262,.309],0,-.03)
    assert len(s.arm_geoms['left'])>=15
    assert len(s.arm_geoms['right'])>=15
    assert not set(s.arm_geoms['left']) & set(s.arm_geoms['right'])
    poses={'left':np.zeros(6),'right':np.zeros(6)}
    s.set_encoders(poses)
    assert s.distances()['minimum_interarm_distance_m']>.3
    root=s.m.body('left_arm').id
    s.m.body_pos[root]=0
    s.m.body_quat[root]=[1,0,0,0]
    poses['left']=np.array(s.calibration['arms']['right']['joint_offsets_rad'])-np.array(s.calibration['arms']['left']['joint_offsets_rad'])
    s.set_encoders(poses)
    assert s.distances()['minimum_interarm_distance_m']<0
