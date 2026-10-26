"""Check camera extrinsics, lens crop and raw-frame validation against the saved rig."""
import json
from pathlib import Path
import mujoco
import numpy as np
from blind_insertion.scenes import pc_bench as S, physical_rig as R
from blind_insertion.assets.real_camera import real_policy_image
m,l=S.build_scene();d=mujoco.MjData(m);R.reference_pose(m,d,l);mujoco.mj_forward(m,d)
s=R.load();ref=json.loads(Path('data/sim_ref/2026-10-25_scene/pointed.json').read_text())
report={}
for side,prefix in [('right',''),('left','left_')]:
 c=m.camera(prefix+'wrist_cam').id
 p,q=R.camera_pose(ref['wrist_cam_in_right_base'][side])
 rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q)
 assert np.linalg.norm(d.cam_xpos[c]-p)<1e-4
 assert np.max(np.abs(d.cam_xmat[c]-rot))<1e-4
 w=s['wrist'][side]
 assert np.allclose(m.cam_intrinsic[c,:2],[w['intrinsics']['fx'],w['intrinsics']['fy']])
 assert np.allclose(m.cam_sensorsize[c],w['policy_crop_px'])
 report[side+'_extrinsics_and_crop']=True
 try:real_policy_image(np.zeros((480,640,3),dtype=np.uint8),s,320,side)
 except ValueError:pass
 else:raise AssertionError('wrong capture resolution was accepted')
report['wrong_raw_resolution_rejected']=True
assert len(l.pad_geoms)==32
assert np.allclose(m.body('left_arm').pos,s['arms']['left_base_in_right_base']['pos'])
report['left_base_transform']=True
report['rigid_contact_spheres']=len(l.pad_geoms)
Path('results/H/physical_rig/calibration_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
