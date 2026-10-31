"""Render optical-axis markers to verify subpixel crop projection (EGL required)."""
from pathlib import Path
import mujoco,numpy as np,json
from blind_insertion.scenes import physical_rig as R
s=R.load();out={}
m=mujoco.MjModel.from_xml_string('<mujoco><visual><global offwidth="640" offheight="640"/></visual><worldbody><camera name="c" pos="0 0 0"/><geom type="sphere" size=".012" pos="0 0 -1"/></worldbody></mujoco>');d=mujoco.MjData(m)
r=mujoco.Renderer(m,640,640);r.enable_segmentation_rendering()
for name in ['left','right','top']:
 if name=='top':
  K=s['top']['intrinsics_rectified']['HD1080'];w,h=K['resolution'];crop=2*min(round(K['cx']),w-round(K['cx']),round(K['cy']),h-round(K['cy']))
 else:K=s['wrist'][name]['intrinsics'];crop=s['wrist'][name]['policy_crop_px']
 m.cam_sensorsize[0]=crop;m.cam_resolution[0]=crop;m.cam_intrinsic[0]=[K['fx'],K['fy'],*R._pp_offset(K['cx'],K['cy'])]
 mujoco.mj_forward(m,d);r.update_scene(d,camera='c');im=r.render();ys,xs=np.where(im[:,:,0]==0)
 expected=(np.array([K['cx']-round(K['cx']),K['cy']-round(K['cy'])])+crop/2+.5)*640/crop-.5
 actual=np.array([xs.mean(),ys.mean()]);out[name]={'expected':expected.tolist(),'rendered':actual.tolist(),'error':(actual-expected).tolist()}
r.close()
assert max(abs(v) for row in out.values() for v in row['error']) < .15
report=Path('results/usbc/bimanual_20261029/principal_point_check.json')
report.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
