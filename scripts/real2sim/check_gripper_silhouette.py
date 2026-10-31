"""Diagnostic opening sweep against recorded wrist yellow-tip silhouettes.

Fits a single stationary pose only; does not recalibrate cameras or establish
hardware gripper encoder mapping. Color segmentation and the bottom ROI are assumptions.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2,mujoco,numpy as np
from PIL import Image
from blind_insertion.scenes import pc_bench as S,physical_rig as R
from blind_insertion.assets.real_camera import real_policy_image
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
manifest=json.loads(Path('results/usbc/bimanual_20261029/overlays_pad_correction/manifest.json').read_text())
spec=R.load();digest=hashlib.sha256(R.SPEC.read_bytes()).hexdigest()
assert digest==manifest['calibration_sha256']
profile=json.loads((a.bundle/'reference/right_arm_20261027/rig_profile_provisional_v0.json').read_text())
m,l=S.build_scene(detailed=True,bimanual=True,support_mode='free',profile=profile,measured_support=True)
d=mujoco.MjData(m);S.place_pc(m,d);S.place_reader(m,d,l)
a.out.mkdir(parents=True,exist_ok=True);report={'calibration_sha256':digest,'views':{},'scope':'stationary opening diagnostic only; no calibration or encoder mapping adopted'}
r=mujoco.Renderer(m,640,640)
try:
 for side,prefix in [('right',''),('left','left_')]:
  view=manifest['views']['wrist_'+side];t=int(view['host_receive_ns'])/1e9
  for arm,pre in [('right',''),('left','left_')]:
   with (a.bundle/'motion'/manifest['trial']/f'encoder_{arm}.csv').open() as f:rows=list(csv.DictReader(f))
   row=min(rows,key=lambda x:abs(float(x['host_arrival_s'])-t))
   for j in range(1,7):d.qpos[m.joint(pre+f'joint{j}').qposadr]=float(row[f'q_{j}_rad'])+spec['arms'][arm]['joint_offsets_rad'][j-1]
   for finger in ('left','right'):d.qpos[m.joint(pre+finger+'_finger').qposadr]=.0475
   if arm==side:encoder=float(row['q_7_rad'])
  raw=cv2.imread(str(a.bundle/'cameras'/manifest['trial']/('wrist_'+side)/view['frame']))
  real=real_policy_image(raw,spec,640,side);hsv=cv2.cvtColor(real,cv2.COLOR_RGB2HSV)
  mask=(hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>=100)&(hsv[:,:,2]>=80);mask[:400]=False
  gids=[g for g in range(m.ngeom) if m.geom_bodyid[g] in [m.body(prefix+'left_finger').id,m.body(prefix+'right_finger').id] and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.mesh(m.geom_dataid[g]).name.endswith('soft_tips')]
  assert gids and mask.sum()>50
  scores=[];best=None
  for opening in [.025,.030,.035,.040,.043,.045,.0475,.048]:
   for finger in ('left','right'):d.qpos[m.joint(prefix+finger+'_finger').qposadr]=opening
   mujoco.mj_forward(m,d);r.enable_segmentation_rendering();r.update_scene(d,camera=prefix+'wrist_cam');seg=r.render()
   sim=np.isin(seg[:,:,0],gids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM));sim[:400]=False
   iou=float((sim&mask).sum()/max(1,(sim|mask).sum()));scores.append({'opening_m':opening,'iou':iou})
   if best is None or iou>best[0]:best=(iou,opening,sim.copy())
  overlay=real.copy();overlay[mask]=(.5*overlay[mask]+.5*np.array([0,255,0])).astype('uint8');overlay[best[2]]=(.5*overlay[best[2]]+.5*np.array([255,0,255])).astype('uint8')
  Image.fromarray(overlay).save(a.out/(side+'_mask_overlay.png'))
  report['views'][side]={'frame':view['frame'],'gripper_encoder':encoder,'scores':scores,'best_opening_m':best[1],'best_iou':best[0],'overlay':'green real HSV mask; magenta simulated yellow-tip mesh'}
finally:r.close()
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
