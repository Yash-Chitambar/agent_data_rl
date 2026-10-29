"""Replay captured encoder poses and render three calibrated camera comparisons.
Timestamp pairing uses host arrivals, not hardware exposure synchronization.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2,mujoco,numpy as np
from PIL import Image,ImageDraw
from blind_insertion.scenes import pc_bench as S,physical_rig as R
from blind_insertion.assets.real_camera import real_policy_image


def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--annotation',type=Path);p.add_argument('--trial',default='bimanual_stationary_001');a=p.parse_args()
 spec=R.load();capture=a.bundle/'cameras'/a.trial
 meta=json.loads((capture/'capture.json').read_text())
 digest=hashlib.sha256(R.SPEC.read_bytes()).hexdigest()
 if meta['calibration_sha256']!=digest:raise ValueError('capture and render calibration differ')
 enc={}
 for side in ('right','left'):
  with (a.bundle/'motion'/a.trial/f'encoder_{side}.csv').open() as f:rows=list(csv.DictReader(f))
  enc[side]=(np.array([float(r['host_arrival_s']) for r in rows]),np.array([[float(r[f'q_{j}_rad']) for j in range(1,8)] for r in rows]))
 start=max(v[0][0] for v in enc.values());end=min(v[0][-1] for v in enc.values())
 if end<=start:raise ValueError('no simultaneous encoder coverage')
 profile=json.loads((a.bundle/'reference/right_arm_20261027/rig_profile_provisional_v0.json').read_text())
 m,l=S.build_scene(detailed=True,support_mode='free',profile=profile,bimanual=True,measured_support=True)
 d=mujoco.MjData(m);S.place_pc(m,d);S.place_reader(m,d,l)
 annotation=None
 if a.annotation:
  annotation=json.loads(a.annotation.read_text())
  if annotation['trial']!=a.trial:raise ValueError('annotation trial mismatch')
  from blind_insertion.scenes.measured_support import dimensions
  z=dimensions()['table_top_z_m']+float(profile['pc']['measurements']['height_with_feet']['value'])
  top=spec['top'];k=top['intrinsics_rectified']['HD1080'];pose=top['extrinsics_cam_in_right_base']
  cx,cy=round(k['cx']),round(k['cy']);crop=2*min(cx,1920-cx,cy,1080-cy)
  u,v=np.array(annotation['pc_lid_center_px'])*crop/640+np.array([cx-crop/2,cy-crop/2])
  q=np.array(pose['quat']);q/=np.linalg.norm(q);rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q)
  ray=rot.reshape(3,3)@np.array([(u-k['cx'])/k['fx'],(v-k['cy'])/k['fy'],1.])
  center=np.array(pose['pos'])+ray*((z-pose['pos'][2])/ray[2])
  if annotation['pc_yaw_deg']!=0:raise ValueError('nonzero annotation yaw needs rotated center-to-port transform')
  depth=float(profile['pc']['measurements']['depth']['value']);width=float(profile['pc']['measurements']['width']['value'])
  port=center[:2]+[-depth/2,width/2-S.PORT_FROM_LEFT]
  S.place_pc(m,d,port_xy=port)
  annotation['derived_port_xy_m']=port.tolist()
  if not annotation['reader_visible']:d.qpos[l.peg_qadr:l.peg_qadr+3]=[0,0,-10]

 a.out.mkdir(parents=True,exist_ok=True);report={'scene_annotation':annotation,'calibration_sha256':digest,'trial':a.trial,'views':{},
  'limitations':['host arrival alignment, exposure timing unknown','PC placement approximate from annotation when supplied; reader omitted only if annotated absent','finger opening approximated; encoder mapping not identified','procedural background and pad geometry remain approximate']}
 renderer=mujoco.Renderer(m,height=640,width=640)
 try:
  for label,camera,side in [('zed','top',None),('wrist_right','wrist_cam','right'),('wrist_left','left_wrist_cam','left')]:
   with (capture/label/'frames.csv').open() as f:frames=[r for r in csv.DictReader(f) if r['raw_png'] and start<=int(r['host_receive_ns'])/1e9<=end]
   if not frames:raise ValueError(f'no saved {label} frames inside encoder coverage')
   row=min(frames,key=lambda r:abs(int(r['host_receive_ns'])/1e9-(start+end)/2));t=int(row['host_receive_ns'])/1e9
   errors={}
   for arm,prefix in [('right',''),('left','left_')]:
    times,qs=enc[arm];idx=int(np.argmin(abs(times-t)));q=qs[idx,:6]+spec['arms'][arm]['joint_offsets_rad']
    errors[arm]=float(abs(times[idx]-t)*1000)
    for j,val in enumerate(q,1):d.qpos[m.joint(prefix+f'joint{j}').qposadr]=val
    for finger in ('left','right'):d.qpos[m.joint(prefix+finger+'_finger').qposadr]=.0475
   mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera);sim=renderer.render().copy()
   raw=cv2.imread(str(capture/label/row['raw_png']))
   real=R.real_top_policy_image(raw,640) if side is None else real_policy_image(raw,spec,640,side)
   pair=Image.new('RGB',(1280,670));pair.paste(Image.fromarray(real),(0,30));pair.paste(Image.fromarray(sim),(640,30))
   draw=ImageDraw.Draw(pair);draw.text((8,8),'REAL — recorded frame',fill='white');draw.text((648,8),'SIM - recorded joints; approximate scene',fill='white')
   pair.save(a.out/f'{label}_comparison.png');Image.blend(Image.fromarray(real),Image.fromarray(sim),.5).save(a.out/f'{label}_overlay.png')
   report['views'][label]={'frame':row['raw_png'],'host_receive_ns':row['host_receive_ns'],'nearest_encoder_arrival_delta_ms':errors}
 finally:renderer.close()
 (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
