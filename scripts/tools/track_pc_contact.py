"""Offline PC displacement from sampled raw overhead frames; no hardware access.

Uses the working checkout's calibrated projection. The bright top-surface detector
needs manual review under occlusion; it is not a contact or safety sensor.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rig-code-root',type=Path,required=True)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--geometry',type=Path,required=True)
    ap.add_argument('--center',type=float,nargs=2,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.out.exists():raise SystemExit('refusing overwrite')
    sys.path.insert(0,str(a.rig_code_root))
    import cv2
    import numpy as np
    from deploy.checks.placement import TopCam,find_blob
    geometry=json.loads(a.geometry.read_text());height=geometry['pc']['top_z_model_estimate_m']
    if height is None:raise SystemExit('model-frame case-top height unresolved')
    with (a.capture/'zed/frames.csv').open() as f:frames=list(csv.DictReader(f))
    rows=[];cam=TopCam();reference=None;previous_yaw=None;unwrapped=0.;yaw_valid=True
    for frame in frames:
        if not frame.get('raw_png'):continue
        image=cv2.imread(str(a.capture/'zed'/frame['raw_png']))
        hit=None if image is None else find_blob(image,cam,np.array(a.center),height,.12,.1025*.097)
        row={'host_receive_ns':int(frame['host_receive_ns']),'frame_id':int(frame['local_frame_id']),
             'raw_png':frame['raw_png'],'detected':hit is not None}
        if hit is not None:
            xy,yaw,contour=hit
            if reference is None:reference={'xy':xy.tolist(),'yaw_deg_mod90':float(yaw),'frame_id':row['frame_id']}
            if previous_yaw is not None:unwrapped+=(float(yaw)-previous_yaw+45)%90-45
            previous_yaw=float(yaw)
            delta=xy-np.array(reference['xy'])
            row.update(xy_m=xy.tolist(),yaw_deg_mod90=float(yaw),delta_xy_m=delta.tolist(),
                       displacement_m=float(np.linalg.norm(delta)),delta_yaw_deg=unwrapped if yaw_valid else None,
                       yaw_continuity_valid=yaw_valid)
        else:
            # Never unwrap across a tracking loss; yaw continuity is unknown afterward.
            previous_yaw=None
            if reference is not None:yaw_valid=False
            row['tracking_lost']=True
        rows.append(row)
    result={'reference':reference,'top_plane_model_z_m':height,'frames':rows,
            'limitations':['First detected frame is reference; confirm it precedes contact',
                'Host arrival timestamps, not exposure synchronization',
                'Bright-blob identity requires manual review, especially under occlusion',
                'Yaw has 90-degree ambiguity; no reliable continuity across tracking loss',
                'Relative image-derived motion only; absolute calibration uncertainty remains']}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(result,f,indent=1);f.write('\n')
    detected=[r for r in rows if r['detected']]
    print(json.dumps({'frames':len(rows),'detected':len(detected),'maximum_relative_motion_m':max((r['displacement_m'] for r in detected),default=None),'out':str(a.out)},indent=2))

if __name__=='__main__':main()
