"""Offline diagnostic only: track the exposed front label, without robot access.

Fixed-baseline template correlation is not a validated safety or yaw detector.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from deploy.checks.placement import TopCam,find_blob


def main():
    root=Path('/home/yashc/left_bimanual_session')
    trial=json.loads((root/'contact/interactive_002/trial.json').read_text())
    folder=root/'cameras/contact_interactive_002/zed'
    files=sorted(folder.glob('raw_*.png'))
    cam=TopCam();xy=np.array(trial['pc_center_xy_m'])
    z=json.loads((root/'support_geometry_profile.json').read_text())['pc']['top_z_model_estimate_m']
    reference=cv2.imread(str(trial['pc_source_image']))
    hit=find_blob(reference,cam,xy,z,.12,.1025*.097)
    if hit is None:raise RuntimeError('baseline not detected')
    x,y,w,h=cv2.boundingRect(hit[2])
    # Camera-specific diagnostic ROI: visible front label and adjacent lid.
    x+=w//5;y+=int(.68*h);w=int(.6*w);h=int(.26*h)
    template=cv2.cvtColor(reference[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    rows=[]
    for path in files:
        image=cv2.imread(str(path))
        if image is None:continue
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        pad=18;search=gray[y-pad:y+h+pad,x-pad:x+w+pad]
        scores=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
        _,score,_,loc=cv2.minMaxLoc(scores)
        delta=np.array(loc)-pad
        p0=cam.ground(x+w/2,y+h/2,z)
        p1=cam.ground(x+w/2+delta[0],y+h/2+delta[1],z)
        rows.append({'image':path.name,'correlation':score,'pixel_delta':delta.tolist(),
                     'translation_m':(p1-p0).tolist(),'search_boundary':bool(np.any(np.abs(delta)>=pad))})
    out=root/'contact/interactive_002/label_replay.json'
    out.write_text(json.dumps({'status':'offline diagnostic; not enabled for motion',
        'roi_xywh':[x,y,w,h],'baseline':trial['pc_source_image'],'yaw_observed':False,'rows':rows},indent=1)+'\n')
    scores=np.array([r['correlation'] for r in rows]);print('frames',len(rows),'correlation min/median',np.min(scores),np.median(scores))
    print('boundary matches',sum(r['search_boundary'] for r in rows));print(out)
    cv2.rectangle(reference,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imwrite(str(out.with_suffix('.jpg')),reference)

if __name__=='__main__':main()
