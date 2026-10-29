"""Two-lid-marker RGB measurements. Offline commissioning; not a safety sensor."""
import cv2
import numpy as np


def detect_pair(image, roi):
    if image is None:raise ValueError('missing image')
    x,y,w,h=map(int,roi)
    if x<0 or y<0 or w<=0 or h<=0 or x+w>image.shape[1] or y+h>image.shape[0]:
        raise ValueError('invalid marker ROI')
    hsv=cv2.cvtColor(image[y:y+h,x:x+w],cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([15,50,100]),np.array([45,255,255]))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if area<100:continue
        bx,by,bw,bh=cv2.boundingRect(contour)
        if bx==0 or by==0 or bx+bw>=w or by+bh>=h:
            raise ValueError('marker clipped at ROI boundary')
        m=cv2.moments(contour)
        candidates.append({'pixel':[x+m['m10']/m['m00'],y+m['m01']/m['m00']],'area_px':area})
    if len(candidates)!=2:raise ValueError('expected exactly two visible yellow lid markers')
    return sorted(candidates,key=lambda c:c['pixel'][0])


def compare_pair(reference,current,cam,z):
    a=np.array([cam.ground(*r['pixel'],z)[:2] for r in reference])
    b=np.array([cam.ground(*r['pixel'],z)[:2] for r in current])
    va=a[1]-a[0];vb=b[1]-b[0];la=np.linalg.norm(va);lb=np.linalg.norm(vb)
    if la<.02 or not .9*la<=lb<=1.1*la:raise ValueError('marker separation inconsistent')
    for r,c in zip(reference,current):
        if not .8<=c['area_px']/r['area_px']<=1.2:raise ValueError('marker area changed; possible occlusion')
    angle=np.arctan2(va[0]*vb[1]-va[1]*vb[0],np.dot(va,vb))
    return {'marker_midpoint_translation_m':(b.mean(axis=0)-a.mean(axis=0)).tolist(),
            'relative_yaw_rad':float(angle),'marker_separation_m':float(lb)}
