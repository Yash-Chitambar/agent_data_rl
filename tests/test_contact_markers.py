import cv2
import numpy as np
import pytest
from deploy.contact_markers import detect_pair,compare_pair

class Camera:
    def ground(self,u,v,z):return np.array([u*.001,v*.001,z])

def image(dx=0,dy=0,hide=False):
    out=np.full((200,300,3),150,np.uint8)
    cv2.rectangle(out,(60+dx,70+dy),(80+dx,100+dy),(0,240,255),-1)
    if not hide:cv2.rectangle(out,(140+dx,70+dy),(160+dx,100+dy),(0,240,255),-1)
    return out

def test_translation_and_missing_marker_refusal():
    roi=(20,20,220,150);ref=detect_pair(image(),roi)
    result=compare_pair(ref,detect_pair(image(5,-3),roi),Camera(),.01)
    assert np.allclose(result['marker_midpoint_translation_m'],[.005,-.003])
    assert abs(result['relative_yaw_rad'])<1e-9
    with pytest.raises(ValueError,match='exactly two'):detect_pair(image(hide=True),roi)

def test_deformed_pair_refused():
    ref=detect_pair(image(),(20,20,220,150))
    current=[dict(r,pixel=list(r['pixel'])) for r in ref];current[1]['pixel'][0]+=30
    with pytest.raises(ValueError,match='separation'):compare_pair(ref,current,Camera(),.01)


def test_yaw_is_separate_from_translation_and_occlusion_is_rejected():
    ref=[{'pixel':[70.,85.],'area_px':600.},{'pixel':[150.,85.],'area_px':600.}]
    theta=np.deg2rad(7);rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    center=np.array([110.,85.]);current=[{'pixel':(rotation@(np.array(r['pixel'])-center)+center).tolist(),'area_px':600.} for r in ref]
    result=compare_pair(ref,current,Camera(),.01)
    assert np.isclose(result['relative_yaw_rad'],theta)
    assert np.allclose(result['marker_midpoint_translation_m'],0,atol=1e-12)
    current[0]['area_px']=200.
    with pytest.raises(ValueError,match='area changed'):compare_pair(ref,current,Camera(),.01)
