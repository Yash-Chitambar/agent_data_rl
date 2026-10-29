"""Rear-panel photo mapped through UVs, with a real hole at the unchanged USB-C cavity.
No bitmap rescaling or generated image is used. Photo-derived centers are estimates.
"""
import json
from pathlib import Path
import cv2
import mujoco
import numpy as np
from blind_insertion.assets.builder import USBC_REC
ASSET=Path(__file__).resolve().parents[1]/'assets/pc_rear'


def layout(profile):
    a=json.loads((ASSET/'landmarks.json').read_text());v=profile['pc']['measurements']
    w,h=float(v['width']['value']),float(v['body_height']['value'])
    metric=np.float32([[0,h],[w,h],[w,0],[0,0]])
    H=cv2.getPerspectiveTransform(np.float32(a['case_corners_px_tl_tr_br_bl']),metric)
    centers={}
    for name,px in a['port_centers_px'].items():
        p=H@[*px,1.];centers[name]=(p[:2]/p[2]).tolist()
    target=centers['usbc_2_target'];feet=float(profile['pc']['pad_height_derived']['value'])
    return {'width':w,'height':h,'target_from_left':target[0],
            'port_height':target[1]+feet,'centers':centers,'image_to_metric':H.tolist()}


def add(sp,profile):
    cfg=layout(profile);w,h=cfg['width'],cfg['height'];tx,ty=cfg['centers']['usbc_2_target']
    fix=next(b for b in sp.bodies if b.name=='fixture')
    # Remove procedural connector art only; physical target cavity remains untouched.
    tokens=('usbc','usb_a','rj45','hdmi','dp','lock','ethernet','led','vent')
    for g in fix.geoms:
        if g.contype==0 and g.conaffinity==0 and (g.name.startswith('vis_') or
            (g.name.startswith('detail_') and any(t in g.name for t in tokens))):g.rgba=[0,0,0,0]
    # Dark internal socket backing, behind the panel; visual only, not an insertion stop.
    fix.add_geom(name='pc_socket_dark_interior',type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0,0,-.0001],size=[USBC_REC[0]/2,USBC_REC[1]/2,.00002],
        rgba=[.025,.028,.03,1],contype=0,conaffinity=0,mass=0)
    sp.add_texture(name='pc_rear_photo',type=mujoco.mjtTexture.mjTEXTURE_2D,file=str(ASSET/'source.png'))
    mat=sp.add_material(name='pc_rear_photo',rgba=[1,1,1,1],specular=.15,shininess=.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB]='pc_rear_photo'
    inv=np.linalg.inv(np.array(cfg['image_to_metric']));xs=np.unique(np.r_[np.linspace(0,w,45),tx-USBC_REC[0]/2,tx+USBC_REC[0]/2]);ys=np.unique(np.r_[np.linspace(0,h,25),ty-USBC_REC[1]/2,ty+USBC_REC[1]/2])
    verts=[];uv=[];faces=[]
    for y in ys:
        for x in xs:
            px=inv@[x,y,1];px=px[:2]/px[2]
            verts.append([x-tx,y-ty,.0009]);uv.append([px[0]/1700,px[1]/1070])
    n=len(xs)
    for j in range(len(ys)-1):
        for i in range(n-1):
            x=(xs[i]+xs[i+1])/2;y=(ys[j]+ys[j+1])/2
            if abs(x-tx)<USBC_REC[0]/2 and abs(y-ty)<USBC_REC[1]/2:continue
            k=j*n+i;faces.extend([[k,k+1,k+n+1],[k,k+n+1,k+n]])
    # Additional rear vertices give a nondegenerate convex hull for the massless visual mesh.
    for x,y in ((0,0),(w,0),(w,h),(0,h)):
        verts.append([x-tx,y-ty,.0008]);uv.append([0,0])
    sp.add_mesh(name='pc_rear_photo_mesh',inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,uservert=np.array(verts).ravel(),userface=np.array(faces).ravel(),
        usertexcoord=np.array(uv).ravel(),userfacetexcoord=np.array(faces).ravel())
    fix.add_geom(name='pc_rear_photo_surface',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='pc_rear_photo_mesh',
        material='pc_rear_photo',contype=0,conaffinity=0,mass=0)
