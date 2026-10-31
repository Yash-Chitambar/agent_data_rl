"""Diagnostic geometry overlay on the source photo; does not refit or alter geometry."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blind_insertion.scenes import pc_rear
from blind_insertion.assets.builder import USBC_REC
p=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
cfg=pc_rear.layout(p);land=json.loads((pc_rear.ASSET/'landmarks.json').read_text())
photo=np.asarray(Image.open(pc_rear.ASSET/'source.png'))
H=np.linalg.inv(np.array(cfg['image_to_metric']))
def pixels(xy):
 v=np.c_[xy,np.ones(len(xy))]@H.T
 return v[:,:2]/v[:,2:]
w,h=USBC_REC;r=h/2
outline=np.array([(cx+r*np.cos(a),r*np.sin(a)) for cx,angles in [(w/2-r,np.linspace(-np.pi/2,np.pi/2,60)),(-w/2+r,np.linspace(np.pi/2,3*np.pi/2,60))] for a in angles])
out=Path('results/usbc/bimanual_20261029/overlays');out.mkdir(parents=True,exist_ok=True)
fig,axs=plt.subplots(1,2,figsize=(15,5.8),layout='constrained')
for ax in axs:ax.imshow(photo);ax.set_xlim(100,1610);ax.set_ylim(850,155);ax.axis('off')
axs[0].set_title('Source photo (unmodified)')
axs[1].set_title('Simulation geometry projected onto photo')
rect=pixels(np.array([[0,0],[cfg['width'],0],[cfg['width'],cfg['height']],[0,cfg['height']],[0,0]]))
axs[1].plot(*rect.T,color='cyan',lw=1.7,label='102.5 × 41.2 mm case envelope')
for name,color in [('usbc_1','orange'),('usbc_2_target','lime')]:
 center=np.array(cfg['centers'][name]);curve=pixels(outline+center)
 axs[1].plot(*np.vstack([curve,curve[0]]).T,color=color,lw=2,label=('Port 2 physical aperture' if name.endswith('target') else 'Port 1 nominal USB-C size'))
 centerpx=pixels(center[None])[0];axs[1].plot(*centerpx,marker='+',color=color,ms=10)
axs[1].legend(loc='lower center',fontsize=9)
fig.suptitle('Registration check: centers and case boundary fitted from this photo; aperture size preserved independently',fontsize=12)
fig.savefig(out/'rear_photo_geometry_overlay.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.imshow(photo)
curve=pixels(outline+cfg['centers']['usbc_2_target']);ax.plot(*np.vstack([curve,curve[0]]).T,color='lime',lw=2)
ax.set_xlim(475,660);ax.set_ylim(550,445);ax.axis('off');ax.set_title('Port 2: green = unchanged 8.34 × 2.56 mm simulated aperture\nPhoto registration is not an independent metrology check')
fig.savefig(out/'target_port_registration.png',dpi=150);plt.close(fig)
print(out)
