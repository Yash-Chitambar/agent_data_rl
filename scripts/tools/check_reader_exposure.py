"""Verify measured overmold placement without changing the original metal meshes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from blind_insertion.assets import builder as B

r=json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())['reader']['measurements']
exposure=float(r['plug_protrusion']['value'])
neck=tuple(float(r[k]['value']) for k in ('neck_width','neck_thickness','neck_length'))
body=(B.READER_BODY[0],B.READER_BODY[1],B.READER_TOTAL_LENGTH-exposure-neck[2])
old=B.reader_variant(neck=neck,body=(body[0],body[1],B.READER_TOTAL_LENGTH-B.USBC_PLUG_LEN-neck[2]))
new=B.reader_variant(neck=neck,body=body,exposed_length=exposure)
a,b=B.extract_pieces(old),B.extract_pieces(new)
metal=lambda p:[v for v in p.peg if v[:,2].max()<=B.USBC_PLUG_LEN+1e-9]
assert len(metal(a))==len(metal(b))
for x,y in zip(metal(a),metal(b)): np.testing.assert_array_equal(x,y)
handle=b.peg[-2]
np.testing.assert_allclose([handle[:,2].min(),handle[:,2].max()],[exposure,exposure+neck[2]],atol=1e-10)
np.testing.assert_allclose(np.concatenate(b.peg)[:,2].max(),B.READER_TOTAL_LENGTH,atol=1e-10)
fig,ax=plt.subplots(figsize=(12,4))
for part,color in zip(b.peg,['silver']*(len(b.peg)-2)+['#333333','#555555']):
    z,y=part[:,2]*1000,part[:,1]*1000
    ax.fill([z.min(),z.max(),z.max(),z.min()],[y.min(),y.min(),y.max(),y.max()],color=color,alpha=.8)
for lo,hi,label,height in [(0,5.8,'5.8 mm exposed',9),(5.8,49.2,'43.4 mm neck',9),(49.2,90.46,'41.26 mm outer housing',9),(0,90.46,'90.46 mm overall',16)]:
    ax.annotate('',(hi,height),(lo,height),arrowprops={'arrowstyle':'<->'})
    ax.text((lo+hi)/2,height+1,label,ha='center',fontsize=9)
ax.axvline(6.65,color='tab:red',linestyle='--',label='Unchanged metal ends at 6.65 mm (0.85 mm embedded)')
ax.set(xlabel='Distance from plug tip (mm)',ylabel='Thickness coordinate (mm)',ylim=(-8,22),title='Measured reader side envelope — collision geometry, not a photograph')
ax.legend(loc='lower right',fontsize=8);fig.tight_layout()
out=Path('results/usbc/bimanual_20261029/object_audit');out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'reader_exposure.png',dpi=150)
report={'metal_meshes_bitwise_unchanged':True,'exposed_mm':exposure*1000,'embedded_mm':new.handle_embed*1000,'neck_mm':neck[2]*1000,'outer_housing_mm':body[2]*1000,'total_mm':B.READER_TOTAL_LENGTH*1000,'scope':'geometry verification; no force or electrical seating validation'}
(out/'reader_exposure.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
