"""Check measured reader envelope and preserve the original plug mesh exactly."""
import json
from pathlib import Path
import numpy as np
from blind_insertion.assets import builder as B
v = B.reader_variant()
old = B.reader_variant(body=(.036,.012,.036), neck=(.014,.007,.028))
p, previous = B.extract_pieces(v), B.extract_pieces(old)
def plug(pieces):
    return [a for a in pieces.peg if a[:,2].max() <= B.USBC_PLUG_LEN + 1e-9]
a, b = plug(p), plug(previous)
assert len(a) == len(b) and all(np.array_equal(x,y) for x,y in zip(a,b))
verts = np.concatenate(p.peg)
assert np.allclose(np.ptp(verts, axis=0), [.03473,.00883,.09046],atol=1e-9)
assert np.isclose(v.handle['width'], .01287)
assert np.isclose(v.grip_half, .03473/2)
assert np.isclose(v.grasp_z, .06946)
report = dict(body_width_mm=34.73, body_thickness_mm=8.83, neck_width_mm=12.87,
              overall_length_mm=90.46, plug_mm=[8.25,2.40,6.65],
              plug_mesh_identical=True, estimated_body_length_mm=42.,
              derived_neck_length_mm=41.81, estimated_neck_thickness_mm=4.5,
              collision_envelope_verified=True)
Path('results/H/scene_review/reader_dimensions.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
