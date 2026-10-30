"""Export a pinned i2rt inverse-dynamics model without importing CAN drivers.

This candidate reconstructs the documented linear_4310 / 0.5 kg driver model.
It does not establish the exact deployed source revision or actual wrist payload.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

PIN='4c43932cdb8e74dcb576419a81a76b47b90f2705'
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--out-dir',type=Path,required=True)
a=p.parse_args();source=a.source.resolve()
git=lambda *args:subprocess.check_output(['git','-C',str(source),*args],text=True).strip()
if git('rev-parse','HEAD')!=PIN or git('status','--porcelain'):
    raise SystemExit('requires clean pinned i2rt source '+PIN)
path=source/'i2rt/robots/utils.py'
tree=ast.parse(path.read_text())
# Only definitions needed for XML construction: no CAN import, motor detection,
# calibration or controller code is loaded.
keep=[]
for node in tree.body:
    if isinstance(node,ast.ClassDef) and node.name=='JointMapper':break
    if isinstance(node,ast.ImportFrom) and node.module=='i2rt.motor_drivers.dm_driver':continue
    keep.append(node)
tree.body=keep
sys.path.insert(0,str(source))
module=types.ModuleType('_offline_i2rt_xml')
module.__file__=str(path);sys.modules[module.__name__]=module
exec(compile(tree,str(path),'exec'),module.__dict__)
combined=Path(module.combine_arm_and_gripper_xml(module.ArmType.YAM,module.GripperType.LINEAR_4310,ee_mass=.5))
try:
    original=mujoco.MjModel.from_xml_path(str(combined))
finally:combined.unlink()
original.geom_contype[:]=0;original.geom_conaffinity[:]=0;original.jnt_limited[:]=0
original.opt.gravity[:]=[0,0,-9.81]
a.out_dir.mkdir(parents=True,exist_ok=True)
dest=a.out_dir/'driver_gravity.xml'
with tempfile.TemporaryDirectory() as tmp:
    xml=Path(tmp)/'canonical.xml';mujoco.mj_saveLastXML(str(xml),original)
    root=ET.parse(xml).getroot()
    # Compiled XML has explicit inertias. Remove rendering assets/geometry only.
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ('geom','asset','light','camera'):parent.remove(child)
    ET.ElementTree(root).write(dest,encoding='unicode')
candidate=mujoco.MjModel.from_xml_path(str(dest))
for m in (original,candidate):
    m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.jnt_limited[:]=0
d0,d1=mujoco.MjData(original),mujoco.MjData(candidate)
rng=np.random.default_rng(812);errors=[]
for _ in range(100):
    q=rng.uniform(-1.2,1.2,6)
    for m,d in ((original,d0),(candidate,d1)):
        d.qpos[:6]=q;d.qvel[:]=0;d.qacc[:]=0;mujoco.mj_inverse(m,d)
    errors.append(float(np.max(abs(d0.qfrc_inverse[:6]-d1.qfrc_inverse[:6]))))
assert max(errors)<1e-5,max(errors)
shutil.copyfile(source/'LICENSE',a.out_dir/'I2RT_LICENSE')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'source_commit':PIN,'source_remote':git('remote','get-url','origin'),
        'construction':'YAM + LINEAR_4310; gripper inertial mass override 0.5 kg',
        'mujoco_version':mujoco.__version__,'nq':candidate.nq,'nv':candidate.nv,
        'source_utils_sha256':sha(path),'export_sha256':sha(dest),
        'max_static_inverse_difference_nm':max(errors),'verification_poses':100,
        'limitations':['deployed source revision not recorded in captures',
                       '0.5 kg is driver configuration, not measured payload',
                       'geometry removed after compilation; inverse dynamics preserved',
                       'candidate only; no hardware controller imported or commanded']}
(a.out_dir/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(report)
