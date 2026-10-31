"""Add the encoder arm-pose envelope to a ZED gate checkpoint (zed_estimator.ZedEstimator.arm_ok).

v1 ZED net (runs/perception/zed.pt): trained with both arms at the reference (rest) pose and the right
fingers fully open. Envelope: each arm within --tol rad (max over joints) of its rest pose, right
fingers >= --finger-min m. Encoder-offset error (+/-1.5 deg) is inside the tolerance.

    PYTHONPATH=. .venv/bin/python scripts/perception/add_arm_envelope.py --gate zed_presence --out zed_v1_gate_arm
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
import numpy as np
import torch

from blind_insertion.scenes import pc_bench as S, physical_rig as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', default='zed_presence'); ap.add_argument('--out', default='zed_v1_gate_arm')
    ap.add_argument('--tol', type=float, default=.05); ap.add_argument('--finger-min', type=float, default=.04)
    ap.add_argument('--left-extra', default='', help='json list of extra left reference poses')
    a = ap.parse_args()
    prof = json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
    m, lay = S.build_scene(detailed=False, bimanual=True, support_mode='free', profile=prof, measured_support=True)
    d = mujoco.MjData(m); R.reference_pose(m, d, lay)
    q = lambda pre: [float(d.qpos[m.joint(pre + f'joint{i}').qposadr[0]]) for i in range(1, 7)]   # noqa: E731
    cp = torch.load(f'runs/perception/{a.gate}.pt', map_location='cpu')
    left = [q('left_')] + (json.loads(a.left_extra) if a.left_extra else [])
    cp['arm_envelope'] = {'left_refs': left, 'right_refs': [q('')], 'tol_rad': a.tol, 'right_finger_min_m': a.finger_min}
    torch.save(cp, f'runs/perception/{a.out}.pt')
    print(json.dumps(cp['arm_envelope']))


if __name__ == '__main__':
    main()
