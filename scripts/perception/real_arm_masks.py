"""Arm masks for real ZED frames from the logged encoders: the sim model at the measured joint angles
(encoder + calibrated joint offsets, physical_rig.load()), rendered from the nominal calibrated ZED
camera (no DR), segmentation -> arm pixels, dilated like training. Only sessions with encoder logs
(motion/<session>/encoder_{left,right}.csv) and static arms are supported (one mask per session).

    PYTHONPATH=. .venv/bin/python scripts/perception/real_arm_masks.py --session bimanual_stationary_001
Writes runs/perception/real_masks/<session>.npz (mask (320, 320) bool, q_left, q_right)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import numpy as np
import torch

from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
from blind_insertion.perception.arm_mask import ArmMasker
from blind_insertion.scenes import physical_rig as R

ROOT = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--session', default='bimanual_stationary_001')
    a = ap.parse_args()
    rt = BimanualRuntime(BimanualCfg(nworld=1, device='cuda:0', render_rgb=True, visual_dr=None, camera_resolution=320,
                                     measured_camera_cadence=False))
    cal = R.load()
    out = {}
    for side in ('left', 'right'):
        enc = np.loadtxt(ROOT / 'motion' / a.session / f'encoder_{side}.csv', delimiter=',', skiprows=1)
        q = enc[:, 2:8].mean(0) + np.asarray(cal['arms'][side]['joint_offsets_rad'])
        grip = float(enc[:, 8].mean())                       # 1 = open
        rt.q[:, rt.indices[side][0]] = torch.tensor(q, device=rt.device, dtype=rt.q.dtype)
        pre = '' if side == 'right' else 'left_'
        fq = [int(rt.mjm.joint(pre + f + '_finger').qposadr[0]) for f in ('left', 'right')]
        rt.q[:, fq] = .0475 * min(max(grip, 0.), 1.)
        out[f'q_{side}'] = q
    rt.q[:, rt.pc_qadr:rt.pc_qadr + 3] = torch.tensor([3., 3., -2.], device=rt.device)   # mask the arms only
    with wp.ScopedDevice(rt.cfg.device):
        mjw.forward(rt.m, rt.d)
    out['mask'] = ArmMasker(rt)()[0].cpu().numpy()
    Path('runs/perception/real_masks').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'runs/perception/real_masks/{a.session}.npz', **out)
    print(a.session, 'mask pixels', int(out['mask'].sum()), {k: np.round(v, 3).tolist() for k, v in out.items() if k != 'mask'})


if __name__ == '__main__':
    main()
