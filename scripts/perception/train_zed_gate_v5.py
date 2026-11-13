"""Gate for the zed_v5 pose nets (docs/real/ZED_V5_20261114.md): scripts/perception/train_zed_gate_v2.py unchanged
(presence CNN + Mahalanobis OOD at 1.25 x p99.9 + two-net disagreement at p99.5), with the
positives and the feature statistics at the arm pose the pose nets were trained at (--arms 0: reset / capture pose, as
zed_v5; --arms 1: varied arms, as gate_v2 for zed_v4 / zed_v5a; the 'beyond training' test stays at vary_arms 2.5), the
placement v2 layout, the v5 PC box (+/-40 mm, +/-20 deg about the real spot) and the v5 crop. Then the gate becomes the
estimator key: it names its pose nets ('pose_nets'), carries the nominal and the crop, and gets the arm-pose envelope
of zed_v1_gate_arm (same ZED capture pose: configs/zed_capture_pose.json, right fingers open >= 40 mm).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl python scripts/perception/train_zed_gate_v5.py
Writes runs/perception/zed_v5_gate.pt and results/usbc/policy/approach/zed_v5_gate.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_zed_v5 import BOX, LAYOUT, NOMINAL_V5, ZED_CROP_V5  # noqa: E402


def main():
    import torch
    import train_zed_gate_v2 as G
    from blind_insertion.perception.zed_estimator import zed_crop
    G.LAYOUT = LAYOUT
    G.zed_crop = lambda X: zed_crop(X, ZED_CROP_V5)   # noqa: E731
    argv = sys.argv[1:]
    arms = 0.
    if '--arms' in argv:
        i = argv.index('--arms'); arms = float(argv[i + 1]); del argv[i:i + 2]
    render0 = G.render
    G.render = lambda r, n, present, scale: render0(r, n, present, arms if scale == 1. else scale)   # noqa: E731
    for k, v in {'--pose': ['zed_v5s0'], '--pose2': ['zed_v5s1'], '--tag': ['zed_v5_gate'], '--pc-range': list(BOX)}.items():
        if k not in argv:
            argv += [k, *v]
    sys.argv = [sys.argv[0]] + argv
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--pose'); p.add_argument('--pose2'); p.add_argument('--tag')
    a, _ = p.parse_known_args(argv)
    G.main()
    f = Path('runs/perception') / f'{a.tag}.pt'
    gate = torch.load(f, map_location='cpu', weights_only=False)
    env = torch.load('runs/perception/zed_v1_gate_arm.pt', map_location='cpu', weights_only=False)['arm_envelope']
    gate.update(gate_arms_scale=arms, pose_nets=[a.pose, a.pose2], nominal_port_xy=NOMINAL_V5, zed_crop=ZED_CROP_V5, arm_envelope=env, layout=LAYOUT,
                pc_pose_range=[float(x) for x in BOX])
    torch.save(gate, f)
    print('wrote', f, 'pose_nets', gate['pose_nets'], 'ood', gate['ood_thresh'] * gate['ood_scale'])


if __name__ == '__main__':
    main()
