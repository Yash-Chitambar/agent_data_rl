"""ZED PC pose net for the placement v2 layout (docs/real/ZED_V5_20261114.md): scripts/perception/train_approach_estimators.py
unchanged (--which zed, policy/v1 visual DR, 40 epochs, width 32, lr 2e-3, n 49152), arms at the reset (capture)
pose as the rig-proven v1 net (zed.pt; --vary-arms 0): nets trained with varied arm poses (zed_v4 recipe, vary_arms 0.7)
read real frames 7-15 mm toward +y (FINDINGS 2026-11-01 Round 6), and so did this script with --vary-arms 0.7
(zed_v5a, kept as a sim-only study: +4..+9 mm y on the measured real spots). With
  * the layout results/placement/task_layout_real_v2.json (PC nominal = the real spot (231, 394) mm, 4 deg),
  * a wider PC box: +/-40 mm in x and y, +/-20 deg (zed_v4: +/-30 mm, +/-10 deg about (220, 360) mm, 0 deg),
  * labels relative to the new nominal (NOMINAL_V5 below; zed/zed_v4: (220, 360) mm),
  * the crop moved with the nominal (ZED_CROP_V5; zed/zed_v4: x 110, y 70, 160 px): the v5 box projects to
    u 109-188, v 149-216 px in the 320-px frame, the old crop starts at u 110.
The checkpoint stores 'nominal_port_xy' and 'zed_crop'; perception/zed_estimator.ZedEstimator reads both (old
checkpoints: the module defaults, so the old nets behave as before).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl python scripts/perception/train_zed_v5.py --seed 10 --tag _v5s0
Writes runs/perception/zed_v5s0.pt and runs/perception/zed_v5/estimator_zed_v5s0.json (new names; old nets untouched).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAYOUT = 'results/placement/task_layout_real_v2.json'
NOMINAL_V5 = (.231, .394)
ZED_CROP_V5 = (96, 72, 160)
BOX = ('0.04', '0.04', '20')


def main():
    import torch
    import train_approach_estimators as T
    global ZED_CROP_V5
    argv = sys.argv[1:]
    if '--crop' in argv:          # study: another crop (e.g. the v1 crop 110 70 160)
        i = argv.index('--crop'); ZED_CROP_V5 = tuple(int(v) for v in argv[i + 1:i + 4]); del argv[i:i + 4]
    T.NOMINAL['port_xy'] = NOMINAL_V5
    T.ZED_CROP = ZED_CROP_V5
    defaults = {'--which': ['zed'], '--n': ['49152'], '--epochs': ['40'], '--vary-arms': ['0'], '--pc-range': list(BOX),
                '--task-layout': [LAYOUT], '--out-dir': ['runs/perception/zed_v5']}
    for k, v in defaults.items():
        if k not in argv:
            argv += [k, *v]
    sys.argv = [sys.argv[0]] + argv
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--tag', default=''); p.add_argument('--out-dir'); p.add_argument('--pc-range', type=float, nargs=3)
    a, _ = p.parse_known_args(argv)
    T.main()
    src = Path(a.out_dir) / f'zed{a.tag}.pt'
    ck = torch.load(src, map_location='cpu', weights_only=False)
    ck.update(nominal_port_xy=NOMINAL_V5, zed_crop=ZED_CROP_V5, layout=LAYOUT, pc_pose_range=list(a.pc_range), argv=argv)
    torch.save(ck, Path('runs/perception') / f'zed{a.tag}.pt')
    print('wrote', Path('runs/perception') / f'zed{a.tag}.pt')


if __name__ == '__main__':
    main()
