"""reader_zed for the placement v2 layout (docs/real/ROBUSTNESS_PLAN_20261113.md): scripts/perception/train_reader_zed.py
unchanged, with the layout results/placement/task_layout_real_v2.json (PC nominal = the real spot (231, 394) mm, 4 deg;
reader body centres x 85-175, y 335-425 mm, plug yaw 50-130 deg) and a wider PC box (default +/-30 mm, +/-15 deg).
Labels, crop, strip centre and yaw0 stay as in v1, so the deploy loader reads the new file the same way.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. MUJOCO_GL=egl python scripts/perception/train_reader_zed_v2.py --tag _v2 [--pc-box 0.03 0.03 15]
Writes runs/perception/reader_zed_v2.pt and results/usbc/policy/pick/reader_zed_v2.json (new names; v1 untouched).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAYOUT = 'results/placement/task_layout_real_v2.json'


def main():
    box = (0.03, 0.03, 15.)
    if '--pc-box' in sys.argv:
        i = sys.argv.index('--pc-box')
        box = tuple(float(v) for v in sys.argv[i + 1:i + 4])
        del sys.argv[i:i + 4]
    if '--tag' not in sys.argv:
        sys.argv += ['--tag', '_v2']
    import train_reader_zed as T
    BC, PC = T.BimanualCfg, T.PickTaskCfg

    def bcfg(*a, **kw):
        kw['task_layout'] = LAYOUT
        return BC(*a, **kw)

    def pcfg(*a, **kw):
        kw['pc_pose_range'] = box
        return PC(*a, **kw)
    T.BimanualCfg, T.PickTaskCfg = bcfg, pcfg
    T.main()


if __name__ == '__main__':
    main()
