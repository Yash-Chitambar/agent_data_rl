"""Presence head for reader_zed_v2 (docs/real/ROBUSTNESS_PLAN_20261113.md): scripts/perception/train_reader_presence.py
unchanged, with the v2 layout (results/placement/task_layout_real_v2.json) and a +/-30 mm, +/-15 deg PC box. Adds the
presence net to runs/perception/reader_zed_v2.pt; the report goes to results/usbc/policy/pick/reader_presence_v2.json
(the v1 report reader_presence.json is restored).

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=.:scripts/perception MUJOCO_GL=egl python scripts/perception/train_reader_presence_v2.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAYOUT = 'results/placement/task_layout_real_v2.json'
REP = ROOT / 'results/usbc/policy/pick/reader_presence.json'


def main():
    import train_reader_presence as P
    BC, PC = P.BimanualCfg, P.PickTaskCfg
    P.BimanualCfg = lambda *a, **kw: BC(*a, **{**kw, 'task_layout': LAYOUT})
    P.PickTaskCfg = lambda *a, **kw: PC(*a, **{**kw, 'pc_pose_range': (0.03, 0.03, 15.)})
    if '--ckpt' not in sys.argv:
        sys.argv += ['--ckpt', 'runs/perception/reader_zed_v2.pt']
    keep = REP.read_text() if REP.exists() else None
    try:
        P.main()
        shutil.copy(REP, REP.with_name('reader_presence_v2.json'))
    finally:
        if keep is not None:
            REP.write_text(keep)


if __name__ == '__main__':
    main()
