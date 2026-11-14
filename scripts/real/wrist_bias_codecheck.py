"""Check of the deploy code path of align.secondary.bias_model on real logged frames (no arm, no camera):
deploy.bimanual.align.stop_tip_y on the prepare frames of a run (via port_vision.plug_depth + measure) must give the
value of the offline analysis (wrist_bias.json tip_cy_stop), and WristNetSource.cur_bias must give the model offset;
without bias_model it must give bias_mm exactly. Prints FAILED: none or the failures.

    PYTHONPATH=. python scripts/real/wrist_bias_codecheck.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.bimanual import align as A, port_vision as PV  # noqa: E402

RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')


def main():
    fails = []
    T = {(t['date'], t['run']): t for t in json.loads((ROOT / 'results/placement/perception_bias/wrist_bias.json').read_text())['table']}
    cfg = json.loads((ROOT / 'deploy/config/bimanual_align10_all_netgrasp.json').read_text())['align']['secondary']
    for key in [('2026-11-13', 'full_171043_all9b_full_1'), ('2026-11-11', 'full_155351_demo_1'), ('2026-11-10', 'full_152717_k4_1')]:
        run = RUNS / key[0] / key[1]
        fr = [json.loads(x) for x in (run / 'frames.jsonl').read_text().splitlines() if x.strip()]
        imgs = [cv2.imread(str(run / f['file'])) for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw' and str(f['tag']).startswith('align_depth')]
        z = PV.plug_depth(imgs)
        y = A.stop_tip_y(PV.measure, imgs, z, {})
        ref = T[key]['tip_cy_stop']
        src = A.WristNetSource.__new__(A.WristNetSource)
        src.bias, src.bias_model = np.asarray(cfg['bias_mm'], float), cfg['bias_model']
        src.primary = type('P', (), {'stop_tip_y': y})()
        b = src.cur_bias()
        exp = float(np.clip(cfg['bias_model']['a_mm'] + cfg['bias_model']['b'] * y, *cfg['bias_model']['clip_mm']))
        src.bias_model = None
        b0 = src.cur_bias()
        ok = abs(y - ref) < 1e-6 and abs(b[1] - exp) < 1e-9 and np.allclose(b[[0, 2]], cfg['bias_mm'][0::2]) and np.allclose(b0, cfg['bias_mm'])
        print(f'{key[1]}: stop tip y {y:.3f} mm (analysis {ref:.3f}), up offset {b[1]:.2f} mm, without model {b0.tolist()} -> {"ok" if ok else "FAIL"}')
        if not ok:
            fails.append(key[1])
    try:
        A.WristNetSource.__init__(A.WristNetSource.__new__(A.WristNetSource), None, dict(cfg, bias_model={'kind': 'other'}))
        fails.append('bad bias_model kind accepted')
    except ValueError:
        pass
    print('FAILED:', ', '.join(fails) if fails else 'none')


if __name__ == '__main__':
    main()
