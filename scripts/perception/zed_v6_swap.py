"""PC swap test for a pose net given by name (zed_v6): real frame with the twin PC region and the twin with the real PC
region (zed_bias_swap.pc_mask), against the twin truth. docs/real/PERCEPTION_BIAS_20261115.md, Part A.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/perception/zed_v6_swap.py --twin-dir <dir> --twin-json <json> --nets zed_v6s0 zed_v6s1
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts/perception'))
import fit_pc_edges as F  # noqa: E402
from zed_bias_swap import pc_mask  # noqa: E402
from zed_v6_replay import load_net, estimate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--twin-dir', required=True); ap.add_argument('--twin-json', required=True)
    ap.add_argument('--nets', nargs='+', default=['zed_v6s0', 'zed_v6s1'])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    nets = [load_net(n, 'cuda:0') for n in a.nets]
    tw = np.load(Path(a.twin_dir) / 'twin_rgb.npy')
    rows = json.loads(Path(a.twin_json).read_text())['rows']
    fr = {(f['run'], f['date']): f for f in json.loads((ROOT / 'results/placement/zed_map/frames.json').read_text())['frames'] if f['src'] == 'run'}
    cam = F.camera()
    out = {k: [] for k in ('real', 'twin', 'twin_with_real_pc', 'real_with_twin_pc')}
    T = []
    for i, r in enumerate(rows):
        f = fr[(r['run'], r['date'])]
        real = cv2.cvtColor(cv2.imread(f['png']), cv2.COLOR_BGR2RGB)
        m = pc_mask(f['edge_real']['xy_mm'] + [f['edge_real']['yaw_deg']], f['edge_sim']['xy_mm'] + [f['edge_sim']['yaw_deg']], cam)[..., None]
        imgs = np.stack([real, tw[i], tw[i] * (1 - m) + real * m, real * (1 - m) + tw[i] * m]).astype(np.uint8)
        E, _ = estimate(nets, imgs, 'cuda:0')
        for k, e in zip(out, E):
            out[k].append(e)
        T.append(r['truth'])
    T = np.array(T)
    summ = {k: np.median(np.array(v) - T, 0).round(2).tolist() for k, v in out.items()}
    summ.update(n=len(T), nets=a.nets)
    print(json.dumps(summ))
    if a.out:
        Path(a.out).write_text(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
