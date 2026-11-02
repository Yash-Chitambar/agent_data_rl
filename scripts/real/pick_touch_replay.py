"""Offline check of the touch-down detector (blind_insertion/pipeline/grasp_v3.TouchDetector) on real logs.

Free motion: no real log has the right arm descending slowly, so the false-positive margin comes from the LEFT arm's
guarded descent at 10 mm/s (stage hold, same arm type, approach_insert and hold runs): largest drop before contact.
The first 1.2 s of each descent is skipped (the left swing arrives at speed; the deployed descent starts after a settle).
Contact: the right arm from the last pick waypoint to the close (2026-11-01): first detection, tip height there.
Limit: the right descent there runs at 50 mm/s and then creeps on the integral, not at the deployed 8 mm/s.

    PYTHONPATH=. .venv/bin/python scripts/real/pick_touch_replay.py --drop-mm 8 --eps-mm 0.5 --out results/real/touch_replay.json
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def replay(zc, zm, drop, eps):
    from blind_insertion.pipeline.grasp_v3 import TouchDetector
    det = TouchDetector(1, drop, eps, 'cpu')
    out, hit = [], None
    for i in range(len(zc)):
        t, d = det.update(torch.tensor([zc[i]], dtype=torch.float32), torch.tensor([zm[i]], dtype=torch.float32))
        out.append(float(d[0]))
        if hit is None and bool(t[0]):
            hit = i
    return np.array(out), hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/real_runs/2026-11-01')
    ap.add_argument('--drop-mm', type=float, default=8.0)
    ap.add_argument('--eps-mm', type=float, default=0.5)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    from deploy.bimanual.scene import SceneModel
    from deploy.safety_task import FlexChecker
    sc = SceneModel('cpu')
    off = sc.encoder_offsets.numpy()
    chk = FlexChecker(np.zeros(6), .04, -.015, None, -.033)
    tip = lambda q: np.array([min(chk.heights(x)[f] for f in ('left_finger', 'right_finger')) for x in q]) + .033   # noqa: E731
    fk = lambda ch, q: ch.fk(torch.tensor(q, dtype=torch.float32))[0][:, 2].numpy()  # noqa: E731
    res = dict(args=vars(a), left_free=[], right_contact=[])
    for d in sorted(glob.glob(a.root + '/*')):
        if '_dry_' in d or not os.path.exists(d + '/ticks.npz'):
            continue
        t = np.load(d + '/ticks.npz')
        ks = np.where(t['stage'] == 2)[0]
        if len(ks) >= 50:
            zm, zc = fk(sc.chain['left'], t['q'][ks, 1] + off[1]), fk(sc.chain['left'], t['target_b'][ks, 1])
            v = np.gradient(zc) / .02
            idx = np.where((v < -.005) & (v > -.02))[0]
            if len(idx) >= 20:
                a0, b0 = idx[0] + 60, idx[-1]       # 1.2 s after the swing: the deployed descent starts settled
                dr, hit = replay(zc[a0:b0 + 1], zm[a0:b0 + 1], a.drop_mm * 1e-3, a.eps_mm * 1e-3)
                # the hand stops ~drop / 10 mm/s before the hit; a true contact keeps the measured height flat from there on
                z_end = zm[a0:b0 + 1]
                res['left_free'].append(dict(run=os.path.basename(d), n=int(b0 - a0), hit_i=hit, z_at_hit_mm=None if hit is None else float(z_end[hit] * 1e3),
                                             meas_fall_after_hit_mm=None if hit is None else float((z_end[hit] - z_end[hit:].min()) * 1e3),
                                             cmd_fall_after_hit_mm=None if hit is None else float((zc[a0:b0 + 1][hit] - zc[a0:b0 + 1].min()) * 1e3)))
                print('L', res['left_free'][-1], flush=True)
        ev = [json.loads(x) for x in open(d + '/events.jsonl')]
        fr = [e for e in ev if e['kind'] == 'servo_frame' and e.get('frame_kind') == 'waypoint']
        pk = np.where(t['stage'] == 1)[0]
        if len(fr) < 3 or not len(pk):
            continue
        i0 = int(np.searchsorted(t['k'], int(fr[-1]['frame_tag'].split('@')[1])))
        g0 = np.median(t['grip_cmd'][pk[:50], 0])
        below = pk[t['grip_cmd'][pk, 0] < g0 - .01]
        end = int(below[0]) if len(below) else int(pk[-1])
        rel = [e['tick'] for e in ev if e['kind'] == 'close_press_relief' and e.get('state') == 'on']
        if rel:
            end = min(end, int(np.searchsorted(t['k'], rel[0])))
        q, tb = t['q'][i0:end, 0] + off[0], t['target_b'][i0:end, 0]
        zm, zc = fk(sc.tcp_chain, q), fk(sc.tcp_chain, tb)
        tips = tip(q)
        dr, hit = replay(zc, zm, a.drop_mm * 1e-3, a.eps_mm * 1e-3)
        res['right_contact'].append(dict(run=os.path.basename(d), n=len(q), hit_i=hit, tip_start_mm=float(tips[0] * 1e3),
                                         tip_at_hit_mm=None if hit is None else float(tips[hit] * 1e3), tip_final_mm=float(tips[-1] * 1e3),
                                         drop_max_mm=float(dr.max() * 1e3)))
        print('R', res['right_contact'][-1], flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
