"""Largest hand move the command did not ask for, in the last ticks of the approach, from a run's log. No motion. Pass limit for the next rig run: docs/real/ANSWERS_20261103.md.

    python scripts/real/approach_jump.py data/real_runs/<date>/full_<id> [--ticks 300] [--window-s 0.5]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--ticks', type=int, default=300)
    ap.add_argument('--window-s', type=float, default=.5)
    a = ap.parse_args()
    ev = [json.loads(x) for x in open(Path(a.run) / 'events.jsonl')]
    st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
    end = st.get('align', st.get('5 insert', st.get('collect')))
    t0 = st['4 approach']
    d = {k: [] for k in ('k', 'q', 'target_b')}
    for f in sorted(glob.glob(str(Path(a.run) / 'chunks/ticks/*.npz'))):
        z = np.load(f, allow_pickle=True)
        for k in d:
            d[k].append(z[k])
    if not d['k'][0].size and len(d['k']) == 1:
        raise SystemExit('no ticks')
    d = {k: np.concatenate(v) for k, v in d.items()}
    i = (d['k'] >= max(t0, end - a.ticks)) & (d['k'] < end)
    q = d['q'][i, 0]
    tip = L.FK.fk(torch.as_tensor(q + L.OFF, dtype=torch.float32))[0].cpu().numpy() * 1e3
    w = int(round(a.window_s / .02))
    cmd = L.FK.fk(torch.as_tensor(d['target_b'][i, 0] + L.OFF, dtype=torch.float32))[0].cpu().numpy() * 1e3
    mv = np.linalg.norm((tip[w:] - tip[:-w]) - (cmd[w:] - cmd[:-w]), axis=1)      # move the command did not ask for: a slip
    j = int(mv.argmax())
    e = (d['target_b'][i, 0] - q) * 1e3
    print(json.dumps(dict(run=Path(a.run).name, approach_ticks=[int(t0), int(end)], window_s=a.window_s, max_uncommanded_move_mm=round(float(mv.max()), 2), at_tick=int(d['k'][i][j]),
                          err_j2_j3_abs_max_mrad=np.abs(e[:, 1:3]).max(0).round(1).tolist(), err_j2_j3_end_mrad=e[-1, 1:3].round(1).tolist())))


if __name__ == '__main__':
    main()
