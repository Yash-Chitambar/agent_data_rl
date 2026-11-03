"""Does the real reader move in the jaws under push? The camera is rigid to the hand, so a change of the shell row in the frame
is a change of the plug in the hand. Shell: port vision's detector (real/port-vision, deploy/bimanual/port_vision.py; read only);
silver front edge: inhand_measure's walk. Against the grip reading, from the end of the pick to the end of the run.

A free 6-dof fit on the 4 coplanar silver corners falls into the planar ambiguity (it gave 37 deg): not used.

    PYTHONPATH=. python scripts/real2sim/inhand_push.py
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inhand_measure import RUNS, detect, undist  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--runs', nargs='+', default=['full_174850_full_A', 'full_175742_full_A2', 'collect_182100_collect1'])
ap.add_argument('--pv', default='/home/yashchitambar/wt/port-vision/deploy/bimanual/port_vision.py')
ap.add_argument('--out', default='results/real/inhand_real/push.csv')
a = ap.parse_args()
spec = importlib.util.spec_from_file_location('port_vision', a.pv)
PV = importlib.util.module_from_spec(spec)
sys.modules['port_vision'] = PV
spec.loader.exec_module(PV)
rows = []
for name in a.runs:
    run = RUNS / name
    ev = [json.loads(x) for x in open(run / 'events.jsonl')]
    z = np.load(run / 'ticks.npz')
    t0 = next(e['t'] for e in ev if e.get('event') == 'end_hold')
    fr = [json.loads(x) for x in open(run / 'frames.jsonl')]
    fr = [f for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw' and f['t_capture'] >= t0]
    for f in fr:
        raw = cv2.imread(str(run / f['file']))
        pl, _why = PV.detect_plug(PV.undistort(raw)[0])
        r, _ = detect(undist(raw), silver_only=True, dark=52, min_v=75)
        i = min(np.searchsorted(z['t'], f['t_capture']), len(z['t']) - 1)
        row = dict(run=name, t=f['t_capture'] - t0, grip=float(z['grip'][i, 0]), tag=f['tag'], qd=float(np.abs(z['qd'][i, 0]).max()))
        if pl is not None:
            row.update(ve=float(pl['ve']), w=float(pl['w']), u=float(pl['ul'] + pl['ur']) / 2)
        if r is not None:
            row['front_y'] = float(np.nanmean(r['pts'][:2, 1]))
            row['front_w'] = r['silver_px_w']
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(a.out, index=False)
pd.set_option('display.width', 200, 'display.max_rows', 400)
print(df.columns.tolist(), len(df))
col = 've'
for name, x in df.groupby('run'):
    x = x[x.qd < .02].copy()
    x['gb'] = (x.grip * 500).round() / 500
    print(name)
    print(x.groupby('gb').agg(t0=('t', 'min'), t1=('t', 'max'), n=('t', 'size'), shell=(col, 'median'), w=('w', 'median'), n_shell=(col, 'count'), front=('front_y', 'median')).round(2).sort_index(ascending=False).to_string())
