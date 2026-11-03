"""Plug angle against the panel from one raw wrist frame, no encoders: panel in the camera (port_vision.panel_pose) and reader in the camera (scripts/real2sim/inhand_measure fit). No motion.

    python scripts/real/plug_angle_vision.py --json results/real_align/plug_angle_vision.json

Test: collect sweep frames against FK rotation labels (differences only: the labels' constant is open), slopes per axis; then A and A2 at the stop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
sys.path.append(str(ROOT / 'scripts/real2sim'))
import inhand_measure as IM  # noqa: E402
from deploy.bimanual import port_vision as PV  # noqa: E402

D = ROOT / 'data/real_runs'


def measure(raw, hand=None):
    und, K = PV.undistort(raw, 'right')
    pl, _ = PV.detect_plug(und)
    pose, why = PV.panel_pose(und, K, PV.plug_box(pl) if pl else None, PV.MAX_REPROJ_PX, PV.load_model())
    if pose is None:
        return None, 'panel: ' + why
    und2 = IM.undist(raw)
    r, why = IM.detect(und2)
    if r is None:
        return None, 'reader: ' + str(why)
    Rrc, t, rms, _ = IM.fit_pose(r['pts'], x0=hand.x0() if hand else None)
    if rms > 4.:
        return None, f'reader fit {rms:.1f} px'
    Rpr = np.asarray(pose['R'], float).T @ Rrc
    return dict(rv=Rot.from_matrix(Rpr).as_rotvec(degrees=True), rms=rms, reader_t=t.tolist()), ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    hand = IM.Hand()
    man = [json.loads(x) for x in open(ROOT / 'data/real_training/port_v1/manifest.jsonl')]
    man = [m for m in man if m['source'] == 'collect' and m.get('raw')]
    rows, why = [], {}
    for m in man:
        raw = cv2.imread(str(ROOT / m['raw']))
        o, w = measure(raw, hand)
        if o is None:
            why[w.split(':')[0]] = why.get(w.split(':')[0], 0) + 1
            continue
        rows.append(dict(rv=o['rv'].tolist(), label=m['rv_deg'], p=m['p_mm'], rms=o['rms']))
    V, Y = np.array([r['rv'] for r in rows]), np.array([r['label'] for r in rows])
    out = dict(n_frames=len(man), n_ok=len(rows), fail=why)
    if len(rows) > 5:
        x, y = Y - Y.mean(0), V - V.mean(0)
        sl = (x * y).sum(0) / (x * x).sum(0)
        out.update(slope=sl.round(3).tolist(), resid_sd=(y - x * sl).std(0).round(3).tolist(), mean_vision=V.mean(0).round(2).tolist(), mean_label=Y.mean(0).round(2).tolist(), corr=[float(np.corrcoef(x[:, i], y[:, i])[0, 1].round(3)) for i in range(3)], label_sd=Y.std(0).round(2).tolist())
    print(json.dumps(out))
    stops = {}
    sys.path.append(str(ROOT / 'scripts/real'))
    import align_replay as AR
    for name in AR.RUNS[:3]:
        v = []
        for t, raw, pol in AR.stop_frames(D / name):
            o, w = measure(cv2.imread(str(raw)), hand)
            if o is not None:
                v.append(o['rv'])
        stops[name] = dict(n=len(v), mean=np.mean(v, 0).round(2).tolist() if v else None, sd=np.std(v, 0).round(2).tolist() if v else None)
        print(name, stops[name])
    out['stops'] = stops
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
