"""Hand against the panel from port_vision's panel pose, on real frames: does the rotation follow FK? No motion.

    python scripts/real/panel_rotation_check.py --json results/real_align/panel_rotation.json

Collect sweep: rotation of the camera against the panel (port_vision) against the FK rotation label (rv_deg), differences, by regression. Rotation labels come from FK, the plug rigid in the hand. A and A2 at the stop: absolute, with one constant from the sweep.
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
from deploy.bimanual import port_vision as PV  # noqa: E402

D = ROOT / 'data/real_runs'


def rv_of(raw, plug_z=None):
    m = PV.measure(raw, side='right', **({'plug_z_mm': plug_z} if plug_z else {}))
    if not m.ok or not m.debug.get('R') is not None:
        return None
    return np.asarray(m.debug['R'], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    man = [json.loads(x) for x in open(ROOT / 'data/real_training/port_v1/manifest.jsonl')]
    man = [m for m in man if m['source'] == 'collect' and m.get('raw')]
    R, Y, P = [], [], []
    for m in man:
        Rc = rv_of(cv2.imread(str(ROOT / m['raw'])))
        if Rc is not None:
            R.append(Rc)
            Y.append(m['rv_deg'])
            P.append(m['p_mm'])
    R, Y, P = np.array(R), np.array(Y), np.array(P)
    Rbar = Rot.from_matrix(R).mean().as_matrix()
    # panel in camera: X_cam = R X_panel. Rotation of the camera (rigid with the hand) against the panel, in panel axes: R^T changes as rot(label) applied in port axes
    V = Rot.from_matrix(np.einsum('ij,njk->nik', Rbar.T, R)).as_rotvec(degrees=True)
    A = Rot.from_matrix(np.einsum('nij,jk->nik', R.transpose(0, 2, 1), Rbar)).as_rotvec(degrees=True)
    x = Y - Y.mean(0)
    out = dict(n_frames=len(man), n_ok=len(R))
    for name, v in (('Rbar^T R', V), ('R^T Rbar', A)):
        y = v - v.mean(0)
        C = np.linalg.lstsq(x, y, rcond=None)[0]          # y = x C
        out[name] = dict(matrix_label_to_vision=C.round(3).tolist(), resid_sd=(y - x @ C).std(0).round(3).tolist())
    y = A - A.mean(0)
    near = P[:, 2] < 15
    sl = (x * y).sum(0) / (x * x).sum(0)
    out['slope_per_axis_R^T_Rbar'] = sl.round(3).tolist()
    out['near_only_resid_sd'] = (y - x * sl)[near].std(0).round(3).tolist()
    out['label_sd'] = Y.std(0).round(2).tolist()
    # absolute: plug against the panel = panel in camera, camera in site (calibrated mount), reader in site (mean of the measured real grasps)
    import torch
    sys.path.append(str(ROOT / 'scripts/real2sim'))
    import inhand_measure as IM
    from blind_insertion.envs import inhand_real as IR
    hand = IM.Hand()
    c = IR.load('v2')['inhand']
    d = np.mean(np.asarray(c['rows'], float), 0) if 'rows' in c else np.r_[c['mean_mm'], c['mean_rotvec_deg']]
    Rrs = IR.reader_in_site(torch.tensor(d[None], dtype=torch.float32), 'cpu')[0][0].numpy().astype(float)
    def plug_rv(Rcp):
        return Rot.from_matrix(np.asarray(Rcp, float).T @ hand.R_cs.T @ Rrs).as_rotvec(degrees=True)
    PR = np.array([plug_rv(x) for x in R])
    out['inhand_mean_delta'] = np.round(d, 2).tolist()
    out['sweep_plug_rv_minus_label_mean'] = (PR - Y).mean(0).round(2).tolist()
    out['sweep_plug_rv_minus_label_sd'] = (PR - Y).std(0).round(2).tolist()
    out['near_mean'] = (PR - Y)[near].mean(0).round(2).tolist()
    import align_replay as AR
    stops = {}
    for name in AR.RUNS[:3]:
        fr = AR.stop_frames(D / name)
        pz = PV.plug_depth([cv2.imread(str(x[1])) for x in fr])
        v = []
        for t, raw, pol in fr:
            Rc = rv_of(cv2.imread(str(raw)), pz)
            if Rc is not None:
                v.append(plug_rv(Rc))
        stops[name] = dict(n=len(v), last=np.round(v[-1], 2).tolist() if v else None, mean=np.mean(v, 0).round(2).tolist() if v else None, sd=np.std(v, 0).round(2).tolist() if v else None)
    out['stops_plug_rv_deg'] = stops
    print(json.dumps(out))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
