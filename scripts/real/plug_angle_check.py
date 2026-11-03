"""Plug angle from FK at the close and FK at the stop: real runs, and dry runs against sim truth. No motion.

    python scripts/real/plug_angle_check.py --dry <dir of full dry runs> --json results/real_align/plug_angle.json
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402
from blind_insertion.envs.inhand import nominal_reader_in_site  # noqa: E402
from deploy.bimanual.plug_angle import close_rotation, plug_rv_deg, reader_in_site_from_close  # noqa: E402

R0 = nominal_reader_in_site(torch.device('cpu'))[0].numpy().astype(float)
D = ROOT / 'data/real_runs'
REAL = ['2026-11-02/full_174850_full_A', '2026-11-02/full_175742_full_A2', '2026-11-02/collect_182100_collect1']


def one(run):
    ev = L.events(run)
    sq = [e for e in ev if e['kind'] == 'grasp_v3' and e.get('event') == 'squeeze_end']
    if not sq:
        return None
    t = dict(np.load(os.path.join(run, 'ticks.npz'), allow_pickle=True))
    k = t['k'].astype(int)
    pc = [e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc'][-1]
    eR = L.port_R(pc['yaw_deg'])
    st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
    k_end = st.get('5 insert', st.get('align', st.get('collect')))
    if k_end is None:
        return None
    qc = t['q'][np.searchsorted(k, sq[0]['tick']), 0] + L.OFF
    Rc = close_rotation(L.FK, qc)
    q_now = t['q'][np.searchsorted(k, k_end) - 1, 0] + L.OFF
    Rn = close_rotation(L.FK, q_now)
    Rrs = reader_in_site_from_close(Rc, R0)
    rv = plug_rv_deg(Rn, Rrs, eR)
    tilt_err = np.degrees(np.arcsin((Rc @ R0)[2, 1]))        # reader y axis out of vertical if the grasp were nominal
    out = dict(run=os.path.basename(run), rv_deg=rv.round(2).tolist(), rv_nominal_inhand_deg=plug_rv_deg(Rn, R0, eR).round(2).tolist(), k_end=int(k_end))
    ft = [e for e in ev if e['kind'] == 'frame_truth' and int(e['frame_tag'].split('@')[-1]) <= k_end]
    if ft:
        out['truth_deg'] = np.round(ft[-1]['plug_rot_deg'], 2).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry', nargs='*', default=[])
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    rows = dict(real=[], dry=[])
    for r in REAL:
        o = one(str(D / r))
        rows['real'].append(o)
        print(json.dumps(o))
    for root in a.dry:
        for d in sorted(glob.glob(root + '/*/*/')):
            try:
                o = one(d.rstrip('/'))
            except Exception as e:  # noqa: BLE001
                print(d, repr(e)[:80])
                continue
            if o:
                rows['dry'].append(o)
    if rows['dry']:
        T = np.array([x['truth_deg'] for x in rows['dry'] if 'truth_deg' in x])
        P = np.array([x['rv_deg'] for x in rows['dry'] if 'truth_deg' in x])
        N = np.array([x['rv_nominal_inhand_deg'] for x in rows['dry'] if 'truth_deg' in x])
        rows['dry_summary'] = dict(n=len(T), err_mean=(P - T).mean(0).round(2).tolist(), err_sd=(P - T).std(0).round(2).tolist(), nominal_err_sd=(N - T).std(0).round(2).tolist(), truth_sd=T.std(0).round(2).tolist())
        print(rows['dry_summary'])
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
