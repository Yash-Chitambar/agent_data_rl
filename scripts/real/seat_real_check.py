"""Old and new seat detectors on the real logs. No motion. A and A2 must not seat; the jogged seats of collect1 and ai4 (operator: seated) show a true seat.

    python scripts/real/seat_real_check.py --json results/real_align/seat_real.json

Hand-over measurement variants: port_vision on the stop frames; FK label from the collect anchor (A, A2) or the run's own seat (ai4). The stage is off in these runs, so the
stop here is the approach stop (gap about 11 mm): the new detector takes the measured gap wherever the hand-over is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import align_replay as AR  # noqa: E402
import insertion_log_analysis as L  # noqa: E402
from deploy.bimanual import port_vision as PV  # noqa: E402
from deploy.bimanual.insert import SeatDetector  # noqa: E402
from deploy.bimanual.insert_guard import SeatDetector2  # noqa: E402

D = ROOT / 'data/real_runs'
SEAT = dict(margin_mm=1.0, gate_mm=(3.0, 6.0), hold_s=.3, speed_mm_s=3., still_mm=99.)


def pv_at(run, tick_end, tick_from=None, cam='wrist_right', pre=300):
    fr = AR.stop_frames_upto(run, tick_end, pre)
    vs = []
    pz = PV.plug_depth([cv2.imread(str(raw)) for t, raw, _ in fr]) if fr else None
    for t, raw, _ in fr:
        m = PV.measure(cv2.imread(str(raw)), side='right', **({'plug_z_mm': pz} if pz else {}))
        if m.ok:
            vs.append([m.across_mm, m.up_mm, m.gap_mm])
    return (np.median(np.array(vs), 0).round(2).tolist() if vs else None), len(vs), len(fr)


def replay(q_b, eR, tip0, g0, lat0, k=None, dt=.02):
    rel = (L.fk_tip(q_b)[0] - tip0) @ eR * 1e3
    det = SeatDetector2(1, torch.device('cpu'), g0, lat0, **SEAT)
    t_seat, last = None, None
    for i, r in enumerate(rel):
        o = det.update(torch.tensor(r, dtype=torch.float32)[None])
        last = o
        if t_seat is None and bool(o['seated'][0]):
            t_seat = i * dt
    d = -rel[:, 2] - g0
    return dict(g0=g0, lat0=lat0, seated=t_seat is not None, t_seat_s=t_seat, depth_end=float(d[-1]), depth_max=float(d.max()), lat_end=(np.asarray(lat0) + rel[-1, :2]).round(2).tolist())


def old_replay(r, tcp_q, dt=.02):
    det = SeatDetector(torch.as_tensor(r['eR'], dtype=torch.float32), .006, .0058, .001, .2, .005, dt)
    tt = L.fk_tip(tcp_q)[1]
    t_seat = None
    for i, p in enumerate(tt):
        d, v, s = det.update(p)
        if s and t_seat is None:
            t_seat = i * dt
    return dict(seated=t_seat is not None, t_seat_s=t_seat, depth_end_mm=d * 1e3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    out = []
    an = L.anchor(str(D / AR.COLLECT))
    # A, A2
    for name in ('2026-11-02/full_174850_full_A', '2026-11-02/full_175742_full_A2'):
        run = D / name
        r = L.load(str(run))
        tr = L.trace(r)
        pv, nok, nf = pv_at(run, r['st']['5 insert'])
        lab = (an['rel'] + an['eR'].T @ (tr['tip0'] - an['tip']) * 1e3)
        row = dict(run=name, truth='not seated (operator)', old=old_replay(r, tr['q_m']), port_vision=pv, pv_frames=[nok, nf], fk_label=lab.round(2).tolist())
        row['new_fk_label_gap'] = replay(tr['q_m'], r['eR'], tr['tip0'], float(lab[2]), lab[:2].tolist())
        if pv:
            row['new_port_vision'] = replay(tr['q_m'], r['eR'], tr['tip0'], pv[2], pv[:2])
        row['new_old_gap_assumption'] = replay(tr['q_m'], r['eR'], tr['tip0'], 6., [0., 0.])
        out.append(row)
    # A and A2 with the port vision agent's stop values (docs/real/PORT_VISION.md, one depth per run), label offset known
    for row, v in zip(out, ((-.3, 5.0, 8.4), (-5.2, 8.9, 8.0))):
        r = L.load(str(D / row['run']))
        tr = L.trace(r)
        row['new_port_vision_doc_values'] = replay(tr['q_m'], r['eR'], tr['tip0'], v[2], list(v[:2]))
    # ai4: seated in the run; own seat as anchor
    name = '2026-11-01/approach_insert_171123_ai4'
    run = D / name
    r = L.load(str(run))
    tr = L.trace(r)
    pv, nok, nf = pv_at(run, r['st']['5 insert'])
    end = tr['tip_m'][-1]
    lab = np.array([0, 0, -5.8]) + (tr['tip0'] - end) @ r['eR'] * 1e3
    row = dict(run=name, truth='seated (operator, 2026-11-01)', old=old_replay(r, tr['q_m']), port_vision=pv, pv_frames=[nok, nf], fk_label_own_seat=lab.round(2).tolist())
    row['new_own_seat_label'] = replay(tr['q_m'], r['eR'], tr['tip0'], float(lab[2]), lab[:2].tolist())
    if pv:
        row['new_port_vision'] = replay(tr['q_m'], r['eR'], tr['tip0'], pv[2], pv[:2])
    out.append(row)
    # collect1: jogged seats. Hand-over: the approach stop. Up to the anchor and to the recheck
    run = D / AR.COLLECT
    r = L.load(str(run))
    t = r['t']
    k = t['k'].astype(int)
    k0 = r['st']['collect']
    ev = r['ev']
    ka = next(e['tick'] for e in ev if e['kind'] == 'collect_anchor')
    i0, ia = np.searchsorted(k, k0), np.searchsorted(k, ka)
    q = t['q'][:, 0] + L.OFF
    tip0 = L.fk_tip(q[i0])[0][0]
    pv, nok, nf = pv_at(run, k0)
    jog = [e for e in ev if e['kind'] == 'collect_jog']
    kr = max(e['tick'] for e in jog)
    ir = np.searchsorted(k, kr)
    k_samp = [e['tick'] for e in ev if e['kind'] == 'collect_sample']
    # first seat
    rel_a = (L.fk_tip(q[ia])[0][0] - tip0) @ r['eR'] * 1e3
    lab = np.array([0, 0, -5.8]) - rel_a
    row = dict(run=AR.COLLECT, truth='jogged in, anchor', advance_to_seat_mm=rel_a.round(2).tolist(), port_vision_stop=pv, pv_frames=[nok, nf], label_stop=lab.round(2).tolist(), grip_stop=float(t['grip'][i0, 0]), grip_seat=float(t['grip'][ia, 0]))
    row['new_label_to_anchor'] = replay(q[i0:ia + 1], r['eR'], tip0, float(lab[2]), lab[:2].tolist())
    if pv:
        row['new_port_vision_to_anchor'] = replay(q[i0:ia + 1], r['eR'], tip0, pv[2], pv[:2])
    # recheck seat: plug jogged in again, with the hand 4.5 mm lower (reader pitch); anchor-frame reading
    i1 = np.searchsorted(k, max(k_samp))
    rel_r = (L.fk_tip(q[ir])[0][0] - tip0) @ r['eR'] * 1e3
    row['recheck_advance_from_stop_mm'] = rel_r.round(2).tolist()
    row['new_label_to_recheck'] = replay(np.r_[q[ia:ia + 1], q[ir - 500:ir + 1]], r['eR'], tip0, float(lab[2]), lab[:2].tolist())
    if pv:
        row['new_port_vision_to_recheck'] = replay(np.r_[q[ia:ia + 1], q[ir - 500:ir + 1]], r['eR'], tip0, pv[2], pv[:2])
    row['grip_recheck'] = float(t['grip'][ir, 0])
    out.append(row)
    for row in out:
        print(json.dumps(row))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
