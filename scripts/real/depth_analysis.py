"""Insertion depth on the real logs: FK advance along the port axis from the hand-over to the release, per run.

    python scripts/real/depth_analysis.py [RUN_DIR ...] --json results/real_align/depth/real_depth.json [--png results/real_align/depth/real_depth.png]

depth = advance of the plug tip (and of the TCP) along -port_z since the first insertion tick, minus the measured hand-over gap.
Marks: K4 seat tick, after the 0.2 s measured-pose hold, push end, after the push drop, last tick before the release.
PD axial force = J^-T kp (target - q) projected on -port_z (right arm; believed angles).
"""
from __future__ import annotations

import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402

DATA = '/home/yashchitambar/peg_test/data/real_runs'
# operator notes, docs/real/SESSION_20261110/11_duchamp3.md
OP = {'full_145824_kjrc_1': 'seated, hangs', 'full_150628_kjrc_2': 'seated, hangs, ~1 mm short', 'full_152149_kjrc_3_push': 'seated, ~3 mm short',
      'full_152717_k4_1': 'seated, ~3 mm short', 'full_154422_k4_2': 'in, not detected, pulled out', 'full_155541_k4_3': 'seated, really well',
      'full_161011_k4_4': 'seated (push stopped by jaw watch)', 'full_163515_k4_5': 'flush and seated', 'full_170521_k4_6': 'seated',
      'full_155351_demo_1': 'seated, hangs slightly (push skipped)', 'full_160310_demo_2': 'seated', 'full_161223_demo_3': 'seated',
      'full_161705_demo_4': 'fully inserted (best)', 'full_163144_speed075_1': 'success (push skipped)', 'full_171905_fastmove_1': 'inserted perfectly'}


def axial_force(q, tgt, z_b):
    """PD force on the TCP along -port_z (N), from kp (target - q)."""
    kp = L.SC.kp[0].float() if hasattr(L.SC, 'kp') else None
    if kp is None:
        return np.full(len(q), np.nan)
    out = []
    qt, tt = torch.as_tensor(q, dtype=torch.float32), torch.as_tensor(tgt, dtype=torch.float32)
    for i in range(0, len(qt), 64):
        _, _, J = L.FK.fk(qt[i:i + 64], with_jac=True)
        tau = kp * (tt[i:i + 64] - qt[i:i + 64])
        F = torch.linalg.lstsq(J.transpose(1, 2), tau[..., None]).solution[:, :3, 0]
        out.append(-(F.numpy() @ z_b))
    return np.concatenate(out)


def one(run):
    r = L.load(run)
    ev = r['ev']
    if not any(e.get('kind') == 'insert_start' for e in ev):
        return None
    ho = next((e for e in ev if e.get('kind') == 'align_handover'), None)
    gap = ho['gap_mm'] if ho else float('nan')
    t = r['t']
    k = t['k'].astype(int)
    i0 = int(np.searchsorted(k, r['st']['5 insert']))
    i1 = int(np.searchsorted(k, r['st'].get('6 release', k[-1])))
    i2 = min(len(k), i1 + 100)
    q = np.asarray(t['q'][i0 - 1:i2, 0], float) + L.OFF
    tgt = np.asarray(t['target_b'][i0 - 1:i2, 0], float) if 'target_b' in t else q
    tip, tcp, _ = L.fk_tip(q)
    eR = r['eR']
    rel_tip = (tip - tip[0]) @ eR * 1e3
    rel_tcp = (tcp - tcp[0]) @ eR * 1e3
    d_tip, d_tcp = -rel_tip[:, 2] - gap, -rel_tcp[:, 2] - gap
    F = axial_force(q, tgt + (L.OFF if np.abs(tgt[0] - q[0]).max() > np.abs(tgt[0] + L.OFF - q[0]).max() else 0), eR[:, 2])
    seat = next((e for e in ev if e.get('kind') == 'seated_detected'), None)
    push = next((e for e in ev if e.get('kind') == 'insert_final_push'), None)
    skip = next((e for e in ev if e.get('kind') == 'insert_final_push_skipped'), None)
    res = next((e for e in ev if e.get('kind') == 'insert_result'), {})
    n_ins = i1 - i0
    js = None if seat is None else int(seat['step']) + 1          # index in the trace (trace[0] = the tick before the insertion)
    out = dict(run=r['name'], operator=OP.get(r['name']), outcome=res.get('outcome'), gap_mm=gap, lat_mm=None if ho is None else ho['lat_mm'],
               seat_step=None if seat is None else seat['step'], seat_depth2_mm=None if seat is None else seat['depth_mm'],
               push_advance_mm=None if push is None else push['advance_mm'], push_stopped=None if push is None else push.get('stopped'),
               push_skipped=None if skip is None else skip['reason'], result_advance_mm=res.get('advance_mm'))
    g = np.load(os.path.join(run, 'guard.npz')) if os.path.exists(os.path.join(run, 'guard.npz')) else None
    if g is not None and len(g['lat2_mm']):
        out['seat_lat2_mm'] = np.asarray(g['lat2_mm'][-1], float).round(2).tolist()
    def at(j):
        j = int(min(max(j, 0), len(d_tcp) - 1))
        return dict(tcp=round(float(d_tcp[j]), 2), tip=round(float(d_tip[j]), 2), up=round(float(rel_tip[j, 1]), 2), F=round(float(F[j]), 1))
    if js is not None:
        out['at_seat'] = at(js)
        out['after_hold'] = at(js + 10)
    out['max_depth'] = dict(tcp=round(float(d_tcp[:n_ins + 1].max()), 2), tip=round(float(d_tip[:n_ins + 1].max()), 2), F_max=round(float(np.nanmax(F[:n_ins + 1])), 1))
    out['before_release'] = at(n_ins)
    if push is not None:
        jp = int(np.searchsorted(k, push['tick'])) - (i0 - 1)
        out['push_end'] = at(jp - 15)                       # before the 0.3 s drop
        out['push_F_max'] = round(float(np.nanmax(F[js + 10:jp - 14])), 1)
    grip = np.asarray(t['grip'][i0 - 1:i2, 0], float)
    out['grip'] = [round(float(grip[0]), 4), round(float(grip[min(n_ins, len(grip) - 1)]), 4)]
    out['_trace'] = dict(t=((np.arange(len(d_tcp)) - 1) * .02), d_tcp=d_tcp, d_tip=d_tip, up=rel_tip[:, 1], across=rel_tip[:, 0], F=F, n_ins=n_ins, js=js)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*'); ap.add_argument('--json'); ap.add_argument('--png')
    a = ap.parse_args()
    runs = a.runs or sorted(x for d in ('2026-11-10', '2026-11-11') for x in glob.glob(f'{DATA}/{d}/full_*') if '_dry' not in x)
    rows = []
    for run in runs:
        try:
            o = one(run)
        except Exception as e:  # noqa: BLE001
            print('skip', os.path.basename(run), type(e).__name__, e); continue
        if o:
            rows.append(o)
            print(json.dumps({k: v for k, v in o.items() if k != '_trace'}))
    if a.png:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        n = len(rows); f, ax = plt.subplots(2, n, figsize=(3.2 * n, 6), squeeze=False, sharey='row')
        for j, o in enumerate(rows):
            tr = o['_trace']
            ax[0, j].plot(tr['t'], tr['d_tcp'], label='TCP'); ax[0, j].plot(tr['t'], tr['d_tip'], label='tip'); ax[0, j].axvline(tr['n_ins'] * .02, color='k', ls=':')
            if tr['js']: ax[0, j].axvline(tr['js'] * .02, color='g', ls='--')
            ax[0, j].axhline(5.8, color='r', lw=.5); ax[0, j].set_title(f"{o['run'][5:]}\n{o['operator']}", fontsize=7); ax[0, j].grid(alpha=.3)
            ax[1, j].plot(tr['t'], tr['F']); ax[1, j].grid(alpha=.3); ax[1, j].set_xlabel('s')
        ax[0, 0].set_ylabel('FK depth mm'); ax[1, 0].set_ylabel('PD axial force N'); ax[0, 0].legend(fontsize=6)
        f.tight_layout(); f.savefig(a.png, dpi=80)
    if a.json:
        json.dump([{k: v for k, v in o.items() if k != '_trace'} for o in rows], open(a.json, 'w'), indent=1)
