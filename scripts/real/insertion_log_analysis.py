"""Insertion phase of real and dry runs in the port frame: hand-over pose, climb, target against measured, contact, stop.

    python scripts/real/insertion_log_analysis.py data/real_runs/2026-11-02/full_174850_full_A ...  [--anchor <collect run>] [--json out.json]

All distances come from encoder FK in the believed frame. Displacements are good; a constant error stays (joint offsets). With --anchor
(a collect run, same PC spot), the "gap" and "across" values are relative to its seated pose: see docs/real/ALIGN_STAGE.md for the limit.
Axes: x across, y up, z out of the port (ZED yaw).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.envs.inhand import measured_mean  # noqa: E402
from deploy import conventions as C  # noqa: E402
from deploy.bimanual.scene import SceneModel  # noqa: E402

SC = SceneModel('cpu')
FK = SC.chain['right']
OFF = C.joint_offsets('right')
Rs, ps = measured_mean(device='cpu')
TIP_S = (Rs.reshape(3, 3).float().T @ ps.reshape(3).float())


def events(run):
    return [json.loads(x) for x in open(os.path.join(run, 'events.jsonl'))]


def port_R(yaw_deg):
    y = math.radians(yaw_deg)
    z = np.array([-math.cos(y), -math.sin(y), 0.])
    up = np.array([0., 0., 1.])
    return np.c_[np.cross(up, z), up, z]


def fk_tip(q_b):
    """(n, 3) plug tip, (n, 3) TCP, (n, 3, 3) R of believed angles (n, 6)."""
    q = torch.as_tensor(np.asarray(q_b), dtype=torch.float32).reshape(-1, 6)
    ps_, Rs_ = [], []
    for i in range(0, len(q), 16):
        p, R = FK.fk(q[i:i + 16])
        ps_.append(p), Rs_.append(R)
    p, R = torch.cat(ps_), torch.cat(Rs_)
    return (p - (R @ TIP_S[:, None])[..., 0]).numpy().astype(float), p.numpy().astype(float), R.numpy().astype(float)


def load(run):
    ev = events(run)
    meta = json.load(open(os.path.join(run, 'meta.json')))
    pc = [e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc'][-1]
    st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
    t = dict(np.load(os.path.join(run, 'ticks.npz'), allow_pickle=True))
    pol = dict(np.load(os.path.join(run, 'policy.npz'), allow_pickle=True)) if os.path.exists(os.path.join(run, 'policy.npz')) else None
    hold = dict(np.load(os.path.join(run, 'hold.npz'), allow_pickle=True)) if os.path.exists(os.path.join(run, 'hold.npz')) else None
    return dict(ev=ev, meta=meta, pc=pc, st=st, t=t, pol=pol, hold=hold, eR=port_R(pc['yaw_deg']), name=os.path.basename(run.rstrip('/')))


def trace(r):
    """Insertion phase: per policy tick, plug tip (measured and from the target) relative to the hand-over tip, port axes, mm."""
    pol, eR = r['pol'], r['eR']
    q_m = np.asarray(pol['obs'][:, :6], float) + OFF
    jt = np.asarray(pol['joint_target'][:, 0], float)
    tip_m, tcp_m, R_m = fk_tip(q_m)
    tip_t, tcp_t, _ = fk_tip(jt)
    k = np.asarray(pol['k']).astype(int)
    tk = r['t']['k'].astype(int)
    i0 = np.searchsorted(tk, r['st']['5 insert'])
    q0 = np.asarray(r['t']['q'][max(i0 - 1, 0), 0], float) + OFF
    tip0, tcp0, _ = fk_tip(q0)
    rel_m = (tip_m - tip0) @ eR * 1e3      # (n, 3) port axes
    rel_t = (tip_t - tip0) @ eR * 1e3
    return dict(k=k, rel_m=rel_m, rel_t=rel_t, tip0=tip0[0], tcp0=tcp0[0], tip_m=tip_m, jt=jt, q_m=q_m, action=np.asarray(pol['action'], float),
                depth=np.asarray(pol['seat_depth_mm'], float), contact=np.asarray(pol['contact_share'], float), t=np.arange(len(k)) * .02)


def anchor(run):
    ev = events(run)
    e = next(x for x in ev if x['kind'] == 'collect_anchor')
    return dict(tip=np.array(e['tip'], float), eR=np.array(e['port_R'], float), rel=np.array(e['rel_mm'], float), q_b=np.array(e['q_b'], float))


def summary(r, a=None):
    tr = trace(r)
    t_, m, c = tr['t'], tr['rel_m'], tr['rel_t']
    pick = lambda s: int(min(np.searchsorted(t_, s), len(t_) - 1))  # noqa: E731
    out = dict(run=r['name'], n=len(t_), outcome=(r['meta'].get('outcome') or {}).get('label'), inhand6=next((e['inhand6'] for e in r['ev'] if e['kind'] == 'insert_start'), None))
    out['measured_mm'] = {f't{s}': m[pick(s)].round(2).tolist() for s in (1, 2, 4, 6, 8, 10)}
    out['target_mm'] = {f't{s}': c[pick(s)].round(2).tolist() for s in (1, 2, 4, 6, 8, 10)}
    out['max_up_mm'] = float(m[:, 1].max())
    out['max_in_mm'] = float(-m[:, 2].min())
    out['range_across_mm'] = [float(m[:, 0].min()), float(m[:, 0].max())]
    if a is not None:
        out['start_rel_anchor_mm'] = (a['rel'] + a['eR'].T @ (tr['tip0'] - a['tip']) * 1e3).round(2).tolist()
        out['end_rel_anchor_mm'] = (a['rel'] + a['eR'].T @ (tr['tip_m'][-1] - a['tip']) * 1e3).round(2).tolist()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--anchor', default=None)
    ap.add_argument('--json', default=None)
    A = ap.parse_args()
    an = anchor(A.anchor) if A.anchor else None
    rows = []
    for run in A.runs:
        rows.append(summary(load(run), an))
        print(json.dumps(rows[-1]))
    if A.json:
        Path(A.json).write_text(json.dumps(rows, indent=1))
