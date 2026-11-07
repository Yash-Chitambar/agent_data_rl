"""Real insertion hand-over states, one row per run, for the real-to-sim replay (docs/real/K2_PROGRESS.md,
"Real-to-sim replay and roll").

Per run that reached the insertion (insert_start event):
  * vision at the hand-over: align_handover lat_mm / gap_mm (align-stage runs); otherwise port_vision on the last raw
    right-wrist stream frames before insert_start (offline, median of up to 3 good frames, one shell depth per grasp);
  * plug angles: rv_deg of the last align_measure cycles (rx, ry, rz = roll; align runs only), and the applied turn;
  * the hand-over target lead (policy.npz obs[38:44] at t = 0, rad), the FK pose features obs[50:56], the in-hand
    estimate (insert_start inhand6);
  * outcome: the log's insert_result and the operator's truth (TRUTH below, from docs/real/SESSION_*.md and
    results/real_align/guard_real.json); plant: plain PD or the precision controller in the insertion.

  python scripts/real/real_handover_states.py [RUN_DIR ...] --out results/real_align/replay/real_handovers.json
Without run dirs: every non-dry run under data/real_runs/2026-11-0{1,2,3,6} with insert_start.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = pathlib.Path('/home/yashchitambar/peg_test/data/real_runs')

# Operator truth (seated in the port or not), with the source. A run not listed: truth unknown.
TRUTH = {
    'approach_insert_164528_ai2': (False, 'SESSION_20261101: operator, plug stopped short of the port (seated_detected was a false positive); reader placed by hand'),
    'approach_insert_165634_ai3': (True, 'guard_real.json: seated, stopped early by the old detector'),
    'approach_insert_171123_ai4': (True, 'guard_real.json: seated at about 4 s'),
    'approach_insert_173731_ai5': (False, 'guard_real.json: failed, see the notes'),
    'full_174850_full_A': (False, 'SESSION_20261102: plug on the case'),
    'full_175742_full_A2': (False, 'SESSION_20261102: climbed over the port'),
    'full_155211_align_full2': (True, 'SESSION_20261103: operator confirmed fully in'),
    'full_164912_align_full5': (False, 'SESSION_20261103: stalled on the case'),
    'full_165244_align_full8': (False, 'SESSION_20261106: controller in the insertion, K-J climbed 9.7 mm, guard stop'),
    'full_173941_align_full9': (True, 'SESSION_20261106: operator confirmed'),
    'full_175030_align_full10': (False, 'SESSION_20261106: stalled on the case'),
}


def events(run):
    return [json.loads(l) for l in open(run / 'events.jsonl')]


def offline_vision(run, t_insert, n_last=3):
    """port_vision on the last raw right-wrist frames captured before t_insert. Returns dict or None."""
    import cv2
    from deploy.bimanual import port_vision as PV
    rows = [json.loads(l) for l in open(run / 'frames.jsonl')]
    rows = [r for r in rows if r.get('cam') == 'wrist_right' and r.get('kind') == 'raw' and tuple(r.get('shape', ())[:2]) == (1024, 1280)
            and r.get('t_use', r.get('t_capture', 0)) <= t_insert]
    if not rows:
        return None
    rows = rows[-12:]
    imgs = [cv2.imread(str(run / r['file'])) for r in rows]
    imgs = [i for i in imgs if i is not None]
    z = PV.plug_depth(imgs) if len(imgs) >= 3 else None
    ms = []
    for img, r in list(zip(imgs, rows))[::-1]:
        m = PV.measure(img, plug_z_mm=z)
        if m.ok:
            ms.append((m.across_mm, m.up_mm, m.gap_mm, r['t_use'] - t_insert))
        if len(ms) >= n_last:
            break
    if not ms:
        return dict(ok=False, plug_z_mm=z)
    a = np.array(ms)
    return dict(ok=True, lat_mm=np.median(a[:, :2], 0).tolist(), gap_mm=float(np.median(a[:, 2])), n=len(ms),
                dt_s=a[:, 3].tolist(), spread_mm=(a[:, :3].max(0) - a[:, :3].min(0)).tolist(), plug_z_mm=z)


def row(run, do_vision=True):
    run = pathlib.Path(run)
    ev = events(run)
    get = lambda k: [e for e in ev if e['kind'] == k]  # noqa: E731
    ist = get('insert_start')
    if not ist:
        return None
    ist = ist[-1]
    res = (get('insert_result') or [{}])[-1]
    ho = (get('align_handover') or [None])[-1]
    meas = [e for e in get('align_measure') if e.get('rv_deg') is not None]
    turns = get('align_turn')
    pol = np.load(run / 'policy.npz') if (run / 'policy.npz').exists() else None
    out = dict(run=run.name, day=run.parent.name, outcome_log=res.get('outcome'), depth_mm_log=res.get('depth_mm'), t_seat_s=res.get('t_seat_s'))
    tr = TRUTH.get(run.name)
    out['truth_seated'], out['truth_src'] = (tr if tr else (None, 'unknown'))
    ac = [e for e in ev if e['kind'] in ('precision_active',) and e.get('active')]
    stg = [e for e in ev if e['kind'] == 'stage']
    out['plant'] = 'controller' if any('5 insert' in str(a.get('stage', '')) for a in ac) or run.name.endswith('align_full8') else 'plain_pd'
    if ho:
        out.update(vision_src='align_handover', lat_mm=ho['lat_mm'], gap_mm=ho['gap_mm'], handover_ok=ho.get('ok'), handover_lead_cfg=ho.get('lead'))
    if meas:
        rv = np.array([m['rv_deg'] for m in meas[-3:]], float)
        out['rv_deg_last3_median'] = np.median(rv, 0).tolist()
        out['rv_deg_last3_spread'] = (rv.max(0) - rv.min(0)).tolist()
        out['rv_deg_all_cycles'] = [m['rv_deg'] for m in meas]
        out['rot_applied_last'] = meas[-1].get('rot_applied')
    if turns:
        out['n_turns'] = len(turns)
    if do_vision:
        ov = offline_vision(run, ist['t'])
        out['offline_vision'] = ov
        if not ho and ov and ov.get('ok'):
            out.update(vision_src='offline_port_vision', lat_mm=ov['lat_mm'], gap_mm=ov['gap_mm'])
    out['inhand6'] = ist.get('inhand6')
    if pol is not None and len(pol['obs']):
        o = pol['obs'][0]
        out['lead0_mrad'] = (o[38:44] * 1e3).round(2).tolist()
        out['left_lead0_mrad'] = (o[44:50] * 1e3).round(2).tolist()
        out['fk_tcp_obs'] = o[50:53].tolist()          # (TCP - nominal) x 10, m
        out['fk_eul_obs'] = o[53:56].tolist()
        out['n_policy_steps'] = int(len(pol['obs']))
        g = run / 'guard.npz'
        if g.exists():
            G = np.load(g)
            d = G['depth2_mm'][:len(pol['obs'])]
            out['fk_depth_max_mm'] = float(np.nanmax(d)) if np.isfinite(d).any() else None
            out['tgt_up_max_mm'] = float(G['rel_t'][:, 1].max())
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--out', default=str(ROOT / 'results/real_align/replay/real_handovers.json'))
    ap.add_argument('--no-vision', action='store_true')
    a = ap.parse_args()
    runs = a.runs or sorted(d for day in ('2026-11-01', '2026-11-02', '2026-11-03', '2026-11-06') for d in glob.glob(str(DATA / day / '*'))
                            if 'dry' not in d and pathlib.Path(d, 'events.jsonl').exists())
    rows = [r for r in (row(d, not a.no_vision) for d in runs) if r]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(rows, indent=1, default=float))
    for r in rows:
        ov = r.get('offline_vision') or {}
        print(f"{r['run']:28s} {r['plant']:10s} log {str(r['outcome_log']):16s} truth {str(r['truth_seated']):5s} src {r.get('vision_src', '-'):20s} "
              f"lat {np.round(r.get('lat_mm', [np.nan, np.nan]), 2)} gap {r.get('gap_mm', np.nan):.2f} rv {np.round(r.get('rv_deg_last3_median', [np.nan] * 3), 2)} "
              f"offline {np.round(ov.get('lat_mm', [np.nan, np.nan]), 2)} {ov.get('gap_mm', np.nan) if ov.get('ok') else ov.get('ok')}")
