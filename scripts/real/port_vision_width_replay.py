"""Replay of the port vision shell detector on every real align frame: old detector against the shell reference (plug_ref, off by default).

    PYTHONPATH=. .venv/bin/python scripts/real/port_vision_width_replay.py [--days 2026-11-0* 2026-11-1*] [--procs 24]

For each run folder with raw right wrist align frames: the 9 prepare frames (align_depth_*) give the grasp's shell depth (the logged one
when the run accepted it) and the shell reference (port_vision.plug_ref). Then every align frame (prepare, step cycles, servo, wrist net
frames, in-hand) is measured twice: old = measure(raw, plug_z_mm=z), new = measure(raw, plug_z_mm=z, plug_ref=ref).
Also: the old replay against the logged per-frame values (code drift), the shell edges against the reference, the bright patch next to
the right edge and where the case bottom edge line sits against the shell rows. Output: results/real_runs_review/port_vision_width.{json,csv}.
docs/real/PORT_VISION_WIDTH_20261114.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import multiprocessing as mp
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # scripts/real/deploy.py would shadow the deploy package
ROOT = Path('/home/yashchitambar/peg_test/data/real_runs')
OUT = Path(__file__).resolve().parents[2] / 'results/real_runs_review'
STEP = re.compile(r'^align_(\d{3})_(\d+)$')


def kind_of(tag):
    if tag.startswith('align_depth_'):
        return 'prepare'
    if STEP.match(tag):
        return 'step'
    if tag.startswith('align_servo') or re.match(r'^align\d+_servo', tag):
        return 'servo'
    if tag == 'align_inhand':
        return 'inhand'
    if re.match(r'^align\d+_\d+$', tag):
        return 'wrist_net'
    return 'other'


def runs(days):
    out = []
    for pat in days:
        for d in sorted(glob.glob(str(ROOT / pat / 'full_*'))):
            f = Path(d) / 'frames.jsonl'
            if not f.exists():
                continue
            fr = []
            for line in f.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get('cam') == 'wrist_right' and r.get('kind') == 'raw' and r['tag'].startswith('align') and (Path(d) / r['file']).exists():
                    fr.append(r)
            if fr:
                out.append((d, fr))
    return out


def events(d):
    ev = []
    p = Path(d) / 'events.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return ev


def ev_kind(e):
    return e.get('kind', e.get('event'))


def diag(PV, und, ref, pose):
    """Bright patch right of the shell (grey levels over the background) and the case bottom edge row at the shell's right edge."""
    import cv2
    g = cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
    ve, ur = int(round(ref['ve'])), int(round(ref['ur']))
    band = g[ve - 26:ve - 5]
    patch = float(band[:, ur + 2:ur + 9].mean() - band[:, ur + 14:ur + 21].mean())
    vb = None
    if pose is not None:
        K = PV.undistort.__globals__['_MAPS']['right'][0]
        L = PV.load_model()['lines']['bottom']
        xs = np.linspace(-15, 25, 81)
        P = np.array([PV._proj(K, pose['R'], pose['t'], PV.line_point(L, 'bottom', x))[:2] for x in xs])
        vb = float(np.interp(ur, P[:, 0], P[:, 1])) if P[0, 0] < ur < P[-1, 0] else None
    return patch, (None if vb is None else float((ve - 26) - vb))     # > 0: the edge line is above the side band rows


def one(job):
    d, recs, z, ref = job
    import cv2
    cv2.setNumThreads(1)
    from deploy.bimanual import port_vision as PV
    out = []
    for r in recs:
        raw = cv2.imread(str(Path(d) / r['file']))
        row = dict(run=Path(d).name, day=Path(d).parent.name, tag=r['tag'], kind=kind_of(r['tag']), file=r['file'], tick=r.get('tick'))
        if raw is None or raw.shape[:2] != (1024, 1280):
            row['err'] = 'bad frame'
            out.append(row)
            continue
        for name, kw in (('old', dict(plug_z_mm=z)), ('new', dict(plug_z_mm=z, plug_ref=ref))):
            if z is None:
                row[f'{name}_ok'], row[f'{name}_reason'] = False, 'no shell depth for this grasp'
                continue
            if name == 'new' and ref is None:
                kw = dict(plug_z_mm=z)
            m = PV.measure(raw, **kw)
            pl = m.debug.get('plug') or {}
            row.update({f'{name}_ok': bool(m.ok), f'{name}_reason': m.reason, f'{name}_v': None if not m.ok else [m.across_mm, m.up_mm, m.gap_mm],
                        f'{name}_ul': pl.get('ul'), f'{name}_ur': pl.get('ur'), f'{name}_ve': pl.get('ve'), f'{name}_w': pl.get('w'),
                        f'{name}_fix': pl.get('ref_fix', '')})
            if name == 'old' and ref is not None:
                und, _ = PV.undistort(raw)
                pose = m.debug if 'R' in m.debug else None
                row['patch'], row['edge_line_px'] = diag(PV, und, ref, pose)
        out.append(row)
    return out


FAILED_RUN = ROOT / '2026-11-13/full_163924_all9_align_1'


def selftest():
    """The rule must not hide a moved shell. Failed-run frames of cycles 3-7 with the true reference: measured. With a reference as if the
    shell had moved since the prepare (both edges shifted, the end row shifted, the width changed): no edge is replaced, the old result stays."""
    import cv2
    from deploy.bimanual import port_vision as PV
    fr = [json.loads(x) for x in (FAILED_RUN / 'frames.jsonl').read_text().splitlines()]
    fr = [r for r in fr if r['cam'] == 'wrist_right' and r['kind'] == 'raw']
    dep = [cv2.imread(str(FAILED_RUN / r['file'])) for r in fr if r['tag'].startswith('align_depth_')]
    late = [cv2.imread(str(FAILED_RUN / r['file'])) for r in fr if STEP.match(r['tag']) and int(STEP.match(r['tag']).group(1)) >= 2]
    ref, z = PV.plug_ref(dep), PV.plug_depth(dep)
    cases = dict(true_ref={}, shell_moved_across_2px=dict(ul=2, ur=2), shell_moved_across_3px=dict(ul=3, ur=3), shell_moved_across_m3px=dict(ul=-3, ur=-3),
                 end_row_moved_4px=dict(ve=4), deeper_width_m3px=dict(ul=1.5, ur=-1.5, w=-3), nearer_width_p3px=dict(ul=-1.5, ur=1.5, w=3),
                 right_edge_ref_off_3px=dict(ur=3, w=3))
    res, ok_all = {}, True
    for name, sh in cases.items():
        r2 = {k: (v + sh.get(k, 0.0) if k in ('ul', 'ur', 've', 'w') else v) for k, v in ref.items()}
        ms = [PV.measure(f, plug_z_mm=z, plug_ref=r2) for f in late]
        fixes = sum(bool(m.debug.get('plug') and m.debug['plug'].get('ref_fix')) for m in ms)
        n_ok = sum(m.ok for m in ms)
        want = (n_ok == len(ms) and fixes == len(ms)) if name == 'true_ref' else (fixes == 0 and n_ok == 0)
        ok_all &= want
        res[name] = dict(n=len(ms), measured=n_ok, edge_replaced=fixes, pass_=bool(want))
        print(f"{'ok  ' if want else 'FAIL'} {name}: {n_ok}/{len(ms)} measured, edge replaced {fixes}")
    print('FAILED: none' if ok_all else 'FAILED: selftest')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', nargs='+', default=['2026-11-0*', '2026-11-1*'])
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--selftest', action='store_true', help='only the moved-shell test on the failed run (about 20 s)')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import cv2
    cv2.setNumThreads(1)
    from deploy.bimanual import port_vision as PV
    RS = runs(a.days)
    jobs, meta = [], {}
    for d, fr in RS:
        ev = events(d)
        prep = [e for e in ev if ev_kind(e) == 'align_prepare']
        dep = [cv2.imread(str(Path(d) / r['file'])) for r in fr if kind_of(r['tag']) == 'prepare']
        dep = [f for f in dep if f is not None]
        z_c = PV.plug_depth(dep) if len(dep) >= 3 else None
        ref = PV.plug_ref(dep) if len(dep) >= 3 else None
        z_log = prep[0].get('plug_z_mm') if prep else None
        acc = prep[0].get('accepted') if prep else None
        z = (z_log if acc else None) if prep else z_c
        # logged per-frame values of the step stage, in frame order
        logged = [f for e in ev if ev_kind(e) == 'align_measure' for f in (e.get('frames') or [])]
        steps = [r for r in fr if kind_of(r['tag']) == 'step']
        meta[Path(d).name] = dict(day=Path(d).parent.name, plug_z_logged=z_log, accepted=acc, plug_z_replay=z_c, plug_z_used=z, ref=ref,
                                  n_frames=len(fr), n_step=len(steps), n_logged=len(logged))
        for r, lg in zip(steps, logged) if len(steps) == len(logged) else ():
            r['logged'] = dict(ok=lg.get('ok'), v=lg.get('v'), reason=lg.get('reason'))
        for i in range(0, len(fr), 12):
            jobs.append((d, fr[i:i + 12], z, ref))
    with mp.get_context('spawn').Pool(a.procs) as p:      # fork after cv2 work in this process deadlocks the workers
        rows = [x for part in p.imap(one, jobs, chunksize=1) for x in part]
    # attach logged values
    lg = {}
    for d, fr in RS:
        for r in fr:
            if 'logged' in r:
                lg[(Path(d).name, r['file'])] = r['logged']
    for r in rows:
        x = lg.get((r['run'], r['file']))
        if x:
            r['log_ok'], r['log_v'], r['log_reason'] = x['ok'], x['v'], x['reason']

    S = summarise(rows, meta)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'port_vision_width.json').write_text(json.dumps(dict(summary=S, runs=meta, rows=rows), indent=1, default=float))
    keys = sorted({k for r in rows for k in r})
    with open(OUT / 'port_vision_width.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    print(json.dumps(S, indent=1, default=float))


def summarise(rows, meta):
    S = dict(n_runs=len(meta), n_frames=len(rows), by_kind={})
    def cnt(rs):
        both = [r for r in rs if r.get('old_ok') and r.get('new_ok')]
        same_rej = [r for r in rs if not r.get('old_ok') and not r.get('new_ok')]
        rec = [r for r in rs if not r.get('old_ok') and r.get('new_ok')]
        newrej = [r for r in rs if r.get('old_ok') and not r.get('new_ok')]
        d = np.array([np.abs(np.subtract(r['new_v'], r['old_v'])) for r in both]) if both else np.zeros((0, 3))
        chg = [r for r in both if r.get('new_fix')]
        return dict(n=len(rs), both_ok=len(both), both_reject=len(same_rej), recovered=len(rec), newly_rejected=len(newrej),
                    both_ok_with_fix=len(chg), max_abs_diff_mm=d.max(0).round(3).tolist() if len(d) else None,
                    identical=int((d.max(1) == 0).sum()) if len(d) else 0)
    S['all'] = cnt(rows)
    for k in sorted({r['kind'] for r in rows}):
        S['by_kind'][k] = cnt([r for r in rows if r['kind'] == k])
    S['recovered'] = [dict(run=r['run'], tag=r['tag'], old=r.get('old_reason'), new_v=np.round(r['new_v'], 2).tolist(), fix=r.get('new_fix'),
                           w_old=r.get('old_w'), w_new=r.get('new_w')) for r in rows if not r.get('old_ok') and r.get('new_ok')]
    S['newly_rejected'] = [dict(run=r['run'], tag=r['tag'], new=r.get('new_reason')) for r in rows if r.get('old_ok') and not r.get('new_ok')]
    S['fix_fired'] = sorted({(r['run'], r.get('new_fix')) for r in rows if r.get('new_fix')})
    reasons = {}
    for r in rows:
        if not r.get('old_ok'):
            k = re.sub(r'[-\d.]+', '#', r.get('old_reason') or r.get('err', ''))
            reasons[k] = reasons.get(k, 0) + 1
    S['old_reject_reasons'] = reasons
    # shell edges against the reference (old detector), frames the old one measured
    dv = np.array([[r['old_ul'] - meta[r['run']]['ref']['ul'], r['old_ur'] - meta[r['run']]['ref']['ur'], r['old_ve'] - meta[r['run']]['ref']['ve']]
                   for r in rows if r.get('old_ok') and meta[r['run']]['ref'] and r.get('old_ul') is not None])
    if len(dv):
        S['edge_dev_px_old_ok'] = dict(n=len(dv), abs_p50=np.percentile(np.abs(dv), 50, 0).round(2).tolist(), abs_p99=np.percentile(np.abs(dv), 99, 0).round(2).tolist(),
                                       abs_max=np.abs(dv).max(0).round(2).tolist(), over_tol=int(((np.abs(dv[:, :2]) > 1.5).any(1) | (np.abs(dv[:, 2]) > 3)).sum()))
    # logged against the old replay (code drift since the run)
    L = [r for r in rows if r.get('log_ok') is not None and r.get('old_ok') is not None]
    agree = [r for r in L if bool(r['log_ok']) == bool(r['old_ok'])]
    dd = np.array([np.abs(np.subtract(r['old_v'], r['log_v'])) for r in L if r['log_ok'] and r['old_ok'] and r.get('log_v') and None not in r['log_v']])
    S['old_vs_logged'] = dict(n=len(L), ok_agree=len(agree), max_abs_diff_mm=dd.max(0).round(3).tolist() if len(dd) else None,
                              p99_abs_diff_mm=np.percentile(dd, 99, 0).round(3).tolist() if len(dd) else None)
    return S


if __name__ == '__main__':
    main()
