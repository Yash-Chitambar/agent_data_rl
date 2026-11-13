"""Align stage time budget from the real step-mode logs, and an offline model of the fast settings (docs/real/ALIGN_FAST_20261113.md). No motion.

    python scripts/real/align_fast_model.py [--json results/real_align/align_fast/model.json] [runs ...]

Per run (wall clock from ticks.npz): prepare, probe cycles, correction cycles, confirm cycles, press + post-press; per cycle the measurement
(rest wait, frames, second source) and the step. The model replays every logged step through deploy/bimanual/align.reach_ok on the FK trace
(the loose reach test), drops the probe cycles (fixed slope prior), recounts the gap cycles with the larger gap steps, and prices each
measurement with the new frame count, frame wait and second-source rule. Limits of the model: an earlier step end changes the next pose
(the next correction makes up for it); the time per frame without the two-frame wait is estimated (port_vision 26 ms in the servo's worker + one
camera frame, 24 frames/s); the gap steps use each run's median step fraction.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402
from deploy.bimanual.align import DEFAULTS, reach_ok  # noqa: E402

RUNS = [r for d in ('2026-11-10', '2026-11-11') for r in sorted(glob.glob(f'/home/yashchitambar/peg_test/data/real_runs/{d}/full_*')) if 'dry' not in r]
FAST = dict(reach_mode='loose', reach_rel=.5, reach_side_mm=.6, reach_hold_ticks=3, reach_frac_min=.6, reach_stall_s=.5, reach_stall_mm=.15, step_timeout_s=2.0,
            frames=3, settle_s=.1, gap_step_ratio=.5, gap_step_max_mm=3.0, fresh_frames='first', secondary_near_mm=1.0)
FRAME_FIRST_S = .087 + .020          # first frame: two-frame wait + port_vision, plus its control tick (median of the logs)
FRAME_NEXT_S = .026 + .021 + .020    # later frames: port_vision (26 ms) + one new camera frame (about 21 ms on average) + the tick


def gap_cycles(g0, target, tol, cap_fn, frac):
    g, n = g0, 0
    while g - target > tol * .5 and n < 60:
        g -= frac * cap_fn(g)
        n += 1
    return n


def run_budget(run, fast):
    if not (Path(run) / 'ticks.npz').exists() or '"align_measure"' not in (Path(run) / 'events.jsonl').read_text():
        return None
    r = L.load(run)
    ev, t, eR = r['ev'], r['t'], r['eR']
    tw = t['t']
    q = t['q'][:, 0] + L.OFF
    qd = np.abs(t['qd'][:, 0]).max(1)
    st = [e for e in ev if e['kind'] == 'align_start']
    if not st or not any(e['kind'] == 'align_measure' for e in ev):
        return None
    t0 = st[0]['t']
    res = [e for e in ev if e['kind'] == 'align_result']
    t_end = res[0]['t'] if res else max(e['t'] for e in ev if e['kind'].startswith('align_'))
    prep = [e for e in ev if e['kind'] == 'align_prepare']
    press = [e for e in ev if e['kind'] == 'align_press_on']
    al = [e for e in ev if e['kind'] in ('align_measure', 'align_step', 'align_press_on', 'align_handover')]
    out = dict(run=Path(run).name, stage_s=t_end - t0, prepare_s=(prep[0]['t'] - t0) if prep else None, result=res[0]['reason'] if res else None)
    cyc = []
    prev_t = prep[0]['t'] if prep else t0
    prev_step_tick = None
    pre_press = True
    for i, e in enumerate(al):
        if e['kind'] == 'align_press_on':
            pre_press = False
            prev_t = e['t']
            continue
        if e['kind'] == 'align_measure':
            nfr = len(e['frames'])
            c = dict(cycle=e['cycle'], pre=pre_press, reason=e['decision_reason'], decision=e['decision'], meas_s=e['t'] - prev_t, frames=nfr,
                     second=e.get('second') is not None, near=bool(e['ok'] and np.all(np.abs(np.array([e['across'], e['up'], e['gap']]) - [0, 0, 6]) <= 2.0)),
                     gap=e['gap'], step_s=0., step=None)
            # rest wait: ticks from the step end (or the previous event) to the first frame tick
            k_end = e['tick']
            k_frames0 = k_end - nfr - (1 if c['second'] else 0)
            c['still_s'] = float(tw[k_frames0] - tw[prev_step_tick]) if prev_step_tick is not None and k_frames0 > prev_step_tick else None
            cyc.append(c)
            prev_t = e['t']
        elif e['kind'] == 'align_step':
            k1, n = e['tick'], e['ticks']
            k0 = k1 - n
            tip, _, _ = L.fk_tip(q[k0:k1 + 1])
            rel = (tip - tip[0]) @ eR * 1e3
            hist, k_new = [], None
            for j in range(1, len(rel)):
                hist.append(rel[j])
                if reach_ok({**DEFAULTS, **fast}, hist, e['delta']):
                    k_new = j
                    break
            n_to = int(round(fast['step_timeout_s'] / .02))
            k_new = min(k_new if k_new is not None else n, n_to)
            # rest after the new end: first still reading after settle_s
            ks = k0 + k_new + int(round(fast['settle_s'] / .02))
            j = ks
            while j < k1 + 75 and j < len(qd) and qd[j] >= DEFAULTS['still_rad_s']:
                j += 1
            d_ = np.asarray(e['delta'], float)
            frac_new = float(rel[k_new] @ d_ / max(d_ @ d_, 1e-9))
            cyc[-1].update(step_s=e['t'] - prev_t, step=dict(asked=e['delta'], reached=e['reached'], ticks=n, frac=e['frac'], ticks_new=int(k_new), frac_new=frac_new,
                                                             reached_new=k_new < n_to and (k_new < n or e['reached']), still_new_s=float((j - (k0 + k_new)) * .02)))
            prev_t = e['t']
            prev_step_tick = k1
    out['cycles'] = cyc
    pre = [c for c in cyc if c['pre']]
    post = [c for c in cyc if not c['pre']]
    out['n_pre'], out['n_probe'] = len(pre), sum(c['reason'] == 'probe' for c in pre)
    out['n_correct'] = sum(c['reason'] not in ('probe',) and c['decision'] == 'step' for c in pre)
    out['n_confirm'] = sum(c['decision'] in ('retry', 'done') for c in pre)
    steps = [c['step'] for c in cyc if c['step']]
    out['n_steps'], out['n_timeout'] = len(steps), sum(not s['reached'] for s in steps)
    out['step_s'] = sum(c['step_s'] for c in cyc)
    out['meas_s'] = sum(c['meas_s'] for c in cyc)
    out['probe_s'] = sum(c['step_s'] + c['meas_s'] for c in pre if c['reason'] == 'probe')
    out['post_s'] = (([e for e in ev if e['kind'] == 'align_handover'] or [dict(t=t_end)])[0]['t'] - press[0]['t'] + 1.0) if press else 0.
    out['meas_med_s'] = float(np.median([c['meas_s'] for c in cyc]))
    out['still_med_s'] = float(np.median([c['still_s'] for c in cyc if c['still_s'] is not None])) if any(c['still_s'] is not None for c in cyc) else None

    # ---- model
    def meas_new(c, still_s):
        sec = c['second'] and (fast['secondary_near_mm'] is None or c['near'])
        fr = FRAME_FIRST_S + (fast['frames'] - 1) * (FRAME_NEXT_S if fast['fresh_frames'] == 'first' else FRAME_FIRST_S)
        return still_s + fr + (FRAME_FIRST_S if sec else 0.)

    stepc = [c for c in cyc if c['step'] and c['reason'] != 'probe']
    c_step = float(np.median([meas_new(c, fast['settle_s']) + max(fast['settle_s'], c['step']['still_new_s']) - fast['settle_s'] + c['step']['ticks_new'] * .02 + .02
                              for c in stepc]))
    c_meas = float(np.median([meas_new(c, fast['settle_s']) for c in cyc]))
    # gap cycles: each run's median gap progress per correction step (old end, new end), entry gap to the target
    corr = [c for c in pre if c['reason'] == 'correct' and c['step'] and abs(c['step']['asked'][2]) > .5]
    fr_old = float(np.median([c['step']['frac'] for c in corr])) if corr else 1.
    fr_new = float(np.median([c['step']['frac_new'] for c in corr])) if corr else 1.
    g0 = next((c['gap'] for c in pre if c['gap'] is not None), 18.)
    old_n = gap_cycles(g0, 6., 1., lambda g: 1., fr_old)
    cap_new = (lambda g: float(np.clip(fast['gap_step_ratio'] * (g - 6.), 1., fast['gap_step_max_mm']))) if fast['gap_step_ratio'] else (lambda g: 1.)
    new_n = gap_cycles(g0, 6., 1., cap_new, fr_new)
    n_corr_new = max(new_n, out['n_correct'] - old_n + new_n)
    n_post_steps = sum(1 for c in post if c['step'])
    n_post_meas = len(post) - n_post_steps
    other = out['stage_s'] - out['step_s'] - out['meas_s'] - (out['prepare_s'] or 0.) - 1.0 * bool(press)
    model = dict(c_step_s=c_step, c_meas_s=c_meas, gap_frac_old=fr_old, gap_frac_new=fr_new, gap_cycles_old=old_n, gap_cycles_new=new_n, n_correct_new=n_corr_new,
                 pre_s=n_corr_new * c_step + out['n_confirm'] * c_meas, post_s=n_post_steps * c_step + n_post_meas * c_meas + 1.0 * bool(press),
                 prepare_s=out['prepare_s'] or 0., other_s=max(other, 0.))
    model['step_s'] = (n_corr_new + n_post_steps) * (c_step - c_meas)
    model['meas_s'] = model['pre_s'] + model['post_s'] - model['step_s'] - 1.0 * bool(press)
    model['stage_s'] = model['pre_s'] + model['post_s'] + model['prepare_s'] + model['other_s']
    # the same accounting with the old settings (checks the model against the log)
    out['model'] = model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--json')
    ap.add_argument('--fast', default='{}', help='JSON overrides of the fast settings')
    a = ap.parse_args()
    fast = {**FAST, **json.loads(a.fast)}
    rows = [x for x in (run_budget(r, fast) for r in (a.runs or RUNS)) if x is not None]
    hdr = 'run                         stage  prep probe(n,s)  corr conf steps(to)  step_s meas_s meas/cyc post | model: step meas corr(gap frac old/new) stage'
    print(hdr)
    for x in rows:
        m = x['model']
        print(f"{x['run'][:27]:27s} {x['stage_s']:5.1f} {x['prepare_s'] or 0:4.1f} {x['n_probe']:2d} {x['probe_s']:5.1f} {x['n_correct']:5d} {x['n_confirm']:4d} {x['n_steps']:3d}({x['n_timeout']:2d}) "
              f"{x['step_s']:7.1f} {x['meas_s']:6.1f} {x['meas_med_s']:6.2f} {x['post_s']:5.1f} | {m['step_s']:5.1f} {m['meas_s']:5.1f} {x['n_correct']}->{m['n_correct_new']} ({m['gap_frac_old']:.2f}/{m['gap_frac_new']:.2f}) {m['stage_s']:5.1f}")
    st = np.array([x['stage_s'] for x in rows])
    ms = np.array([x['model']['stage_s'] for x in rows])
    steps = [s for x in rows for c in x['cycles'] if (s := c['step'])]
    print(f'n = {len(rows)} runs; stage wall s p50 {np.median(st):.1f} (range {st.min():.1f}-{st.max():.1f}); model p50 {np.median(ms):.1f} (range {ms.min():.1f}-{ms.max():.1f})')
    print(f'steps n = {len(steps)}: timeouts {sum(not s["reached"] for s in steps)}; with the loose test {sum(not s["reached_new"] for s in steps)}; '
          f'step ticks p50 {np.median([s["ticks"] for s in steps]):.0f} -> {np.median([s["ticks_new"] for s in steps]):.0f}')
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(fast=fast, runs=rows), indent=1, default=float))


if __name__ == '__main__':
    main()
