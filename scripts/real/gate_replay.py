"""Gate replay: every gate and stop limit of the deploy graph, evaluated on every logged real full-chain run (2026-11-03 .. 2026-11-11). No motion, CPU.

    PYTHONPATH=. python scripts/real/gate_replay.py [--root /home/yashchitambar/peg_test/data/real_runs] [--out results/real_runs_review]

Configs: deploy/config/bimanual_align5_k4_push15_seat6_fastmove.json ("base", the milestone config) and deploy/config/bimanual_align9_all.json ("all9",
the combined candidate). Code: deploy/bimanual/{align,insert,insert_guard,guards,chain}.py, blind_insertion/perception/zed_estimator.py.

For every gate with a logged value: value, limit, margin (positive = inside the limit, in the gate's units), fired (the limit would stop the run, or
skip the push), stopped (the logged run did stop there), fine (the run's stage was fine by vision or by the operator; labels below), and the
missed-stop test (gate passed, but the run shows a failure that this gate is meant to catch). Each gate is marked "safety" (protects the hardware or
people) or "quality" (protects the outcome).

Outputs: gate_replay.json (summary per gate and config, rows per run) and gate_replay.csv (one row per gate x config x run).
Limits of the replay: values are the logged ones (FK in the believed frame, PD torque estimates, port vision), so a gate that depends on motion
the new config changes (fast align steps, the force push, the no-jump approach) is replayed on the old motion only, and is marked so.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))

DT = .02
DATA = '/home/yashchitambar/peg_test/data/real_runs'
DATES = ('2026-11-03', '2026-11-06', '2026-11-10', '2026-11-11')
CONFIGS = dict(base='deploy/config/bimanual_align5_k4_push15_seat6_fastmove.json', all9='deploy/config/bimanual_align9_all.json')
ZED_OOD_SCALE = 1.25          # blind_insertion/perception/zed_estimator.py OOD_SCALE

# ---------------------------------------------------------------------------------------------------------------------------- labels
# From the session documents (SESSION_20261103/06/10/11_duchamp3.md) and scripts/real/review_runs.py OPERATOR.
# align: the align stage was fine by vision (hand-over within tolerance, reader held), None = not reached or not judged.
# insert: the plug was in the port by the operator (also when the run was then stopped), None = not reached.
# full: operator said fully in (True), short or hanging (False), not stated (None).
# tags: failures a gate may be meant to catch.
LAB = {
    'full_144822_measure1': dict(align=None, insert=None, full=None, tags=[], note='measure mode, no align motion'),
    'full_151410_measure1': dict(align=None, insert=None, full=None, tags=[], note='connected and quit, no motion'),
    'full_151615_align1': dict(align=True, insert=None, full=None, tags=[], note='scale gate stop, position and angle in tolerance (doc: stop looks false)'),
    'full_152618_align2': dict(align=True, insert=None, full=None, tags=[], note='aligned, align mode'),
    'full_153808_align_full1': dict(align=False, insert=None, full=None, tags=['reader_slip'], note='grip gate: reader moving in the jaws (doc: gate doing its job)'),
    'full_155211_align_full2': dict(align=True, insert=True, full=True, tags=[], note='seated, fully in'),
    'full_162735_align_full3': dict(align=False, insert=None, full=None, tags=['reader_slip'], note='reader lost in the probes'),
    'full_163516_align_full3': dict(align=False, insert=None, full=None, tags=['reader_slip'], note='reader lost in 5 drops'),
    'full_164427_align_full3': dict(align=None, insert=None, full=None, tags=[], note='no stage events'),
    'full_164445_align_full4': dict(align=None, insert=None, full=None, tags=[], note='pick only'),
    'full_164912_align_full5': dict(align=True, insert=False, full=None, tags=['stall_on_case'], note='good hand-over, no seat, guard bound'),
    'full_154548_align_prec1': dict(align=None, insert=None, full=None, tags=['software'], note='software crash in hold'),
    'full_155102_align_prec2': dict(align=None, insert=None, full=None, tags=['table_contact'], note='controller in the pick, gripper on the table, safety stop'),
    'full_160310_align_prec3': dict(align=None, insert=None, full=None, tags=['software'], note='capture stop (bug)'),
    'full_160705_align_prec4': dict(align=True, insert=None, full=None, tags=[], note='aligned, align mode'),
    'full_161426_align_prec5': dict(align=True, insert=None, full=None, tags=[], note='aligned, align mode'),
    'full_162512_align_prec6': dict(align=False, insert=None, full=None, tags=['stuck'], note='stalled at step 9 (joint 2 outside the window)'),
    'full_163339_align_prec7': dict(align=True, insert=None, full=None, tags=[], note='aligned, align mode'),
    'full_164558_align_full7': dict(align=True, insert=None, full=None, tags=[], note='disagree stop 4.49 mm up; vision in tolerance (net bias, ROBUSTNESS_PLAN part B)'),
    'full_165244_align_full8': dict(align=True, insert=False, full=None, tags=['climb'], note='K-J climbed 9.7 mm, guard stop'),
    'full_173941_align_full9': dict(align=True, insert=True, full=True, tags=[], note='seated, fully in'),
    'full_175030_align_full10': dict(align=True, insert=False, full=None, tags=['stall_on_case'], note='plug stalled on the case, guard bound'),
    'full_143002_a4p_1': dict(align=None, insert=None, full=None, tags=[], note='start aborted, no events'),
    'full_143024_a4p_1': dict(align=True, insert=None, full=None, tags=[], note='aligned, align mode (precision controller)'),
    'full_144059_servo_1': dict(align=None, insert=None, full=None, tags=['software'], note='software stop at the align start'),
    'full_144605_servo_1': dict(align=True, insert=None, full=None, tags=[], note='servo aligned'),
    'full_145249_servo_2': dict(align=True, insert=None, full=None, tags=[], note='servo aligned'),
    'full_145824_kjrc_1': dict(align=True, insert=True, full=False, tags=['short'], note='seated, hangs'),
    'full_150628_kjrc_2': dict(align=True, insert=True, full=False, tags=['short'], note='seated, about 1 mm short'),
    'full_152149_kjrc_3_push': dict(align=True, insert=True, full=False, tags=['short'], note='seated, about 3 mm short'),
    'full_152717_k4_1': dict(align=True, insert=True, full=False, tags=['short'], note='seated, about 3 mm short'),
    'full_154422_k4_2': dict(align=True, insert=True, full=None, tags=['not_detected'], note='plug in, seat not detected, guard pulled it out'),
    'full_155541_k4_3': dict(align=True, insert=True, full=True, tags=[], note='seated, really well (counted fully in, FULL_DEPTH)'),
    'full_160352_k4_4': dict(align=True, insert=None, full=None, tags=[], note='disagree stop 3.54 mm up; vision in tolerance'),
    'full_155351_demo_1': dict(align=True, insert=True, full=False, tags=['short', 'push_skipped'], note='seated, hanging slightly, push skipped'),
    'full_160310_demo_2': dict(align=True, insert=True, full=None, tags=[], note='seated'),
    'full_160801_demo_3': dict(align=None, insert=None, full=None, tags=['pick_empty'], note='empty close'),
    'full_161223_demo_3': dict(align=True, insert=True, full=None, tags=[], note='seated'),
    'full_161705_demo_4': dict(align=True, insert=True, full=True, tags=[], note='fully inserted'),
    'full_162246_demo_5': dict(align=True, insert=None, full=None, tags=[], note='disagree stop 3.51 mm up; vision in tolerance'),
    'full_163144_speed075_1': dict(align=True, insert=True, full=None, tags=['push_skipped'], note='success, push skipped'),
    'full_171905_fastmove_1': dict(align=True, insert=True, full=True, tags=[], note='inserted perfectly'),
}
# Runs of 2026-11-10 with empty folders on mason: values copied from SESSION_20261110_duchamp3.md (via review_runs.DOC_ONLY), only where a gate value is stated.
DOC = {
    'full_161011_k4_4': dict(lab=dict(align=True, insert=True, full=None, tags=[], note='seated; push stopped by the jaw watch (doc only)'),
                             push_grip_change=0.020, push_left_mm=0.15, push_stopped=True, disagree_up=2.61),
    'full_162427_k4_5': dict(lab=dict(align=None, insert=None, full=None, tags=['hand_jump'], note='post-press step limit, hand moved 2.8 mm by itself (doc only)'),
                             post_press_fired=True),
    'full_163515_k4_5': dict(lab=dict(align=True, insert=True, full=True, tags=[], note='flush and seated (doc only)'), push_grip_change=0.009, push_left_mm=0.07),
    'full_170521_k4_6': dict(lab=dict(align=True, insert=True, full=None, tags=[], note='seated (doc only)'), push_grip_change=0.011, push_left_mm=0.25),
}


def cfg_load(path):
    c = json.load(open(ROOT / path))
    from deploy.bimanual.align import align_cfg
    return c, align_cfg(c)


def r2(x, n=3):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(x) else round(x, n)


def npz(run, name):
    p = os.path.join(run, name + '.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


# ---------------------------------------------------------------------------------------------------------------------------- per-run extraction
def extract(run, L):
    """All gate inputs of one run (logged values only)."""
    name = os.path.basename(run)
    evp = os.path.join(run, 'events.jsonl')
    ev = [json.loads(x) for x in open(evp)] if os.path.exists(evp) and os.path.getsize(evp) else []
    R = dict(run=name, date=Path(run).parent.name, lab=LAB.get(name, dict(align=None, insert=None, full=None, tags=[], note='')), n_events=len(ev))
    if not ev:
        if name in DOC:
            R.update(source='doc', lab=DOC[name]['lab'], doc=DOC[name])
        else:
            R['source'] = 'none'
        return R
    R['source'] = 'logs'
    meta = json.load(open(os.path.join(run, 'meta.json')))
    R['config'] = Path(meta['args'].get('config', '?')).stem
    R['stop_mm'] = meta['config'].get('chain_args', {}).get('stop')
    by = {}
    for e in ev:
        by.setdefault(e['kind'], []).append(e)
    R['stopped'] = [e['reason'] for e in by.get('stopped', [])]
    st = {e['stage']: e['tick'] for e in by.get('stage', [])}
    R['stages'] = st
    for e in by.get('estimate', []):
        if e.get('est') == 'zed_pc':
            R['zed'] = dict(maha=e.get('mahalanobis'), presence=e.get('presence'), arm_ok=e.get('arm_ok'), ok=e.get('ok'), yaw=e.get('yaw_deg'))
        if e.get('est') == 'zed_reader':
            R['zed_reader'] = dict(ok=e.get('ok'), tries=e.get('tries'))
    po = by.get('pick_outcome', [None])[-1]
    R['pick'] = po and po['outcome']
    g = npz(run, 'grasp')
    if g is not None and len(g['reading']):
        R['pick_reading'] = float(g['reading'][-1])
    hr = by.get('hold_result', [None])[-1]
    if hr:
        R['hold'] = dict(latched=hr['latched'], Fz=hr['F_est'][2])
    hd = npz(run, 'hold')
    if hd is not None and len(hd['pressing']):
        R['press_s'] = float(hd['pressing'].sum() * DT)
    pc = by.get('preinsert_check', [None])[-1]
    if pc:
        R['preinsert_lat_mm'] = pc['fk_lateral_mm']
    tk = npz(run, 'ticks')
    if tk is not None and len(tk['k']):
        tau = np.abs(tk['tau_est'])                       # (n, 2, 6) PD torque estimate as the guards compute it
        n5 = 5
        if len(tau) >= n5:
            win = np.lib.stride_tricks.sliding_window_view(tau, n5, axis=0).min(-1)      # sustained over tau_peak_s = 0.1 s
            R['tau_peak_sust'] = win.max(0)               # (2, 6)
        R['tau_abs_max'] = tau.max(0)
        R['tau_cont_s'] = {}                               # filled per config (limit dependent)
        R['_tau'] = tau
        R['qd_max'] = float(np.abs(tk['qd']).max())
    # ---- align (step stage)
    ms = by.get('align_measure', [])
    steps = by.get('align_step', [])
    press = by.get('align_press_on', [None])[0]
    R['align_mode'] = 'servo' if ('align_servo_end' in by or 'align_servo_frame' in by) else ('step' if ms else None)
    ar = by.get('align_result', [None])[-1]
    R['align_result'] = ar and dict(ok=ar['ok'], reason=ar['reason'], cycles=ar.get('cycles'), seconds=ar.get('seconds'), slope=ar.get('slope'))
    R['align_ms'] = ms
    R['align_steps'] = steps
    R['press_t'] = press and press['t']
    R['handover'] = by.get('align_handover', [None])[-1]
    # ---- insertion
    R['insert_result'] = by.get('insert_result', [None])[-1]
    R['seated_ev'] = by.get('seated_detected', [None])[-1]
    R['push'] = by.get('insert_final_push', [None])[-1]
    R['push_skip'] = by.get('insert_final_push_skipped', [None])[-1]
    R['insert_stop'] = by.get('insert_stop', [None])[-1]
    gd = npz(run, 'guard')
    R['guard'] = gd if gd is not None and len(gd['k']) else None
    pol = npz(run, 'policy')
    R['policy_npz'] = pol if pol is not None and len(pol['k']) else None
    R['policy'] = 'K4' if 'insert_vision_prior' in by else ('K-J' if 'insert_start' in by else None)
    return R


# ---------------------------------------------------------------------------------------------------------------------------- helpers on the align log
def mvec(m):
    return None if not m.get('ok') else np.array([m['across'], m['up'], m['gap']], float)


def bad_reason(m, A):
    if not m.get('ok'):
        return 'no measurement'
    if m.get('conf', 1.) < A['min_conf']:
        return 'conf'
    if m.get('sigma') is not None and m['sigma'] > A['max_sigma_mm']:
        return 'sigma'
    if m.get('spread', 0.) > A['frame_spread_mm']:
        return 'spread'
    return None


def phases(R):
    """Split the align measures into the pre-press and post-press loops."""
    ms = R.get('align_ms') or []
    pt = R.get('press_t')
    pre = [m for m in ms if pt is None or m['t'] < pt]
    post = [m for m in ms if pt is not None and m['t'] > pt]
    return pre, post


def slope_se(levels, A):
    """deploy/bimanual/align.AlignLogic.slope_se on (tip, measurement) levels."""
    if len(levels) < 3:
        return [None] * 3, [float('inf')] * 3
    Ar = np.asarray(levels, float)
    sl, se = [], []
    for i in range(3):
        x, y = Ar[:, i] - Ar[:, i].mean(), Ar[:, i + 3] - Ar[:, i + 3].mean()
        sxx = float(x @ x)
        if sxx < 1e-6:
            sl.append(None), se.append(float('inf'))
            continue
        b = float(x @ y) / sxx
        res = y - b * x
        sd = max(float(np.sqrt(res @ res / max(len(x) - 2, 1))), A['slope_noise_floor_mm'][i])
        sl.append(b), se.append(sd / np.sqrt(sxx))
    return sl, se


def levels_until_done(pre):
    lv = []
    for m in pre:
        if bad_reason(m, dict(min_conf=.5, max_sigma_mm=1.5, frame_spread_mm=1.5)) is None:
            lv.append(np.r_[np.asarray(m['tip_rel'], float), mvec(m)])
        if m.get('decision') == 'done':
            break
    return lv


def check_points(R):
    """Measurements where the hand-over checks ran: decision done, or a fail at the disagreement / scale gate."""
    out = []
    for m in R.get('align_ms') or []:
        dr = str(m.get('decision_reason') or '')
        if m.get('decision') == 'done' or 'sources disagree' in dr or 'scale gate' in dr:
            out.append(m)
    return out


# ---------------------------------------------------------------------------------------------------------------------------- gates
class G:
    def __init__(self, gid, stage, kind, reason, limit_key, unit, fn, stage_fine='align', catches=(), stop_match=None, note='', comparable=None):
        self.gid, self.stage, self.kind, self.reason, self.limit_key, self.unit = gid, stage, kind, reason, limit_key, unit
        self.fn, self.stage_fine, self.catches, self.stop_match, self.note, self.comparable = fn, stage_fine, tuple(catches), stop_match, note, comparable


def fine_of(R, which):
    lab = R['lab']
    if which == 'none':
        return None
    if which == 'capture':
        ms = R.get('align_ms') or []
        if R.get('align_mode') == 'servo' or R.get('handover'):
            return True
        if ms and ms[0].get('ok'):
            return bool(abs(ms[0]['across']) < 10 and abs(ms[0]['up']) < 10)
        return None
    if which == 'pick':
        return None if R.get('pick') is None else R['pick'] == 'held'
    if which == 'full':
        return lab.get('full')
    return lab.get(which)


def g_zed_ood(R, C, A):
    z = R.get('zed')
    if not z or z.get('maha') is None:
        return None
    lim = C['_zed_ood']
    return dict(value=z['maha'], limit=lim, margin=lim - z['maha'], fired=z['maha'] > lim)


def g_zed_presence(R, C, A):
    z = R.get('zed')
    if not z or z.get('presence') is None:
        return None
    return dict(value=z['presence'], limit=.5, margin=z['presence'] - .5, fired=z['presence'] < .5)


def g_pick_band(R, C, A):
    if R.get('pick_reading') is None:
        return None
    lo, hi = C['pick']['grasp_v3']['held_band']
    v = R['pick_reading']
    return dict(value=v, limit=[lo, hi], margin=min(v - lo, hi - v), fired=not (lo <= v <= hi))


def g_hold_time(R, C, A):
    if R.get('press_s') is None:
        return None
    lim = C['safety']['hold_max_s']
    return dict(value=R['press_s'], limit=lim, margin=lim - R['press_s'], fired=R['press_s'] > lim,
                note='total press time over the run; the guard counter may reset between presses, so this is an upper bound')


def g_tau_peak(R, C, A):
    if R.get('tau_peak_sust') is None:
        return None
    lim = np.asarray(C['safety']['tau_peak_nm'], float)
    r = R['tau_peak_sust'] / lim
    s, j = np.unravel_index(int(r.argmax()), r.shape)
    return dict(value=float(R['tau_peak_sust'][s, j]), limit=float(lim[j]), margin=float(lim[j] - R['tau_peak_sust'][s, j]), fired=bool((r > 1).any()),
                where=f'{("right", "left")[s]} joint {j + 1}')


def g_tau_cont(R, C, A):
    if R.get('_tau') is None:
        return None
    lim = np.asarray(C['safety']['tau_cont_nm'], float)
    t = (R['_tau'] > lim).sum(0) * DT                     # (2, 6) seconds above, whole run
    s, j = np.unravel_index(int(t.argmax()), t.shape)
    L = C['safety']['tau_cont_s']
    return dict(value=float(t[s, j]), limit=L, margin=float(L - t[s, j]), fired=bool((t > L).any()), where=f'{("right", "left")[s]} joint {j + 1}')


def g_qd(R, C, A):
    if R.get('qd_max') is None:
        return None
    lim = C['safety']['qd_max_rad_s']
    return dict(value=R['qd_max'], limit=lim, margin=lim - R['qd_max'], fired=R['qd_max'] > lim)


def g_preinsert(R, C, A):
    if R.get('preinsert_lat_mm') is None:
        return None
    lim = C['insert'].get('wrong_port_mm')
    v = R['preinsert_lat_mm']
    return dict(value=v, limit=lim, margin=None if lim is None else lim - v, fired=False if lim is None else v > lim, note='log only (limit null)')


def g_entry_gap(R, C, A):
    ms = R.get('align_ms') or []
    if not ms or A.get('entry_min_gap_mm') is None:
        return None
    m = ms[0]
    if not m.get('ok'):
        return None
    lim = A['entry_min_gap_mm']
    return dict(value=m['gap'], limit=lim, margin=m['gap'] - lim, fired=m['gap'] < lim, comparable=R.get('stop_mm') == C['chain_args']['stop'])


def g_measure_quality(R, C, A):
    ms = R.get('align_ms') or []
    if not ms:
        return None
    run_, worst, spreads = 0, 0, []
    for m in ms:
        b = bad_reason(m, A)
        run_ = run_ + 1 if b else 0
        worst = max(worst, run_)
        if m.get('ok'):
            spreads.append(m.get('spread', 0.))
    lim = A['max_no_measure']
    return dict(value=worst, limit=lim, margin=lim - worst, fired=worst >= lim, extra=dict(spread_max=r2(max(spreads) if spreads else None), n_bad=sum(bad_reason(m, A) is not None for m in ms)))


def g_frame_spread(R, C, A):
    ms = [m for m in (R.get('align_ms') or []) if m.get('ok')]
    if not ms:
        return None
    v = max(m.get('spread', 0.) for m in ms)
    lim = A['frame_spread_mm']
    run_, worst = 0, 0
    for m in R['align_ms']:
        run_ = run_ + 1 if (m.get('ok') and m.get('spread', 0.) > lim) else 0
        worst = max(worst, run_)
    return dict(value=v, limit=lim, margin=lim - v, fired=worst >= A['max_no_measure'], extra=dict(n_over=sum(m.get('spread', 0.) > lim for m in ms), n=len(ms)),
                note='a single frame spread over the limit only retries; the stop needs max_no_measure in a row')


def disagree_vals(R, A, bias):
    pts = check_points(R)
    out = []
    for m in pts:
        sec = m.get('second') or {}
        if not sec.get('ok') or not m.get('ok'):
            continue
        d = np.asarray(sec['v'], float) - np.asarray(bias, float) - mvec(m)
        out.append(d)
    return out


def g_disagree(R, C, A):
    bias = (A.get('secondary') or {}).get('bias_mm', [0., 0., 0.])
    lim = np.asarray(A['disagree_mm'], float)
    if R.get('source') == 'doc' and R['doc'].get('disagree_up') is not None:
        v = R['doc']['disagree_up'] - bias[1]
        return dict(value=v, limit=float(lim[1]), margin=float(lim[1] - abs(v)), fired=abs(v) > lim[1], axis='up', note='doc value (net - vision up)')
    ds = disagree_vals(R, A, bias)
    if not ds:
        return None
    marg = [lim - np.abs(d) for d in ds]
    k = int(np.argmin([mg.min() for mg in marg]))
    ax = int(np.argmin(marg[k]))
    return dict(value=float(ds[k][ax]), limit=float(lim[ax]), margin=float(marg[k][ax]), fired=bool(any((np.abs(d) > lim).any() for d in ds)), axis=('across', 'up', 'gap')[ax],
                extra=dict(net_minus_vis=[r2(x, 2) for x in ds[k]], n_checks=len(ds)))


def g_scale(R, C, A):
    pre, _ = phases(R)
    if not pre or not any(m.get('decision') == 'done' or 'scale gate' in str(m.get('decision_reason')) for m in pre):
        return None
    lv = levels_until_done(pre)
    sl, se = slope_se(lv, A)
    lo, hi = A['slope_min'], A['slope_max']
    if A.get('slope_prior'):
        known = list(A['slope_prior'].get('known_axes', [2]))
        fired, why = False, []
        for i in range(3):
            if sl[i] is None:
                if i in known:
                    why.append(f'{i}: no motion')
                continue
            if se[i] <= A['slope_se_max']:
                if not lo <= sl[i] <= hi:
                    fired = True
            elif i in known:
                why.append(f'{i}: se {se[i]:.2f}')
            elif sl[i] + 2 * se[i] < lo or sl[i] - 2 * se[i] > hi:
                fired = True
        note = 'replayed on the logged levels (they include the probe motion; all9 has no probes)' + (f'; would ask extra probes: {why}' if why else '')
    else:
        fired = any(s is not None and se_ <= A['slope_se_max'] and not lo <= s <= hi for s, se_ in zip(sl, se))
        note = 'levels recomputed from align_measure tip_rel and vision'
    ok = [(i, s) for i, s in enumerate(sl) if s is not None]
    if not ok:
        return None
    marg = [(min(s - lo, hi - s), i, s) for i, s in ok]
    mg, i, s = min(marg)
    return dict(value=s, limit=[lo, hi], margin=mg, fired=bool(fired), axis=('across', 'up', 'gap')[i],
                extra=dict(slope=[r2(x) for x in sl], se=[r2(x) for x in se], logged=R['align_result'] and R['align_result'].get('slope')), note=note)


def g_step_limit(R, C, A):
    pre, _ = phases(R)
    if not pre:
        return None
    n = len(pre)
    lim = A['max_cycles']
    return dict(value=n, limit=lim, margin=lim - n, fired=n > lim)


def g_post_press(R, C, A):
    if R.get('source') == 'doc' and R['doc'].get('post_press_fired'):
        return dict(value=A['post_press_cycles'] + 1, limit=A['post_press_cycles'], margin=-1, fired=True, note='doc: step limit after the press')
    _, post = phases(R)
    if not post:
        return None
    n = len(post)
    lim = A['post_press_cycles']
    return dict(value=n, limit=lim, margin=lim - n, fired=n > lim)


def g_time(R, C, A):
    ar = R.get('align_result')
    if not ar or ar.get('seconds') is None or R.get('align_mode') != 'step':
        return None
    lim = A['max_time_s']
    return dict(value=ar['seconds'], limit=lim, margin=lim - ar['seconds'], fired=ar['seconds'] > lim, note='stage tick time of the pre-press loop')


def g_tip_lat(R, C, A):
    ms = R.get('align_ms') or []
    if not ms:
        return None
    t = np.array([m['tip_rel'] for m in ms], float)
    v = float(np.abs(t[:, :2]).max())
    lim = A['max_lat_mm']
    return dict(value=v, limit=lim, margin=lim - v, fired=v > lim)


def g_tip_adv(R, C, A):
    ms = R.get('align_ms') or []
    if not ms:
        return None
    t = np.array([m['tip_rel'] for m in ms], float)
    v = float((-t[:, 2]).max())
    lim = A['max_advance_mm']
    return dict(value=v, limit=lim, margin=lim - v, fired=v > lim)


def g_grip(R, C, A):
    ms = [m for m in (R.get('align_ms') or []) if m.get('grip') is not None]
    if not ms:
        return None
    g0 = ms[0]['grip']
    v = max(abs(m['grip'] - g0) for m in ms)
    lim = A['grip_change']
    return dict(value=v, limit=lim, margin=lim - v, fired=v > lim)


def loop_pairs(R, A):
    """(prev, cur) pairs of good measurements inside one loop (the post-press loop resets last_m)."""
    out = []
    for loop in phases(R):
        prev = None
        for m in loop:
            if bad_reason(m, A) is not None:
                continue
            if prev is not None:
                out.append((prev, m))
            prev = m
    return out


def g_drift(R, C, A):
    pairs = loop_pairs(R, A)
    if not pairs:
        return None
    vals, run_, worst = [], 0, 0
    for p, m in pairs:
        if p.get('decision') == 'retry':          # no command since the last good measurement (confirm or retry)
            d = float(np.linalg.norm(np.asarray(m['tip_rel'], float) - np.asarray(p['tip_rel'], float)))
            vals.append(d)
            run_ = run_ + 1 if d > A['drift_mm'] else 0
            worst = max(worst, run_)
    if not vals:
        return None
    v = max(vals)
    return dict(value=v, limit=A['drift_mm'], margin=A['drift_mm'] - v, fired=worst >= A['drift_n'], extra=dict(n=len(vals), n_over=int(sum(x > A['drift_mm'] for x in vals))),
                note='hand motion between two measurements with no command (confirm cycles); stop after drift_n in a row')


def g_pred(R, C, A):
    pairs = loop_pairs(R, A)
    if not pairs:
        return None
    worst_v, run_, worst = -float('inf'), 0, 0
    for p, m in pairs:
        dtip = np.asarray(m['tip_rel'], float) - np.asarray(p['tip_rel'], float)
        dm = mvec(m) - mvec(p)
        v = float(np.abs(dm - dtip).max()) - 2 * float(m.get('sigma') or 0.)      # against pred_mm (the limit per cycle is pred_mm + 2 sigma)
        worst_v = max(worst_v, v)
        run_ = run_ + 1 if v > A['pred_mm'] else 0
        worst = max(worst, run_)
    return dict(value=worst_v, limit=A['pred_mm'], margin=A['pred_mm'] - worst_v, fired=worst >= A['pred_n'],
                note='value = largest |measured change - FK change| minus 2 sigma of that cycle; stop after pred_n in a row')


def g_min_clear(R, C, A):
    ms = [m for m in (R.get('align_ms') or []) if m.get('ok')]
    if not ms:
        return None
    v = min(min(m['gap'], (m.get('fused') or [0, 0, m['gap']])[2]) for m in ms)
    lim = A['min_clear_fail_mm'] if A.get('min_clear_fail_mm') is not None else A['min_clear_mm']
    return dict(value=v, limit=lim, margin=v - lim, fired=v < lim)


def g_angle(R, C, A):
    ms = [m for m in (R.get('align_ms') or []) if m.get('rot_applied') is not None]
    if not ms or not A.get('angle'):
        return None
    v = max(float(np.abs(m['rot_applied']).max()) for m in ms)
    lim = A['angle']['max_deg']
    return dict(value=v, limit=lim, margin=lim - v, fired=v > lim)


def g_stuck(R, C, A):
    st = R.get('align_steps') or []
    if not st:
        return None
    run_, worst, fr = 0, 0, []
    for s in st:
        stuck = (not s['reached']) and s['frac'] < .25
        fr.append(s['frac'])
        run_ = run_ + 1 if stuck else 0
        worst = max(worst, run_)
    lim = A['stuck_n']
    return dict(value=worst, limit=lim, margin=lim - worst, fired=worst >= lim, extra=dict(n_steps=len(st), n_low=int(sum((not s['reached']) and s['frac'] < .25 for s in st))))


def g_step_timeouts(R, C, A, L=None):
    """Step end: strict (0.3 mm, 3 s) as logged, or the loose reach test of all9 replayed on the FK trace of each logged step."""
    st = R.get('align_steps') or []
    if not st or R.get('_q') is None:
        return None
    from deploy.bimanual.align import reach_ok
    n_to = int(round(A['step_timeout_s'] / DT))
    n_out, ticks = 0, []
    for s in st:
        k1, n = s['tick'], s['ticks']
        idx = np.searchsorted(R['_k'], [k1 - n, k1])
        q = R['_q'][idx[0]:idx[1] + 1]
        if len(q) < 2:
            continue
        tip, _, _ = L.fk_tip(q)
        rel = (tip - tip[0]) @ R['_eR'] * 1e3
        hist, kn = [], None
        for j in range(1, len(rel)):
            hist.append(rel[j])
            if reach_ok(A, hist, s['delta']):
                kn = j
                break
        if A.get('reach_mode', 'strict') != 'loose':
            kn = None if not s['reached'] else n
        reached = kn is not None and kn < n_to and (kn < n or s['reached'])
        n_out += not reached
        ticks.append(min(kn if kn is not None else n, n_to))
    if not ticks:
        return None
    frac = n_out / len(ticks)
    return dict(value=n_out, limit=None, margin=None, fired=False, extra=dict(n_steps=len(ticks), timeouts=n_out, timeout_frac=r2(frac), step_ticks_p50=r2(np.median(ticks), 1),
                                                                              step_s_total=r2(sum(ticks) * DT, 1)),
                note='not a stop: step-end test (timeout count). all9 replays the loose reach test and the 2 s timeout on the logged FK')


def seat_replay(R, C):
    """SeatDetector2 with the config's seat2 values on the logged guard trace; returns per-tick dicts (seated, deep_ok, depth, lat)."""
    import torch
    from deploy.bimanual.insert_guard import SeatDetector2
    gd, ho = R.get('guard'), R.get('handover')
    if gd is None or not ho or ho.get('gap_mm') is None:
        return None
    s2 = C['insert']['seat2']
    det = SeatDetector2(1, 'cpu', ho['gap_mm'], ho['lat_mm'], seat_depth_mm=5.8, margin_mm=s2['margin_mm'], gate_mm=s2['gate_mm'], hold_s=s2['hold_s'],
                        speed_mm_s=s2['speed_mm_s'], still_mm=s2['still_mm'], dt=DT)
    out = []
    for rm in gd['rel_m']:
        o = det.update(torch.as_tensor(rm, dtype=torch.float32)[None])
        out.append(dict(seated=bool(o['seated'][0]), deep_ok=bool(det.deep_ok[0]), depth=float(o['depth'][0]), lat=o['lat'][0].numpy().astype(float), speed=float(o['speed'][0])))
    return out


def g_seat(R, C, A):
    sr = seat_replay(R, C)
    if sr is None:
        return None
    first = next((i for i, x in enumerate(sr) if x['seated']), None)
    dmax = max(x['depth'] for x in sr)
    need = 5.8 - C['insert']['seat2']['margin_mm']
    return dict(value=dmax, limit=need, margin=dmax - need, fired=first is None, extra=dict(seat_tick=first, seat_s=r2(first * DT if first is not None else None, 2),
                                                                                       logged_seat_step=R['seated_ev'] and R['seated_ev']['step'], n_ticks=len(sr)),
                note='fired = no seat detected in the logged trace (the trace ends at the logged seat or stop, so a later seat is not visible)')


def g_guard_bound(R, C, A):
    gd = R.get('guard')
    sr = seat_replay(R, C)
    if gd is None or sr is None:
        return None
    gc = C['insert']['guard']
    lim = float(gc['bound_mm'][0]) + float(gc['hard_mm'])
    lat = np.abs(gd['rel_m'][:, :2]).max(1)
    exempt = np.array([x['seated'] or x['deep_ok'] for x in sr])
    v = float(lat[~exempt].max()) if (~exempt).any() else 0.
    return dict(value=v, limit=lim, margin=lim - v, fired=v >= lim, extra=dict(lat_max_all=r2(lat.max()), clamped=int(gd['clamped'][-1])),
                note='|across|, |up| of the FK tip against the hand-over, ticks not exempt (seated or at depth on the axis)')


def g_press_abort(R, C, A):
    gd = R.get('guard')
    sr = seat_replay(R, C)
    if gd is None or sr is None:
        return None
    gc = C['insert']['guard']
    n_stall, n_min = int(round(gc['stall_s'] / DT)), int(round(gc['t_min_s'] / DT))
    best, tb, worst, fired = 0., 0, 0, False
    for t, (rm, rt) in enumerate(zip(gd['rel_m'], gd['rel_t']), 1):
        adv = -rm[2]
        if adv > best + gc['stall_eps_mm']:
            best, tb = adv, t
        lead_in = -(rt[2] - rm[2])
        if lead_in >= gc['lead_min_mm'] and t >= n_min and not sr[t - 1]['seated']:
            worst = max(worst, t - tb)
            fired = fired or (t - tb >= n_stall)
    return dict(value=worst * DT, limit=gc['stall_s'], margin=gc['stall_s'] - worst * DT, fired=fired, note='longest time with no axial progress while the target pushes in')


def g_insert_time(R, C, A):
    ir = R.get('insert_result')
    if not ir:
        return None
    v = ir['steps'] * DT
    lim = C['insert']['insert_s']
    return dict(value=v, limit=lim, margin=lim - v, fired=v >= lim)


def push_point(R, C):
    gd = R.get('guard')
    if gd is None or not R.get('seated_ev'):
        return None
    return float(gd['depth2_mm'][-1]), np.asarray(gd['lat2_mm'][-1], float)


def g_push_depth(R, C, A):
    pp = push_point(R, C)
    if pp is None:
        return None
    lim = C['insert']['final_push']['min_depth_mm']
    return dict(value=pp[0], limit=lim, margin=pp[0] - lim, fired=pp[0] < lim)


def g_push_lat(R, C, A):
    pp = push_point(R, C)
    if pp is None:
        return None
    lim = C['insert']['final_push']['max_lat_mm']
    v = float(np.abs(pp[1]).max())
    gd = R['guard']
    d, l2 = gd['depth2_mm'], gd['lat2_mm']
    deep = np.where(d >= C['insert']['final_push']['min_depth_mm'])[0][-25:]
    dm = float(np.abs(np.median(l2[deep], 0)).max()) if len(deep) else None
    return dict(value=v, limit=lim, margin=lim - v, fired=v > lim, extra=dict(lat2=[r2(x, 2) for x in pp[1]], deep_median_25=r2(dm, 2), deep_median_fired=None if dm is None else dm > lim))


def push_worst(R):
    if R.get('source') == 'doc':
        d = R['doc']
        if d.get('push_grip_change') is None:
            return None
        return dict(grip_change=d['push_grip_change'], left_mm=d['push_left_mm'], advance_mm=None, doc=True)
    p = R.get('push')
    if not p or not p.get('worst'):
        return None
    w = p['worst']
    return dict(grip_change=max(w['grip_change']), left_mm=w['left_mm'], advance_mm=w.get('advance_mm'), lat_err_mm=w.get('lat_err_mm'))


def g_push_adv(R, C, A):
    w = push_worst(R)
    if not w or w.get('advance_mm') is None:
        return None
    lim = C['insert']['final_push']['max_advance_mm']
    return dict(value=w['advance_mm'], limit=lim, margin=lim - w['advance_mm'], fired=w['advance_mm'] > lim)


def g_push_left(R, C, A):
    w = push_worst(R)
    if not w:
        return None
    lim = C['insert']['final_push']['max_left_mm']
    return dict(value=w['left_mm'], limit=lim, margin=lim - w['left_mm'], fired=w['left_mm'] > lim)


def g_push_grip(R, C, A):
    w = push_worst(R)
    if not w:
        return None
    lim = C['insert']['final_push']['max_grip_change']
    v = w['grip_change']
    fired = v > lim or (w.get('doc') and v >= lim)        # the doc run (0.020) stopped at the watch: rounding in the doc
    return dict(value=v, limit=lim, margin=lim - v, fired=bool(fired), note='doc value' if w.get('doc') else '')


def g_push_line(R, C, A):
    fp = C['insert']['final_push']
    if fp.get('mode') != 'force':
        return None
    p = R.get('push')
    if not p or R.get('_q') is None:
        return None
    # hand off a straight push line during the logged position push (peak of |across, up| motion after the push start); force mode holds a line by design
    k1 = p['tick']
    ks = R['seated_ev'] and R['stages'].get('5 insert')
    if ks is None:
        return None
    i_seat = np.searchsorted(R['_k'], ks + R['seated_ev']['step'] + 12)
    i_end = np.searchsorted(R['_k'], k1) - 15
    if i_end <= i_seat:
        return None
    tip, _, _ = R['_L'].fk_tip(R['_q'][i_seat:i_end])
    rel = (tip - tip[0]) @ R['_eR'] * 1e3
    v = float(np.linalg.norm(rel[:, :2], axis=1).max())
    lim = fp['max_lat_err_mm']
    return dict(value=v, limit=lim, margin=lim - v, fired=v > lim, comparable=False,
                note='hand off its line during the logged 15 mm POSITION push; the force push holds the line with a spring (dry runs 0.13-1.3 mm), so this is an upper bound')


def rest_depth(R):
    ir, ho = R.get('insert_result'), R.get('handover')
    if not ir or not ho or ho.get('gap_mm') is None or ir.get('advance_mm') is None or ir['outcome'] != 'seated_detected':
        return None
    return ir['advance_mm'] - ho['gap_mm']


def g_full_depth(R, C, A):
    fp = C['insert']['final_push']
    if fp.get('full_depth_mm') is None:
        return None
    v = rest_depth(R)
    if v is None:
        return None
    return dict(value=v, limit=fp['full_depth_mm'], margin=v - fp['full_depth_mm'], fired=v < fp['full_depth_mm'], note='report only (FULL / SHORT), not a stop')


GATES = [
    G('zed_ood', 'capture', 'safety', 'a PC pose from an out-of-distribution frame steers the 0.4 m transport and approach toward the case', 'zed_gate ood_thresh x 1.25', 'Mahalanobis', g_zed_ood, 'capture'),
    G('zed_presence', 'capture', 'quality', 'no PC in the crop: the chain has nothing to approach', 'zed_gate presence_thresh', 'p', g_zed_presence, 'capture'),
    G('pick_held_band', 'pick', 'quality', 'an empty or bad grasp makes the rest of the chain useless; no hardware risk', 'pick.grasp_v3.held_band', 'gripper reading', g_pick_band, 'pick', catches=['pick_empty'],
      stop_match='pick failed'),
    G('hold_max_s', 'hold', 'safety', 'left joint 2 sits near its rating under the 25 N press', 'safety.hold_max_s', 's', g_hold_time, 'none'),
    G('tau_peak', 'all', 'safety', 'motor peak torque rating (DM4340 / DM4310)', 'safety.tau_peak_nm (0.1 s)', 'N m', g_tau_peak, 'none', stop_match='torque estimate'),
    G('tau_cont', 'all', 'safety', 'motor continuous rating and heating', 'safety.tau_cont_nm (20 s total)', 's above', g_tau_cont, 'none', stop_match='continuous torque'),
    G('qd_max', 'all', 'safety', 'joint speed: runaway or a slip', 'safety.qd_max_rad_s', 'rad/s', g_qd, 'none', stop_match='speed'),
    G('preinsert_lateral', 'approach', 'quality', 'neighbouring-port lock (log only)', 'insert.wrong_port_mm', 'mm', g_preinsert, 'none'),
    G('align_entry_gap', 'align', 'safety', 'slow steps that start close to the case slipped 7 mm into it (2026-11-03)', 'align.entry_min_gap_mm', 'mm', g_entry_gap, 'align', stop_match='entry gap'),
    G('align_no_measure', 'align', 'quality', 'no usable vision: the stage cannot aim', 'align.max_no_measure (+ min_conf, max_sigma_mm, frame_spread_mm)', 'bad cycles in a row', g_measure_quality, 'align'),
    G('align_frame_spread', 'align', 'quality', 'frames of one measurement disagree (motion or a bad fit)', 'align.frame_spread_mm', 'mm', g_frame_spread, 'align', stop_match='frames disagree'),
    G('align_sources_disagree', 'align', 'quality', 'second source (wrist net) against port vision at the hand-over: catches a wrong vision fit; vision alone drives the stage', 'align.disagree_mm (- secondary.bias_mm)', 'mm', g_disagree, 'align',
      stop_match='sources disagree'),
    G('align_scale', 'align', 'quality', 'vision must follow FK motion (slope 0.6-1.6): catches a wrong camera model', 'align.slope_min/max (+ slope_prior)', 'mm/mm', g_scale, 'align', stop_match='scale gate'),
    G('align_step_limit', 'align', 'quality', 'cycle budget of the stage', 'align.max_cycles', 'cycles', g_step_limit, 'align', catches=['stuck'], stop_match='step limit'),
    G('align_post_press', 'align', 'quality', 'cycle budget under the press (the press can move the hand)', 'align.post_press_cycles', 'cycles', g_post_press, 'align', catches=['hand_jump'], stop_match='after the press: step limit'),
    G('align_time', 'align', 'quality', 'time budget of the stage', 'align.max_time_s', 's', g_time, 'align', stop_match='time limit'),
    G('align_tip_lateral', 'align', 'safety', 'hand must stay near the 18 mm stop line (no wandering near the case)', 'align.max_lat_mm', 'mm', g_tip_lat, 'align', stop_match='moved too far'),
    G('align_tip_advance', 'align', 'safety', 'hand must not advance more than the stop-to-hand-over distance', 'align.max_advance_mm', 'mm', g_tip_adv, 'align', stop_match='moved too far'),
    G('align_grip_change', 'align', 'quality', 'reader moving in the jaws (it falls on the table at worst)', 'align.grip_change', 'gripper reading', g_grip, 'align', catches=['reader_slip'], stop_match='gripper reading changed'),
    G('align_drift', 'align', 'safety', 'hand moving with no command near the case (stick-slip release)', 'align.drift_mm x drift_n', 'mm', g_drift, 'align', catches=['hand_jump'], stop_match='drifted'),
    G('align_pred', 'align', 'quality', 'measurement change must match FK change', 'align.pred_mm (+2 sigma) x pred_n', 'mm', g_pred, 'align', stop_match='measurement against FK'),
    G('align_min_clear', 'align', 'safety', 'plug must not touch the case during the alignment', 'align.min_clear_fail_mm', 'mm', g_min_clear, 'align', stop_match='below the'),
    G('align_angle', 'align', 'quality', 'angle correction budget', 'align.angle.max_deg', 'deg', g_angle, 'align', stop_match='angle correction above'),
    G('align_stuck', 'align', 'quality', 'steps that do not move the hand (friction beyond the lead limit)', 'align.stuck_n', 'steps in a row', g_stuck, 'align', catches=['stuck'], stop_match='did not move the hand'),
    G('align_step_end', 'align', 'quality', 'step-end test and timeout (time only)', 'align.reach_mode / step_timeout_s', 'timeouts', g_step_timeouts, 'none'),
    G('seat2_detect', 'insert', 'quality', 'seat detection ends the policy (too late: the guard pulls a seated plug out; too early: short)', 'insert.seat2', 'mm depth', g_seat, 'insert'),
    G('guard_bound', 'insert', 'safety', 'side load on the port and the plug (lateral bound + hard margin)', 'insert.guard.bound_mm + hard_mm', 'mm', g_guard_bound, 'insert', catches=['stall_on_case', 'climb']),
    G('guard_press_abort', 'insert', 'safety', 'pressing on the case with no progress', 'insert.guard.stall_s', 's', g_press_abort, 'insert', catches=['stall_on_case']),
    G('insert_time', 'insert', 'quality', 'policy time budget', 'insert.insert_s', 's', g_insert_time, 'insert'),
    G('push_min_depth', 'push', 'quality', 'push only into a plug that reads as in the port', 'insert.final_push.min_depth_mm', 'mm', g_push_depth, 'insert'),
    G('push_max_lat', 'push', 'quality', 'push only near the port axis (no side load); a skip leaves the plug short', 'insert.final_push.max_lat_mm', 'mm', g_push_lat, 'insert', catches=[], stop_match=None),
    G('push_watch_advance', 'push', 'safety', 'plug moving more than a seated plug can (slip in the jaws or a broken port)', 'insert.final_push.max_advance_mm', 'mm', g_push_adv, 'insert'),
    G('push_watch_left', 'push', 'safety', 'the PC press gives way (PC moves on the table)', 'insert.final_push.max_left_mm', 'mm', g_push_left, 'insert'),
    G('push_watch_grip', 'push', 'safety', 'reader turning in the jaws under the push load', 'insert.final_push.max_grip_change', 'gripper reading', g_push_grip, 'insert'),
    G('push_line', 'push', 'safety', 'force push: hand off its line = side load on the port', 'insert.final_push.max_lat_err_mm (force mode)', 'mm', g_push_line, 'insert'),
    G('full_depth_report', 'push', 'quality', 'report FULL / SHORT from the depth at rest (no stop)', 'insert.final_push.full_depth_mm', 'mm', g_full_depth, 'full'),
]

NOT_REPLAYED = [
    ('tcp_box', 'safety', 'TCP targets of both arms in the box: the left-arm base transform is not in the replay; no box stop in any log'),
    ('table_clearance (SafeArm)', 'safety', 'per-command FlexChecker clearance; values are not logged per tick (one stop: 2026-11-06 prec2, gripper on the table, a true stop)'),
    ('move_to goal not attained (0.02 rad)', 'safety', 'two stops on 2026-11-06 (prec3, full7 home): both software bugs of that day'),
    ('camera max_age_ms', 'quality', 'one stop on 2026-11-10 16:47 (left wrist frame 48.8 s old, doc only): a true stop'),
    ('align correction grows (grow_n, grow_abs_mm)', 'quality', 'needs the fused error per cycle; never fired in the logs'),
    ('approach_nojump deadband (all9)', 'quality', 'changes the approach motion itself: needs the rig'),
    ('pick max_attempts 2 (all9)', 'quality', 'a retry changes the outcome of demo_3 (16:08); cannot be replayed'),
    ('force push tau_max_nm, ramp, still end (all9)', 'safety', 'force mode never ran on the rig'),
]


# ---------------------------------------------------------------------------------------------------------------------------- run
def matches_stop(R, g):
    """Did the logged run stop at this gate?"""
    if g.stop_match is None:
        if g.gid == 'push_max_lat':
            return bool(R.get('push_skip') and 'lateral' in R['push_skip']['reason'])
        if g.gid == 'push_min_depth':
            return bool(R.get('push_skip') and 'depth' in R['push_skip']['reason'])
        if g.gid == 'guard_bound':
            return bool(R.get('insert_stop') and R['insert_stop']['outcome'] == 'guard_bound')
        if g.gid == 'guard_press_abort':
            return bool(R.get('insert_stop') and R['insert_stop']['outcome'] == 'press_abort')
        if g.gid == 'push_watch_grip':
            p = R.get('push')
            return bool((p and p.get('stopped') and 'gripper' in str(p['stopped'])) or (R.get('source') == 'doc' and R['doc'].get('push_stopped')))
        return False
    texts = list(R.get('stopped') or [])
    if R.get('align_result'):
        texts.append(R['align_result']['reason'])
    if R.get('source') == 'doc' and g.gid == 'align_post_press':
        return bool(R['doc'].get('post_press_fired'))
    return any(g.stop_match in str(t) for t in texts)


def evaluate(runs, cfgs, L):
    rows = []
    for cname, (C, A) in cfgs.items():
        for g in GATES:
            for R in runs:
                if R.get('source') == 'none':
                    continue
                try:
                    if g.gid == 'align_step_end':
                        v = g.fn(R, C, A, L)
                    else:
                        v = g.fn(R, C, A)
                except Exception as e:  # noqa: BLE001
                    v = dict(value=None, limit=None, margin=None, fired=False, note=f'error {type(e).__name__}: {e}')
                if v is None:
                    continue
                fine = fine_of(R, g.stage_fine)
                tags = R['lab'].get('tags', [])
                comp = v.get('comparable', True)
                rows.append(dict(config=cname, gate=g.gid, stage=g.stage, kind=g.kind, run=R['run'], date=R['date'], source=R.get('source'), run_config=R.get('config'),
                                 value=v['value'] if isinstance(v['value'], (int, float)) or v['value'] is None else v['value'], limit=v['limit'], margin=v.get('margin'),
                                 fired=bool(v['fired']), stopped=matches_stop(R, g), fine=fine, comparable=comp, tags=tags,
                                 caught_tag=bool(set(tags) & set(g.catches)), axis=v.get('axis'), where=v.get('where'), extra=v.get('extra'), note=v.get('note', ''),
                                 operator=R['lab'].get('note', '')))
    return rows


def summarize(rows):
    out = []
    for cname in dict.fromkeys(r['config'] for r in rows):
        for g in GATES:
            rr = [r for r in rows if r['config'] == cname and r['gate'] == g.gid]
            if not rr:
                out.append(dict(config=cname, gate=g.gid, stage=g.stage, kind=g.kind, reason=g.reason, limit_key=g.limit_key, n=0))
                continue
            rc = [r for r in rr if r['comparable'] is not False]
            good = [r for r in rc if r['fine'] is True]
            gv = [r['value'] for r in good if isinstance(r['value'], (int, float))]
            gm = [r['margin'] for r in good if isinstance(r['margin'], (int, float))]
            fs = [r['run'] for r in rc if r['fired'] and r['fine'] is True]
            ms = [r['run'] for r in rc if not r['fired'] and r['caught_tag']]
            tp = [r['run'] for r in rc if r['fired'] and r['fine'] is False]
            near = min(good, key=lambda r: r['margin']) if gm else None
            av = [r['value'] for r in rc if isinstance(r['value'], (int, float))]
            am = [r for r in rc if isinstance(r['margin'], (int, float))]
            near_all = min(am, key=lambda r: r['margin']) if am else None
            other = {(r['run'], r['stage']) for r in rows if r['config'] == cname and r['gate'] != g.gid and r['fired']}
            ms_other = [x for x in ms if (x, g.stage) in other]
            ms = [x for x in ms if (x, g.stage) not in other]
            out.append(dict(config=cname, gate=g.gid, stage=g.stage, kind=g.kind, reason=g.reason, limit_key=g.limit_key, unit=g.unit, limit=rr[0]['limit'],
                            n=len(rc), n_not_comparable=len(rr) - len(rc), n_good=len(good), n_fired=sum(r['fired'] for r in rc), n_stopped_logged=sum(r['stopped'] for r in rr),
                            good_value=dict(min=r2(min(gv)), p50=r2(np.median(gv)), max=r2(max(gv))) if gv else None,
                            false_stops=len(fs), false_stop_runs=fs, true_stops=len(tp), true_stop_runs=tp, missed_stops=len(ms), missed_stop_runs=ms,
                            missed_caught_by_other_gate=ms_other,
                            not_comparable=[dict(run=r['run'], value=r2(r['value']), fired=r['fired']) for r in rr if r['comparable'] is False],
                            nearest_good_margin=r2(near['margin']) if near else None, nearest_good_run=near['run'] if near else None,
                            all_value=dict(min=r2(min(av)), p50=r2(np.median(av)), max=r2(max(av))) if av else None,
                            nearest_margin_all=r2(near_all['margin']) if near_all else None, nearest_run_all=near_all['run'] if near_all else None,
                            nearest_where_all=near_all.get('where') if near_all else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DATA)
    ap.add_argument('--out', default='results/real_runs_review')
    a = ap.parse_args()
    import insertion_log_analysis as L
    cfgs = {k: cfg_load(p) for k, p in CONFIGS.items()}
    for C, A in cfgs.values():
        import torch
        cp = torch.load(ROOT / 'runs/perception' / f"{C['chain_args']['zed_gate']}.pt", map_location='cpu', weights_only=False)
        C['_zed_ood'] = float(cp['ood_thresh']) * ZED_OOD_SCALE
    runs = []
    for d in DATES:
        for run in sorted(glob.glob(os.path.join(a.root, d, 'full_*'))):
            if '_dry' in os.path.basename(run):
                continue
            R = extract(run, L)
            if R.get('source') == 'logs' and os.path.exists(os.path.join(run, 'ticks.npz')) and R.get('zed'):
                t = np.load(os.path.join(run, 'ticks.npz'))
                R['_q'] = t['q'][:, 0] + L.OFF
                R['_k'] = t['k'].astype(int)
                R['_eR'] = L.port_R(R['zed']['yaw'])
                R['_L'] = L
            runs.append(R)
            print(R['run'], R.get('source'), R['lab'].get('note', ''), flush=True)
    rows = evaluate(runs, cfgs, L)
    summ = summarize(rows)
    os.makedirs(a.out, exist_ok=True)
    inv = [dict(run=R['run'], date=R['date'], source=R.get('source'), config=R.get('config'), **{k: R['lab'].get(k) for k in ('align', 'insert', 'full', 'tags', 'note')}) for R in runs]
    json.dump(dict(configs=CONFIGS, runs=inv, gates=summ, not_replayed=[dict(gate=g, kind=k, why=w) for g, k, w in NOT_REPLAYED], rows=rows),
              open(os.path.join(a.out, 'gate_replay.json'), 'w'), indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
    keys = ['config', 'gate', 'stage', 'kind', 'run', 'date', 'source', 'run_config', 'value', 'limit', 'margin', 'fired', 'stopped', 'fine', 'comparable', 'caught_tag', 'axis', 'where', 'extra', 'note']
    with open(os.path.join(a.out, 'gate_replay.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else json.dumps(r[k], default=str) if isinstance(r[k], (dict, list)) else r[k]) for k in keys})
    print(f'\n{len(runs)} runs, {len(rows)} rows')
    for s in summ:
        if s['n'] or s.get('not_comparable'):
            print(f"{s['config']:5s} {s['gate']:24s} {s['kind']:7s} n={s['n']:2d} good={s['n_good']:2d} fired={s['n_fired']:2d} logged_stops={s['n_stopped_logged']} "
                  f"FS={s['false_stops']} {s['false_stop_runs']} TS={s['true_stops']} MS={s['missed_stops']} {s['missed_stop_runs']} (+other {s['missed_caught_by_other_gate']}) good={s['good_value']} near={s['nearest_good_margin']} ({s['nearest_good_run']}) all={s['all_value']} near_all={s['nearest_margin_all']} {s['nearest_run_all']} {s['nearest_where_all']}")


if __name__ == '__main__':
    main()
