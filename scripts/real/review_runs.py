"""One table of every real full-chain run of 2026-11-10 and 2026-11-11 (task 0 of docs/real/SERVER_TASKS_20261112.md). No motion, CPU.

    PYTHONPATH=. python scripts/real/review_runs.py [--root /home/yashchitambar/peg_test/data/real_runs] [--out results/real_runs_review]

Writes runs_20261110_11.csv / .json (one row per run), steps_20261110_11.json (every align step and measurement) and plots.
Runs whose logs did not reach mason (2026-11-10 from 16:10) get a row from the session document only (source = "session doc").
All distances are encoder FK in the believed frame, port axes (x across, y up, z out of the port), mm. Times in s.
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
STAGES = ['0 capture', '1 pick', '2 hold', '3 transport', '4 approach', 'align', '5 insert', '6 release', '7 home']
SHORT = dict(zip(STAGES, ['capture', 'pick', 'hold', 'transport', 'approach', 'align', 'insert', 'release', 'home']))

# Operator notes and the "fully in" judgement, from docs/real/SESSION_20261110_duchamp3.md and SESSION_20261111_duchamp3.md.
# full: True = operator said fully in / flush / perfect; False = operator said short or hanging; None = seated, depth not stated, or no seat.
OPERATOR = {
    'full_143002_a4p_1': ('start aborted, no events', None),
    'full_143024_a4p_1': ('align mode; hand sagged at the controller switch-off', None),
    'full_144059_servo_1': ('software stop at the align start (log_event keyword), no servo motion', None),
    'full_144605_servo_1': ('align mode; plug swayed a little in the servo', None),
    'full_145249_servo_2': ('align mode; servo OK, tip speed check failed', None),
    'full_145824_kjrc_1': ('seated, reader hangs (droops) more than it should', False),
    'full_150628_kjrc_2': ('seated, still hangs; about 1 mm more would make it right', False),
    'full_152149_kjrc_3_push': ('seated and held the reader; about 3 mm short after the push', False),
    'full_152717_k4_1': ('seated and held the reader; about 3 mm short after the push', False),
    'full_154422_k4_2': ('plug was in, seat not detected, guard back-off pulled the reader out', None),
    'full_155541_k4_3': ('seated, worked really well', None),
    'full_160352_k4_4': ('align stop at the hand-over (vision-net 3.54 mm); operator held the reader', None),
    'full_161011_k4_4': ('seated; push stopped by the gripper watch', None),
    'full_162427_k4_5': ('align stop after the press (hand jump 2.8 mm, budget); operator held the reader', None),
    'full_163515_k4_5': ('flush and seated', True),
    'full_164410_k4_6': ('not in the session document', None),
    'full_164756_k4_6': ('stop in capture: left wrist camera frame 48.8 s old', None),
    'full_165527_k4_6': ('ended in transport, no stop event (process end)', None),
    'full_170521_k4_6': ('seated', None),
    'full_155351_demo_1': ('seated, reader hanging slightly (push skipped)', False),
    'full_160310_demo_2': ('seated', None),
    'full_160801_demo_3': ('failed: unsuccessful grasp', None),
    'full_161223_demo_3': ('seated (success)', None),
    'full_161705_demo_4': ('seated, fully inserted (best demo)', True),
    'full_162246_demo_5': ('failed, cause not seen (align stop, vision-net 3.51 mm)', None),
    'full_163144_speed075_1': ('success (push skipped)', None),
    'full_171905_fastmove_1': ('seated, inserted perfectly', True),
}
# Numbers of the runs without logs on mason, copied from SESSION_20261110_duchamp3.md (not checked against logs here).
DOC_ONLY = {
    'full_161011_k4_4': dict(config='bimanual_align5_k4_push15_seat6', approach_jump_mm=2.35, handover_gap_mm=6.50, handover_across_mm=-.17, handover_up_mm=.78,
                             net_minus_vis_up_mm=2.61, seat_t_s=1.44, seat_depth_mm=7.76, push_advance_mm=2.74, push_stopped='gripper watch (right jaw -0.020)',
                             push_left_mm=.15, push_jaw_change=.020, outcome='seated_detected'),
    'full_162427_k4_5': dict(config='bimanual_align5_k4_push15_seat6', outcome='align_stop', stop_reason='after the press: step limit (hand moved 2.8 mm by itself)'),
    'full_163515_k4_5': dict(config='bimanual_align5_k4_push15_seat6', approach_jump_mm=1.97, handover_gap_mm=6.69, handover_across_mm=.02, handover_up_mm=.30,
                             seat_t_s=1.54, seat_depth_mm=8.35, push_advance_mm=2.62, push_left_mm=.07, push_jaw_change=.009, outcome='seated_detected'),
    'full_164410_k4_6': dict(outcome='unknown'),
    'full_164756_k4_6': dict(config='bimanual_align5_k4_push15_seat6', outcome='capture_stop', stop_reason='wrist_left camera frame is 48774 ms old'),
    'full_165527_k4_6': dict(config='bimanual_align5_k4_push15_seat6', outcome='process_end', stop_reason='ended in transport, no stop event'),
    'full_170521_k4_6': dict(config='bimanual_align5_k4_push15_seat6', seat_t_s=1.36, seat_depth_mm=7.76, push_advance_mm=2.46, push_left_mm=.25, push_jaw_change=.011,
                             outcome='seated_detected'),
}


def r2(x, n=2):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, n)


KP = np.array([80., 80., 80., 40., 10., 10.])       # right-arm PD gains on the rig (event arm_connected)


def fk_b(L, qb):
    """Believed-frame joints (n, 6) -> TCP position mm (n, 3)."""
    import torch
    return L.FK.fk(torch.as_tensor(np.asarray(qb), dtype=torch.float32).reshape(-1, 6))[0].numpy().astype(float) * 1e3


def wrench_port(L, qb, tau, eR):
    """Hand force (N, port axes) that balances the joint torques tau at believed joints qb: solve J^T w = tau with the 6x6 Jacobian (finite
    differences of the believed FK). tau here is the PD torque kp * (target - measured): friction and gravity-model error are not removed,
    so this is a rough estimate (a few N)."""
    import torch
    def fkR(q):
        p_, R_ = L.FK.fk(torch.as_tensor(q, dtype=torch.float32).reshape(1, 6))
        return p_[0].numpy().astype(float), R_[0].numpy().astype(float)
    J = np.zeros((6, 6))
    _, R0 = fkR(qb)
    for i in range(6):
        d = np.zeros(6)
        d[i] = 1e-3
        (p1, R1), (p2, R2) = fkR(qb + d), fkR(qb - d)
        J[:3, i] = (p1 - p2) / 2e-3
        dR = (R1 - R2) / 2e-3 @ R0.T
        J[3:, i] = [dR[2, 1], dR[0, 2], dR[1, 0]]
    w = np.linalg.solve(J.T, tau)
    return eR.T @ w[:3]


def load_npz(run, name):
    p = os.path.join(run, name + '.npz')
    return dict(np.load(p, allow_pickle=True)) if os.path.exists(p) else None


def review(run, L):
    import torch
    name = os.path.basename(run)
    note, full = OPERATOR.get(name, ('', None))
    row = dict(run=name, date=Path(run).parent.name, source='logs', operator=note, operator_fully_in=full)
    evp = os.path.join(run, 'events.jsonl')
    ev = [json.loads(x) for x in open(evp)] if os.path.exists(evp) and os.path.getsize(evp) else []
    if not ev:
        row.update(source='session doc' if name in DOC_ONLY else 'none', **DOC_ONLY.get(name, dict(outcome='no events')))
        if 'seat_depth_mm' in row and 'push_advance_mm' in row:
            row['seat_plus_push_mm'] = r2(row['seat_depth_mm'] + row['push_advance_mm'])
        return row, [], []
    meta = json.load(open(os.path.join(run, 'meta.json')))
    a = meta['args']
    row.update(config=Path(a['config']).stem, speed=a['speed'], align_mode_arg=a['align_mode'], stream=bool(a.get('stream')),
               move_max_vel=meta['config'].get('safety', {}).get('move_max_vel_rad_s', .15))
    by = {}
    for e in ev:
        by.setdefault(e['kind'], []).append(e)
    one = lambda k: by[k][-1] if k in by else None  # noqa: E731

    # ---- stage times (wall clock)
    st = [(e['stage'], e['t'], e['tick']) for e in by.get('stage', [])]
    t_end = ev[-1]['t']
    for i, (s, t, _) in enumerate(st):
        row['t_' + SHORT.get(s, s)] = r2((st[i + 1][1] if i + 1 < len(st) else t_end) - t, 1)
    row['t_total'] = r2(t_end - st[0][1], 1) if st else None
    pk = [e['t'] for e in by.get('prompt', []) if str(e.get('what', '')).startswith('pick')]      # no stage event for the pick: it starts at its prompt
    nxt = [t for s, t, _ in st if s != '0 capture']
    if st and pk:
        row['t_pick'] = r2((nxt[0] if nxt else t_end) - pk[0], 1)
        row['t_capture'] = r2(pk[0] - st[0][1], 1)
    row['last_stage_before_home'] = st[-2][0] if len(st) > 1 else None
    tick = {s: k for s, _, k in st}
    stop = one('stopped')
    row['stop_reason'] = stop['reason'] if stop else None

    tk = load_npz(run, 'ticks')
    k = tk['k'].astype(int) if tk is not None else None

    # ---- ZED estimates
    for e in by.get('estimate', []):
        if e.get('est') == 'zed_pc':
            row.update(port_x_mm=r2(e['port_xy'][0] * 1e3, 1), port_y_mm=r2(e['port_xy'][1] * 1e3, 1), port_yaw_deg=r2(e['yaw_deg'], 1))

    # ---- pick
    po = one('pick_outcome')
    row['pick'] = po['outcome'] if po else None
    g = load_npz(run, 'grasp')
    if tk is not None and '2 hold' in tick:
        i = np.searchsorted(k, tick['2 hold']) - 1
        row['grip_after_pick'] = r2(tk['grip'][i, 0], 4)
    hr = one('hold_result')
    if hr:
        row.update(hold_latched=hr['latched'], hold_Fz_N=r2(hr['F_est'][2], 1))

    # ---- approach jump (scripts/real/approach_jump.py, same numbers)
    if tk is not None and '4 approach' in tick:
        end = tick.get('align', tick.get('5 insert'))
        if end is not None:
            i = (k >= max(tick['4 approach'], end - 300)) & (k < end)
            q, tg = tk['q'][i, 0], tk['target_b'][i, 0]
            tip = L.FK.fk(torch.as_tensor(q + L.OFF, dtype=torch.float32))[0].numpy() * 1e3
            cmd = L.FK.fk(torch.as_tensor(tg + L.OFF, dtype=torch.float32))[0].numpy() * 1e3
            w = 25
            dv = (tip[w:] - tip[:-w]) - (cmd[w:] - cmd[:-w])
            mv = np.linalg.norm(dv, axis=1)
            e = (tg - q) * 1e3                       # as approach_jump.py: ticks 'q' is the encoder frame, target_b the believed frame, so this
            off = np.asarray(L.OFF, float) * 1e3     # "error" holds the constant joint offset (joint 2: +12.7 mrad). e - off is the tracking error.
            et = e - off
            j = int(mv.argmax())
            de = et[j + w] - et[j]
            jj = int(np.abs(de[:3]).argmax())
            eRp = L.port_R(row['port_yaw_deg']) if row.get('port_yaw_deg') is not None else np.eye(3)
            row.update(approach_jump_mm=r2(mv.max()), approach_jump_s_before_end=r2((len(mv) - j) * DT, 1), approach_err_j2_mrad=r2(np.abs(e[:, 1]).max(), 1),
                       approach_err_j3_mrad=r2(np.abs(e[:, 2]).max(), 1), approach_end_err_j2_mrad=r2(e[-1, 1], 1),
                       approach_true_err_j2_absmax_mrad=r2(np.abs(et[:, 1]).max(), 1), approach_true_end_err_j2_mrad=r2(et[-1, 1], 1),
                       approach_true_end_err_j3_mrad=r2(et[-1, 2], 1), approach_true_end_err_j5_mrad=r2(et[-1, 4], 1),
                       jump_joint=jj + 1, jump_joint_release_mrad=r2(de[jj], 1), jump_up_mm=r2((dv[j] @ eRp)[1]), jump_across_mm=r2((dv[j] @ eRp)[0]),
                       jump_true_err_before_mrad=r2(et[j, jj], 1))
            ia = (k >= tick['4 approach']) & (k < end)
            row['approach_peak_qd_rad_s'] = r2(np.abs(tk['qd'][ia, 0]).max())
    wu = one('align_windup_at_entry')
    if wu:
        row['windup_entry_max_mrad'] = r2(np.abs(np.asarray(wu['cmd']) - np.asarray(wu['q_b'])).max() * 1e3, 1)

    # ---- align
    steps, meas = [], []
    ar, ho = one('align_result'), one('align_handover')
    if 'align_start' in by:
        servo = 'align_servo_frame' in by or 'align_servo_end' in by
        row['align'] = 'servo' if servo else 'step'
        t_al = by['align_start'][0]['t']
        if ar:
            row.update(align_ok=ar['ok'], align_reason=ar['reason'], align_cycles=ar['cycles'], align_s=r2(ar['seconds'], 1),
                       slope_across=r2(ar['slope'][0]), slope_up=r2(ar['slope'][1]), slope_gap=r2(ar['slope'][2]))
        ss = by.get('align_step', [])
        ms = by.get('align_measure', [])
        pr = 'align_press_on' in by and by['align_press_on'][0]['t']
        for s in ss:
            d = np.asarray(s['delta'], float)
            probe = bool((np.abs(d) > 1.5).any())
            steps.append(dict(run=name, tick=s['tick'], delta=s['delta'], moved=s['moved'], frac=s['frac'], reached=s['reached'], ticks=s['ticks'], probe=probe,
                              axis=int(np.abs(d).argmax()), size=float(np.abs(d).max()), sign=float(np.sign(d[np.abs(d).argmax()])), n_axes=int((np.abs(d) > 1e-6).sum()),
                              post_press=bool(pr and s['t'] > pr)))
        prev_t = t_al
        for m in ms:
            fr = m.get('frames', [])
            sec = m.get('second') or {}
            meas.append(dict(run=name, cycle=m['cycle'], tick=m['tick'], ok=m['ok'], n_frames=len(fr), frames_ms=float(sum(f['ms'] for f in fr)),
                             first_frame_ms=float(fr[0]['ms']) if fr else None, v=[m.get('across'), m.get('up'), m.get('gap')], fused=m.get('fused'),
                             second=sec.get('v') if sec.get('ok') else None, decision=m.get('decision'), grip=m.get('grip'), dt_since_prev_event=m['t'] - prev_t))
            prev_t = m['t']
        if ss:
            tks = np.array([s['ticks'] for s in ss])
            to = np.array([not s['reached'] for s in ss])
            pb = np.array([s['probe'] for s in steps])
            row.update(align_steps=len(ss), align_step_s=r2(tks.sum() * DT, 1), align_timeouts=int(to.sum()), align_timeout_s=r2(tks[to].sum() * DT, 1),
                       align_probe_steps=int(pb.sum()), align_probe_step_s=r2(tks[pb].sum() * DT, 1), align_turns=len(by.get('align_turn', [])),
                       timeout_frac_p50=r2(np.median([abs(s['frac']) for s, t in zip(ss, to) if t]) if to.any() else None))
        if ms:
            row.update(align_measures=len(ms), align_measure_frames_s=r2(sum(m['frames_ms'] for m in meas) / 1e3, 1), first_gap_mm=r2(ms[0].get('gap')),
                       first_across_mm=r2(ms[0].get('across')), first_up_mm=r2(ms[0].get('up')))
            # wrist net (second source) minus vision, last measurement that has both
            both = [m for m in ms if (m.get('second') or {}).get('ok') and m.get('ok')]
            if both:
                m = both[-1]
                ref = np.asarray(m.get('fused') or [m['across'], m['up'], m['gap']], float)
                d = np.asarray(m['second']['v'], float) - ref
                row.update(net_minus_vis_across_mm=r2(d[0]), net_minus_vis_up_mm=r2(d[1]), net_minus_vis_gap_mm=r2(d[2]))
                dd = np.array([np.asarray(x['second']['v'], float) - np.asarray([x['across'], x['up'], x['gap']], float) for x in both])
                row.update(net_minus_vis_up_all_p50=r2(np.median(dd[:, 1])), net_minus_vis_n=len(both))
        if servo:
            se = one('align_servo_end')
            fr = by.get('align_servo_frame', [])
            row.update(servo_frames=len(fr), servo_frame_ms_p50=r2(np.median([f['ms'] for f in fr]), 1) if fr else None)
            if se:
                row.update(align_s=r2(se['seconds'], 1), align_ok=se['ok'])
        if row.get('t_align') is not None and ss:
            row['align_other_wall_s'] = r2(row['t_align'] - row['align_step_s'] - row.get('align_measure_frames_s', 0), 1)
        if ho:
            row.update(handover_ok=ho['ok'], handover_gap_mm=r2(ho['gap_mm']), handover_across_mm=r2(ho['lat_mm'][0]) if ho['lat_mm'] else None,
                       handover_up_mm=r2(ho['lat_mm'][1]) if ho['lat_mm'] else None)
        if pr and ms:
            post = [m for m in ms if m['t'] > pr]
            row['align_post_press_measures'] = len(post)
            row['align_post_press_s'] = r2((by['align_handover'][-1]['t'] if ho else ar['t'] if ar else post[-1]['t']) - pr, 1) if post else None

    # ---- insertion
    ir, sd = one('insert_result'), one('seated_detected')
    gd, pol = load_npz(run, 'guard'), load_npz(run, 'policy')
    if ir:
        row.update(outcome=ir['outcome'], insert_steps=ir['steps'], seat_t_s=r2(sd['step'] * DT) if sd else None, seat_depth_mm=r2(sd['depth_mm']) if sd else None,
                   insert_result_advance_mm=r2(ir.get('advance_mm')) if ir.get('advance_mm') is not None else None)
        row['policy'] = 'K4' if 'insert_vision_prior' in by else 'K-J recentred' if 'insert_obs_recenter' in by else 'K-J'
    if gd is not None and len(gd['k']):
        rm, rt = gd['rel_m'], gd['rel_t']
        d2 = gd['depth2_mm']
        row.update(fk_advance_k4_end_mm=r2(-rm[-1, 2]), fk_advance_k4_max_mm=r2(-rm[:, 2].min()), tip_up_max_mm=r2(rm[:, 1].max()), tip_up_min_mm=r2(rm[:, 1].min()),
                   tip_across_absmax_mm=r2(np.abs(rm[:, 0]).max()), target_up_max_mm=r2(rt[:, 1].max()), target_across_absmax_mm=r2(np.abs(rt[:, 0]).max()),
                   target_in_max_mm=r2(-rt[:, 2].min()), lead_up_end_mm=r2(rt[-1, 1] - rm[-1, 1]), lead_in_end_mm=r2(-(rt[-1, 2] - rm[-1, 2])),
                   depth2_max_mm=r2(np.nanmax(d2)) if np.isfinite(d2).any() else None, seat_lat2_across_mm=r2(gd['lat2_mm'][-1, 0]), seat_lat2_up_mm=r2(gd['lat2_mm'][-1, 1]),
                   guard_clamped=int(gd['clamped'][-1]))
        if np.isfinite(d2).any():
            deep = np.where(d2 >= 4.8)[0]
            row['t_first_deep_s'] = r2(deep[0] * DT) if len(deep) else None
        if pol is not None and 'contact_share' in pol:
            cs = pol['contact_share']
            row['contact_inside_share_end'] = r2(cs[-1, 2])
    fp, fs = one('insert_final_push'), one('insert_final_push_skipped')
    if fp:
        wr = fp.get('worst') or {}          # the 5 mm push (first version) logged no watch values
        row.update(push='stopped' if fp.get('stopped') else 'done', push_mm_cmd=fp['mm'], push_advance_mm=r2(fp['advance_mm']), push_left_mm=r2(wr.get('left_mm')),
                   push_jaw_change=r2(max(np.abs(wr['grip_change'])), 4) if wr else None, push_stopped=fp.get('stopped'))
        if sd:
            row['seat_plus_push_mm'] = r2(sd['depth_mm'] + fp['advance_mm'])
    elif fs:
        row.update(push='skipped', push_stopped=fs['reason'])
        if sd:
            row['seat_plus_push_mm'] = r2(sd['depth_mm'])
    elif ir:
        row['push'] = 'no push in config' if not meta['config'].get('insert', {}).get('final_push') else 'not run (no seat)'
        if sd:
            row['seat_plus_push_mm'] = r2(sd['depth_mm'])
    # ---- forces and FK advance at the end of the policy and at the peak of the push (ticks; believed q = encoder q + joint offsets)
    if ir and tk is not None and '6 release' in tick and row.get('port_yaw_deg') is not None:
        off = np.asarray(L.OFF, float)
        eR = L.port_R(row['port_yaw_deg'])
        ia, i1 = np.searchsorted(k, tick['5 insert']), np.searchsorted(k, tick['6 release'])
        qb, tg = tk['q'][ia:i1, 0] + off, tk['target_b'][ia:i1, 0]
        tip, tt = L.fk_tip(qb)[0] * 1e3, L.fk_tip(tg)[0] * 1e3          # plug tip (mean real grasp), as the guard's rel_m / rel_t
        rel, lead = (tip - tip[0]) @ eR, (tt - tip) @ eR
        # last policy tick: policy.npz 'k' is the chain tick of each policy step (the insertion starts with warm-up ticks)
        ns = int(np.clip((np.searchsorted(k, int(pol['k'][-1])) - ia - 2) if pol is not None and len(pol['k']) else int(ir['steps']) - 1, 0, len(qb) - 1))

        def F(j):
            s_ = slice(max(j - 5, 0), j + 1)
            return wrench_port(L, qb[j], KP * (tg[s_] - qb[s_]).mean(0), eR)
        f = F(ns)
        row.update(k_end_adv_mm=r2(-rel[ns, 2]), k_end_up_mm=r2(rel[ns, 1]), k_end_lead_up_mm=r2(lead[ns, 1], 1), k_end_lead_in_mm=r2(-lead[ns, 2], 1),
                   k_end_F_in_N=r2(-f[2], 1), k_end_F_up_N=r2(f[1], 1), k_end_F_across_N=r2(f[0], 1),
                   k_end_depth_mm=r2(-rel[ns, 2] - row['handover_gap_mm']) if row.get('handover_gap_mm') else None)
        if fp:
            n0 = min(ns + 12, len(qb) - 1)            # after the 0.2 s measured-pose hold
            j = n0 + int((-lead[n0:, 2]).argmax())
            f = F(j)
            row.update(relaxed_adv_mm=r2(-rel[n0 - 2:n0 + 1, 2].mean()), relaxed_up_mm=r2(rel[n0 - 2:n0 + 1, 1].mean()), push_peak_adv_mm=r2(-rel[j, 2]), push_peak_up_mm=r2(rel[j, 1]),
                       push_peak_lead_in_mm=r2(-lead[j, 2], 1), push_F_in_N=r2(-f[2], 1), push_F_up_N=r2(f[1], 1), push_F_across_N=r2(f[0], 1),
                       push_net_gain_mm=r2(-rel[j, 2] + rel[ns, 2]), end_adv_mm=r2(-rel[-1, 2]), end_up_mm=r2(rel[-1, 1]),
                       push_grip_start=r2(tk['grip'][ia + ns, 0], 4), push_grip_end=r2(tk['grip'][i1 - 1, 0], 4))
            if row.get('handover_gap_mm'):
                row['peak_depth_mm'] = r2(max(-rel[j, 2], -rel[ns, 2]) - row['handover_gap_mm'])
        elif row.get('handover_gap_mm'):
            row['peak_depth_mm'] = row['k_end_depth_mm']
            row.update(end_adv_mm=r2(-rel[-1, 2]), end_up_mm=r2(rel[-1, 1]))
    if not ir:
        row['outcome'] = ('pick_failed' if po and po['outcome'] != 'held' else 'align_only' if a['align_mode'] == 'align' and ar and ar['ok'] else
                          'align_stop' if ar and not ar['ok'] else 'stopped' if stop else 'no_insert')

    # ---- loop timing
    if tk is not None:
        for s, lab in (('5 insert', 'insert'), ('align', 'align')):
            if s in tick:
                nxt = [x[2] for x in st if x[2] > tick[s]]
                i = (k >= tick[s]) & (k < (nxt[0] if nxt else k[-1] + 1))
                if i.any():
                    row[f'compute_ms_p50_{lab}'] = r2(np.median(tk['compute_ms'][i]), 1)
                    row[f'compute_ms_p99_{lab}'] = r2(np.percentile(tk['compute_ms'][i], 99), 1)
                    row[f'late_ticks_{lab}'] = int((tk['late_s'][i] > .005).sum())
                    row[f'ticks_{lab}'] = int(i.sum())
        row['compute_ms_p99_run'] = r2(np.percentile(tk['compute_ms'], 99), 1)
        row['late_ticks_run'] = int((tk['late_s'] > .005).sum())
        if pol is not None and len(pol['k']) > 1:
            row['policy_tick_gap_max'] = int(np.diff(pol['k'].astype(int)).max())
    # ---- slow moves
    mv = by.get('move_to', [])
    md = by.get('moved', [])
    if mv:
        row['n_slow_moves'] = len(mv)
    return row, steps, meas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/yashchitambar/peg_test/data/real_runs')
    ap.add_argument('--dates', nargs='+', default=['2026-11-10', '2026-11-11'])
    ap.add_argument('--out', default='results/real_runs_review')
    a = ap.parse_args()
    import insertion_log_analysis as L
    os.makedirs(a.out, exist_ok=True)
    rows, steps, meas = [], [], []
    for d in a.dates:
        for run in sorted(glob.glob(os.path.join(a.root, d, 'full_*'))):
            if '_dry' in os.path.basename(run):
                continue
            r, s, m = review(run, L)
            rows.append(r), steps.extend(s), meas.extend(m)
            print(r['run'], r.get('source'), r.get('outcome'), flush=True)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    stem = os.path.join(a.out, 'runs_20261110_11')
    with open(stem + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    json.dump(rows, open(stem + '.json', 'w'), indent=1, default=str)
    json.dump(dict(steps=steps, measures=meas), open(os.path.join(a.out, 'steps_20261110_11.json'), 'w'), default=str)
    print('wrote', stem + '.csv', len(rows), 'rows;', len(steps), 'align steps;', len(meas), 'measurements')


if __name__ == '__main__':
    main()
