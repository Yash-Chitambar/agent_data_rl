"""Force-push spring-back on the real logs: timeline of the right hand along the port axis from the K4 seat to the end of the release.

    python scripts/real/springback_timeline.py RUN_DIR [RUN_DIR ...] [--json out.json] [--csv-dir DIR]

Per tick (ticks.npz, believed FK = encoder + joint offsets; port axes from the ZED yaw, as the chain):
  adv_mm      TCP advance along -port_z since the push start (the same reference as insert_final_push advance_mm)
  lat_mm      TCP across / up against the push start
  F_pd_n      PD axial force kp (target - q) through J^-T, along -port_z (what the controller commands)
  F_eff_n     axial force from the measured joint torques (eff) minus their mean in the 0.2 s seat hold (gravity at about the
              same pose), through J^-T. Motor-side: includes joint friction; sign as F_pd (positive = into the port)
  dq_mrad     max |target - q| over the right joints
  grip, grip_cmd   right gripper reading and command (i2rt units; finger_m = jaw opening, m)
  left_mm     left TCP displacement since the push start
Phases from the events: seat hold (0.2 s), push (centring, ramp, hold, ended still/hold), drop (0.3 s, target = measured each tick),
release open (insert.release_open_s, target = the pose at the release start, jaws open), retreat (port +z), unload left.
FK sees the hand, not the plug: a hand that comes back with closed jaws may or may not carry the plug.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402

KP = L.SC.kp[0].float()


def jac(q):
    out_p, out_J = [], []
    qt = torch.as_tensor(q, dtype=torch.float32)
    for i in range(0, len(qt), 64):
        p, _, J = L.FK.fk(qt[i:i + 64], with_jac=True)
        out_p.append(p), out_J.append(J)
    return torch.cat(out_p).numpy().astype(float), torch.cat(out_J)


def cart_force(J, tau):
    """(n, 6, 6) J, (n, 6) joint torques -> (n, 3) force with J^T [F, M] = tau (least squares)."""
    return torch.linalg.lstsq(J.transpose(1, 2), torch.nan_to_num(torch.as_tensor(tau, dtype=torch.float32))[..., None]).solution[:, :3, 0].numpy()


def timeline(run):
    r = L.load(run)
    ev, t, eR = r['ev'], r['t'], r['eR']
    k = t['k'].astype(int)
    seat = next((e for e in ev if e['kind'] == 'seated_detected'), None)
    push = next((e for e in ev if e['kind'] == 'insert_final_push'), None)
    if seat is None or push is None:
        return None, None
    dep = next((e for e in ev if e['kind'] == 'insert_depth'), None)
    cfg = json.load(open(os.path.join(run, 'meta.json'))).get('args', {}).get('config')
    k_ins = r['st']['5 insert']
    k_rel = r['st']['6 release']
    k_home = r['st'].get('7 home', k[-1] + 1)
    k_seat = k_ins + int(seat['step']) + 1                  # first tick after the seated tick
    k_push0 = k_seat + 10                                   # after the 0.2 s measured-pose hold
    un = push.get('unload') or None                        # insert.final_push.unload (DEPTH_SPRINGBACK_20261114): ramp-down / open / relax
    k_drop = int(un['tick_start']) if un else int(push['tick']) - 15      # push loop end: 0.3 s drop (or the unload) follows
    speed = float(r['meta']['args'].get('speed', 1.0))
    ic = json.load(open(os.path.join(ROOT, cfg)))['insert'] if cfg and os.path.exists(os.path.join(ROOT, cfg)) else {}
    n_open = int(round(ic.get('release_open_s', 0.4) / 0.02))
    n_ret = max(1, int(round(ic.get('retreat_m', 0.05) / (ic.get('retreat_v_m_s', 0.02) * speed) / 0.02)))
    k_ret0, k_ret1 = k_rel + n_open, k_rel + n_open + n_ret
    i0, i1 = int(np.searchsorted(k, k_seat - 12)), int(np.searchsorted(k, k_home))
    sl = slice(i0, i1)
    q = np.asarray(t['q'][sl, 0], float) + L.OFF
    ql = np.asarray(t['q'][sl, 1], float) + L.C.joint_offsets('left')
    tgt = np.asarray(t['target_b'][sl, 0], float)
    eff = np.asarray(t['eff'][sl, 0], float)
    p, J = jac(q)
    pl, _ = L.SC.chain['left'].fk(torch.as_tensor(ql, dtype=torch.float32))
    pl = pl.numpy().astype(float)
    kk = k[sl]
    j0 = int(np.searchsorted(kk, k_push0))
    rel = (p - p[j0]) @ eR * 1e3
    adv = -rel[:, 2]
    zin = -eR[:, 2]
    Fpd = cart_force(J, (KP.numpy() * (tgt - q))) @ zin
    hs = slice(int(np.searchsorted(kk, k_seat)), j0)
    Feff = cart_force(J, eff - eff[hs].mean(0)) @ zin
    left = np.linalg.norm(pl - pl[j0], axis=1) * 1e3
    grip = np.asarray(t['grip'][sl, 0], float)
    gcmd = np.asarray(t['grip_cmd'][sl, 0], float)
    fing = np.asarray(t['finger_m'][sl, 0], float) * 1e3
    dq = np.abs(tgt - q).max(1) * 1e3

    def ph(kx):
        if kx < k_push0: return 'seat_hold'
        if kx < k_drop: return 'push'
        if un and kx < int(un['tick_open']): return 'unload_ramp'
        if un and kx < int(un['tick_relax']): return 'unload_open'
        if kx < k_rel: return 'unload_relax' if un else 'drop'
        if kx < k_ret0: return 'open'
        if kx < k_ret1: return 'retreat'
        return 'unload_left'
    rows = [dict(k=int(kx), phase=ph(kx), adv_mm=adv[j], across_mm=rel[j, 0], up_mm=rel[j, 1], F_pd_n=Fpd[j], F_eff_n=Feff[j],
                 dq_mrad=dq[j], grip=grip[j], grip_cmd=gcmd[j], finger_mm=fing[j], left_mm=left[j]) for j, kx in enumerate(kk)]

    def at(kx):
        j = int(min(np.searchsorted(kk, kx), len(kk) - 1))
        return dict(k=int(kk[j]), adv_mm=round(float(adv[j]), 2), up_mm=round(float(rel[j, 1]), 2), across_mm=round(float(rel[j, 0]), 2),
                    F_pd_n=round(float(Fpd[j]), 1), F_eff_n=round(float(Feff[j]), 1), grip=round(float(grip[j]), 4),
                    finger_mm=round(float(fing[j]), 2), left_mm=round(float(left[j]), 2), dq_mrad=round(float(dq[j]), 1))
    jd = int(np.searchsorted(kk, k_drop))
    jr = int(np.searchsorted(kk, k_rel))
    jo = int(np.searchsorted(kk, k_ret0))
    # when does the hand come back: per-phase change of the advance
    seg = {}
    for name, a, b in (('push', j0, jd), ('drop', jd - 1, jr - 1), ('open', jr - 1, jo - 1)):
        seg[name] = dict(start=round(float(adv[a]), 2), end=round(float(adv[b]), 2), change=round(float(adv[b] - adv[a]), 2))
    # drop phase per tick: how fast the hand goes back once the target follows the measured pose
    drop = [dict(k=int(kk[j]), adv_mm=round(float(adv[j]), 2), F_pd_n=round(float(Fpd[j]), 1), F_eff_n=round(float(Feff[j]), 1))
            for j in range(jd - 2, jr + 1)]
    # retreat: does the jaw (still opening) carry anything? advance against the commanded retreat path
    jaw_open_tick = next((int(kk[j]) for j in range(jr, len(kk)) if grip[j] - grip[jr - 1] > 0.01), None)
    full_open_tick = next((int(kk[j]) for j in range(jr, len(kk)) if grip[j] >= grip[jr:].max() - 0.002), None)
    out = dict(run=r['name'], config=cfg, speed=speed, ticks=dict(insert=k_ins, seat=k_seat, push0=k_push0, drop=k_drop, release=k_rel,
                                                                   retreat=k_ret0, retreat_end=k_ret1, home=int(k_home)),
               event=dict(mode=push.get('mode', 'position'), advance_mm=push['advance_mm'], rest_advance_mm=push.get('rest_advance_mm'),
                          force=push.get('force_reached_n'), stopped=push.get('stopped'),
                          end=push.get('end'), centre_mm=push.get('centre_mm'), rest_depth_mm=None if dep is None else dep['rest_depth_mm'],
                          gap_mm=None if dep is None else dep['gap_mm']),
               marks=dict(push_start=at(k_push0), push_peak=at(int(kk[j0 + int(np.argmax(adv[j0:jd]))])), push_end=at(k_drop - 1),
                          drop_end=at(k_rel - 1), open_end=at(k_ret0 - 1), retreat_5mm=at(k_ret0 + n_ret // 3)),
               phases=seg, drop_ticks=drop, jaw_moves_tick=jaw_open_tick, jaw_full_open_tick=full_open_tick,
               grip_change_push=round(float(grip[j0] - grip[jd - 1]), 4))
    return out, rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--json', default=None)
    ap.add_argument('--csv-dir', default=None)
    A = ap.parse_args()
    res = []
    for run in A.runs:
        try:
            o, rows = timeline(run)
        except (IndexError, KeyError, StopIteration, FileNotFoundError) as e:
            print(f'{run}: incomplete log ({type(e).__name__}: {e}), skipped')
            continue
        if o is None:
            print(f'{run}: no seat or no final push, skipped')
            continue
        res.append(o)
        print(json.dumps({k: v for k, v in o.items() if k != 'drop_ticks'}, indent=1))
        print('drop ticks:', ' '.join(f"{d['k']}:{d['adv_mm']:.2f}/{d['F_pd_n']:.1f}/{d['F_eff_n']:.1f}" for d in o['drop_ticks']))
        if A.csv_dir:
            os.makedirs(A.csv_dir, exist_ok=True)
            with open(os.path.join(A.csv_dir, o['run'] + '.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                for x in rows:
                    w.writerow({a: (round(float(b), 4) if isinstance(b, (float, np.floating)) else b) for a, b in x.items()})
    if A.json:
        Path(A.json).parent.mkdir(parents=True, exist_ok=True)
        Path(A.json).write_text(json.dumps(res, indent=1, default=float))
