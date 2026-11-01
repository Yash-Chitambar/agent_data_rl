"""Dry run of the real runner (deploy/run_bimanual.py: SimIO = one eval_chain world, the real code path above it)
against scripts/eval/eval_chain.py on the same seeds and the same Chain object.

--mode expert: the runner with eval_chain's default pick (ZED reader estimate, scripted expert, no servo) and
  eval_chain.Chain.run_batch(seed) (believed frame) on the same seed: per-stage decisions side by side (ZED estimate
  and gate, reader estimate, pick, hold latch, pre-insert error, insertion outcome). The runner's ZED frame is the
  same render (same seed, same camera-noise draw): its estimate must match eval_chain's exactly.
--mode servo: the real configuration (servo-descent pick): end-to-end outcomes and the seated detector against the
  sim truth (MuJoCo Warp is not bitwise deterministic: compare outcomes and distributions, not trajectories).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_dryrun --mode expert --seeds 7000 7001 7002 7003
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_dryrun --mode servo --seeds 7000 ... 7007
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def scal(v):
    if torch.is_tensor(v):
        v = v.detach().cpu().numpy()
    v = np.asarray(v)
    return v.reshape(-1)[0].item() if v.size else None


def mine_row(out):
    r = out['results']
    lg = r.get('chain_log', {})
    z = lg.get('zed', {})
    ins = r.get('insert', {})
    return dict(completed=out['completed'], stop=out.get('stop'), zed_xy=z.get('port_xy'), zed_yaw_deg=z.get('yaw_deg'),
                zed_ok=z.get('ok'), zed_err_mm=z.get('err_xy_mm'), pick_ok=r.get('pick_ok'),
                hold_latched=scal(lg['hold_latched']) if 'hold_latched' in lg else None,
                pre_lateral_mm=scal(lg['pre_lateral_mm']) if 'pre_lateral_mm' in lg else None,
                pre_tilt_deg=scal(lg['pre_tilt_deg']) if 'pre_tilt_deg' in lg else None,
                insert=ins.get('outcome'), t_detect_s=ins.get('t_seat_s'), true_seat_tick=ins.get('true_seat_tick'),
                obs_max_abs_err=ins.get('obs_max_abs_err'), true_seated_after_release=r.get('true_seated_after_release'),
                seated_end=(out.get('sim_truth_end') or {}).get('seated'), ticks=out.get('ticks'))


def ref_row(ch, seed):
    import pandas as pd
    ev = ch.EC if hasattr(ch, 'EC') else None
    out = ch.run_batch(seed)
    df = pd.DataFrame(out)
    from deploy.bimanual.io_sim import eval_chain
    df['e2e_class'] = eval_chain().classify(df)
    r = df.iloc[0]
    return dict(zed_ok=bool(r.zed_start_ok), zed_err_mm=float(r.zed_start_xy_mm), pick_ok=bool(r.pick_ok),
                reader_est_err_mm=float(r.get('pick_est_pos_err_mm', np.nan)), hold_latched=bool(r.hold_latched),
                pre_lateral_mm=float(r.pre_lateral_mm), pre_tilt_deg=float(r.pre_tilt_deg), insert=str(r.insert_outcome),
                e2e=str(r.e2e_class))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('expert', 'servo'), required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[7000, 7001, 7002, 7003])
    ap.add_argument('--no-ref', action='store_true', help='expert mode without the eval_chain reference run')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    from deploy import run_bimanual as RB
    torch.set_num_threads(2)
    args = RB.parse(['--dry-run', '--stage', 'full', '--pick-mode', a.mode, '--check-obs', '--name', f'eq_{a.mode}'])
    r = RB.Runner(args)
    rows = []
    for s in a.seeds:
        t0 = time.time()
        row = {'seed': s}
        if a.mode == 'expert' and not a.no_ref:
            if r.io is None:
                from deploy.bimanual.io_sim import SimIO
                r.io = SimIO(r.cfg, r.gmaps, r.policy_path)
            row['eval_chain'] = ref_row(r.io.ch, s)
        out = r.episode(s)
        row['runner'] = mine_row(out)
        row['wall_s'] = time.time() - t0
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    summ = summarize(rows, a.mode)
    res = {'mode': a.mode, 'seeds': a.seeds, 'rows': rows, 'summary': summ}
    print(json.dumps(summ, indent=1, default=float))
    out = Path(a.out or ROOT / f'results/usbc/real_deploy/bimanual_dryrun_{a.mode}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))


def summarize(rows, mode):
    m = [x['runner'] for x in rows]
    s = {'n': len(rows), 'runner_completed': sum(bool(x['completed']) for x in m),
         'runner_seated_after_release': sum(bool(x['true_seated_after_release']) for x in m),
         'runner_detected_seated': sum(x['insert'] == 'seated_detected' for x in m),
         'obs_max_abs_err': max([x['obs_max_abs_err'] for x in m if x['obs_max_abs_err'] is not None] or [None]) if any(x['obs_max_abs_err'] is not None for x in m) else None,
         'stops': [x['stop'] for x in m if x['stop']]}
    det = [(x['t_detect_s'], x['true_seat_tick']) for x in m if x['t_detect_s'] is not None and x['true_seat_tick'] is not None]
    if det:
        s['detector_delay_s_after_true_seat'] = [round(t - k * 0.02, 3) for t, k in det]
    if mode == 'expert' and rows and 'eval_chain' in rows[0]:
        e = [x['eval_chain'] for x in rows]
        s['eval_chain_e2e_success'] = sum(x['e2e'] == 'success' for x in e)
        s['zed_ok_agree'] = sum(bool(a_['zed_ok']) == bool(b['zed_ok']) for a_, b in zip(m, e))
        s['zed_err_mm_pairs'] = [(round(a_['zed_err_mm'] or np.nan, 3), round(b['zed_err_mm'], 3)) for a_, b in zip(m, e)]
        s['pick_agree'] = sum(bool(a_['pick_ok']) == bool(b['pick_ok']) for a_, b in zip(m, e))
        s['hold_agree'] = sum(bool(a_['hold_latched']) == bool(b['hold_latched']) for a_, b in zip(m, e) if a_['hold_latched'] is not None)
        s['pre_lateral_mm_pairs'] = [(a_['pre_lateral_mm'], b['pre_lateral_mm']) for a_, b in zip(m, e)]
        s['insert_pairs'] = [(a_['insert'], a_['true_seated_after_release'], b['insert']) for a_, b in zip(m, e)]
    return s


if __name__ == '__main__':
    main()
