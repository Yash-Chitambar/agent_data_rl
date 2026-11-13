"""Approach stage, iteration by iteration, from a run's log (no motion): where the uncommanded hand move happens.

For each servo iteration of eval_chain.stage_approach (frame -> IK -> ramp -> settle) it prints the commanded tip step,
the net's estimate, the measured tip move in the ramp and in the settle, the tip lag (FK(target) - FK(measured)) and the
joint 2/3 errors at the end of the settle, all in the believed frame (target_b vs q + joint offsets). The largest
uncommanded move (tip move minus command move over --window-s, as scripts/real/approach_jump.py) is located in its
iteration and phase.

    python scripts/real/approach_iters.py data/real_runs/<date>/full_<id> [...] [--window-s 0.5] [--json out.json]
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402


def ticks(run):
    d = {k: [] for k in ('k', 'q', 'target_b', 'ff')}
    for f in sorted(glob.glob(str(Path(run) / 'chunks/ticks/*.npz'))):
        z = np.load(f, allow_pickle=True)
        for k in d:
            d[k].append(z[k])
    return {k: np.concatenate(v) for k, v in d.items()}


def fk(qb):
    return L.FK.fk(torch.as_tensor(qb, dtype=torch.float32))[0].cpu().numpy() * 1e3


def analyse(run, window_s=.5):
    ev = [json.loads(x) for x in open(Path(run) / 'events.jsonl')]
    st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
    if '4 approach' not in st:
        return None
    t0, end = st['4 approach'], st.get('align', st.get('5 insert'))
    frames = sorted(int(e['frame_tag'].split('@')[1]) for e in ev if e.get('kind') == 'net' and str(e.get('frame_tag', '')).startswith('4_approach')
                    and 'inhand' not in e['net'])
    nets = {int(e['frame_tag'].split('@')[1]): e['out'][0] for e in ev if e.get('kind') == 'net' and str(e.get('frame_tag', '')).startswith('4_approach') and 'inhand' not in e['net']}
    d = ticks(run)
    i = (d['k'] >= t0) & (d['k'] < end)
    k = d['k'][i]
    qb = d['q'][i, 0] + np.asarray(L.OFF)                 # believed measured
    tb = d['target_b'][i, 0]                               # believed target
    tip, cmd = fk(qb), fk(tb)
    left = fk_left = None
    w = int(round(window_s / .02))
    mv = np.linalg.norm((tip[w:] - tip[:-w]) - (cmd[w:] - cmd[:-w]), axis=1)
    j = int(mv.argmax())
    jump_k = int(k[j + w])          # end of the worst window
    # ramp ends: first tick after a frame where the target stops changing
    dtb = np.r_[0, np.abs(np.diff(tb, axis=0)).max(1)]
    its = []
    for n, f in enumerate(frames):
        f_next = frames[n + 1] if n + 1 < len(frames) else end
        s = (k >= f) & (k < f_next)
        idx = np.nonzero(s)[0]
        if not idx.size:
            continue
        moving = idx[dtb[idx] > 1e-7]
        r_end = int(moving.max()) + 1 if moving.size else idx[0]
        a, b = idx[0], idx[-1]
        e_end = (tb[b] - qb[b]) * 1e3
        its.append(dict(it=n, tick=int(k[a]), ramp_ticks=int(r_end - a), net=np.round(nets.get(f, [np.nan] * 3)[:3], 2).tolist(),
                        cmd_step_mm=np.round(cmd[b] - cmd[a], 2).tolist(), cmd_step_norm=round(float(np.linalg.norm(cmd[b] - cmd[a])), 2),
                        tip_ramp_mm=round(float(np.linalg.norm(tip[min(r_end, b)] - tip[a])), 2), tip_settle_mm=round(float(np.linalg.norm(tip[b] - tip[min(r_end, b)])), 2),
                        lag_end_mm=np.round(cmd[b] - tip[b], 2).tolist(), lag_end_norm=round(float(np.linalg.norm(cmd[b] - tip[b])), 2),
                        err_j2_j3_end_mrad=np.round(e_end[1:3], 1).tolist(), ff=bool(np.abs(d['ff'][i][b, 0]).max() > 1e-6),
                        has_jump=bool(k[a] <= jump_k < (f_next))))
    e = (tb - qb) * 1e3
    # at the stop: the final iterations (eval_chain final_iters; every iteration whose net gap is within 3 mm of the
    # last one), the window approach_jump.py checks; wind-up = growth of the command-tip lag while the tip stays still
    n_stop = int(sum(1 for f in frames if abs(nets.get(f, [0, 0, 1e9])[2] - nets.get(frames[-1], [0, 0, 0])[2]) < 3.))
    s0 = int(np.searchsorted(k, frames[-n_stop])) if frames else 0
    mv2 = mv[max(0, s0):]
    j2 = int(mv2.argmax()) + max(0, s0) if mv2.size else j
    jk2 = int(k[j2 + w]) if mv2.size else jump_k
    lag = cmd - tip
    pre = max(s0, j2)                                # start of the worst window at the stop
    stop = dict(n_stop_iters=n_stop, start_tick=int(k[s0]), max_uncommanded_mm=round(float(mv2.max()) if mv2.size else 0., 2), jump_tick=jk2,
                jump_iter=int(sum(1 for f in frames if f <= jk2) - 1),
                cmd_path_mm=round(float(np.linalg.norm(np.diff(cmd[s0:], axis=0), axis=1).sum()), 2),
                cmd_net_mm=round(float(np.linalg.norm(cmd[-1] - cmd[s0])), 2),
                tip_net_mm=round(float(np.linalg.norm(tip[-1] - tip[s0])), 2),
                lag_growth_to_jump_mm=round(float(np.linalg.norm(lag[pre] - lag[s0])), 2),
                lag_growth_vec_mm=np.round(lag[pre] - lag[s0], 2).tolist(),
                err_j2_j3_at_jump_mrad=np.round(e[pre, 1:3], 1).tolist(),
                err_j2_j3_start_mrad=np.round(e[s0, 1:3], 1).tolist())
    return dict(stop=stop, run=Path(run).name, approach=[int(t0), int(end)], n_iters=len(its), max_uncommanded_mm=round(float(mv.max()), 2), jump_tick=jump_k,
                jump_ticks_before_align=int(end - jump_k), err_j2_abs_max_mrad=round(float(np.abs(e[:, 1]).max()), 1),
                err_j3_abs_max_mrad=round(float(np.abs(e[:, 2]).max()), 1), iters=its)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--window-s', type=float, default=.5)
    ap.add_argument('--json')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    out = []
    for r in a.runs:
        if not (Path(r) / 'events.jsonl').exists():
            continue
        res = analyse(r, a.window_s)
        if res is None:
            continue
        out.append(res)
        print(f"{res['run']}: jump {res['max_uncommanded_mm']} mm at tick {res['jump_tick']} ({res['jump_ticks_before_align']} before align), "
              f"|e2| max {res['err_j2_abs_max_mrad']} |e3| max {res['err_j3_abs_max_mrad']} mrad (believed frame), iters {res['n_iters']}")
        print('    at the stop:', json.dumps(res['stop']))
        if not a.quiet:
            for it in res['iters']:
                print('   ', ' '.join(f'{kk}={v}' for kk, v in it.items()))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
