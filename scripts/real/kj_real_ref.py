"""Real K-J reference traces (plain PD): plug tip and commanded target against the hand-over, in port axes
(x across, y up, z out of the port; depth = -z - gap). Reads guard.npz / policy.npz / events.jsonl of a run.

  python scripts/real/kj_real_ref.py RUN_DIR [RUN_DIR ...] [--npz OUT.npz] [--png OUT.png]
"""
import argparse, json, pathlib
import numpy as np

RUNS = {'2026-11-03/full_155211_align_full2': 'seat', '2026-11-03/full_164912_align_full5': 'stall',
        '2026-11-06/full_173941_align_full9': 'seat', '2026-11-06/full_175030_align_full10': 'stall'}


def load(run):
    run = pathlib.Path(run)
    g, p = np.load(run / 'guard.npz'), np.load(run / 'policy.npz')
    ev = [json.loads(l) for l in open(run / 'events.jsonl')]
    ho = next(e for e in ev if e['kind'] == 'align_handover')
    res = next(e for e in ev if e['kind'] == 'insert_result')
    n = min(len(g['t']), len(p['t']))
    return dict(name=run.name, t=p['t'][:n] * 0.02, tip=g['rel_m'][:n], tgt=g['rel_t'][:n], act=p['action'][:n],
                share=p['contact_share'][:n], depth=g['depth2_mm'][:n], gap=ho['gap_mm'], lat=ho['lat_mm'],
                lead=ho.get('lead'), outcome=res['outcome'], obs=p['obs'][:n], jt=p['joint_target'][:n])


def summary(d):
    tip, tgt, t = d['tip'], d['tgt'], d['t']
    first = lambda c: (float(t[np.argmax(c)]) if c.any() else None)  # noqa: E731
    out = dict(run=d['name'], outcome=d['outcome'], gap=round(d['gap'], 2), lat=[round(x, 2) for x in d['lat']],
               t_end=round(float(t[-1]), 2), depth_max=round(float(np.nanmax(d['depth'])), 2),
               t_depth_gt0=first(d['depth'] > 0), t_depth_gt3=first(d['depth'] > 3),
               tgt_up_max=round(float(tgt[:, 1].max()), 1), t_tgt_up_gt5=first(tgt[:, 1] > 5),
               tip_up_max=round(float(tip[:, 1].max()), 1), lead_up_end=round(float(tgt[-1, 1] - tip[-1, 1]), 1),
               lead_in_end=round(float(-(tgt[-1, 2] - tip[-1, 2])), 1))
    for ts in (0.3, 0.6, 1.0, 1.5, 2.0):
        i = min(int(ts / 0.02), len(t) - 1)
        out[f'@{ts}s tip(x,y,depth)/tgt'] = ([round(float(tip[i, 0]), 1), round(float(tip[i, 1]), 1), round(float(-tip[i, 2] - d['gap']), 1)],
                                            [round(float(tgt[i, 0]), 1), round(float(tgt[i, 1]), 1), round(float(-tgt[i, 2] - d['gap']), 1)])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='*')
    ap.add_argument('--root', default='data/real_runs')
    ap.add_argument('--npz'); ap.add_argument('--png')
    a = ap.parse_args()
    runs = a.runs or [f'{a.root}/{r}' for r in RUNS]
    ds = [load(r) for r in runs]
    for d in ds:
        print(json.dumps(summary(d)))
    if a.npz:
        np.savez(a.npz, **{f"{d['name']}__{k}": np.asarray(v, float) for d in ds for k, v in d.items() if k in ('t', 'tip', 'tgt', 'act', 'share', 'depth', 'obs', 'jt')})
    if a.png:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        f, ax = plt.subplots(4, len(ds), figsize=(4.2 * len(ds), 11), sharex=True, squeeze=False)
        for j, d in enumerate(ds):
            t = d['t']
            for i, (lab, k) in enumerate([('across mm', 0), ('up mm', 1), ('depth mm', 2)]):
                y = (lambda v: -v[:, 2] - d['gap']) if k == 2 else (lambda v, k=k: v[:, k])
                ax[i, j].plot(t, y(d['tip']), label='tip (FK)'); ax[i, j].plot(t, y(d['tgt']), '--', label='target')
                ax[i, j].set_ylabel(lab); ax[i, j].grid(alpha=.3)
            ax[0, j].set_title(f"{d['name']}\n{d['outcome']}, gap {d['gap']:.1f}", fontsize=9); ax[0, j].legend(fontsize=7)
            ax[3, j].plot(t, d['act']); ax[3, j].set_ylabel('action (j1..j6)'); ax[3, j].set_xlabel('s')
        f.tight_layout(); f.savefig(a.png, dpi=90)
