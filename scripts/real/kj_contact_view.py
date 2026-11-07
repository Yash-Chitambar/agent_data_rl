"""K-J's contact head on real (or dry-run) insertion logs: per-point class (free / touching / inside) on the plug's
bottom edge, re-computed from the logged actor observations (policy.npz obs; 30-frame history, first frame repeated
as in run_insertion). Same head as Bronars et al.'s "predicted contact points"; the policy never acts on it.

  python scripts/real/kj_contact_view.py RUN_DIR [...] --png OUT.png [--times 0.3,0.6,1.0,1.5]
"""
import argparse, json, pathlib, sys
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402

POL = ROOT / 'deploy/bundle/models/insert_KJ.pt'
COL = np.array([[.62, .62, .62], [.95, .55, .1], [.15, .65, .3]])   # free, touching, inside
NAMES = ('free', 'touching', 'inside')


def outline():
    v = B.usbc_spec_variant()
    return B._resample_outline(B.extract_pieces(v).peg_poly, 64) * 1e3


def head(pol, obs):
    """obs (T, 68) logged -> probs (T, 64, 3)."""
    o = torch.as_tensor(obs, dtype=torch.float32)
    H = int(pol.meta['history'])
    hist = o[:1].expand(H, -1).clone()
    ac = pol.members[0][0]
    out = []
    for t in range(len(o)):
        hist = torch.cat([hist[1:], o[t:t + 1]], 0)
        h = hist[None, :, :pol.obs_dim].clone()
        if pol.mask_idx:
            h[..., list(pol.mask_idx)] = 0
        with torch.no_grad():
            z, _ = ac.encode(h.to(next(ac.parameters()).device))
            out.append(torch.softmax(ac.contact_logits(z), -1)[0].cpu().numpy())
    return np.stack(out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--png', required=True)
    ap.add_argument('--times', default='0.2,0.4,0.6,0.8,1.0,1.5')
    ap.add_argument('--json')
    ap.add_argument('--policy', default=str(POL), help='checkpoint (K-J default; K3: obs_dim 71, logged obs used unchanged)')
    a = ap.parse_args()
    pol = InsertionPolicy(a.policy, 'cpu')
    xy = outline()
    ts = [float(x) for x in a.times.split(',')]
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    f, ax = plt.subplots(len(a.runs), len(ts) + 1, figsize=(2.3 * (len(ts) + 1) + 2, 1.9 * len(a.runs)), squeeze=False,
                         gridspec_kw=dict(width_ratios=[1] * len(ts) + [2.2]))
    summ = {}
    for i, r in enumerate(a.runs):
        r = pathlib.Path(r)
        p = np.load(r / 'policy.npz')
        if p['obs'].shape[1] != pol.obs_dim:
            raise SystemExit(f"{r.name}: logged obs has {p['obs'].shape[1]} dims, the policy {pol.obs_dim}: use the policy the run used")
        P = head(pol, p['obs'])
        t = p['t'] * 0.02
        g = np.load(r / 'guard.npz') if (r / 'guard.npz').exists() else None
        depth = g['depth2_mm'][:len(t)] if g is not None else None
        res = [json.loads(l) for l in open(r / 'events.jsonl') if '"insert_result"' in l]
        out = res[0]['outcome'] if res else '?'
        for j, ts_ in enumerate(ts):
            k = min(int(round(ts_ / 0.02)), len(t) - 1)
            c = P[k].argmax(-1)
            ax[i, j].scatter(xy[:, 0], xy[:, 1], c=COL[c], s=22)
            ax[i, j].set_aspect('equal'); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ttl = f'{t[k]:.2f} s'
            if depth is not None:
                ttl += f'\nFK depth {depth[k]:.1f}'
            ax[i, j].set_title(ttl, fontsize=8)
        sh = P.mean(1)
        for c in range(3):
            ax[i, -1].plot(t, sh[:, c], color=COL[c], label=NAMES[c])
        if depth is not None:
            a2 = ax[i, -1].twinx(); a2.plot(t, depth, 'k:', lw=1); a2.set_ylabel('FK depth mm', fontsize=7)
        ax[i, -1].set_ylim(0, 1); ax[i, -1].set_title(f'{r.name}: {out}', fontsize=8); ax[i, -1].tick_params(labelsize=7)
        ax[i, 0].set_ylabel(r.name.split('_', 2)[-1], fontsize=8)
        summ[r.name] = dict(outcome=out, **{f'share@{x}s': P[min(int(round(x / .02)), len(t) - 1)].mean(0).round(2).tolist() for x in ts},
                            t_first_inside50=float(t[np.argmax(sh[:, 2] > .5)]) if (sh[:, 2] > .5).any() else None)
    ax[0, -1].legend(fontsize=7, loc='upper left')
    f.tight_layout(); f.savefig(a.png, dpi=110)
    print(json.dumps(summ, indent=1))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(summ, indent=1))
