"""Figures and tables of docs/real/PERCEPTION_BIAS_20261115.md from results/placement/perception_bias/*.json.

    PYTHONPATH=. python scripts/perception/zed_bias_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[2] / 'results/placement/perception_bias'


def finger_sweep():
    files = {0.0: 'twin_sim_left_closed.json', 10.0: 'twin_sim_lf0.01.json', 20.0: 'twin_sim_lf0.02.json', 30.0: 'twin_sim_lf0.03.json',
             40.0: 'twin_sim_lf0.04.json', 47.5: 'twin_sim.json'}
    xs, med, lo, hi, real = [], [], [], [], None
    for mm, f in files.items():
        d = json.loads((R / f).read_text())
        rows = d['rows']
        dy = np.array([r['twin_v5'][1] - r['truth'][1] for r in rows])
        xs.append(mm); med.append(np.median(dy)); lo.append(np.percentile(dy, 10)); hi.append(np.percentile(dy, 90))
        real = d['summary']['v5']['real_minus_edge_med'][1]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=130)
    ax.fill_between(xs, lo, hi, color='#9db4d6', alpha=.5, lw=0, label='sim twin p10-p90')
    ax.plot(xs, med, 'o-', color='#1f4e8c', label='sim twin median')
    ax.axhline(real, color='#c0392b', ls='--', label=f'real frames median ({real:+.1f} mm)')
    ax.axhline(0, color='0.5', lw=.8)
    ax.set_xlabel('left finger opening in the sim twin (mm per finger; training: 47.5)')
    ax.set_ylabel('zed_v5 y error (mm)')
    ax.set_title('zed_v5 y error against the left gripper opening (n = 89 per point)', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(R / 'finger_sweep.png')


def wrist_bias():
    d = json.loads((R / 'wrist_bias.json').read_text())
    T = d['table']
    keys = [('tip_cy_stop', 'plug tip y in the camera at the stop (mm)'), ('vis_gap', 'vision gap at the hand-over (mm)'), ('tip_cz', 'plug tip in the camera, z (mm)'),
            ('bright_crop', 'wrist frame brightness (grey, net crop)'), ('net_ry', 'net rotation ry (deg)'), ('g_rot_deg', 'grasp rotation at the squeeze (deg)')]
    keys = [k for k in keys if k[0] in T[0]]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=110)
    dates = sorted({t['date'] for t in T})
    cols = dict(zip(dates, plt.cm.viridis(np.linspace(0, .9, len(dates)))))
    for ax, (k, lab) in zip(axs.ravel(), keys):
        for dt in dates:
            S = [t for t in T if t['date'] == dt]
            ax.scatter([t[k] for t in S if t['handover']], [t['d_up'] for t in S if t['handover']], color=cols[dt], s=20, label=dt)
            ax.scatter([t[k] for t in S if not t['handover']], [t['d_up'] for t in S if not t['handover']], facecolor='none', edgecolor=cols[dt], s=20)
        c, ch = d['corr'].get(k, {}), d['corr_handover'].get(k, {})
        ax.set_title(f"rho {c.get('spearman', float('nan')):+.2f} (n {c.get('n', 0)} runs); hand-over runs {ch.get('spearman', float('nan')):+.2f} (n {ch.get('n', 0)})", fontsize=8)
        ax.set_xlabel(lab, fontsize=8); ax.set_ylabel('net up - vision up (mm)', fontsize=8)
    axs.ravel()[0].legend(fontsize=6)
    fig.suptitle('Wrist-net up minus port-vision up, gap <= 10 mm (one point per run; filled: runs with a hand-over)', fontsize=10)
    fig.tight_layout()
    fig.savefig(R / 'wrist_bias.png')


if __name__ == '__main__':
    finger_sweep()
    if (R / 'wrist_bias.json').exists() and 'corr_handover' in json.loads((R / 'wrist_bias.json').read_text()):
        wrist_bias()
