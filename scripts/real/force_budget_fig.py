"""Figure: hand advance against the estimated force at the plug for the logged pushes, with the fitted 1-D plant
(results/real_align/force_budget/push_traces.png). Run after force_budget.py contact and plant_fit.py."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results/real_align/force_budget'
z = np.load(OUT / 'plant_fit_traces.npz')
lab = {c['run']: c['label'] for c in json.loads((OUT / 'contact.json').read_text())}
runs = sorted({k.split('__')[0] for k in z.files})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for n in runs:
    F, a, p, jd = z[n + '__F'], z[n + '__adv'], z[n + '__pred'], int(z[n + '__jd'])
    test = n.startswith('all9')
    col = {'full': '#2a7', 'short': '#c43'}.get(lab.get(n), '#888')
    ax[0].plot(F[:jd], a[:jd], color=col, lw=1.8 if test else 0.8, alpha=0.9)
    t = np.arange(len(a)) * 0.02
    if test:
        ax[1].plot(t, a, color=col, lw=1.5, label=f'{n} real')
        ax[1].plot(t, p, color=col, lw=1.2, ls='--', label=f'{n} plant fit')
ax[0].set_xlabel('estimated force at the plug, along the port axis (N)'); ax[0].set_ylabel('hand advance, encoder FK (mm)')
ax[0].set_title('Load ramps, 12 pushes (green: fully in n=4, red: short n=4, grey: unlabelled n=4; thick: force pushes)', fontsize=8)
ax[1].set_xlabel('time from the push start (s)'); ax[1].set_ylabel('hand advance (mm)')
ax[1].set_title('Held-out force pushes (2026-11-13), n=2: real and the fitted plant', fontsize=8); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / 'push_traces.png', dpi=130)
