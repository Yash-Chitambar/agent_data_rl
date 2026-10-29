"""Compare all twelve recorded-rest replays without claiming transfer validation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
rows = []
for side in ('left', 'right'):
    for joint in range(1, 7):
        b = json.loads((a.directory / f'{side}_j{joint}_baseline.json').read_text())
        ff_path = a.directory / f'{side}_j{joint}_logged_feedforward.json'
        f = json.loads(ff_path.read_text()) if ff_path.exists() else None
        if f is not None and b['input_sha256'] != f['input_sha256']:
            raise ValueError('cannot compare different source recordings')
        rows.append({'side': side, 'joint': joint,
                     'baseline_rmse_mrad': b['joint_rmse_rad'][joint-1]*1000,
                     'logged_ff_rmse_mrad': f['joint_rmse_rad'][joint-1]*1000 if f else None,
                     'comparison_status': 'available' if f else 'no logged-feedforward replay available'})
report = {'trials': rows, 'interpretation': 'same recorded trajectories; no plant fit; logged feedforward execution timing uncertain; not held-out transfer validation'}
(a.directory / 'feedforward_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, side in zip(axes, ('left', 'right')):
    selected = [r for r in rows if r['side'] == side]
    x = np.arange(1, 7)
    ax.bar(x-.18, [r['baseline_rmse_mrad'] for r in selected], .36, label='Ideal gravity baseline')
    ax.bar(x+.18, [r['logged_ff_rmse_mrad'] if r['logged_ff_rmse_mrad'] is not None else np.nan for r in selected], .36, label='Logged driver feedforward')
    if all(r['logged_ff_rmse_mrad'] is None for r in selected):
        ax.text(.5, .95, 'Driver feedforward absent from these captures', transform=ax.transAxes, ha='center', va='top', fontsize=9)
    ax.set(title=f'{side.capitalize()} arm: folded-rest trials', xlabel='Excited joint', xticks=x)
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('Excited-joint RMSE (mrad)')
axes[1].legend()
fig.suptitle('Controller input audit — unchanged plant physics, not a transfer accuracy claim')
fig.tight_layout()
fig.savefig(a.directory / 'feedforward_comparison.png', dpi=160)
print(json.dumps(report, indent=2))
