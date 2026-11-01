"""Draw the GaP-style graph of the bimanual USB-C policy (stages, models, gates, retries, measured rates).

    python scripts/report/policy_graph.py --out results/usbc/policy/chain/graph/policy_graph
Numbers: realistic chain (believed frame, +/-1.5 deg encoder offsets, full DR, RGB only),
chain_real_servo_bf1.5_n2048 + r9chain_ih5_v18ws_n1024 (docs/campaign/FINDINGS.md, 2026-11-01).
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C = {'percept': '#dbeafe', 'script': '#e5e7eb', 'learned': '#dcfce7', 'gate': '#fef3c7', 'fail': '#fee2e2', 'done': '#bbf7d0'}
EDGE = {'percept': '#1d4ed8', 'script': '#374151', 'learned': '#15803d', 'gate': '#b45309', 'fail': '#b91c1c', 'done': '#15803d'}

# (key, x, y, w, h, kind, title, body)
NODES = [
    ('zedpc', 0, 0, 5.2, 1.25, 'percept', 'ZED frame #1: PC pose',
     'capture pose, grippers open\nZED PC net + presence/OOD + arm-pose gate'),
    ('zedrd', 0, -1.9, 5.2, 1.25, 'percept', 'Park left arm -> ZED frame #2: reader pose',
     'reader ZED net + presence + arm-envelope gate'),
    ('gate1', 0, -3.65, 3.4, 0.9, 'gate', 'Reader accepted?', ''),
    ('hover', 0, -5.3, 5.2, 1.25, 'script', 'Pick: hover above reader (encoder FK)',
     'wrist frame -> servo_hover net (grasp rel. to hand)'),
    ('gate2', 0, -7.05, 3.4, 0.9, 'gate', 'Correction > 8 mm?', ''),
    ('descent', 0, -8.7, 5.2, 1.4, 'percept', 'Servo descent: frames at 50 / 80 / 92 %',
     'servo_near (position) + near6 (rotation)\nshort straight move, close 12 N, lift'),
    ('hold', 0, -10.6, 5.2, 1.25, 'script', 'Left arm hold',
     'guarded press 25 N, 2 cm in front of case centre\nencoder force estimate, latch + K^-1 J^T F'),
    ('transport', 0, -12.35, 5.2, 1.0, 'script', 'Transport to look pose', 'Cartesian path, reorient plug to port'),
    ('approach', 0, -14.0, 5.2, 1.4, 'percept', 'Approach: wrist visual servo',
     'wrist_v14w -> wrist_v18ws (< 7 mm), friction FF\nroll fix, stop 6 mm from port'),
    ('inhand', 0, -15.85, 5.2, 1.0, 'percept', 'In-hand estimate at the stop', 'inhand_ih5: reader pose in hand (6-D)'),
    ('handover', 0, -17.35, 5.2, 0.8, 'script', 'Bumpless torque hand-over', ''),
    ('insert', 0, -19.0, 5.2, 1.6, 'learned', 'Insertion: PPO policy (K-J)',
     'TCN over 30-step joint history + in-hand est.\ncontact head, force/speed penalties\nseated detector (FK depth)'),
    ('done', 0, -21.0, 4.2, 0.9, 'done', 'Seated -> release, retreat', ''),
    ('retake', 4.6, -3.65, 2.6, 0.9, 'script', 'Re-take frame (1x)', ''),
    ('reject', 8.3, -3.65, 2.5, 1.0, 'fail', 'Give up:\nreader rejected', ''),
    ('rehover', 4.6, -7.05, 2.6, 0.9, 'script', 'Re-hover (<= 2x)', ''),
]
STATS = {
    'zedpc': 'accept 100 %', 'gate1': 'reject 5 / 2048', 'descent': 'pick 2043 / 2048\n(99.6 % alone, any offset 0.25-1.5 deg)',
    'hold': 'PC still 100 % at 3-10 N push', 'transport': '100 %', 'approach': 'lateral p99 1.47 mm\ntilt p99 1.63 deg',
    'inhand': 'err p99 1.4 mm / 1.8 deg', 'insert': 'seated 100 % of reached (r9)\npush p99 ~15 N, max < 25 N',
    'gate2': 'fires in ~2/3 of picks',
}


def box(ax, key, x, y, w, h, kind, title, body):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.04,rounding_size=0.15',
                                fc=C[kind], ec=EDGE[kind], lw=1.6))
    if body:
        ax.text(x, y + h / 2 - 0.22, title, ha='center', va='top', fontsize=10.5, weight='bold')
        ax.text(x, y - 0.12, body, ha='center', va='center', fontsize=8.6, color='#1f2937', linespacing=1.25)
    else:
        ax.text(x, y, title, ha='center', va='center', fontsize=10, weight='bold')
    if key in STATS:
        ax.text(x - w / 2 - 0.25, y, STATS[key], ha='right', va='center', fontsize=8.6, color='#065f46', style='italic')


def arrow(ax, a, b, label='', color='#374151', rad=0.0):
    ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='-|>', color=color, lw=1.4, connectionstyle=f'arc3,rad={rad}'))
    if label:
        ax.text((a[0] + b[0]) / 2 + 0.12, (a[1] + b[1]) / 2, label, fontsize=8.5, color=color, va='center')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='results/usbc/policy/chain/graph/policy_graph'); a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(13, 18.5))
    N = {k: (x, y, w, h) for k, x, y, w, h, *_ in NODES}
    for n in NODES:
        box(ax, *n)
    bot = lambda k: (N[k][0], N[k][1] - N[k][3] / 2)
    top = lambda k: (N[k][0], N[k][1] + N[k][3] / 2)
    right = lambda k: (N[k][0] + N[k][2] / 2, N[k][1])
    left = lambda k: (N[k][0] - N[k][2] / 2, N[k][1])
    seq = ['zedpc', 'zedrd', 'gate1', 'hover', 'gate2', 'descent', 'hold', 'transport', 'approach', 'inhand', 'handover', 'insert', 'done']
    for p, q in zip(seq, seq[1:]):
        lab = {'gate1': 'yes', 'gate2': 'no'}.get(p, '')
        arrow(ax, bot(p), top(q), lab, '#15803d' if lab else '#374151')
    arrow(ax, right('gate1'), left('retake'), 'no', '#b45309')
    arrow(ax, (N['retake'][0], N['retake'][1] + .45), (N['zedrd'][0] + 2.6, N['zedrd'][1] - .3), 'park pose, new frame', '#b45309', rad=.25)
    arrow(ax, right('retake'), left('reject'), '', '#b91c1c')
    ax.text((right('retake')[0] + left('reject')[0]) / 2, N['retake'][1] + .3, 'no again', fontsize=8.5, color='#b91c1c', ha='center')
    arrow(ax, right('gate2'), left('rehover'), 'yes', '#b45309')
    arrow(ax, (N['rehover'][0], N['rehover'][1] + .45), (N['hover'][0] + 2.6, N['hover'][1] - .3), 'new hover, new frame', '#b45309', rad=.25)
    ax.text(5.2, -11.0, 'press engaged before the approach;\nthe wrist servo measures the live port,\nso PC motion from the press is absorbed', fontsize=8.5, color='#374151')
    ax.text(5.2, -19.0, 'actor input: joint encoders, velocities,\nprevious actions, target errors,\nTCP pose/twist (FK), in-hand estimate\n6 right-arm joint increments, 50 Hz,\naccumulated targets (cap 0.05 rad)',
            fontsize=8.5, color='#374151', va='center')
    legend = [('perception (RGB nets)', 'percept'), ('scripted / IK on encoder FK', 'script'), ('learned RL policy', 'learned'),
              ('gate / decision', 'gate'), ('failure exit', 'fail')]
    for i, (t, k) in enumerate(legend):
        ax.add_patch(FancyBboxPatch((5.0, 0.4 - i * 0.55), 0.45, 0.32, boxstyle='round,pad=0.02', fc=C[k], ec=EDGE[k]))
        ax.text(5.6, 0.56 - i * 0.55, t, fontsize=9, va='center')
    ax.set_title('Bimanual USB-C insertion policy graph (GaP-style)\n'
                 'Realistic sim chain: encoders off by up to +/-1.5 deg, full visual + physics DR, RGB only\n'
                 'end to end 99.32 % (n = 2048, round-8 nets) -> 99.80 % (n = 1024, round-9 nets ih5 + v18ws)',
                 fontsize=12.5, weight='bold')
    ax.set_xlim(-5.6, 9.6); ax.set_ylim(-21.8, 1.2); ax.axis('off')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(f'{a.out}.{ext}', dpi=130)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
