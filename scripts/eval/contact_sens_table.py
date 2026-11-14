"""Tables of docs/real/CONTACT_SENS_20261115.md from results/real_align/contact_sens/raw/*.json (contact_sens.py, funnel_rings.py).

    python scripts/eval/contact_sens_table.py   -> results/real_align/contact_sens/{test1,test2}.{md,json}
"""
from __future__ import annotations

import json
from pathlib import Path

D = Path(__file__).resolve().parents[2] / 'results/real_align/contact_sens'
ORDER = ['base', 'pmu_lo', 'pmu_hi', 'pmu_vhi', 'ptc_2ms', 'ptc_8ms', 'pdamp_05', 'psolimp_soft', 'padmu_08', 'padmu_20', 'padtc_6ms',
         'grip_06', 'grip_14']
DESC = {'base': 'training contact (port mu 0.10-0.20, port solref 3-4 ms)', 'pmu_lo': 'port mu 0.05-0.08', 'pmu_hi': 'port mu 0.35-0.60',
        'pmu_vhi': 'port mu 0.80-1.00', 'ptc_2ms': 'port solref 2 ms (stiffer)', 'ptc_8ms': 'port solref 8 ms (softer)',
        'pdamp_05': 'port damping ratio 0.5', 'psolimp_soft': 'port solimp (0.90, 0.95)', 'padmu_08': 'pad mu 0.8',
        'padmu_20': 'pad mu 2.0', 'padtc_6ms': 'pad solref 6 ms', 'grip_06': 'squeeze x0.6', 'grip_14': 'squeeze x1.4'}
FAIL = ('lateral_miss', 'grasp_slip', 'tilt', 'jam', 'deep_unseated', 'pc_pushed', 'nonfinite', 'full_no_seat_test')


def fails(c):
    x = [f'{k} {v}' for k, v in sorted(c.items(), key=lambda kv: -kv[1]) if k != 'seated']
    return ', '.join(x) if x else '-'


def row(r):
    t = r['t_seat_p50_p90_p99']
    return (f"{100 * r['rate']:.1f} [{100 * r['wilson'][0]:.1f}, {100 * r['wilson'][1]:.1f}] | {100 * r['full_depth_rate']:.1f} | "
            f"{t[0]:.2f} / {t[2]:.2f} | {r['release_in_k']}/{r['release_n']} | {fails(r['classes'])}" if t else
            f"{100 * r['rate']:.1f} | {100 * r['full_depth_rate']:.1f} | - | {r['release_in_k']}/{r['release_n']} | {fails(r['classes'])}")


def test1():
    rows = []
    for f in sorted((D / 'raw').glob('*.json')):
        if f.name.startswith('funnel_'):
            continue
        rows += json.loads(f.read_text())
    out = ['| Policy | Place | Pads | Plug | Variant | In training | n | Seat % [95 % CI] | Full depth % | Seat time p50 / p99 s | In after release | Failures |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    key = lambda r: (r['tag'], r['place'], r['pad'] != '4x1.2', r['pad'], r['plug'] != 'nom', r['plug'], ORDER.index(r['variant']))  # noqa: E731
    for r in sorted(rows, key=key):
        out.append(f"| {r['tag'].upper()} | {r['place']} | {r['pad']} | {r['plug']} | {r['variant']}: {DESC[r['variant']]} | "
                   f"{'yes' if r['in_training'] else 'no'} | {r['n']} | {row(r)} |")
    (D / 'test1.md').write_text('\n'.join(out) + '\n')
    (D / 'test1.json').write_text(json.dumps(rows, indent=1))
    return rows


def test2():
    rows = []
    for f in sorted((D / 'raw').glob('funnel_*.json')):
        rows += json.loads(f.read_text())
    out = ['| Policy | Prior mode | Ring mm | n | Seat % [95 % CI] | Full depth % | Seat time p50 / p90 / p99 s | Rest depth p5 / p50 mm | In after release | Failures |',
           '|---|---|---|---|---|---|---|---|---|---|']
    for r in sorted(rows, key=lambda r: (r['tag'], r['mode'], r['ring_mm'])):
        t = r['t_seat_p50_p90_p99'] or [float('nan')] * 3
        dr = r['rest_depth_mm_p5_p50'] or [float('nan')] * 2
        out.append(f"| {r['tag'].upper()} | {r['mode']} | {r['ring_mm']} | {r['n']} | {100 * r['rate']:.1f} [{100 * r['wilson'][0]:.1f}, "
                   f"{100 * r['wilson'][1]:.1f}] | {100 * r['full_depth_rate']:.1f} | {t[0]:.2f} / {t[1]:.2f} / {t[2]:.2f} | {dr[0]:.2f} / {dr[1]:.2f} | "
                   f"{r['release_in_k']}/{r['release_n']} | {fails(r['classes'])} |")
    (D / 'test2.md').write_text('\n'.join(out) + '\n')
    (D / 'test2.json').write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == '__main__':
    a, b = test1(), test2()
    print((D / 'test1.md').read_text())
    print((D / 'test2.md').read_text())
