"""Tables of scripts/eval/mating_force_eval.py results (docs/real/MATING_FORCE_20261119.md).

    python scripts/eval/mating_force_table.py results/real_align/mating_force/*.json > results/real_align/mating_force/table.md
"""
import json
import sys


def r(x):
    return f"{100 * x['rate']:.1f} [{100 * x['wilson'][0]:.1f}, {100 * x['wilson'][1]:.1f}] ({x['k']}/{x['n']})"


def rng(v, i=(0, 2)):
    return '-' if v is None else f'{v[i[0]]:.2f}-{v[i[1]]:.2f}'


rows = []
for f in sys.argv[1:]:
    rows += json.load(open(f))
print('## Insertion and seat detector (all worlds of a run)\n')
print('| Policy | Cond | n | Sim seat | FK detector fired | Detector with plug > 1 mm short | > 0.5 mm short | True depth at detector p5-p95 (mm) |')
print('|---|---|---|---|---|---|---|---|')
seen = set()
for d in rows:
    k = (d['tag'], d['cond'], d['n'], d.get('groups_spec'))
    if k in seen:
        continue
    seen.add(k)
    print(f"| {d['tag']} | {d['cond']} | {d['n']} | {r(d['sim_seat'])} | {r(d['det'])} | {r(d['det_short_gt_1mm'])} | {r(d['det_short_gt_0p5mm'])} | {rng(d['det_true_depth_mm_p5_p50_p95'])} |")
print('\n## Push outcome per group\n')
print('| Policy | Cond | Group | n | Pushed | Watch stop (of pushed) | Full under load | Full at rest (0 N, jaws closed) | Rest short 1-2 mm | Rest short > 2 mm | Short and tilted > 3 deg | Rest depth p5-p50-p95 (mm) | FK adv end p5-p95 (mm) | Cmd axial at end p5-p95 (N) | Reader turn in jaws, hand-over to push p50 (deg) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for d in rows:
    for g, v in d['groups'].items():
        rd = v['rest_depth_mm_p5_p50_p95']
        turn = v.get('reader_turn_in_jaws_handover_to_push_deg_p5_p50_p95')
        print(f"| {d['tag']} | {d['cond']} | {g} | {v['n']} | {v['pushed_n']} | {r(v['watch_stop'])} | {r(v['full_under_load'])} | {r(v['full_at_rest'])} | "
              f"{r(v['rest_short_1_2'])} | {r(v['rest_short_gt2'])} | {r(v['short_tilted_gt3deg'])} | {'-' if rd is None else '-'.join(f'{x:.2f}' for x in rd)} | "
              f"{rng(v['fk_adv_end_mm_p5_p50_p95'])} | {rng(v['f_cmd_axial_end_n_p5_p50_p95'])} | {'-' if not turn else f'{turn[1]:.1f}'} |")
