"""Safe test box from the wide-placement grid (docs/real/WIDE_PLACEMENT_READY_20261116.md): per PC pose, the first
failing chain node (wide_grid_summary.py) and the clearance flags (wide_clearance.py) against the same-seed sim
nominal pose (pc +0 +0 +0) and the real-run reference (results/placement/wide/clearance_real.json).

Flags (a pose is 'clear' when it has none):
  AA   arm-arm model distance in a stage (or the home move) under 30 mm AND more than 5 mm below both the same-seed
       sim nominal and the smallest value of 13 real runs (the arms are closer than on any proven run)
  TAB  a table margin (FlexChecker body clearances, motor frame: what SafeArm checks) under 5 mm in the hold/align/
       insert/release stages ('STOP' when under 0: SafeArm would stop the run there); the right fingertips in the
       pick are commanded near the table by design and are left out
  J    a joint margin to the SafeArm joint box under 0.30 rad
  CAM  left wrist camera (visual) to the PC more than 8 mm below the nominal pose of the same seed
    PYTHONPATH=. python scripts/eval/wide_safe_box.py --summary results/placement/wide/summary.json --out results/placement/wide/safe_box.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGES = ('hold', 'transport', 'approach', 'align', 'insert', 'release', 'home_move')


REAL = {}


def flags(c, nom):
    f = []
    if not c:
        return ['no clearance data']
    for st in STAGES:
        if st in c and st in nom and 'arm_arm' in c[st] and 'arm_arm' in nom[st]:
            ref = min(nom[st]['arm_arm'][0], REAL.get(st, 1e9))
            if c[st]['arm_arm'][0] < 30.0 and c[st]['arm_arm'][0] < ref - 5.0:
                f.append(f"AA {st} {c[st]['arm_arm'][0]:.0f} (nominal {nom[st]['arm_arm'][0]:.0f}) mm")
    for st in ('hold', 'align', 'insert', 'release'):
        for k in ('table_right', 'table_left'):
            if st in c and k in c[st] and c[st][k][0] < 5.0:
                f.append(f"TAB{' STOP' if c[st][k][0] < 0 else ''} {k.split('_')[1]} {c[st][k][1]} {c[st][k][0]:.1f} mm")
    for st, v in c.items():
        for k in ('joint_right', 'joint_left'):
            if k in v and v[k][0] < 0.30:
                f.append(f"J {k.split('_')[1]} joint {v[k][1]} {v[k][0]:.2f} rad ({st})")
    lc = min((v['left_body_pc'][0] for v in c.values() if 'left_body_pc' in v), default=None)
    ln = min((v['left_body_pc'][0] for v in nom.values() if 'left_body_pc' in v), default=None)
    if lc is not None and ln is not None and lc < ln - 8.0:
        f.append(f'CAM left camera-PC {lc:.0f} (nominal {ln:.0f}) mm')
    f = [x for x in f]
    return sorted(set(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', default=str(ROOT / 'results/placement/wide/summary.json'))
    ap.add_argument('--out', default=str(ROOT / 'results/placement/wide/safe_box.json'))
    a = ap.parse_args()
    R = json.loads(Path(a.summary).read_text())
    for r in json.loads((ROOT / 'results/placement/wide/clearance_real.json').read_text())['rows']:
        for st, v in r['mins'].items():
            if 'arm_arm' in v:
                REAL[st] = min(REAL.get(st, 1e9), v['arm_arm'][0])
    nom = {}
    for r in R:
        if r['tag'] in ('k7_pc+0_+0_+0_s7043', 'k4_pc+0_+0_+0_s7043'):
            nom[(r['policy'], r['seed'])] = r['clearance']
        if r['tag'] == 'k7_rd+0_+0_y110_s7045':     # no (0, 0, 0) PC run with seed 7045: nearest is the reader-yaw run at PC (0, +4, 0)
            nom[('k7', 7045)] = r['clearance']
    out = []
    for r in R:
        if not r['tag'].startswith(('k7_pc', 'k4_pc', 'k7_c_')):
            continue
        n = nom.get((r['policy'], r['seed'])) or nom.get((r['policy'], 7043))
        fl = flags(r['clearance'], n or {})
        out.append(dict(tag=r['tag'], policy=r['policy'], seed=r['seed'], d=r['d'], rd=r['rd'], node=r['node'], why=r.get('why'),
                        depth_end_mm=r.get('true_depth_end_mm'), flags=fl))
    Path(a.out).write_text(json.dumps(out, indent=1))
    by = defaultdict(list)
    for o in out:
        by[tuple(o['d'])].append(o)
    for d in sorted(by, key=lambda d: (d[2], d[0], d[1])):
        print(d, ' || '.join(f"{o['tag'].split('_pc')[0]} s{o['seed']}: {o['node']}" + (f" [{'; '.join(o['flags'])}]" if o['flags'] else '') for o in by[d]))


if __name__ == '__main__':
    main()
