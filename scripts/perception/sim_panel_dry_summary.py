"""Table of the edge-fit dry runs with the sim fit frame (docs/real/SIM_RELEASE_PANEL_20261116.md): per run folder, the fit
frame source, the fit error against the sim truth, the fit gate, the v1 / zed_v5 errors, the insertion and the release.

    PYTHONPATH=. python scripts/perception/sim_panel_dry_summary.py results/placement/sim_panel/dry/ef [--md out.md]
"""
import argparse
import json
from pathlib import Path


def row(run):
    ev = [json.loads(line) for line in open(run / 'events.jsonl') if line.strip()]
    meta = json.loads((run / 'meta.json').read_text())
    e = next((x for x in ev if x.get('kind') == 'estimate' and x.get('est') == 'zed_pc'), {})
    f = e.get('edge_fit') or {}
    ins = next((x for x in ev if x.get('kind') == 'insert_result'), {})
    res = (meta.get('result') or {}).get('results') or {}
    pt = run.parent / 'point.json'
    return dict(run=run.name, seed=meta.get('seed'), point=json.loads(pt.read_text())['pc'] if pt.exists() else None,
                frame=f.get('frame_source'), fit_ok=f.get('ok'), why=f.get('why'), edges=f.get('edges_used'),
                fit_err_mm=[round(v, 2) for v in f['err_mm']] if 'err_mm' in f else None,
                fit_err_xy_mm=None if 'err_xy_mm' not in f else round(f['err_xy_mm'], 2),
                fit_err_yaw_deg=None if 'err_yaw_deg' not in f else round(f['err_yaw_deg'], 2),
                v1_err_mm=None if 'net' not in e else round(e['net'].get('err_xy_mm', float('nan')), 2),
                v5_err_mm=None if 'start' not in e else round(e['start'].get('err_xy_mm', float('nan')), 2),
                shift_mm=None if 'shift_mm' not in f else round(f['shift_mm'], 2),
                insert=ins.get('outcome') or res.get('insert', {}).get('outcome'),
                seated_after_release=res.get('true_seated_after_release'), outcome=meta.get('outcome'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--md', default=None)
    a = ap.parse_args()
    rows = [row(p.parent) for p in sorted(Path(a.root).rglob('events.jsonl'))]
    lines = ['| run | PC (x, y m, yaw deg) | fit frame | fit gate | fit err x, y (mm) | \\|err\\| (mm) | yaw err (deg) | v1 err (mm) | zed_v5 err (mm) | shift (mm) | insert | seated after release |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['run']} | {r['point'] or 'seed ' + str(r['seed'])} | {r['frame']} | {'ok' if r['fit_ok'] else 'REJECT: ' + '; '.join(r['why'] or [])} | "
                     f"{r['fit_err_mm']} | {r['fit_err_xy_mm']} | {r['fit_err_yaw_deg']} | {r['v1_err_mm']} | {r['v5_err_mm']} | {r['shift_mm']} | {r['insert']} | {r['seated_after_release']} |")
    print('\n'.join(lines))
    if a.md:
        Path(a.md).write_text('\n'.join(lines) + '\n')
        Path(a.md).with_suffix('.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
