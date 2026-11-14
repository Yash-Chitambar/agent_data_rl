"""Table of the edge-fit dry runs (docs/real/ZED_EDGEFIT_20261115.md): per run folder under results/placement/zed_edgefit/dry,
the capture estimate (net / start / fit error against the sim truth, gate), the first wrist-net reading at the look pose,
the align result, and the insertion outcome.

    PYTHONPATH=. python scripts/perception/zed_edgefit_dry_summary.py results/placement/zed_edgefit/dry
"""
import json
import sys
from pathlib import Path


def row(run):
    ev = [json.loads(l) for l in open(run / 'events.jsonl') if l.strip()]
    meta = json.loads((run / 'meta.json').read_text())
    e = next((x for x in ev if x.get('kind') == 'estimate' and x.get('est') == 'zed_pc'), {})
    first = next((x['out'][0] for x in ev if x.get('kind') == 'net' and 'wrist_r10w' in x.get('net', '')), None)
    al = next((x for x in ev if x.get('kind') == 'align_result'), {})
    ins = next((x for x in ev if x.get('kind') == 'insert_result'), {})
    f = e.get('edge_fit', {})
    rf = f.get('render_fit', {})
    return dict(run=run.name, cfg=Path(meta['args']['config']).name, seed=meta.get('seed'), sim_phys=meta['args'].get('sim_phys', ''),
                gate_ok=e.get('ok'), used_err_mm=e.get('err_xy_mm'), net_err_mm=(e.get('net') or {}).get('err_xy_mm', e.get('err_xy_mm')),
                fit_src=f.get('frame_source'), fit_ok=f.get('ok'), fit_why=f.get('why'), shift_mm=f.get('shift_mm'),
                render_fit_err=None if not rf else [round((a - b) * 1e3, 1) for a, b in zip(rf['fit_xy'], e['true_port_xy'])],
                first_wrist=None if first is None else [round(v, 2) for v in first[:3]],
                align=(al.get('ok'), al.get('reason'), al.get('cycles'), al.get('seconds')), insert=ins.get('outcome'),
                true_seated=ins.get('true_seated_end'), outcome=meta.get('outcome'))


def main(root):
    rows = [row(p.parent) for p in sorted(Path(root).rglob('events.jsonl')) if (p.parent / 'meta.json').exists()]
    for r in rows:
        print(json.dumps(r))
    Path(root, 'summary.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
