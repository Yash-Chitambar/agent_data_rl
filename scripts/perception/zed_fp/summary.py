"""Tables for docs/real/FOUNDATIONPOSE_20261116.md from eval_sim.py / eval_real.py outputs (any python with numpy).

    python scripts/perception/zed_fp/summary.py --sim results/perception/fp/sim_eval.json \
        --sim-raw results/perception/fp/sim_eval_noplanefix.json --real results/perception/fp/real_zed_gt.json \
        --out results/perception/fp/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def st(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return dict(n=0)
    return dict(n=int(len(v)), p50=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                p90=float(np.percentile(v, 90)), mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def fmt(s, k=('p50', 'min', 'max'), d=2):
    if not s.get('n'):
        return '-'
    return f"{s['p50']:+.{d}f} [{s['min']:+.{d}f}, {s['max']:+.{d}f}]"


def sim_table(path):
    J = json.loads(Path(path).read_text())
    rows = J['rows']
    out = {'n': len(rows)}
    for v in ('oracle', 'stereo', 'full'):
        R = [r[v] for r in rows if 'dxy_mm' in r[v]]
        out[v] = dict(n_ok=len(R), dxy=st([r['dxy_mm'] for r in R]), dx=st([r['dx_mm'] for r in R]), dy=st([r['dy_mm'] for r in R]),
                      dz=st([r['dz_mm'] for r in R]), dyaw=st([r['dyaw_deg'] for r in R]), absdyaw=st([abs(r['dyaw_deg']) for r in R]),
                      score=st([r['score'] for r in R]), iou_fp_sam=st([r['iou_fp_sam'] for r in R]),
                      dz_raw=st([r.get('dz_mm') for r in R]), k=[int(r.get('quarter_turns', 0)) for r in R],
                      time_s=st([r['time_s'] for r in R]), fp_s=st([r['fp_s'] for r in R]),
                      stereo_s=st([r.get('stereo_s') for r in R]), gpu=st([r.get('gpu_peak_gb') for r in R]))
        if v == 'full':
            out[v]['sam_iou_truth'] = st([(r.get('sam') or {}).get('iou_truth') for r in R])
            out[v]['sam_iou_edgefit'] = st([(r.get('sam') or {}).get('iou_edgefit_render') for r in R])
            out[v]['sam_s'] = st([(r.get('sam') or {}).get('time_s') for r in R])
            out[v]['sam_prompt'] = {p: sum(1 for r in R if (r.get('sam') or {}).get('prompt') == p) for p in
                                    sorted(set((r.get('sam') or {}).get('prompt') for r in R), key=str)}
            out[v]['fails'] = [r['case'] for r in rows if 'dxy_mm' not in r[v]]
    E = [r['edge_fit'] for r in rows if 'dx_mm' in r.get('edge_fit', {})]
    out['edge_fit'] = dict(n=len(E), dxy=st([np.hypot(e['dx_mm'], e['dy_mm']) for e in E]), dx=st([e['dx_mm'] for e in E]),
                           dy=st([e['dy_mm'] for e in E]), dyaw=st([e['dyaw_deg'] for e in E]),
                           ok_quality=sum(1 for e in E if e['inlier_rms_px'] is not None and e['inlier_rms_px'] <= 1.2
                                          and all(e['inl'][k] >= 30 for k in ('panel', 'right', 'back'))))
    out['truthmask_vs_render_iou'] = st([r.get('iou_truthmask_vs_truth_render') for r in rows])
    return out


def real_table(path):
    J = json.loads(Path(path).read_text())
    rows = [r for r in J['rows'] if r['fp'].get('pose')]
    out = dict(n=len(J['rows']), n_ok=len(rows))
    out['fp_minus_edge'] = dict(dx=st([r['fp_minus_edge']['dx_mm'] for r in rows]), dy=st([r['fp_minus_edge']['dy_mm'] for r in rows]),
                                dyaw=st([r['fp_minus_edge']['dyaw_deg'] for r in rows]),
                                dxy=st([np.hypot(r['fp_minus_edge']['dx_mm'], r['fp_minus_edge']['dy_mm']) for r in rows]))
    f = lambda k: st([r['fp'].get(k) for r in rows])   # noqa: E731
    out.update(score=f('score'), iou_fp_sam=f('iou_fp_sam'), dz_raw=f('dz_mm'), depth_valid=f('depth_valid_frac'),
               time_s=f('time_s'), fp_s=f('fp_s'), stereo_s=f('stereo_s'), gpu=f('gpu_peak_gb'),
               tilt=st([r['fp']['pose'][4] for r in rows]), k=[r['fp']['quarter_turns'] for r in rows],
               sam_iou_edgefit=st([r['fp']['sam']['iou_edgefit_render'] for r in rows]),
               sam_score=st([r['fp']['sam']['score'] for r in rows]), sam_s=st([r['fp']['sam']['time_s'] for r in rows]),
               fp_vs_edge_render_iou=st([r['fp']['iou_fp_render_vs_edgefit_render'] for r in rows]),
               edge_rms=st([r['edge_fit']['inlier_rms_px'] for r in rows]))
    # repeatability: spot2 group (PC not moved; 10 frames), spot1 (8 frames)
    rep = {}
    for name, dirs in (('spot1', ('spot1',)), ('spot2_all', ('spot2', 'spot2_repeat', 'spot2_repeat2'))):
        g = [r for r in rows if r['dir'] in dirs]
        rep[name] = dict(n=len(g), fp_sd_mm_deg=[float(np.std([r['fp']['pose'][i] for r in g], ddof=1) * (1e3 if i < 2 else 1)) for i in range(3)],
                         edge_sd_mm_deg=[float(np.std([r['edge_fit']['pose'][i] for r in g], ddof=1) * (1e3 if i < 2 else 1)) for i in range(3)],
                         fp_range_mm_deg=[float(np.ptp([r['fp']['pose'][i] for r in g]) * (1e3 if i < 2 else 1)) for i in range(3)],
                         edge_range_mm_deg=[float(np.ptp([r['edge_fit']['pose'][i] for r in g]) * (1e3 if i < 2 else 1)) for i in range(3)])
    out['repeat'] = rep
    # spot1 -> spot2 move (operator: PC slid toward the arms): FP vs edge fit
    m = {}
    for k in ('fp', 'edge'):
        get = (lambda r: r['fp']['pose'][:3]) if k == 'fp' else (lambda r: r['edge_fit']['pose'][:3])
        a = np.median([get(r) for r in rows if r['dir'] == 'spot1'], 0)
        b = np.median([get(r) for r in rows if r['dir'].startswith('spot2')], 0)
        m[k] = [float((b[0] - a[0]) * 1e3), float((b[1] - a[1]) * 1e3), float(b[2] - a[2])]
    out['move_spot1_to_spot2'] = m
    npr = [r.get('fp_noprior') for r in J['rows'] if r.get('fp_noprior')]
    if npr:
        yaws = [((p['pose_raw'][2] - r['edge_fit']['pose'][2] + 180) % 360 - 180) for r, p in zip(J['rows'], npr) if p and p.get('pose_raw')]
        out['noprior_raw_yaw_minus_edge'] = dict(n=len(yaws), within_10deg=sum(abs(y) < 10 for y in yaws),
                                                 near_180=sum(abs(abs(y) - 180) < 10 for y in yaws),
                                                 near_90=sum(abs(abs(y) - 90) < 10 for y in yaws))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', default='results/perception/fp/sim_eval.json')
    ap.add_argument('--sim-raw', default='results/perception/fp/sim_eval_noplanefix.json')
    ap.add_argument('--real', default='results/perception/fp/real_zed_gt.json')
    ap.add_argument('--out', default='results/perception/fp/summary.json')
    a = ap.parse_args()
    S = {}
    for k, p in (('sim', a.sim), ('sim_raw', a.sim_raw)):
        if Path(p).exists():
            S[k] = sim_table(p)
    if Path(a.real).exists():
        S['real'] = real_table(a.real)
    Path(a.out).write_text(json.dumps(S, indent=1))
    for k in ('sim', 'sim_raw'):
        if k not in S:
            continue
        s = S[k]
        print(f'## {k} (n = {s["n"]})')
        for v in ('oracle', 'stereo', 'full'):
            x = s[v]
            print(f"{v}: ok {x['n_ok']}, |dxy| p50 {x['dxy']['p50']:.2f} p90 {x['dxy']['p90']:.2f} max {x['dxy']['max']:.2f} mm; dx {fmt(x['dx'])}; dy {fmt(x['dy'])}; "
                  f"dyaw {fmt(x['dyaw'])} deg; |dyaw| max {x['absdyaw']['max']:.2f}; dz raw {fmt(x['dz_raw'], d=1)}; k counts {np.bincount(x['k'], minlength=4).tolist()}; "
                  f"score {fmt(x['score'], d=1)}; IoU FP/SAM {fmt(x['iou_fp_sam'], d=3)}; t {x['time_s']['p50']:.2f} s (FP {x['fp_s']['p50']:.2f}); GPU {x['gpu']['max']:.2f} GB")
        x = s['full']
        print(f"SAM IoU vs truth {fmt(x['sam_iou_truth'], d=3)}; vs edge-fit render {fmt(x['sam_iou_edgefit'], d=3)}; SAM t p50 {x['sam_s']['p50']:.2f} s; prompts {x['sam_prompt']}; fails {x['fails']}")
        e = s['edge_fit']
        print(f"edge fit: n {e['n']}, quality ok {e['ok_quality']}, |dxy| p50 {e['dxy']['p50']:.2f} max {e['dxy']['max']:.2f}; dx {fmt(e['dx'])}; dy {fmt(e['dy'])}; dyaw {fmt(e['dyaw'])}")
        print(f"truth mask vs truth render IoU {fmt(s['truthmask_vs_render_iou'], d=3)}")
    if 'real' in S:
        r = S['real']
        print(f"## real zed_gt (n = {r['n']}, ok {r['n_ok']})")
        for k, v in r['fp_minus_edge'].items():
            print(f'FP - edge {k}: {fmt(v)}')
        for k in ('score', 'iou_fp_sam', 'dz_raw', 'tilt', 'sam_iou_edgefit', 'sam_score', 'fp_vs_edge_render_iou', 'depth_valid', 'time_s', 'fp_s', 'stereo_s', 'sam_s', 'gpu', 'edge_rms'):
            print(f'{k}: {fmt(r[k], d=3)}')
        print('k counts', np.bincount(r['k'], minlength=4).tolist())
        print('repeat', json.dumps(r['repeat']))
        print('move spot1->spot2', r['move_spot1_to_spot2'])
        print('no prior', r.get('noprior_raw_yaw_minus_edge'))


if __name__ == '__main__':
    main()
