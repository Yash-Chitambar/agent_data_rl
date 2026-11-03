"""Offline: what would the align stage have seen and commanded at the real stops? No motion. Real logged frames through the measurement sources and AlignLogic.

    python scripts/real/align_replay.py --config deploy/config/bimanual_align1.json --json results/real_align/replay.json

For each run: raw wrist frames from the end of the approach (settled, last before the insertion), port_vision and the fine wrist net on each; FK-label reference from the collect anchor
(same PC spot; add the open up offset, up to 4.5 mm: see docs/real/SESSION_20261102_duchamp3.md). The first decision of AlignLogic on the median. Then the scale gate: port_vision and the net against FK labels on the collect sweep
(the frames move with known FK differences), the same regression the stage does online.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402
from deploy.bimanual import port_vision as PV  # noqa: E402
from deploy.bimanual.align import DEFAULTS, AlignLogic, PortMeasure  # noqa: E402

D = ROOT / 'data/real_runs'
RUNS = ['2026-11-02/full_174850_full_A', '2026-11-02/full_175742_full_A2', '2026-11-01/approach_insert_171123_ai4', '2026-11-01/approach_insert_173731_ai5', '2026-11-01/approach_insert_164528_ai2', '2026-11-01/approach_insert_165634_ai3']
COLLECT = '2026-11-02/collect_182100_collect1'


def net_loader(name):
    from deploy.bimanual import netcache as NC
    return NC.wrist(name, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


def net_eval(w, png):
    dev = next(w[0].parameters()).device
    im = cv2.cvtColor(cv2.imread(str(png)), cv2.COLOR_BGR2RGB)
    fr = torch.as_tensor(im, device=dev)[None]
    if w[2] == 'crop':
        img = fr[:, 40:232, 64:256].contiguous()
    else:
        img = torch.nn.functional.interpolate(fr.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
    with torch.no_grad():
        return (w[0](img) * w[1].to(dev))[0, :3].cpu().numpy().astype(float)


def stop_frames(run):
    ev = L.events(run)
    k1 = next(e['tick'] for e in ev if e['kind'] == 'stage' and e['stage'] == '5 insert')
    return stop_frames_upto(run, k1)


def stop_frames_upto(run, k1, pre=300):
    r = [json.loads(x) for x in open(run / 'frames.jsonl')]
    out = []
    tags = {}
    for x in r:
        if x['cam'] == 'wrist_right':
            t = x.get('tick')
            tags.setdefault(x['tag'] + str(t), {})[x['kind']] = x
    for key, d in tags.items():
        x = d.get('raw')
        if x is None or 'policy' not in d:
            continue
        t = x.get('tick')
        if t is not None and k1 - pre <= t <= k1:
            out.append((t, run / x['file'], run / d['policy']['file']))
    return sorted(out, key=lambda z: z[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--json', default=None)
    A = ap.parse_args()
    c = dict(DEFAULTS)
    if A.config:
        c.update(json.loads(Path(A.config).read_text()).get('align', {}))
    an = L.anchor(str(D / COLLECT))
    nets = {n: net_loader(n) for n in ('wrist_v18ws', 'wrist_v14w')}
    res = []
    for name in RUNS:
        run = D / name
        fr = stop_frames(run)
        rows = []
        for t, raw, pol in fr:
            im = cv2.imread(str(raw))
            m = PV.measure(im, side='right')
            rows.append(dict(tick=t, ok=m.ok, reason=m.reason, conf=m.conf, v=[m.across_mm, m.up_mm, m.gap_mm], sigma=m.sigma_mm, net={n: net_eval(w, pol).round(2).tolist() for n, w in nets.items()}))
        gd = [r for r in rows if r['ok']]
        out = dict(run=name, frames=len(rows), port_vision_ok=len(gd), reasons=sorted({r['reason'] for r in rows if not r['ok']}))
        if gd:
            out['port_vision_median'] = np.median(np.array([r['v'] for r in gd], float), 0).round(2).tolist()
            out['port_vision_spread'] = (np.array([r['v'] for r in gd], float).max(0) - np.array([r['v'] for r in gd], float).min(0)).round(2).tolist()
        if rows:
            for n in nets:
                out[n] = np.median(np.array([r['net'][n] for r in rows]), 0).round(2).tolist()
        try:
            rr = L.load(str(run))
            tr = L.trace(rr)
            ep = an['rel'] + an['eR'].T @ (tr['tip0'] - an['tip']) * 1e3
            out['fk_label_collect_anchor'] = ep.round(2).tolist()
        except Exception as e:  # noqa: BLE001
            out['fk_label_collect_anchor'] = repr(e)
        if gd:
            v = np.median(np.array([r['v'] for r in gd], float), 0)
            lg = AlignLogic(dict(c, probe_mm=[0, 0, 0]))
            dec = lg.update(PortMeasure(True, 'replay', 1., *v, 0.5), None, np.zeros(3))
            out['first_decision_port_vision'] = dict(kind=dec['kind'], delta=np.asarray(dec.get('delta')).round(2).tolist(), reason=dec.get('reason'), err=np.asarray(dec.get('err', [0, 0, 0])).round(2).tolist())
        if rows:
            v = np.array(out['wrist_v18ws'])
            lg = AlignLogic(dict(c, probe_mm=[0, 0, 0]))
            dec = lg.update(PortMeasure(True, 'replay', 1., *v, 0.5), None, np.zeros(3))
            out['first_decision_wrist_v18ws'] = dict(kind=dec['kind'], delta=np.asarray(dec.get('delta')).round(2).tolist(), reason=dec.get('reason'))
        res.append(out)
        print(json.dumps(out), flush=True)
    # scale gate on the collect sweep
    run = D / COLLECT
    man = [json.loads(x) for x in open(ROOT / 'data/real_training/port_v1/manifest.jsonl')]
    man = [m for m in man if m['source'] == 'collect' and m['p_mm'][2] < 15]
    P, V, N = [], [], {n: [] for n in nets}
    for m in man:
        p = np.array(m['p_mm'], float)
        raw = m.get('raw')
        if raw is None:
            continue
        rawp, polp = Path(raw) if Path(raw).is_absolute() else ROOT / raw, Path(m['image']) if Path(m['image']).is_absolute() else ROOT / m['image']
        pm = PV.measure(cv2.imread(str(rawp)), side='right')
        P.append(p)
        V.append([pm.across_mm, pm.up_mm, pm.gap_mm] if pm.ok else [np.nan] * 3)
        for n, w in nets.items():
            N[n].append(net_eval(w, polp))
    P, V = np.array(P), np.array(V, float)
    ok = np.isfinite(V).all(1)
    def slope(x, y):
        x, y = x - x.mean(0), y - y.mean(0)
        return ((x * y).sum(0) / (x * x).sum(0)).round(3).tolist(), [float(np.corrcoef(x[:, i], y[:, i])[0, 1].round(3)) for i in range(3)], (y - x * (x * y).sum(0) / (x * x).sum(0)).std(0).round(3).tolist()
    sc = dict(n=len(P), port_vision_ok=int(ok.sum()))
    if ok.sum() > 5:
        s, cc, rs = slope(P[ok], V[ok])
        sc.update(port_vision_slope=s, port_vision_corr=cc, port_vision_resid_mm=rs, port_vision_mean_err_mm=(V[ok] - P[ok]).mean(0).round(2).tolist())
    for n in nets:
        s, cc, rs = slope(P, np.array(N[n]))
        sc[n] = dict(slope=s, corr=cc, resid_mm=rs)
    print(json.dumps(sc))
    if A.json:
        Path(A.json).parent.mkdir(parents=True, exist_ok=True)
        Path(A.json).write_text(json.dumps(dict(stops=res, scale=sc), indent=1))


if __name__ == '__main__':
    main()
