"""Real hand-over frames: the direct plug-tip-to-port reading (port_vision.measure) against 'port pose from vision + plug tip from the
camera mount and the in-hand estimate' (the port-only path). Same panel pose (R, t) for both, so the difference is only the tip:
image tip vs mount x in-hand model. Per run: last align frame before the insertion; in-hand = the run's align_inhand 'after' (the
net the policy got) and the nominal grasp (zero delta). Also the direct reading's spread over the hand-over confirm frames.
  PYTHONPATH=. python scripts/real/relvis_compare.py"""
import json, glob, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.append(str(ROOT / 'scripts/real2sim'))
import cv2, numpy as np
from deploy.bimanual import port_vision as PV
import inhand_measure as IM

hand = IM.Hand()
pc = np.asarray(PV.load_model().get('port_center_in_model_mm', PV.PORT_CENTER_MM), float)
out = {}
for d in sorted(glob.glob('data/real_runs/2026-11-0[67]/full_*')):
    fr = [json.loads(l) for l in open(f'{d}/frames.jsonl')]
    fr = [r for r in fr if r['cam'] == 'wrist_right' and r['kind'] == 'raw' and r['tag'].startswith('align_') and r['tag'][6:9].isdigit()]
    ev = [json.loads(l) for l in open(f'{d}/events.jsonl')]
    ih = [e for e in ev if e['kind'] == 'align_inhand']
    if not fr:
        continue
    # direct readings over the last 6 frames (the hand-over confirm window, hand at rest)
    Xs, last = [], None
    for r in fr[-6:]:
        m = PV.measure(cv2.imread(f"{d}/{r['file']}"))
        if m.ok:
            Xs.append([m.across_mm, m.up_mm, m.gap_mm]); last = m
    if last is None:
        continue
    R, t = np.asarray(last.debug['R']), np.asarray(last.debug['t'])
    X = np.array(Xs[-1])
    rec = dict(n_rest=len(Xs), rest_sd_mm=np.std(Xs, 0).round(3).tolist(), direct_mm=X.round(2).tolist())
    for lab, dl in (('nominal', np.zeros(3)), ('inhand_net', np.asarray(ih[-1]['after'][:3]) if ih else None)):
        if dl is None:
            continue
        p_site = hand.p_nom * 1e3 + dl                       # reader tip in the grasp site (mm); in-hand delta on the TCP = site axes
        tip_cam = hand.R_cs.T @ (p_site - hand.p_cs * 1e3)   # camera frame (mm)
        Xa = R.T @ (tip_cam - t) - pc
        rec[f'alt_{lab}_mm'] = Xa.round(2).tolist()
        rec[f'direct_minus_alt_{lab}_mm'] = (X - Xa).round(2).tolist()
    out[Path(d).name] = rec
    print(Path(d).name, rec)
for lab in ('nominal', 'inhand_net'):
    D = np.array([v[f'direct_minus_alt_{lab}_mm'] for v in out.values() if f'direct_minus_alt_{lab}_mm' in v])
    print(lab, 'n', len(D), 'mean', D.mean(0).round(2), 'sd over runs', D.std(0, ddof=1).round(2))
    out[f'summary_{lab}'] = dict(n=len(D), mean=D.mean(0).round(3).tolist(), sd=D.std(0, ddof=1).round(3).tolist())
Path('results/real_align/relvis/compare_inhand.json').write_text(json.dumps(out, indent=1))
