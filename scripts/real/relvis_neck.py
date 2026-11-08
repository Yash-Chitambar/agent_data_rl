"""Is the reader's black neck (measured 12.96 mm wide, data/real_yam/usbc_lab_20261027/measurements.csv) usable as a second depth scale?
Per align frame: column profile of the band just below the shell's sleeve end; edge steps at the expected half-width
(12.96 / 8.15 x shell half-width in px) against the profile noise. Prints per run: n frames, edge SNR, width px, implied depth vs shell depth.
  PYTHONPATH=. python scripts/real/relvis_neck.py"""
import json, glob, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2, numpy as np
from deploy.bimanual import port_vision as PV

NECK_W_MM = 12.96
out = {}
for d in sorted(glob.glob('data/real_runs/2026-11-0[67]/full_*')):
    rows = [json.loads(l) for l in open(f'{d}/frames.jsonl')]
    rows = [r for r in rows if r['cam'] == 'wrist_right' and r['kind'] == 'raw' and r['tag'].startswith('align_') and r['tag'][6:9].isdigit()]
    res = []
    for r in rows[::3]:
        und, K = PV.undistort(cv2.imread(f"{d}/{r['file']}"))
        pl, _ = PV.detect_plug(und)
        if pl is None:
            continue
        uc, ve, w = 0.5 * (pl['ul'] + pl['ur']), pl['ve'], pl['w']
        half = 0.5 * w * NECK_W_MM / PV.SHELL_W_EFF_MM
        g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY).astype(np.float32)
        band = g[int(ve) + 4:int(ve) + 30, int(uc - 2.2 * half):int(uc + 2.2 * half)]
        prof = cv2.GaussianBlur(band.mean(0)[None], (1, 5), 0)[0]
        gr = np.gradient(prof)
        c = len(prof) // 2
        L = np.argmax(np.abs(gr[int(c - 1.5 * half):int(c - 0.5 * half)])) + int(c - 1.5 * half)
        R = np.argmax(np.abs(gr[int(c + 0.5 * half):int(c + 1.5 * half)])) + int(c + 0.5 * half)
        noise = np.std(np.diff(prof)) + 1e-6
        snr = min(abs(gr[L]), abs(gr[R])) / noise
        f = 0.5 * (K[0, 0] + K[1, 1])
        res.append((snr, R - L, f * NECK_W_MM / max(R - L, 1), f * PV.SHELL_W_EFF_MM / w, abs(gr[L]), abs(gr[R])))
    if res:
        a = np.array(res)
        out[Path(d).name] = dict(n=len(a), snr_p50=float(np.median(a[:, 0])), neck_px_p50=float(np.median(a[:, 1])), neck_px_sd=float(a[:, 1].std()),
                                 z_neck_p50=float(np.median(a[:, 2])), z_neck_sd=float(a[:, 2].std()), z_shell_p50=float(np.median(a[:, 3])), z_shell_sd=float(a[:, 3].std()),
                                 step_LR_p50=np.median(a[:, 4:6], 0).round(1).tolist())
        print(Path(d).name, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out[Path(d).name].items()})
Path('results/real_align/relvis/neck.json').write_text(json.dumps(out, indent=1))
