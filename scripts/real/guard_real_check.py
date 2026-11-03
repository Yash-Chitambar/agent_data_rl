"""InsertGuard on the real insertion logs, open loop (the logged run does not react): when would the bound clamp, when would it stop, and why. No motion.

    python scripts/real/guard_real_check.py --json results/real_align/guard_real.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402
from deploy.bimanual.insert_guard import InsertGuard  # noqa: E402

D = ROOT / 'data/real_runs'
RUNS = {'2026-11-02/full_174850_full_A': 'failed, plug on the case', '2026-11-02/full_175742_full_A2': 'failed, climbed over the port', '2026-11-01/approach_insert_171123_ai4': 'seated at about 4 s', '2026-11-01/approach_insert_173731_ai5': 'failed, see the notes', '2026-11-01/approach_insert_165634_ai3': 'seated, stopped early by the old detector'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align1.json'))
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    gc = {k: v for k, v in json.loads(Path(a.config).read_text())['insert']['guard'].items() if not k.startswith('_')}
    out = []
    for name, truth in RUNS.items():
        r = L.load(str(D / name))
        tr = L.trace(r)
        s = (L.Rs.reshape(3, 3).float().T @ L.ps.reshape(3).float())[None]
        g = InsertGuard(L.FK, s, torch.tensor(r['eR'], dtype=torch.float32), 1, torch.device('cpu'), dt=.02, **gc)
        q = torch.as_tensor(tr['q_m'], dtype=torch.float32)
        jt = torch.as_tensor(tr['jt'], dtype=torch.float32)
        g.reset(torch.tensor([0]), q[:1])
        t_clamp, t_stop, kind = None, None, 0
        for i in range(len(q)):
            _, info = g.limit(q[i:i + 1], jt[i:i + 1])
            if t_clamp is None and bool(info['cut'].any()):
                t_clamp = i * .02
            if t_stop is None and int(g.stop[0]):
                t_stop, kind = i * .02, int(g.stop[0])
        rel = tr['rel_m']
        out.append(dict(run=name, truth=truth, first_clamp_s=t_clamp, stop_s=t_stop, stop={0: None, 1: 'bound', 2: 'press'}[kind], up_at_stop=None if t_stop is None else float(rel[int(t_stop / .02), 1]), across_at_stop=None if t_stop is None else float(rel[int(t_stop / .02), 0]), up_max=float(np.abs(rel[:, 1]).max()), across_max=float(np.abs(rel[:, 0]).max()), n=len(q)))
        print(json.dumps(out[-1]))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
