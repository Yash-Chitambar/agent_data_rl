"""K-J's contact head against the sim's true contact labels (free / touching / inside per edge point), from the
real-like hand-over (gap 5.5 mm, ring 0.5 mm, real target lead). Per 0.1 s bin: per-point accuracy, and the share each
class gets in truth and prediction; saves a few worlds' traces for plots.

  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/eval/kj_contact_sim.py --nworld 256 --out results/real_align/kj_contact_sim.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402  (also patches angle conditions)
from scripts.eval.kj_start_map import build, rows_base, run_cond  # noqa: E402

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--seed', type=int, default=3300)
    ap.add_argument('--cond', default='{"gap": 5.5, "ring": 0.5}')
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    torch.manual_seed(A.seed)
    env, pol = build(A.nworld, A.seed, extra=json.loads(A.env_extra))
    ac = pol.members[0][0]
    T = 150
    rec = dict(true=[], pred=[], t=0)

    def hook(kind, e, act=None):
        if kind == 'reset':
            q0 = e._measured_model_angles()[:, 0]
            e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
            return None
        if rec['t'] < T:
            h = e.hist[..., :pol.obs_dim].clone()
            if pol.mask_idx:
                h[..., list(pol.mask_idx)] = 0
            with torch.no_grad():
                z, _ = ac.encode(h)
                rec['pred'].append(ac.contact_logits(z).argmax(-1).cpu().numpy().astype(np.int8))
            rec['true'].append(e.current_labels().cpu().numpy().astype(np.int8))
        rec['t'] += 1
        return act
    r = run_cond(env, pol, json.loads(A.cond), rows_base(), A.seed, on_step=hook)
    Tt, Pp = np.stack(rec['true']), np.stack(rec['pred'])          # (T, N, 64); worlds reset when done: count the first episode only
    out = dict(cond=json.loads(A.cond), seat=dict(k=r['k'], n=r['n'], rate=r['rate'], wilson=r['wilson']), bins=[])
    for b in range(0, len(Tt), 5):
        tt, pp = Tt[b:b + 5].reshape(-1), Pp[b:b + 5].reshape(-1)
        conf = np.zeros((3, 3), int)
        np.add.at(conf, (tt, pp), 1)
        out['bins'].append(dict(t=round(b * .02, 2), acc=float((tt == pp).mean()), true_share=(np.bincount(tt, minlength=3) / len(tt)).round(3).tolist(),
                                pred_share=(np.bincount(pp, minlength=3) / len(pp)).round(3).tolist(), confusion=conf.tolist()))
    tt, pp = Tt.reshape(-1), Pp.reshape(-1)
    conf = np.zeros((3, 3), int); np.add.at(conf, (tt, pp), 1)
    out['overall'] = dict(acc=float((tt == pp).mean()), confusion_true_rows_pred_cols=conf.tolist(),
                          recall=(np.diag(conf) / np.maximum(conf.sum(1), 1)).round(3).tolist(), precision=(np.diag(conf) / np.maximum(conf.sum(0), 1)).round(3).tolist())
    Path(A.out).write_text(json.dumps(out, indent=1))
    np.savez_compressed(Path(A.out).with_suffix('.npz'), true=Tt[:, :8], pred=Pp[:, :8])
    print(json.dumps(out['overall']), json.dumps(out['seat']))
