"""K3 robustness in the sim (real-like plant, real hand-over lead): how much it trusts the camera prior, whether encoder
offset errors reach it, and how right its contact head is. n per cell = --nworld (first episode of each world).

Conditions (on top of the cell's start):
  prior_ok          the prior as trained (truth + N(0, (0.4, 0.95, 0.63) mm))
  prior_up+1/2/3    prior biased by +1/+2/+3 mm in up (the vision up error seen on 2026-11-03 was 2-3 mm)
  prior_across+2    +2 mm across
  prior_nominal     prior = (0, 0, 6) mm for every world (uninformative: the align stage said "on target")
  enc3              encoder offset DR +-3 deg (training: +-1.5); prior as trained
Contact head (--contact): per-point accuracy against the sim's labels over the first 150 steps.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python scripts/eval/k3_robust.py --policy runs/k3_forkB_a.pt --out results/real_align/k3/robust_a.json
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.eval.kj_real4  # noqa: E402,F401
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, run_cond  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402

CELLS = {'real_ho': dict(gap=5.5, ring=.5), 'ring3': dict(gap=5.5, ring=3.)}
CONDS = {'prior_ok': None, 'prior_up+1': (0, 1.), 'prior_up+2': (0, 2.), 'prior_up+3': (0, 3.), 'prior_across+2': (2., 0), 'prior_nominal': 'nominal'}


def build(policy, nworld, seed, extra):
    pol = InsertionPolicy(policy, 'cuda:0')
    ej = dict(pol.meta['env_json'], hard_frac=0., scale_range_frac=0., live_frac=0., extra_ang_frac=0., handover_lead_frac=0.,
              friction_scale_range=[1.3, 2.0], gravity_scale_jitter=0.12)
    ej.update(extra)
    env = make_env(argparse.Namespace(nworld=nworld, seed=seed, action_mode='accumulate', env_json=json.dumps(ej)))
    return env, pol


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--nworld', type=int, default=512)
    ap.add_argument('--seed', type=int, default=6100)
    ap.add_argument('--only', default='')
    ap.add_argument('--contact', action='store_true')
    ap.add_argument('--enc', action='store_true', help='run the enc3 condition (own env)')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {(d['cell'], d['cond']) for d in done}
    torch.manual_seed(A.seed)

    def save(r):
        done.append(r); out.write_text(json.dumps(done, indent=1))
        print(r['cell'], r['cond'], f"{r['k']}/{r['n']} = {100 * r['rate']:.1f} % [{100 * r['wilson'][0]:.1f}, {100 * r['wilson'][1]:.1f}]",
              {k: v for k, v in r.get('classes', {}).items() if k != 'seated'}, r.get('contact', ''), flush=True)

    jobs = [(c, k) for c in CELLS for k in CONDS] if not A.enc else [(c, 'enc3') for c in CELLS]
    if A.only:
        jobs = [j for j in jobs if j[1] in A.only.split(',')]
    jobs = [j for j in jobs if j not in seen]
    if jobs or A.contact:
        env, pol = build(A.policy, A.nworld, A.seed, {'enc_offset_deg': 3.0} if A.enc else {})
        assert env.icfg.vision_prior_obs, 'not a K3 policy'
        base = rows_base()
    for i, (cell, cond) in enumerate(jobs):
        mod = CONDS.get(cond)

        def hook(kind, e, act=None, mod=mod):
            if kind == 'reset':
                q0 = e._measured_model_angles()[:, 0]
                e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
                if mod == 'nominal':
                    e.vision_prior[:] = torch.tensor([0., 0., 5.5], device=e.device)
                elif mod is not None:
                    e.vision_prior[:, 0] += mod[0]; e.vision_prior[:, 1] += mod[1]
                e.hist[:] = e._actor_obs()[:, None]
                return None
            return act
        t0 = time.time()
        r = run_cond(env, pol, CELLS[cell], base, A.seed + i, on_step=hook)
        r.update(cell=cell, cond=cond, policy=A.policy, wall_s=round(time.time() - t0))
        save(r)
    if A.contact and ('real_ho', 'contact') not in seen:
        ac = pol.members[0][0]
        rec = dict(true=[], pred=[], t=0)

        def hook2(kind, e, act=None):
            if kind == 'reset':
                q0 = e._measured_model_angles()[:, 0]
                e.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
                e.hist[:] = e._actor_obs()[:, None]
                return None
            if rec['t'] < 150:
                h = e.hist[..., :pol.obs_dim].clone()
                if pol.mask_idx:
                    h[..., list(pol.mask_idx)] = 0
                with torch.no_grad():
                    z, _ = ac.encode(h)
                    rec['pred'].append(ac.contact_logits(z).argmax(-1).cpu().numpy().astype(np.int8))
                rec['true'].append(e.current_labels().cpu().numpy().astype(np.int8))
            rec['t'] += 1
            return act
        r = run_cond(env, pol, CELLS['real_ho'], base, A.seed + 99, on_step=hook2)
        tt, pp = np.stack(rec['true']).reshape(-1), np.stack(rec['pred']).reshape(-1)
        conf = np.zeros((3, 3), int); np.add.at(conf, (tt, pp), 1)
        r.update(cell='real_ho', cond='contact', contact=dict(acc=round(float((tt == pp).mean()), 4), recall=(np.diag(conf) / np.maximum(conf.sum(1), 1)).round(3).tolist(),
                                                               confusion_true_rows_pred_cols=conf.tolist()))
        save(r)
