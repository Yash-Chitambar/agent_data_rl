"""Does the insertion policy's history encoder already know the hidden offsets? Linear (ridge) probes from the TCN latent z, from the
current observation, and from the flattened raw history, to privileged quantities, on K4 rollouts in acceptance cells
(scripts/eval/k2_matrix.py setup: the policy's env_json, real plant, real right-arm lead). Fit on 70 % of the worlds, test on 30 %.

Targets (all known in the sim only):
  p_now      plug tip in the live port frame now (mm, 3)          -- what the policy must find
  prior_err  vision prior - true plug position at the reset (mm, 3) -- the error a history could remove
  enc_r      right encoder offset error (mrad, 6)
  lead       joint target - measured at the reset (mrad, 6)
  inhand     reader in TCP minus the reset value... (true in-hand error vs the policy's in-hand estimate, mm/deg, 6)
Baseline for p_now: the probe from the current observation only (o; it holds the prior and the hand-over-relative pose).
Per time bin since the first contact: test RMSE (mm) of p_now lateral (x, y) and of prior_err.

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/rma_probe.py --policy runs/k4s0_final.pt --out results/real_align/rma/probe_k4.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.eval.kj_real4  # noqa: E402,F401
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, cond_rows  # noqa: E402
from scripts.eval.k2_matrix import CELLS, PLANT  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402


def collect(env, pol, cell, base, seed, every=5):
    g = np.random.default_rng(seed)
    env._npz = torch.tensor(cond_rows(CELLS[cell], base, g), device=env.device, dtype=torch.float32)
    env.refresh_pool(); env.reset_all()
    q0 = env._measured_model_angles()[:, 0]
    env.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
    hist, _ = env.observe()
    N = env.N
    p0, _ = env.runtime.plug_in_port()
    prior0 = env.vision_prior.clone()
    prior_err = (prior0 - p0 * 1e3)
    enc_r = env.enc_error[:, 0] * 1e3
    lead = (env.joint_target[:, 0] - env._measured_model_angles()[:, 0]) * 1e3
    inh_true = env.inhand_error(env.ref_rel)
    inh_err = env.inhand_est - inh_true
    ac = pol.members[0][0]
    live = torch.ones(N, dtype=torch.bool, device=env.device)
    t_contact = torch.full((N,), -1, dtype=torch.long, device=env.device)
    ok = torch.zeros(N, dtype=torch.bool, device=env.device)
    rows = []
    for t in range(env.max_steps):
        lab = env.current_labels()
        touch = (lab > 0).any(1)
        t_contact = torch.where((t_contact < 0) & touch & live, torch.full_like(t_contact, t), t_contact)
        if t % every == 0:
            h = hist[..., :pol.obs_dim].clone()
            if pol.mask_idx:
                h[..., list(pol.mask_idx)] = 0
            with torch.no_grad():
                z, o = ac.encode(h)
            p, _ = env.runtime.plug_in_port()
            # baseline: prior + believed FK TCP displacement since the hand-over, in port axes (rel_pose_obs pos = p_tcp - ho_p)
            qb = env._measured_model_angles()[:, 0]
            pt, _ = env.right_chain.fk(qb)
            _, port_R = env._port_pose(env.pc_start)
            d = (port_R.transpose(1, 2) @ (pt - env.ho_p)[..., None])[..., 0] * 1e3
            ap_ = AUX[0](z)[:, :3].detach().cpu().numpy() if AUX else None
            rows.append(dict(t=t, auxp=ap_, live=live.cpu().numpy().copy(), z=z.cpu().numpy(), o=o.cpu().numpy(),
                             hraw=ac.obs_norm(h)[:, ::6].reshape(N, -1).cpu().numpy(),
                             p=(p * 1e3).cpu().numpy(), base=(prior0 + d).cpu().numpy(), tc=t_contact.cpu().numpy().copy()))
        act = pol(hist)
        hist, _, _, done, info = env.step(act)
        if 'final' in info:
            ids = info['final_ids']
            ok[ids] = torch.where(live[ids], info['final']['success'].to(ok.device).bool(), ok[ids])
        live &= ~done
        if not live.any():
            break
    fixed = dict(prior_err=prior_err.cpu().numpy(), enc_r=enc_r.cpu().numpy(), lead=lead.cpu().numpy(), inh_err=inh_err.cpu().numpy(), ok=ok.cpu().numpy())
    return rows, fixed


def ridge(Xtr, Ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    A1, B1 = np.c_[A, np.ones(len(A))], np.c_[B, np.ones(len(B))]
    W = np.linalg.solve(A1.T @ A1 + lam * np.eye(A1.shape[1]), A1.T @ Ytr)
    return B1 @ W


def r2(y, yh):
    return (1 - ((y - yh) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0).clip(1e-9)).round(3).tolist()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--cells', default='real_ho,ring4,mix')
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=5300)
    ap.add_argument('--env_extra', default='{}')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    torch.manual_seed(A.seed)
    pol = InsertionPolicy(A.policy, 'cuda:0')
    ck_ = torch.load(A.policy, map_location='cuda:0', weights_only=False)
    AUX = []
    if 'aux' in ck_:     # train_bimanual --aux_coef: the policy's own estimate of the plug in the port (mm, first 3 outputs)
        from blind_insertion.policies.tcn_actor_critic import mlp
        h_ = mlp(pol.members[0][0].latent, (128,), ck_['aux']['2.weight'].shape[0]).to('cuda:0')
        h_.load_state_dict(ck_['aux']); h_.eval()
        AUX.append(h_)
    ej = dict(pol.meta['env_json'], hard_frac=0., scale_range_frac=0., live_frac=0., extra_ang_frac=0., handover_lead_frac=0.,
              vision_prior_sigma_scale=None)     # training-only DR off (K5: wider prior error in training)
    ej.update(PLANT['real']); ej.update(json.loads(A.env_extra))
    env = make_env(argparse.Namespace(nworld=A.nworld, seed=A.seed, action_mode='accumulate', env_json=json.dumps(ej)))
    base = rows_base()
    out = {}
    for ci, cell in enumerate(A.cells.split(',')):
        rows, fx = collect(env, pol, cell, base, A.seed + ci)
        N = A.nworld
        rng = np.random.default_rng(0)
        tr = rng.random(N) < .7
        te = ~tr
        res = dict(n=N, seated=float(fx['ok'].mean()))
        # fixed targets from the latent at chosen times (since the start, s)
        for tsec in (0.0, 0.2, 0.5, 1.0, 1.5):
            r = min(rows, key=lambda x: abs(x['t'] * env.dt - tsec))
            m = r['live']
            for src in ('z', 'o', 'zo', 'hraw'):
                X = np.c_[r['z'], r['o']] if src == 'zo' else r[src]
                for tgt in ('prior_err', 'enc_r', 'lead', 'inh_err'):
                    Y = fx[tgt]
                    a, b = tr & m, te & m
                    if a.sum() < 50 or b.sum() < 20:
                        continue
                    yh = ridge(X[a], Y[a], X[b], lam=10.)
                    res[f'{tgt}|{src}|t{tsec}'] = dict(r2=r2(Y[b], yh), rmse=np.sqrt(((Y[b] - yh) ** 2).mean(0)).round(3).tolist(), sd=Y[b].std(0).round(3).tolist(), n_live=int(m.sum()))
        # p_now lateral, pooled over time, binned by time since the first contact
        bins = [(-1e9, 0), (0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 1e9)]
        Xz, Xo, Y, Bs, T, W = [], [], [], [], [], []
        for r in rows:
            m = r['live']
            Xz.append(np.c_[r['z'], r['o']][m]); Xo.append(r['o'][m]); Y.append(r['p'][m]); Bs.append(r['base'][m])
            tc = r['tc'][m]
            T.append(np.where(tc < 0, -1.0, (r['t'] - tc) * env.dt)); W.append(np.nonzero(m)[0])
        Xz, Xo, Y, Bs, T, W = map(np.concatenate, (Xz, Xo, Y, Bs, T, W))
        a, b = tr[W], te[W]
        yz = ridge(Xz[a], Y[a], Xz[b], lam=10.)
        yo = ridge(Xo[a], Y[a], Xo[b], lam=10.)
        Xzz = np.concatenate([r['z'][r['live']] for r in rows])
        yzz = ridge(Xzz[a], Y[a], Xzz[b], lam=10.)
        tb = T[b]
        pb = {}
        for lo, hi in bins:
            k = (tb >= lo) & (tb < hi) if lo > -1e8 else (tb < 0)
            if k.sum() < 20:
                continue
            e = lambda yh: np.sqrt(((Y[b][k] - yh[k]) ** 2).mean(0)).round(3).tolist()  # noqa: E731
            pb[f'{lo:g}..{hi:g}' if lo > -1e8 else 'before_contact'] = dict(n=int(k.sum()), sd=Y[b][k].std(0).round(3).tolist(), rmse_xyz_z=e(yzz), rmse_xyz_o=e(yo), rmse_xyz_zo=e(yz))
        if AUX:
            Pa = np.concatenate([r['auxp'][r['live']] for r in rows])
            for kk, (lo, hi) in zip(list(pb), [x for x in bins if ((tb >= x[0]) & (tb < x[1]) if x[0] > -1e8 else (tb < 0)).sum() >= 20]):
                k = (tb >= lo) & (tb < hi) if lo > -1e8 else (tb < 0)
                pb[kk]['rmse_xyz_aux_head'] = np.sqrt(((Y[b][k] - Pa[b][k]) ** 2).mean(0)).round(3).tolist()
        res['p_now_lat_by_time_since_contact'] = pb
        out[cell] = res
        print(cell, json.dumps({k: v for k, v in res.items() if k.startswith('prior_err|z') or k.startswith('p_now') or k == 'seated'}), flush=True)
    Path(A.out).parent.mkdir(parents=True, exist_ok=True)
    Path(A.out).write_text(json.dumps(out, indent=1))
