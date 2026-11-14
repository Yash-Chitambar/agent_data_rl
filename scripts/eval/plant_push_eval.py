"""K4 / K7 insertion + deploy-like seat detector + force push + slow unload, in the fitted push plant (docs/real/FORCE_BUDGET_20261115.md
Part 2). Sim only.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/plant_push_eval.py --policy runs/insert_K4.pt --tag k4 \
        --conds rigid_m0,comp_m0,comp_m10,comp_m20 --out results/real_align/force_budget/plant_eval.json

Cell 'real_ho' (gap 5.5 mm, ring 0.5 mm), real-like plant, real right-arm lead, rig-spot layout (v2), as scripts/eval/k7_depth.py.
Conditions: 'rigid' = the present sim arm; 'comp' = BimanualCfg.right_series_stiffness (joint 1, k = 640 N m/rad = 0.22 mm/N along the
port axis, the Part 2 fit); m<N> = InsertionCfg.mate_force_n [N, N] (spring model: zero up to 2 mm of depth, linear to N at the seat).
Per world (first episode; the sim success end is switched off):
  * seat: the sim seat test held (true depth within 0.25 mm of 5.8 mm for hold_steps) at any time = the usual sim seat rate;
  * det: the deploy seat2 detector on the ENCODER FK depth (true depth at the reset + FK advance along the port axis), depth >=
    seat - 1 mm and |speed| < 6 mm/s for 0.3 s; the true depth when it fires;
  * then (deploy insert.final_push force mode, bimanual_align10_all): 0.2 s measured-pose hold, axial load ramp 3 N/s (4 N/s x speed
    0.75) to F_push, 1.5 s hold, lateral spring 3 N/mm (cap 8 N) to the start line, joint torques scaled under tau_max; then the
    unload ramp to 0 N at 3 N/s with the jaws closed, 0.2 s measured-pose hold. Rotation spring left out.
  * full depth under load: true depth at the end of the push hold >= 5.3 mm (0.5 mm short of the seat); at 0 N (jaws closed) only for
    mate 0 (the spring model pushes a short plug back out, the real port holds it).
  * FK advance of the hand during the push (to compare with the logged 2.6 mm at 10 N and 4.1 mm at 14 N).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval.k3_robust import build  # noqa: E402
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.kj_start_map import rows_base, run_cond  # noqa: E402

RIG = {'task_layout': 'results/placement/task_layout_real_v2.json', 'pc_pose_range': [0.005, 0.005, 1.5], 'vision_prior_sigma_scale': None,
       'slip_m': 0.02, 'slip_deg': 20.0}     # only a gross slip ends the episode: the push and unload run to the end
K1 = 640.0
COND = {'rigid': {}, 'comp': {'right_series_stiffness': [K1, None, None, None, None, None]}}
TAU_MAX = [6.0, 6.0, 6.0, 2.5, 2.0, 2.0]


def pct(x, q):
    x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else None


def one(policy, cond, nworld, seed, f_push, episode_s):
    arm, mate = cond.split('_m')
    mate = float(mate)
    extra = dict(RIG, mate_force_n=None if mate == 0 else [mate, mate], **COND[arm])
    env, pol = build(policy, nworld, seed, extra)
    hold0 = env.hold_steps
    env.hold_steps = 10 ** 9
    env.max_steps = int(round(episode_s / env.dt))
    N, dev, dt = env.N, env.device, env.dt
    seat_mm = env.seat_depth * 1e3
    rt = env.runtime
    act_r = list(rt.indices['right'][2])
    kp = torch.tensor(rt.mjm.actuator_gainprm[act_r, 0], device=dev, dtype=torch.float32)   # nominal kp (the CPU model is unchanged)
    tmax = torch.tensor(TAU_MAX, device=dev)
    n_det, n_hold, n_full = int(round(0.3 / dt)), int(round(0.2 / dt)), int(round(1.5 / dt))
    ramp = 3.0 * dt
    S = {}
    orig = env._update_joint_targets

    def patched(action, measured):                       # the push target bypasses the policy's tracking cap (deploy sends it directly)
        p = orig(action, measured)
        m = S.get('override')
        if m is not None and m.any():
            p[m, 0] = S['target'][m]
            env.joint_target.copy_(p)
        return p
    env._update_joint_targets = patched

    def port():
        fa = rt.pc_qadr
        from blind_insertion.envs import kinematics as K
        R = K.quat_to_mat(rt.q[:, fa + 3:fa + 7])
        return R

    def hook(kind, e, act=None):
        meas = e._measured_model_angles()[:, 0]
        p_fk, _ = e.right_chain.fk(meas)
        R = port()
        zin = -R[:, :, 2]
        pp, _ = rt.plug_in_port()
        d = -pp[:, 2] * 1e3
        if kind == 'reset':
            e.joint_target[:, 0] = meas + torch.as_tensor(REAL_LEAD, dtype=meas.dtype, device=dev)
            e.hist[:] = e._actor_obs()[:, None]
            nan = torch.full((N,), float('nan'), device=dev)
            S.update(live=torch.ones(N, dtype=torch.bool, device=dev), prev_steps=e.steps.clone(), d0=d.clone(), p0=p_fk.clone(),
                     prev_fk=None, run=torch.zeros(N, dtype=torch.long, device=dev), phase=torch.zeros(N, dtype=torch.long, device=dev),
                     cnt=torch.zeros(N, dtype=torch.long, device=dev), f=torch.zeros(N, device=dev), p_push=p_fk.clone(),
                     t_seat=nan.clone(), t_det=nan.clone(), d_det=nan.clone(), d_load=nan.clone(), d_rest=nan.clone(), d_max=torch.zeros(N, device=dev),
                     adv_load=nan.clone(), adv_rest=nan.clone(), f_reached=nan.clone(), adv10=nan.clone(),
                     target=meas.clone(), override=torch.zeros(N, dtype=torch.bool, device=dev))
            return None
        live = S['live'] & (e.steps >= S['prev_steps'])
        S['live'] = live
        S['prev_steps'] = e.steps.clone()
        t = e.steps.float() * dt
        S['d_max'] = torch.where(live, torch.maximum(S['d_max'], d), S['d_max'])
        new_seat = live & torch.isnan(S['t_seat']) & (e.seated_steps >= hold0)
        S['t_seat'] = torch.where(new_seat, t, S['t_seat'])
        fk_d = S['d0'] + ((p_fk - S['p0']) * zin).sum(1) * 1e3            # deploy-like depth: FK advance from the hand-over
        v = torch.zeros_like(fk_d) if S['prev_fk'] is None else (fk_d - S['prev_fk']).abs() / dt
        S['prev_fk'] = fk_d.clone()
        ph = S['phase']
        # phase 0: policy; detector
        ok = (fk_d >= seat_mm - 1.0) & (v < 6.0)
        S['run'] = torch.where(ok, S['run'] + 1, torch.zeros_like(S['run']))
        fire = live & (ph == 0) & (S['run'] >= n_det)
        S['t_det'] = torch.where(fire, t, S['t_det'])
        S['d_det'] = torch.where(fire, d, S['d_det'])
        S['cnt'] = torch.where(fire, torch.zeros_like(S['cnt']), S['cnt'] + 1)
        ph = torch.where(fire, torch.ones_like(ph), ph)
        # phase 1: 0.2 s measured hold, then the push starts from this pose
        go = live & (ph == 1) & (S['cnt'] >= n_hold)
        S['p_push'] = torch.where(go[:, None], p_fk, S['p_push'])
        ph = torch.where(go, torch.full_like(ph, 2), ph)
        S['cnt'] = torch.where(go, torch.zeros_like(S['cnt']), S['cnt'])
        # push / unload load law
        f = S['f']
        up = live & (ph == 2)
        f = torch.where(up, (f + ramp).clamp(max=f_push), f)
        full = up & (f >= f_push - 1e-6)
        S['cnt'] = torch.where(up & ~full, torch.zeros_like(S['cnt']), S['cnt'])
        adv = ((p_fk - S['p_push']) * zin).sum(1) * 1e3
        S['adv10'] = torch.where(up & (f >= 10.0) & torch.isnan(S['adv10']), adv, S['adv10'])
        end_push = full & (S['cnt'] >= n_full)
        S['d_load'] = torch.where(end_push, d, S['d_load'])
        S['adv_load'] = torch.where(end_push, adv, S['adv_load'])
        S['f_reached'] = torch.where(end_push, f, S['f_reached'])
        ph = torch.where(end_push, torch.full_like(ph, 3), ph)
        down = live & (ph == 3)
        f = torch.where(down, (f - ramp).clamp(min=0.), f)
        zero = down & (f <= 0)
        ph = torch.where(zero, torch.full_like(ph, 4), ph)
        S['cnt'] = torch.where(zero, torch.zeros_like(S['cnt']), S['cnt'])
        rest = live & (ph == 4) & (S['cnt'] >= n_hold)
        S['d_rest'] = torch.where(rest & torch.isnan(S['d_rest']), d, S['d_rest'])
        S['adv_rest'] = torch.where(rest & torch.isnan(S['adv_rest']), adv, S['adv_rest'])
        ph = torch.where(rest, torch.full_like(ph, 5), ph)
        S['f'], S['phase'] = f, ph
        # targets
        loaded = live & ((ph == 2) | (ph == 3))
        tgt = meas.clone()
        if loaded.any():
            _, _, J = e.right_chain.fk(meas, with_jac=True)
            rel = torch.einsum('ni,nij->nj', p_fk - S['p_push'], R) * 1e3
            F_lat = (3.0 * (-rel[:, :2])).clamp(-8., 8.)
            F = R[:, :, 0] * F_lat[:, :1] + R[:, :, 1] * F_lat[:, 1:2] + zin * f[:, None]
            W = torch.cat([F, torch.zeros_like(F)], 1)
            tau = torch.einsum('nji,nj->ni', J, W)
            sc = (tmax / tau.abs().clamp(min=1e-9)).clamp(max=1.).min(1).values
            tgt = meas + tau * sc[:, None] / kp
        hold_w = live & (ph >= 1)
        S['target'] = torch.where(hold_w[:, None], tgt, S['target'])
        S['override'] = hold_w
        act = torch.where(hold_w[:, None], torch.zeros_like(act), act)
        return act
    t0 = time.time()
    r = run_cond(env, pol, dict(gap=5.5, ring=.5), rows_base(), seed, on_step=hook)
    g = {k: S[k].cpu().numpy() for k in ('t_seat', 't_det', 'd_det', 'd_load', 'd_rest', 'd_max', 'adv_load', 'adv_rest', 'f_reached', 'adv10')}
    det = np.isfinite(g['t_det'])
    pushed = np.isfinite(g['d_load'])
    rest = np.isfinite(g['d_rest'])
    full_mm = seat_mm - 0.5
    out = dict(policy=policy, cond=cond, f_push_n=f_push, n=N, seed=seed, seat_depth_mm=seat_mm, env_extra=extra,
               sim_seat_rate=float(np.isfinite(g['t_seat']).mean()),
               det_rate=float(det.mean()), t_det_p50_p99=pct(g['t_det'], [50, 99]), det_true_depth_mm_p5_p50_p95=pct(g['d_det'], [5, 50, 95]),
               det_short_gt_1mm=float(((seat_mm - g['d_det']) > 1.0)[det].mean()) if det.any() else None,
               pushed_n=int(pushed.sum()), full_under_load_rate=float((g['d_load'] >= full_mm).sum() / N),
               full_under_load_of_pushed=float((g['d_load'][pushed] >= full_mm).mean()) if pushed.any() else None,
               load_true_depth_mm_p5_p50_p95=pct(g['d_load'], [5, 50, 95]),
               fk_adv_at_10n_mm_p5_p50_p95=pct(g['adv10'], [5, 50, 95]), fk_adv_end_mm_p5_p50_p95=pct(g['adv_load'], [5, 50, 95]),
               fk_adv_rest_mm_p5_p50_p95=pct(g['adv_rest'], [5, 50, 95]), f_reached_p5_p50=pct(g['f_reached'], [5, 50]),
               rest_n=int(rest.sum()), max_depth_mm_p5_p50=pct(g['d_max'], [5, 50]),
               end_classes_not_a_rate=r['classes'], wall_s=round(time.time() - t0))
    if '_m0' in cond and rest.any():
        out.update(rest_true_depth_mm_p5_p50_p95=pct(g['d_rest'], [5, 50, 95]), full_at_rest_rate=float((g['d_rest'] >= full_mm).sum() / N))
    del env
    torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--conds', default='rigid_m0,comp_m0,comp_m10,comp_m20')
    ap.add_argument('--f_push', type=float, default=14.0)
    ap.add_argument('--episode_s', type=float, default=16.0)
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=6500)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {(d['tag'], d['cond'], d['f_push_n'], d['n']) for d in done}
    torch.manual_seed(A.seed)
    for c in A.conds.split(','):
        if (A.tag, c, A.f_push, A.nworld) in seen:
            continue
        r = one(A.policy, c, A.nworld, A.seed, A.f_push, A.episode_s)
        r['tag'] = A.tag
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k != 'env_extra'}), flush=True)
