"""K4 / K7 + deploy-like FK seat detector + final push (force 10 N, force 14 N, 15 mm position push) + unload, in the fitted push
plant with the measured USB-C mating force (docs/real/MATING_FORCE_20261119.md). Sim only. Built on scripts/eval/plant_push_eval.py.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/eval/mating_force_eval.py --policy runs/insert_K4.pt --tag k4 \
        --conds comp_f6.4 --out results/real_align/mating_force/eval.json

Condition '<arm>_<model><peak>[_s<start mm>]': arm 'rigid' | 'comp' (BimanualCfg.right_series_stiffness joint 1 640 N m/rad = 0.22 mm/N);
model 'f' = InsertionCfg.mate_model 'friction', 's' = 'spring' (the old model), 'm0' = no mating force; peak = force at the seat (N);
the profile starts at mate_onset_frac x peak at the engagement depth (default 2 mm, InsertionCfg.mate_start_m) and rises linearly to
the peak at the seat (5.8 mm). Measured 2026-11-19: 2.0 N -> 6.4 N (onset fraction 0.31; kept for the +-30 % cells).
The worlds split in three equal groups (n = nworld / 3 each, same insertion start distribution): force push 10 N, force push 14 N
(deploy final_push_force: 3 N/s ramp, 1.5 s hold, 'still' end, lateral spring 3 N/mm cap 8 N, tau_max, then the unload ramp to 0 N with
the jaws closed, 0.2 s measured hold), position push (deploy final_push: target 15 mm into the port at 3 mm/s smoothstep, 0.5 s hold,
0.3 s measured hold). Advance watch 4 mm on the FK hand advance stops either push (then the 0.3 s measured hold, as the rig).
"""
from __future__ import annotations

import argparse
import json
import math
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
       'slip_m': 0.02, 'slip_deg': 20.0}
K1 = 640.0
ARM = {'rigid': {}, 'comp': {'right_series_stiffness': [K1, None, None, None, None, None]}}
TAU_MAX = [6.0, 6.0, 6.0, 2.5, 2.0, 2.0]
ONSET = 2.0 / 6.4
MODES = ('force10', 'force14', 'pos15')
DEFAULT_GROUPS = 'force10:yaw0,force14:yaw0,pos15:yaw0'


def parse_groups(spec):
    """'mode:axisDEG,...' (axis yaw | pitch) -> [(mode, axis, deg)]. The push frame (axis, lateral line, advance watch) is the true port
    frame turned by deg about the port y axis (yaw: the vertical) or the port x axis (pitch: across); the sign alternates per world.
    'plugSIGMA': the push frame is the plug's own frame at the push start (the 'plug' axis source: hand FK x in-hand estimate) with an
    estimate error N(0, SIGMA deg) per tilt axis. The push frame is frozen at the push start (as deploy).
    'plugfkSIGMA': as deploy final_push.axis_source 'plug': the hand rotation (FK) at the push start x the reader-in-TCP rotation at the
    hand-over (the in-hand estimate is read before the insertion; a reader that turned in the jaws since then is not seen), plus the
    estimate error N(0, SIGMA deg). 'plug' = the TRUE plug frame (an ideal fresh estimate).
    'rotrefMAX': force push with insert.final_push.rot_ref 'handover' (the other worker's option, PUSH_ANGLE_20261119): the rotation
    spring pulls the hand toward its orientation at the insertion hand-over (here: the reset pose), turn clipped to MAX deg, scaled
    with f / force_n; push axis = the true port axis."""
    out = []
    for g in spec.split(','):
        m, a = g.split(':')
        ax = next(k for k in ('yaw', 'pitch', 'plugfk', 'plug', 'rotref') if a.startswith(k))
        out.append((m, ax, float(a[len(ax):])))
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return [float('nan')] * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def pct(x, q):
    x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else None


def parse(cond):
    parts = cond.split('_')
    arm, mf = parts[0], parts[1]
    start = float(parts[2][1:]) * 1e-3 if len(parts) > 2 else 0.002
    if mf == 'm0':
        return arm, None, 0.0, start
    return arm, {'f': 'friction', 's': 'spring'}[mf[0]], float(mf[1:]), start


def rate(mask, sub):
    k, n = int((mask & sub).sum()), int(sub.sum())
    return dict(k=k, n=n, rate=round(k / max(n, 1), 4), wilson=wilson(k, n))


def one(policy, cond, nworld, seed, episode_s, onset, groups=DEFAULT_GROUPS, env_more=None):
    arm, model, peak, start = parse(cond)
    extra = dict(RIG, **ARM[arm])
    if model is None:
        extra['mate_force_n'] = None
    else:
        extra.update(mate_force_n=[peak, peak], mate_model=model, mate_onset_frac=onset, mate_start_m=start)
    extra.update(env_more or {})
    env, pol = build(policy, nworld, seed, extra)
    hold0 = env.hold_steps
    env.hold_steps = 10 ** 9
    env.max_steps = int(round(episode_s / env.dt))
    N, dev, dt = env.N, env.device, env.dt
    seat_mm = env.seat_depth * 1e3
    rt = env.runtime
    act_r = list(rt.indices['right'][2])
    kp = torch.tensor(rt.mjm.actuator_gainprm[act_r, 0], device=dev, dtype=torch.float32)
    tmax = torch.tensor(TAU_MAX, device=dev)
    n_det, n_hold, n_full, n_drop = int(round(0.3 / dt)), int(round(0.2 / dt)), int(round(1.5 / dt)), int(round(0.3 / dt))
    n_still = int(round(0.6 / dt))
    ramp = 3.0 * dt
    GR = parse_groups(groups)
    grp = torch.arange(N, device=dev) * len(GR) // N
    mode = torch.tensor([MODES.index(g[0]) for g in GR], device=dev)[grp]
    f_push = torch.where(mode == 0, 10.0, 14.0).to(dev)
    is_pos = mode == 2
    sgn = 1.0 - 2.0 * (torch.arange(N, device=dev) % 2).float()
    ang = torch.tensor([g[2] for g in GR], device=dev)[grp] * sgn * math.pi / 180
    is_yaw = torch.tensor([g[1] == 'yaw' for g in GR], device=dev)[grp]
    c_, s_ = torch.cos(ang), torch.sin(ang)
    Roff = torch.eye(3, device=dev).repeat(N, 1, 1)
    # yaw: about port y; pitch: about port x
    Roff[:, 0, 0] = torch.where(is_yaw, c_, 1.); Roff[:, 0, 2] = torch.where(is_yaw, s_, 0.)
    Roff[:, 2, 0] = torch.where(is_yaw, -s_, 0.); Roff[:, 2, 2] = c_
    Roff[:, 1, 1] = torch.where(is_yaw, 1., c_); Roff[:, 1, 2] = torch.where(is_yaw, 0., -s_)
    Roff[:, 2, 1] = torch.where(is_yaw, 0., s_)
    is_plug = torch.tensor([g[1] in ('plug', 'plugfk') for g in GR], device=dev)[grp]
    is_pfk = torch.tensor([g[1] == 'plugfk' for g in GR], device=dev)[grp]
    is_rr = torch.tensor([g[1] == 'rotref' for g in GR], device=dev)[grp]
    rr_max = torch.tensor([g[2] for g in GR], device=dev)[grp] * math.pi / 180
    Roff = torch.where((is_plug | is_rr)[:, None, None], torch.eye(3, device=dev), Roff)
    Rnoise_ok = ~is_rr
    gen = torch.Generator(device=dev); gen.manual_seed(seed + 17)
    rv = torch.randn(N, 3, device=dev, generator=gen) * torch.tensor([1., 1., 0.], device=dev) \
        * torch.tensor([g[2] for g in GR], device=dev)[grp][:, None] * math.pi / 180
    from blind_insertion.envs import kinematics as K
    Rnoise = K.axis_angle_mat(rv / rv.norm(dim=1, keepdim=True).clamp(min=1e-12), rv.norm(dim=1)) if hasattr(K, 'axis_angle_mat') else None
    n_pos = int(round(15.0 / 3.0 / dt))
    n_pos_hold = int(round(0.5 / dt))
    S = {}
    orig = env._update_joint_targets

    def patched(action, measured):
        p = orig(action, measured)
        m = S.get('override')
        if m is not None and m.any():
            p[m, 0] = S['target'][m]
            env.joint_target.copy_(p)
        return p
    env._update_joint_targets = patched
    from blind_insertion.envs import kinematics as K

    def hook(kind, e, act=None):
        meas = e._measured_model_angles()[:, 0]
        p_fk, Rh = e.right_chain.fk(meas)
        fa = rt.pc_qadr
        Rt = K.quat_to_mat(rt.q[:, fa + 3:fa + 7])
        zt = -Rt[:, :, 2]                        # true axis (detector depth: the deploy seat2 uses the port estimate; small angles)
        R = Rt @ Roff                            # push frame (estimated port frame)
        zin = -R[:, :, 2]
        pp, pR_ = rt.plug_in_port()
        d = -pp[:, 2] * 1e3
        tilt = torch.rad2deg(torch.acos(pR_[:, 2, 2].clamp(-1, 1)))
        if kind == 'reset':
            e.joint_target[:, 0] = meas + torch.as_tensor(REAL_LEAD, dtype=meas.dtype, device=dev)
            e.hist[:] = e._actor_obs()[:, None]
            nan = torch.full((N,), float('nan'), device=dev)
            z = lambda: torch.zeros(N, dtype=torch.long, device=dev)
            S.update(live=torch.ones(N, dtype=torch.bool, device=dev), prev_steps=e.steps.clone(), d0=d.clone(), p0=p_fk.clone(),
                     prev_fk=None, run=z(), phase=z(), cnt=z(), hist_adv=torch.zeros(N, n_still + 1, device=dev),
                     f=torch.zeros(N, device=dev), p_push=p_fk.clone(), q_push=meas.clone(), Jinv=torch.zeros(N, 6, 6, device=dev),
                     t_seat=nan.clone(), t_det=nan.clone(), d_det=nan.clone(), d_load=nan.clone(), d_rest=nan.clone(),
                     d_max=torch.zeros(N, device=dev), adv_load=nan.clone(), adv_rest=nan.clone(), f_reached=nan.clone(),
                     f_cmd_end=nan.clone(), watch=torch.zeros(N, dtype=torch.bool, device=dev), slip_end=nan.clone(),
                     R_push=torch.eye(3, device=dev).repeat(N, 1, 1), Rh0=torch.eye(3, device=dev).repeat(N, 1, 1), Rh_ho=Rh.clone(),
                     w_ref=torch.zeros(N, 3, device=dev), theta_push=nan.clone(), theta_rest=nan.clone(), rr_turn=nan.clone(), d_pre_push=nan.clone(), ih_tilt=nan.clone(), ih_turn=nan.clone(), port_tilt_push=nan.clone(), tilt_rest=nan.clone(), target=meas.clone(), override=torch.zeros(N, dtype=torch.bool, device=dev))
            return None
        live = S['live'] & (e.steps >= S['prev_steps'])
        S['live'] = live
        S['prev_steps'] = e.steps.clone()
        t = e.steps.float() * dt
        S['d_max'] = torch.where(live, torch.maximum(S['d_max'], d), S['d_max'])
        new_seat = live & torch.isnan(S['t_seat']) & (e.seated_steps >= hold0)
        S['t_seat'] = torch.where(new_seat, t, S['t_seat'])
        fk_d = S['d0'] + ((p_fk - S['p0']) * zt).sum(1) * 1e3
        v = torch.zeros_like(fk_d) if S['prev_fk'] is None else (fk_d - S['prev_fk']).abs() / dt
        S['prev_fk'] = fk_d.clone()
        ph, cnt = S['phase'], S['cnt']
        # 0: policy + seat2 detector on the FK depth
        ok = (fk_d >= seat_mm - 1.0) & (v < 6.0)
        S['run'] = torch.where(ok, S['run'] + 1, torch.zeros_like(S['run']))
        fire = live & (ph == 0) & (S['run'] >= n_det)
        S['t_det'] = torch.where(fire, t, S['t_det'])
        S['d_det'] = torch.where(fire, d, S['d_det'])
        cnt = torch.where(fire, torch.zeros_like(cnt), cnt + 1)
        ph = torch.where(fire, torch.ones_like(ph), ph)
        # 1: 0.2 s measured hold, then the push starts from this pose
        go = live & (ph == 1) & (cnt >= n_hold)
        if go.any():
            _, _, J0 = e.right_chain.fk(meas, with_jac=True)
            S['Jinv'] = torch.where(go[:, None, None], torch.linalg.pinv(J0), S['Jinv'])
        S['p_push'] = torch.where(go[:, None], p_fk, S['p_push'])
        S['Rh0'] = torch.where(go[:, None, None], Rh, S['Rh0'])
        if go.any():
            w = K.rot_log(S['Rh_ho'] @ Rh.transpose(1, 2))
            wn = w.norm(dim=1, keepdim=True)
            w = w * (rr_max[:, None] / wn.clamp(min=1e-9)).clamp(max=1.)
            S['w_ref'] = torch.where((go & is_rr)[:, None], w, S['w_ref'])
            S['rr_turn'] = torch.where(go, wn[:, 0] * 180 / math.pi, S['rr_turn'])
            if hasattr(e, 'mate_theta'):
                S['theta_push'] = torch.where(go, e.mate_theta, S['theta_push'])
        S['q_push'] = torch.where(go[:, None], meas, S['q_push'])
        S['d_pre_push'] = torch.where(go, d, S['d_pre_push'])
        pa = rt.lay.peg_qadr
        Rpeg = torch.where(is_pfk[:, None, None], Rh @ e.ref_rel[:, :3, :3], K.quat_to_mat(rt.q[:, pa + 3:pa + 7])) @ Rnoise
        S['R_push'] = torch.where(go[:, None, None], torch.where(is_plug[:, None, None], Rpeg, R), S['R_push'])
        R = S['R_push']
        zin = -R[:, :, 2]
        if go.any():
            rel_now = e._reader_in_tcp()
            ih = e.inhand_error(rel_now)[:, 3:].norm(dim=1)
            S['ih_tilt'] = torch.where(go, ih, S['ih_tilt'])
            dR = K.rot_log(rel_now[:, :3, :3] @ e.ref_rel[:, :3, :3].transpose(1, 2)).norm(dim=1) * 180 / math.pi
            S['ih_turn'] = torch.where(go, dR, S['ih_turn'])
            S['port_tilt_push'] = torch.where(go, tilt, S['port_tilt_push'])
        ph = torch.where(go, torch.where(is_pos, torch.full_like(ph, 6), torch.full_like(ph, 2)), ph)
        cnt = torch.where(go, torch.zeros_like(cnt), cnt)
        adv = ((p_fk - S['p_push']) * zin).sum(1) * 1e3
        # force push: 2 ramp/hold, 3 unload ramp, 4 rest hold; position push: 6 move, 7 hold; 8 drop (0.3 s measured); 5 done
        f = S['f']
        up = live & (ph == 2)
        f = torch.where(up, torch.minimum(f + ramp, f_push), f)
        full = up & (f >= f_push - 1e-6)
        cnt = torch.where(up & ~full, torch.zeros_like(cnt), cnt)
        S['hist_adv'] = torch.where(full[:, None], torch.roll(S['hist_adv'], -1, 1), S['hist_adv'])
        S['hist_adv'][:, -1] = torch.where(full, adv, S['hist_adv'][:, -1])
        still = full & (cnt > n_still) & ((S['hist_adv'][:, -1] - S['hist_adv'][:, 0]).abs() < 0.15)
        watch = live & ((ph == 2) | (ph == 6) | (ph == 7)) & (adv > 4.0)
        S['watch'] = S['watch'] | watch
        end_f = (full & ((cnt >= n_full) | still)) & ~watch
        pos_mv = live & (ph == 6)
        pos_end = live & (ph == 7) & (cnt >= n_pos_hold) & ~watch
        end_load = end_f | pos_end | watch
        S['d_load'] = torch.where(end_load, d, S['d_load'])
        S['adv_load'] = torch.where(end_load, adv, S['adv_load'])
        S['f_reached'] = torch.where(end_load & ~is_pos, f, S['f_reached'])
        if hasattr(e, '_last'):
            S['slip_end'] = torch.where(end_load, e._last['d_rel'] * 1e3, S['slip_end'])
        # commanded axial force at the push end: least-squares wrench of the PD torque, force part along the axis (tip map)
        if end_load.any():
            _, _, J = e.right_chain.fk(meas, with_jac=True)
            tau = kp * (S['target'] - meas)
            W = torch.linalg.lstsq(J.transpose(1, 2), tau[..., None]).solution[..., 0]
            S['f_cmd_end'] = torch.where(end_load, (W[:, :3] * zt).sum(1), S['f_cmd_end'])
        ph = torch.where(end_f, torch.full_like(ph, 3), ph)
        ph = torch.where(pos_end | watch, torch.full_like(ph, 8), ph)
        cnt = torch.where(pos_end | watch, torch.zeros_like(cnt), cnt)
        f = torch.where(watch, torch.zeros_like(f), f)
        down = live & (ph == 3)
        f = torch.where(down, (f - ramp).clamp(min=0.), f)
        zero = down & (f <= 0)
        ph = torch.where(zero, torch.full_like(ph, 4), ph)
        cnt = torch.where(zero, torch.zeros_like(cnt), cnt)
        mv_done = pos_mv & (cnt >= n_pos)
        ph = torch.where(mv_done, torch.full_like(ph, 7), ph)
        cnt = torch.where(mv_done, torch.zeros_like(cnt), cnt)
        rest = live & (((ph == 4) & (cnt >= n_hold)) | ((ph == 8) & (cnt >= n_drop)))
        S['d_rest'] = torch.where(rest & torch.isnan(S['d_rest']), d, S['d_rest'])
        S['adv_rest'] = torch.where(rest & torch.isnan(S['adv_rest']), adv, S['adv_rest'])
        S['tilt_rest'] = torch.where(rest & torch.isnan(S['tilt_rest']), tilt, S['tilt_rest'])
        if hasattr(e, 'mate_theta'):
            S['theta_rest'] = torch.where(rest & torch.isnan(S['theta_rest']), e.mate_theta, S['theta_rest'])
        ph = torch.where(rest, torch.full_like(ph, 5), ph)
        S['f'], S['phase'], S['cnt'] = f, ph, cnt
        # targets
        tgt = meas.clone()
        loaded = live & ((ph == 2) | (ph == 3))
        if loaded.any():
            _, _, J = e.right_chain.fk(meas, with_jac=True)
            rel = torch.einsum('ni,nij->nj', p_fk - S['p_push'], R) * 1e3
            F_lat = (3.0 * (-rel[:, :2])).clamp(-8., 8.)
            F = R[:, :, 0] * F_lat[:, :1] + R[:, :, 1] * F_lat[:, 1:2] + zin * f[:, None]
            sref = (f / f_push).clamp(0, 1)
            wr = S['w_ref'] * sref[:, None]
            Rref = K.axis_angle_mat(wr / wr.norm(dim=1, keepdim=True).clamp(min=1e-12), wr.norm(dim=1)) @ S['Rh0']
            Re = Rref @ Rh.transpose(1, 2)            # deploy rotation spring: 6 N m/rad to the push-start orientation, cap 0.5 N m
            w = torch.stack([Re[:, 2, 1] - Re[:, 1, 2], Re[:, 0, 2] - Re[:, 2, 0], Re[:, 1, 0] - Re[:, 0, 1]], 1) / 2
            W = torch.cat([F, (6.0 * w).clamp(-0.5, 0.5)], 1)
            tau = torch.einsum('nji,nj->ni', J, W)
            sc = (tmax / tau.abs().clamp(min=1e-9)).clamp(max=1.).min(1).values
            tgt = torch.where(loaded[:, None], meas + tau * sc[:, None] / kp, tgt)
        posw = live & ((ph == 6) | (ph == 7))
        if posw.any():
            u = (cnt.float() / n_pos).clamp(max=1.)
            u = torch.where(ph == 7, torch.ones_like(u), u)
            u = u * u * (3 - 2 * u)
            dx = torch.cat([zin * (15e-3 * u)[:, None], torch.zeros_like(zin)], 1)
            qt = S['q_push'] + torch.einsum('nij,nj->ni', S['Jinv'], dx)
            tgt = torch.where(posw[:, None], qt, tgt)
        hold_w = live & (ph >= 1)
        # phases 1, 4, 5, 8 hold the measured pose (tgt = meas)
        S['target'] = torch.where(hold_w[:, None], tgt, S['target'])
        S['override'] = hold_w
        act = torch.where(hold_w[:, None], torch.zeros_like(act), act)
        return act
    t0 = time.time()
    r = run_cond(env, pol, dict(gap=5.5, ring=.5), rows_base(), seed, on_step=hook)
    keys = ('t_seat', 't_det', 'd_det', 'd_load', 'd_rest', 'd_max', 'adv_load', 'adv_rest', 'f_reached', 'f_cmd_end', 'slip_end', 'd_pre_push', 'ih_tilt', 'ih_turn', 'theta_push', 'theta_rest', 'rr_turn', 'port_tilt_push', 'tilt_rest')
    g = {k: S[k].cpu().numpy() for k in keys}
    watch = S['watch'].cpu().numpy()
    G = grp.cpu().numpy()
    sgnn = sgn.cpu().numpy()
    full_mm = seat_mm - 0.5
    det = np.isfinite(g['t_det'])
    allw = np.ones(N, bool)
    out = dict(policy=policy, cond=cond, mate_model=model, mate_peak_n=peak, mate_onset_n=round(peak * onset, 3) if model else 0.,
               mate_start_mm=start * 1e3, n=N, seed=seed, seat_depth_mm=seat_mm, env_extra=extra,
               sim_seat=rate(np.isfinite(g['t_seat']), allw), det=rate(det, allw),
               t_det_p50_p99=pct(g['t_det'], [50, 99]), det_true_depth_mm_p5_p50_p95=pct(g['d_det'], [5, 50, 95]),
               det_short_gt_1mm=rate((seat_mm - g['d_det']) > 1.0, det), det_short_gt_0p5mm=rate((seat_mm - g['d_det']) > 0.5, det),
               max_depth_mm_p5_p50=pct(g['d_max'], [5, 50]), end_classes_not_a_rate=r['classes'], groups={})
    out['groups_spec'] = groups
    for gi, (gm, gax, gdeg) in enumerate(GR):
        name = f'{gm}:{gax}{gdeg:g}'
        sub = G == gi
        pushed = sub & np.isfinite(g['d_load'])
        rest = sub & np.isfinite(g['d_rest'])
        sh = seat_mm - g['d_rest']
        out['groups'][name] = dict(
            n=int(sub.sum()), det=rate(det, sub), pushed_n=int(pushed.sum()), watch_stop=rate(watch, pushed),
            full_under_load=rate(g['d_load'] >= full_mm, sub), full_under_load_of_pushed=rate(g['d_load'] >= full_mm, pushed),
            full_at_rest=rate(g['d_rest'] >= full_mm, sub), full_at_rest_of_pushed=rate(g['d_rest'] >= full_mm, rest),
            rest_short_0p5_1=rate((sh > 0.5) & (sh <= 1.0), rest), rest_short_1_2=rate((sh > 1.0) & (sh <= 2.0), rest), rest_short_gt2=rate(sh > 2.0, rest),
            pre_push_depth_mm_p5_p50_p95=pct(g['d_pre_push'][sub], [5, 50, 95]),
            load_depth_mm_p5_p50_p95=pct(g['d_load'][sub], [5, 50, 95]), rest_depth_mm_p5_p50_p95=pct(g['d_rest'][sub], [5, 50, 95]),
            fk_adv_end_mm_p5_p50_p95=pct(g['adv_load'][sub], [5, 50, 95]), fk_adv_rest_mm_p5_p50_p95=pct(g['adv_rest'][sub], [5, 50, 95]),
            f_cmd_axial_end_n_p5_p50_p95=pct(g['f_cmd_end'][sub], [5, 50, 95]), f_reached_p5_p50=pct(g['f_reached'][sub], [5, 50]),
            reader_shift_in_jaws_end_mm_p50_p95=pct(g['slip_end'][sub], [50, 95]),
            inhand_rot_at_push_deg_p5_p50_p95=pct(g['ih_tilt'][sub], [5, 50, 95]),
            reader_turn_in_jaws_handover_to_push_deg_p5_p50_p95=pct(g['ih_turn'][sub], [5, 50, 95]),
            hand_angle_theta_push_deg_p5_p50_p95=pct(g['theta_push'][sub], [5, 50, 95]), hand_angle_theta_rest_deg_p5_p50_p95=pct(g['theta_rest'][sub], [5, 50, 95]),
            hand_turn_handover_to_push_deg_p5_p50_p95=pct(g['rr_turn'][sub], [5, 50, 95]),
            by_theta_push={f'{lo}-{hi}': rate(g['d_rest'] >= full_mm, rest & (g['theta_push'] >= lo) & (g['theta_push'] < hi))
                           for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 7.5), (7.5, 90))}, plug_tilt_in_port_at_push_deg_p5_p50_p95=pct(g['port_tilt_push'][sub], [5, 50, 95]),
            plug_tilt_in_port_rest_deg_p50_p95=pct(g['tilt_rest'][sub], [50, 95]),
            short_tilted_gt3deg=rate((sh > 0.5) & (g['tilt_rest'] > 3.0), rest),
            by_sign={sg: dict(full_at_rest=rate(g['d_rest'] >= full_mm, sub & (sgnn == sv))) for sg, sv in (('pos', 1.0), ('neg', -1.0))},
            by_inhand_rot={f'{lo}-{hi}': rate(g['d_rest'] >= full_mm, rest & (g['ih_tilt'] >= lo) & (g['ih_tilt'] < hi))
                           for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 90))})
    out['wall_s'] = round(time.time() - t0)
    del env
    torch.cuda.empty_cache()
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--conds', default='comp_f6.4')
    ap.add_argument('--onset', type=float, default=ONSET)
    ap.add_argument('--groups', default=DEFAULT_GROUPS)
    ap.add_argument('--env_more', default='', help='JSON merged into the env extra (e.g. a wider in-hand rotation)')
    ap.add_argument('--episode_s', type=float, default=20.0)
    ap.add_argument('--nworld', type=int, default=3072)
    ap.add_argument('--seed', type=int, default=6600)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(A.seed)
    for c in A.conds.split(','):
        done = json.loads(out.read_text()) if out.exists() else []
        if (A.tag, c, A.nworld, A.groups) in {(d['tag'], d['cond'], d['n'], d.get('groups_spec', DEFAULT_GROUPS)) for d in done}:
            continue
        r = one(A.policy, c, A.nworld, A.seed, A.episode_s, A.onset, A.groups, json.loads(A.env_more) if A.env_more else None)
        r['tag'] = A.tag
        done = json.loads(out.read_text()) if out.exists() else []
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k not in ('env_extra', 'groups')}), flush=True)
        for k, v in r['groups'].items():
            print(' ', k, json.dumps(v), flush=True)
