"""K-J in the batched insertion env with and without InsertGuard, and the old and new seat detectors replayed on the same runs.

Episodes do not end at the sim's success (the robot has no such signal): the policy goes on for the whole 10 s and the detectors run on believed FK. Truth: the env's
seated hold. A guard or abort stop before the true seat counts as a failure (conservative: on the robot the run ends there).

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/insert_guard_eval.py --set live --nworld 1024 --out results/real_align/guard.json
Sets: live (held-out chain hand-over states), r3x15 (training pool x 1.5), case (plug on the case: gap 11.5 mm, 5-14 mm high), gap115 (aligned, gap 11.5 mm),
windup (live pool + real-like target offset at the hand-over).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/eval'))
from blind_insertion.envs.inhand import measured_mean  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy, wilson  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402
from deploy.bimanual.insert import SeatDetector  # noqa: E402
from deploy.bimanual.insert_guard import InsertGuard, SeatDetector2  # noqa: E402

POL = ROOT / 'deploy/bundle/models/insert_KJ.pt'
BASE = ROOT / 'results/usbc/policy/insertion/aux/starts_round3_plus_chain.npz'
# real tracking at the hand-over (run full_A/A2/ai4 mean): joint target - believed, rad
GRASP_SITE = 'grasp_site'
REAL_WINDUP = torch.tensor([-.001, -.027, .017, -.001, .029, .001])


def rows(kind, g, n=4096):
    z = np.load(BASE)
    r = np.stack([z[k] for k in ('x_mm', 'y_mm', 'gap_mm', 'rx_deg', 'ry_deg', 'roll_deg')], 1)
    r = r[np.isfinite(r).all(1) & (np.hypot(r[:, 0], r[:, 1]) <= 8) & (r[:, 2] >= .5) & (r[:, 2] <= 15)]
    r = r[g.integers(0, len(r), n)].copy()
    if kind == 'case':
        r[:, 0] = g.uniform(-1, 1, n)
        r[:, 1] = g.uniform(5, 14, n)
        r[:, 2] = 11.5
    elif kind == 'gap115':
        r[:, 2] = 11.5
    elif kind == 'r3x15':
        r[:, [0, 1, 3, 4, 5]] *= 1.5
    return r


def build(a):
    pol = InsertionPolicy(str(POL), 'cuda:0')
    ej = dict(pol.meta['env_json'], hard_frac=0., scale_range_frac=0.)
    ej['live_frac'] = 1. if a.set in ('live', 'windup') else 0.
    if ej['live_frac']:
        ej['live_split'] = 'eval'
    if a.friction:
        ej['friction_scale_range'] = json.loads(a.friction)
    if a.enc_offset_deg is not None:
        ej['enc_offset_deg'] = a.enc_offset_deg
    env = make_env(argparse.Namespace(nworld=a.nworld, seed=a.seed, action_mode='accumulate', env_json=json.dumps(ej)))
    return env, pol


def run(env, pol, a, guard_on):
    dev, N, T = env.device, env.N, env.max_steps
    g = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    if a.set not in ('live', 'windup'):
        env._npz = torch.tensor(rows(a.set, g), device=dev, dtype=torch.float32)
    env.refresh_pool()
    Rs, ps = measured_mean(device=dev)
    s = (Rs.reshape(3, 3).float().T @ ps.reshape(3).float())
    from blind_insertion.envs import kinematics as K
    m = env.runtime.mjm
    ch = K.Chain(m, m.site(GRASP_SITE).id, np.array(env.runtime.indices['right'][0]), device=dev)
    latch, alive, fail = [torch.zeros(N, dtype=torch.bool, device=dev) for _ in range(3)]
    alive[:] = True
    t_true = torch.full((N,), -1, device=dev)
    orig_tr = env._transition

    def tr():
        sc, term, trunc, info = orig_tr()
        su = info['success']
        new = su & ~latch & alive
        t_true[new] = env.steps[new]
        latch.logical_or_(su & alive)
        term2 = term & ~su
        trunc2 = env.steps >= T
        env.done = term2 | trunc2
        fail.logical_or_(term2 & alive & ~latch)
        return sc, term2, trunc2, info
    env._transition = tr
    env.reset_all()
    pR = env._port_pose(env.pc_start)[1]
    q0 = env._measured_model_angles()[:, 0]
    if a.set == 'windup' or a.windup:
        env.joint_target[:, 0] = q0 + REAL_WINDUP.to(dev)
    gd = InsertGuard(ch, s[None].expand(N, 3), pR, N, dev, bound_mm=a.bound, hard_mm=a.hard, stall_s=a.stall_s, stall_eps_mm=a.stall_eps, lead_min_mm=a.lead_min)
    gd.reset(torch.arange(N, device=dev), q0[:, None].squeeze(1), pR)
    orig_upd = env._update_joint_targets
    sidx = {}

    def upd(action, measured):
        proposed = orig_upd(action, measured)
        lim, info = gd.limit(measured[:, 0], proposed[:, 0], latch)
        sidx.update(lead=info['rel_t'] - info['rel_m'], cut=info['cut'])
        if guard_on:
            proposed = proposed.clone()
            proposed[:, 0] = lim
            env.joint_target.copy_(proposed)
        return proposed
    env._update_joint_targets = upd
    # truth-based gap/lateral at the hand-over, with noise like a measurement
    start = env.start['err_p'].clone() if a.set not in ('live', 'windup') else None
    p, _ = env.runtime.plug_in_port()
    g0_true, lat0_true = p[:, 2].clone() * 1e3, p[:, :2].clone() * 1e3
    gn = torch.randn(N, 3, device=dev)
    g0 = g0_true + gn[:, 2] * a.sig_gap
    lat0 = lat0_true + gn[:, :2] * a.sig_lat
    det2 = SeatDetector2(N, dev, g0, lat0, seat_depth_mm=5.8, margin_mm=a.margin, gate_mm=a.gate)
    det2b = SeatDetector2(N, dev, g0_true, lat0_true, seat_depth_mm=5.8, margin_mm=a.margin, gate_mm=a.gate)
    old = [SeatDetector(pR[i].cpu(), .006, .0058, .001, .2, .005, env.dt) for i in range(min(N, a.old_n))]
    tcp_ch = env.right_chain
    p0, _ = tcp_ch.fk(q0)
    for i, d in enumerate(old):
        d.update(p0[i].cpu().numpy())
    hist, _ = env.observe()
    t2, t2b, told = [torch.full((N,), -1, device=dev) for _ in range(3)]
    tstop = torch.full((N,), -1, device=dev)
    stop_kind = torch.zeros(N, dtype=torch.long, device=dev)
    max_lat, depth_max = torch.zeros(N, 2, device=dev), torch.zeros(N, device=dev)
    series = []
    with torch.no_grad():
        for t in range(T):
            act = pol(hist)
            hist, _, _, done, info = env.step(act)
            alive_now = alive & (env.steps > 0) & ~done
            qb = env._measured_model_angles()[:, 0]
            rm = gd.rel(qb)
            rt = gd.rel(env.joint_target[:, 0])
            r2 = det2.update(rm)
            r2b = det2b.update(rm)
            a2 = alive & (t2 < 0) & r2['seated']
            t2[a2] = t
            a2 = alive & (t2b < 0) & r2b['seated']
            t2b[a2] = t
            pn = tcp_ch.fk(qb)[0].cpu().numpy()
            for i, d in enumerate(old):
                if alive[i]:
                    if d.update(pn[i])[2] and told[i] < 0:
                        told[i] = t
            max_lat = torch.where(alive[:, None], torch.maximum(max_lat, rm[:, :2].abs()), max_lat)
            depth_max = torch.where(alive, torch.maximum(depth_max, -rm[:, 2]), depth_max)
            st = gd.stop.clone()
            nw = alive & (tstop < 0) & (st > 0)
            tstop[nw], stop_kind[nw] = t, st[nw]
            if t % 25 == 0:
                series.append(dict(t=t * env.dt, rel_m=rm[alive].mean(0).tolist() if alive.any() else None, rel_m_notseated=rm[alive & ~latch].mean(0).tolist() if (alive & ~latch).any() else None, up_p90=float(rm[alive & ~latch, 1].quantile(.9)) if (alive & ~latch).any() else None, rel_t_notseated=rt[alive & ~latch].mean(0).tolist() if (alive & ~latch).any() else None))
            alive = alive & ~done
    env._transition, env._update_joint_targets = orig_tr, orig_upd
    if not guard_on:
        pass
    ok = latch & ~fail
    if guard_on:      # a stop before the true seat ends the run: failure
        ok = ok & ~((tstop >= 0) & ((t_true < 0) | (tstop < t_true)))
    n = N
    k = int(ok.sum())
    tr_s = t_true.cpu().numpy() >= 0
    tr_ok = ok.cpu().numpy()
    def det(tt):
        d = tt.cpu().numpy() >= 0
        tp = int((d & tr_ok).sum())
        fp = int((d & ~tr_ok).sum())
        return dict(detected=int(d.sum()), true_seat=tp, false_seat=fp, missed=int((~d & tr_ok).sum()), false_rate=fp / max((~tr_ok).sum(), 1), false_ci=wilson(fp, int((~tr_ok).sum())), miss_rate=int((~d & tr_ok).sum()) / max(tr_ok.sum(), 1), miss_ci=wilson(int((~d & tr_ok).sum()), int(tr_ok.sum())))
    tr_old_ok = tr_ok[:len(old)]
    d_old = told[:len(old)].cpu().numpy() >= 0
    out = dict(set=a.set, guard=guard_on, n=n, k=k, rate=k / n, wilson=wilson(k, n), stops={str(j): int((stop_kind == j).sum()) for j in (1, 2)}, stop_before_seat=int(((tstop >= 0) & ((t_true < 0) | (tstop < t_true))).sum()), clamped_worlds=int((gd.clamped > 0).sum()),
               max_lat_seated_p50_p99_max=[[float(np.percentile(max_lat[latch].cpu().numpy()[:, i], x)) for x in (50, 99, 99.9, 100)] for i in range(2)] if latch.any() else None, depth_seated=[float(x) for x in np.percentile(depth_max[latch].cpu().numpy(), [0.1, 1, 50, 99])] if latch.any() else None,
               t_true_p50_p99_max=[float(np.percentile(t_true[t_true >= 0].cpu().numpy(), x)) * env.dt for x in (50, 99, 100)] if (t_true >= 0).any() else None, seat2=det(t2), seat2_exactgap=det(t2b),
               seat_old=dict(n=len(old), false_seat=int((d_old & ~tr_old_ok).sum()), not_seated=int((~tr_old_ok).sum()), missed=int((~d_old & tr_old_ok).sum()), seated=int(tr_old_ok.sum())), series=series, params=dict(bound=a.bound, hard=a.hard, stall_s=a.stall_s, gate=a.gate, margin=a.margin, sig_gap=a.sig_gap, sig_lat=a.sig_lat, friction=a.friction, windup=a.windup, enc=a.enc_offset_deg))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True, choices=('live', 'r3x15', 'case', 'gap115', 'windup'))
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=3000)
    ap.add_argument('--bound', type=float, nargs=2, default=(5., 5.))
    ap.add_argument('--hard', type=float, default=3.)
    ap.add_argument('--stall_s', type=float, default=3.)
    ap.add_argument('--stall_eps', type=float, default=.3)
    ap.add_argument('--lead_min', type=float, default=3.)
    ap.add_argument('--gate', type=float, nargs=2, default=(3., 6.))
    ap.add_argument('--margin', type=float, default=1.)
    ap.add_argument('--sig_gap', type=float, default=1.)
    ap.add_argument('--sig_lat', type=float, default=.5)
    ap.add_argument('--friction', default='')
    ap.add_argument('--windup', action='store_true')
    ap.add_argument('--enc_offset_deg', type=float, default=None)
    ap.add_argument('--old_n', type=int, default=1024)
    ap.add_argument('--guard', choices=('on', 'off', 'both'), default='both')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    env, pol = build(a)
    out = Path(a.out)
    done = json.loads(out.read_text()) if out.exists() else []
    for gon in {'on': [True], 'off': [False], 'both': [False, True]}[a.guard]:
        r = run(env, pol, a, gon)
        done.append(r)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(done, indent=1))
        print(json.dumps({k: v for k, v in r.items() if k != 'series'}), flush=True)
