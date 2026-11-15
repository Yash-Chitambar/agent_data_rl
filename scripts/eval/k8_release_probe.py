"""Batched release probe with the deploy retreat (docs/real/K8_20261115.md, task B). Sim only.

contact_sens.py's start (cell real_ho, rig spot, real-like plant) and seat detector; then, per world, the deploy release order of
RealChain.retreat_right + unload_left: 0.2 s measured-pose hold, jaws open with the right target at the measured pose and the left
press target at its measured pose (0.4 s, insert.release_open_s), right TCP back 50 mm along the port axis at 20 mm/s (smoothstep,
IK), left hold 25 ticks, left hand up 30 mm at 20 mm/s. True depth at: rest, open end, retreat end, unload end.
'in after release' = depth at the end >= seat - 0.25 mm (the full-depth rule of contact_sens.py). Slip / far / pushed ends are off.

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=. python scripts/eval/k8_release_probe.py --policy runs/insert_K4.pt --tag k4 \
        --plug nom --variants base,pmu03
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
OUT = ROOT / 'results/real_align/k8/release_probe'
EXTRA = {'pmu03': dict(port_mu=(0.30, 0.30)), 'pmu02_04': dict(port_mu=(0.20, 0.40))}


def main():
    from scripts.eval import contact_sens as CS
    from scripts.eval.kj_start_map import rows_base
    from scripts.eval.kj_real4 import REAL_LEAD
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--plug', default='nom', choices=sorted(CS.PLUGS))
    ap.add_argument('--variants', default='base,pmu03')
    ap.add_argument('--n', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=8700)
    ap.add_argument('--open_m', type=float, default=1.0, help='right finger position target at the release (m); 1.0 = past full open; '
                    '0.024 = the dry-run chain (SimIO caps the right jaws at gripper.right_approach_open_m)')
    A = ap.parse_args()
    CS.VARIANTS.update(EXTRA)
    names = A.variants.split(',')
    CS.patch_scene('4x1.2', A.plug)
    torch.manual_seed(A.seed)
    N = A.n * len(names)
    env, pol = CS.build(A.policy, N, A.seed, 'real')
    c = env.icfg
    c.slip_m, c.slip_deg, c.far_m, c.pc_move_m, c.pc_tilt_deg = 1e9, 1e9, 1e9, 1e9, 1e9
    g = np.random.default_rng(A.seed)
    rows, _ = CS.ring_rows(rows_base(), g, N * env.icfg.pool_per_world, 0.5)
    dev, dt, rt = env.device, env.dt, env.runtime
    gen = torch.Generator(device=dev); gen.manual_seed(A.seed)
    CS.fixed_pool(env, rows)
    env.hold_steps = 10 ** 9
    rt.grip_target = torch.full((N,), float('nan'), device=dev)
    env.reset_all()
    n = A.n
    for k, name in enumerate(names):
        CS.apply_variant(env, torch.arange(k * n, (k + 1) * n, device=dev), CS.VARIANTS[name], gen)
    q0 = env._measured_model_angles()[:, 0]
    env.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=dev)
    env.hist[:] = env._actor_obs()[:, None]
    hist, _ = env.observe()
    seat_mm = env.seat_depth * 1e3
    r = lambda s: int(round(s / dt))   # noqa: E731
    n_det, n_rest, n_open, n_ret, n_uh, n_lift = r(0.3), r(0.2), r(0.4), r(0.05 / 0.02), 25, r(0.03 / 0.02)
    e_rest, e_open = n_rest, n_rest + n_open
    e_ret, e_uh, e_end = e_open + n_ret, e_open + n_ret + n_uh, e_open + n_ret + n_uh + n_lift
    nan = lambda: torch.full((N,), float('nan'), device=dev)   # noqa: E731
    S = {k: nan() for k in ('t_seat', 't_det', 'd_rest', 'd_open', 'd_ret', 'd_end', 'lat_end', 'finger_open_mm')}
    run = torch.zeros(N, dtype=torch.long, device=dev); h = torch.full((N,), -1, dtype=torch.long, device=dev)
    live = torch.ones(N, dtype=torch.bool, device=dev)
    qr0 = torch.zeros(N, 6, device=dev); ql0 = torch.zeros(N, 6, device=dev)
    p0 = torch.zeros(N, 3, device=dev); R0 = torch.zeros(N, 3, 3, device=dev); z = torch.zeros(N, 3, device=dev)
    pl0 = torch.zeros(N, 3, device=dev); Rl0 = torch.zeros(N, 3, 3, device=dev); qlift = torch.zeros(N, 6, device=dev)
    qret = torch.zeros(N, 6, device=dev)
    prev_d = None
    for _ in range(env.max_steps + 5):
        p, _ = rt.plug_in_port()
        d = -p[:, 2] * 1e3
        t = env.steps.float() * dt
        S['t_seat'] = torch.where(live & torch.isnan(S['t_seat']) & (env.seated_steps >= env.hold_steps_sim), t, S['t_seat'])
        v = torch.zeros_like(d) if prev_d is None else (d - prev_d).abs() / dt
        prev_d = d.clone()
        run = torch.where((d >= seat_mm - CS.DET_MARGIN_MM) & (v < CS.DET_SPEED_MM_S), run + 1, torch.zeros_like(run))
        fire = live & torch.isnan(S['t_det']) & (run >= n_det)
        S['t_det'] = torch.where(fire, t, S['t_det'])
        h = torch.where(fire, torch.zeros_like(h), h)
        meas = env._measured_model_angles()
        for key, e in (('d_rest', e_rest), ('d_open', e_open), ('d_ret', e_ret), ('d_end', e_end)):
            S[key] = torch.where(live & (h == e), d, S[key])
        fo = rt.q[:, rt.finger_qadr_right].mean(1) * 1e3
        S['finger_open_mm'] = torch.where(live & (h == e_open), fo, S['finger_open_mm'])
        S['lat_end'] = torch.where(live & (h == e_end), p[:, :2].norm(dim=1) * 1e3, S['lat_end'])
        st = live & (h == e_rest)                          # release start
        if st.any():
            rt.grip_target[st] = A.open_m
            qr0[st], ql0[st] = meas[st, 0], meas[st, 1]
            env.left_press_q[st] = meas[st, 1]
            pp, RR = env.right_chain.fk(meas[st, 0]); p0[st], R0[st] = pp, RR
            pc = rt.q[st, rt.pc_qadr:rt.pc_qadr + 7]
            _, port_R = env._port_pose(pc); z[st] = port_R[:, :, 2]
            qret[st] = meas[st, 0]
        hold = live & (h >= 0) & (h < e_rest)
        opn = live & (h >= e_rest) & (h < e_open)
        ret = live & (h >= e_open) & (h < e_ret)
        uh = live & (h >= e_ret) & (h < e_uh)
        lift = live & (h >= e_uh)
        act = pol(hist)
        busy = live & (h >= 0)
        act = torch.where(busy[:, None], torch.zeros_like(act), act)
        if hold.any():
            env.joint_target[hold, 0] = meas[hold, 0]
        if opn.any():
            env.joint_target[opn, 0] = qr0[opn]
        if ret.any():
            u = ((h[ret] - e_open + 1).float() / n_ret).clamp(0, 1)[:, None]
            u = u * u * (3 - 2 * u)
            q = env.right_chain.ik(p0[ret] + z[ret] * 0.05 * u, R0[ret], qret[ret], iters=10)[0]
            qret[ret] = q
            env.joint_target[ret, 0] = q
        if uh.any():
            env.joint_target[uh, 0] = qret[uh]
            first = uh & (h == e_ret)
            if first.any():
                ql0[first] = meas[first, 1]
                pp, RR = env.left_chain.fk(meas[first, 1]); pl0[first], Rl0[first] = pp, RR; qlift[first] = meas[first, 1]
            env.left_press_q[uh] = ql0[uh]
        if lift.any():
            env.joint_target[lift, 0] = qret[lift]
            u = ((h[lift] - e_uh + 1).float() / n_lift).clamp(0, 1)[:, None]
            tgt = pl0[lift] + torch.tensor([0., 0., 0.03], device=dev) * u
            q = env.left_chain.ik(tgt, Rl0[lift], qlift[lift], iters=10)[0]
            qlift[lift] = q
            env.left_press_q[lift] = q
        h = torch.where(h >= 0, h + 1, h)
        live = live & ~(h > e_end)
        hist, _, _, done, info = env.step(act)
        if 'final' in info:
            ids = info['final_ids']
            live[ids] = False
        if not live.any():
            break
    G = {k: v.cpu().numpy() for k, v in S.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for k, name in enumerate(names):
        sl = slice(k * n, (k + 1) * n)
        full_rest = G['d_rest'][sl] >= seat_mm - CS.FULL_MM
        done = np.isfinite(G['d_end'][sl])
        base = full_rest & done
        pc = lambda x: [round(float(y), 3) for y in np.percentile(x, [1, 5, 50])] if len(x) else None   # noqa: E731
        row = dict(tag=A.tag, policy=A.policy, plug=A.plug, open_m=A.open_m, variant=name, params=CS.VARIANTS[name], n=n, seed=A.seed,
                   sim_seat=int(np.isfinite(G['t_seat'][sl]).sum()), detected=int(np.isfinite(G['t_det'][sl]).sum()),
                   full_at_rest=int(full_rest.sum()), release_done=int(base.sum()))
        for key in ('d_open', 'd_ret', 'd_end'):
            x = G[key][sl][base]
            row[f'in_after_{key[2:]}'] = int((x >= seat_mm - CS.FULL_MM).sum())
            row[f'{key}_mm_p1_p5_p50'] = pc(x)
        row['out_frac_end'] = round(1 - row['in_after_end'] / max(row['release_done'], 1), 4)
        row['drop_open_mm_p50_p95'] = [round(float(y), 3) for y in np.percentile((G['d_rest'] - G['d_open'])[sl][base], [50, 95])] if base.any() else None
        row['drop_retreat_mm_p50_p95'] = [round(float(y), 3) for y in np.percentile((G['d_open'] - G['d_ret'])[sl][base], [50, 95])] if base.any() else None
        row['drop_unload_mm_p50_p95'] = [round(float(y), 3) for y in np.percentile((G['d_ret'] - G['d_end'])[sl][base], [50, 95])] if base.any() else None
        row['finger_mm_at_retreat_p5_p50_p95'] = [round(float(y), 2) for y in np.percentile(G['finger_open_mm'][sl][base], [5, 50, 95])] if base.any() else None
        res.append(row)
        print(json.dumps(row), flush=True)
    f = OUT / f'{A.tag}_{A.plug}{"" if A.open_m >= 1 else f"_open{A.open_m*1e3:.0f}"}.json'
    old = json.loads(f.read_text()) if f.exists() else []
    f.write_text(json.dumps(old + res, indent=1))


if __name__ == '__main__':
    main()
