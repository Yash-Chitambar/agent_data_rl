"""Contact-model sensitivity of an insertion policy (docs/real/CONTACT_SENS_20261115.md, test 1). Sim only.

One process = one scene build (pad model, plug clearance, PC placement) and one or more contact variants. Each variant
gets its own group of --n worlds in the same batch (per-world contact parameters, set after the reset), so one process
evaluates several variants at once. Every world runs its first episode only.

Start: cell 'real_ho' of k2_matrix.py (gap 5.5 mm, lateral ring 0.5 mm, angles from the chain pool), real right-arm lead,
real-like plant (joint friction x1.3-2.0, gravity error 12 %), rig-spot layout v2 (PC +/-5 mm, 4 +/- 1.5 deg) or the
step-2 box (--place box: PC +/-25 mm, 4 +/- 15 deg), no mating force, training-only DR off. As k7_depth.py, the sim
success end is off and each world runs on after the seat; per world the script records:
  * seat: the sim seat test held 0.2 s (the usual success; t_seat = its time, as t_ok of k2_matrix);
  * the deploy-like seat detector on the true depth (depth >= seat - 1 mm, axial speed < 6 mm/s for 0.3 s), then 0.2 s
    measured-pose hold (action 0): depth 'at rest'; full depth = rest depth >= seat - 0.25 mm;
  * then a release probe: right jaws open (position target past full open), right arm held at the measured pose, left
    press target = its measured pose; true depth 0.5 s later. 'seated after release' = that depth >= seat - 0.25 mm.
  * failure class: 'seated' if the seat test held, else the class of the first episode's end (failure_class).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYTHONPATH=. python scripts/eval/contact_sens.py --policy runs/insert_K4.pt --tag k4 \
        --pad 4x1.2 --plug nom --place real --variants base,pmu_lo,pmu_hi
Output: <out_dir>/<tag>_<place>_<pad>_<plug>.json (one row per variant, resumable); per-world arrays in runs/contact_sens/.
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

# Contact variants. Keys: port_mu, port_tc (solref time constant, s), port_damp (solref damping ratio), port_solimp
# (dmin, dmax, width), pad_mu, pad_tc (s), grip (squeeze scale). A missing key keeps the training value (DEFAULT_CONTACT_DR of
# insertion_task.py: port mu U(0.10, 0.20), port tc U(3, 4) ms, damping ratio 1, solimp (0.99, 0.999, 0.001); pad mu 1.2, pad
# tc 2 ms (scene), squeeze x1.0).
VARIANTS = {
    'base': {},
    'pmu_lo': dict(port_mu=(0.05, 0.08)),
    'pmu_hi': dict(port_mu=(0.35, 0.60)),
    'pmu_vhi': dict(port_mu=(0.80, 1.00)),
    'ptc_2ms': dict(port_tc=(0.002, 0.002)),
    'ptc_8ms': dict(port_tc=(0.008, 0.008)),
    'pdamp_05': dict(port_damp=0.5),
    'psolimp_soft': dict(port_solimp=(0.90, 0.95, 0.001)),
    'padmu_08': dict(pad_mu=(0.8, 0.8)),
    'padmu_20': dict(pad_mu=(2.0, 2.0)),
    'padtc_6ms': dict(pad_tc=(0.006, 0.006)),
    'grip_06': dict(grip=(0.6, 0.6)),
    'grip_14': dict(grip=(1.4, 1.4)),
}
# Variants inside the training randomization of K4 and K7 (both use DEFAULT_CONTACT_DR, pads 4x4 r 1.2 mm, nominal plug)
IN_TRAINING = {'base'}
PADS = {'4x1.2': (4, 0.0012), '2x1.2': (2, 0.0012), '6x0.8': (6, 0.0008)}     # spheres per pad side, radius (m)
# Plug outline (width, thickness, m) in the nominal receptacle (8.34 x 2.56 mm): clearance per side width / thickness
PLUGS = {'nom': None,                  # 8.25 x 2.40: 45 / 80 um (builder.USBC_PLUG)
         'tight': (0.00830, 0.00246),  # 20 / 50 um: tightest USB-C spec fit
         'loose': (0.00816, 0.00234)}  # 90 / 110 um: loosest spec fit
PLACES = {'real': [0.005, 0.005, 1.5], 'box': [0.025, 0.025, 15.0]}
RIG = {'task_layout': 'results/placement/task_layout_real_v2.json', 'mate_force_n': None, 'vision_prior_sigma_scale': None}
DET_MARGIN_MM, DET_SPEED_MM_S, DET_HOLD_S, REST_S, REL_S, FULL_MM = 1.0, 6.0, 0.3, 0.2, 0.5, 0.25
OUT = ROOT / 'results/real_align/contact_sens'
RAW = ROOT / 'runs/contact_sens'


def patch_scene(pad='4x1.2', plug='nom', batch_solimp=True):
    """Scene patches for this process only (before the env is built): pad spheres, plug outline, per-world port solimp."""
    from blind_insertion.assets import builder as B
    n, r = PADS[pad]
    if (n, r) != PADS['4x1.2']:
        orig = B.sphere_pads
        B.sphere_pads = lambda spec, info, n_=3, radius=0.0012: orig(spec, info, n, r)
    if PLUGS[plug] is not None:
        orig_rv = B.reader_variant
        B.reader_variant = lambda *a, **k: orig_rv(*a, **dict(k, plug=PLUGS[plug]))
    if batch_solimp:
        import mujoco_warp as mjw
        orig_pm = mjw.put_model

        def put_model(m, batch_sizes=None, **kw):
            bs = dict(batch_sizes or {})
            if 'geom_solref' in bs:
                bs['geom_solimp'] = bs['geom_solref']
            return orig_pm(m, batch_sizes=bs or None, **kw)
        mjw.put_model = put_model


def build(policy, nworld, seed, place='real', extra=None):
    from scripts.eval.k3_robust import build as k3_build
    ex = dict(RIG, pc_pose_range=PLACES[place])
    ex.update(extra or {})
    env, pol = k3_build(policy, nworld, seed, ex)   # real-like plant, training-only DR off
    env.hold_steps_sim = env.hold_steps
    return env, pol


def apply_variant(env, ids, v, gen):
    """Per-world contact parameters for worlds `ids` (after the reset sampled the training values)."""
    rt = env.runtime
    n = len(ids)
    u = lambda lo, hi: lo + (hi - lo) * torch.rand(n, 1, device=env.device, generator=gen)   # noqa: E731
    P, D = rt.port_geoms, rt.pad_geoms
    if 'port_mu' in v:
        rt.geom_friction[ids[:, None], P, 0] = u(*v['port_mu']).expand(-1, len(P))
    if 'port_tc' in v:
        rt.geom_solref[ids[:, None], P, 0] = u(*v['port_tc']).expand(-1, len(P))
    if 'port_damp' in v:
        rt.geom_solref[ids[:, None], P, 1] = float(v['port_damp'])
    if 'port_solimp' in v:
        import warp as wp
        si = wp.to_torch(rt.m.geom_solimp)
        assert si.shape[0] == env.N, 'geom_solimp is not batched per world (patch_scene)'
        si[ids[:, None], P, :3] = torch.tensor(v['port_solimp'], device=env.device, dtype=si.dtype)
    if 'pad_mu' in v:
        rt.geom_friction[ids[:, None], D, 0] = u(*v['pad_mu']).expand(-1, len(D))
    if 'pad_tc' in v:
        rt.geom_solref[ids[:, None], D, 0] = u(*v['pad_tc']).expand(-1, len(D))
    if 'grip' in v:
        rt.actuator_gainprm[ids[:, None], rt.grip_acts, 0] = u(*v['grip']) * rt._grip_gain0


def ring_rows(base, g, n, ring, gap=5.5, real_prior=False):
    """Start rows (x, y, gap mm; rx, ry, roll deg) on a lateral ring; angles from the chain pool. real_prior: the rows are
    the vision measurement m on the ring and the true start is m + N(0, VISION_SIGMA) (replay_real_handovers.py); returns
    (rows = truth, m)."""
    from scripts.eval.replay_real_handovers import VISION_SIGMA
    r = base[g.integers(0, len(base), n)].copy()
    th = g.uniform(0, 2 * math.pi, n)
    m = np.stack([ring * np.cos(th), ring * np.sin(th), np.full(n, gap)], 1)
    r[:, 0:3] = m + (g.normal(0, 1, (n, 3)) * VISION_SIGMA if real_prior else 0)
    r[:, 2] = np.clip(r[:, 2], 1.0, 20.0)
    return r, m


def fixed_pool(env, rows):
    """Start pool entry j from row j (sample_starts draws rows at random: torch.randint is replaced by arange for this call).
    Returns a bool per world: True if its first start is the intended row (IK retries redraw from the start of the rows)."""
    orig = torch.randint
    nrow = len(rows)

    def ri(high, size, **kw):
        if high == nrow:
            return torch.arange(size[0], device=kw.get('device')) % high
        return orig(high, size, **kw)
    env._npz = torch.tensor(rows, device=env.device, dtype=torch.float32)
    torch.randint = ri
    try:
        env.refresh_pool()
    finally:
        torch.randint = orig
    P = env.icfg.pool_per_world
    first = env.pool['err_p'][torch.arange(env.N, device=env.device) * P] * 1e3
    want = env._npz[torch.arange(env.N, device=env.device) * P, :3]
    return ((first - want).abs().max(1).values < 1e-3).cpu().numpy()


def run_probe(env, pol, groups, rows, seed, prior=None):
    """Run every world's first episode with the seat / depth / release probe.
    groups: list of (name, variant dict); world w belongs to group w // n. rows: N * pool_per_world start rows in world order.
    prior: None (the env's prior: truth + N(0, sigma)) or (N, 3) mm per world (the vision measurement fed to the policy; NaN rows
    keep the env's prior)."""
    from scripts.eval.kj_real4 import REAL_LEAD
    from blind_insertion.envs.insertion_task import failure_class
    N, dev, dt = env.N, env.device, env.dt
    n = N // len(groups)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    ok_row = fixed_pool(env, rows)
    hold0 = env.hold_steps_sim
    env.hold_steps = 10 ** 9           # no success end: each world runs on after the seat
    rt = env.runtime
    rt.grip_target = torch.full((N,), float('nan'), device=dev)
    env.reset_all()
    for k, (_, v) in enumerate(groups):
        apply_variant(env, torch.arange(k * n, (k + 1) * n, device=dev), v, gen)
    q0 = env._measured_model_angles()[:, 0]
    env.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=dev)
    if prior is not None:              # rows with NaN keep the env's prior
        pr = torch.as_tensor(prior, dtype=torch.float32, device=dev)
        m = torch.isfinite(pr).all(1)
        env.vision_prior[m] = pr[m]
    env.hist[:] = env._actor_obs()[:, None]
    hist, _ = env.observe()
    seat_mm = env.seat_depth * 1e3
    n_det, n_rest, n_rel = (int(round(x / dt)) for x in (DET_HOLD_S, REST_S, REL_S))
    nan = lambda: torch.full((N,), float('nan'), device=dev)   # noqa: E731
    S = dict(t_seat=nan(), t_det=nan(), d_det=nan(), d_rest=nan(), d_rel=nan(), d_max=torch.zeros(N, device=dev),
             rel_full=torch.zeros(N, dtype=torch.bool, device=dev),
             run=torch.zeros(N, dtype=torch.long, device=dev), hold=torch.full((N,), -1, dtype=torch.long, device=dev))
    live = torch.ones(N, dtype=torch.bool, device=dev)
    cls = np.array([''] * N, dtype=object)
    prev_d = None
    for _ in range(env.max_steps + 5):
        p, _ = rt.plug_in_port()
        d = -p[:, 2] * 1e3
        t = env.steps.float() * dt
        S['d_max'] = torch.where(live, torch.maximum(S['d_max'], d), S['d_max'])
        new_seat = live & torch.isnan(S['t_seat']) & (env.seated_steps >= hold0)
        S['t_seat'] = torch.where(new_seat, t, S['t_seat'])
        v = torch.zeros_like(d) if prev_d is None else (d - prev_d).abs() / dt
        prev_d = d.clone()
        okd = (d >= seat_mm - DET_MARGIN_MM) & (v < DET_SPEED_MM_S)
        S['run'] = torch.where(okd, S['run'] + 1, torch.zeros_like(S['run']))
        fire = live & torch.isnan(S['t_det']) & (S['run'] >= n_det)
        S['t_det'] = torch.where(fire, t, S['t_det'])
        S['d_det'] = torch.where(fire, d, S['d_det'])
        S['hold'] = torch.where(fire, torch.zeros_like(S['hold']), S['hold'])
        h = S['hold']
        rest_end = live & (h == n_rest)
        S['d_rest'] = torch.where(rest_end, d, S['d_rest'])
        in_rel = live & (h > n_rest) & (h <= n_rest + n_rel)     # last live depth of the release window (0.5 s after the open)
        S['d_rel'] = torch.where(in_rel, d, S['d_rel'])
        S['rel_full'] = S['rel_full'] | (live & (h == n_rest + n_rel))
        meas = env._measured_model_angles()
        if rest_end.any():             # release start: jaws open, left press target = measured
            rt.grip_target[rest_end] = 1.0
            env.left_press_q[rest_end] = meas[rest_end, 1]
        holding = live & (h >= 0)
        act = pol(hist)
        if holding.any():
            env.joint_target[holding, 0] = meas[holding, 0]
            act = torch.where(holding[:, None], torch.zeros_like(act), act)
        S['hold'] = torch.where(h >= 0, h + 1, h)
        hist, _, _, done, info = env.step(act)
        if 'final' in info:
            ids = info['final_ids']
            fc = failure_class(info['final'])
            lv = live[ids].cpu().numpy()
            for j, w in enumerate(ids.tolist()):
                if lv[j]:
                    cls[w] = fc[j]
            live[ids] = False
            rt.grip_target[ids] = float('nan')
        if not live.any():
            break
    g = {k: S[k].cpu().numpy() for k in ('t_seat', 't_det', 'd_det', 'd_rest', 'd_rel', 'd_max', 'rel_full')}
    seated = np.isfinite(g['t_seat'])
    full = g['d_rest'] >= seat_mm - FULL_MM
    # not seated by the sim test but fully in at rest: the aperture-polygon test failed (penetration), not the insertion;
    # the end class of such a world comes after the release probe and would mislead
    cls = np.where(seated, 'seated', np.where(full, 'full_no_seat_test', cls))
    return g, cls, ok_row


def pct(x, q):
    x = x[np.isfinite(x)]
    return [round(float(v), 3) for v in np.percentile(x, q)] if len(x) else None


def summarize(g, cls, ok, seat_mm):
    """Summary of one group of worlds (boolean mask ok selects the valid worlds)."""
    from blind_insertion.eval_insertion import wilson
    n = int(ok.sum())
    seated = np.isfinite(g['t_seat'][ok])
    full = (g['d_rest'][ok] >= seat_mm - FULL_MM)
    rel_done = np.isfinite(g['d_rel'][ok])
    rel_in = g['d_rel'][ok] >= seat_mm - FULL_MM
    k = int(seated.sum())
    c = {}
    for x in cls[ok]:
        c[x] = c.get(x, 0) + 1
    return dict(n=n, k=k, rate=k / max(n, 1), wilson=list(wilson(k, n)),
                full_depth_rate=float(full.mean()), full_k=int(full.sum()),
                t_seat_p50_p90_p99=pct(g['t_seat'][ok], [50, 90, 99]),
                rest_depth_mm_p5_p50=pct(g['d_rest'][ok], [5, 50]),
                release_n=int(rel_done.sum()), release_in_k=int(rel_in.sum()), release_ended_early=int((rel_done & ~g['rel_full'][ok]).sum()),
                release_in_frac=float(rel_in.sum() / max(rel_done.sum(), 1)),
                release_depth_mm_p1_p5_p50=pct(g['d_rel'][ok], [1, 5, 50]),
                classes=c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--pad', default='4x1.2', choices=sorted(PADS))
    ap.add_argument('--plug', default='nom', choices=sorted(PLUGS))
    ap.add_argument('--place', default='real', choices=sorted(PLACES))
    ap.add_argument('--variants', default='base')
    ap.add_argument('--n', type=int, default=1024, help='worlds per variant')
    ap.add_argument('--seed', type=int, default=8100)
    ap.add_argument('--out_dir', default=str(OUT / 'raw'))
    A = ap.parse_args()
    names = A.variants.split(',')
    out = Path(A.out_dir) / f'{A.tag}_{A.place}_{A.pad}_{A.plug}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    names = [x for x in names if x not in {d['variant'] for d in done}]
    if not names:
        return
    patch_scene(A.pad, A.plug)
    torch.manual_seed(A.seed)
    N = A.n * len(names)
    env, pol = build(A.policy, N, A.seed, A.place)
    from scripts.eval.kj_start_map import rows_base
    g = np.random.default_rng(A.seed)
    rows, _ = ring_rows(rows_base(), g, N * env.icfg.pool_per_world, 0.5)
    t0 = time.time()
    res, cls, ok_row = run_probe(env, pol, [(x, VARIANTS[x]) for x in names], rows, A.seed)
    seat_mm = env.seat_depth * 1e3
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW / f'{A.tag}_{A.place}_{A.pad}_{A.plug}_{"-".join(names)}.npz', cls=cls.astype(str), ok_row=ok_row, **res)
    for k, name in enumerate(names):
        sel = np.zeros(N, bool); sel[k * A.n:(k + 1) * A.n] = True
        s = summarize(res, cls, sel, seat_mm)
        s.update(variant=name, params=VARIANTS[name], tag=A.tag, policy=A.policy, place=A.place, pad=A.pad, plug=A.plug, seed=A.seed,
                 in_training=(name in IN_TRAINING and A.pad == '4x1.2' and A.plug == 'nom'), start_row_mismatch=int((~ok_row[sel]).sum()),
                 wall_s=round(time.time() - t0), batch_worlds=N)
        done.append(s)
        print(name, f"{s['k']}/{s['n']} = {100 * s['rate']:.1f} % full {100 * s['full_depth_rate']:.1f} % t {s['t_seat_p50_p90_p99']} "
              f"rel_in {s['release_in_k']}/{s['release_n']}", {a: b for a, b in s['classes'].items() if a != 'seated'}, flush=True)
    out.write_text(json.dumps(done, indent=1))


if __name__ == '__main__':
    main()
