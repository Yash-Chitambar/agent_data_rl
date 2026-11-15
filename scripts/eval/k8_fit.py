"""Per-world plug fit and friction schedule for the K8 fine-tune (docs/real/K8_20261115.md). Sim only, off by default.

Nothing here changes library code. `install(...)` monkeypatches this process only, before the env is built:
  * fit: the scene gets one reader variant per plug outline of FIT_GRID (variant 0 = the nominal 45 / 80 um plug, so every
    `[0]` lookup in the env stays nominal). Receptacle = nominal 8.34 x 2.56 mm in every variant (fixture slots stay variant 0;
    the script checks they are equal). Each world gets one plug (peg slots: geom_dataid/pos/quat/size/aabb/rbound batched per
    world), drawn once at build. The seat test and the contact labels use that world's own plug-edge points.
  * mu_sched: port friction upper bound grows linearly with env steps (curriculum), lower bound fixed.
  * live_redraw: worlds restored from saved chain states (live_states) get new contact values (the saved ones are the
    training values of K4/K7).

    from scripts.eval.k8_fit import install; install(fit='spec_train')  # before InsertionTask(...)
"""
from __future__ import annotations

import inspect
import textwrap

import numpy as np
import torch

USBC_REC = (0.00834, 0.00256)
# (width, thickness) clearance per side, m. USB-C spec fit: 20-90 um across the width, 50-110 um across the thickness.
# spec_full: the whole spec box (fit check). spec_train: thickness clearance 50-80 um only: at 110 um the sim seat test fails on
# fully inserted plugs (aperture penetration; fitcheck_k7.json: K7 seat 43.9 % at 90/110 um with 90.2 % full depth), so those
# worlds would give a false failure signal in training.
_grid = lambda cs: [(45e-6, 80e-6)] + [(cx, c) for cx in (20e-6, 45e-6, 67.5e-6, 90e-6) for c in cs if (cx, c) != (45e-6, 80e-6)]  # noqa: E731
FIT_GRID = {'spec_full': _grid((50e-6, 80e-6, 110e-6)), 'spec_train': _grid((50e-6, 65e-6, 80e-6)),
            'tight': [(20e-6, 50e-6)], 'loose': [(90e-6, 110e-6)], 'nom': [(45e-6, 80e-6)]}
STATE = {}


def plug_of(cx, c):
    return (USBC_REC[0] - 2 * cx, USBC_REC[1] - 2 * c)


def _rewrite(cls, name, old, new, count):
    src = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    assert src.count(old) == count, f'{cls.__name__}.{name}: expected {count} x {old!r}'
    ns = {}
    mod = inspect.getmodule(cls)
    exec(compile(src.replace(old, new), f'<k8_fit {name}>', 'exec'), mod.__dict__, ns)
    setattr(cls, name, ns[name])


def install(fit='nom', mu_sched=None, live_redraw=False, fit_seed=0):
    """fit: key of FIT_GRID ('nom' = no change). mu_sched: (lo, hi0, hi1, steps): port mu U(lo, hi(t)), hi from hi0 to hi1
    over `steps` env steps (all worlds). live_redraw: redraw contact values after a live-state restore."""
    from blind_insertion.assets import builder as B
    from blind_insertion.envs import bimanual_runtime as BR
    from blind_insertion.envs import insertion_task as IT
    STATE.update(fit=fit, mu_sched=mu_sched, steps=0)
    grid = FIT_GRID[fit]
    if fit != 'nom':
        orig_rv, orig_build = B.reader_variant, B.build
        last = {}

        def reader_variant(*a, **k):
            last.clear(); last.update(a=a, k=k)
            return orig_rv(*a, **k)

        def build(robot, variants, **kw):
            if len(variants) == 1 and variants[0].kind == 'reader' and last:
                k = dict(last['k']); assert not last['a'], 'reader_variant with positional args'
                variants = [orig_rv(**dict(k, plug=plug_of(cx, c))) for cx, c in grid]
            return orig_build(robot, variants, **kw)
        B.reader_variant, B.build = reader_variant, build

        import mujoco_warp as mjw
        orig_pm = mjw.put_model
        FIELDS = ('geom_dataid', 'geom_pos', 'geom_quat', 'geom_size', 'geom_aabb', 'geom_rbound')

        def put_model(m, batch_sizes=None, **kw):
            bs = dict(batch_sizes or {})
            assert bs, 'k8_fit needs a batched model (contact_dr) to know nworld'
            n = max(bs.values())
            for f in FIELDS:
                bs[f] = n
            return orig_pm(m, batch_sizes=bs, **kw)
        mjw.put_model = put_model

        orig_init = BR.BimanualRuntime.__init__

        def init(self, cfg=None):
            orig_init(self, cfg)
            _assign_fit(self, fit_seed)
        BR.BimanualRuntime.__init__ = init

        def insertion_metrics(self, seat_tolerance_m=.00025, tilt_limit_deg=2., aperture_tol_m=0.):
            p, rotation = self.plug_in_port()
            xy = self.fit_edge.to(self.q.dtype)                                        # (N, 64, 2) per world
            points = torch.cat((xy, torch.zeros_like(xy[..., :1])), 2) @ rotation.transpose(1, 2) + p[:, None]
            polygon = torch.as_tensor(self.lay.hole_polys[0], device=self.device, dtype=self.q.dtype)
            edge = torch.roll(polygon, -1, 0) - polygon
            delta = points[:, :, :2, None].transpose(2, 3) - polygon[None, None]
            cross = edge[None, None, :, 0] * delta[:, :, :, 1] - edge[None, None, :, 1] * delta[:, :, :, 0]
            if aperture_tol_m:
                inside = (cross / edge.norm(dim=1)[None, None] >= -aperture_tol_m).all(dim=2).all(dim=1)
            else:
                inside = (cross >= -1e-9).all(dim=2).all(dim=1)
            depth = float((self.lay.seating_depth if self.lay.seating_depth is not None else self.lay.depth)[0])
            axial = points[:, :, 2]
            tilt = torch.rad2deg(torch.acos(rotation[:, 2, 2].clamp(-1, 1)))
            seated = inside & (axial <= -depth + seat_tolerance_m).all(1) & (axial >= -depth - seat_tolerance_m).all(1)
            finite = torch.isfinite(self.q).all(1)
            return {'geometric_seated': seated & (tilt <= tilt_limit_deg) & finite,
                    'aperture_fit': inside, 'tip_depth_m': -p[:, 2], 'tilt_deg': tilt, 'finite': finite}
        BR.BimanualRuntime.insertion_metrics = insertion_metrics

        # InsertionTask: self.edge becomes (N, 64, 3); the two batched transforms drop their [None]
        if not getattr(IT.InsertionTask, '_k8_edge', False):
            _rewrite(IT.InsertionTask, '_transition', 'self.edge[None] @ R.transpose(1, 2)', 'self.edge @ R.transpose(1, 2)', 1)
            _rewrite(IT.InsertionTask, 'current_labels', 'self.edge[None] @ R.transpose(1, 2)', 'self.edge @ R.transpose(1, 2)', 1)
            orig_ti = IT.InsertionTask.__init__

            def task_init(self, *a, **k):
                orig_ti(self, *a, **k)
                xy = self.runtime.fit_edge.float()
                self.edge = torch.cat([xy, torch.zeros_like(xy[..., :1])], 2)
            IT.InsertionTask.__init__ = task_init
            IT.InsertionTask._k8_edge = True

    if mu_sched is not None:
        orig_sc = BR.BimanualRuntime._sample_contact

        def sample_contact(self, ids):
            lo, h0, h1, T = mu_sched
            hi = h0 + (h1 - h0) * min(1.0, STATE['steps'] / T)
            self.cfg.contact_dr['port_friction'] = (lo, hi)
            STATE['mu_hi'] = hi
            return orig_sc(self, ids)
        BR.BimanualRuntime._sample_contact = sample_contact
        orig_step = IT.InsertionTask.step

        def step(self, action):
            STATE['steps'] += self.N
            return orig_step(self, action)
        IT.InsertionTask.step = step

    if live_redraw:
        orig_rl = IT.InsertionTask._restore_live

        def restore_live(self, ids):
            orig_rl(self, ids)
            if self.runtime.cfg.contact_dr:
                self.runtime._sample_contact(ids)
        IT.InsertionTask._restore_live = restore_live


def _assign_fit(rt, seed):
    """One plug variant per world (uniform over the grid), written into the batched peg-slot fields."""
    import warp as wp
    lay, N = rt.lay, rt.cfg.nworld
    K = len(lay.variants)
    for pre in ('dataid', 'pos', 'quat', 'size', 'rbound'):   # receptacle is the same in every variant
        a = getattr(lay, f'fix_{pre}')
        if pre != 'dataid':
            assert np.allclose(a, a[:1], atol=1e-9), f'fixture {pre} differs between fit variants'
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    v = torch.randint(K, (N,), generator=g).numpy()
    rt.fit_id = torch.as_tensor(v, device=rt.device)
    ps = torch.as_tensor(lay.peg_slots, device=rt.device)
    tt = lambda a: torch.as_tensor(np.asarray(a), device=rt.device)   # noqa: E731
    for f, src in (('geom_dataid', lay.peg_dataid), ('geom_pos', lay.peg_pos), ('geom_quat', lay.peg_quat),
                   ('geom_size', lay.peg_size), ('geom_rbound', lay.peg_rbound)):
        t = wp.to_torch(getattr(rt.m, f))
        t[:, ps] = tt(src[v]).to(t.dtype)
    t = wp.to_torch(rt.m.geom_aabb)
    t[:, ps] = tt(lay.peg_aabb[v]).to(t.dtype).reshape(N, len(ps), *t.shape[2:])
    rt.fit_edge = tt(lay.edge_points[v]).float()                     # (N, 64, 2)
    cl = np.array(FIT_GRID[STATE['fit']])
    rt.fit_clear_um = torch.as_tensor(cl[v] * 1e6, device=rt.device)  # (N, 2) width / thickness clearance per side
    counts = np.bincount(v, minlength=K)
    print(f'[k8_fit] {K} plug variants, worlds per variant {counts.min()}-{counts.max()}', flush=True)


def fitcheck(policy, n, seed, out):
    """Check of the per-world fit: contact_sens probe (base contact, rig spot) with one plug per world; seat rate per plug.
    Compare with contact_sens.py whole-scene plugs (tight / nom / loose) of the same policy."""
    import json
    from pathlib import Path
    from scripts.eval import contact_sens as CS
    from scripts.eval.kj_start_map import rows_base
    install(fit='spec_full', fit_seed=seed)
    torch.manual_seed(seed)
    env, pol = CS.build(policy, n, seed, 'real')
    g = np.random.default_rng(seed)
    rows, _ = CS.ring_rows(rows_base(), g, n * env.icfg.pool_per_world, 0.5)
    res, cls, ok = CS.run_probe(env, pol, [('base', {})], rows, seed)
    fid = env.runtime.fit_id.cpu().numpy()
    seat_mm = env.seat_depth * 1e3
    rows_out = []
    for k, (cx, c) in enumerate(FIT_GRID['spec_full']):
        s = CS.summarize(res, cls, fid == k, seat_mm)
        s.update(fit_um=[cx * 1e6, c * 1e6], policy=policy, seed=seed)
        rows_out.append(s)
        print(f'fit {cx*1e6:.1f}/{c*1e6:.0f} um: seat {s["k"]}/{s["n"]} = {100*s["rate"]:.1f} % full {100*s["full_depth_rate"]:.1f} %',
              {a: b for a, b in s['classes'].items() if a != 'seated'}, flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rows_out, indent=1))


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--n', type=int, default=3072)
    ap.add_argument('--seed', type=int, default=8600)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    fitcheck(A.policy, A.n, A.seed, A.out)
