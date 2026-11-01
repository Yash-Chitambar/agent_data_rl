"""Reachability of the Part 1 tipped side grasp over the reader region (GPU IK, no physics).

Seeds: 6 fixed + 24 random joint configurations; a seed counts when pre-grasp, grasp and lift all solve.
For every feasible (cell, yaw bin) of the task layout (`placement/report.json`, the region the
runtime samples) at its centre and 4 jittered points: pre-grasp, grasp and lift IK (1 mm, ~0.6 deg)
with every joint >= 0.05 rad inside its limits, for each tilt (50-65 deg) and hand roll (0: the
confirmed frame; 180: the same grasp with the hand turned about its tool axis, camera on the other
side). Writes results/usbc/policy/pick/reach_map.{json,png}.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/policy/pick_reach_map.py
"""
import json, math
from pathlib import Path
import mujoco, numpy as np, torch
from blind_insertion.envs import kinematics as K
from blind_insertion.scenes import pc_bench as S, physical_rig as R, reader_grasp as G

dev = 'cuda:0'
N_RANDOM_SEEDS = 24
prof = json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m, l = S.build_scene(detailed=False, bimanual=True, support_mode='free', profile=prof, measured_support=True)
d = mujoco.MjData(m); R.reference_pose(m, d, l); mujoco.mj_forward(m, d)
hand = G.hand_geometry(m, d)
qadr = [int(m.joint(f'joint{i}').qposadr[0]) for i in range(1, 7)]
chain = K.Chain(m, l.tcp_site, qadr, device=dev)
rep = json.loads(Path('results/usbc/bimanual_20261029/placement/report.json').read_text())['reader_region']
smp = np.array(rep['samples_xy_yaw']); h, hb = rep['cell_m'] / 2, rep['yaw_bin_deg'] / 2
rng = np.random.default_rng(0)
pts = [smp] + [smp + rng.uniform(-1, 1, smp.shape) * [h, h, hb] for _ in range(4)]
P = np.concatenate(pts); Nn = len(P)
# reader pose for (x, y, yaw) as place_reader does
S.place_reader(m, d, l, xy=(0., 0.), yaw_deg=0.); q0 = d.qpos[l.peg_qadr:l.peg_qadr + 7].copy()
R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, q0[3:]); R0 = torch.tensor(R0.reshape(3, 3), dtype=torch.float32, device=dev)
yaw = torch.tensor(np.radians(P[:, 2]), dtype=torch.float32, device=dev)
c, s = torch.cos(yaw), torch.sin(yaw); z, o = torch.zeros_like(c), torch.ones_like(c)
Rz = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
p0 = torch.tensor(q0[:3], dtype=torch.float32, device=dev)
pr = torch.tensor(np.c_[P[:, :2], np.zeros(Nn)], dtype=torch.float32, device=dev) + (Rz @ torch.tensor([p0[0], p0[1], 0.], device=dev)[:, None])[..., 0]
pr[:, 2] = p0[2]
Rr = Rz @ R0
lo, hi = chain.lo + .05, chain.hi - .05
flip = torch.diag(torch.tensor([-1., -1., 1.], device=dev))
res = {}
for roll in (0, 180):
    for tilt in (60., 55., 65., 50.):
        off, rngh = G.grasp_offset(l, hand, tilt)
        W = G.grasp_poses(pr, Rr, off, tilt)
        ok_all = torch.ones(Nn, dtype=torch.bool, device=dev); qs = None
        best = None
        aim = torch.atan2(pr[:, 1], pr[:, 0])
        for name in ('pregrasp', 'grasp', 'lift'):
            p_t, R_t = W[name]
            if roll: R_t = R_t @ flip
            if qs is None:
                gen = torch.Generator(device=dev); gen.manual_seed(1)
                fixed = [torch.cat([aim[:, None], torch.tensor(sd, device=dev).expand(Nn, 5)], 1) for sd in
                         ([1.0, 1.0, -.5, 0, 0], [1.6, 1.3, -.9, 0, 0], [2.2, 1.8, -1.2, 0, 0], [.6, .5, -.3, 0, 0], [1.6, 1.3, -.9, 0, 1.5], [1.6, 1.3, -.9, 0, -1.5])]
                rand = [chain.lo + (chain.hi - chain.lo) * torch.rand(Nn, 6, device=dev, generator=gen) for _ in range(N_RANDOM_SEEDS)]
                for seed in fixed + rand:
                    seed = seed.clamp(chain.lo + .05, chain.hi - .05)
                    q, pe, re = chain.ik(p_t, R_t, seed, iters=100)
                    good = (pe < 1e-3) & (re < 1e-2) & ((q >= lo) & (q <= hi)).all(1)
                    # the grasp and lift must also solve from this pre-grasp solution
                    qg, pg, rg = chain.ik(W['grasp'][0], W['grasp'][1] @ flip if roll else W['grasp'][1], q, iters=60)
                    ql, pl, rl = chain.ik(W['lift'][0], W['lift'][1] @ flip if roll else W['lift'][1], qg, iters=60)
                    good &= (pg < 1e-3) & (rg < 1e-2) & ((qg >= lo) & (qg <= hi)).all(1) & (pl < 1e-3) & (rl < 1e-2) & ((ql >= lo) & (ql <= hi)).all(1)
                    if best is None: best, bg = q, good
                    else:
                        take = good & ~bg; best = torch.where(take[:, None], q, best); bg = bg | good
                qs, g = best, bg
            else:
                q, pe, re = chain.ik(p_t, R_t, qs, iters=100)
                g = (pe < 1e-3) & (re < 1e-2) & ((q >= lo) & (q <= hi)).all(1); qs = q
            ok_all &= g
        res[(roll, tilt)] = ok_all.cpu().numpy()
        print(f'roll {roll} tilt {tilt}: reachable {ok_all.float().mean().item():.3f}', flush=True)
r60 = res[(0, 60.)]
any0 = np.any([res[(0, t)] for t in (60., 55., 65., 50.)], 0)
anyall = np.any(list(res.values()), 0)
any60 = res[(0, 60.)] | res[(180, 60.)]
summary = {'n_points': int(Nn), 'per_option': {f'roll{r}_tilt{int(t)}': float(v.mean()) for (r, t), v in res.items()},
           'tilt60_roll0': float(r60.mean()), 'tilt60_any_roll': float(any60.mean()), 'tilt50_65_roll0': float(any0.mean()), 'tilt50_65_any_roll': float(anyall.mean())}
yb = np.round(P[:, 2] / 20) * 20
summary['by_yaw_bin'] = {str(int(y)): {'tilt60_roll0': float(r60[yb == y].mean()), 'tilt60_any_roll': float(any60[yb == y].mean()), 'any': float(anyall[yb == y].mean()), 'n': int((yb == y).sum())} for y in np.unique(yb)}
print(json.dumps(summary, indent=1))
out = Path('results/usbc/policy/pick'); out.mkdir(parents=True, exist_ok=True)
(out / 'reach_map.json').write_text(json.dumps(summary, indent=1) + '\n')
np.savez_compressed(out / 'reach_map.npz', points=P, **{f'roll{r}_tilt{int(t)}': v for (r, t), v in res.items()})
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a_, (v, ttl) in zip(ax, ((r60, 'tilt 60, roll 0 (confirmed frame)'), (any60, 'tilt 60, roll 0 or 180'), (anyall, 'tilt 50-65, roll 0 or 180'))):
    a_.scatter(P[:, 1] * 100, P[:, 0] * 100, c=np.where(v, 'tab:green', 'tab:red'), s=3)
    a_.set(title=f'{ttl}: {v.mean():.1%} (n={Nn})', xlabel='y (cm)', ylabel='x (cm)'); a_.invert_xaxis(); a_.set_aspect('equal')
fig.tight_layout(); fig.savefig(out / 'reach_map.png', dpi=110)
