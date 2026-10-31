"""New reader start region for Part 1 (user decision 2026-10-31): a compact window between the right
arm base and the PC where the confirmed 60 deg side grasp reaches with margin.

Candidates: 1 cm cells, x 0.06-0.34 m, y -0.06-0.32 m, yaw bins of 10 deg over +/-YAW (yaw 0 =
plug along +x, towards the port side). A (cell, bin) is feasible when, at all 4 cell corners x both
bin edges (8 reader poses):
  * pre-grasp, grasp and lift IK (tilt 60 deg, roll 0) solve to 1 mm / 0.6 deg with every joint
    >= MARGIN rad inside its limits (6 fixed + 24 random seeds);
  * the reader footprint (body + neck + plug, +1.5 cm) is outside the hull of every PC pose of the
    training box (port (0.22, 0.36) +/-3 cm, +/-10 deg), as in explore_pc_placement.
The region is the cell-aligned SIDE x SIDE square with every (cell, bin) feasible that has the
largest feasible margin around it. Written in the task_layout `reader_region` format to
results/usbc/policy/pick/reader_region.json (+ png).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/policy/pick_region.py
"""
import argparse, json
from pathlib import Path
import cv2, mujoco, numpy as np, torch
from blind_insertion.envs import kinematics as K
from blind_insertion.scenes import pc_bench as S, physical_rig as R, reader_grasp as G
from blind_insertion.scenes.pc_rear import layout as rear_layout

ap = argparse.ArgumentParser()
ap.add_argument('--yaw', type=float, default=30.)
ap.add_argument('--side', type=float, default=.12)
ap.add_argument('--margin', type=float, default=.10)
ap.add_argument('--tilt', type=float, default=60.)
a = ap.parse_args()
dev = 'cuda:0'
CELL, BIN = .01, 10.
READER_BACK, READER_FRONT, READER_HALF_W, READER_MARGIN = .02, .07, .018, .015
PORT, PC_BOX = (.22, .36, 0.), (.03, .03, 10.)
prof = json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m, l = S.build_scene(detailed=False, bimanual=True, support_mode='free', profile=prof, measured_support=True)
d = mujoco.MjData(m); R.reference_pose(m, d, l); mujoco.mj_forward(m, d)
hand = G.hand_geometry(m, d)
off, pad_h = G.grasp_offset(l, hand, a.tilt)
chain = K.Chain(m, l.tcp_site, [int(m.joint(f'joint{i}').qposadr[0]) for i in range(1, 7)], device=dev)
xs = np.round(np.arange(.06, .34 + 1e-9, CELL), 3); ys = np.round(np.arange(-.06, .32 + 1e-9, CELL), 3)
yb = np.arange(-a.yaw + BIN / 2, a.yaw - BIN / 2 + 1e-9, BIN)
X, Y, T = np.meshgrid(xs, ys, yb, indexing='ij')
cells = np.stack([X.ravel(), Y.ravel(), T.ravel()], 1)
corner = np.array([[sx * CELL / 2, sy * CELL / 2, st * BIN / 2] for sx in (-1, 1) for sy in (-1, 1) for st in (-1, 1)])
P = (cells[:, None] + corner[None]).reshape(-1, 3); n = len(P)
# reader poses (place_reader: xy = body centre)
S.place_reader(m, d, l, xy=(0., 0.), yaw_deg=0.); q0 = d.qpos[l.peg_qadr:l.peg_qadr + 7].copy()
R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, q0[3:]); R0 = torch.tensor(R0.reshape(3, 3), dtype=torch.float32, device=dev)
yaw = torch.tensor(np.radians(P[:, 2]), dtype=torch.float32, device=dev)
c, s = torch.cos(yaw), torch.sin(yaw); z, o = torch.zeros_like(c), torch.ones_like(c)
Rz = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
pr = torch.tensor(np.c_[P[:, :2], np.zeros(n)], dtype=torch.float32, device=dev) + (Rz @ torch.tensor([q0[0], q0[1], 0.], dtype=torch.float32, device=dev)[:, None])[..., 0]
pr[:, 2] = float(q0[2]); Rr = Rz @ R0
W = G.grasp_poses(pr, Rr, off, a.tilt)
lo, hi = chain.lo + a.margin, chain.hi - a.margin
aim = torch.atan2(pr[:, 1], pr[:, 0])
gen = torch.Generator(device=dev); gen.manual_seed(1)
seeds = [torch.cat([aim[:, None], torch.tensor(sd, device=dev).expand(n, 5)], 1) for sd in
         ([1.0, 1.0, -.5, 0, 0], [1.6, 1.3, -.9, 0, 0], [2.2, 1.8, -1.2, 0, 0], [.6, .5, -.3, 0, 0], [1.6, 1.3, -.9, 0, 1.5], [1.6, 1.3, -.9, 0, -1.5])]
seeds += [chain.lo + (chain.hi - chain.lo) * torch.rand(n, 6, device=dev, generator=gen) for _ in range(24)]
reach = torch.zeros(n, dtype=torch.bool, device=dev)
inside = lambda q: ((q >= lo) & (q <= hi)).all(1)   # noqa: E731
best_margin = torch.full((n,), -1., device=dev)
for sd in seeds:
    q1, p1, r1 = chain.ik(*W['pregrasp'], sd.clamp(chain.lo + .05, chain.hi - .05), iters=100)
    q2, p2, r2 = chain.ik(*W['grasp'], q1, iters=60)
    q3, p3, r3 = chain.ik(*W['lift'], q2, iters=60)
    good = (p1 < 1e-3) & (r1 < 1e-2) & (p2 < 1e-3) & (r2 < 1e-2) & (p3 < 1e-3) & (r3 < 1e-2) & inside(q1) & inside(q2) & inside(q3)
    marg = torch.stack([torch.minimum(q - chain.lo, chain.hi - q).min(1).values for q in (q1, q2, q3)], 1).min(1).values
    best_margin = torch.where(good, torch.maximum(best_margin, marg), best_margin)
    reach |= good
reach = reach.cpu().numpy().reshape(len(cells), 8).all(1)
# PC clearance
pc = prof['pc']['measurements']; dep, wid = pc['depth']['value'], pc['width']['value']; tfl = rear_layout(prof)['target_from_left']
hull = []
for dx in (-PC_BOX[0], PC_BOX[0]):
    for dy in (-PC_BOX[1], PC_BOX[1]):
        for dt in (-PC_BOX[2], 0, PC_BOX[2]):
            t = np.radians(PORT[2] + dt); Rm = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
            cen = np.array([PORT[0] + dx, PORT[1] + dy]) + Rm @ [dep / 2, -(wid / 2 - tfl)]
            hull += [cen + Rm @ [sx * dep / 2, sy * wid / 2] for sx in (-1, 1) for sy in (-1, 1)]
hull = cv2.convexHull(np.array(hull, np.float32))
u, v = np.meshgrid(np.linspace(-READER_BACK, READER_FRONT, 9), np.linspace(-READER_HALF_W, READER_HALF_W, 5))
foot = np.stack([u.ravel(), v.ravel()], -1)
clear = np.ones(len(cells), bool)
for i, (x, y, t0) in enumerate(cells):
    for cx, cy, tt in corner:
        t = np.radians(t0 + tt)
        pts = np.array([x + cx, y + cy]) + foot @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        if max(cv2.pointPolygonTest(hull, (float(px), float(py)), True) for px, py in pts) > -READER_MARGIN:
            clear[i] = False; break
ok = (reach & clear).reshape(len(xs), len(ys), len(yb))
cell_ok = ok.all(-1)
# compact square: all cells feasible; maximize the feasible margin ring (cells within 2 cm outside)
k = int(round(a.side / CELL))
best = None
for i in range(len(xs) - k + 1):
    for j in range(len(ys) - k + 1):
        if not cell_ok[i:i + k, j:j + k].all(): continue
        ring = cell_ok[max(0, i - 2):i + k + 2, max(0, j - 2):j + k + 2].mean()
        dist = np.hypot(xs[i:i + k].mean(), ys[j:j + k].mean())       # tie-break: nearer the base (better dexterity)
        score = (ring, -dist)
        if best is None or score > best[0]: best = (score, i, j)
summary = {'grid_cells': int(cell_ok.size), 'cells_all_bins_feasible': int(cell_ok.sum()), 'reach_frac': float(reach.mean()),
           'clear_frac': float(clear.mean()), 'tilt_deg': a.tilt, 'joint_margin_rad': a.margin, 'yaw_half_range_deg': a.yaw,
           'pad_contact_height_m': pad_h}
out = Path('results/usbc/policy/pick'); out.mkdir(parents=True, exist_ok=True)
if best is None:
    print(json.dumps(summary, indent=1)); raise SystemExit(f'no fully feasible {a.side} m square')
_, i, j = best
sel = [(ii, jj) for ii in range(i, i + k) for jj in range(j, j + k)]
samples = [[float(xs[ii]), float(ys[jj]), float(t)] for ii, jj in sel for t in yb]
region = {'samples_xy_yaw': samples, 'cell_m': CELL, 'yaw_bin_deg': BIN,
          'x_range_m': [float(xs[i] - CELL / 2), float(xs[i + k - 1] + CELL / 2)], 'y_range_m': [float(ys[j] - CELL / 2), float(ys[j + k - 1] + CELL / 2)],
          'yaw_range_deg': [-a.yaw, a.yaw], 'centre_xy_m': [float(xs[i:i + k].mean()), float(ys[j:j + k].mean())],
          'area_m2': round(k * k * CELL * CELL, 4), 'n_cells': k * k, 'mean_yaw_bins_per_cell': float(len(yb)),
          'clearance': f'reader footprint (+{READER_MARGIN * 100:.1f} cm) outside the hull of every PC pose of the box; '
                       f'60 deg side grasp pre-grasp/grasp/lift IK with >= {a.margin} rad joint margin at all cell corners and bin edges',
          'feasible_ring_fraction': float(best[0][0]), 'search': summary,
          'note': 'samples are cell centres (cell_m) x yaw bin centres (yaw_bin_deg); BimanualRuntime._sample_layout jitters within them'}
(out / 'reader_region.json').write_text(json.dumps(region, indent=1) + '\n')
np.savez_compressed(out / 'reader_region_search.npz', xs=xs, ys=ys, yaw_bins=yb, feasible=ok, reach=reach.reshape(ok.shape), clear=clear.reshape(ok.shape))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 7))
frac = ok.mean(-1)
im = ax.imshow(frac, origin='lower', extent=[ys[0] * 100 - .5, ys[-1] * 100 + .5, xs[0] * 100 - .5, xs[-1] * 100 + .5], cmap='RdYlGn', vmin=0, vmax=1)
ax.invert_xaxis(); plt.colorbar(im, ax=ax, label=f'fraction of yaw bins feasible (|yaw| <= {a.yaw:g})')
h = hull.reshape(-1, 2) * 100; ax.fill(h[:, 1], h[:, 0], color='k', alpha=.3, label='PC hull (all box poses)')
y0, y1 = region['y_range_m']; x0, x1 = region['x_range_m']
ax.plot(np.array([y0, y1, y1, y0, y0]) * 100, np.array([x0, x0, x1, x1, x0]) * 100, 'm-', lw=2, label=f'reader region {a.side*100:.0f} cm')
ax.plot(0, 0, 'b^', ms=12, label='right arm base'); ax.set(xlabel='y (cm)', ylabel='x (cm)', title='Part 1 reader start region (60 deg side grasp)')
ax.set_xlim(ys[-1] * 100 + 2, -8); ax.set_ylim(-4, 50); ax.legend(loc='upper right', fontsize=8)
fig.tight_layout(); fig.savefig(out / 'reader_region.png', dpi=110)
print(json.dumps({kk: vv for kk, vv in region.items() if kk != 'samples_xy_yaw'}, indent=1))
