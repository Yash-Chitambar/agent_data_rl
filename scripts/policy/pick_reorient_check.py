"""After the Part 1 pick: can the right arm turn the held reader to the port? (kinematic check)

For every reader start pose of the Part 1 region (cell corners x yaw-bin edges) and every PC pose of
the training box (port +/-1.5 cm, +/-5 deg: 3 x 3 x 3 corners), with the nominal in-hand pose of the
grasp (reader_grasp, tilt and flip from the region file):
  * lift pose IK (start) and pre-insert pose IK (plug tip 10 mm out of the port, aligned) with every
    joint >= 0.05 rad inside its limits;
  * at the pre-insert pose, the right hand (fingers closed on the reader) stays >= 5 mm outside the
    PC box (below its top + 5 mm) and >= 1 cm from the left arm in its nominal press pose.
Joint-space path between the two poses is not checked (Part 2 plans it).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/policy/pick_reorient_check.py
Writes results/usbc/policy/pick/reorient_check.json.
"""
import json, math
from pathlib import Path
import mujoco, numpy as np, torch
from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.scenes import pc_bench as S, physical_rig as R, reader_grasp as G

dev = 'cuda:0'
reg = json.loads(Path('results/usbc/policy/pick/reader_region.json').read_text())
tilt, flip = reg['grasp']['tilt_deg'], reg['grasp']['flip']
prof = json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m, l = S.build_scene(detailed=False, bimanual=True, support_mode='free', profile=prof, measured_support=True)
d = mujoco.MjData(m); R.reference_pose(m, d, l); mujoco.mj_forward(m, d)
hand = G.hand_geometry(m, d)
off, _ = G.grasp_offset(l, hand, tilt, flip=flip)
p_in, R_in = G.nominal_inhand(off, tilt, flip)            # reader in grasp_site
chain = K.Chain(m, l.tcp_site, [int(m.joint(f'joint{i}').qposadr[0]) for i in range(1, 7)], device=dev)
lo, hi = chain.lo + .05, chain.hi - .05

# hand points with the fingers closed on the reader (pad gap = body width)
for f in ('left', 'right'):
    d.qpos[m.joint(f + '_finger').qposadr] = .019
mujoco.mj_forward(m, d)
sid = m.site('grasp_site').id; Rs0, ps0 = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
def subtree(b):
    return [b] + [c for c in range(m.nbody) if m.body_parentid[c] == b and c != b for c in subtree(c)]
def pts_of(bodies, dd, frame=None):
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bodies: continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]; V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            sz = m.geom_size[g]; V = np.array([[a * sz[0], b * sz[1], c * sz[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        else: continue
        W = V @ dd.geom_xmat[g].reshape(3, 3).T + dd.geom_xpos[g]
        out.append(W if frame is None else (W - frame[1]) @ frame[0])
    return np.concatenate(out)
hp = pts_of(set(subtree(m.body('link_6').id)), d, (Rs0, ps0))
hp = torch.tensor(hp[np.random.default_rng(0).choice(len(hp), 500, replace=False)], dtype=torch.float32, device=dev)
rep = json.loads(Path('results/usbc/bimanual_20261029/placement/report.json').read_text())['nominal']
dl = mujoco.MjData(m); dl.qpos[:] = d.qpos
lq = [int(m.joint(f'left_joint{i}').qposadr[0]) for i in range(1, 7)]; dl.qpos[lq] = rep['q_left']; mujoco.mj_kinematics(m, dl)
left_b = {b for b in range(m.nbody) if m.body(b).name.startswith(('left_link_', 'left_gripper', 'left_left_finger', 'left_right_finger', 'left_wrist'))}
lp = torch.tensor(pts_of(left_b, dl), dtype=torch.float32, device=dev)

# reader start poses (lift) and PC poses
s = np.array(reg['samples_xy_yaw']); h, hb = reg['cell_m'] / 2, reg['yaw_bin_deg'] / 2
starts = np.concatenate([s + [sx * h, sy * h, st * hb] for sx in (-1, 1) for sy in (-1, 1) for st in (-1, 1)])
S.place_reader(m, d, l, xy=(0., 0.), yaw_deg=0.); q0 = d.qpos[l.peg_qadr:l.peg_qadr + 7].copy()
R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, q0[3:]); R0 = R0.reshape(3, 3)
pc_box = BimanualCfg.pc_pose_range; port0 = (.22, .36)
pcs = [(port0[0] + dx, port0[1] + dy, dt) for dx in (-pc_box[0], 0, pc_box[0]) for dy in (-pc_box[1], 0, pc_box[1]) for dt in (-pc_box[2], 0, pc_box[2])]
pc = prof['pc']['measurements']; dep, wid = pc['depth']['value'], pc['width']['value']
from blind_insertion.scenes.pc_rear import layout as rear_layout
tfl = rear_layout(prof)['target_from_left']; top = float(m.geom('table').pos[2] + m.geom('table').size[2]) + pc['height_with_feet']['value']

def ik_ok(p, Rm, seeds):
    best = torch.zeros(len(p), dtype=torch.bool, device=dev); qb = seeds[0].clone()
    for sd in seeds:
        q, pe, re = chain.ik(p, Rm, sd.clamp(chain.lo + .05, chain.hi - .05), iters=100)
        g = (pe < 1e-3) & (re < 1e-2) & ((q >= lo) & (q <= hi)).all(1)
        qb = torch.where((g & ~best)[:, None], q, qb); best |= g
    return best, qb

out = {'n_starts': int(len(starts)), 'n_pc_poses': len(pcs), 'tilt_deg': tilt, 'flip': flip}
# lift IK at every start
yaw = np.radians(starts[:, 2]); n = len(starts)
Rz = np.stack([np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]]) for t in yaw])
pr = np.c_[starts[:, :2], np.zeros(n)] + np.einsum('nij,j->ni', Rz, [q0[0], q0[1], 0.]); pr[:, 2] = q0[2]
Rr = Rz @ R0
W = G.grasp_poses(torch.tensor(pr, dtype=torch.float32, device=dev), torch.tensor(Rr, dtype=torch.float32, device=dev), off, tilt,
                  flip=torch.full((n,), bool(flip), device=dev))
gen = torch.Generator(device=dev); gen.manual_seed(0)
aim = torch.atan2(W['lift'][0][:, 1], W['lift'][0][:, 0])
seeds = [torch.cat([aim[:, None], torch.tensor(sd, device=dev).expand(n, 5)], 1) for sd in ([1.0, 1.0, -.5, 0, 0], [1.6, 1.3, -.9, 0, 0], [2.2, 1.8, -1.2, 0, 0])]
seeds += [chain.lo + (chain.hi - chain.lo) * torch.rand(n, 6, device=dev, generator=gen) for _ in range(12)]
lift_ok, q_lift = ik_ok(*W['lift'], seeds)
out['lift_ik_ok'] = float(lift_ok.float().mean())
# pre-insert per PC pose (independent of the start: same in-hand pose)
res = []
for (px, py, dt) in pcs:
    S.place_pc(m, d, port_xy=(px, py), yaw_deg=dt); mujoco.mj_forward(m, d)
    fa = int(m.joint('pc_free_joint').qposadr[0]); Rp = np.zeros(9); mujoco.mju_quat2Mat(Rp, d.qpos[fa + 3:fa + 7]); Rp = Rp.reshape(3, 3)
    p_reader = d.qpos[fa:fa + 3] + Rp @ [0, 0, .010]; R_reader = Rp
    R_site = R_reader @ R_in.T; p_site = p_reader - R_site @ p_in
    P = torch.tensor(p_site, dtype=torch.float32, device=dev)[None]; Rm = torch.tensor(R_site, dtype=torch.float32, device=dev)[None]
    sds = [q_lift[lift_ok][:64]] if lift_ok.any() else []
    sds = [x for x in sds if len(x)] + [chain.lo + (chain.hi - chain.lo) * torch.rand(64, 6, device=dev, generator=gen) for _ in range(4)]
    ok_any = False; q_sol = None
    for sd in sds:
        q, pe, re = chain.ik(P.expand(len(sd), 3), Rm.expand(len(sd), 3, 3), sd.clamp(chain.lo + .05, chain.hi - .05), iters=150)
        g = (pe < 1e-3) & (re < 1e-2) & ((q >= lo) & (q <= hi)).all(1)
        if g.any(): ok_any = True; q_sol = q[g][0]; break
    # hand clearance at pre-insert
    H = P[0] + hp @ Rm[0].T
    t = math.radians(dt); Rz2 = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    cen = np.array([px, py]) + Rz2 @ [dep / 2, -(wid / 2 - tfl)]
    loc = (H[:, :2].cpu().numpy() - cen) @ Rz2                                  # PC box frame (x depth, y width)
    inside = (np.abs(loc[:, 0]) < dep / 2 + .005) & (np.abs(loc[:, 1]) < wid / 2 + .005) & (H[:, 2].cpu().numpy() < top + .005)
    dleft = float(torch.cdist(H, lp).min())
    res.append({'port_xy': [px, py], 'yaw_deg': dt, 'ik_ok': ok_any, 'hand_in_pc_points': int(inside.sum()), 'left_arm_min_dist_m': dleft,
                'ok': bool(ok_any and inside.sum() == 0 and dleft >= .01)})
out['pre_insert'] = res
out['pre_insert_ok'] = f"{sum(r['ok'] for r in res)}/{len(res)}"
print(json.dumps({k: v for k, v in out.items() if k != 'pre_insert'}, indent=1))
for r in res:
    if not r['ok']: print('fail', r)
Path('results/usbc/policy/pick/reorient_check.json').write_text(json.dumps(out, indent=1) + '\n')
