"""Evaluate the Part 1 scripted expert (top-down side grasp) on the GPU bimanual runtime with all DR.

Success per episode: the expert reaches the end of its hold, the reader is lifted >= 60 mm, its
tilt (thickness axis vs world up) is < 3 deg at the start and end of the 1 s hold, and the in-hand
pose moves < 1 mm / 1 deg during the hold. Writes per-episode parquet + summary JSON and the
in-hand pose spread (`results/usbc/policy/pick/inhand_spread.json`, read by reader_grasp.inhand_pose).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. .venv/bin/python scripts/policy/pick_expert_eval.py --episodes 2048 --seed 1000
"""
import argparse, json, math, time
from pathlib import Path
import numpy as np
import torch
import warp as wp
wp.config.quiet = True
from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickExpert, PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.scenes import reader_grasp as G

ap = argparse.ArgumentParser()
ap.add_argument('--episodes', type=int, default=2048)
ap.add_argument('--nworld', type=int, default=1024)
ap.add_argument('--seed', type=int, default=1000)
ap.add_argument('--out', default='results/usbc/policy/pick/expert')
ap.add_argument('--no_dr', action='store_true', help='nominal physics: no contact/control DR')
ap.add_argument('--write_spread', action='store_true')
ap.add_argument('--impratio', type=float, default=None)
ap.add_argument('--grip_range', type=float, nargs=2, default=None)
ap.add_argument('--pad_mu_range', type=float, nargs=2, default=None)
ap.add_argument('--layout', default='results/usbc/policy/pick/task_layout.json')
a = ap.parse_args()


def wilson_lo(k, n, z=1.96):
    if n == 0: return 0.
    p = k / n; d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


cfg = PickTaskCfg(physics=BimanualCfg(nworld=a.nworld, device='cuda:0', task_layout=a.layout))
if a.impratio is not None: cfg.impratio = a.impratio
if a.grip_range: cfg.grip_force_scale_range = tuple(a.grip_range)
if a.pad_mu_range: cfg.pad_friction_range = tuple(a.pad_mu_range)
if a.no_dr:
    cfg.pad_friction_range = cfg.reader_table_friction_range = cfg.grip_force_scale_range = None
    cfg.latency_steps = (0,); cfg.enc_offset_deg = 0.
    cfg.physics.friction_scale_range = (1., 1.); cfg.physics.gravity_scale_jitter = 0.
t0 = time.time()
task = PickTask(cfg)
print('built', round(time.time() - t0, 1), 's', flush=True)
torch.manual_seed(a.seed)
rows = []
for batch in range(math.ceil(a.episodes / a.nworld)):
    task.reset(); ex = PickExpert(task, seed=a.seed + batch); ex.reset()
    N = a.nworld
    start_hold = {k: torch.zeros(N, *s, device=task.device) for k, s in (('p', (3,)), ('R', (3, 3)), ('tilt', ()))}
    steps = torch.zeros(N, dtype=torch.long, device=task.device)
    final = None
    for k in range(task.max_steps - 1):
        prev = ex.phase.clone()
        act = ex.act()
        task.step(act)
        m = task.pick_metrics()
        entered = (ex.phase == 5) & (prev == 4)
        start_hold['p'][entered] = m['inhand_p'][entered]; start_hold['R'][entered] = m['inhand_R'][entered]
        start_hold['tilt'][entered] = m['tilt_deg'][entered]
        fin = (ex.phase == 6) & (prev == 5)
        if final is None: final = {kk: v.clone() for kk, v in m.items()}
        for kk, v in m.items(): final[kk][fin] = v[fin]
        steps[fin] = k + 1
        if ex.done.all(): break
    m_now = task.pick_metrics()
    for kk, v in m_now.items(): final[kk][~ex.done] = v[~ex.done]
    dp = (final['inhand_p'] - start_hold['p']).norm(dim=1)
    dR = torch.rad2deg(K.rot_log(final['inhand_R'] @ start_hold['R'].transpose(1, 2)).norm(dim=1))
    nominal_p = torch.zeros(N, 3, device=task.device); nominal_R = torch.zeros(N, 3, 3, device=task.device)
    for tl, off in task.offsets.items():
        np_, nR = G.nominal_inhand(off, tl); sel = ex.tilt == tl
        nominal_p[sel] = torch.tensor(np_, dtype=torch.float32, device=task.device); nominal_R[sel] = torch.tensor(nR, dtype=torch.float32, device=task.device)
    nominal_p[~ex.ik_ok] = nominal_p[ex.ik_ok][:1] if ex.ik_ok.any() else 0; nominal_R[~ex.ik_ok] = torch.eye(3, device=task.device)
    rel_p = final['inhand_p'] - nominal_p
    rel_r = K.rot_log(final['inhand_R'] @ nominal_R.transpose(1, 2))
    ok = ex.done & ~task.failed & final['finite'] & (final['lift_m'] > .06) & (final['tilt_deg'] < 3) & (start_hold['tilt'] < 3) & (dp < 1e-3) & (dR < 1.)
    pr = task.reader0
    d = {'batch': batch, 'success': ok, 'done': ex.done, 'ik_ok': ex.ik_ok, 'tilt_cmd_deg': ex.tilt, 'failed': task.failed, 'nonfinite': task.nonfinite, 'nonfinite_part': task.nonfinite_part, 'nonfinite_step': task.nonfinite_step, 'phase_end': ex.phase, 'lift_mm': final['lift_m'] * 1e3,
         'tilt_deg': final['tilt_deg'], 'tilt_start_hold_deg': start_hold['tilt'], 'hold_drift_mm': dp * 1e3, 'hold_rot_drift_deg': dR,
         'settle_err_mm': ex.settle_err[:, 0] * 1e3, 'settle_err_deg': ex.settle_err[:, 1], 'steps': steps,
         'reader_x': pr[:, 0], 'reader_y': pr[:, 1],
         'reader_yaw_deg': torch.rad2deg(torch.atan2(-K.quat_to_mat(pr[:, 3:])[:, 1, 2], -K.quat_to_mat(pr[:, 3:])[:, 0, 2])),
         'latency': task.latency, 'grip_scale': task.runtime.grip_scale if task.runtime.grip_scale is not None else torch.ones(N, device=task.device),
         'pad_mu': task.runtime.geom_friction[:, task.pads[0], 0], 'reader_mu': task.runtime.geom_friction[:, task.reader_geoms[0], 0]}
    for i, nm in enumerate('xyz'): d[f'inhand_d{nm}_mm'] = rel_p[:, i] * 1e3
    for i, nm in enumerate('xyz'): d[f'inhand_r{nm}_deg'] = torch.rad2deg(rel_r[:, i])
    d['inhand_p'] = final['inhand_p']; d['inhand_R'] = final['inhand_R']
    rows.append({kk: v.detach().cpu().numpy() if torch.is_tensor(v) else np.full(N, v) for kk, v in d.items()})
    print(f'batch {batch}: success {int(ok.sum())}/{N} done {int(ex.done.sum())} t={time.time()-t0:.0f}s', flush=True)

cat = {k: np.concatenate([r[k] for r in rows])[:a.episodes] for k in rows[0]}
P, R = cat.pop('inhand_p'), cat.pop('inhand_R')
import pandas as pd
df = pd.DataFrame(cat)
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
tag = out.name + ('_nodr' if a.no_dr else '') + f'_s{a.seed}'
df.to_parquet(out.parent / f'{tag}.parquet')
n, k = len(df), int(df.success.sum())
summ = {'n': n, 'success': k, 'rate': k / n, 'wilson95_lo': wilson_lo(k, n), 'seed': a.seed, 'dr': not a.no_dr,
        'failures': {'nonfinite': int(df.nonfinite.sum()), 'not_done': int((~df.done).sum()), 'ik_fail': int((~df.ik_ok).sum()), 'lift<60mm': int((df.lift_mm <= 60).sum()),
                     'tilt>=3': int((df.tilt_deg >= 3).sum()), 'hold_drift>=1mm': int((df.hold_drift_mm >= 1).sum())},
        'tilt_deg_p50_p99_max': [float(np.percentile(df.tilt_deg, q)) for q in (50, 99, 100)],
        'settle_err_mm_p50_p99': [float(np.percentile(df.settle_err_mm, q)) for q in (50, 99)],
        'episode_s_p50_p99': [float(np.percentile(df.steps[df.done], q) * task.dt) for q in (50, 99)] if df.done.any() else None}
okm = df.success.values
sp = {'n_success': int(okm.sum())}
for c in [c for c in df.columns if c.startswith('inhand_')]:
    v = df[c].values[okm]
    sp[c] = {'mean': float(v.mean()), 'std': float(v.std()), 'p1': float(np.percentile(v, 1)), 'p99': float(np.percentile(v, 99))}
summ['inhand_spread'] = sp
print(json.dumps(summ, indent=1))
(out.parent / f'{tag}.json').write_text(json.dumps(summ, indent=1) + '\n')
if a.write_spread:
    Pm = P[okm].mean(0); Rm = R[okm].mean(0); U, _, Vt = np.linalg.svd(Rm); Rm = U @ Vt
    from blind_insertion.scenes.reader_grasp import _mat_to_quat
    spread = {'source': f'{tag}.parquet', 'n': int(okm.sum()), 'mean_pos_m': Pm.tolist(), 'mean_quat_wxyz': _mat_to_quat(Rm).tolist(),
              'nominal': {str(tl): {'pos_m': G.nominal_inhand(off, tl)[0].tolist(), 'quat_wxyz': _mat_to_quat(G.nominal_inhand(off, tl)[1]).tolist(), 'pad_contact_height_m': task.pad_contact_height[tl]} for tl, off in task.offsets.items()},
              'tilt_share': {str(tl): float((df.tilt_cmd_deg[okm] == tl).mean()) for tl in task.offsets}, 'about_nominal': sp,
              'units': 'inhand_d* in mm (grasp_site frame), inhand_r* rotation vector in deg (grasp_site frame) about the nominal'}
    G.SPREAD_FILE.write_text(json.dumps(spread, indent=1) + '\n')
    print('wrote', G.SPREAD_FILE)
