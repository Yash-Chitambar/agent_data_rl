"""Closed-loop CPU replay of recorded real commands: current sim vs identified dynamics.

Each recorded command stream (CAN send times, recorded gains, encoder offsets) drives the
bimanual scene (contacts off, 1 ms). Variants:

- `current`: what the bimanual runtime does now: perfect gravity cancellation (gravcomp),
  scene frictionloss 0.01 N m, right joint 2 at 1.0 N m.
- `identified_held_out`: `tau = f_g * g_driver(q_enc) - s * g_scene(q)` replaces the perfect
  cancellation, joint frictionloss = identified Coulomb with a stiff friction impedance,
  using parameters fitted only on the *other* day's trials (honest held-out test).
- `identified`: same with parameters fitted on all trials (what training would use).

Metrics: joint RMSE against the real encoders at observation times and tool (grasp site)
position error from forward kinematics of the simulated vs real joints, in mm.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import identify_arm_dynamics as I
from blind_insertion.envs import arm_residual as AR
from blind_insertion.scenes import pc_bench as S

_SCENE = {}


def scene():
    if not _SCENE:
        m, _ = S.build_scene(bimanual=True)
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        _SCENE['m'] = m
        _SCENE['driver'] = mujoco.MjModel.from_xml_path(str(AR.DRIVER_XML))
    return _SCENE['m'], _SCENE['driver']


def arm_indices(m, side):
    pre = 'left_' if side == 'left' else ''
    return ([int(m.joint(pre + f'joint{i}').qposadr[0]) for i in range(1, 7)],
            [int(m.joint(pre + f'joint{i}').dofadr[0]) for i in range(1, 7)],
            [m.actuator(pre + f'joint{i}').id for i in range(1, 7)], m.site(pre + 'grasp_site').id)


def simulate(tr, params):
    """Simulated joint angles (encoder frame) at the trial's observation times."""
    m, driver = scene()
    qadr, dof, act, _ = arm_indices(m, tr['side'])
    saved = (m.dof_frictionloss.copy(), m.actuator_gainprm.copy(), m.actuator_biasprm.copy(),
             m.dof_solref.copy(), m.dof_solimp.copy())
    try:
        for i, kp, kd in zip(act, tr['kp'], tr['kd']):
            m.actuator_gainprm[i, 0], m.actuator_biasprm[i, 1], m.actuator_biasprm[i, 2] = kp, -kp, -kd
        if params is None:
            if tr['side'] == 'right':
                m.dof_frictionloss[dof[1]] = 1.0
        else:
            m.dof_frictionloss[dof] = params['coulomb_nm']
            m.dof_solref[dof], m.dof_solimp[dof] = params['friction_solref'], params['friction_solimp']
            scale, gfac = np.asarray(params['gravity_scale']), tr['gfac']
        d, dd = mujoco.MjData(m), mujoco.MjData(driver)
        off = tr['offsets']
        d.qpos[qadr] = tr['q_enc'][0] + off
        d.ctrl[act] = tr['send_cmd'][0] + off
        mujoco.mj_forward(m, d)
        T = tr['t'][-1]
        n = int(np.ceil(T / m.opt.timestep)) + 1
        out = np.zeros((n, 6)); k = 0
        for step in range(n):
            while k < len(tr['send_t']) and tr['send_t'][k] <= d.time:
                d.ctrl[act] = tr['send_cmd'][k] + off; k += 1
            if params is not None:
                dd.qpos[:6] = d.qpos[qadr] - off
                mujoco.mj_kinematics(driver, dd); mujoco.mj_comPos(driver, dd)
                mujoco.mj_rne(driver, dd, 0, dd.qfrc_bias)          # zero velocity: gravity only
                d.qfrc_applied[dof] = gfac * dd.qfrc_bias[:6] - scale * d.qfrc_gravcomp[dof]
            out[step] = d.qpos[qadr] - off
            mujoco.mj_step(m, d)
        t_sim = np.arange(n) * m.opt.timestep
        return np.column_stack([np.interp(tr['t'], t_sim, out[:, j]) for j in range(6)])
    finally:
        (m.dof_frictionloss[:], m.actuator_gainprm[:], m.actuator_biasprm[:],
         m.dof_solref[:], m.dof_solimp[:]) = saved


def tool_positions(side, q_enc, offsets):
    m, _ = scene()
    qadr, _, _, site = arm_indices(m, side)
    d = mujoco.MjData(m)
    out = np.zeros((len(q_enc), 3))
    for i, q in enumerate(q_enc):
        d.qpos[qadr] = q + offsets
        mujoco.mj_kinematics(m, d)
        out[i] = d.site_xpos[site]
    return out


def job(args):
    path, side, profile = args
    tr = I.load_trial(Path(path), side)
    arm = profile['arms'][side]
    real_tool = tool_positions(side, tr['q_enc'], tr['offsets'])
    res = {'name': tr['name'], 'day': tr['day'], 'side': side, 'duration_s': float(tr['t'][-1]),
           'elevated': bool(np.median(tr['q_enc'][:, 1]) > 0.3), 'variants': {}}
    curves = {'t': tr['t'], 'real': tr['q_enc'], 'cmd': tr['cmd']}
    stiff = {k: profile[k] for k in ('friction_solref', 'friction_solimp')}
    for name, params in (('current', None), ('identified_held_out', {**arm['held_out_by_day'][tr['day']], **stiff}),
                         ('identified', {**arm, **stiff})):
        q = simulate(tr, params)
        tool = tool_positions(side, q, tr['offsets'])
        err_mm = np.linalg.norm(tool - real_tool, axis=1) * 1e3
        res['variants'][name] = {'joint_rmse_mrad': (np.sqrt(np.mean((q - tr['q_enc']) ** 2, 0)) * 1e3).round(2).tolist(),
                                 'tool_error_mm_mean': round(float(err_mm.mean()), 2),
                                 'tool_error_mm_p95': round(float(np.percentile(err_mm, 95)), 2),
                                 'tool_error_mm_max': round(float(err_mm.max()), 2)}
        curves[name] = q; curves[name + '_tool_mm'] = err_mm
    return res, curves


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', type=Path, default=I.ROOT / 'results/usbc/bimanual_20261029/dynamics_id/profile.json')
    p.add_argument('--out', type=Path, default=I.ROOT / 'results/usbc/bimanual_20261029/dynamics_id/replay')
    p.add_argument('--only', nargs='*', help='trial names to replay')
    p.add_argument('--workers', type=int, default=24)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    profile = json.loads(a.profile.read_text())
    jobs = [(str(path), side, profile) for side, paths in I.TRIALS.items() for path in paths
            if not a.only or (path.stem if path.suffix else path.name) in a.only]
    with Pool(min(a.workers, len(jobs))) as pool:
        results = pool.map(job, jobs, chunksize=1)
    report = {'method': __doc__.strip(), 'profile': str(a.profile), 'trials': [r for r, _ in results]}
    for res, curves in results:
        np.savez_compressed(a.out / f'{res["side"]}_{res["name"]}.npz', **curves)
    summary = {}
    for res, _ in results:
        key = f'{res["side"]}_{"elevated" if res["elevated"] else "folded_rest"}'
        for v, e in res['variants'].items():
            s = summary.setdefault(key, {}).setdefault(v, {'sq': [], 'w': [], 'tool': []})
            s['sq'].append(np.square(e['joint_rmse_mrad'])); s['w'].append(res['duration_s']); s['tool'].append(e['tool_error_mm_mean'])
    report['summary'] = {k: {v: {'joint_rmse_mrad': np.sqrt(np.average(s['sq'], axis=0, weights=s['w'])).round(2).tolist(),
                                 'tool_error_mm_mean': round(float(np.average(s['tool'], weights=s['w'])), 2)}
                             for v, s in vs.items()} for k, vs in summary.items()}
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    for res, _ in results:
        print(f'{res["side"]:5s} {res["name"]:36s} {res["day"]:8s} ' + '  '.join(
            f'{v[:12]:12s} {e["tool_error_mm_mean"]:6.2f}mm j{e["joint_rmse_mrad"]}' for v, e in res['variants'].items()))
    print(json.dumps(report['summary'], indent=1))
    plot(a.out, results)


def plot(out, results):
    picks = [rc for rc in results if rc[0]['elevated'] and rc[0]['duration_s'] > 5][:6]
    fig, axes = plt.subplots(len(picks), 3, figsize=(16, 2.8 * len(picks)), squeeze=False)
    for row, (res, c) in zip(axes, picks):
        for ax, j in zip(row[:2], (1, 2)):
            ax.plot(c['t'], (c['cmd'][:, j] - c['real'][:, j]) * 1e3, lw=.7, label='real')
            ax.plot(c['t'], (c['cmd'][:, j] - c['current'][:, j]) * 1e3, lw=.7, label='current sim')
            ax.plot(c['t'], (c['cmd'][:, j] - c['identified_held_out'][:, j]) * 1e3, lw=.7, label='identified (held-out day)')
            ax.set_title(f'{res["side"]} {res["name"]}: j{j + 1} cmd - q (mrad)', fontsize=9); ax.grid(alpha=.3)
        row[2].plot(c['t'], c['current_tool_mm'], lw=.7, color='C1', label='current sim')
        row[2].plot(c['t'], c['identified_held_out_tool_mm'], lw=.7, color='C2', label='identified (held-out day)')
        row[2].set_title('tool position error vs real (mm)', fontsize=9); row[2].grid(alpha=.3)
    axes[0, 0].legend(fontsize=7); axes[0, 2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'replay_elevated.png', dpi=100); plt.close(fig)


if __name__ == '__main__':
    main()
