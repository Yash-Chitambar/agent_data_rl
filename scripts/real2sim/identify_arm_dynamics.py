"""Identify per-arm gravity and friction residuals from recorded real YAM motion.

Torque balance at each 50 Hz observation (motor output, encoder frame for the PD):

    kp (q_cmd - q) - kd qd + f_g * g_driver(q_enc) = g_true(q_model) + F_c d

`g_driver` is the reconstructed i2rt driver gravity model with its recorded factors
`f_g`; its Coulomb compensation is not included, so F_c is the effective friction the
plant shows under that driver. `g_true = (1 + psi) * g_scene`: the scene's gravity
(Menagerie + FlexPoint + camera) with one torque scale per joint, ridge-regularised toward
0 with the weight chosen by cross-day error. Free per-link inertial fits (20 or 5
parameters) generalised worse across days; both arms independently give j3 +11-17%.

The arms creep in stick-slip (encoder velocity is spikes), so `d` is the direction of the
last commanded move, latched for `--hold-s` after the command stops: a PD-driven joint
stops on the side it approached from. Joints never commanded to move, and j2/j3 samples
on their lower stops (folded rest), are excluded for that joint. Inertia is ignored and
high-acceleration samples dropped. Validation: leave-one-trial-out, and across days
(Oct 25-26 captures vs the Oct 28 bundle).

Writes `profile.json` (consumed by `blind_insertion.envs.arm_residual`) and `report.json`.
"""
import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import savgol_filter

from blind_insertion.envs import arm_residual as AR

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2')
RIGHT = BUNDLE / 'reference/right_arm_20261027/motion'
LEGACY = ROOT / 'data/real_yam'
# Free-space trials only: contact and payload (reader grasp) trials are reserved for later checks.
TRIALS = {
    'left': [*(BUNDLE / f'motion/left_rest_j{i}_fit' for i in range(1, 7)), BUNDLE / 'motion/left_elevated_j3_001',
             *(BUNDLE / f'contact/free_settle_00{i}' for i in range(1, 4)),
             *(LEGACY / f'left_steps_joint{i}.csv' for i in range(1, 7)),
             *(LEGACY / f'left_probe_joint{i}.csv' for i in range(1, 7)),
             *(LEGACY / f'left_break_joint3_{a}.csv' for a in ('0.06', '0.08', '0.10'))],
    'right': [*(RIGHT / f'right_rest_empty_j{i}_triangle_001' for i in range(1, 7)),
              RIGHT / 'right_pointed_empty_j3_triangle_001', RIGHT / 'right_pickup_mid_empty_hold_001',
              RIGHT / 'right_rest_reconnect_j3_001',
              *(LEGACY / f'steps_joint{i}.csv' for i in range(1, 7)),
              *(LEGACY / f'probe_joint{i}.csv' for i in range(1, 7)),
              *(LEGACY / f'right_break_joint3_{a}.csv' for a in ('0.06', '0.08', '0.10'))],
}
STOP_MARGIN = 0.08      # rad above the j2/j3 lower limit (0): folded rest sits on the stops


def day(path):
    return 'oct25_26' if LEGACY in path.parents else 'oct28'


def load_trial(path, side):
    """Observation-aligned arrays; the active command at an observation is the last one sent before it."""
    if path.suffix == '.csv':                                   # Oct 25-26 deploy.py logs
        meta = json.loads(path.with_name(path.stem + '_meta.json').read_text())
        ctrl = meta['arm'] if isinstance(meta['arm'], dict) else ast.literal_eval(meta['arm'])
        offsets = np.array(meta['joint_offsets_rad'])
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        t = np.array([float(r['t']) for r in rows])
        q = np.array([[float(r[f'q_{i}']) for i in range(1, 7)] for r in rows])
        cmd = np.array([[float(r[f'q_cmd_{i}']) for i in range(1, 7)] for r in rows])
        t_send = np.array([float(r['t_cmd']) for r in rows])
    elif (path / 'trial.json').exists():                        # contact-runner free-settle trials
        meta = json.loads((path / 'trial.json').read_text())
        ctrl = meta['controllers'][side]
        rows = [r for r in meta['rows'] if f'executed_{side}_encoder' in r]
        t = np.array([r['observations'][side]['observation_time'] for r in rows])
        q = np.array([r['observations'][side]['q_encoder'][:6] for r in rows])
        cmd = np.array([r[f'executed_{side}_encoder'][:6] for r in rows])
        t_send = np.array([r[f'{side}_first_can_send_host_s'] for r in rows])
        offsets = np.array(json.loads((BUNDLE / 'motion/left_rest_j3_fit/run.json').read_text())
                           ['arm'][side]['joint_offset_rad'])
    else:                                                       # measure_motion.py runs
        meta = json.loads((path / 'run.json').read_text())
        ctrl, offsets = meta['arm'][side]['gains'], np.array(meta['arm'][side]['joint_offset_rad'])
        rows = []
        for name in ('approach.csv', 'steps.csv'):
            if (path / name).exists():
                with (path / name).open(newline='') as f:
                    rows += [r for r in csv.DictReader(f) if r['side'] == side and r['q_cmd'] and r['t_send']]
        rows.sort(key=lambda r: float(r['t_obs']))
        t = np.array([float(r['t_obs']) for r in rows])
        q = np.array([json.loads(r['q'])[:6] for r in rows])
        cmd = np.array([json.loads(r['q_cmd'])[:6] for r in rows])
        t_send = np.array([float(r['t_send']) for r in rows])
    order = np.argsort(t_send, kind='stable')
    active = np.searchsorted(t_send[order], t, side='left') - 1
    keep = active >= 0
    t, q, cmd_active = t[keep], q[keep], cmd[order][active[keep]]
    return {'name': path.stem if path.suffix else path.name, 'path': str(path), 'day': day(path), 'side': side,
            't': t - t[0], 'q_enc': q, 'cmd': cmd_active,
            'send_t': t_send[order] - (t[0] + 0.0), 'send_cmd': cmd[order],
            'kp': np.array(ctrl['kp'], float), 'kd': np.array(ctrl['kd'], float),
            'gfac': np.array(ctrl['gravity_comp_factor'][:6], float), 'offsets': offsets}


def latched_direction(t, cmd, v0, hold_s):
    """Sign of the last commanded move per joint, held for `hold_s` after it stops (0 otherwise)."""
    step = np.diff(cmd, axis=0, prepend=cmd[:1])
    dt = np.maximum(np.diff(t, prepend=t[0] - 0.02), 1e-3)
    moving = np.abs(step) / dt[:, None] > v0
    out = np.zeros_like(cmd)
    for j in range(cmd.shape[1]):
        last_sign, last_t = 0.0, -np.inf
        for k in range(len(t)):
            if moving[k, j]:
                last_sign, last_t = np.sign(step[k, j]), t[k]
            out[k, j] = last_sign if t[k] - last_t <= hold_s else 0.0
    return out


def build_samples(tr, model, driver, a):
    q_model = tr['q_enc'] + tr['offsets']
    dt = np.median(np.diff(tr['t']))
    win = 11 if len(tr['t']) > 11 else (len(tr['t']) // 2) * 2 - 1
    v = savgol_filter(tr['q_enc'], win, 2, deriv=1, delta=dt, axis=0)
    acc = savgol_filter(tr['q_enc'], win, 2, deriv=2, delta=dt, axis=0)
    dd = mujoco.MjData(driver)
    g_drv = np.zeros_like(q_model)
    for k, qe in enumerate(tr['q_enc']):
        dd.qpos[:6] = qe
        mujoco.mj_forward(driver, dd)
        g_drv[k] = dd.qfrc_bias[:6]
    g_drv *= tr['gfac']
    dirn = latched_direction(tr['t'], tr['cmd'], a.v0, a.hold_s)
    usable = (np.abs(acc[:, 1:]) < a.max_acc).all(1)[:, None] & (dirn != 0)
    usable[:, 1] &= q_model[:, 1] > STOP_MARGIN
    usable[:, 2] &= q_model[:, 2] > STOP_MARGIN
    return {'Y': np.stack([model.regressor(qm) for qm in q_model]),
            'tau': tr['kp'] * (tr['cmd'] - tr['q_enc']) - tr['kd'] * v + g_drv,
            'dir': dirn, 'usable': usable, 'g_driver': g_drv}


def fit(samples, model, lam):
    """Returns [psi j1..j6 (6) | coulomb j1..j6 (6)]; j1 has no gravity, so psi_1 stays 0."""
    rows, ys = [], []
    for s in samples:
        for j in range(6):
            ok = s['usable'][:, j]
            X = np.zeros((ok.sum(), 12))
            g = s['Y'][ok, j, :] @ model.theta_scene
            X[:, j], X[:, 6 + j] = g, s['dir'][ok, j]
            rows.append(X); ys.append(s['tau'][ok, j] - g)
    X, y = np.vstack(rows), np.concatenate(ys)
    return np.linalg.solve(X.T @ X + np.diag(np.r_[np.full(6, lam), np.full(6, 1e-6)]), X.T @ y)


def predicted_error(phi, s, tr, model, mode):
    """Joint error the PD must hold (rad). `ideal`: current sim (perfect gravity cancellation);
    `scene`: scene gravity, driver ff, no friction; `fit`: identified scale and friction."""
    if mode == 'ideal':
        return np.zeros_like(tr['cmd'])
    g = np.einsum('njp,p->nj', s['Y'], model.theta_scene)
    if mode == 'fit':
        g = g * (1 + phi[:6]) + phi[6:] * s['dir']
    return (g - s['g_driver']) / tr['kp']


def rmse(pred, tr, s):
    err = np.where(s['usable'], pred - (tr['cmd'] - tr['q_enc']), np.nan)
    with np.errstate(invalid='ignore'):
        return [None if not np.isfinite(x) else round(float(x) * 1e3, 2)
                for x in np.sqrt(np.nanmean(err ** 2, 0))] if s['usable'].any() else [None] * 6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lam', type=float, nargs='+', default=[1e1, 1e2, 1e3, 1e4, 1e5],
                   help='ridge weights toward unit gravity scales; the best cross-day one is kept')
    p.add_argument('--v0', type=float, default=0.003, help='commanded speed counted as a move (rad/s)')
    p.add_argument('--hold-s', type=float, default=5.0, help='seconds the last move direction stays latched')
    p.add_argument('--max-acc', type=float, default=0.5, help='drop samples with larger |qdd| (rad/s^2)')
    p.add_argument('--out', type=Path, default=ROOT / 'results/usbc/bimanual_20261029/dynamics_id')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    driver = mujoco.MjModel.from_xml_path(str(AR.DRIVER_XML))
    report = {'method': __doc__.strip(), 'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              'driver_xml_sha256': hashlib.sha256(AR.DRIVER_XML.read_bytes()).hexdigest(), 'sides': {}}
    # MuJoCo's default soft frictionloss creeps under sustained sub-threshold load (a held
    # arm drifts by residual/kp in ~5 s); the real joints stick. A stiff friction impedance
    # holds them (replay: uncommanded j2/j3 drift 10-13 -> 1-1.5 mrad). solref >= 2 dt.
    profile = {'source': 'scripts/real2sim/identify_arm_dynamics.py',
               'model': 'g_true = gravity_scale * g_scene; frictionloss = coulomb_nm',
               'friction_solref': [0.004, 1.0], 'friction_solimp': [0.995, 0.9995, 0.001, 0.5, 2.0], 'arms': {}}
    for side, paths in TRIALS.items():
        model = AR.ArmRegressor(side)
        trials = [load_trial(path, side) for path in paths]
        samples = [build_samples(tr, model, driver, a) for tr in trials]
        sweep = {}
        for lam in a.lam:
            by_day = {d: fit([s for s, tr in zip(samples, trials) if tr['day'] != d], model, lam)
                      for d in ('oct28', 'oct25_26')}
            sq = [np.concatenate([(predicted_error(by_day[tr['day']], s, tr, model, 'fit') - (tr['cmd'] - tr['q_enc']))
                                  [s['usable'][:, j], j] ** 2 for tr, s in zip(trials, samples)]) for j in range(6)]
            sweep[lam] = float(np.mean([np.sqrt(x.mean()) for x in sq if len(x)]))
        lam = min(sweep, key=sweep.get)
        theta = fit(samples, model, lam)
        by_day = {d: fit([s for s, tr in zip(samples, trials) if tr['day'] != d], model, lam)
                  for d in ('oct28', 'oct25_26')}
        rep = {'trials': {}, 'lam_sweep_cross_day_mean_rmse_mrad': {f'{k:g}': round(v * 1e3, 2) for k, v in sweep.items()},
               'lam': lam}
        for i, (tr, s) in enumerate(zip(trials, samples)):
            loo = fit(samples[:i] + samples[i + 1:], model, lam)
            meas = np.where(s['usable'], tr['cmd'] - tr['q_enc'], np.nan)
            with np.errstate(invalid='ignore'):
                meas_rms = np.sqrt(np.nanmean(meas ** 2, 0)) * 1e3
            rep['trials'][tr['name']] = {
                'day': tr['day'], 'path': tr['path'], 'usable_samples_per_joint': s['usable'].sum(0).tolist(),
                'measured_error_rms_mrad': [None if not np.isfinite(x) else round(float(x), 2) for x in meas_rms],
                'current_sim_rmse_mrad': rmse(predicted_error(None, s, tr, model, 'ideal'), tr, s),
                'scene_gravity_driver_ff_no_friction_rmse_mrad': rmse(predicted_error(theta, s, tr, model, 'scene'), tr, s),
                'identified_leave_one_out_rmse_mrad': rmse(predicted_error(loo, s, tr, model, 'fit'), tr, s),
                'identified_other_day_rmse_mrad': rmse(predicted_error(by_day[tr['day']], s, tr, model, 'fit'), tr, s)}
        rep['gravity_scale'] = (1 + theta[:6]).round(4).tolist()
        rep['coulomb_nm'] = theta[6:].round(4).tolist()
        for d, th in by_day.items():
            other = 'fit_on_' + ('oct25_26' if d == 'oct28' else 'oct28')
            rep.setdefault('by_day', {})[other] = {'gravity_scale': (1 + th[:6]).round(4).tolist(),
                                                   'coulomb_nm': th[6:].round(4).tolist()}
        report['sides'][side] = rep
        profile['arms'][side] = {'gravity_scale': (1 + theta[:6]).tolist(),
                                 'coulomb_nm': np.maximum(theta[6:], 0).tolist(),
                                 'driver_gravity_comp_factor': trials[0]['gfac'].tolist(),
                                 'held_out_by_day': {d: {'gravity_scale': (1 + th[:6]).tolist(),
                                                         'coulomb_nm': np.maximum(th[6:], 0).tolist()}
                                                     for d, th in by_day.items()}}
        plot_side(a.out / f'{side}_errors.png', side, trials, samples, theta, model)
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    (a.out / 'profile.json').write_text(json.dumps(profile, indent=1) + '\n')
    summarise(report)


def summarise(report):
    for side, rep in report['sides'].items():
        print(f'== {side}: lam {rep["lam"]:g} sweep {rep["lam_sweep_cross_day_mean_rmse_mrad"]}')
        print(f'   gravity scale {rep["gravity_scale"]}  coulomb {rep["coulomb_nm"]}')
        print(f'   by day {rep["by_day"]}')
        agg = {}
        for name, e in rep['trials'].items():
            for key in ('current_sim_rmse_mrad', 'identified_leave_one_out_rmse_mrad', 'identified_other_day_rmse_mrad'):
                for j, (x, n) in enumerate(zip(e[key], e['usable_samples_per_joint'])):
                    if x is not None and n:
                        agg.setdefault((key, j), []).append((x ** 2, n))
            print(f'  {name:36s} {e["day"]:8s} sim {e["current_sim_rmse_mrad"]}  loo {e["identified_leave_one_out_rmse_mrad"]}'
                  f'  xday {e["identified_other_day_rmse_mrad"]}')
        for key in ('current_sim_rmse_mrad', 'identified_leave_one_out_rmse_mrad', 'identified_other_day_rmse_mrad'):
            vals = []
            for j in range(6):
                x = agg.get((key, j), [])
                vals.append(round(float(np.sqrt(sum(a * n for a, n in x) / max(1, sum(n for _, n in x)))), 2))
            print(f'  pooled {key:40s} j1-6 {vals}')


def plot_side(path, side, trials, samples, theta, model):
    elevated = [(tr, s) for tr, s in zip(trials, samples) if s['usable'][:, 1:4].any()
                and np.median(tr['q_enc'][:, 1]) > 0.3][:4]
    if not elevated:
        return
    fig, axes = plt.subplots(len(elevated), 3, figsize=(15, 3 * len(elevated)), squeeze=False)
    for row, (tr, s) in zip(axes, elevated):
        pred = predicted_error(theta, s, tr, model, 'fit')
        for ax, j in zip(row, (1, 2, 4)):
            ax.plot(tr['t'], (tr['cmd'] - tr['q_enc'])[:, j] * 1e3, lw=.6, label='real')
            ax.plot(tr['t'], np.where(s['usable'][:, j], pred[:, j], np.nan) * 1e3, lw=.8, label='identified (in-sample)')
            ax.axhline(0, color='k', lw=.6, ls=':', label='current sim (ideal gravcomp)')
            ax.set_title(f'{tr["name"]} j{j + 1}', fontsize=9); ax.grid(alpha=.3)
        row[0].set_ylabel('cmd - q (mrad)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f'{side} arm: tracking error, real vs model (usable samples)')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
