"""Replay recorded real commands through the GPU bimanual runtime: legacy vs identified plant.

Drives `BimanualRuntime` (MuJoCo Warp, full scene, 50 Hz control, 20 substeps) with the
recorded command stream of free-space trials (randomization off), and compares the
simulated arm with the real encoders: joint RMSE and grasp-site position error in mm.
Trials are chosen so the runtime's controller profile matches the recorded gains.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

import identify_arm_dynamics as I
from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime

TRIALS = [('left', I.BUNDLE / 'motion/left_elevated_j3_001', 'identification_20261027'),
          ('left', I.LEGACY / 'left_break_joint3_0.08.csv', 'identification_20261027'),
          ('left', I.LEGACY / 'left_steps_joint3.csv', 'identification_20261027'),
          ('right', I.RIGHT / 'right_pickup_mid_empty_hold_001', 'identification_20261027'),
          ('right', I.LEGACY / 'steps_joint3.csv', 'identification_20261027'),
          ('right', I.LEGACY / 'right_break_joint3_0.08.csv', 'identification_20261027')]


def replay(e, tr):
    side = tr['side']
    q, v, act, grip, _ = e.indices[side]
    k = 0 if side == 'right' else 1
    off = torch.as_tensor(tr['offsets'], device=e.device, dtype=e.q.dtype)
    e.reset()
    e.q[:, q] = torch.as_tensor(tr['q_enc'][0], device=e.device, dtype=e.q.dtype) + off
    with wp.ScopedDevice(e.cfg.device):
        mjw.forward(e.m, e.d)
    hold = torch.cat([e.q[:, e.indices[s][0]] for s in ('right', 'left')], 1)
    action = torch.zeros((e.cfg.nworld, 14), device=e.device)
    action[:, 0:6], action[:, 7:13] = hold[:, :6], hold[:, 6:]
    action[:, 6] = action[:, 13] = 1
    dt = e.mjm.opt.timestep * e.cfg.substeps
    n = int(np.ceil(tr['t'][-1] / dt)) + 1
    sim = np.zeros((n, 6))
    for i in range(n):
        j = np.searchsorted(tr['send_t'], i * dt, side='right') - 1
        if j >= 0:
            action[:, 7 * k:7 * k + 6] = torch.as_tensor(tr['send_cmd'][j], device=e.device, dtype=e.q.dtype) + off
        sim[i] = (e.q[0, q] - off).cpu().numpy()
        e.step(action)
    t = np.arange(n) * dt
    return np.column_stack([np.interp(tr['t'], t, sim[:, j]) for j in range(6)])


def tool_mm(side, qa, qb, offsets, m):
    pre = 'left_' if side == 'left' else ''
    qadr = [int(m.joint(pre + f'joint{i}').qposadr[0]) for i in range(1, 7)]
    site, d = m.site(pre + 'grasp_site').id, mujoco.MjData(m)
    out = []
    for a, b in zip(qa, qb):
        pts = []
        for x in (a, b):
            d.qpos[qadr] = x + offsets
            mujoco.mj_kinematics(m, d)
            pts.append(d.site_xpos[site].copy())
        out.append(np.linalg.norm(pts[0] - pts[1]) * 1e3)
    return np.array(out)


def main():
    out = {'scope': 'GPU runtime replay of recorded free-space commands; randomization off; encoder comparison '
                     '(not independent tool metrology)', 'trials': []}
    runtimes = {}
    for side, path, profile in TRIALS:
        tr = I.load_trial(path, side)
        row = {'side': side, 'trial': tr['name'], 'day': tr['day'], 'controller_profile': profile}
        for name, dyn in (('legacy_ideal_gravcomp', None), ('identified', 'results/usbc/bimanual_20261029/dynamics_id/profile.json')):
            key = (profile, dyn)
            if key not in runtimes:
                runtimes[key] = BimanualRuntime(BimanualCfg(nworld=1, controller_profile=profile, dynamics_profile=dyn,
                                                            friction_scale_range=(1.0, 1.0), gravity_scale_jitter=0.0))
            e = runtimes[key]
            sim = replay(e, tr)
            err = tool_mm(side, sim, tr['q_enc'], tr['offsets'], e.mjm)
            row[name] = {'joint_rmse_mrad': (np.sqrt(np.mean((sim - tr['q_enc']) ** 2, 0)) * 1e3).round(2).tolist(),
                         'tool_error_mm_mean': round(float(err.mean()), 2), 'tool_error_mm_p95': round(float(np.percentile(err, 95)), 2)}
        out['trials'].append(row)
        print(f"{side:5s} {tr['name']:34s} legacy {row['legacy_ideal_gravcomp']['tool_error_mm_mean']:6.2f} mm"
              f"  identified {row['identified']['tool_error_mm_mean']:6.2f} mm  joints {row['identified']['joint_rmse_mrad']}")
    path = I.ROOT / 'results/usbc/bimanual_20261029/dynamics_id/gpu_replay.json'
    path.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
