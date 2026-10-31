"""Live insertion-start states from the chain (eval_chain.py --save-handover): restore into an InsertionTask,
complete the file (actor observation, geom names), write README.json, and check them with a policy rollout.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python scripts/eval/handover_states.py \
        --states runs/chain/handover/handover_r5_hold020_n4096.npz --policy runs/chain/P4-insert-best_v11.pt

Restore (in a training env, any nworld; states are indexed per world):

    from scripts.eval.handover_states import restore      # or copy the function
    env = InsertionTask(cfg)                               # same scene; render on or off
    z = np.load(path)
    restore(env, z, ids=torch.arange(env.N), rows=np.arange(env.N))
    hist = env.hist                                        # (N, 30, obs_dim): the first actor observation repeated
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import warp as wp  # noqa: E402
wp.config.quiet = True
import mujoco_warp as mjw  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from blind_insertion.envs.insertion_task import InsertionCfg, InsertionTask, failure_class  # noqa: E402
from blind_insertion.eval_insertion import load_insertion_policy, wilson  # noqa: E402

START_KEYS = ('pc', 'q_right', 'reader', 'fingers', 'q_left_hover', 'q_left_press', 'inhand_R', 'inhand_p', 'err_p', 'err_ang')


def geom_map(env, z):
    """Saved geom columns -> this model's geom ids (by name; the chain model has 2 extra visual decal geoms)."""
    m = env.runtime.mjm
    names = [m.geom(g).name for g in range(m.ngeom)]
    if 'geom_names' not in (z.files if hasattr(z, 'files') else z):
        raise KeyError('states file lacks geom_names: run this script once on it (it adds them)')
    saved = list(z['geom_names'])
    idx = {n: i for i, n in enumerate(saved)}
    cols = [idx[n] for n in names]
    return torch.as_tensor(cols, device=env.device)


def restore(env: InsertionTask, z, ids, rows):
    """Exact restore of saved hand-over states into worlds `ids` of an InsertionTask (rows: indices into the file).
    Physics state, control DR (encoder offsets, latency, accumulated target incl. the torque hand-over offset, latency
    buffer), the left hold target and the per-world physics/contact DR are all restored; the episode bookkeeping is
    reset as InsertionTask.reset does (steps 0, warm-up, history = the first observation repeated)."""
    rt, dev = env.runtime, env.device
    t = lambda k: torch.as_tensor(z[k][rows], device=dev)   # noqa: E731
    rt.q[ids] = t('qpos'); rt.v[ids] = t('qvel'); rt.ctrl[ids] = t('ctrl')
    wp.to_torch(rt.d.qacc_warmstart)[ids] = t('qacc_warmstart')
    cols = geom_map(env, z)
    if hasattr(rt, 'frictionloss'):
        rt.frictionloss[ids] = t('dof_frictionloss'); rt.gravity_scale[ids] = t('gravity_scale')
    if hasattr(rt, 'geom_friction'):
        rt.geom_friction[ids[:, None], torch.arange(len(cols), device=dev)[None], 0] = t('geom_friction')[:, cols]
    if hasattr(rt, 'geom_solref'):
        rt.geom_solref[ids[:, None], torch.arange(len(cols), device=dev)[None], 0] = t('geom_solref')[:, cols]
    if hasattr(rt, 'actuator_gainprm'):
        rt.actuator_gainprm[ids, :, 0] = t('actuator_gainprm')
    if hasattr(rt, 'contact_params'):
        rt.contact_params[ids] = t('contact_params')
    env.enc_error[ids] = t('enc_error'); env.latency[ids] = t('latency').long()
    env.joint_target[ids] = t('joint_target')
    env.sent_history[:, ids] = t('sent_history').transpose(0, 1)
    env.left_press_q[ids] = t('left_press_q')
    with wp.ScopedDevice(rt.cfg.device):
        mjw.forward(rt.m, rt.d)
    env.steps[ids] = 0; env.seated_steps[ids] = 0; env.done[ids] = False; env.previous_action[ids] = 0
    env.pc_start[ids] = t('pc'); env.captured[ids] = False; env.strict_run[ids] = 0; env.strict_best[ids] = 0
    env.ref_rel[ids] = env._reader_in_tcp()[ids]
    for k in ('err_p', 'err_ang'):
        if k in env.start:
            env.start[k][ids] = t(k)
    env.phi[ids] = env._potential()[ids]
    o = env._actor_obs()
    env.hist[ids] = o[ids, None].expand(-1, env.icfg.history, -1)
    return o[ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', required=True)
    ap.add_argument('--policy', default='runs/chain/P4-insert-best_v11.pt')
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--geom-names-from-chain', action='store_true', help='add geom_names (chain model incl. visual decals)')
    a = ap.parse_args()
    path = ROOT / a.states
    z = dict(np.load(path))
    if 'geom_names' not in z:   # the chain runtime renders: visual-DR assets included
        from blind_insertion.envs import visual_dr as VD
        from blind_insertion.scenes import pc_bench as S
        prof = json.loads((ROOT / 'data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
        m, _ = S.build_scene(detailed=True, bimanual=True, measured_support=True, support_mode='free', profile=prof,
                             extra_spec=VD.add_assets)
        assert m.ngeom == z['geom_friction'].shape[1]
        z['geom_names'] = np.array([m.geom(g).name for g in range(m.ngeom)])
    pol = load_insertion_policy(str(ROOT / a.policy), 'cuda:0')
    cfg = InsertionCfg()
    for k, v in pol.meta['env_json'].items():
        tgt = cfg if hasattr(cfg, k) else cfg.task if hasattr(cfg.task, k) else cfg.task.physics
        setattr(tgt, k, tuple(v) if isinstance(v, list) else v)
    cfg.task.physics.nworld = a.nworld
    env = InsertionTask(cfg, seed=1)
    n_all = len(z['qpos'])
    obs_all = np.zeros((n_all, env.obs_dim), np.float32)
    rows_out = []
    for b0 in range(0, n_all, a.nworld):
        rows = np.arange(b0, min(b0 + a.nworld, n_all))
        ids = torch.arange(len(rows), device=env.device)
        env.reset_all()
        o = restore(env, z, ids, rows)
        obs_all[rows] = o.cpu().numpy()
        first = torch.zeros(env.N, dtype=torch.bool, device=env.device)
        succ = torch.zeros_like(first); cls = ['timeout'] * env.N
        for _ in range(env.max_steps):
            _, _, _, done, info = env.step(pol(env.hist))
            if 'final' in info:
                c = failure_class(info['final'])
                for j, w in enumerate(info['final_ids'].tolist()):
                    if not first[w]:
                        cls[w] = c[j]
                first |= done
            if first[:len(rows)].all():
                break
        rows_out += cls[:len(rows)]
        print(f'rows {rows[0]}-{rows[-1]}: seated {sum(c_ == "seated" for c_ in cls[:len(rows)])}/{len(rows)}', flush=True)
    k = sum(c_ == 'seated' for c_ in rows_out)
    lo, hi = wilson(k, n_all)
    z['actor_obs'] = obs_all
    np.savez_compressed(path, **z)
    shapes = {k_: [list(v.shape), str(v.dtype)] for k_, v in z.items()}
    readme = {
        'path': str(path),
        'what': 'Live insertion-start states of the end-to-end chain (scripts/eval/eval_chain.py, policy/chain): one '
                'continuous simulation per world (Part 1 scripted pick, Part 3 guarded 25 N press 2 cm in front of the case '
                'centre, Cartesian transport, Part 2 round-5 wrist servo wrist_v12w/v12ws with the chain stop offset, '
                'bumpless torque hand-over), stopped at the first insertion step. Full DR per world (layout, arm dynamics, '
                'latency, encoder offsets, PC feet / reader / port / pad contact, visual DR during perception).',
        'n': n_all, 'seeds': sorted(set(int(s_) for s_ in z['seed'])),
        'fields': {
            'InsertionTask.reset(ids, starts) keys': list(START_KEYS),
            'pc, reader': 'free-joint qpos (pos, quat wxyz), world',
            'q_right, q_left_hover': 'true joint angles (model frame, rad)',
            'q_left_press': 'force offset such that q_left_hover - enc_error[:, 1] + q_left_press = left hold target (believed frame), as InsertionTask guarded hold',
            'fingers': 'right finger joint q (m)',
            'inhand_R, inhand_p': 'reader in the right TCP frame (x_tcp = R x_reader + p)',
            'err_p, err_ang': 'plug tip in the live port frame (m) and rotation vector (rad; InsertionTask samples rpy: equal to first order)',
            'exact restore': 'qpos (nq), qvel (nv), ctrl (nu), qacc_warmstart, enc_error (N,2,6 rad: right, left), latency (steps), '
                             'joint_target (N,2,6 believed accumulated target incl. the torque hand-over offset tau_ff/kp), '
                             'sent_history (N,L,14 physical commands in the latency buffer), left_press_q (believed hold target)',
            'per-world DR': 'dof_frictionloss (nv), gravity_scale (N,2,6), geom_friction / geom_solref (column 0, chain model geoms: '
                            'map by geom_names; the chain model has 2 visual decal geoms more than a non-rendering one), '
                            'actuator_gainprm (column 0), contact_params (port_friction, port_timeconst, pad_friction, pad_timeconst, grip_gain_scale)',
            'actor_obs': 'first 62-D actor observation at the insertion start (InsertionTask._actor_obs, unmasked); the actor history '
                         'is this repeated 30 times (as at an InsertionTask reset); previous actions are zero',
            'outcome, e2e_class': 'not_run (the chain stopped at the hand-over); e2e_class = success unless an earlier stage failed',
            'seed': 'chain batch seed'},
        'shapes': shapes,
        'snippet': ['import numpy as np, torch, sys; sys.path.insert(0, "scripts/eval")',
                    'from handover_states import restore',
                    'z = np.load(PATH)',
                    'env = InsertionTask(cfg)      # e.g. make_env(...) of train_bimanual.py',
                    'rows = np.random.randint(len(z["qpos"]), size=env.N)',
                    'restore(env, z, torch.arange(env.N, device=env.device), rows)   # exact state + DR + bookkeeping; env.hist ready',
                    '# or, approximate (InsertionTask samples its own DR/control): env.reset(ids, {k: torch.as_tensor(z[k][rows]).cuda() for k in START_KEYS})'],
        'check': {'policy': a.policy, 'seated': k, 'n': n_all, 'rate': k / n_all, 'wilson95': [lo, hi],
                  'outcomes': {c_: rows_out.count(c_) for c_ in set(rows_out)},
                  'note': 'policy rolled out from the restored states in a fresh InsertionTask (no rendering)'}}
    (path.parent / 'README.json').write_text(json.dumps(readme, indent=1) + '\n')
    print(json.dumps(readme['check'], indent=1))


if __name__ == '__main__':
    main()
