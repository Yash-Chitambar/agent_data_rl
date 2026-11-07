"""Replay real insertion hand-overs in the sim ("learn from real failures": docs/real/K2_PROGRESS.md, "Real-to-sim replay and roll").

Input: rows of scripts/real/real_handover_states.py (results/real_align/replay/real_handovers.json). For each run, n worlds start
from that run's hand-over and the policy runs on a real-like plant (friction x1.3-2.0, gravity error 12 %, real velocity noise).

Start (true state, per world): the run's vision measurement + N(0, VISION_SIGMA) (vision is the only measurement; its error
is the pose-dependent + run error of port_vision: sigma (0.4, 0.95, 0.63) mm), angles = the run's last rv_deg (median of
3 cycles) + N(0, 0.5 deg) when the run had an align stage, else the chain pool's angles (unknown). Joint target lead = the run's
logged lead (obs[38:44] at t = 0). In-hand estimate input = the run's logged inhand6 (both policies).
Observation, per policy:
  * absolute-pose policies (K-J): obs[50:56] set to the run's logged values (the real FK offsets), as kj_obs_shift.py;
  * rel_pose_obs policies (K2/K3): relative to the hand-over by construction (nothing to set);
  * vision_prior_obs policies (K3): the prior = the run's vision measurement exactly (what the robot would feed); the true
    start differs from it by the vision error above, as on the robot.

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/replay_real_handovers.py --policy deploy/bundle/models/insert_KJ.pt \
      --tag kj --out results/real_align/replay/replay_kj.json [--runs full_175030_align_full10,...] [--nworld 256]
New runs: add them with real_handover_states.py RUN_DIR --out <file> and pass --states <file>.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.eval.kj_start_map as KSM  # noqa: E402
from scripts.eval.kj_start_map import rows_base, run_cond  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402

VISION_SIGMA = np.array([0.4, 0.95, 0.63])
ANG_SIGMA = 0.5
REAL_PLANT = {'friction_scale_range': [1.3, 2.0], 'gravity_scale_jitter': 0.12, 'vel_obs_noise': [0.006, 0.003, 0.003, 0.018, 0.016, 0.028]}


def make_rows(st, base, g, n=4096, vision_noise=True, angles=True):
    r = base[g.integers(0, len(base), n)].copy()
    v = np.array(list(st['lat_mm']) + [st['gap_mm']], float)
    r[:, 0:3] = v[None] + (g.normal(0, 1, (n, 3)) * VISION_SIGMA if vision_noise else 0)
    r[:, 2] = np.clip(r[:, 2], 1.0, 20.0)
    if angles and st.get('rv_deg_last3_median') is not None:
        r[:, 3:6] = np.asarray(st['rv_deg_last3_median'], float)[None] + g.normal(0, ANG_SIGMA, (n, 3))
    return r


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--states', default=str(ROOT / 'results/real_align/replay/real_handovers.json'))
    ap.add_argument('--runs', default='', help='comma list of run names (default: every row with a vision measurement)')
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--seed', type=int, default=6100)
    ap.add_argument('--plant', default='real', choices=('real', 'train'))
    ap.add_argument('--no-real-obs', action='store_true', help='K-J: keep the sim\'s own pose features (no real FK offsets)')
    ap.add_argument('--no-vision-noise', action='store_true', help='true start = the vision measurement exactly')
    ap.add_argument('--no-inhand-obs', action='store_true', help='keep the sim\'s own in-hand estimate input (not the run\'s logged one)')
    ap.add_argument('--angles', default='run', choices=('run', 'pool'), help='run: the run\'s last rv_deg + noise; pool: the chain pool\'s angles')
    ap.add_argument('--env_extra', default='{}', help='e.g. {"grasp": "real"} for the measured real grasp')
    ap.add_argument('--real-left-lead', action='store_true', help='observation only: left joint target - measured (obs[44:49]) = the run\'s logged value')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    rows = [r for r in json.loads(Path(A.states).read_text()) if r.get('lat_mm') is not None and r.get('gap_mm') is not None]
    if A.runs:
        keep = set(A.runs.split(','))
        rows = [r for r in rows if r['run'] in keep]
    out = Path(A.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else []
    seen = {d['run'] for d in done}
    torch.manual_seed(A.seed)
    pol = InsertionPolicy(A.policy, 'cuda:0')
    ej0 = pol.meta['env_json']
    rel, prior = bool(ej0.get('rel_pose_obs')), bool(ej0.get('vision_prior_obs'))
    ej = dict(ej0, hard_frac=0., scale_range_frac=0., live_frac=0., extra_ang_frac=0., handover_lead_frac=0.)
    if A.plant == 'real':
        ej.update(REAL_PLANT)
    ej.update(json.loads(A.env_extra))
    env = make_env(argparse.Namespace(nworld=A.nworld, seed=A.seed, action_mode='accumulate', env_json=json.dumps(ej)))
    dev = env.device
    base_obs = env._actor_obs
    state = {'shift': None}

    def shifted():
        o = base_obs()
        return o if state['shift'] is None else o + state['shift']
    env._actor_obs = shifted
    base = rows_base()
    _cr = KSM.cond_rows
    for i, st in enumerate(rows):
        if st['run'] in seen:
            continue
        KSM.cond_rows = lambda c, b, g, st=st: make_rows(st, b, g, vision_noise=not A.no_vision_noise, angles=A.angles == 'run')  # noqa: E731

        def hook(kind, e, act=None, st=st):
            if kind != 'reset':
                return act
            q0 = e._measured_model_angles()[:, 0]
            if st.get('lead0_mrad') is not None:
                e.joint_target[:, 0] = q0 + torch.tensor(st['lead0_mrad'], dtype=q0.dtype, device=dev) * 1e-3
            if prior:
                e.vision_prior[:] = torch.tensor(list(st['lat_mm']) + [st['gap_mm']], dtype=torch.float32, device=dev)
            state['shift'] = None
            o = base_obs()
            s = torch.zeros_like(o)
            if st.get('inhand6') is not None and not A.no_inhand_obs:
                s[:, 62:68] = torch.tensor(st['inhand6'], dtype=o.dtype, device=dev) - o[:, 62:68]
            if not rel and not A.no_real_obs and st.get('fk_tcp_obs') is not None:
                s[:, 50:53] = torch.tensor(st['fk_tcp_obs'], dtype=o.dtype, device=dev) - o[:, 50:53]
                s[:, 53:56] = torch.tensor(st['fk_eul_obs'], dtype=o.dtype, device=dev) - o[:, 53:56]
            if A.real_left_lead and st.get('left_lead0_mrad') is not None:
                s[:, 44:49] = torch.tensor(st['left_lead0_mrad'][:5], dtype=o.dtype, device=dev) * 1e-3 - o[:, 44:49]
            state['shift'] = s
            e.hist[:] = (o + s)[:, None]
            return None
        t0 = time.time()
        r = run_cond(env, pol, {'real': st['run']}, base, A.seed + i, on_step=hook)
        r.update(run=st['run'], tag=A.tag, policy=A.policy, plant=A.plant, truth_seated=st.get('truth_seated'), outcome_log=st.get('outcome_log'),
                 real_plant=st.get('plant'), vision=[*st['lat_mm'], st['gap_mm']], angles=st.get('rv_deg_last3_median'),
                 real_obs=(not rel and not A.no_real_obs), vision_noise=not A.no_vision_noise, inhand_obs=not A.no_inhand_obs,
                 angles_src=A.angles, env_extra=json.loads(A.env_extra), real_left_lead=A.real_left_lead)
        done.append(r)
        out.write_text(json.dumps(done, indent=1))
        print(A.tag, st['run'], f"truth {st.get('truth_seated')}", f"{r['k']}/{r['n']} = {100 * r['rate']:.1f} % [{100 * r['wilson'][0]:.1f}, {100 * r['wilson'][1]:.1f}]",
              {k: v for k, v in r['classes'].items() if k != 'seated'}, f'{time.time() - t0:.0f} s', flush=True)
    KSM.cond_rows = _cr
