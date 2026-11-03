"""K-J on the real in-hand grasp: success against hand-over offset, old and new in-hand distribution.

Hand-over: plug at (offset x, offset y, gap) in the port frame, no rotation error unless --angle. K-J's own env otherwise.
Also logs what the push does to the grasp: reader pitch in the jaws (about the closing axis), slip and finger opening
against the values at the end of the warm-up, the sim's counterpart of the real grip reading. Resumable.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python scripts/eval/eval_kj_inhand.py --out results/real/inhand_real/kj_inhand.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.insertion_task import failure_class
from blind_insertion.eval_insertion import InsertionPolicy, wilson
from blind_insertion.train_bimanual import make_env

OFFSETS = [(0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (0, 1), (0, -1), (0, 2), (0, -2)]
# In-hand input: the real net's mean on the three real grasps it saw (inhand_ih5, 2026-11-02 full_A, full_A2, collect1)
NET_OBS = (3.63, 0.01, -2.00, -0.65, -0.75, 1.00)


def run_cell(env, ens, seed):
    torch.manual_seed(seed)
    env.refresh_pool()
    env.reset_all()
    hist, _ = env.observe()
    N, dev = env.N, env.device
    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    rows = []
    rt = env.runtime
    fq = [int(rt.mjm.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
    pitch_max = torch.zeros(N, device=dev)
    slip_max = torch.zeros(N, device=dev)
    fing0 = torch.full((N,), float('nan'), device=dev)
    fing_drop = torch.zeros(N, device=dev)
    R0 = None
    for t in range(env.max_steps + 10):
        with torch.no_grad():
            act = ens(hist)
        hist, _, _, done, info = env.step(act)
        T = env._reader_in_tcp()
        dR = T[:, :3, :3] @ env.ref_rel[:, :3, :3].transpose(1, 2)
        rv = K.rot_log(dR) * 180 / np.pi                 # on the TCP axes; y = the closing axis
        dp = (T[:, :3, 3] - env.ref_rel[:, :3, 3]).norm(dim=1) * 1e3
        warm = (env.steps > env.icfg.warmup_steps) & ~done_once
        pitch_max = torch.where(warm, torch.maximum(pitch_max, rv[:, 1].abs()), pitch_max)
        slip_max = torch.where(warm, torch.maximum(slip_max, dp), slip_max)
        f = rt.q[:, fq].mean(1)
        first = warm & torch.isnan(fing0)
        fing0 = torch.where(first, f, fing0)
        fing_drop = torch.where(warm, torch.maximum(fing_drop, (fing0 - f) * 1e3), fing_drop)
        if 'final' in info:
            ids = info['final_ids']
            cls = failure_class(info['final'])
            for j, w in enumerate(ids.tolist()):
                if not done_once[w]:
                    rows.append((w, cls[j], float(pitch_max[w]), float(slip_max[w]), float(fing_drop[w])))
            done_once[ids] = True
        if done_once.all():
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='runs/chain/P4-insert-best_KJ.pt')
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--gap_mm', type=float, default=6.)
    ap.add_argument('--angle', type=float, default=0.)
    ap.add_argument('--envs', nargs='+', default=['realistic', 'base'])
    ap.add_argument('--out', default='results/real/inhand_real/kj_inhand.csv')
    ap.add_argument('--preset', default='v1', help='real grasp preset; other than v1 the grasp column reads real_<preset>')
    ap.add_argument('--only_exact', action='store_true')
    ap.add_argument('--grasps', default='real:true,real:net,part1:true', help='grasp:obs cells')
    a = ap.parse_args()
    dev = 'cuda:0'
    ens = InsertionPolicy(a.policy, dev)
    ck = ens.members[0][1]
    ej = json.loads(ck['env_json'])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            have.add((r['env'], r['grasp'], r['obs'], r['ox'], r['oy'], r['gap_mm'], r['angle']))
    new = not out.exists()
    fh = open(out, 'a', newline='')
    wr = None
    for name in a.envs:
        over = dict(ej, pool_per_world=1, start_source='uniform', live_frac=0., live_states='', lateral_m=[0., 0.], angle_deg=a.angle, gap_m=[a.gap_mm * 1e-3, a.gap_mm * 1e-3])
        if name == 'realistic':
            over.update(grasp_fidelity='v2_real')
        args = argparse.Namespace(nworld=a.nworld, seed=1000, action_mode=ck.get('action_mode', 'accumulate'), env_json=json.dumps(over))
        env = make_env(args)
        print('env', name, 'obs', env.obs_dim, flush=True)
        for grasp, obs in (x.split(':') for x in a.grasps.split(',')):
            for ox, oy in (OFFSETS[:1] if a.only_exact else OFFSETS):
                label = grasp if grasp == 'part1' or a.preset == 'v1' else f'{grasp}_{Path(a.preset).stem}'
                key = (name, label, obs, str(ox), str(oy), str(a.gap_mm), str(a.angle))
                if key in have:
                    continue
                c = env.icfg
                c.grasp, c.grasp_real, c.start_offset_m = grasp, a.preset, (ox * 1e-3, oy * 1e-3)
                c.inhand_extra_frac = ej.get('inhand_extra_frac', 1.) if grasp == 'part1' else 0.
                c.inhand_obs_fixed = NET_OBS if obs == 'net' else None
                t0 = time.time()
                rows = run_cell(env, ens, 1000 + sum(map(ord, "".join(key))) % 1000)
                cls = [r[1] for r in rows]
                n = len(rows)
                k = sum(x == 'seated' for x in cls)
                lo, hi = wilson(k, n)
                ok = [r for r in rows if r[1] == 'seated']
                count = {x: cls.count(x) for x in sorted(set(cls))}
                row = dict(env=name, grasp=label, obs=obs, ox=ox, oy=oy, gap_mm=a.gap_mm, angle=a.angle, n=n, seated=k, rate=round(k / max(n, 1), 4), lo=round(lo, 4), hi=round(hi, 4),
                           classes=json.dumps(count), pitch_med_deg=round(float(np.median([r[2] for r in rows])), 3),
                           pitch_p90_deg=round(float(np.quantile([r[2] for r in rows], .9)), 3), slip_med_mm=round(float(np.median([r[3] for r in rows])), 3),
                           slip_p90_mm=round(float(np.quantile([r[3] for r in rows], .9)), 3), finger_drop_med_mm=round(float(np.median([r[4] for r in rows])), 3),
                           finger_drop_p90_mm=round(float(np.quantile([r[4] for r in rows], .9)), 3))
                if wr is None:
                    wr = csv.DictWriter(fh, fieldnames=list(row))
                    if new:
                        wr.writeheader()
                wr.writerow(row)
                fh.flush()
                print(row, f'{time.time() - t0:.0f}s', flush=True)
        del env
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
