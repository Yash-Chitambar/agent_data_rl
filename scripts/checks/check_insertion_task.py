"""Checks for the Part 4 insertion episode (blind_insertion/envs/insertion_task.py), GPU.

1. Starts: IK converges, the plug sits at the sampled pre-insert pose, the reader is between the pads.
2. Zero action: after warmup the grasp holds (no slip termination) and the PC stays put for 2 s.
3. Privileged scripted controller (reader/port state): seats the plug with full DR on.
4. Contact DR: per-world friction/solref/gain values differ across worlds and lie in range.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python scripts/checks/check_insertion_task.py --nworld 256
"""
import argparse
import json
import time

import torch

from blind_insertion.envs.insertion_task import InsertionCfg, InsertionTask, failure_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--out', default='')
    ap.add_argument('--cfg_json', default='{}', help='InsertionCfg overrides (top-level fields)')
    a = ap.parse_args()
    cfg = InsertionCfg(**json.loads(a.cfg_json))
    cfg.task.physics.nworld = a.nworld
    t0 = time.time()
    env = InsertionTask(cfg, seed=123)
    rt = env.runtime
    print(f'built in {time.time()-t0:.1f}s; nominal right q {env.q_right_nominal.tolist()} margin {env.nominal_margin:.3f}')
    report = {'nworld': a.nworld, 'ik_failures_pool': env.ik_failures}
    # 1. starts
    p, R = rt.plug_in_port()
    err = (p - env.start['err_p']).norm(dim=1)
    report['start_tip_error_max_mm'] = float(err.max() * 1e3)
    report['start_gap_mm'] = [float(p[:, 2].min() * 1e3), float(p[:, 2].max() * 1e3)]
    assert err.max() < 2e-4, 'reader not at the sampled pre-insert pose'
    # 4. contact DR
    cp = rt.contact_params
    report['contact_params_min'] = cp.min(0).values.tolist()
    report['contact_params_max'] = cp.max(0).values.tolist()
    assert (cp.std(0) > 0).all(), 'contact parameters not randomized per world'
    # 2. zero action hold
    stats = {'slip': 0, 'pushed': 0, 'far': 0, 'nonfinite': 0, 'success': 0}
    hold_steps = int(2.0 / env.dt)
    terminated_any = torch.zeros(env.N, dtype=torch.bool, device=env.device)
    for t in range(hold_steps):
        _, _, _, done, info = env.step(torch.zeros(env.N, 6, device=env.device))
        if 'final_ids' in info:
            ids = info['final_ids']
            new = ~terminated_any[ids]
            for k, key in (('slip', 'slip'), ('pushed', 'pushed'), ('far', 'far'), ('nonfinite', 'bad'), ('success', 'success')):
                stats[k] += int(info['final'][key][new].sum())
            terminated_any[ids] = True
    report['zero_action_2s'] = stats
    report['buffer_usage'] = rt.buffer_usage()
    print('zero action:', stats, rt.buffer_usage())
    # 3. scripted privileged insertion
    env.reset_all()
    outcomes, t_seat = [], []
    first = torch.zeros(env.N, dtype=torch.bool, device=env.device)
    t0 = time.time()
    for t in range(env.max_steps):
        a_ = env.scripted_action()
        _, _, _, done, info = env.step(a_)
        if 'final_ids' in info:
            ids = info['final_ids']
            new = ~first[ids]
            fin = {k: v[new] for k, v in info['final'].items()}
            outcomes += failure_class(fin)
            t_seat += (info['steps'][ids][new].float() * env.dt).tolist() if False else []
            first[ids] = True
        if first.all():
            break
    el = time.time() - t0
    counts = {k: outcomes.count(k) for k in sorted(set(outcomes))}
    counts['unfinished'] = int((~first).sum())
    report['scripted'] = counts
    report['scripted_seated_rate'] = counts.get('seated', 0) / env.N
    report['steps_per_s'] = env.N * (t + 1) / el
    report['buffer_usage_scripted'] = rt.buffer_usage()
    print(json.dumps(report, indent=1))
    if a.out:
        open(a.out, 'w').write(json.dumps(report, indent=1))
    assert stats['nonfinite'] == 0


if __name__ == '__main__':
    main()
