"""The real insertion observation (deploy/bimanual/insert.InsertObs: encoders, joint targets, previous action,
in-hand estimate, chain constants from the bundle) equals InsertionTask._actor_obs for the same state.

InsertionTask worlds (K-J's env_json via eval_chain's Chain) are reset to random starts (encoder-offset error,
latency, quantized encoders, masked left joint 6) and stepped with random actions; every step, per world, the
real builder gets what the robot would read (encoder reading = sim encoder - offset error, i.e. the motor frame;
velocities; the controller's joint targets and previous action; the in-hand input) and must reproduce the env's
68-D observation. The live dry run repeats this every insertion tick (run_bimanual --check-obs).

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_obs [--nworld 8 --steps 40]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nworld', type=int, default=8)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    from deploy.bimanual.io_sim import chain_argv, eval_chain
    from deploy.bimanual.insert import InsertObs
    from deploy.bimanual.scene import SceneModel
    from blind_insertion.envs.insertion_task import InsertionTask
    cfg = json.loads((ROOT / 'deploy/config/bimanual.json').read_text())
    consts = json.loads((ROOT / cfg['bundle_dir'] / 'configs/chain_constants.json').read_text())
    EC = eval_chain()
    ch = EC.Chain(EC.parse(chain_argv(cfg, ROOT / cfg['bundle_dir'] / 'models/insert_KJ.pt', a.nworld)))
    env, rt = ch.env, ch.rt
    InsertionTask.reset(env, None)
    sc = SceneModel('cpu')
    ob = InsertObs(sc, consts, torch.device('cpu'))
    g = torch.Generator(device=env.device).manual_seed(0)
    env.inhand_est.copy_(torch.randn(a.nworld, 6, device=env.device, generator=g))
    worst, n = 0.0, 0
    for t in range(a.steps):
        act = torch.rand(a.nworld, 6, device=env.device, generator=g) * 2 - 1
        env.step(act)
        o_env = env._actor_obs().cpu()
        pr = rt.proprioception()
        enc = (pr['encoder_rad'].reshape(-1, 2, 6) - env.enc_error).cpu()
        vel = pr['encoder_velocity_rad_s'].reshape(-1, 2, 6).cpu()
        for w in range(a.nworld):
            o = ob(enc[w], vel[w], env.previous_action[w].cpu(), env.joint_target[w].cpu(), env.inhand_est[w].cpu())
            worst = max(worst, float((o[0] - o_env[w]).abs().max()))
            n += 1
    ok = worst < 1e-4
    res = dict(ok=ok, compared=n, obs_dim=int(o_env.shape[1]), max_abs_diff=worst, nworld=a.nworld, steps=a.steps,
               enc_error_deg_range=float(torch.rad2deg(env.enc_error.abs().max())), latency=env.latency.tolist())
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
