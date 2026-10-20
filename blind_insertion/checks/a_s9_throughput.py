"""A-S9: env steps/s vs number of worlds (random actions, auto-resets on)."""
import argparse
import json
import time

import torch

from blind_insertion.envs.peg_env import EnvCfg, PegEnv


def bench(n, robot="flexiv", steps=50, nvar=None, **kw):
    env = PegEnv(EnvCfg(robot=robot, nworld=n, n_variants=nvar or min(n, 1024), **kw))
    a = torch.zeros(n, env.n_arm, device="cuda")
    for _ in range(5):
        env.step(a)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(steps):
        a = 0.5 * torch.randn(n, env.n_arm, device="cuda")
        env.step(a)
    torch.cuda.synchronize()
    dt = time.time() - t
    sps = n * steps / dt
    ncon = env.ncon().item() / n
    del env
    torch.cuda.empty_cache()
    return sps, ncon


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="flexiv")
    ap.add_argument("--ns", default="1024,4096,8192")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = []
    for n in [int(x) for x in args.ns.split(",")]:
        sps, ncon = bench(n, args.robot)
        print(f"N={n}: {sps:,.0f} env steps/s ({sps*20:,.0f} physics steps/s), contacts/world {ncon:.1f}",
              flush=True)
        rows.append(dict(nworld=n, env_steps_per_s=sps, contacts_per_world=ncon))
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
