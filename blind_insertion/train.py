"""PPO with an asymmetric critic and contact supervision (paper §2.3, App. A.1).

Usage: python -m blind_insertion.train --name B-peg-proprio-base-s0 --seed 0 --steps 300e6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.envs.peg_env import EnvCfg, PegEnv
from blind_insertion.policies.tcn_actor_critic import ActorCritic

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("BI_RUNS_DIR", ROOT / "runs"))   # e.g. cluster scratch


def save_atomic(obj, path: Path):
    """torch.save via a temp file + rename, so a kill mid-write never leaves a truncated file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "nogit"


class PPOCfg:
    rollout = 32
    epochs = 5
    minibatches = 20
    lr = 3e-4
    clip = 0.2
    gamma = 0.98
    lam = 0.95
    ent = 0.003
    vf = 1.0
    max_grad = 1.0
    contact_coef = 1.0
    class_w = (0.25, 3.0, 2.0)


# USB-C curriculum (plan §11.3): (clearance per side, median start offset)
USBC_STAGES = [(3.0e-4, 5e-4), (1.5e-4, 1.0e-3), (8e-5, 1.5e-3), (5e-5, 2.0e-3), ((2e-5, 1e-4), 3.0e-3)]
MEDIAN_RADIAL_PER_HALFWIDTH = 0.765   # median |(x,y)| for x,y ~ U(-a, a) is ~0.765 a


def make_variants(args):
    from blind_insertion.assets import builder as B  # noqa: PLC0415
    if getattr(args, "task", "peg") == "gear":
        from blind_insertion.assets import gear_builder as GB  # noqa: PLC0415
        if args.family == "study":
            return GB.sample_study_gears(args.n_variants, args.variant_seed)
        if args.family.startswith("ood:"):
            return GB.sample_ood_gears(args.n_variants, args.variant_seed, args.family[4:])
        raise ValueError(args.family)
    if args.family == "study":
        return B.sample_study_pegs(args.n_variants, args.variant_seed)
    if args.family.startswith("scale:"):     # scale:<opening m>:<clearance m>
        _, o, c = args.family.split(":")
        return B.sample_scale_cell(args.n_variants, args.variant_seed, float(o), float(c))
    if args.family.startswith("ood:"):
        return B.sample_ood_pegs(args.n_variants, args.variant_seed, args.family[4:])
    if args.family == "usbc":
        vkw = json.loads(getattr(args, "variant_kw", "") or "{}")   # e.g. {"chamfer": [2e-4, 5e-4]}
        if getattr(args, "curriculum", "none") == "usbc":
            per = args.n_variants // len(USBC_STAGES)
            vs = []
            for i, (c, _) in enumerate(USBC_STAGES):
                cr = c if isinstance(c, tuple) else (c, c)
                vs += B.sample_usbc(per, args.variant_seed + i, clearance=cr, **vkw)
            return vs
        return B.sample_usbc(args.n_variants, args.variant_seed, **vkw)
    raise ValueError(args.family)


def make_env(args, **over) -> PegEnv:
    kw = dict(robot=args.robot, nworld=args.nworld, seed=args.seed, n_variants=args.n_variants,
              variant_seed=args.variant_seed)
    if args.env_file:
        kw.update(json.load(open(args.env_file)))
    kw.update(json.loads(args.env_json) if args.env_json else {})
    kw.update(over)
    kw["variants"] = make_variants(args)
    if getattr(args, "task", "peg") == "gear":
        from blind_insertion.envs.gear_env import GearCfg, GearEnv  # noqa: PLC0415
        return GearEnv(GearCfg(**kw))
    return PegEnv(EnvCfg(**kw))


class Curriculum:
    """Advance a stage when rolling training success >= thr over `window` env steps."""

    def __init__(self, env, stages, thr=0.9, window=5e6):
        self.env, self.stages, self.thr, self.window = env, stages, thr, window
        self.stage = -1
        self.per = env.nK // len(stages)
        self.hist = []      # (env_steps, successes, episodes)
        self.log = []
        self.advance(0)

    def advance(self, total):
        self.stage += 1
        c, med = self.stages[self.stage]
        env = self.env
        base = self.stage * self.per
        env.set_variants(base + torch.arange(env.N, device=env.device) % self.per)
        env.cfg.peg_offset = med / MEDIAN_RADIAL_PER_HALFWIDTH
        env.refresh_pool()
        env.reset_all(desync=True)
        self.hist = []
        self.log.append(dict(stage=self.stage + 1, env_steps=total, clearance=str(c), median_offset=med))
        print(f"[curriculum] stage {self.stage + 1} at {total/1e6:.1f}M steps: clearance {c}, "
              f"median offset {1000*med:.1f} mm", flush=True)

    def update(self, total, k, n):
        self.hist.append((total, k, n))
        recent = [h for h in self.hist if h[0] > total - self.window]
        span = total - (recent[0][0] if recent else total)
        K_, N_ = sum(h[1] for h in recent), sum(h[2] for h in recent)
        if self.stage < len(self.stages) - 1 and span >= self.window * 0.95 and N_ > 0 and K_ / N_ >= self.thr:
            self.advance(total)
            return True
        return False


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    if (run / "DONE").exists() and args.resume:   # finished: skip, unless asked for more steps
        done = json.loads((run / "DONE").read_text())
        if done.get("reason") == "stop_success" or done.get("steps", 0) >= args.steps:
            print(f"[{args.name}] already DONE: {done}", flush=True)
            return
        (run / "DONE").unlink()
    env = make_env(args)
    dev = env.device
    cur = Curriculum(env, USBC_STAGES) if args.curriculum == "usbc" else None
    pc = PPOCfg()
    pc.contact_coef = args.contact_coef
    task = getattr(args, "task", "peg")
    if args.minibatches:
        pc.minibatches = args.minibatches
    elif task == "gear":
        pc.minibatches = 24     # paper App. A.1 (gear and nut)
    n_lab = getattr(env, "n_labels", 64)
    lab_fn = getattr(env, "current_labels", None)
    ac = ActorCritic(env.obs_dim, env.n_arm, env.critic_dim, history=env.cfg.history,
                     encoder=args.encoder, n_points=n_lab).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=pc.lr)
    meta = dict(args=vars(args), env=asdict(env.cfg) | {"variants": None, "variant_sampler": None},
                git=git_hash(), versions={p: version(p) for p in ("mujoco", "mujoco-warp", "warp-lang",
                                                                   "torch", "mjfit")})
    resuming = args.resume and (run / "last.pt").exists()
    json.dump(meta, open(run / ("config_resume.json" if resuming else "config.json"), "w"), indent=1, default=str)

    N, T, H, D, A = env.N, pc.rollout, env.cfg.history, env.obs_dim, env.n_arm
    buf = dict(hist=torch.zeros(T, N, H, D, device=dev), crit=torch.zeros(T, N, env.critic_dim, device=dev),
               act=torch.zeros(T, N, A, device=dev), logp=torch.zeros(T, N, device=dev),
               val=torch.zeros(T, N, device=dev), rew=torch.zeros(T, N, device=dev),
               done=torch.zeros(T, N, device=dev), tout=torch.zeros(T, N, device=dev),
               lab=torch.zeros(T, N, n_lab, dtype=torch.long, device=dev))
    env.reset_all(desync=True)
    hist, crit = env.observe()
    cw = torch.tensor(pc.class_w, device=dev)
    total, it, best, wall0 = 0, 0, -1, 0.0
    if resuming:
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        ac.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        it, total, best, wall0 = ck["iter"], ck["steps"], ck.get("best", -1), ck.get("wall", 0.0)
        stage = ck.get("stage")
        if cur is not None and not stage and (run / "curriculum.json").exists():   # older checkpoints
            stage = max(r["stage"] for r in json.load(open(run / "curriculum.json"))
                        if r["env_steps"] <= total)
        if cur is not None and stage:
            while cur.stage + 1 < stage:
                cur.advance(total)
            if (run / "curriculum.json").exists():
                cur.log = [r for r in json.load(open(run / "curriculum.json")) if r["stage"] <= stage]
            env.reset_all(desync=True)
        if "rng" in ck:
            torch.set_rng_state(ck["rng"]["torch"].cpu())
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu(), dev)
            env.gen.set_state(ck["rng"]["env"].cpu())
        hist, crit = env.observe()
        print(f"[{args.name}] resumed at it {it}, {total/1e6:.1f}M steps"
              + (f", stage {cur.stage + 1}" if cur is not None else ""), flush=True)
    log_path = run / "train_log.csv"
    log = None
    if resuming and log_path.exists() and log_path.stat().st_size:
        with open(log_path) as f_:
            fields = next(csv.reader(f_))
        rows = [r for r in csv.DictReader(open(log_path)) if int(r["iter"]) <= it]   # drop rows past last.pt
        log_f = open(log_path, "w", newline="")
        log = csv.DictWriter(log_f, fieldnames=fields)
        log.writeheader()
        log.writerows(rows)
    else:
        log_f = open(log_path, "w", newline="")
    t0 = time.time() - wall0
    ep_stats = {"succ": [], "len": [], "cap": []}
    reason, succ = "steps", float("nan")

    def ckpt():
        return {"model": ac.state_dict(), "iter": it, "steps": total, "obs_dim": env.obs_dim,
                "act_dim": env.n_arm, "critic_dim": env.critic_dim, "encoder": args.encoder, "task": task,
                "n_points": n_lab, "opt": opt.state_dict(), "best": best, "wall": time.time() - t0,
                "stage": (cur.stage + 1) if cur is not None else None,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev),
                        "env": env.gen.get_state()}}
    while total < args.steps:
        it += 1
        t_roll = time.time()
        ac.eval()
        with torch.no_grad():
            for t in range(T):
                ac.obs_norm.update(hist[:, -1])
                ac.crit_norm.update(crit)
                d, _ = ac.dist(hist)
                a = d.sample()
                buf["hist"][t] = hist
                buf["crit"][t] = crit
                buf["act"][t] = a
                buf["logp"][t] = d.log_prob(a).sum(-1)
                buf["val"][t] = ac.value(hist, crit)
                # labels of the state the latent encodes (before stepping)
                if lab_fn is not None:
                    buf["lab"][t] = lab_fn()
                else:
                    p, R, _, _ = env.peg_in_fixture()
                    buf["lab"][t] = env.labels(env.edge_points_fixture(p, R))[0]
                hist, crit, r, done, info = env.step(a)
                buf["rew"][t] = r
                buf["done"][t] = done.float()
                buf["tout"][t] = info["timeout"].float()
                if done.any():
                    ep_stats["succ"].append(info["success"][done].float().cpu())
                    ep_stats["len"].append(info["steps"][done].float().cpu())
                    ep_stats["cap"].append(info["captured"][done].float().cpu())
            last_val = ac.value(hist, crit)
        total += T * N
        env.refresh_pool()
        roll_time = time.time() - t_roll

        # GAE (bootstrap timeouts with V(s_t), as rsl_rl)
        with torch.no_grad():
            rew = buf["rew"] + pc.gamma * buf["val"] * buf["tout"]
            adv = torch.zeros_like(rew)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_val if t == T - 1 else buf["val"][t + 1]
                nd = 1.0 - buf["done"][t]
                delta = rew[t] + pc.gamma * nv * nd - buf["val"][t]
                last = delta + pc.gamma * pc.lam * nd * last
                adv[t] = last
            ret = adv + buf["val"]

        # update
        t_up = time.time()
        ac.train()
        nb = T * N
        mb = nb // pc.minibatches
        flat = {k: v.reshape(nb, *v.shape[2:]) for k, v in buf.items()}
        fadv, fret = adv.reshape(nb), ret.reshape(nb)
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "ce": 0.0, "acc": 0.0, "kl": 0.0}
        nupd = 0
        for _ in range(pc.epochs):
            perm = torch.randperm(nb, device=dev)
            for i in range(pc.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                h = flat["hist"][j]
                dist, z = ac.dist(h)
                logp = dist.log_prob(flat["act"][j]).sum(-1)
                ratio = torch.exp(logp - flat["logp"][j])
                a_ = fadv[j]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(1 - pc.clip, 1 + pc.clip) * a_).mean()
                v = ac.value(h, flat["crit"][j])
                v_old = flat["val"][j]
                v_clip = v_old + (v - v_old).clamp(-pc.clip, pc.clip)
                vf = torch.max((v - fret[j]) ** 2, (v_clip - fret[j]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + pc.vf * vf - pc.ent * ent
                if ac.decoder is not None and pc.contact_coef > 0:
                    logits = ac.contact_logits(z)
                    lab = flat["lab"][j]
                    ce = F.cross_entropy(logits.reshape(-1, 3), lab.reshape(-1), weight=cw)
                    loss = loss + pc.contact_coef * ce
                    with torch.no_grad():
                        stats["ce"] += ce.item()
                        stats["acc"] += _balanced_acc(logits.argmax(-1), lab)
                elif ac.decoder is not None:
                    # no supervision of the encoder: train the decoder on a detached latent,
                    # as a monitor of how decodable contact is from z
                    logits = ac.contact_logits(z.detach())
                    lab = flat["lab"][j]
                    ce = F.cross_entropy(logits.reshape(-1, 3), lab.reshape(-1), weight=cw)
                    loss = loss + ce
                    with torch.no_grad():
                        stats["ce"] += ce.item()
                        stats["acc"] += _balanced_acc(logits.argmax(-1), lab)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), pc.max_grad)
                opt.step()
                with torch.no_grad():
                    stats["kl"] += (flat["logp"][j] - logp).mean().item()
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                stats["ent"] += ent.item()
                nupd += 1
        up_time = time.time() - t_up
        stats = {k: v / nupd for k, v in stats.items()}

        if ep_stats["succ"]:
            succ = torch.cat(ep_stats["succ"]).mean().item()
            eplen = torch.cat(ep_stats["len"]).mean().item()
            cap = torch.cat(ep_stats["cap"]).mean().item()
            neps = sum(len(x) for x in ep_stats["succ"])
        else:
            succ = eplen = cap = float("nan")
            neps = 0
        ep_stats = {"succ": [], "len": [], "cap": []}
        stage = None
        if cur is not None:
            if neps and cur.update(total, succ * neps, neps):
                hist, crit = env.observe()
            stage = cur.stage + 1
            json.dump(cur.log, open(run / "curriculum.json", "w"), indent=1)
        row = dict(stage=stage, iter=it, env_steps=total, wall=time.time() - t0, success=succ, capture=cap,
                   ep_len=eplen, episodes=neps, sps=T * N / (roll_time + up_time),
                   roll_s=roll_time, update_s=up_time, std=ac.log_std.exp().mean().item(),
                   reward=buf["rew"].sum().item() / N,
                   ov_nefc=int(env.overflow_counts[0]), ov_narrow=int(env.overflow_counts[3]),
                   ov_iter=int(env.overflow_counts[9]), ov_ls=int(env.overflow_counts[10]), **stats)
        env.overflow_counts.zero_()
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_f.flush()
        if it % args.print_every == 0 or it == 1:
            print(f"[{args.name}] it {it} steps {total/1e6:.1f}M succ {succ:.3f} cap {cap:.3f} "
                  f"len {eplen:.0f} std {row['std']:.3f} vf {stats['vf']:.4f} ce {stats['ce']:.3f} "
                  f"acc {stats['acc']:.3f} kl {stats['kl']:.4f} sps {row['sps']:.0f}", flush=True)
        if it % args.ckpt_every == 0 or total >= args.steps:
            if succ == succ and succ >= best:
                best = succ
                save_atomic(ckpt(), run / "best.pt")
            save_atomic(ckpt(), run / "last.pt")
        if it % (args.ckpt_every * 10) == 0:
            save_atomic(ckpt(), run / f"ckpt_{total // 1_000_000:04d}M.pt")
        if args.stop_success and succ == succ and succ >= args.stop_success and total >= args.min_steps:
            save_atomic(ckpt(), run / "last.pt")
            print(f"[{args.name}] reached training success {succ:.3f} at {total/1e6:.1f}M steps", flush=True)
            reason = "stop_success"
            break
    log_f.close()
    json.dump(dict(steps=total, iter=it, wall_s=time.time() - t0, reason=reason,
                   success=succ if succ == succ else None),
              open(run / "DONE", "w"))


def _balanced_acc(pred, lab, n=3):
    accs = []
    for c in range(n):
        m = lab == c
        if m.any():
            accs.append((pred[m] == c).float().mean().item())
    return float(np.mean(accs)) if accs else float("nan")


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--robot", default="flexiv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nworld", type=int, default=8192)
    ap.add_argument("--n_variants", type=int, default=1024)
    ap.add_argument("--variant_seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=300e6)
    ap.add_argument("--encoder", default="tcn")
    ap.add_argument("--contact_coef", type=float, default=1.0)
    ap.add_argument("--env_json", default="")
    ap.add_argument("--env_file", default="")
    ap.add_argument("--family", default="study")
    ap.add_argument("--curriculum", default="none")
    ap.add_argument("--variant_kw", default="", help="usbc: JSON kwargs for builder.sample_usbc, e.g. "
                    "'{\"chamfer\": [2e-4, 5e-4]}'")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--task", default="peg", choices=["peg", "gear"])
    ap.add_argument("--minibatches", type=int, default=0, help="0: task default (peg 20, gear 24)")
    ap.add_argument("--stop_success", type=float, default=0.0, help="stop once training success >= this ...")
    ap.add_argument("--min_steps", type=float, default=0.0, help="... after at least this many env steps")
    ap.add_argument("--resume", action="store_true",
                    help="continue from runs/<name>/last.pt if it exists (else start fresh); no-op if DONE")
    return ap


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    train(parser().parse_args())
