"""PPO for the wrist-RGB visual policy (same task, rewards, critic, contact loss as the blind one).

Rollout frames are stored once per step (uint8), with the previous rollout's last H-1 frames
carried over; minibatches are drawn by world so each frame is embedded once per minibatch,
and histories are reconstructed by index (clamped at episode starts, as in acting).

python -m blind_insertion.train_visual --name B-peg-rgb-base-s0 --seed 0 --nworld 4096
Phase F (USB-C on the YAM, plan §12): the env flags of train.py plus --modality rgb|depth|rgbd and
--proprio (fusion with the joint-history encoder), e.g.
python -m blind_insertion.train_visual --name F-usbc-pd-curr-s0 --modality depth --proprio \
    --family usbc --curriculum usbc --n_variants 320 --env_file blind_insertion/configs/phase_f_usbc.json
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.policies.visual import EmbHistory, VisualActorCritic
from blind_insertion.train import RUNS, USBC_STAGES, Curriculum, PPOCfg, _balanced_acc, git_hash, make_env, save_atomic


def window_index(T, H, count):
    """count (T, W): frames in episode at each step (<= H). -> (T, W, H) indices into the
    frame buffer of length T+H-1, oldest first, clamped at the episode's first frame."""
    t = torch.arange(T, device=count.device)[:, None, None]
    back = torch.arange(H - 1, -1, -1, device=count.device)[None, None, :]      # H-1..0
    back = torch.minimum(back, (count - 1)[..., None])
    return H - 1 + t - back


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
    env = make_env(args, camera=True, cam_res=args.res, cam_modality=args.modality)
    dev = env.device
    cur = Curriculum(env, USBC_STAGES) if args.curriculum == "usbc" else None
    pc = PPOCfg()
    pc.contact_coef = args.contact_coef
    H = env.cfg.history
    ac = VisualActorCritic(env.obs_dim, env.n_arm, env.critic_dim, history=H, res=args.res,
                           modality=args.modality, proprio=args.proprio).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=pc.lr)
    meta = dict(args=vars(args), env=asdict(env.cfg) | {"variants": None, "variant_sampler": None},
                git=git_hash(), modality=args.modality, proprio=args.proprio,
                versions={p: version(p) for p in ("mujoco", "mujoco-warp", "torch", "mjfit")})
    resuming = args.resume and (run / "last.pt").exists()
    json.dump(meta, open(run / ("config_resume.json" if resuming else "config.json"), "w"), indent=1, default=str)

    N, T, D, A, R = env.N, pc.rollout, env.obs_dim, env.n_arm, args.res
    C = env.frame.shape[1]
    frames = torch.zeros(T + H - 1, N, C, R, R, dtype=torch.uint8, device=dev)
    buf = dict(obs=torch.zeros(T, N, D, device=dev), crit=torch.zeros(T, N, env.critic_dim, device=dev),
               act=torch.zeros(T, N, A, device=dev), logp=torch.zeros(T, N, device=dev),
               val=torch.zeros(T, N, device=dev), rew=torch.zeros(T, N, device=dev),
               done=torch.zeros(T, N, device=dev), tout=torch.zeros(T, N, device=dev),
               lab=torch.zeros(T, N, 64, dtype=torch.long, device=dev),
               count=torch.ones(T, N, dtype=torch.long, device=dev))
    if args.proprio:   # joint histories for the fused policy's TCN
        buf["hist"] = torch.zeros(T, N, H, D, device=dev)
    total, it, best, wall0 = 0, 0, -1.0, 0.0
    if resuming:
        ck = torch.load(run / "last.pt", map_location=dev, weights_only=False)
        ac.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        it, total, best, wall0 = ck["iter"], ck["steps"], ck.get("best", -1.0), ck.get("wall", 0.0)
        stage = ck.get("stage")
        if cur is not None and not stage and (run / "curriculum.json").exists():   # older checkpoints
            stage = max(r["stage"] for r in json.load(open(run / "curriculum.json"))
                        if r["env_steps"] <= total)
        if cur is not None and stage:
            while cur.stage + 1 < stage:
                cur.advance(total)
            if (run / "curriculum.json").exists():
                cur.log = [r for r in json.load(open(run / "curriculum.json")) if r["stage"] <= stage]
        if "rng" in ck:
            torch.set_rng_state(ck["rng"]["torch"].cpu())
            torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu(), dev)
            env.gen.set_state(ck["rng"]["env"].cpu())
        print(f"[{args.name}] resumed at it {it}, {total/1e6:.1f}M steps", flush=True)
    env.reset_all(desync=True)
    env.render()
    hist, crit = env.observe()
    eh = EmbHistory(N, H, ac.emb, dev)
    with torch.no_grad():
        e0 = ac.cnn(env.frame)
    eh.push(e0, torch.ones(N, dtype=torch.bool, device=dev))
    frames[:] = env.frame[None]
    cw = torch.tensor(pc.class_w, device=dev)
    log_path = run / "train_log.csv"
    log = None
    if resuming and log_path.exists() and log_path.stat().st_size:
        with open(log_path) as f_:
            fields = next(csv.reader(f_))
        rows = [r for r in csv.DictReader(open(log_path))   # a kill mid-write can leave a partial last row
                if (r.get("iter") or "").strip().isdigit() and int(r["iter"]) <= it]
        log_f = open(log_path, "w", newline="")
        log = csv.DictWriter(log_f, fieldnames=fields)
        log.writeheader()
        log.writerows(rows)
    else:
        log_f = open(log_path, "w", newline="")
    t0 = time.time() - wall0
    reason, succ = "steps", float("nan")
    ep = {"succ": [], "len": [], "cap": []}
    while total < args.steps:
        it += 1
        t_roll = time.time()
        ac.eval()
        frames[:H - 1] = frames[T:].clone()       # carry the last H-1 frames
        with torch.no_grad():
            # re-embed the carried window with the current CNN so acting matches the update
            frames[H - 1] = env.frame
            E = torch.cat([ac.cnn(frames[:H, j:j + 256].reshape(-1, C, R, R)).reshape(H, -1, ac.emb)
                           for j in range(0, N, 256)], 1)
            idx0 = window_index(1, H, eh.count[None])[0]               # (N, H)
            eh.h = E[idx0, torch.arange(N, device=dev)[:, None].expand_as(idx0)]
            for t in range(T):
                o_now = hist[:, -1]
                ac.obs_norm.update(o_now)
                ac.crit_norm.update(crit)
                frames[H - 1 + t] = env.frame
                buf["count"][t] = eh.count
                if args.proprio:
                    buf["hist"][t] = hist
                d, _ = ac.dist_from_emb(eh.h, hist)
                a = d.sample()
                buf["obs"][t], buf["crit"][t], buf["act"][t] = o_now, crit, a
                buf["logp"][t] = d.log_prob(a).sum(-1)
                buf["val"][t] = ac.value(o_now, crit)
                p, Rm, _, _ = env.peg_in_fixture()
                buf["lab"][t] = env.labels(env.edge_points_fixture(p, Rm))[0]
                hist, crit, r, done, info = env.step(a)
                buf["rew"][t], buf["done"][t] = r, done.float()
                buf["tout"][t] = info["timeout"].float()
                eh.push(ac.cnn(env.frame), done)
                if done.any():
                    ep["succ"].append(info["success"][done].float().cpu())
                    ep["len"].append(info["steps"][done].float().cpu())
                    ep["cap"].append(info["captured"][done].float().cpu())
            last_val = ac.value(hist[:, -1], crit)
        total += T * N
        env.refresh_pool()
        roll_time = time.time() - t_roll
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
            widx = window_index(T, H, buf["count"])          # (T, N, H)

        t_up = time.time()
        ac.train()
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "ce": 0.0, "acc": 0.0, "kl": 0.0}
        nupd = 0
        nw = N // pc.minibatches
        for _ in range(pc.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(pc.minibatches):
                W = perm[i * nw:(i + 1) * nw]
                fr = frames[:, W]                                    # (T+H-1, nw, 3, R, R)
                E = ac.cnn(fr.reshape(-1, C, R, R)).reshape(T + H - 1, nw, -1)
                idx = widx[:, W]                                     # (T, nw, H)
                wcol = torch.arange(nw, device=dev)[None, :, None].expand_as(idx)
                eh_mb = E[idx, wcol]                                 # (T, nw, H, emb)
                eh_mb = eh_mb.reshape(T * nw, H, -1)
                sel = lambda x: x[:, W].reshape(T * nw, *x.shape[2:])  # noqa: E731
                dist, z = ac.dist_from_emb(eh_mb, sel(buf["hist"]) if args.proprio else None)
                act, logp_old = sel(buf["act"]), sel(buf["logp"])
                logp = dist.log_prob(act).sum(-1)
                ratio = torch.exp(logp - logp_old)
                a_ = sel(adv)
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pg = -torch.min(ratio * a_, ratio.clamp(1 - pc.clip, 1 + pc.clip) * a_).mean()
                v = ac.value(sel(buf["obs"]), sel(buf["crit"]))
                v_old, r_ = sel(buf["val"]), sel(ret)
                v_clip = v_old + (v - v_old).clamp(-pc.clip, pc.clip)
                vf = torch.max((v - r_) ** 2, (v_clip - r_) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                lab = sel(buf["lab"])
                logits = ac.contact_logits(z if pc.contact_coef > 0 else z.detach())
                ce = F.cross_entropy(logits.reshape(-1, 3), lab.reshape(-1), weight=cw)
                # coef 0: the decoder still trains on a detached latent (monitor only)
                loss = pg + pc.vf * vf - pc.ent * ent + (pc.contact_coef if pc.contact_coef > 0 else 1.0) * ce
                if nupd == 0:   # model unchanged since the rollout: reconstruction must match
                    stats["hist_check"] = (logp - logp_old).abs().max().item()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), pc.max_grad)
                opt.step()
                with torch.no_grad():
                    stats["kl"] += (logp_old - logp).mean().item()
                    stats["acc"] += _balanced_acc(logits.argmax(-1), lab)
                stats["pg"] += pg.item()
                stats["vf"] += vf.item()
                stats["ent"] += ent.item()
                stats["ce"] += ce.item()
                nupd += 1
        up_time = time.time() - t_up
        hc = stats.pop("hist_check", float("nan"))
        stats = {k: v / nupd for k, v in stats.items()}
        stats["hist_check"] = hc
        if ep["succ"]:
            succ = torch.cat(ep["succ"]).mean().item()
            eplen = torch.cat(ep["len"]).mean().item()
            cap = torch.cat(ep["cap"]).mean().item()
            neps = sum(len(x) for x in ep["succ"])
        else:
            succ = eplen = cap = float("nan")
            neps = 0
        ep = {"succ": [], "len": [], "cap": []}
        stage = None
        if cur is not None:
            if neps and cur.update(total, succ * neps, neps):   # every world was reset
                hist, crit = env.observe()
                with torch.no_grad():
                    eh.push(ac.cnn(env.frame), torch.ones(N, dtype=torch.bool, device=dev))
            stage = cur.stage + 1
            json.dump(cur.log, open(run / "curriculum.json", "w"), indent=1)
        row = dict(stage=stage, iter=it, env_steps=total, wall=time.time() - t0, success=succ, capture=cap,
                   ep_len=eplen, episodes=neps, sps=T * N / (roll_time + up_time), roll_s=roll_time,
                   update_s=up_time, std=ac.log_std.exp().mean().item(),
                   reward=buf["rew"].sum().item() / N, **stats)
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_f.flush()
        if it % args.print_every == 0 or it == 1:
            print(f"[{args.name}] it {it} steps {total/1e6:.1f}M succ {succ:.3f} cap {cap:.3f} len {eplen:.0f} "
                  f"std {row['std']:.3f} ce {stats['ce']:.3f} acc {stats['acc']:.3f} kl {stats['kl']:.4f} "
                  f"sps {row['sps']:.0f} (roll {roll_time:.1f}s upd {up_time:.1f}s) histchk {hc:.1e}", flush=True)
        def ckpt():
            return {"model": ac.state_dict(), "iter": it, "steps": total, "obs_dim": env.obs_dim,
                    "act_dim": env.n_arm, "critic_dim": env.critic_dim, "modality": args.modality,
                    "proprio": args.proprio, "res": args.res, "opt": opt.state_dict(), "best": best,
                    "wall": time.time() - t0, "stage": stage,
                    "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(dev),
                            "env": env.gen.get_state()}}
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
                   success=succ if succ == succ else None), open(run / "DONE", "w"))


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--robot", default="flexiv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--n_variants", type=int, default=1024)
    ap.add_argument("--variant_seed", type=int, default=0)
    ap.add_argument("--steps", type=float, default=300e6)
    ap.add_argument("--res", type=int, default=64)
    ap.add_argument("--contact_coef", type=float, default=1.0)
    ap.add_argument("--env_json", default="")
    ap.add_argument("--print_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=25)
    ap.add_argument("--stop_success", type=float, default=0.0)
    ap.add_argument("--min_steps", type=float, default=0.0)
    ap.add_argument("--env_file", default="")
    ap.add_argument("--family", default="study", help="as in train.py (study | usbc | scale:o:c | ood:bin)")
    ap.add_argument("--curriculum", default="none", help="none | usbc")
    ap.add_argument("--variant_kw", default="", help="usbc: JSON kwargs for builder.sample_usbc")
    ap.add_argument("--modality", default="rgb", choices=["rgb", "depth", "rgbd"])
    ap.add_argument("--proprio", action="store_true", help="fuse with the joint-history encoder (F-PR/PD/PRD)")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt (no-op if DONE)")
    return ap


if __name__ == "__main__":
    train(parser().parse_args())
