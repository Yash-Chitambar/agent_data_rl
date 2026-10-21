"""Evaluation harness (A-12): matched start states, one row per episode, optional rollout logs.

python -m blind_insertion.eval --policy runs/<name>/last.pt --episodes 2048 --out results/B/x.parquet
Policies: a checkpoint path, or one of zero | random | spiral.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.peg_env import INSIDE, EnvCfg, PegEnv
from blind_insertion.policies.tcn_actor_critic import ActorCritic


def load_policy(path, env, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("modality") in ("rgb", "depth", "rgbd"):
        from blind_insertion.policies.visual import VisualActorCritic  # noqa: PLC0415
        ac = VisualActorCritic(ck["obs_dim"], ck["act_dim"], ck["critic_dim"], history=env.cfg.history,
                               res=ck.get("res", 64), modality=ck["modality"],
                               proprio=ck.get("proprio", False)).to(device)
        ac.load_state_dict(ck["model"])
        ac.eval()
        return ac
    ac = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["critic_dim"], history=env.cfg.history,
                     encoder=ck.get("encoder", "tcn"), n_points=ck.get("n_points", 64)).to(device)
    ac.load_state_dict(ck["model"])
    ac.eval()
    ac.obs_norm.freeze()
    ac.crit_norm.freeze()
    return ac


class Spiral:
    """Optimistic scripted spiral search (plan §8.6), privileged: knows the hole pose.

    Centred on the peg's start, pitch = 2*clearance (the capture band), phase chosen so the
    spiral passes over the hole centre ('best starting direction'), the peg held aligned
    with the hole axis while pressing down.
    """

    def __init__(self, env, speed=0.02, press=0.0, gain=10.0, press_force=3.0):
        self.env, self.speed, self.press, self.gain = env, speed, press, gain
        self.press_force = press_force
        self.mass = torch.as_tensor(env.lay.peg_mass, device=env.device, dtype=torch.float32)
        self.kp = torch.as_tensor(env.lay.robot.kp, device=env.device, dtype=torch.float32)
        lay = env.lay
        self.c = torch.as_tensor([v.clearance for v in lay.variants], device=env.device,
                                 dtype=torch.float32)
        self.reset_ids(torch.arange(env.N, device=env.device))

    def reset_ids(self, ids):
        env = self.env
        if not hasattr(self, "theta"):
            self.theta = torch.zeros(env.N, device=env.device)
            self.phase0 = torch.zeros(env.N, device=env.device)
            self.center = torch.zeros(env.N, 2, device=env.device)
        p, _, _, _ = env.peg_in_fixture()
        self.center[ids] = p[ids, :2]
        pitch = 2 * self.c[env.variant[ids]]
        rh = self.center[ids].norm(dim=1)
        th_star = 2 * math.pi * rh / pitch
        to_hole = torch.atan2(-self.center[ids, 1], -self.center[ids, 0])
        self.phase0[ids] = to_hole - th_star
        self.theta[ids] = 0.0
        if not hasattr(self, "inserting"):
            self.inserting = torch.zeros(env.N, dtype=torch.bool, device=env.device)
        self.inserting[ids] = False

    def __call__(self, hist):
        env = self.env
        dt = env.control_dt
        pitch = 2 * self.c[env.variant]
        r = pitch * self.theta / (2 * math.pi)
        self.theta = self.theta + self.speed * dt / torch.clamp(r, min=pitch)
        r = pitch * self.theta / (2 * math.pi)
        ang = self.phase0 + self.theta
        xy = self.center + torch.stack([r * torch.cos(ang), r * torch.sin(ang)], 1)
        # desired peg pose in fixture frame: aligned, pressing below the top face
        p, R, _, _ = env.peg_in_fixture()
        lab, _ = env.labels(env.edge_points_fixture(p, R))
        self.inserting |= (lab == INSIDE).all(1)           # dropped in: stop searching, insert
        ins = self.inserting
        xy = torch.where(ins[:, None], torch.zeros_like(xy), xy)
        self.theta = torch.where(ins, self.theta - self.speed * dt / torch.clamp(r, min=pitch), self.theta)
        z_des = torch.where(ins, -(env.depth[env.variant] + 0.002), torch.full_like(ins, -self.press, dtype=torch.float32))
        p_des = torch.cat([xy, z_des[:, None]], 1)
        # current tcp and peg in world -> relative transform (true, privileged)
        q = env._q_meas
        tp, tR, J = env.chain.fk(q, with_jac=True)
        Rf, pf = env.fix_R, env.fix_p
        R_peg_w = Rf @ R
        p_peg_w = pf + (Rf @ p[..., None])[..., 0]
        R_rel = tR.transpose(1, 2) @ R_peg_w            # peg in tcp
        p_rel = (tR.transpose(1, 2) @ (p_peg_w - tp)[..., None])[..., 0]
        R_peg_des_w = Rf
        p_peg_des_w = pf + (Rf @ p_des[..., None])[..., 0]
        R_tcp_des = R_peg_des_w @ R_rel.transpose(1, 2)
        p_tcp_des = p_peg_des_w - (R_tcp_des @ p_rel[..., None])[..., 0]
        e = torch.cat([p_tcp_des - tp, K.rot_log(R_tcp_des @ tR.transpose(1, 2))], 1)
        JJt = J @ J.transpose(1, 2) + 1e-4 * torch.eye(6, device=q.device)
        dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e[..., None]))[..., 0]
        # privileged feed-forward: cancel the peg's weight and press along the hole axis
        m = self.mass[env.variant]
        F = torch.zeros(env.N, 6, device=q.device)
        F[:, 2] = 9.81 * m
        pf_ = torch.where(ins, torch.full_like(m, 3 * self.press_force), torch.full_like(m, self.press_force))
        F[:, :3] -= pf_[:, None] * Rf[:, :, 2]
        ff = (J.transpose(1, 2) @ F[..., None])[..., 0] / self.kp
        return ((self.gain * dq + ff) / env.scale).clamp(-3, 3)


def evaluate(env: PegEnv, policy, *, seed=12345, episodes=None, log_worlds=(), log_latent=False,
             spiral=None, max_steps=None):
    """Run each world's first episode from matched starts; return a DataFrame."""
    dev = env.device
    N = env.N
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = torch.arange(N, device=dev)
    starts = env.sample_starts(ids, gen=gen)
    env.reset(ids, starts)
    if spiral is not None:
        spiral.reset_ids(ids)
    finished = torch.zeros(N, dtype=torch.bool, device=dev)
    rows = {k: torch.zeros(N, device=dev) for k in
            ("success", "time", "travel", "t_partial", "tilt_partial", "tilt_final", "captured",
             "far", "slip", "timeout", "bad", "n_inside_final", "final_z")}
    start_xy = env.start_xy.clone()
    log = {w: {k: [] for k in ("qpos", "action", "peg_p", "peg_R", "labels", "z", "hist_obs",
                                "mocap_pos", "mocap_quat")} for w in log_worlds}
    hist, _ = env.observe()
    T = max_steps or env.max_steps
    visual = type(policy).__name__ == "VisualActorCritic"
    if visual:
        from blind_insertion.policies.visual import EmbHistory  # noqa: PLC0415
        eh = EmbHistory(N, env.cfg.history, policy.emb, dev)
        with torch.no_grad():
            eh.push(policy.cnn(env.frame), torch.ones(N, dtype=torch.bool, device=dev))
    for t in range(T + 1):
        with torch.no_grad():
            if visual:
                a, z = policy.act_from_emb(eh.h, hist)
            elif isinstance(policy, ActorCritic):
                a, z = policy.act_mean(hist)
            else:
                a, z = policy(hist), None
        for w in log_worlds:
            if not finished[w]:
                L = log[w]
                L["qpos"].append(env.t_qpos[w].cpu().numpy().copy())
                L["mocap_pos"].append(env.t_mocap_pos[w, 0].cpu().numpy().copy())
                L["mocap_quat"].append(env.t_mocap_quat[w, 0].cpu().numpy().copy())
                L["action"].append(a[w].cpu().numpy())
                p, R, _, _ = env.peg_in_fixture()
                L["peg_p"].append(p[w].cpu().numpy())
                L["peg_R"].append(R[w].cpu().numpy())
                L["labels"].append(env.labels(env.edge_points_fixture(p, R))[0][w].cpu().numpy())
                L["hist_obs"].append(hist[w, -1].cpu().numpy())
                if z is not None and log_latent:
                    L["z"].append(z[w].cpu().numpy())
        hist, _, _, done, info = env.step(a)
        if visual:
            with torch.no_grad():
                eh.push(policy.cnn(env.frame), done)
        new = done & ~finished
        if new.any():
            for k in ("success", "far", "slip", "timeout", "bad", "captured"):
                rows[k][new] = info[k][new].float()
            rows["time"][new] = info["steps"][new].float() * env.control_dt
            rows["travel"][new] = info["travel"][new]
            rows["t_partial"][new] = info["t_partial"][new]
            rows["tilt_partial"][new] = info["tilt_partial"][new]
            rows["tilt_final"][new] = info["tilt"][new]
            rows["n_inside_final"][new] = info["n_inside"][new].float()
            rows["final_z"][new] = info["peg_p"][new, 2]
            finished |= new
        if spiral is not None and done.any():
            spiral.reset_ids(done.nonzero()[:, 0])
        if finished.all():
            break
    df = pd.DataFrame({k: v.cpu().numpy() for k, v in rows.items()})
    df["world"] = np.arange(N)
    df["variant"] = env.variant.cpu().numpy()
    df["start_x"], df["start_y"] = start_xy[:, 0].cpu().numpy(), start_xy[:, 1].cpu().numpy()
    df["start_dist"] = np.hypot(df.start_x, df.start_y)
    df["clearance"] = [env.lay.variants[k].clearance for k in df.variant]
    df["failure"] = np.select(
        [df.success > 0, df.bad > 0, df.slip > 0, df.far > 0, df.n_inside_final >= 64,
         df.n_inside_final > 0, df.captured > 0],
        ["success", "nan", "slipped", "left_fixture", "jammed_inside", "wedged_partial",
         "captured_then_lost"], "never_found")
    if episodes is not None:
        df = df.iloc[:episodes]
    logs = {w: {k: np.asarray(v) for k, v in L.items() if len(v)} for w, L in log.items()}
    return df, logs


def add_env_args(ap, n_variants=1024, variant_seed=1000, seed=12345):
    """Environment flags shared by eval.py and analysis/probes.py."""
    ap.add_argument("--robot", default="flexiv")
    ap.add_argument("--n_variants", type=int, default=n_variants)
    ap.add_argument("--variant_seed", type=int, default=variant_seed)   # unseen geometries, same dist.
    ap.add_argument("--seed", type=int, default=seed)
    ap.add_argument("--env_json", default="")
    ap.add_argument("--ood", default="", help="OOD bin from builder.OOD_BINS")
    ap.add_argument("--env_file", default="")
    ap.add_argument("--family", default="", help="as in train.py (study | usbc | scale:o:c | ood:bin)")
    ap.add_argument("--clearance", default="", help="usbc: fixed clearance (m) or lo,hi")
    ap.add_argument("--variant_kw", default="", help="usbc: JSON kwargs for builder.sample_usbc (as train.py)")
    ap.add_argument("--peg_offset", type=float, default=None)


def env_kwargs(args, nworld) -> dict:
    """EnvCfg kwargs from the add_env_args flags (env_file, then env_json, then the explicit flags)."""
    kw = dict(robot=args.robot, nworld=nworld, n_variants=args.n_variants,
              variant_seed=args.variant_seed, seed=args.seed)
    if args.ood:
        from blind_insertion.assets import builder as B  # noqa: PLC0415
        kw["variant_sampler"] = lambda n, sd, b=args.ood: B.sample_ood_pegs(n, sd, b)
    if args.env_file:
        kw.update(json.load(open(args.env_file)))
    kw.update(json.loads(args.env_json) if args.env_json else {})
    if args.peg_offset is not None:
        kw["peg_offset"] = args.peg_offset
    if args.family:
        from blind_insertion.assets import builder as B  # noqa: PLC0415
        from blind_insertion.train import make_variants  # noqa: PLC0415
        if args.family == "usbc" and args.clearance:
            cr = [float(x) for x in args.clearance.split(",")]
            kw["variants"] = B.sample_usbc(args.n_variants, args.variant_seed,
                                           clearance=(cr[0], cr[-1]), **json.loads(args.variant_kw or "{}"))
        else:
            kw["variants"] = make_variants(argparse.Namespace(
                family=args.family, curriculum="none", n_variants=args.n_variants,
                variant_seed=args.variant_seed, variant_kw=args.variant_kw))
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--episodes", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log_worlds", default="")
    ap.add_argument("--task", default="peg", choices=["peg", "gear"])
    add_env_args(ap)
    args = ap.parse_args()
    if args.task == "gear":
        from blind_insertion.eval_gear import run as run_gear  # noqa: PLC0415
        return run_gear(args)
    kw = env_kwargs(args, args.episodes)
    if args.policy.endswith(".pt"):
        ck = torch.load(args.policy, map_location="cpu", weights_only=False)
        if ck.get("modality") in ("rgb", "depth", "rgbd"):
            kw.update(camera=True, cam_res=ck.get("res", 64), cam_modality=ck["modality"])
    env = PegEnv(EnvCfg(**kw))
    spiral = None
    if args.policy == "zero":
        pol = lambda h: torch.zeros(env.N, env.n_arm, device=env.device)  # noqa: E731
    elif args.policy == "random":
        pol = lambda h: 0.5 * torch.randn(env.N, env.n_arm, device=env.device)  # noqa: E731
    elif args.policy.startswith("spiral"):
        kw_s = json.loads(args.policy[len("spiral"):] or "{}")
        spiral = Spiral(env, **kw_s)
        pol = spiral
    else:
        pol = load_policy(args.policy, env, env.device)
    lw = [int(x) for x in args.log_worlds.split(",") if x]
    df, logs = evaluate(env, pol, seed=args.seed, log_worlds=lw, log_latent=True, spiral=spiral)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    if logs:
        np.savez_compressed(out.with_suffix(".rollouts.npz"),
                            **{f"w{w}_{k}": v for w, L in logs.items() for k, v in L.items()})
    s = df.success.mean()
    print(f"{args.policy}: success {s:.3f} (n={len(df)}), median time {df[df.success>0].time.median():.2f}s, "
          f"median travel {1000*df[df.success>0].travel.median():.1f} mm; failures: "
          f"{df.failure.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
