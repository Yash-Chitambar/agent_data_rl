"""Gear-meshing evaluation (called by `blind_insertion.eval --task gear`).

Matched starts, each world's first episode, one row per episode, optional rollout logs:
python -m blind_insertion.eval --task gear --policy runs/<run>/last.pt --episodes 2048 \
    --out results/G/x.parquet [--ood tighter|taller|much_taller|mild_size|extreme_size|random_tooth]
Policies: a checkpoint path, or zero | random.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from blind_insertion.assets import gear_builder as GB
from blind_insertion.envs.gear_env import GearCfg, GearEnv
from blind_insertion.policies.tcn_actor_critic import ActorCritic


def evaluate_gear(env: GearEnv, policy, *, seed=12345, log_worlds=(), log_latent=False, max_steps=None,
                  demo_k=0):
    dev, N = env.device, env.N
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = torch.arange(N, device=dev)
    starts = env.sample_starts(ids, gen=gen)
    env.reset(ids, starts)
    finished = torch.zeros(N, dtype=torch.bool, device=dev)
    keys = ("success", "time", "travel", "captured", "meshed", "t_capture", "t_mesh", "far", "slip",
            "timeout", "bad", "tilt_final", "final_z", "n_inside_final", "n_mesh_final")
    rows = {k: torch.zeros(N, device=dev) for k in keys}
    start_xy = env.start_xy.clone()
    start_phase = env.phase_error().clone()
    log = {w: {k: [] for k in ("qpos", "action", "peg_p", "peg_R", "labels", "z", "hist_obs",
                                "mocap_pos", "mocap_quat")} for w in log_worlds}
    hist, _ = env.observe()
    T = max_steps or env.max_steps
    rec = {k: [] for k in ("qpos", "mocap_pos", "mocap_quat", "action", "peg_p", "peg_R", "labels")} if demo_k else None
    for _t in range(T + 1):
        with torch.no_grad():
            if isinstance(policy, ActorCritic):
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
                L["labels"].append(env.current_labels()[w].cpu().numpy())
                L["hist_obs"].append(hist[w, -1].cpu().numpy())
                if z is not None and log_latent:
                    L["z"].append(z[w].cpu().numpy())
        if rec is not None:
            from blind_insertion.eval import record_step  # noqa: PLC0415
            record_step(env, a, rec, env.current_labels())
        hist, _, _, done, info = env.step(a)
        new = done & ~finished
        if new.any():
            for k in ("success", "far", "slip", "timeout", "bad", "captured", "meshed", "t_capture", "t_mesh"):
                rows[k][new] = info[k][new].float()
            rows["time"][new] = info["steps"][new].float() * env.control_dt
            rows["travel"][new] = info["travel"][new]
            rows["tilt_final"][new] = info["tilt"][new]
            rows["final_z"][new] = info["peg_p"][new, 2]
            rows["n_inside_final"][new] = info["n_inside"][new].float()
            rows["n_mesh_final"][new] = info["n_mesh"][new].float()
            finished |= new
        if finished.all():
            break
    df = pd.DataFrame({k: v.cpu().numpy() for k, v in rows.items()})
    df["world"] = np.arange(N)
    k = env.variant.cpu().numpy()
    df["variant"] = k
    df["start_x"], df["start_y"] = start_xy[:, 0].cpu().numpy(), start_xy[:, 1].cpu().numpy()
    df["start_dist"] = np.hypot(df.start_x, df.start_y)
    sp = start_phase.abs().cpu().numpy()
    df["start_phase0"], df["start_phase1"] = sp[:, 0], sp[:, 1]
    vs = env.lay.variants
    df["n_anchor"] = [len(vs[i].anchors) for i in k]
    df["module"] = [vs[i].module for i in k]
    df["teeth"] = [vs[i].teeth for i in k]
    df["backlash"] = [min(a["backlash"] for a in vs[i].anchors) for i in k]
    df["failure"] = np.select(
        [df.success > 0, df.bad > 0, df.slip > 0, df.far > 0, (df.meshed > 0) & (df.captured > 0),
         df.captured > 0],
        ["success", "nan", "slipped", "left_fixture", "meshed_not_seated", "on_shaft_not_meshed"],
        "never_captured")
    logs = {w: {k_: np.asarray(v) for k_, v in L.items() if len(v)} for w, L in log.items()}
    if rec is not None:
        from blind_insertion.eval import select_demos  # noqa: PLC0415
        logs.update(select_demos(df, rec, env.control_dt, demo_k))
    return df, logs


def run(args):
    from blind_insertion.eval import load_policy  # noqa: PLC0415
    kw = dict(robot=args.robot, nworld=args.episodes, n_variants=args.n_variants,
              variant_seed=args.variant_seed, seed=args.seed)
    if args.env_file:
        kw.update(json.load(open(args.env_file)))
    kw.update(json.loads(args.env_json) if args.env_json else {})
    if args.peg_offset is not None:
        kw["peg_offset"] = args.peg_offset
    fam = args.family or (f"ood:{args.ood}" if args.ood else "study")
    if fam == "study":
        kw["variants"] = GB.sample_study_gears(args.n_variants, args.variant_seed)
    elif fam.startswith("ood:"):
        kw["variants"] = GB.sample_ood_gears(args.n_variants, args.variant_seed, fam[4:])
    else:
        raise ValueError(fam)
    env = GearEnv(GearCfg(**kw))
    if args.policy == "zero":
        pol = lambda h: torch.zeros(env.N, env.n_arm, device=env.device)  # noqa: E731
    elif args.policy == "random":
        pol = lambda h: 0.5 * torch.randn(env.N, env.n_arm, device=env.device)  # noqa: E731
    else:
        pol = load_policy(args.policy, env, env.device)
    lw = [int(x) for x in args.log_worlds.split(",") if x]
    df, logs = evaluate_gear(env, pol, seed=args.seed, log_worlds=lw, log_latent=True, demo_k=args.demo_k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    if logs:
        np.savez_compressed(out.with_suffix(".rollouts.npz"), meta=np.array(json.dumps(vars(args))),
                            **{f"w{w}_{k}": v for w, L in logs.items() for k, v in L.items()})
    s = df.success.mean()
    ok = df[df.success > 0]
    print(f"{args.policy} [gear {fam}]: success {s:.4f} (n={len(df)}), median time {ok.time.median():.2f}s, "
          f"median travel {1000*ok.travel.median():.1f} mm; failures: {df.failure.value_counts().to_dict()}")
