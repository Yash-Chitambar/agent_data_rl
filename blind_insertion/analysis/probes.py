"""Linear probe of the history latent (B-R4) and latent-swap test (B-R5), paper §3.2 / App. C.

python -m blind_insertion.analysis.probes --policy runs/X/last.pt --out results/B/probe_s0
Other tasks take eval.py's env flags, e.g. the YAM (C-R3):
python -m blind_insertion.analysis.probes --policy runs/C-.../last.pt --env_file blind_insertion/configs/phase_c_yam.json ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from blind_insertion.envs.peg_env import FREE, INSIDE, EnvCfg, PegEnv  # noqa: E402
from blind_insertion.eval import add_env_args, env_kwargs, load_policy  # noqa: E402
from blind_insertion.policies.tcn_actor_critic import ActorCritic  # noqa: E402

MODES = ["lifted", "off", "far", "near", "partial", "moving", "stuck"]
SWAP_MODES = ["far", "near", "partial", "moving", "stuck"]


@torch.no_grad()
def collect(env: PegEnv, ac, *, seed=777, near_mm=5.0, stuck_mm=0.5, stuck_steps=10):
    """One deterministic episode per world; per-step z, normalized o_t, raw hist, mode."""
    dev, N = env.device, env.N
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    ids = torch.arange(N, device=dev)
    env.reset(ids, env.sample_starts(ids, gen=gen))
    hist, _ = env.observe()
    finished = torch.zeros(N, dtype=torch.bool, device=dev)
    depth_hist = []
    Z, O, Hs, M, W, T = [], [], [], [], [], []
    for t in range(env.max_steps + 1):
        mu, z = ac.act_mean(hist)
        p, R, _, _ = env.peg_in_fixture()
        pts = env.edge_points_fixture(p, R)
        lab, _ = env.labels(pts)
        n_in = (lab == INSIDE).sum(1)
        all_free = (lab == FREE).all(1)
        depth = -pts[..., 2].mean(1)
        depth_hist.append(depth)
        k = env.variant
        off = p[:, :2].abs().max(1).values > env.plate_half[k]
        dist = p[:, :2].norm(dim=1)
        mode = torch.full((N,), 2, dtype=torch.long, device=dev)            # far
        mode[dist <= near_mm / 1000] = 3                                     # near
        mode[(n_in > 0) & (n_in < 64)] = 4                                   # partial
        inside = n_in == 64
        if len(depth_hist) > stuck_steps:
            prog = depth - depth_hist[-1 - stuck_steps]
            stuck = prog < stuck_mm / 1000
        else:
            stuck = torch.zeros_like(inside)
        mode[inside] = 5
        mode[inside & stuck] = 6
        mode[all_free & ~inside] = 0
        mode[off & ~inside] = 1
        live = ~finished
        Z.append(z[live].cpu()), O.append(ac.obs_norm(hist[:, -1])[live].cpu())
        Hs.append(hist[live].cpu()), M.append(mode[live].cpu())
        W.append(ids[live].cpu()), T.append(torch.full((int(live.sum()),), t))
        hist, _, _, done, info = env.step(mu)
        finished |= done
        if finished.all():
            break
    return dict(z=torch.cat(Z), o=torch.cat(O), hist=torch.cat(Hs), mode=torch.cat(M),
                world=torch.cat(W), t=torch.cat(T))


def fit_probe(X, y, train, n_classes=7, epochs=300, device="cuda"):
    """Multinomial logistic regression (class-balanced CE), standardized inputs; returns test preds."""
    X = X.to(device).float()
    y = y.to(device)
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Xn = (X - mu) / sd
    lin = torch.nn.Linear(X.shape[1], n_classes).to(device)
    cnt = torch.bincount(y[train], minlength=n_classes).float()
    w = torch.where(cnt > 0, cnt.sum() / (n_classes * cnt.clamp(min=1)), torch.zeros_like(cnt))
    opt = torch.optim.LBFGS(lin.parameters(), lr=0.5, max_iter=epochs, line_search_fn="strong_wolfe")
    Xt, yt = Xn[train], y[train]

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(Xt), yt, weight=w) + 1e-4 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return lin(Xn).argmax(1).cpu()


def balanced(pred, y, n_classes=7):
    rec = []
    for c in range(n_classes):
        m = y == c
        rec.append(float((pred[m] == c).float().mean()) if m.any() else np.nan)
    return float(np.nanmean(rec)), rec


def confusion(pred, y, n=7):
    cm = np.zeros((n, n))
    for a, b in zip(y.numpy(), pred.numpy()):
        cm[a, b] += 1
    return 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)


def boot_ci_bal(pred, y, worlds, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    uw = np.unique(worlds.numpy())
    accs = []
    for _ in range(n_boot):
        pick = rng.choice(uw, len(uw))
        m = torch.from_numpy(np.isin(worlds.numpy(), pick))
        accs.append(balanced(pred[m], y[m])[0])
    return float(np.nanpercentile(accs, 2.5)), float(np.nanpercentile(accs, 97.5))


@torch.no_grad()
def latent_swap(ac, data, n_pairs=20000, seed=0, device="cuda"):
    """Receiving state keeps its o_t, gets a donor's z; ||delta action|| normalized by the mean
    change for donors in the receiver's own mode (paper Fig. 15)."""
    rng = np.random.default_rng(seed)
    idx_by = {m: np.where(data["mode"].numpy() == MODES.index(m))[0] for m in SWAP_MODES}
    out = np.full((5, 5), np.nan)
    raw = np.full((5, 5), np.nan)
    for i, mr in enumerate(SWAP_MODES):
        if len(idx_by[mr]) < 50:
            continue
        r = rng.choice(idx_by[mr], n_pairs)
        o_r = data["o"][r].to(device)
        z_r = data["z"][r].to(device)
        a_r = ac.actor(torch.cat([z_r, o_r], 1))
        for j, md in enumerate(SWAP_MODES):
            if len(idx_by[md]) < 50:
                continue
            dnr = rng.choice(idx_by[md], n_pairs)
            z_d = data["z"][dnr].to(device)
            a_s = ac.actor(torch.cat([z_d, o_r], 1))
            raw[i, j] = (a_s - a_r).norm(dim=1).mean().item()
        out[i] = raw[i] / raw[i, i]
    return out, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--worlds", type=int, default=1536)
    ap.add_argument("--out", required=True)
    ap.add_argument("--near_mm", type=float, default=5.0, help="far/near split (mm); scale with part size")
    add_env_args(ap, n_variants=0, variant_seed=2000, seed=11)   # n_variants 0: one per world
    args = ap.parse_args()
    args.n_variants = args.n_variants or args.worlds
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = PegEnv(EnvCfg(**env_kwargs(args, args.worlds)))
    ac = load_policy(args.policy, env, env.device)
    data = collect(env, ac, near_mm=args.near_mm)
    y, worlds = data["mode"], data["world"]
    train = (worlds % 2 == 0)
    test = ~train
    # random encoder with the trained policy's normalizer
    rnd = ActorCritic(ac.obs_dim, ac.act_dim, env.critic_dim, history=env.cfg.history).to(env.device)
    rnd.obs_norm.load_state_dict(ac.obs_norm.state_dict())
    with torch.no_grad():
        z_rand = torch.cat([rnd.encode(data["hist"][i:i + 8192].to(env.device))[0].cpu()
                            for i in range(0, len(y), 8192)])
    res = {"n_states": int(len(y)), "mode_counts": {m: int((y == i).sum()) for i, m in enumerate(MODES)},
           "chance": 1 / 7, "near_mm": args.near_mm}
    preds = {}
    for name, X in (("latent", data["z"]), ("current_obs", data["o"]), ("random_encoder", z_rand)):
        pred = fit_probe(X, y, train)
        bal, rec = balanced(pred[test], y[test])
        ci = boot_ci_bal(pred[test], y[test], worlds[test])
        res[name] = dict(balanced_acc=bal, ci=ci, recall=dict(zip(MODES, rec)))
        preds[name] = pred
        print(f"probe[{name}]: balanced acc {bal:.3f} CI {ci[0]:.3f}-{ci[1]:.3f}; recall "
              + ", ".join(f"{m} {r:.2f}" for m, r in zip(MODES, rec)))
    cm = confusion(preds["latent"][test], y[test])
    swap, raw = latent_swap(ac, data)
    res["swap_ratio"] = swap.tolist()
    res["swap_raw"] = raw.tolist()
    json.dump(res, open(str(out) + ".json", "w"), indent=1)
    # figures: confusion, per-step strip, swap heat map
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw=dict(width_ratios=[1, 1.4, 1]))
    ax = axes[0]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 50 else "black")
    ax.set_xticks(range(7), MODES, rotation=45)
    ax.set_yticks(range(7), MODES)
    ax.set_xlabel("decoded mode")
    ax.set_ylabel("true mode")
    ax.set_title(f"latent probe, balanced acc {res['latent']['balanced_acc']:.0%}\n"
                 f"(obs {res['current_obs']['balanced_acc']:.0%}, random enc "
                 f"{res['random_encoder']['balanced_acc']:.0%}, chance 14%)")
    # strip: the test world with the most distinct modes
    ax = axes[1]
    tw = worlds[test].unique()
    best_w = max(tw.tolist()[:300], key=lambda w: len(set(y[worlds == w].tolist())))
    m = worlds == best_w
    tt = data["t"][m].numpy() * 0.02
    ax.step(tt, y[m].numpy(), where="post", label="true", lw=2)
    ax.step(tt, preds["latent"][m].numpy(), where="post", label="decoded", lw=1, ls="--")
    ax.set_yticks(range(7), MODES)
    ax.set_xlabel("time (s)")
    ax.set_title(f"one test start (world {best_w}): true vs decoded mode")
    ax.legend()
    ax = axes[2]
    im = ax.imshow(swap, cmap="magma", vmin=0.8, vmax=max(2.0, np.nanmax(swap)))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, "base" if i == j else f"{swap[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if (swap[i, j] if np.isfinite(swap[i, j]) else 0) < 1.4 else "black")
    ax.set_xticks(range(5), SWAP_MODES, rotation=45)
    ax.set_yticks(range(5), SWAP_MODES)
    ax.set_xlabel("donor latent's mode")
    ax.set_ylabel("receiving state's mode")
    ax.set_title("latent swap: action change / within-mode change")
    fig.suptitle(f"B-R4/B-R5 probe and latent swap (n={len(y):,} states, {args.worlds} worlds; "
                 "train even, test odd worlds)")
    fig.tight_layout()
    fig.savefig(str(out) + ".png", dpi=120)
    print("swap ratios:\n", np.round(swap, 2))


if __name__ == "__main__":
    main()
