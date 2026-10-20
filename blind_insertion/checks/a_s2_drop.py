"""A-S2: drop test for every variant in the batched (MuJoCo Warp) env.

The peg is released (gripper open, arm parked) centred and aligned 1 mm above its hole;
it must fall to the floor without touching the walls.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from blind_insertion.envs import kinematics as K  # noqa: E402
from blind_insertion.envs.peg_env import EnvCfg, PegEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/A")
    ap.add_argument("--n_variants", type=int, default=1024)
    ap.add_argument("--variant_seed", type=int, default=0)
    ap.add_argument("--height", type=float, default=0.001)
    ap.add_argument("--tag", default="study")
    args = ap.parse_args()
    out = Path(args.out)
    n = args.n_variants
    env = PegEnv(EnvCfg(nworld=n, n_variants=n, variant_seed=args.variant_seed, episode_s=100,
                        far_z=1e6, far_xy_margin=1e6, slip_dist=1e6, seed=5, auto_reset=False, fix_tilt_deg=0.0))
    lay = env.lay
    dev = env.device
    # open the gripper, park the arm high, release the peg centred above the hole
    park = env.home.clone()
    park[0] += 1.6     # swing the arm 90 degrees away from the fixtures
    env.t_qpos[:, torch.as_tensor(lay.arm_qadr, device=dev)] = park
    env.t_ctrl[:, torch.as_tensor(lay.arm_act, device=dev)] = park
    for a in lay.finger_qadr:
        env.t_qpos[:, int(a)] = 0.0
    for g in lay.grip_act:
        env.t_ctrl[:, int(g)] = -5.0 if lay.robot.name == "flexiv" else 5.0   # hold open
    pa = lay.peg_qadr
    p = env.fix_p + (env.fix_R @ torch.tensor([0, 0, args.height], device=dev))
    env.t_qpos[:, pa:pa + 3] = p
    env.t_qpos[:, pa + 3:pa + 7] = K.mat_to_quat(env.fix_R)
    env.t_qvel[:] = 0
    env._q_meas = park.expand(n, -1).clone()
    lat, zs = [], []
    for t in range(25):   # 0.5 s
        env.step(torch.zeros(n, env.n_arm, device=dev))
        pp, R, _, _ = env.peg_in_fixture()
        lat.append(pp[:, :2].norm(dim=1).cpu().numpy())
        zs.append(pp[:, 2].cpu().numpy())
    lat, zs = np.array(lat), np.array(zs)
    depth = env.depth[env.variant].cpu().numpy()
    seated = zs[-1] < -(depth - 0.0005)
    df = pd.DataFrame(dict(variant=np.arange(n), clearance=[v.clearance for v in lay.variants],
                           depth=depth, final_z=zs[-1], max_lateral=lat.max(0), seated=seated))
    df.to_csv(out / f"A-S2_drop_{args.tag}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(1000 * df.clearance, 1000 * (df.final_z + df.depth), s=6, c=np.where(seated, "#4477AA", "#EE6677"))
    axes[0].set_xlabel("clearance per side (mm)")
    axes[0].set_ylabel("final height above floor (mm)")
    axes[0].set_title(f"seated {seated.sum()}/{n}")
    axes[1].hist(1000 * df.max_lateral, bins=40, color="#4477AA")
    axes[1].axvline(1000 * df.clearance.min(), color="k", ls="--", label="min clearance")
    axes[1].set_xlabel("max lateral drift during the drop (mm)")
    axes[1].legend()
    fig.suptitle(f"A-S2 drop test in MuJoCo Warp, every variant ({args.tag}, n={n}), released {1000*args.height:.1f} mm above")
    fig.tight_layout()
    fig.savefig(out / f"A-S2_drop_{args.tag}.png", dpi=120)
    print(f"A-S2 {args.tag}: seated {seated.sum()}/{n}; max lateral drift {1000*lat.max():.3f} mm "
          f"(min clearance {1000*df.clearance.min():.3f} mm) -> {'PASS' if seated.all() else 'FAIL'}")


if __name__ == "__main__":
    main()
