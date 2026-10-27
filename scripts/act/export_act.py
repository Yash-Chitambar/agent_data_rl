"""Export a trained LeRobot ACT checkpoint to TorchScript for the simulation venv (LeRobot venv).

The traced module takes (state (B, 7 or 13: gen_demos --cmd_err), top (B, 3, H, W), wrist (B, 3, H, W)) with images in
[0, 1] and returns the unnormalized action chunk (B, chunk, 8). Normalization uses the
dataset's mean/std, as LeRobot's ACT preprocessor does (MEAN_STD for state, images, action).

    .venv-act/bin/python scripts/act/export_act.py --ckpt runs/H-act-pick-v1/checkpoints/last/pretrained_model \
        --dataset data/act/reader_pick_v1 --out runs/H-act-pick-v1/act_traced.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from lerobot.policies.act.modeling_act import ACTPolicy


class Wrapped(torch.nn.Module):
    def __init__(self, model, stats):
        super().__init__()
        self.model = model
        t = lambda k, f: torch.tensor(stats[k][f], dtype=torch.float32)  # noqa: E731
        self.register_buffer("s_mean", t("observation.state", "mean"))
        self.register_buffer("s_std", t("observation.state", "std"))
        self.register_buffer("a_mean", t("action", "mean"))
        self.register_buffer("a_std", t("action", "std"))
        for cam in ("top", "wrist"):
            k = f"observation.images.{cam}"
            self.register_buffer(f"{cam}_mean", t(k, "mean").reshape(1, 3, 1, 1))
            self.register_buffer(f"{cam}_std", t(k, "std").reshape(1, 3, 1, 1))

    def forward(self, state, top, wrist):
        eps = 1e-8
        batch = {"observation.state": (state - self.s_mean) / (self.s_std + eps),
                 "observation.images": [(top - self.top_mean) / (self.top_std + eps),
                                        (wrist - self.wrist_mean) / (self.wrist_std + eps)]}
        a = self.model(batch)[0]
        return a * self.a_std + self.a_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pol = ACTPolicy.from_pretrained(a.ckpt).eval().to("cuda")
    assert list(pol.config.image_features) == ["observation.images.top", "observation.images.wrist"], pol.config.image_features
    stats = json.loads((Path(a.dataset) / "meta" / "stats.json").read_text())
    w = Wrapped(pol.model, stats).eval().to("cuda")
    H, W = pol.config.image_features["observation.images.top"].shape[1:]
    ex = (torch.zeros(2, len(stats["observation.state"]["mean"]), device="cuda"), torch.rand(2, 3, H, W, device="cuda"), torch.rand(2, 3, H, W, device="cuda"))
    with torch.no_grad():
        tr = torch.jit.trace(w, ex, check_trace=False)
        ref, out = w(*ex), tr(*ex)
    print("trace max diff", (ref - out).abs().max().item(), "chunk", tuple(out.shape))
    tr.save(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
