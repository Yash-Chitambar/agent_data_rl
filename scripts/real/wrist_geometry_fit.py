"""Where is the real port in the base frame, and the real plug in the hand? No motion, no labels other than the logged joints.

    PYTHONPATH=. python scripts/real/wrist_geometry_fit.py --out results/real/wrist_nets/geometry_fit_trainblocks.json

Renders the sim PC at the logged q_b over the sweep's training blocks (reader out of view), and finds the port offset (mm, in the
anchored port axes) with the best gradient correlation against the real frames, away from the hand and the reader. A local grid
around --start, per-frame scores kept, so the interval is a bootstrap over frames. From the offset, the plug's tip shift in the TCP:
at the anchor the tip is 5.8 mm inside the port (operator jog, full seat). This uses the anchor only through that one pose.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import warp as wp  # noqa: E402
import mujoco_warp as mjw  # noqa: E402
wp.config.quiet = True

import wrist_net_real_eval as W  # noqa: E402
from wrist_sim_at_real import SimAtReal  # noqa: E402


def grad(im):
    g = F.avg_pool2d(im.mean(-1, keepdim=True).permute(0, 3, 1, 2), 3, 1, 1)
    return ((g[..., :, 2:] - g[..., :, :-2])[..., 1:-1, :] ** 2 + (g[..., 2:, :] - g[..., :-2, :])[..., :, 1:-1] ** 2).sqrt()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--start', type=float, nargs=3, default=(-2.34, -3.01, 6.99))
    ap.add_argument('--half', type=float, nargs=3, default=(3., 3., 5.))
    ap.add_argument('--step', type=float, default=.5)
    ap.add_argument('--look', default=None)
    a = ap.parse_args()
    co = W.collect_rows()
    tr, _, _ = W.split_collect(len(co))
    sel = [co[i] for i in tr]
    B = len(sel)
    S = SimAtReal(B, look=a.look)
    dev = S.dev
    ev = [json.loads(x) for x in open(f"data/real_runs/2026-11-02/{co[0]['run']}/events.jsonl")]
    an = [e for e in ev if e['kind'] == 'collect_anchor'][0]
    ql = [e for e in ev if e['kind'] == 'collect_press_removed'][0]['q_left']
    eR, Ra = torch.tensor(an['port_R'], device=dev), torch.tensor(an['tcp_R'], device=dev)
    tip_s = S.R_sr.reshape(3, 3).T @ S.p_sr.reshape(3)
    Y = torch.tensor(np.array([r['p_mm'] for r in sel]), device=dev).float() * 1e-3
    q = torch.tensor([r['q_b'] for r in sel], device=dev)
    sp, sR = S.h.chain['right'].fk(q)
    port0 = ((sp - sR @ tip_s) - (eR @ Y[..., None])[..., 0]).mean(0)
    real = torch.tensor(np.stack([W.read_rgb(r['image']) for r in sel]), device=dev).float()
    Gr = grad(real)
    mask = torch.zeros_like(Gr)
    mask[:, :, 95:235, :] = 1
    mask[:, :, 150:, 130:194] = 0
    ql_t = torch.tensor(ql, device=dev).expand(B, 6)

    def score(off):
        port = port0 + eR @ torch.tensor(off, device=dev, dtype=torch.float32) * 1e-3
        S.set_state(q, ql_t, port.expand(B, 3), eR.expand(B, 3, 3))
        pa = S.rt.lay.peg_qadr
        S.rt.q[:, pa:pa + 3] = torch.tensor([3., 3., 0.], device=dev)
        with wp.ScopedDevice(S.rt.cfg.device):
            mjw.forward(S.rt.m, S.rt.d)
        Gs = grad(S.rt.render()['wrist_cam'].float())
        x, y, m = (Gs * mask).flatten(1), (Gr * mask).flatten(1), mask.flatten(1)
        x = (x - (x.sum(1, keepdim=True) / m.sum(1, keepdim=True))) * m
        y = (y - (y.sum(1, keepdim=True) / m.sum(1, keepdim=True))) * m
        return ((x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + 1e-6)).cpu().numpy()

    axes = [np.arange(c - h, c + h + 1e-6, a.step) for c, h in zip(a.start, a.half)]
    grid = list(itertools.product(*axes))
    sc = np.stack([score(g) for g in grid])            # (grid, frames)
    grid = np.array(grid)
    best = grid[sc.mean(1).argmax()]
    g = np.random.default_rng(0)
    bs = [grid[sc[:, g.integers(0, B, B)].mean(1).argmax()] for _ in range(1000)]
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    off = best
    rep = dict(frames=B, frames_source='training blocks only', best_offset_mm=best.tolist(), ci95_frames=ci.T.round(2).tolist(), score=float(sc.mean(1).max()),
               port_p_m=(port0 + eR @ torch.tensor(off, device=dev, dtype=torch.float32) * 1e-3).tolist(), port_R=an['port_R'],
               offset_mm_port_axes=off.tolist(), tip_shift_tcp_mm=(Ra.T @ (eR @ torch.tensor(off, device=dev, dtype=torch.float32))).tolist(),
               method=__doc__.strip().splitlines()[0])
    Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
