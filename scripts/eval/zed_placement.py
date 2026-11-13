"""ZED perception against the placement (sim, batched): the PC port estimate and its gate, and the reader estimate and its
gate, with the PC and the reader at forced poses (one pose per world, a grid around the real spot), at the capture pose
of the real chain (eval_chain.run_batch: the 'start' frame), with the visual DR of the chain. Nets from the config's
chain_args (zed_v1 + zed_v1_gate_arm) and pick.reader_estimator (reader_zed_cap).

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. MUJOCO_GL=egl python scripts/eval/zed_placement.py --out results/placement/zed_placement.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REAL_PC = (0.231, 0.394, 4.0)
REAL_RD = (0.131, 0.448, 92.0)


def forced_layout(pc, rd):
    """BimanualRuntime._sample_layout replacement: world i gets pc[i] (port x, y, yaw deg) and rd[i] (plug-tip origin x, y, yaw)."""
    from blind_insertion.envs import bimanual_runtime as BR

    def f(self, ids):
        dev = self.device
        P, Q = torch.as_tensor(pc, dtype=torch.float32, device=dev)[ids], torch.as_tensor(rd, dtype=torch.float32, device=dev)[ids]
        for qadr, ref, X, body in ((self.pc_qadr, self._pc0, P, False), (self.lay.peg_qadr, self._reader0, Q, True)):
            a = torch.deg2rad(X[:, 2])
            c, s_ = torch.cos(a), torch.sin(a)
            p0 = ref[:3]
            n = len(ids)
            if body:
                pos = torch.stack([X[:, 0], X[:, 1], p0[2].expand(n)], -1)
            else:
                pos = torch.stack([X[:, 0] + c * p0[0] - s_ * p0[1], X[:, 1] + s_ * p0[0] + c * p0[1], p0[2].expand(n)], -1)
            qz = torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], -1)
            w1, x1, y1, z1 = qz.unbind(-1)
            w2, x2, y2, z2 = ref[3:]
            quat = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
            self.q[ids, qadr:qadr + 3] = pos
            self.q[ids, qadr + 3:qadr + 7] = quat
    BR.BimanualRuntime._sample_layout = f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align8_nojump.json'))
    ap.add_argument('--reps', type=int, default=16, help='worlds per pose')
    ap.add_argument('--seed', type=int, default=8100)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reader-net', default=None, help='reader estimator name in runs/perception (default: the config pick.reader_estimator)')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    poses = []
    for dx in (-40, -30, -20, -10, 0, 10, 20, 30, 40):
        poses.append(('pc', (REAL_PC[0] + dx * 1e-3, REAL_PC[1], REAL_PC[2]), REAL_RD, f'pc dx {dx:+d}'))
    for dy in (-60, -45, -30, -20, -10, 10, 20, 30, 40):
        poses.append(('pc', (REAL_PC[0], REAL_PC[1] + dy * 1e-3, REAL_PC[2]), REAL_RD, f'pc dy {dy:+d}'))
    for dyaw in (-20, -15, -10, -5, 5, 10, 15, 20, 25):
        poses.append(('pc', (REAL_PC[0], REAL_PC[1], REAL_PC[2] + dyaw), REAL_RD, f'pc yaw {dyaw:+d}'))
    for dx in (-60, -40, -20, 20, 40, 60):
        poses.append(('rd', REAL_PC, (REAL_RD[0] + dx * 1e-3, REAL_RD[1], REAL_RD[2]), f'rd dx {dx:+d}'))
    for dy in (-60, -40, -20, 20, 40, 60):
        poses.append(('rd', REAL_PC, (REAL_RD[0], REAL_RD[1] + dy * 1e-3, REAL_RD[2]), f'rd dy {dy:+d}'))
    for yaw in (50, 60, 70, 80, 100, 110, 120, 130):
        poses.append(('rd', REAL_PC, (REAL_RD[0], REAL_RD[1], yaw), f'rd yaw {yaw}'))
    poses.insert(0, ('pc', REAL_PC, REAL_RD, 'real'))
    pc = [p[1] for p in poses for _ in range(a.reps)]
    rd = [p[2] for p in poses for _ in range(a.reps)]
    n = len(pc)
    forced_layout(pc, rd)
    from deploy.bimanual.io_sim import chain_argv, eval_chain
    EC = eval_chain()
    args = EC.parse(chain_argv(cfg, cfg['insert'].get('policy_file', 'runs/insert_K4.pt'), n))
    ch = EC.Chain(args)
    torch.manual_seed(a.seed)
    if hasattr(ch.rt.visual, 'gen'):
        ch.rt.visual.gen.manual_seed(a.seed)
    rt = ch.rt
    ch.pick.reset()
    lq = rt.indices['left'][0]
    rt.q[:, lq] = rt.q[:, lq] + ch.pick.enc_error[:, 1]
    ch.forward()
    p_true, R_true = ch.h.pc_pose()
    yaw_true = torch.atan2(-R_true[:, 1, 2], -R_true[:, 0, 2])
    la, ra, fq = rt.indices['left'][0], rt.indices['right'][0], ch.pick.finger_qadr
    q_park, q_r, q_f = rt.q[:, la].clone(), rt.q[:, ra].clone(), rt.q[:, fq].clone()
    ee = ch.pick.enc_error
    rt.q[:, la] = ch.zed_q['left'] + ee[:, 1]
    rt.q[:, ra] = ch.zed_q['right'] + ee[:, 0]
    rt.q[:, fq] = .0475
    ch.forward()
    xy, eyaw, ok, info = ch.zed_estimate()
    pc_err = ((xy - p_true[:, :2]).norm(dim=1) * 1e3).cpu().numpy()
    pc_yaw_err = torch.rad2deg(torch.atan2(torch.sin(eyaw - yaw_true), torch.cos(eyaw - yaw_true))).cpu().numpy()
    pc_ok = ok.cpu().numpy()
    rt.q[:, la], rt.q[:, ra], rt.q[:, fq] = q_park, q_r, q_f
    ch.forward()
    # reader: the chain's reader estimator on its own frame (left parked, right at the capture pose, jaws open)
    from blind_insertion.pipeline.pick_stage import ZedReader
    zr = ZedReader(str(ROOT / 'runs/perception' / ((a.reader_net or cfg['pick']['reader_estimator']) + '.pt')), ch.dev, retries=0)
    rp, rR, rok = zr(ch)
    tr = ch.pick.reader_pose_q()
    rd_err = ((rp[:, :2] - tr[:, :2]).norm(dim=1) * 1e3).cpu().numpy()
    ryaw = torch.atan2(-rR[:, 1, 2], -rR[:, 0, 2])
    rd_t = torch.deg2rad(torch.as_tensor([r[2] for r in rd], device=ch.dev))
    rd_yaw_err = torch.rad2deg(torch.atan2(torch.sin(ryaw - rd_t), torch.cos(ryaw - rd_t))).cpu().numpy()
    rd_ok = rok.cpu().numpy()
    rows = []
    for i, p in enumerate(poses):
        s = slice(i * a.reps, (i + 1) * a.reps)
        rows.append(dict(tag=p[3], kind=p[0], pc=list(p[1]), reader=list(p[2]), n=a.reps,
                         pc_ok=int(pc_ok[s].sum()), pc_err_p50=float(np.median(pc_err[s])), pc_err_max=float(pc_err[s].max()),
                         pc_yaw_err_p50=float(np.median(np.abs(pc_yaw_err[s]))), pc_yaw_err_max=float(np.abs(pc_yaw_err[s]).max()),
                         rd_ok=int(rd_ok[s].sum()), rd_err_p50=float(np.median(rd_err[s])), rd_err_max=float(rd_err[s].max()),
                         rd_yaw_err_p50=float(np.median(np.abs(rd_yaw_err[s]))), rd_yaw_err_max=float(np.abs(rd_yaw_err[s]).max())))
        r = rows[-1]
        print(f"{r['tag']:12s} PC gate {r['pc_ok']}/{r['n']} err {r['pc_err_p50']:.1f}/{r['pc_err_max']:.1f} mm yaw {r['pc_yaw_err_p50']:.1f}/{r['pc_yaw_err_max']:.1f} | "
              f"reader gate {r['rd_ok']}/{r['n']} err {r['rd_err_p50']:.1f}/{r['rd_err_max']:.1f} mm yaw {r['rd_yaw_err_p50']:.1f}/{r['rd_yaw_err_max']:.1f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
