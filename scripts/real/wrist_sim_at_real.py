"""Render the sim at the logged real poses of the collect run (or A / A2), nominal look, for side-by-side geometry checks.

    PYTHONPATH=. python scripts/real/wrist_sim_at_real.py --out results/real/wrist_nets/sim_at_real [--body 0.0296]

The right arm is at the logged q_b; the left at the logged press-removed q; the PC port is placed so that the plug tip is at
the frame's label (--port label) or at the ZED estimate (--port zed). Reader: measured mean in-hand pose. Writes frames.npy
(N, 320, 320, 3 RGB), meta.json (camera pose, port and tip pixels by pinhole).
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw
wp.config.quiet = True

from blind_insertion.envs import kinematics as K  # noqa: E402
from blind_insertion.envs.inhand import measured_mean  # noqa: E402
from blind_insertion.perception.render_data import ApproachRenderer  # noqa: E402

LAYOUT = 'results/usbc/policy/pick/task_layout.json'


def port_R(yaw_deg):
    y = math.radians(yaw_deg)
    z = np.array([-math.cos(y), -math.sin(y), 0.])
    up = np.array([0., 0., 1.])
    return torch.tensor(np.c_[np.cross(up, z), up, z], dtype=torch.float32)


def load_rows(source=('collect',), man='data/real_training/port_v1/manifest.jsonl'):
    return [r for r in map(json.loads, open(man)) if r['source'] in source]


class SimAtReal:
    def __init__(self, nworld, seed=0, visual_dr=None, reader_look=None, look=None):
        self.r = ApproachRenderer(nworld=nworld, seed=seed, task_layout=LAYOUT, visual_dr=visual_dr, inhand_real=reader_look, pc_real_look=look)
        self.rt, self.h = self.r.rt, self.r.h
        self.n, self.dev = nworld, self.rt.device
        self.h.reset()
        self.R_sr, self.p_sr = (x.to(self.dev).float() for x in measured_mean(device=self.dev))
        self.fq = None

    def set_state(self, q_b, q_left, port_p, port_R_, R_sr=None, p_sr=None):
        """q_b (n, 6) true joint angles; port_p (n, 3) port origin, port_R_ (n, 3, 3)."""
        rt, dev = self.rt, self.dev
        R_sr = self.R_sr.reshape(1, 3, 3).expand(self.n, 3, 3) if R_sr is None else R_sr
        p_sr = self.p_sr.reshape(1, 3).expand(self.n, 3) if p_sr is None else p_sr
        qa, la = rt.indices['right'][0], rt.indices['left'][0]
        rt.q[:, qa] = q_b
        rt.q[:, la] = q_left
        rt.q[:, self.r.left_fingers] = 0.
        fa = rt.pc_qadr
        rt.q[:, fa:fa + 3] = port_p
        rt.q[:, fa + 3:fa + 7] = K.mat_to_quat(port_R_)
        sp, sR = self.h.chain['right'].fk(q_b)
        reader_R = sR @ R_sr.transpose(1, 2)
        reader_p = sp - (reader_R @ p_sr[..., None])[..., 0]
        pa = rt.lay.peg_qadr
        rt.q[:, pa:pa + 3] = reader_p
        rt.q[:, pa + 3:pa + 7] = K.mat_to_quat(reader_R)
        fing = self.r.ins.fingers
        if self.fq is None:
            self.fq = self.r.ins._finger_q(rt.q[0])
        rt.q[:, fing] = self.fq
        rt.v[:] = 0
        with wp.ScopedDevice(rt.cfg.device):
            mjw.forward(rt.m, rt.d)
        return reader_p, reader_R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--port', choices=('label', 'zed'), default='label')
    ap.add_argument('--source', nargs='*', default=['collect'])
    a = ap.parse_args()
    rows = load_rows(tuple(a.source))
    run = rows[0]['run']
    ev = [json.loads(x) for x in open(f'data/real_runs/2026-11-02/{run}/events.jsonl')]
    pc = [e for e in ev if e['kind'] == 'estimate' and e['est'] == 'zed_pc'][-1]
    ql = [e for e in ev if e['kind'] == 'collect_press_removed'][0]['q_left']
    an = [e for e in ev if e['kind'] == 'collect_anchor'][0]
    B = 64
    S = SimAtReal(B)
    dev = S.dev
    print('PC z, nominal q', S.rt.q[0, S.rt.pc_qadr:S.rt.pc_qadr + 7].tolist())
    print('camera names', S.rt.camera_names)
    eR = torch.tensor(an['port_R'], device=dev)
    z0 = float(S.rt.q[0, S.rt.pc_qadr + 2])
    out = []
    for i in range(0, len(rows), B):
        b = rows[i:i + B]
        k = len(b)
        q = torch.tensor([r['q_b'] for r in b] + [b[-1]['q_b']] * (B - k), device=dev)
        p = torch.tensor([r['p_mm'] for r in b] + [b[-1]['p_mm']] * (B - k), device=dev) * 1e-3
        sp, sR = S.h.chain['right'].fk(q)
        tip_s = (S.R_sr.reshape(3, 3).T @ S.p_sr.reshape(3))
        tip = sp - (sR @ tip_s)
        if a.port == 'label':
            port = tip - (eR @ p[..., None])[..., 0]
        else:
            port = torch.tensor(pc['port_xy'] + [z0], device=dev).expand(B, 3)
        S.set_state(q, torch.tensor(ql, device=dev).expand(B, 6), port, eR.expand(B, 3, 3))
        fr = S.rt.render()['wrist_cam']
        out.append(fr[:k].cpu().numpy())
        if i == 0:
            print('port z', port[:3].tolist(), 'z0', z0, 'tip', tip[:3].tolist())
    out = np.concatenate(out)
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, f'frames_{a.port}.npy'), out)
    print(out.shape)


if __name__ == '__main__':
    main()
