"""Sim video of the final alignment (AlignTask): step stage or continuous servo, close camera on port 2, overlay of the
vision estimate, true error and tip speed. Qualitative only.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. MUJOCO_GL=egl python scripts/eval/align_sim_video.py --method servo --ff --worlds 3 --out servo.mp4
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.train_bimanual import make_env  # noqa: E402
from blind_insertion.envs.align_task import ServoPolicy, StepPolicy  # noqa: E402
from scripts.eval.align_compare import ENV  # noqa: E402

FF = {"tau": [0.0275, 0.546, 0.819, 0.0463, 0.0433, 0.0467], "ramp": 0.005}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', default='servo', choices=('servo', 'step'))
    ap.add_argument('--ff', action='store_true')
    ap.add_argument('--worlds', type=int, default=3)
    ap.add_argument('--seed', type=int, default=55)
    ap.add_argument('--seconds', type=float, default=25.)
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    ej = dict(ENV, episode_s=A.seconds + 1, align_done_on='vision')
    if A.ff:
        ej['right_fric_comp'] = FF
    torch.manual_seed(A.seed)
    env = make_env(argparse.Namespace(nworld=A.worlds, seed=A.seed, action_mode='accumulate', env_json=json.dumps(ej)))
    pol = ServoPolicy(env, vmin_mm_s=1.5, deadband_mm=0.4, deadband_deg=0.4) if A.method == 'servo' else StepPolicy(env)
    N = env.N
    rec = {w: [] for w in range(N)}
    done = np.zeros(N, bool)
    outcome = [''] * N
    prev = env._truth()[:, :3].clone()
    for t in range(int(A.seconds / env.dt)):
        a = pol()
        tr = env._truth()
        sp = ((tr[:, :3] - prev).norm(dim=1) / env.dt).cpu().numpy()
        prev = tr[:, :3].clone()
        est = env.est_now().cpu().numpy()
        q = env.runtime.q.cpu().numpy().copy()
        for w in range(N):
            if not done[w]:
                rec[w].append((q[w], tr[w].cpu().numpy(), est[w], sp[w], t * env.dt))
        _, _, _, d, info = env.step(a)
        if 'final' in info:
            f = info['final']
            for j, w in enumerate(info['final_ids'].tolist()):
                if not done[w]:
                    done[w] = True
                    outcome[w] = 'ALIGNED' if bool(f['success'][j]) else 'failed'
            pol.reset(info['final_ids'])
        if done.all():
            break
    m = env.runtime.mjm
    dd = mujoco.MjData(m)
    R = mujoco.Renderer(m, 480, 720)
    cam = mujoco.MjvCamera()
    port_p, port_R = (x.cpu().numpy() for x in env._port_pose(env.pc_start))
    name = ('servo' if A.method == 'servo' else 'step stage') + (' + friction FF' if A.ff else '')
    writer = imageio.get_writer(A.out, fps=25, codec='libx264', quality=7)
    for w in range(N):
        Rp = port_R[w]
        v = Rp[:, 2] + 0.9 * Rp[:, 0] + 0.35 * np.array([0, 0, 1.])
        v /= np.linalg.norm(v)
        cam.lookat[:] = port_p[w] + Rp[:, 2] * 0.008
        cam.distance, cam.azimuth, cam.elevation = 0.14, float(np.degrees(np.arctan2(-v[1], -v[0]))), float(np.degrees(np.arcsin(-v[2])))
        for k, (qp, tr, est, sp, ts) in enumerate(rec[w]):
            if k % 2:
                continue
            dd.qpos[:] = qp
            mujoco.mj_forward(m, dd)
            R.update_scene(dd, cam)
            img = R.render().copy()
            e = tr[:3] - np.array([0, 0, 6.])
            lines = [f'{name}   world {w}   t {ts:5.2f} s', f'true error: across {e[0]:+.1f}  up {e[1]:+.1f}  gap {e[2]:+.1f} mm',
                     f'vision estimate: {est[0]:+.1f} {est[1]:+.1f} {est[2]:.1f} mm    tip speed {sp:4.1f} mm/s']
            for i, s in enumerate(lines):
                cv2.putText(img, s, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
            if k >= len(rec[w]) - 2:
                cv2.putText(img, outcome[w] or 'time limit', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, .8, (40, 200, 80) if outcome[w] == 'ALIGNED' else (220, 50, 50), 2, cv2.LINE_AA)
            writer.append_data(img)
        for _ in range(15):
            writer.append_data(img)
    writer.close(); R.close()
    print(json.dumps(dict(out=A.out, outcomes=outcome, t_end=[r[-1][-1] for r in rec.values()])))
