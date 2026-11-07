"""Sim video of an insertion policy (K-J, K3, ...) from real-like hand-overs: a close free camera on port 2, the policy's
contact-head prediction next to the sim's true contact labels on the plug edge (grey free, orange touching, green inside),
and the outcome. Qualitative only. Records the first episode of each world, then re-renders on the CPU.

  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. MUJOCO_GL=egl python scripts/eval/k_sim_video.py --policy runs/k3.pt --cell ring4 --worlds 4 --out k3_ring4.mp4
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
import scripts.eval.kj_real4  # noqa: E402,F401
from scripts.eval.kj_real4 import REAL_LEAD  # noqa: E402
from scripts.eval.k2_matrix import CELLS, PLANT  # noqa: E402
from scripts.eval.kj_start_map import rows_base  # noqa: E402
import scripts.eval.kj_start_map as KSM  # noqa: E402
from blind_insertion.eval_insertion import InsertionPolicy  # noqa: E402
from blind_insertion.train_bimanual import make_env  # noqa: E402
from blind_insertion.envs.insertion_task import failure_class  # noqa: E402

COL = np.array([[150, 150, 150], [240, 140, 25], [40, 165, 75]], np.uint8)  # RGB


def inset(xy, cls, title, size=260):
    img = np.full((size // 2 + 30, size, 3), 255, np.uint8)
    p = xy - xy.mean(0)
    s = (size - 30) / (np.ptp(p[:, 0]) + 1e-9)
    for (x, y), c in zip(p, cls):
        cv2.circle(img, (int(size / 2 + x * s), int(size // 4 + 22 - y * s)), 5, tuple(int(v) for v in COL[c]), -1)
    cv2.putText(img, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--cell', default='ring4')
    ap.add_argument('--worlds', type=int, default=4)
    ap.add_argument('--seed', type=int, default=777)
    ap.add_argument('--plant', default='real')
    ap.add_argument('--out', required=True)
    A = ap.parse_args()
    torch.manual_seed(A.seed)
    pol = InsertionPolicy(A.policy, 'cuda:0')
    ej = dict(pol.meta['env_json'], hard_frac=0., scale_range_frac=0., live_frac=0., extra_ang_frac=0., handover_lead_frac=0.)
    ej.update(PLANT[A.plant])
    env = make_env(argparse.Namespace(nworld=A.worlds, seed=A.seed, action_mode='accumulate', env_json=json.dumps(ej)))
    g = np.random.default_rng(A.seed)
    env._npz = torch.tensor(KSM.cond_rows(CELLS[A.cell], rows_base(), g)[:4096], device=env.device, dtype=torch.float32)
    env.refresh_pool(); env.reset_all()
    q0 = env._measured_model_angles()[:, 0]
    env.joint_target[:, 0] = q0 + torch.as_tensor(REAL_LEAD, dtype=q0.dtype, device=q0.device)
    hist = env.hist
    ac = pol.members[0][0]
    rt = env.runtime
    N, T = env.N, env.max_steps
    rec = {w: [] for w in range(N)}
    done = np.zeros(N, bool)
    outcome = [None] * N
    xy = env.edge[:, :2].cpu().numpy()
    for t in range(T + 2):
        h = hist[..., :pol.obs_dim].clone()
        if pol.mask_idx:
            h[..., list(pol.mask_idx)] = 0
        with torch.no_grad():
            z, _ = ac.encode(h)
            pred = ac.contact_logits(z).argmax(-1).cpu().numpy()
            a = pol(hist).clamp(-1, 1)
        true = env.current_labels().cpu().numpy()
        p, _ = rt.plug_in_port()
        qpos = rt.q.cpu().numpy().copy()
        for w in range(N):
            if not done[w]:
                rec[w].append((qpos[w], pred[w], true[w], p[w].cpu().numpy() * 1e3, t * env.dt))
        hist, _, _, d, info = env.step(a)
        if 'final' in info:
            cls = failure_class(info['final'])
            for j, w in enumerate(info['final_ids'].tolist()):
                if not done[w]:
                    done[w], outcome[w] = True, cls[j]
        if done.all():
            break
    m = rt.mjm
    d = mujoco.MjData(m)
    R = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    port_p, port_R = (x.cpu().numpy() for x in env._port_pose(env.pc_start))
    writer = imageio.get_writer(A.out, fps=25, codec='libx264', quality=7)
    for w in range(N):
        Rp = port_R[w]
        v = Rp[:, 2] + 0.55 * Rp[:, 0] + 0.45 * np.array([0, 0, 1.])     # outside the port, to the side and above
        v /= np.linalg.norm(v)
        f = -v
        cam.lookat[:] = port_p[w]
        cam.distance = 0.12
        cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
        cam.elevation = float(np.degrees(np.arcsin(f[2])))
        for k, (qp, pr, tr, pp, ts) in enumerate(rec[w]):
            if k % 2:
                continue
            d.qpos[:] = qp
            mujoco.mj_forward(m, d)
            R.update_scene(d, cam)
            img = R.render().copy()
            col = (40, 165, 75) if outcome[w] == 'seated' else (200, 40, 40)
            txt = f'{A.label}  {A.cell}  world {w}  t {ts:.2f} s  tip depth {-pp[2]:.1f} mm  lateral {np.hypot(pp[0], pp[1]):.1f} mm'
            cv2.putText(img, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f'outcome: {outcome[w]}', (10, 44), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2, cv2.LINE_AA)
            side = np.vstack([inset(xy, pr, 'policy contact head'), inset(xy, tr, 'sim truth')])
            side = np.vstack([side, np.full((480 - side.shape[0], side.shape[1], 3), 255, np.uint8)])
            writer.append_data(np.hstack([img, side]))
        for _ in range(12):
            writer.append_data(np.hstack([img, side]))
    writer.close()
    R.close()
    print(json.dumps(dict(out=A.out, outcomes=outcome)))
