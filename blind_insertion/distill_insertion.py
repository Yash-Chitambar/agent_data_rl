"""DAgger distillation of the privileged Part 4 teacher into deployable students (what the wrist camera adds).

Variants (POLICY_PLAN Part 4), all under the same physics DR plus visual DR (BimanualCfg.visual_dr
defaults, measured camera cadence, RGB only):
  blind           proprio history only (control: same pipeline as the camera students)
  wrist           proprio + right-wrist RGB
  wrist_approach  proprio + right-wrist RGB for the first `--vis_s` seconds of the episode, then masked
                  (camera for the approach, proprio for contact)
  wrist_zed       proprio + right-wrist RGB + ZED (top) RGB

The student acts (after a short teacher-driven warm start, beta decays to 0), the teacher labels every
visited state with its mean action (DAgger); loss = MSE to the teacher action + contact-label CE.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python -m blind_insertion.distill_insertion \
        --teacher runs/P4-insert-teacher-acc-s0/last.pt --variant wrist --name P4-student-wrist-s0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from blind_insertion.eval_insertion import load_policy
from blind_insertion.policies.tcn_actor_critic import TCN, RunningNorm, mlp
from blind_insertion.policies.visual import FrameCNN
from blind_insertion.train import RUNS, save_atomic, git_hash
from blind_insertion.train_bimanual import make_env, set_scale

CAMS = {'blind': (), 'wrist': ('wrist_cam',), 'wrist_approach': ('wrist_cam',), 'wrist_zed': ('wrist_cam', 'top')}
PROPRIO = 62


class Student(nn.Module):
    def __init__(self, cams, res, history=30, latent=32, emb=64, n_points=64, act_dim=6):
        super().__init__()
        self.cams, self.res = tuple(cams), res
        self.obs_norm = RunningNorm(PROPRIO)
        self.tcn = TCN(PROPRIO, history, latent)
        self.cnn = nn.ModuleDict({c: FrameCNN(emb, res) for c in self.cams})
        d = latent + PROPRIO + len(self.cams) * (emb + 1) + 1
        self.actor = mlp(d, (256, 256, 128), act_dim)
        self.decoder = mlp(latent, (128, 128), n_points * 3)
        self.n_points = n_points

    def forward(self, hist, frames=None, ages=None, mask=None):
        hn = self.obs_norm(hist)
        z = self.tcn(hn)
        parts = [z, hn[:, -1]]
        m = mask if mask is not None else torch.ones(hist.shape[0], device=hist.device)
        for c in self.cams:
            e = self.cnn[c](frames[c].permute(0, 3, 1, 2))
            parts += [e * m[:, None], (ages[c].clamp(max=0.2) * 10 * m)[:, None]]
        parts.append(m[:, None])
        return self.actor(torch.cat(parts, 1)), self.decoder(z).reshape(-1, self.n_points, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--teacher', required=True)
    ap.add_argument('--variant', required=True, choices=list(CAMS))
    ap.add_argument('--name', required=True)
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--res', type=int, default=96)
    ap.add_argument('--steps', type=float, default=3e6)
    ap.add_argument('--scale', type=float, default=1.25, help='start-error scale (1 = nominal DR)')
    ap.add_argument('--vis_s', type=float, default=1.0)
    ap.add_argument('--beta_steps', type=float, default=3e5, help='env steps over which teacher execution decays to 0')
    ap.add_argument('--buffer', type=int, default=120, help='replay buffer length in control steps (x nworld samples)')
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--updates_per_step', type=int, default=1)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_visual_dr', action='store_true')
    ap.add_argument('--init', default='', help='start the student from this checkpoint (diagnostics: --lr 0 freezes it)')
    ap.add_argument('--refresh_every', type=int, default=0, help='>0: env.refresh_pool() every this many steps')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device('cuda:0')
    teacher, tck = load_policy(a.teacher, dev)
    cams = CAMS[a.variant]
    env_json = json.loads(tck.get('env_json') or '{}') | {'privileged_actor': True}
    if cams:
        env_json |= {'render_rgb': True, 'visual_dr': not a.no_visual_dr, 'camera_resolution': a.res,
                     'camera_names': list(cams)}
    args = argparse.Namespace(nworld=a.nworld, seed=a.seed, action_mode=tck.get('action_mode', 'accumulate'),
                              env_json=json.dumps(env_json))
    env = make_env(args)
    set_scale(env, a.scale)
    env.reset_all()
    run = RUNS / a.name
    run.mkdir(parents=True, exist_ok=True)
    json.dump(dict(args=vars(a), env_json=env_json, git=git_hash(), cams=cams), open(run / 'config.json', 'w'), indent=1)
    st = Student(cams, a.res, history=env.icfg.history, n_points=env.n_labels).to(dev)
    if a.init:
        st.load_state_dict(torch.load(a.init, map_location=dev, weights_only=False)['model'])
    opt = torch.optim.Adam(st.parameters(), lr=max(a.lr, 1e-12))
    N, L, H = env.N, a.buffer, env.icfg.history
    B = {'hist': torch.zeros(L, N, H, PROPRIO, device=dev), 'act': torch.zeros(L, N, 6, device=dev),
         'lab': torch.zeros(L, N, env.n_labels, dtype=torch.long, device=dev), 'mask': torch.zeros(L, N, device=dev)}
    for c in cams:
        B['f_' + c] = torch.zeros(L, N, a.res, a.res, 3, dtype=torch.uint8, device=dev)
        B['a_' + c] = torch.zeros(L, N, device=dev)
    rt = env.runtime
    hist, _ = env.observe()
    cw = torch.tensor((0.25, 3.0, 2.0), device=dev)
    total, k, filled = 0, 0, 0
    t0 = time.time()
    log_f = open(run / 'train_log.csv', 'w', newline='')
    log = None
    succ_n, ep_n, loss_acc, nupd = 0, 0, 0., 0
    it = 0
    while total < a.steps:
        it += 1
        with torch.no_grad():
            frames, ages = rt.camera_sensor.observation(rt.sensor_time) if cams else ({}, {})
            mask = (env.steps.float() * env.dt < a.vis_s).float() if a.variant == 'wrist_approach' else \
                torch.ones(N, device=dev)
            t_act, _ = teacher.act_mean(hist)
            sh = hist[..., :PROPRIO]
            st.obs_norm.update(sh[:, -1])
            s_act, _ = st(sh, frames, ages, mask)
            beta = max(0., 1 - total / a.beta_steps)
            use_t = torch.rand(N, device=dev) < beta
            act = torch.where(use_t[:, None], t_act, s_act)
            j = k % L
            B['hist'][j] = sh; B['act'][j] = t_act; B['lab'][j] = env.current_labels(); B['mask'][j] = mask
            for c in cams:
                B['f_' + c][j] = frames[c]; B['a_' + c][j] = ages[c]
            k += 1; filled = min(filled + 1, L)
            hist, _, _, done, info = env.step(act)
            total += N
            if a.refresh_every and it % a.refresh_every == 0:
                env.refresh_pool()
            if 'final' in info:
                f = info['final']
                stud = ~use_t[info['final_ids']]       # outcome of an episode whose last step the student acted
                succ_n += int(f['success'][stud].sum()); ep_n += int(stud.sum())
        if filled >= 8 and a.lr > 0:
            st.train()
            for _ in range(a.updates_per_step):
                ti = torch.randint(filled, (a.batch,), device=dev)
                wi = torch.randint(N, (a.batch,), device=dev)
                fr = {c: B['f_' + c][ti, wi] for c in cams}
                ag = {c: B['a_' + c][ti, wi] for c in cams}
                mu, logits = st(B['hist'][ti, wi], fr, ag, B['mask'][ti, wi])
                lab = B['lab'][ti, wi]
                loss = F.mse_loss(mu, B['act'][ti, wi]) + 0.2 * F.cross_entropy(logits.reshape(-1, 3), lab.reshape(-1), weight=cw)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(st.parameters(), 1.0)
                opt.step()
                loss_acc += loss.item(); nupd += 1
        if it % 200 == 0:
            row = dict(steps=total, wall=time.time() - t0, beta=beta, loss=loss_acc / max(nupd, 1),
                       student_success=succ_n / max(ep_n, 1), episodes=ep_n, sps=total / (time.time() - t0))
            if log is None:
                log = csv.DictWriter(log_f, fieldnames=list(row)); log.writeheader()
            log.writerow(row); log_f.flush()
            print(f"[{a.name}] {total/1e6:.2f}M beta {beta:.2f} loss {row['loss']:.4f} student succ {row['student_success']:.3f} "
                  f"(n {ep_n}) sps {row['sps']:.0f}", flush=True)
            succ_n, ep_n, loss_acc, nupd = 0, 0, 0., 0
        if it % 2000 == 0:
            save_atomic(dict(model=st.state_dict(), cams=cams, res=a.res, variant=a.variant, vis_s=a.vis_s,
                             history=H, n_points=env.n_labels, env_json=env_json, steps=total,
                             action_mode=args.action_mode), run / 'last.pt')
    save_atomic(dict(model=st.state_dict(), cams=cams, res=a.res, variant=a.variant, vis_s=a.vis_s, history=H,
                     n_points=env.n_labels, env_json=env_json, steps=total, action_mode=args.action_mode), run / 'last.pt')
    (run / 'DONE').write_text(json.dumps(dict(steps=total, wall_s=time.time() - t0)))


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    main()
