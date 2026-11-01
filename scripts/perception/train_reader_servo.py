"""Wrist-camera servo net for the Part 1 pick descent (RGB only): from one right-wrist frame, the grasp pose
RELATIVE TO THE HAND (grasp_site frame): translation t = R_h^T (p_g - p_h) (m) and rotation vector
r = log(R_h^T R_g) (rad). Relative to the hand, so the believed-frame expert composes it with its own
encoder-FK hand pose (T_g = FK_enc(q) * Delta): encoder-offset errors cancel up to the error of the
remaining (short) relative move, and the tool tilt error from the offsets is corrected too.

Training views: the hand anywhere along the approach line from the pre-grasp hover (f = 0) to 8 mm short
of the grasp (f = 0.93), off the line by U(+/-e) per axis with e = max(3 mm, 15 mm (1 - f)) and turned by
U(+/-5 deg) about a random axis (PD tracking, ZED estimate and encoder-offset errors); full visual DR incl.
camera pose error. Loss weighted toward short distances (the last correction sets the grasp).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/perception/train_reader_servo.py
"""
import argparse, json, math, sys, time
from pathlib import Path

import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickTask, PickTaskCfg, PickExpert
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.perception.nets import PoseNet

SCALE = torch.tensor([.05, .05, .10, .06, .06, .06])


def generate(task, n, gen, size=192, f_min=0., f_max=.93, rot_deg=5.):
    X, Y = [], []
    rt = task.runtime; dev = task.device
    while sum(len(x) for x in X) < n:
        task.reset()
        N = task.cfg.physics.nworld
        ex = PickExpert(task); ex.reset()                        # privileged plan = true approach line
        f = f_min + torch.rand(N, device=dev, generator=gen) * (f_max - f_min)
        p_line = ex.p_pre + f[:, None] * (ex.p_g - ex.p_pre)
        e = torch.clamp(.015 * (1 - f), min=.003)
        dp = (torch.rand(N, 3, device=dev, generator=gen) * 2 - 1) * e[:, None]
        ax = torch.randn(N, 3, device=dev, generator=gen); ax = ax / ax.norm(dim=1, keepdim=True)
        ang = (torch.rand(N, device=dev, generator=gen) * 2 - 1) * math.radians(rot_deg)
        R_t = K.axis_angle_mat(ax, ang) @ ex.R_g
        q, pe, re = task.chain.ik(p_line + dp, R_t, ex.q_pre, iters=80)
        ok = ex.ik_ok & (pe < 1e-3)
        rt.q[:, task.qadr] = q
        with wp.ScopedDevice(task.cfg.physics.device):
            mjw.kinematics(rt.m, rt.d)
        fr = rt.render()['wrist_cam'][ok]
        p_h, R_h = task.chain.fk(q)
        t = (R_h.transpose(1, 2) @ (ex.p_g - p_h)[..., None])[..., 0]
        r = K.rot_log(R_h.transpose(1, 2) @ ex.R_g)
        fr = F.interpolate(fr.permute(0, 3, 1, 2).float(), size=(size, size), mode='area').round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        X.append(fr.contiguous().cpu()); Y.append(torch.cat([t, r], 1)[ok].cpu())
    return torch.cat(X)[:n], torch.cat(Y)[:n]


def errors(pred, y):
    d = (pred - y)
    pos = d[:, :3].norm(dim=1) * 1e3; rot = torch.rad2deg(d[:, 3:].norm(dim=1)) if d.shape[1] > 3 else torch.zeros_like(pos)
    dist = y[:, :3].norm(dim=1)
    out = {}
    for name, m in (('all', torch.ones_like(dist, dtype=torch.bool)), ('near_lt40mm', dist < .04), ('mid_40_80mm', (dist >= .04) & (dist < .08)), ('far_ge80mm', dist >= .08)):
        if m.sum() == 0: continue
        out[name] = {'n': int(m.sum()), 'pos_mm': {q: float(np.percentile(pos[m].cpu().numpy(), q)) for q in (50, 90, 99)},
                     'rot_deg': {q: float(np.percentile(rot[m].cpu().numpy(), q)) for q in (50, 90, 99)}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=49152); ap.add_argument('--n_test', type=int, default=4096)
    ap.add_argument('--res', type=int, default=320); ap.add_argument('--size', type=int, default=192)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--tag', default=''); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pos_only', action='store_true', help='predict the grasp position relative to the hand only (3 outputs)')
    ap.add_argument('--f_min', type=float, default=0.); ap.add_argument('--f_max', type=float, default=.93)
    ap.add_argument('--extra', nargs='*', default=[], help='servo DAgger dumps (pick_expert_eval.py --dump_servo): frames + privileged labels, added to training')
    ap.add_argument('--extra_k_min', type=int, default=0, help='use dump frames with waypoint index >= this (-1: hover frames too)')
    ap.add_argument('--init', default=None, help='initialise from this checkpoint (fine-tune)')
    ap.add_argument('--lr', type=float, default=2e-3)
    a = ap.parse_args()
    t0 = time.time(); dev = torch.device('cuda:0')
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=256, device='cuda:0', render_rgb=True, visual_dr=True, camera_resolution=a.res,
                                                    measured_camera_cadence=False, task_layout='results/usbc/policy/pick/task_layout.json')))
    X, Y = generate(task, a.n, gen, a.size, a.f_min, a.f_max)
    gen.manual_seed(999); torch.manual_seed(999)
    Xt, Yt = generate(task, a.n_test, gen, a.size, a.f_min, a.f_max)
    Y, Yt = Y.to(dev), Yt.to(dev)
    t_gen = time.time() - t0
    print(f'generated {len(X)}+{len(Xt)} in {t_gen:.0f} s', flush=True)
    del task; torch.cuda.empty_cache()
    for f_ in a.extra:
        D = torch.load(f_); m_ = D['k'] >= a.extra_k_min
        X = torch.cat([X, D['X'][m_]]); Y = torch.cat([Y, D['Y'][m_].to(dev)])
        print(f'extra {f_}: {int(m_.sum())} frames', flush=True)
    no = 3 if a.pos_only else 6
    if a.pos_only:
        Y, Yt = Y[:, :3], Yt[:, :3]
    sc = SCALE[:no].to(dev)
    net = PoseNet(no, a.size, width=a.width).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init)['net'])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    bs = 128; steps = a.epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=.1)
    pred_of = lambda XX: torch.cat([net(XX[i:i + 128].to(dev)) for i in range(0, len(XX), 128)]) * sc   # noqa: E731
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(X))
        for i in range(len(X) // bs):
            idx = perm[i * bs:(i + 1) * bs]; y = Y[idx.to(dev)]
            w = (.03 / (y[:, :3].norm(dim=1) + .01)).clamp(max=2.)
            l = F.smooth_l1_loss(net(X[idx].to(dev, non_blocking=True)), y / sc, beta=.02, reduction='none').mean(1)
            loss = (w * l).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                e = errors(pred_of(Xt), Yt)
            print(ep, f'loss {loss.item():.4f}', {k: (round(v['pos_mm'][50], 2), round(v['pos_mm'][99], 2), round(v['rot_deg'][50], 2), round(v['rot_deg'][99], 2)) for k, v in e.items()}, flush=True)
    net.eval()
    with torch.no_grad():
        pt = pred_of(Xt)
    Path('runs/perception').mkdir(parents=True, exist_ok=True)
    torch.save({'net': net.state_dict(), 'size': a.size, 'width': a.width, 'n_out': no, 'scale': sc.cpu(), 'res': a.res,
                'camera': 'wrist_cam', 'output': 'grasp pose relative to the hand: t (m, grasp_site frame), rotation vector (rad)'},
               f'runs/perception/reader_servo{a.tag}.pt')
    rep = {'estimator': 'reader_servo' + a.tag, 'method': __doc__.strip(), 'args': vars(a), 'n_train': len(X), 'n_test': len(Xt),
           'heldout_error': errors(pt, Yt), 'generate_s': t_gen, 'total_s': time.time() - t0}
    out = Path(f'results/usbc/policy/pick/reader_servo{a.tag}.json')
    out.write_text(json.dumps(rep, indent=1) + '\n')
    np.savez_compressed(out.with_suffix('.npz'), pred=pt.cpu().numpy(), target=Yt.cpu().numpy())
    print(json.dumps(rep['heldout_error'], indent=1))


if __name__ == '__main__':
    main()
