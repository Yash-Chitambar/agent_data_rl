"""PPO for Part 4 (right-arm USB-C insertion in the bimanual runtime), Bronars et al. recipe.

TCN actor over the proprioceptive history, contact-prediction head (64 plug-edge points x
{free, contact, inside}), privileged critic, sparse capture/seated reward plus potential shaping,
curriculum on the start-error scale (pre-insert lateral/angle and in-hand error together).

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. .venv/bin/python -m blind_insertion.train_bimanual \
        --name P4-insert-proprio-acc-s0 --nworld 4096 --steps 100e6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.envs.insertion_task import InsertionCfg, InsertionTask
from blind_insertion.policies.tcn_actor_critic import ActorCritic
from blind_insertion.train import PPOCfg, RUNS, _balanced_acc, git_hash, save_atomic

BASE = dict(lateral_m=(0.002, 0.002), angle_deg=2.0, inhand_pos_m=(0.003, 0.0, 0.003),
            inhand_rot_deg=(1.5, 8.0, 1.5), gap_m=(0.002, 0.008))


def set_scale(env: InsertionTask, s: float):
    """Scale the start errors (s = 1: the nominal full-DR distribution)."""
    c = env.icfg
    c.lateral_m = tuple(x * s for x in BASE['lateral_m'])
    c.angle_deg = BASE['angle_deg'] * s
    if c.grasp != 'part1':   # Part 1's measured in-hand spread is not scaled by the curriculum
        c.inhand_pos_m = tuple(x * s for x in BASE['inhand_pos_m'])
        c.inhand_rot_deg = tuple(x * s for x in BASE['inhand_rot_deg'])
    c.gap_m = BASE['gap_m']
    c.npz_scale = s          # used when start_source is Part 2's npz (lateral and angles; gap as sampled)
    env.refresh_pool()


def make_env(args, nworld=None, seed=None):
    cfg = InsertionCfg()
    cfg.task.physics.nworld = nworld or args.nworld
    cfg.task.action_mode = args.action_mode
    over = json.loads(args.env_json) if args.env_json else {}
    for k, v in over.items():
        if hasattr(cfg, k):
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
        elif hasattr(cfg.task, k):
            setattr(cfg.task, k, tuple(v) if isinstance(v, list) else v)
        else:
            setattr(cfg.task.physics, k, tuple(v) if isinstance(v, list) else v)
    if getattr(args, 'camera', ''):
        p = cfg.task.physics
        p.render_rgb, p.visual_dr, p.camera_resolution, p.camera_names = True, True, args.res, ('wrist_cam',)
    return InsertionTask(cfg, seed=args.seed if seed is None else seed)


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    env = make_env(args)
    dev = env.device
    stages = [float(x) for x in args.curriculum.split(',')]
    stage = 0
    set_scale(env, stages[0])
    env.reset_all()
    pc = PPOCfg()
    pc.minibatches = args.minibatches
    pc.ent = args.ent
    ac = ActorCritic(env.obs_dim, env.act_dim, env.critic_dim, history=env.icfg.history,
                     n_points=env.n_labels, init_std=args.init_std).to(dev)
    vis = bool(args.camera)
    if vis:
        from blind_insertion.policies.visual_insertion import VisualAC
        ac = VisualAC(ac, args.res).to(dev)
        rt = env.runtime

        def frames():
            fr, age = rt.camera_sensor.observation(rt.sensor_time)
            return fr['wrist_cam'], age['wrist_cam']
    opt = torch.optim.Adam(ac.parameters(), lr=args.lr or pc.lr)
    it, total, t_stage = 0, 0, 0
    if args.init:
        ck0 = torch.load(args.init, map_location=dev, weights_only=False)
        if vis and 'visual' not in ck0:      # proprio trunk from a blind checkpoint; camera head starts at 0
            ac.base.load_state_dict(ck0['model'])
        else:
            ac.load_state_dict(ck0['model'])
        print(f'[{args.name}] initialized from {args.init}', flush=True)
    resuming = args.resume and (run / 'last.pt').exists()
    if resuming:
        ck = torch.load(run / 'last.pt', map_location=dev, weights_only=False)
        ac.load_state_dict(ck['model']); opt.load_state_dict(ck['opt'])
        it, total, stage, t_stage = ck['iter'], ck['steps'], ck['stage'], ck.get('t_stage', ck['steps'])
        set_scale(env, stages[stage]); env.reset_all()
        print(f'[{args.name}] resumed at it {it}, {total/1e6:.1f}M steps, stage {stage}', flush=True)
    meta = dict(args=vars(args), git=git_hash(), obs_dim=env.obs_dim, critic_dim=env.critic_dim,
                grasp=env.grasp_info, contact_dr=env.runtime.cfg.contact_dr, opt=env.runtime.cfg.opt_overrides)
    json.dump(meta, open(run / ('config_resume.json' if resuming else 'config.json'), 'w'), indent=1, default=str)
    N, T, H, D, A, C = env.N, pc.rollout, env.icfg.history, env.obs_dim, env.act_dim, env.critic_dim
    buf = dict(hist=torch.zeros(T, N, H, D, device=dev), crit=torch.zeros(T, N, C, device=dev),
               act=torch.zeros(T, N, A, device=dev), logp=torch.zeros(T, N, device=dev),
               val=torch.zeros(T, N, device=dev), rew=torch.zeros(T, N, device=dev),
               done=torch.zeros(T, N, device=dev), tout=torch.zeros(T, N, device=dev),
               valid=torch.zeros(T, N, device=dev),
               lab=torch.zeros(T, N, env.n_labels, dtype=torch.long, device=dev))
    if vis:
        buf['frames'] = torch.zeros(T, N, args.res, args.res, 3, dtype=torch.uint8, device=dev)
        buf['age'] = torch.zeros(T, N, device=dev)
    hist, crit = env.observe()
    cw = torch.tensor(pc.class_w, device=dev)
    log_f = open(run / 'train_log.csv', 'a' if resuming else 'w', newline='')
    log = None
    t0 = time.time() - (ck.get('wall', 0.) if resuming else 0.)
    recent = []      # (steps, successes, episodes)
    reason = 'steps'

    def ckpt():
        return {'model': ac.state_dict(), 'opt': opt.state_dict(), 'iter': it, 'steps': total, 'stage': stage,
                't_stage': t_stage, 'scale': stages[stage], 'obs_dim': D, 'act_dim': A, 'critic_dim': C,
                'n_points': env.n_labels, 'history': H, 'action_mode': args.action_mode,
                'env_json': args.env_json, 'wall': time.time() - t0,
                **({'visual': {'res': args.res, 'cams': ['wrist_cam']}} if vis else {})}

    while total < args.steps:
        it += 1
        tr = time.time()
        ac.eval()
        outc = {}
        with torch.no_grad():
            for t in range(T):
                ac.obs_norm.update(hist[:, -1])
                ac.crit_norm.update(crit)
                if vis:
                    fr, age = frames()
                    buf['frames'][t], buf['age'][t] = fr, age
                    d, _ = ac.dist(hist, fr, age)
                else:
                    d, _ = ac.dist(hist)
                a = d.sample()
                buf['hist'][t] = hist
                buf['crit'][t] = crit
                buf['act'][t] = a
                buf['logp'][t] = d.log_prob(a).sum(-1)
                buf['val'][t] = ac.value(hist, crit)
                buf['lab'][t] = env.current_labels()
                hist, crit, r, done, info = env.step(a)
                buf['rew'][t] = r
                buf['done'][t] = done.float()
                buf['tout'][t] = info['timeout'].float()
                buf['valid'][t] = info['valid'].float()
                if 'final' in info:
                    f = info['final']
                    for k in ('success', 'slip', 'pushed', 'far', 'bad'):
                        outc[k] = outc.get(k, 0) + int(f[k].sum())
                    outc['n'] = outc.get('n', 0) + len(f['success'])
                    outc['steps'] = outc.get('steps', 0) + int(info['steps'][info['final_ids']][f['success']].sum())
            last_val = ac.value(hist, crit)
        total += T * N
        env.refresh_pool()
        roll = time.time() - tr
        with torch.no_grad():
            rew = buf['rew'] + pc.gamma * buf['val'] * buf['tout']
            adv = torch.zeros_like(rew)
            last = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nv = last_val if t == T - 1 else buf['val'][t + 1]
                nd = 1.0 - buf['done'][t]
                delta = rew[t] + pc.gamma * nv * nd - buf['val'][t]
                last = delta + pc.gamma * pc.lam * nd * last
                adv[t] = last
            ret = adv + buf['val']
        tu = time.time()
        ac.train()
        nb = T * N
        mb = nb // pc.minibatches
        flat = {k: v.reshape(nb, *v.shape[2:]) for k, v in buf.items()}
        fadv, fret = adv.reshape(nb), ret.reshape(nb)
        st = {'pg': 0., 'vf': 0., 'ent': 0., 'ce': 0., 'acc': 0., 'kl': 0.}
        nupd = 0
        for _ep in range(pc.epochs):
            if args.target_kl and nupd and st['kl'] / nupd > args.target_kl:
                break      # KL early stop (fine-tuning a low-std policy)
            perm = torch.randperm(nb, device=dev)
            for i in range(pc.minibatches):
                j = perm[i * mb:(i + 1) * mb]
                h = flat['hist'][j]
                dist, z = ac.dist(h, flat['frames'][j], flat['age'][j]) if vis else ac.dist(h)
                logp = dist.log_prob(flat['act'][j]).sum(-1)
                ratio = torch.exp(logp - flat['logp'][j])
                w = flat['valid'][j]
                a_ = fadv[j]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pgl = -torch.min(ratio * a_, ratio.clamp(1 - pc.clip, 1 + pc.clip) * a_)
                pg = (pgl * w).sum() / w.sum().clamp(min=1)
                v = ac.value(h, flat['crit'][j])
                v_old = flat['val'][j]
                v_clip = v_old + (v - v_old).clamp(-pc.clip, pc.clip)
                vf = torch.max((v - fret[j]) ** 2, (v_clip - fret[j]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + pc.vf * vf - pc.ent * ent
                logits = ac.contact_logits(z)
                lab = flat['lab'][j]
                ce = F.cross_entropy(logits.reshape(-1, 3), lab.reshape(-1), weight=cw)
                loss = loss + args.contact_coef * ce
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), pc.max_grad)
                opt.step()
                with torch.no_grad():
                    st['kl'] += ((flat['logp'][j] - logp) * w).sum().item() / max(w.sum().item(), 1)
                    st['ce'] += ce.item(); st['acc'] += _balanced_acc(logits.argmax(-1), lab)
                st['pg'] += pg.item(); st['vf'] += vf.item(); st['ent'] += ent.item()
                nupd += 1
        st = {k: v / nupd for k, v in st.items()}
        upd = time.time() - tu
        n = outc.get('n', 0)
        succ = outc.get('success', 0) / n if n else float('nan')
        recent.append((total, outc.get('success', 0), n))
        # curriculum: advance when success >= thr over the last `window` env steps of this stage
        win = [x for x in recent if x[0] > max(total - args.window, t_stage)]
        ws, wn = sum(x[1] for x in win), sum(x[2] for x in win)
        if stage < len(stages) - 1 and total - t_stage >= args.window and wn and ws / wn >= args.thr:
            stage += 1
            t_stage = total
            set_scale(env, stages[stage]); env.reset_all()
            hist, crit = env.observe()
            print(f'[{args.name}] curriculum -> stage {stage} scale {stages[stage]} at {total/1e6:.1f}M', flush=True)
        row = dict(iter=it, env_steps=total, wall=time.time() - t0, stage=stage, scale=stages[stage], success=succ,
                   episodes=n, slip=outc.get('slip', 0) / max(n, 1), pushed=outc.get('pushed', 0) / max(n, 1),
                   far=outc.get('far', 0) / max(n, 1), nonfinite=outc.get('bad', 0),
                   t_seat_s=outc.get('steps', 0) * env.dt / max(outc.get('success', 0), 1),
                   reward=buf['rew'].sum().item() / N, std=ac.log_std.exp().mean().item(),
                   sps=T * N / (roll + upd), roll_s=roll, update_s=upd, **st)
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            if not resuming:
                log.writeheader()
        log.writerow(row)
        log_f.flush()
        if it % args.print_every == 0 or it == 1:
            print(f"[{args.name}] it {it} {total/1e6:.1f}M st {stage} succ {succ:.3f} (n {n}) slip {row['slip']:.3f} "
                  f"push {row['pushed']:.3f} far {row['far']:.3f} nan {row['nonfinite']} tseat {row['t_seat_s']:.2f} "
                  f"std {row['std']:.3f} vf {st['vf']:.4f} acc {st['acc']:.3f} kl {st['kl']:.4f} sps {row['sps']:.0f}",
                  flush=True)
        if it % args.ckpt_every == 0:
            save_atomic(ckpt(), run / 'last.pt')
        if it % (args.ckpt_every * 4) == 0:
            save_atomic(ckpt(), run / f'ckpt_{total // 1_000_000:04d}M.pt')
    save_atomic(ckpt(), run / 'last.pt')
    log_f.close()
    json.dump(dict(steps=total, iter=it, wall_s=time.time() - t0, reason=reason, stage=stage), open(run / 'DONE', 'w'))


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--nworld', type=int, default=4096)
    ap.add_argument('--steps', type=float, default=100e6)
    ap.add_argument('--action_mode', default='accumulate', choices=['accumulate', 'relative'])
    ap.add_argument('--curriculum', default='0.25,0.5,0.75,1.0,1.25',
                    help='start-error scales; advance when windowed success >= --thr')
    ap.add_argument('--thr', type=float, default=0.9)
    ap.add_argument('--window', type=float, default=4e6)
    ap.add_argument('--env_json', default='')
    ap.add_argument('--minibatches', type=int, default=16)
    ap.add_argument('--ent', type=float, default=0.0)
    ap.add_argument('--init_std', type=float, default=0.5)
    ap.add_argument('--contact_coef', type=float, default=1.0)
    ap.add_argument('--print_every', type=int, default=5)
    ap.add_argument('--ckpt_every', type=int, default=10)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--camera', default='', help="'wrist': proprio + right-wrist RGB actor (VisualAC)")
    ap.add_argument('--res', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.0, help='0: PPOCfg.lr (3e-4)')
    ap.add_argument('--target_kl', type=float, default=0.0, help='>0: stop the update epochs once mean KL exceeds this')
    ap.add_argument('--init', default='', help='start from this checkpoint (weights + normalizers), fresh optimizer')
    return ap


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    train(parser().parse_args())
