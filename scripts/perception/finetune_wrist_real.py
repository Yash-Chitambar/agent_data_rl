"""Fine-tune a wrist net on real frames, with sim renders in every batch. No motion.

    PYTHONPATH=. python scripts/perception/finetune_wrist_real.py --init runs/perception_real/wrist_r10f.pt --name wrist_r10f_ft \
        --lateral .014 --gap .002 .022 --rot 5

Training frames (scripts/real/wrist_net_real_eval.py fixes the split; its hold-out and A, A2, ai4 are never loaded here):
  collect training blocks: across and gap against the label; up as differences between frames of the batch (the run's constant
    height offset: docs/real/SESSION_20261102_duchamp3.md), or absolute with --label-offset (port vision's estimate, applied to all three axes);
    rotation as differences, low weight.
  ai1, ai2, ai3, ai5 (`unlabeled`, settled frames): FK differences within a run only, all axes. They bring other grasps and another day.
  sim renders (same generator as train_approach_estimators, measured in-hand preset v1, real reader and PC look): all axes absolute. They fix the absolute
    height, which the real loss leaves free.
Selection: --val-blocks are left out of training and scored for the choice of steps and weights. The hold-out is for the final report only.
Output: runs/perception_real/<name>.pt, runs/perception_real/<name>.json (training record).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import wrist_net_real_eval as W  # noqa: E402

ROT_W = .2
MAX_QD_REL = .05


def rel_labels(rows):
    """FK motion in port axes within a run (tip by the training grasp; the pivot's error is small for a few degrees): (n, 6), run id."""
    from blind_insertion.envs import kinematics as K
    from blind_insertion.envs.inhand import measured_mean
    from deploy.bimanual.scene import SceneModel
    fk = SceneModel('cpu').chain['right']
    Rs, ps = measured_mean(device='cpu')
    tip_s = Rs.reshape(3, 3).float().T @ ps.reshape(3).float()
    Y, run = [], []
    runs = sorted({r['run'] for r in rows})
    for k, name in enumerate(runs):
        rr = [r for r in rows if r['run'] == name]
        ev = [json.loads(x) for x in open(ROOT / f"data/real_runs/{rr[0]['day']}/{name}/events.jsonl")]
        yaw = [e for e in ev if e['kind'] == 'estimate' and e['est'] == 'zed_pc'][-1]['yaw_deg']
        y = math.radians(yaw)
        z = np.array([-math.cos(y), -math.sin(y), 0.])
        eR = torch.tensor(np.c_[np.cross([0., 0., 1.], z), [0., 0., 1.], z], dtype=torch.float32)
        p, R = fk.fk(torch.tensor([r['q_b'] for r in rr], dtype=torch.float32))
        tip = p - (R @ tip_s)
        lp = (tip - tip[0]) @ eR * 1e3
        rv = torch.rad2deg(K.rot_log(eR.T @ R @ R[0].T @ eR))
        Y.append(torch.cat([lp, rv], 1))
        run += [k] * len(rr)
    return torch.cat(Y).numpy(), np.array(run)


def augment(x, shift=3):
    """(B, H, W, 3) uint8, net input size: photometric, and a whole-image shift (mount error; the label does not change)."""
    b, h, wd = x.shape[0], x.shape[1], x.shape[2]
    dev = x.device
    im = x.permute(0, 3, 1, 2).float() / 255.
    u = lambda lo, hi, *s: torch.rand(b, *s, device=dev) * (hi - lo) + lo   # noqa: E731
    gain = u(.8, 1.2, 3, 1, 1) * u(.75, 1.25, 1, 1, 1)
    im = im.clamp(1e-3, 1) ** u(.75, 1.3, 1, 1, 1) * gain + u(-.05, .05, 1, 1, 1)
    sg = u(0., 1.3)
    if float(sg.max()) > .05:
        k = 7
        t = torch.arange(k, device=dev).float() - k // 2
        g = torch.exp(-t ** 2 / (2 * sg[:, None] ** 2 + 1e-3))
        g = g / g.sum(1, keepdim=True)
        pad = F.pad(im, (k // 2,) * 4, mode='replicate')
        c = pad.reshape(b * 3, 1, *pad.shape[2:])
        gg = g.repeat_interleave(3, 0)
        c = F.conv2d(c.reshape(1, b * 3, *pad.shape[2:]), gg[:, None, None, :], groups=b * 3)
        c = F.conv2d(c, gg[:, None, :, None], groups=b * 3)
        im = c.reshape(b, 3, h, wd)
    im = im + torch.randn_like(im) * u(0., .02, 1, 1, 1)
    dx = torch.randint(-shift, shift + 1, (b,), device=dev)
    dy = torch.randint(-shift, shift + 1, (b,), device=dev)
    yy, xx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(wd, device=dev), indexing='ij')
    ix = (xx[None] + dx[:, None, None]).clamp(0, wd - 1)
    iy = (yy[None] + dy[:, None, None]).clamp(0, h - 1)
    im = im[torch.arange(b, device=dev)[:, None, None], :, iy, ix].permute(0, 3, 1, 2)
    return (im.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1)


def render_sim(a, n):
    from blind_insertion.perception.render_data import ApproachRenderer, SampleCfg
    import scripts.perception.train_approach_estimators as T
    r = ApproachRenderer(nworld=512, seed=a.seed + 77, task_layout='results/usbc/policy/pick/task_layout.json', visual_dr=dict(blur_sigma_px=(0., 2.5)),
                         inhand_real='v1', pc_real_look=a.look)
    r.ins.cfg.inhand = 'real'
    r.ins.cfg.inhand_real_spread = a.spread
    T.MODE.update(wrist='resize', two_view=False, cpu=True)
    X, Y = T.generate(r, 'wrist', n, SampleCfg(lateral_m=a.lateral, gap_m=tuple(a.gap), rot_deg=a.rot))
    return X.numpy(), Y.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--init', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--lateral', type=float, default=.014)
    ap.add_argument('--gap', type=float, nargs=2, default=(.002, .022))
    ap.add_argument('--rot', type=float, default=5.)
    ap.add_argument('--look', default='blind_insertion/assets/pc_real_look')
    ap.add_argument('--spread', type=float, default=2., help='scale on the in-hand preset half ranges in the renders')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--n-sim', type=int, default=40000)
    ap.add_argument('--bs', type=int, nargs=3, default=(48, 32, 32), help='sim, collect, relative frames per batch')
    ap.add_argument('--w-real', type=float, default=1.)
    ap.add_argument('--label-offset', type=float, nargs=3, default=None, help='true = manifest + offset for the collect run (mm). With it, across, up and gap are absolute; without, up is relative')
    ap.add_argument('--gap-max', type=float, default=1e9, help='collect frames with a label gap above this are not used (mm)')
    ap.add_argument('--no-rel', action='store_true', help='without the ai1-ai5 differences')
    ap.add_argument('--val-blocks', type=int, nargs='*', default=[], help='blocks of the sweep left out for selection (not hold-out blocks)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sim-cache', default=None)
    ap.add_argument('--final', action='store_true', help='write the weights under the given name (else a selection run: record only)')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device('cuda:0')
    t0 = time.time()
    rs = W.rows()
    co = W.collect_rows(rs)
    tr, ho, _ = W.split_collect(len(co))
    assert not set(a.val_blocks) & set(W.HOLDOUT_BLOCKS)
    blk = tr // W.BLOCK
    val = tr[np.isin(blk, a.val_blocks)]
    trn = tr[~np.isin(blk, a.val_blocks)]
    gaps = np.array([r['p_mm'][2] for r in co])
    trn = trn[gaps[trn] <= a.gap_max]
    val = val[gaps[val] <= a.gap_max]
    Yc = np.array([r['p_mm'] + r['rv_deg'] for r in co], dtype=np.float32)
    if a.label_offset:
        Yc[:, :3] += np.array(a.label_offset, dtype=np.float32)
    Xc = np.stack([W.read_rgb(co[i]['image']) for i in range(len(co))])
    net, sc, ck = W.load_net(a.init, dev)
    Xc_in = W.net_input(torch.as_tensor(Xc, device=dev), ck).cpu()
    rl = [r for r in rs if r['source'] == 'unlabeled' and r['qd_max'] <= MAX_QD_REL]
    Yr, rid = rel_labels(rl)
    Xr = np.stack([W.read_rgb(r['image']) for r in rl])
    Xr_in = torch.cat([W.net_input(torch.as_tensor(Xr[i:i + 128], device=dev), ck).cpu() for i in range(0, len(Xr), 128)])
    print(f'collect train {len(trn)}, val {len(val)} (hold-out {len(ho)} not used); relative frames {len(rl)} from {rid.max() + 1} runs', flush=True)
    cache = Path(a.sim_cache) if a.sim_cache else None
    if cache and cache.exists():
        d = np.load(cache)
        Xs, Ys = d['X'], d['Y']
    else:
        Xs, Ys = render_sim(a, a.n_sim)
        if cache:
            np.savez(cache, X=Xs, Y=Ys)
    n_te = 2048
    Xs_t, Ys_t = torch.as_tensor(Xs[-n_te:]), torch.as_tensor(Ys[-n_te:])
    Xs, Ys = torch.as_tensor(Xs[:-n_te]), torch.as_tensor(Ys[:-n_te])
    print(f'sim {len(Xs)} + {len(Xs_t)} test, {time.time() - t0:.0f} s', flush=True)
    opt = torch.optim.AdamW(net.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=.1)
    Yc_t, Yr_t = torch.as_tensor(Yc), torch.as_tensor(Yr, dtype=torch.float32)
    rid_t = torch.as_tensor(rid)
    bs_s, bs_c, bs_r = a.bs
    trn_t = torch.as_tensor(trn)
    u_runs = rid_t.unique()
    log = []

    def eval_set(idx):
        net.eval()
        with torch.no_grad():
            P = torch.cat([net(Xc_in[idx[i:i + 64]].to(dev)) for i in range(0, len(idx), 64)]) * sc
        net.train()
        return P.cpu().numpy()

    def sim_err():
        net.eval()
        with torch.no_grad():
            P = torch.cat([net(Xs_t[i:i + 256].to(dev)) for i in range(0, n_te, 256)]) * sc
        net.train()
        return (P.cpu() - Ys_t).abs().median(0).values.numpy()
    net.train()
    for k in range(a.steps):
        i_s = torch.randint(len(Xs), (bs_s,))
        i_c = trn_t[torch.randint(len(trn_t), (bs_c,))]
        parts = [Xs[i_s], Xc_in[i_c]]
        if not a.no_rel:
            # relative frames: two runs per batch, so each has enough frames for its differences
            rr = u_runs[torch.randperm(len(u_runs))[:2]]
            cand = torch.where((rid_t == rr[0]) | (rid_t == rr[1]))[0]
            i_r = cand[torch.randint(len(cand), (bs_r,))]
            parts.append(Xr_in[i_r])
        x = augment(torch.cat(parts).to(dev))
        o = net(x) * sc
        ys = Ys[i_s].to(dev)
        lsim = F.smooth_l1_loss(o[:bs_s] / sc, ys / sc, beta=.05)
        oc = o[bs_s:bs_s + bs_c]
        yc = Yc_t[i_c].to(dev)
        e = (oc - yc) / sc
        lc = F.smooth_l1_loss(e[:, [0, 2]], torch.zeros_like(e[:, [0, 2]]), beta=.05)
        ey = e[:, [1, 3, 4, 5]].clone()
        if a.label_offset:                    # up absolute (offset from port vision); rotation still as differences
            ey[:, 1:] = ey[:, 1:] - ey[:, 1:].mean(0, keepdim=True)
        else:                                 # up, rotation: differences within the batch
            ey = ey - ey.mean(0, keepdim=True)
        lc = lc + F.smooth_l1_loss(ey[:, 0], torch.zeros_like(ey[:, 0]), beta=.05) + ROT_W * ey[:, 1:].abs().mean()
        lr_ = torch.zeros((), device=dev)
        if not a.no_rel:
            or_ = o[bs_s + bs_c:]
            er = (or_ - Yr_t[i_r].to(dev)) / sc
            for g in rid_t[i_r].unique():
                m = rid_t[i_r].to(dev) == g
                if m.sum() >= 3:
                    d = er[m] - er[m].mean(0, keepdim=True)
                    lr_ = lr_ + F.smooth_l1_loss(d[:, :3], torch.zeros_like(d[:, :3]), beta=.05) + ROT_W * d[:, 3:].abs().mean()
        loss = lsim + a.w_real * (lc + .5 * lr_)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        sched.step()
        if k % 500 == 499 or k == a.steps - 1:
            rec = dict(step=k + 1, loss_sim=round(float(lsim), 4), loss_real=round(float(lc), 4), loss_rel=round(float(lr_), 4), sim_err_p50=sim_err().round(3).tolist())
            if len(val):
                P = eval_set(torch.as_tensor(val))
                Yv = Yc[val]
                rec['val'] = {ax: dict(slope=round(W.slope(Yv[:, i], P[:, i]), 3), resid_sd=round(float((P[:, i] - Yv[:, i]).std()), 2)) for i, ax in enumerate(W.AXES)}
                rec['val_n'] = len(val)
            log.append(rec)
            print(json.dumps(rec), flush=True)
    out = ROOT / 'runs/perception_real'
    out.mkdir(parents=True, exist_ok=True)
    rec = dict(name=a.name, args=vars(a), init=a.init, n_collect_train=len(trn), n_val=len(val), n_relative=len(rl), n_sim=len(Xs), log=log, seconds=time.time() - t0,
               hold_out_not_used=[int(i) for i in ho], split=dict(blocks=list(W.HOLDOUT_BLOCKS), block=W.BLOCK, guard=W.GUARD))
    (out / f'{a.name}.json').write_text(json.dumps(rec, indent=1) + '\n')
    if a.final:
        dst = out / f'{a.name}.pt'
        assert not dst.exists(), f'{dst} exists: new weights take new names'
        net.eval()
        ck2 = dict(ck)
        ck2.update(net=net.state_dict(), parent=str(a.init), real_finetune=rec['args'])
        torch.save(ck2, dst)
        print('saved', dst)


if __name__ == '__main__':
    main()
