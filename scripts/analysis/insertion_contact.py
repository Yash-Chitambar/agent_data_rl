"""Does the Part 4 insertion policy push too hard / too fast, and does it sense the USB-C edges?

Rolls out a policy (deterministic) for the first episode of every world and logs per control step
(state after the 20 substeps, i.e. the last substep's contact forces; elliptic cones: normal force =
efc.force at the contact's first row):
  plug-port normal force (reader geoms vs port/fixture geoms), total reader->PC force (all reader-PC
  contacts), right-pad normal force on the reader, tip pose in the port frame, plug velocity,
  contact labels of the 64 plug-edge points, the actor's TCN latent z and last hidden layer, and the
  contact head's predictions.
Then: per-episode force/speed/contact statistics, contact-head accuracy/F1 (held-out), linear and
MLP probes on the actor state, and the lateral-correction behaviour during rim contact.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/analysis/insertion_contact.py \
        --policy runs/P4-insert-best/best.pt --out results/usbc/policy/insertion/contact_v11
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import warp as wp

from blind_insertion.eval_insertion import load_insertion_policy, wilson
from blind_insertion.envs.insertion_task import FREE, CONTACT, INSIDE, failure_class
from blind_insertion.train_bimanual import make_env, set_scale

R3 = 'results/usbc/policy/approach/preinsert_error_samples.npz'
LV = 'results/usbc/policy/insertion/aux/handover_r5_hold020_n4096.npz'


def rollout(pol, cond, nworld, seed):
    ej = dict(pol.meta['env_json'])
    ej.update(start_source=R3, episode_s=10.0)
    if cond == 'live':
        ej.update(live_states=LV, live_frac=1.0, live_split='eval')
    args = argparse.Namespace(nworld=nworld, seed=seed, action_mode='accumulate', env_json=json.dumps(ej))
    torch.manual_seed(seed)
    env = make_env(args)
    set_scale(env, 1.5 if cond == 'r3x1.5' else 1.0)
    env.reset_all()
    rt, dev, N = env.runtime, env.device, env.N
    m = rt.mjm
    reader = torch.zeros(m.ngeom, dtype=torch.bool, device=dev); reader[list(rt.lay.peg_slots)] = True
    port = torch.zeros_like(reader); port[list(rt.lay.fixture_slots)] = True
    pcb = m.body('fixture').id
    pc = torch.tensor([m.geom_bodyid[g] == pcb for g in range(m.ngeom)], device=dev)
    pads = torch.zeros_like(reader); pads[[g for g in range(m.ngeom) if m.geom(g).name.startswith(('left_pad_s', 'right_pad_s'))]] = True
    ac = pol.members[0][0]
    da = int(m.jnt_dofadr[m.body('peg').jntadr[0]])
    T = env.max_steps
    rec = {k: torch.zeros(T, N, d, device=dev) for k, d in
           (('f_port', 1), ('f_pc', 1), ('f_pad', 1), ('F_port', 3), ('F_pc', 3), ('p', 3), ('v', 3), ('z', ac.latent), ('h', 128), ('a', 6))}
    rec['lab'] = torch.zeros(T, N, env.n_labels, dtype=torch.long, device=dev)
    rec['pred'] = torch.zeros(T, N, env.n_labels, dtype=torch.long, device=dev)
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    length = torch.full((N,), T, dtype=torch.long, device=dev)
    outcome = [None] * N
    starts = {k: v.clone() for k, v in env.start.items()}
    hist, _ = env.observe()
    for t in range(T):
        with torch.no_grad():
            h = hist[..., :pol.obs_dim].clone()
            if pol.mask_idx:
                h[..., list(pol.mask_idx)] = 0
            z, o = ac.encode(h)
            hid = ac.actor[:-1](torch.cat([z, o], 1))
            act = ac.actor[-1](hid).clamp(-1, 1)
            pred = ac.contact_logits(z).argmax(-1)
        rec['z'][t], rec['h'][t], rec['a'][t], rec['pred'][t] = z, hid, act, pred
        rec['lab'][t] = env.current_labels()
        hist, _, _, done, info = env.step(act)
        # contact forces after the step
        d = rt.d
        nc = int(d.nacon.numpy()[0])
        if nc:
            g = wp.to_torch(d.contact.geom)[:nc].long()
            w = wp.to_torch(d.contact.worldid)[:nc].long()
            adr = wp.to_torch(d.contact.efc_address)[:nc, 0].long()
            ok = adr >= 0
            ef = wp.to_torch(d.efc.force)
            fn = torch.zeros(nc, device=dev)
            fn[ok] = ef[w[ok], adr[ok]].abs()
            # full 3D contact force (elliptic cone: rows adr, adr+1, adr+2 = normal, tangent 1, tangent 2)
            dim = wp.to_torch(d.contact.dim)[:nc].long()
            fr = wp.to_torch(d.contact.frame)[:nc].reshape(nc, 3, 3)
            fl = torch.zeros(nc, 3, device=dev)
            for r_ in range(3):
                okr = ok & (dim > r_)
                fl[okr, r_] = ef[w[okr], adr[okr] + r_]
            Fw = (fr.transpose(1, 2) @ fl[..., None])[..., 0]          # world force on geom[1] from geom[0]
            a_, b_ = g[:, 0], g[:, 1]
            on_reader = torch.where(reader[b_], 1., -1.)[:, None] * Fw  # force on the reader
            _, port_R0 = env._port_pose(rt.q[:, rt.pc_qadr:rt.pc_qadr + 7])
            for k, other in (('F_port', port), ('F_pc', pc)):
                msk = (reader[a_] & other[b_]) | (reader[b_] & other[a_])
                acc3 = torch.zeros(N, 3, device=dev)
                acc3.index_add_(0, w[msk], on_reader[msk])
                rec[k][t] = (port_R0.transpose(1, 2) @ acc3[..., None])[..., 0]   # port frame: +z = pushing the plug out
            sel = {'f_port': (reader[a_] & port[b_]) | (reader[b_] & port[a_]),
                   'f_pc': (reader[a_] & pc[b_]) | (reader[b_] & pc[a_]),
                   'f_pad': (reader[a_] & pads[b_]) | (reader[b_] & pads[a_])}
            for k, msk in sel.items():
                acc = torch.zeros(N, device=dev)
                acc.index_add_(0, w[msk], fn[msk])
                rec[k][t, :, 0] = acc
        pp, _ = rt.plug_in_port()
        _, port_R = env._port_pose(rt.q[:, rt.pc_qadr:rt.pc_qadr + 7])
        rec['p'][t] = pp
        rec['v'][t] = (port_R.transpose(1, 2) @ rt.v[:, da:da + 3, None])[..., 0]
        if 'final_ids' in info:
            ids = info['final_ids']
            cls = failure_class(info['final'])
            for j, w_ in enumerate(ids.tolist()):
                if alive[w_]:
                    outcome[w_] = cls[j]
                    length[w_] = t + 1
            alive[ids] = False
        if not alive.any():
            T = t + 1
            break
    rec = {k: v[:T] for k, v in rec.items()}
    return env, rec, length, outcome, starts


def episode_stats(rec, length, outcome, dt):
    N = len(outcome)
    T = rec['f_port'].shape[0]
    tt = torch.arange(T, device=rec['f_port'].device)[:, None]
    valid = tt < length[None]
    fp = rec['f_port'][..., 0] * valid
    fpc = rec['f_pc'][..., 0] * valid
    fpad = rec['f_pad'][..., 0] * valid
    contact = fp > 0.05
    first = torch.where(contact.any(0), contact.float().argmax(0), torch.full((N,), -1, device=fp.device))
    speed = rec['v'].norm(dim=-1) * 1e3        # mm/s
    sp_first = torch.where(first >= 0, speed[first.clamp(min=0), torch.arange(N, device=fp.device)], torch.full((N,), float('nan'), device=fp.device))
    after = (tt >= first[None]) & valid & (first[None] >= 0)
    sp_peak = torch.where(after, speed, torch.zeros_like(speed)).max(0).values
    # rim contact: plug-port contact while not yet inside (tip z > -0.5 mm)
    rim = contact & (rec['p'][..., 2] > -0.0005)
    onsets = (rim[1:] & ~rim[:-1]).sum(0) + rim[0].long()
    nsteps_c = contact.sum(0).clamp(min=1)
    Fp = rec['F_port'] * valid[..., None]
    Fc = rec['F_pc'] * valid[..., None]
    out_net = dict(peak_axial_insertion_force_N=Fp[..., 2].clamp(min=0).max(0).values,
                   mean_axial_force_in_contact_N=Fp[..., 2].clamp(min=0).sum(0) / nsteps_c,
                   peak_lateral_port_force_N=Fp[..., :2].norm(dim=-1).max(0).values,
                   peak_net_port_force_N=Fp.norm(dim=-1).max(0).values,
                   peak_net_reader_to_pc_force_N=Fc.norm(dim=-1).max(0).values)
    out = dict(**out_net, peak_wall_squeeze_sum_N=fp.max(0).values, mean_port_N_in_contact=fp.sum(0) / nsteps_c, peak_pc_N=fpc.max(0).values,
               peak_pad_N=fpad.max(0).values, speed_first_contact_mm_s=sp_first, peak_speed_after_contact_mm_s=sp_peak,
               rim_contact_steps=rim.sum(0).float(), rim_contact_onsets=onsets.float())
    return {k: v.cpu().numpy() for k, v in out.items()}


def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {'p50': float(np.percentile(x, 50)), 'p99': float(np.percentile(x, 99)), 'max': float(x.max()), 'mean': float(x.mean()), 'n': int(len(x))} if len(x) else {}


def head_metrics(rec, length):
    T, N = rec['lab'].shape[:2]
    valid = (torch.arange(T, device=rec['lab'].device)[:, None] < length[None])
    lab, pred = rec['lab'][valid].flatten(), rec['pred'][valid].flatten()
    out = {}
    names = {FREE: 'free', CONTACT: 'rim_contact', INSIDE: 'inside'}
    for c, nm in names.items():
        tp = ((pred == c) & (lab == c)).sum().item(); fp_ = ((pred == c) & (lab != c)).sum().item(); fn = ((pred != c) & (lab == c)).sum().item()
        prec = tp / max(tp + fp_, 1); recall = tp / max(tp + fn, 1)
        out[nm] = {'precision': prec, 'recall': recall, 'f1': 2 * prec * recall / max(prec + recall, 1e-9), 'support': int((lab == c).sum().item())}
    out['accuracy'] = float((lab == pred).float().mean())
    out['balanced_accuracy'] = float(np.mean([out[n]['recall'] for n in names.values()]))
    return out


def probes(rec, length, env, seed=0):
    """Linear and 2-layer MLP probes on the actor state (TCN latent z ++ last hidden layer), split by world."""
    T, N = rec['f_port'].shape[:2]
    valid = (torch.arange(T, device=rec['p'].device)[:, None] < length[None])
    X = torch.cat([rec['z'], rec['h']], -1)
    contact = (rec['f_port'][..., 0] > 0.05).long()
    # touching side: which side of the plug face carries rim-contact labels (edge points: x long side, y short)
    xy = torch.as_tensor(np.asarray(env.runtime.lay.edge_points[0]), device=X.device, dtype=torch.float32)
    rimlab = (rec['lab'] == CONTACT).float()                         # (T, N, 64)
    side_dirs = torch.stack([xy[:, 0] > xy[:, 0].abs().max() * .5, xy[:, 0] < -xy[:, 0].abs().max() * .5,
                             xy[:, 1] > xy[:, 1].abs().max() * .5, xy[:, 1] < -xy[:, 1].abs().max() * .5], 1).float()   # (64, 4)
    side_cnt = rimlab @ side_dirs                                    # (T, N, 4)
    side = torch.where(side_cnt.sum(-1) > 0, side_cnt.argmax(-1), torch.full_like(side_cnt[..., 0], 4, dtype=torch.long))
    lat = rec['p'][..., :2] * 1e3
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    tr, te = perm[: int(.75 * N)], perm[int(.75 * N):]
    def flat(idx, y):
        m = valid[:, idx]
        return X[:, idx][m], y[:, idx][m]
    res = {}
    for name, y, kind in (('contact_yes_no', contact, 'cls'), ('touching_side_(+x,-x,+y,-y,none)', side, 'cls'), ('lateral_offset_mm', lat, 'reg')):
        Xtr, ytr = flat(tr, y); Xte, yte = flat(te, y)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        nout = int(y.max().item()) + 1 if kind == 'cls' else ytr.shape[-1]
        for arch in ('linear', 'mlp'):
            torch.manual_seed(seed)
            net = (torch.nn.Linear(X.shape[-1], nout) if arch == 'linear' else
                   torch.nn.Sequential(torch.nn.Linear(X.shape[-1], 128), torch.nn.ELU(), torch.nn.Linear(128, nout))).to(X.device)
            opt = torch.optim.Adam(net.parameters(), lr=3e-3)
            if kind == 'cls':
                w = torch.bincount(ytr, minlength=nout).float().clamp(min=1); w = (w.sum() / w) / nout
            for it in range(1500):
                j = torch.randint(len(Xtr), (4096,), device=X.device)
                out = net(Xtr[j])
                loss = torch.nn.functional.cross_entropy(out, ytr[j], weight=w) if kind == 'cls' else ((out - ytr[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                out = net(Xte)
                if kind == 'cls':
                    pr = out.argmax(-1)
                    recs = [float((pr[yte == c] == c).float().mean()) for c in range(nout) if (yte == c).any()]
                    res[f'{name}/{arch}'] = {'accuracy': float((pr == yte).float().mean()), 'balanced_accuracy': float(np.mean(recs)),
                                             'class_counts_test': torch.bincount(yte, minlength=nout).tolist()}
                else:
                    err = (out - yte).norm(dim=-1)
                    base = (yte - ytr.mean(0)).norm(dim=-1)
                    r2 = 1 - ((out - yte) ** 2).sum() / ((yte - ytr.mean(0)) ** 2).sum()
                    res[f'{name}/{arch}'] = {'median_err_mm': float(err.median()), 'p90_err_mm': float(err.quantile(.9)),
                                             'baseline_median_err_mm': float(base.median()), 'r2': float(r2)}
    return res


def behaviour(rec, length, dt):
    """During rim contact: does the plug move laterally toward the port axis, or just push?"""
    T, N = rec['p'].shape[:2]
    valid = (torch.arange(T, device=rec['p'].device)[:, None] < length[None])
    fp = rec['f_port'][..., 0]
    rim = (fp > 0.05) & (rec['p'][..., 2] > -0.0005) & valid
    pxy, vxy = rec['p'][..., :2], rec['v'][..., :2]
    off = pxy.norm(dim=-1)
    m = rim & (off > 0.1e-3) & (vxy.norm(dim=-1) > 0.5e-3)          # offset > 0.1 mm, lateral speed > 0.5 mm/s
    cos = (-(pxy * vxy).sum(-1) / (off * vxy.norm(dim=-1)).clamp(min=1e-9))[m]
    # does the axial force rise without lateral progress? per rim episode: offset at first rim step vs at capture/end
    out = {'rim_steps_with_lateral_motion': int(m.sum()), 'cos_lateral_velocity_toward_axis': pct(cos.cpu().numpy()),
           'frac_moving_toward_axis': float((cos > 0).float().mean()) if m.any() else None}
    eps = rim.any(0)
    first = rim.float().argmax(0)
    last = (rim.float() * torch.arange(T, device=rim.device)[:, None]).argmax(0)
    o0 = off[first, torch.arange(N, device=rim.device)]; o1 = off[last, torch.arange(N, device=rim.device)]
    out['rim_episodes'] = int(eps.sum())
    out['offset_at_first_rim_mm'] = pct((o0[eps] * 1e3).cpu().numpy())
    out['offset_at_last_rim_mm'] = pct((o1[eps] * 1e3).cpu().numpy())
    # back-off: axial velocity away from the panel (vz > 0 in the port frame) during rim contact
    vz = rec['v'][..., 2]
    out['frac_rim_steps_backing_off'] = float((vz[rim] > 0.5e-3).float().mean()) if rim.any() else None
    return out


def plot(rec, length, outcome, starts, dt, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    N = len(outcome)
    lat0 = (starts['err_p'][:, :2].norm(dim=1) * 1e3).cpu().numpy()
    rim = ((rec['f_port'][..., 0] > 0.05) & (rec['p'][..., 2] > -0.0005)).any(0).cpu().numpy()
    order = np.argsort(-lat0)
    pick = [int(order[0])]                                    # hardest start
    pick += [int(i) for i in np.nonzero(rim)[0][:3] if int(i) not in pick]
    pick += [int(i) for i in range(N) if outcome[i] != 'seated'][:1]
    pick += [int(i) for i in range(N) if i not in pick][: 6 - len(pick)]
    fig, ax = plt.subplots(len(pick), 1, figsize=(8, 2.2 * len(pick)), sharex=True)
    for k, w in enumerate(pick):
        L = int(length[w]); t = np.arange(L) * dt
        f = rec['F_port'][:L, w, 2].clamp(min=0).cpu().numpy()
        p = rec['p'][:L, w].cpu().numpy() * 1e3
        a1 = ax[k]; a2 = a1.twinx()
        a1.plot(t, f, color='C3', label='axial insertion force on plug (N)')
        a2.plot(t, np.hypot(p[:, 0], p[:, 1]), color='C0', label='lateral offset (mm)')
        a2.plot(t, -p[:, 2] / 5.8, color='C2', ls='--', label='depth / seat depth')
        a1.set_ylabel('N'); a2.set_ylabel('mm, depth frac')
        a1.set_title(f'world {w}: start lateral {lat0[w]:.2f} mm, {outcome[w]}', fontsize=9)
    ax[0].legend(loc='upper left', fontsize=7); a2.legend(loc='upper right', fontsize=7)
    ax[-1].set_xlabel('time (s)')
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='runs/P4-insert-best/best.pt')
    ap.add_argument('--nworld', type=int, default=1024)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pol = load_insertion_policy(a.policy)
    report = {'policy': a.policy, 'note': 'forces sampled at the last of the 20 substeps of each 20 ms control step'}
    for cond in ('live', 'r3x1.5'):
        env, rec, length, outcome, starts = rollout(pol, cond, a.nworld, a.seed)
        dt = env.dt
        st = episode_stats(rec, length, outcome, dt)
        k = sum(o == 'seated' for o in outcome); n = len(outcome)
        r = {'n': n, 'seated': k, 'rate': k / n, 'wilson95': list(wilson(k, n)),
             'outcomes': {o: outcome.count(o) for o in set(outcome)},
             'episode': {key: pct(v) for key, v in st.items()},
             'contact_head': head_metrics(rec, length), 'behaviour': behaviour(rec, length, dt)}
        r['episode']['frac_episodes_peak_axial_over_20N'] = float((st['peak_axial_insertion_force_N'] > 20).mean())
        r['episode']['frac_episodes_peak_net_over_20N'] = float((st['peak_net_port_force_N'] > 20).mean())
        r['probes'] = probes(rec, length, env)
        report[cond] = r
        plot(rec, length, outcome, starts, dt, f'{a.out}_{cond}.png', f'{Path(a.policy).name}, {cond}')
        print(cond, json.dumps({kk: r[kk] for kk in ('n', 'rate', 'outcomes')}), flush=True)
        del env
        torch.cuda.empty_cache()
    Path(a.out + '.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
