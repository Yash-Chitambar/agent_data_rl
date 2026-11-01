"""Replay a logged real (or dry) bimanual run in the sim and compare joint tracking: fit friction, gains, latency
and contact from real runs.

The commands SafeArm actually sent (ticks.npz `sent`, motor frame, both arms + grippers) are replayed open loop in a
BimanualRuntime (the chain's scene, identified dynamics, i2rt native gains, the same driver-gravity residual) from
the first logged reading, in V parallel worlds = a grid of variants (Coulomb friction scale, gravity scale, latency
in control steps, kp scale). The right-arm Coulomb feed-forward is switched on where the run used it. Objects: the
PC at the run's ZED estimate (--objects pc, default) or both PC and reader moved off the table (--objects none:
free-space tracking only). Outputs in <run>/replay/ (or --out):
  replay.json   per variant and stage: RMS / max joint tracking error sim - real (mrad) per joint, best variant
  replay.npz    sim joint trajectories (V, T, 2, 6) motor frame, real q, stages, variants
  joints_<arm>.png, tcp_z.png, hold_force.png   real vs best / nominal variant, stage bands

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. .venv/bin/python scripts/real2sim/replay_real_run.py data/real_runs/<date>/<run> \
        --friction 0.5 1.0 1.5 --latency 0 1 [--kp-scale 1.0] [--gravity 1.0] [--objects pc|none]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SIDES = ('right', 'left')
STAGES = ('0 capture', '1 pick', '2 hold', '3 transport', '4 approach', '5 insert', '6 release', '7 home', 'manual')


def build(n, objects, port, device):
    import mujoco_warp as mjw
    import warp as wp
    from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
    from blind_insertion.envs import kinematics as K
    rt = BimanualRuntime(BimanualCfg(nworld=n, device=device, task_layout='results/usbc/policy/pick/task_layout.json',
                                     friction_scale_range=(1., 1.), gravity_scale_jitter=0.))
    ids = torch.arange(n, device=rt.device)
    rt.reset(ids)
    fa, pa = rt.pc_qadr, rt.lay.peg_qadr
    rt.q[:, pa:pa + 3] = torch.tensor([-0.30, -0.40, rt.q[0, pa + 2]], device=rt.device)   # reader on the table, out of reach
    if objects == 'none' or port is None:
        rt.q[:, fa:fa + 3] = torch.tensor([-0.30, 1.00, rt.q[0, fa + 2]], device=rt.device)
    else:
        p, R = port
        rt.q[:, fa:fa + 3] = torch.as_tensor(p, dtype=torch.float32, device=rt.device)
        rt.q[:, fa + 3:fa + 7] = K.mat_to_quat(torch.as_tensor(R, dtype=torch.float32, device=rt.device)[None])[0]
    with wp.ScopedDevice(device):
        mjw.forward(rt.m, rt.d)
    return rt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run')
    ap.add_argument('--friction', type=float, nargs='+', default=[0.5, 1.0, 1.5], help='x identified Coulomb friction')
    ap.add_argument('--gravity', type=float, nargs='+', default=[1.0], help='x identified gravity scale')
    ap.add_argument('--latency', type=int, nargs='+', default=[0, 1], help='control steps')
    ap.add_argument('--kp-scale', type=float, default=1.0, help='x the arm PD gains (kp and kd), all worlds (gains are not batched)')
    ap.add_argument('--objects', choices=('pc', 'none'), default='pc')
    ap.add_argument('--max-ticks', type=int, default=0)
    ap.add_argument('--out', default='')
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    from deploy.bimanual.runlog import load_run
    from blind_insertion.envs import friction_comp as FC
    meta, st, events, _ = load_run(a.run)
    tk = st['ticks']
    order = np.argsort(tk['k'])
    sent, q_real, stage = tk['sent'][order], tk['q'][order], tk['stage'][order].astype(int)
    ff = tk['ff'][order]
    if a.max_ticks:
        sent, q_real, stage, ff = sent[:a.max_ticks], q_real[:a.max_ticks], stage[:a.max_ticks], ff[:a.max_ticks]
    T = len(sent)
    off = np.asarray([meta['calibration']['joint_offsets_rad'][s] for s in SIDES])
    zed = next((e for e in events if e.get('est') == 'zed_pc'), None)
    port = None
    if zed is not None:
        from deploy.bimanual.scene import SceneModel
        sc = SceneModel('cpu')
        p, R = sc.port_frame(torch.tensor([zed['port_xy']]), torch.tensor([np.radians(zed['yaw_deg'])]))
        port = (p[0].numpy(), R[0].numpy())
    variants = [dict(friction=f, gravity=g, latency=l, kp=a.kp_scale) for f, g, l in itertools.product(a.friction, a.gravity, a.latency)]
    V = len(variants)
    rt = build(V, a.objects, port, a.device)
    dev = rt.device
    fr = torch.tensor([v['friction'] for v in variants], device=dev)
    for k_, sd in enumerate(SIDES):
        dofs = torch.tensor(rt.indices[sd][1], device=dev)
        rt.frictionloss[:, dofs] = rt.coulomb[k_][None] * fr[:, None]
        rt.gravity_scale[:, k_] = rt.gravity_scale_base[k_][None] * torch.tensor([v['gravity'] for v in variants], device=dev)[:, None]
    if a.kp_scale != 1.0:
        import warp as wp
        gp, bp = wp.to_torch(rt.m.actuator_gainprm), wp.to_torch(rt.m.actuator_biasprm)
        acts = [x for sd in SIDES for x in rt.indices[sd][2]]
        gp[..., acts, 0] *= a.kp_scale
        bp[..., acts, 1] *= a.kp_scale
        bp[..., acts, 2] *= a.kp_scale
    off_t = torch.as_tensor(off, dtype=torch.float32, device=dev)
    q0 = torch.as_tensor(q_real[0], dtype=torch.float32, device=dev) + off_t
    for i, sd in enumerate(SIDES):
        rt.q[:, rt.indices[sd][0]] = q0[i]
        rt.v[:, rt.indices[sd][1]] = 0
    lat = torch.tensor([v['latency'] for v in variants], device=dev)
    L = int(lat.max()) + 1
    hist = None
    ff_cfg = FC.parse(meta['config']['fric_comp_right'], side='right')
    q_sim = np.zeros((V, T, 2, 6))
    tt = tk['t'][order][:T]
    gaps = 0
    for k in range(T):
        if k > 0 and tt[k] - tt[k - 1] > 0.1:     # a slow move_to (not in the tick stream) happened: restart from the real reading
            qk = torch.as_tensor(q_real[k - 1] if k > 1 else q_real[0], dtype=torch.float32, device=dev) + off_t
            for i, sd in enumerate(SIDES):
                rt.q[:, rt.indices[sd][0]] = qk[i]
                rt.v[:, rt.indices[sd][1]] = 0
            hist, gaps = None, gaps + 1
        tgt = torch.as_tensor(sent[k, :, :6], dtype=torch.float32, device=dev) + off_t
        g = sent[k, :, 6]
        act = torch.zeros(V, 14, device=dev)
        act[:, :6], act[:, 7:13] = tgt[0], tgt[1]
        gm = meta['config']['gripper']
        act[:, 6] = 1. if g[0] > gm['right']['close'] + 0.05 else -1.
        act[:, 13] = 1. if g[1] > gm['left']['close'] + 0.05 else -1.
        hist = act[None].repeat(L, 1, 1) if hist is None else torch.cat([act[None], hist[:-1]], 0)
        rt.comp['right'] = ff_cfg if np.abs(ff[k, 0]).max() > 0 else None
        rt.step(hist[lat, torch.arange(V, device=dev)])
        q_sim[:, k] = torch.stack([rt.q[:, rt.indices[sd][0]] for sd in SIDES], 1).cpu().numpy() - off[None]
    # tick k logs the command sent at k and the reading after it: q_sim[:, k] (after stepping sent[k]) pairs with q_real[k]
    err = (q_sim - q_real[None]) * 1e3          # mrad
    out = {'run': str(a.run), 'ticks': T, 'restarts_after_moves': gaps, 'objects': a.objects, 'variants': variants, 'stages': {}}
    for s in sorted(set(stage.tolist())):
        m = stage == s
        name = STAGES[s] if 0 <= s < len(STAGES) else str(s)
        rms = np.sqrt((err[:, m] ** 2).mean(1))            # (V, 2, 6)
        best = int(rms.mean((1, 2)).argmin())
        out['stages'][name] = {'n': int(m.sum()), 'rms_mrad': rms.round(2).tolist(), 'max_mrad': np.abs(err[:, m]).max(1).round(2).tolist(),
                               'best_variant': variants[best], 'best_rms_mean_mrad': float(rms[best].mean())}
    tot = np.sqrt((err ** 2).mean((1, 2, 3)))
    out['overall_best'] = variants[int(tot.argmin())]
    out['overall_rms_mrad'] = tot.round(3).tolist()
    od = Path(a.out or Path(a.run) / 'replay')
    od.mkdir(parents=True, exist_ok=True)
    (od / 'replay.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(od / 'replay.npz', q_sim=q_sim, q_real=q_real, stage=stage, variants=json.dumps(variants))
    plots(od, q_sim, q_real, stage, variants, int(tot.argmin()), st, events, meta)
    print(json.dumps({'overall_best': out['overall_best'], 'overall_rms_mrad': out['overall_rms_mrad'],
                      'per_stage_best': {k: v['best_variant'] for k, v in out['stages'].items()}}, indent=1))
    print('wrote', od)


def plots(od, q_sim, q_real, stage, variants, best, st, events, meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = q_real.shape[0]
    t = np.arange(T) * 0.02
    cols = plt.cm.tab10(np.arange(10))

    def bands(ax):
        for s in sorted(set(stage.tolist())):
            m = np.where(stage == s)[0]
            if len(m):
                ax.axvspan(t[m[0]], t[m[-1]], color=cols[s % 10], alpha=0.08)
                ax.text(t[m[0]], ax.get_ylim()[1], STAGES[s] if 0 <= s < len(STAGES) else str(s), fontsize=6, va='top')
    nom = next((i for i, v in enumerate(variants) if v['friction'] == 1.0 and v['latency'] == 0), 0)
    for i, sd in enumerate(SIDES):
        fig, axs = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
        for j in range(6):
            ax = axs[j]
            ax.plot(t, q_real[:, i, j], 'k', lw=1, label='real')
            ax.plot(t, q_sim[best, :, i, j], 'C1', lw=.8, label=f'sim best {variants[best]}')
            if nom != best:
                ax.plot(t, q_sim[nom, :, i, j], 'C0', lw=.6, alpha=.7, label=f'sim {variants[nom]}')
            ax.set_ylabel(f'j{j + 1} (rad)')
            bands(ax)
        axs[0].legend(fontsize=6, loc='lower right')
        axs[-1].set_xlabel('time (s)')
        fig.suptitle(f'{sd} arm, motor frame: real vs sim replay of the sent commands (n = {T} ticks)')
        fig.tight_layout()
        fig.savefig(od / f'joints_{sd}.png', dpi=110)
        plt.close(fig)
    from deploy.bimanual.scene import SceneModel
    sc = SceneModel('cpu')
    off = sc.encoder_offsets.numpy()
    fig, ax = plt.subplots(figsize=(11, 3.5))
    for i, sd in enumerate(SIDES):
        zr = sc.tcp_in_table(sd, torch.tensor(q_real[:, i] + off[i], dtype=torch.float32)).numpy()
        zs = sc.tcp_in_table(sd, torch.tensor(q_sim[best, :, i] + off[i], dtype=torch.float32)).numpy()
        ax.plot(t, zr * 1e3, color=f'C{i}', label=f'{sd} real')
        ax.plot(t, zs * 1e3, '--', color=f'C{i}', label=f'{sd} sim')
    ax.set_ylabel('TCP above table (mm, encoder FK)')
    bands(ax)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(od / 'tcp_z.png', dpi=110)
    plt.close(fig)
    hr = next((e for e in events if e.get('kind') == 'hold_result'), None)
    if hr is not None and 'point' in hr:
        J_point, nrm = torch.tensor(hr['point'])[None], torch.tensor(hr['normal'])
        tgt = st['ticks']['target_b'][np.argsort(st['ticks']['k'])][:T, 1]
        kp = sc.kp[1].numpy()

        def fn(qb):
            p, R, J = sc.chain['left'].fk(torch.tensor(qb, dtype=torch.float32), with_jac=True)
            r = J_point - p
            Jl = J[:, :3] - torch.cross(r[:, :, None].expand(-1, -1, 6), J[:, 3:], dim=1)
            tau = torch.tensor(kp * (tgt - qb), dtype=torch.float32)
            F = (torch.linalg.pinv(Jl.transpose(1, 2)) @ tau[..., None])[..., 0]
            return (F @ nrm).numpy()
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(t, fn(q_real[:, 1] + off[1]), 'k', label='real (PD force estimate)')
        ax.plot(t, fn(q_sim[best, :, 1] + off[1]), 'C1', label='sim replay')
        ax.set_ylabel('left normal force estimate (N)')
        bands(ax)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(od / 'hold_force.png', dpi=110)
        plt.close(fig)


if __name__ == '__main__':
    main()
