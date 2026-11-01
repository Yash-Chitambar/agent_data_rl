"""Part 1 ACT DAgger data: roll out a trained ACT pick policy (TorchScript, scripts/act/export_act.py) and,
at a random time per world, hand the world to the privileged expert (PickExpert.takeover), which
continues from the policy's state: lift if the reader is held, continue the descent if the hand is on
the approach line, otherwise open, go back to the pre-grasp and grasp again. Only the expert part is
recorded (the states the policy reached, labelled with the expert's recovery actions), in the same
format as scripts/policy/gen_pick_demos.py (wrist video + .npz), so pick_to_lerobot.py converts it.

Takeover times: `--early_frac` of the worlds uniformly over [t_lo, t_hi] s; the rest at the first
"trouble" event, which oversamples the v4 failure states: gripper commanded closed with the hand
> 8 mm from the grasp, fingers closed on nothing, hand rising without the reader, or the
policy still open and hovering 2 s after reaching grasp height; worlds with no event by t_hi
take over at t_hi unless the reader is already lifted > 50 mm (those give no episode).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/policy/gen_pick_dagger.py \
        --policy runs/P1-act-pick-v4/act_080000_traced.pt --episodes 1536 --out data/act/pick_raw_dagger1
"""
from __future__ import annotations

import argparse, json, math, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio
import numpy as np
import torch
import warp as wp

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickExpert, PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.scenes import reader_grasp as G

wp.config.quiet = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--episodes', type=int, default=1536)
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--out', default='data/act/pick_raw_dagger1')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--res', type=int, default=224)
    ap.add_argument('--layout', default='results/usbc/policy/pick/task_layout.json')
    ap.add_argument('--ens_k', type=float, default=0.01)
    ap.add_argument('--t_lo', type=float, default=3.0)
    ap.add_argument('--t_hi', type=float, default=12.0)
    ap.add_argument('--early_frac', type=float, default=0.6)
    ap.add_argument('--recover_s', type=float, default=11.0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pol = torch.jit.load(a.policy).eval().cuda()
    cfg = PickTaskCfg(physics=BimanualCfg(nworld=a.nworld, device='cuda:0', task_layout=a.layout, render_rgb=True,
                                          camera_resolution=a.res, visual_dr=True), episode_s=a.t_hi + a.recover_s + 1)
    task = PickTask(cfg)
    pool = ThreadPoolExecutor(16)
    dt = task.dt; T = round((a.t_hi + a.recover_s) / dt)
    off = task.offsets[(G.TILT_DEG, False)]
    grasp_z = float(task.table_top + off[1] + task.lay.variants[0].peg_boxes[0][1][1] + G.SUPPORT_GAP)
    summary, ep0, t0 = [], 0, time.time()
    for b in range(math.ceil(a.episodes / a.nworld)):
        seed = a.seed * 100000 + b
        torch.manual_seed(seed)
        obs = task.reset(); ex = PickExpert(task, seed=seed); ex.reset()
        dev, N = task.device, a.nworld
        n_ep = min(N, a.episodes - ep0)
        g = torch.Generator(device=dev); g.manual_seed(seed + 7)
        early = torch.rand(N, generator=g, device=dev) < a.early_frac
        t_take = torch.where(early, a.t_lo + (a.t_hi - a.t_lo) * torch.rand(N, generator=g, device=dev),
                             torch.full((N,), a.t_hi, device=dev))
        k_take = (t_take / dt).round().long()
        taken = torch.zeros(N, dtype=torch.bool, device=dev); k0 = torch.full((N,), -1, dtype=torch.long, device=dev)
        kind = torch.full((N,), -1, dtype=torch.long, device=dev); cause = torch.zeros(N, dtype=torch.long, device=dev)
        tilt_take = torch.zeros(N, device=dev); lift_take = torch.zeros(N, device=dev)
        closed_t = torch.zeros(N, dtype=torch.long, device=dev); low_t = torch.zeros(N, dtype=torch.long, device=dev)
        writers = {i: imageio.get_writer(out / f'ep{ep0 + i:05d}_wrist.mp4', fps=50, codec='libx264', quality=None,
                                         ffmpeg_params=['-crf', '18', '-preset', 'fast'], macro_block_size=8) for i in range(n_ep)}
        states = [[] for _ in range(n_ep)]; actions = [[] for _ in range(n_ep)]; phases = [[] for _ in range(n_ep)]
        start_hold_p = torch.zeros(N, 3, device=dev); start_hold_R = torch.eye(3, device=dev).repeat(N, 1, 1)
        start_tilt = torch.zeros(N, device=dev); final = None
        for t in range(T):
            enc = obs['encoder_rad'][:, :6] + task.runtime.encoder_offsets[0]
            fo = task.runtime.q[:, task.finger_qadr].sum(1)
            state = torch.cat([enc, fo[:, None]], 1).float()
            wrist = obs['rgb']['wrist_cam']
            # --- policy step (all worlds; used where not taken over)
            img = wrist.permute(0, 3, 1, 2).float() / 255
            with torch.no_grad():
                chunk = torch.cat([pol(state[i:i + 128], img[i:i + 128], img[i:i + 128]) for i in range(0, N, 128)])
            H = chunk.shape[1]
            if t == 0:
                wts = torch.exp(-a.ens_k * torch.arange(H, device=dev, dtype=torch.float32)); cum = torch.cumsum(wts, 0)
                ens, cnt = chunk.clone(), torch.ones(H, dtype=torch.long, device=dev)
            else:
                L = ens.shape[1]
                ens = (ens * cum[cnt - 1][None, :, None] + wts[cnt][None, :, None] * chunk[:, :L]) / cum[cnt][None, :, None]
                cnt = (cnt + 1).clamp(max=H - 1)
                ens = torch.cat([ens, chunk[:, L:]], 1); cnt = torch.cat([cnt, torch.ones(H - L, dtype=torch.long, device=dev)])
            act8 = ens[:, 0]; ens, cnt = ens[:, 1:], cnt[1:]
            pact = torch.zeros(N, 14, device=dev)
            pact[:, :6] = ((act8[:, :6] - task.joint_target[:, 0]) / task.cfg.action_scale_rad).clamp(-1, 1)
            pact[:, 6] = torch.where(act8[:, 6] > 0, 1., -1.); pact[:, 13] = 1.
            # --- trouble events (privileged; only choose WHEN to take over)
            q = task.q_true(); p_a, _ = task.chain.fk(q)
            gerr = (p_a - ex.p_g).norm(dim=1)
            closed_t = torch.where(pact[:, 6] < 0, closed_t + 1, torch.zeros_like(closed_t))
            low_t = torch.where((p_a[:, 2] < grasp_z + .008) & (pact[:, 6] > 0), low_t + 1, torch.zeros_like(low_t))
            m = task.pick_metrics()
            ev_close_far = (pact[:, 6] < 0) & (closed_t <= 3) & (gerr > .008)
            ev_nothing = (closed_t >= round(.8 / dt)) & (fo < .012)
            ev_rise_empty = (closed_t > round(.8 / dt)) & (p_a[:, 2] > grasp_z + .02) & (m['lift_m'] < .005)
            ev_hover = low_t >= round(2.0 / dt)
            ev = ~early & (ev_close_far | ev_nothing | ev_rise_empty | ev_hover)
            lifted_ok = (m['lift_m'] > .05) & (fo < .040)          # the policy already has it: no takeover at t_hi
            go = ~taken & (((k_take == t) & (early | ~lifted_ok)) | ev) & (t > round(1.0 / dt))
            if go.any():
                cls = ex.takeover(go)
                kind[go] = torch.where(cls['holding'][go], 2, torch.where(cls['on_line'][go], 1, 0))
                cause[go & ev] = torch.where(ev_close_far, 1, torch.where(ev_nothing, 2, torch.where(ev_rise_empty, 3, 4)))[go & ev]
                taken |= go; k0[go] = t
                tilt_take[go] = m['tilt_deg'][go]; lift_take[go] = m['lift_m'][go]
            prev = ex.phase.clone()
            eact = ex.act()
            act = torch.where(taken[:, None], eact, pact)
            tgt = task.joint_target[:, 0] + eact[:, :6] * task.cfg.action_scale_rad
            ids = torch.nonzero(taken[:n_ep]).flatten().tolist()
            if ids:
                S = state.cpu().numpy(); A = torch.cat([tgt, eact[:, 6:7], ex.done[:, None].float()], 1).cpu().numpy()
                P = prev.cpu().numpy(); W = wrist.cpu().numpy()
                for i in ids:
                    states[i].append(S[i]); actions[i].append(A[i]); phases[i].append(P[i])
                list(pool.map(lambda i: writers[i].append_data(W[i]), ids))
            obs = task.step(act)[0]
            m = task.pick_metrics()
            ent = taken & (ex.phase == 5) & (prev == 4)
            start_hold_p[ent], start_hold_R[ent], start_tilt[ent] = m['inhand_p'][ent], m['inhand_R'][ent], m['tilt_deg'][ent]
            fin = taken & (ex.phase == 6) & (prev == 5)
            if final is None: final = {k: v.clone() for k, v in m.items()}
            for k, v in m.items(): final[k][fin] = v[fin]
            if taken.all() and ex.done.all(): break
        for w in writers.values(): w.close()
        dp = (final['inhand_p'] - start_hold_p).norm(dim=1)
        dR = torch.rad2deg(K.rot_log(final['inhand_R'] @ start_hold_R.transpose(1, 2)).norm(dim=1))
        ok = (taken & ex.done & ~task.failed & (final['lift_m'] > .06) & (final['tilt_deg'] < 3) & (start_tilt < 3) & (dp < 1e-3) & (dR < 1.)).cpu().numpy()
        for i in range(n_ep):
            if not states[i]:
                np.savez_compressed(out / f'ep{ep0 + i:05d}.npz', state=np.zeros((0, 7), np.float32), action=np.zeros((0, 8), np.float32),
                                    phase=np.zeros(0, np.int8), meta=json.dumps(dict(episode=ep0 + i, ok=False, length=0)))
                summary.append(dict(episode=ep0 + i, ok=False, length=0)); continue
            S, A, P = np.stack(states[i]), np.stack(actions[i]), np.stack(phases[i])
            done_t = int(np.argmax(A[:, 7] > 0)) if A[:, 7].any() else len(A) - 1
            L = min(len(A), done_t + 10)
            meta = dict(episode=ep0 + i, ok=bool(ok[i]) and L >= 60, length=L, takeover_s=float(k0[i] * dt), early=bool(early[i]),
                        kind=['retreat', 'on_line', 'holding'][int(kind[i])], cause=['time', 'close_far', 'closed_on_nothing', 'rise_empty', 'hover_open'][int(cause[i])],
                        lift_mm=float(final['lift_m'][i] * 1e3), tilt_deg=float(final['tilt_deg'][i]),
                        reader_tilt_take_deg=float(tilt_take[i]), reader_lift_take_mm=float(lift_take[i] * 1e3), ik_ok=bool(ex.ik_ok[i]),
                        settle_err_mm=float(ex.settle_err[i, 0] * 1e3) if hasattr(ex, 'settle_err') else -1.,
                        settle_err_deg=float(ex.settle_err[i, 1]) if hasattr(ex, 'settle_err') else -1.)
            np.savez_compressed(out / f'ep{ep0 + i:05d}.npz', state=S[:L].astype(np.float32), action=A[:L].astype(np.float32),
                                phase=P[:L].astype(np.int8), meta=json.dumps(meta))
            summary.append(meta)
        ep0 += n_ep
        okm = [s['ok'] for s in summary]
        print(f'batch {b}: {ep0} episodes, recovered ok {np.mean(okm):.3f}, {time.time() - t0:.0f} s', flush=True)
    import collections
    (out / 'summary.json').write_text(json.dumps(dict(
        episodes=len(summary), ok=float(np.mean([m['ok'] for m in summary])), fps=50, res=a.res, cams=['wrist'],
        kinds=collections.Counter(m.get('kind') for m in summary), causes=collections.Counter(m.get('cause') for m in summary),
        ok_by_kind={k: float(np.mean([m['ok'] for m in summary if m.get('kind') == k])) for k in ('retreat', 'on_line', 'holding')
                    if any(m.get('kind') == k for m in summary)},
        args=vars(a), episodes_meta=summary), indent=1))


if __name__ == '__main__':
    main()
