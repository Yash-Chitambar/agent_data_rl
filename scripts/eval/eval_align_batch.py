"""Align stage on the batched sim chain (eval_chain.Chain): AlignLogic and creep_law from deploy/bimanual/align.py per world, then the unchanged eval_chain insertion. Many worlds per run: the one-world dry run
(deploy/run_bimanual.py) has the same logic and is the code-path check; it takes minutes per episode. Failed worlds hold (the back-off is not simulated here) and do not count as handed over.

Starting point: the sim's approach ends close; an entry offset (uniform box, mm) moves the plug to a real-like stop first (A -0.3, +5.0, 8.4 and A2 -5.2, +8.9, 8.0 by port vision).
Measurement: sim truth with an error model (slope, bias, run constant, pose-dependent noise, frame noise, drop-out). The grasp is the sim's (baseline Part 1), unless --grasp-fidelity is set: say so with each number.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/eval/eval_align_batch.py --name pv --out results/real_align/align_pv.json --nworld 256 --batches 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
from deploy.bimanual.align import DEFAULTS, AlignLogic, PortMeasure, creep_law, creep_state  # noqa: E402
from deploy.bimanual.io_sim import chain_argv, eval_chain  # noqa: E402

EC = eval_chain()


class AlignChain(EC.Chain):
    A = None

    def stage_insert(self, cmd, q_hold, log):
        A, n, dev, h, dt = self.A, self.n, self.dev, self.h, self.dt
        c = self.acfg
        seed = self.seed
        tp0, _ = self.rel_pose()
        in_hand0 = (tp0[:, :2].norm(dim=1) < .015).cpu().numpy()      # the pick lost the reader: not this stage's business, reported apart
        g = np.random.default_rng(seed + 991)
        sm = c['sim_source']
        slope, bias, frame_s, pose_s, run_s = (np.asarray(sm.get(k, d), float) for k, d in (('slope', [1, 1, 1]), ('bias_mm', [0, 0, 0]), ('sigma_mm', [.1, .1, .1]), ('pose_sigma_mm', [.25, .74, .49]), ('run_bias_sigma_mm', [.3, .6, .4])))
        drop = float(sm.get('drop_p', .1))
        rb = g.normal(0, 1, (n, 3)) * run_s
        pose_err, pose_at = np.zeros((n, 3)), np.full((n, 3), np.inf)
        ch = h.chain['right']
        s = (self.R_sr0.T @ self.p_sr0).float().to(dev)
        _, eR = self.port_est
        eRn = eR.cpu().numpy().astype(float)
        ff_lead = FF_fold(self, cmd)
        self.rt.comp['right'] = None
        ql0 = h.believed('left').clone()
        q = lambda: h.believed('right')  # noqa: E731
        cs = creep_state(np.clip(ff_lead, -np.asarray(c['lead_max_rad']), np.asarray(c['lead_max_rad'])))
        qn = q().cpu().numpy().astype(float)
        goal, prev = qn.copy(), qn.copy()

        def fk_tip(qq):
            p, R = ch.fk(qq)
            return p - (R @ s[:, None])[..., 0], p, R

        tip0, _, R0 = fk_tip(q())
        tip0, R0 = tip0.clone(), R0.clone()

        def tip_rel():
            t, _, _ = fk_tip(q())
            return (eR.transpose(1, 2) @ (t - tip0)[..., None])[..., 0].cpu().numpy().astype(float) * 1e3

        def goal_of(ids, delta):
            qq = q()[ids]
            t, p, Rm = fk_tip(qq)
            dw = (eR[ids] @ torch.as_tensor(delta, dtype=torch.float32, device=dev)[..., None])[..., 0] * 1e-3
            pg = p + ((R0[ids] @ s[:, None])[..., 0] - (Rm @ s[:, None])[..., 0]) + dw
            qg, _, _ = ch.ik(pg, R0[ids], qq, iters=30)
            return qg.cpu().numpy().astype(float)

        def tick():
            nonlocal prev
            qq = q().cpu().numpy().astype(float)
            tgt = creep_law(qq, prev, goal, cs, c, dt)
            prev = qq
            self.hstep(torch.as_tensor(tgt, dtype=torch.float32, device=dev), ql0, 'align')

        off = np.asarray(A.entry, float)
        want_off = off[0] + (off[1] - off[0]) * g.random((n, 3))
        # entry: 1 tick per loop, world by world
        phase = np.zeros(n, int)        # 0 entry, 1 settle, 2 step, 3 stopped
        cnt, st_t, tries = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
        want, start = np.zeros((n, 3)), np.zeros((n, 3))
        last_step = [None] * n
        logic = [AlignLogic(c) for _ in range(n)]
        ok = np.zeros(n, bool)
        reason = [''] * n
        fr, ent_got = [[] for _ in range(n)], np.zeros((n, 3))
        t_end = np.zeros(n)
        n0, nmax, ntime = int(c['settle_s'] / dt), int(c['settle_max_s'] / dt), int(c['step_timeout_s'] / dt)
        entered = np.zeros(n, bool)
        def run_phase(max_s):
          nonlocal entered, cnt
          for tk in range(int(max_s / dt)):
            rel = tip_rel()
            # entry steps
            for w in np.where((phase == 0) & (cnt == 0))[0]:
                rem = want_off[w] - rel[w]
                tries[w] += 1
                if np.linalg.norm(rem) < .5 or tries[w] > 30:
                    phase[w], entered[w], cnt[w] = 1, True, 0
                    ent_got[w] = rel[w]
                    continue
                goal[w] = goal_of(torch.tensor([w], device=dev), np.clip(rem, -2., 2.)[None])[0]
                cnt[w] = 1
            cnt[phase == 0] += 1
            cnt[(phase == 0) & ((cnt > 60) | (np.linalg.norm(np.where((phase == 0)[:, None], rel - want_off, 0.), axis=1) < .5))] = 0
            just = np.where((phase == 1) & entered)[0]
            if len(just):      # stage's zero: where the entry ended
                tt, _, RR = fk_tip(q())
                ids = torch.as_tensor(just, device=dev)
                tip0[ids], R0[ids] = tt[ids], RR[ids]
                entered[just] = False
                rel = tip_rel()
            qq = q().cpu().numpy().astype(float)
            still = np.abs(qq - prev).max(1) / dt < c['still_rad_s']
            # steps
            stp = phase == 2
            if stp.any():
                reached = np.linalg.norm(rel - want, axis=1) <= c['move_tol_mm']
                cnt[stp] += 1
                fin = stp & (reached | (cnt >= ntime))
                for w in np.where(fin)[0]:
                    asked = want[w] - start[w]
                    moved = rel[w] - start[w]
                    frac = float(moved @ asked / max(asked @ asked, 1e-9))
                    fr[w].append(frac)
                    last_step[w] = dict(frac=frac, timeout=not reached[w])
                    phase[w], cnt[w] = 1, 0
            # settle -> measure -> decide
            cnt[phase == 1] += 1
            meas = (phase == 1) & (((cnt >= n0) & still) | (cnt >= nmax))
            if meas.any():
                tp, _ = self.rel_pose()
                tp = tp.cpu().numpy().astype(float) * 1e3
                goals_ids, goals_d = [], []
                for w in np.where(meas)[0]:
                    if np.linalg.norm(tp[w] - pose_at[w]) > .3:
                        pose_err[w], pose_at[w] = g.normal(0, 1, 3) * pose_s, tp[w]
                    good = g.random(int(c['frames'])) >= drop
                    frames = (slope * tp[w] + bias + rb[w] + pose_err[w])[None] + g.normal(0, 1, (int(c['frames']), 3)) * frame_s
                    if good.sum() >= int(c['min_good']):
                        v = np.median(frames[good], 0)
                        m = PortMeasure(True, 'sim', float(sm.get('conf', 1.)), float(v[0]), float(v[1]), float(v[2]), float(max(frame_s.max(), pose_s.max())))
                        m.spread = float((frames[good].max(0) - frames[good].min(0)).max())
                    else:
                        m = PortMeasure(False, 'few frames')
                    ls = last_step[w]
                    d = logic[w].update(m, None, rel[w], stuck=bool(ls and ls['timeout'] and ls['frac'] < .25))
                    last_step[w] = None
                    cnt[w] = 0
                    if tk * dt > max_s - 1. and d['kind'] != 'done':
                        d = dict(kind='fail', reason='time limit')
                    if d['kind'] == 'fail':
                        phase[w], reason[w] = 3, d['reason']
                    elif d['kind'] == 'done':
                        phase[w], ok[w], reason[w] = 3, True, 'ok'
                    elif d['kind'] == 'step':
                        goals_ids.append(w)
                        goals_d.append(d['delta'])
                        start[w], want[w] = rel[w], rel[w] + d['delta']
                        phase[w] = 2
                    t_end[w] = tk * dt
                if goals_ids:
                    goal[goals_ids] = goal_of(torch.as_tensor(goals_ids, device=dev), np.asarray(goals_d))
            tick()
            if (phase == 3).all():
                break
        run_phase(c['max_time_s'] + 60)
        for w in np.where(phase != 3)[0]:
            reason[w] = 'time limit (sim loop)'
        tp, trv = self.rel_pose()
        pre = tp.cpu().numpy() * 1e3
        lo = np.asarray(c['handover_lead_rad'], float)
        cs['lead'] = np.clip(cs['lead'], lo[:, 0], lo[:, 1])
        nr = int(c['repress_s'] / dt)
        for k in range(nr):
            u = (k + 1) / nr
            u = u * u * (3 - 2 * u)
            qq = q().cpu().numpy().astype(float)
            tgt = creep_law(qq, prev, goal, cs, c, dt)
            cs['lead'] = np.clip(cs['lead'], lo[:, 0], lo[:, 1])
            prev = qq
            self.hstep(torch.as_tensor(tgt, dtype=torch.float32, device=dev), ql0 + (q_hold - ql0) * u, 'align')
        for _ in range(n0):
            qq = q().cpu().numpy().astype(float)
            tgt = creep_law(qq, prev, goal, cs, c, dt)
            cs['lead'] = np.clip(cs['lead'], lo[:, 0], lo[:, 1])
            prev = qq
            self.hstep(torch.as_tensor(tgt, dtype=torch.float32, device=dev), q_hold, 'align')
        # as AlignStage.handover: a few cycles under the press
        ql0 = q_hold.expand(n, 6).clone()
        ok1 = ok.copy()
        for w in np.where(ok1)[0]:
            lg = logic[w]
            lg.cycle = max(0, c['max_cycles'] - c['post_press_cycles'])
            lg.confirm, lg.last_m, lg.last_e, lg.fused = -1, None, None, []
            phase[w], cnt[w], ok[w], reason[w] = 1, 0, False, 'after the press'
        run_phase(c['post_press_cycles'] * 3.5)
        tp, trv = self.rel_pose()
        post = tp.cpu().numpy() * 1e3
        rot = torch.rad2deg(trv).cpu().numpy()
        handed = ok1 & ok
        cmd2 = torch.as_tensor(q().cpu().numpy().astype(float) + cs['lead'], dtype=torch.float32, device=dev)
        self.ff_was_on = False
        # in-hand net at the hand-over gap, as AlignStage.handover
        if self.inhand_net is not None:
            import torch.nn.functional as F
            net, scale, _ = self.inhand_net
            fr_ = self.render()['wrist_cam']
            img = F.interpolate(fr_.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
            with torch.no_grad():
                self.inhand_est6 = torch.cat([net(img[i:i + 256]) for i in range(0, n, 256)]) * scale
        EC.Chain.stage_insert(self, cmd2, q_hold, log)
        self.result = dict(in_hand0=in_hand0.tolist(), p0=(tp0 * 1e3).tolist(), handed=handed.tolist(), logic_ok=ok.tolist(), reason=reason, pre=pre.tolist(), post=post.tolist(), rot=rot.tolist(), cycles=[l.cycle for l in logic], t_end=t_end.tolist(), entry=ent_got.tolist(), slope=[l.slope() for l in logic], frac=[f for x in fr for f in x], outcome=list(log['insert_outcome']))


def FF_fold(ch, cmd):
    from blind_insertion.envs import friction_comp as FC
    q = ch.h.believed('right')
    lead = cmd - q
    if ch.rt.comp['right'] is not None and ch.fric is not None:
        lead = lead + FC.feedforward(lead, *ch.fric) / ch.h.kp[0]
    return lead.cpu().numpy().astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align1.json'))
    ap.add_argument('--name', default='pv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--batches', type=int, default=4)
    ap.add_argument('--seed', type=int, default=8000)
    ap.add_argument('--entry', default='[[-5,3,0],[1,10,2]]', help='entry offset box, mm: added to the approach stop of the sim (about 0, 0, 6)')
    ap.add_argument('--align_set', action='append', default=[])
    ap.add_argument('--grasp-fidelity', default=None)
    ap.add_argument('--friction', type=float, default=1., help='x friction on right joints 2, 3 from the hand-over stage on (real band about x2.4 on joint 2)')
    ap.add_argument('--enc-offset-deg', type=float, default=1.5)
    ap.add_argument('--sim_phys', default='')
    A = ap.parse_args()
    A.entry = json.loads(A.entry)
    cfg = {k: v for k, v in json.loads(Path(A.config).read_text()).items() if not k.startswith('_')}
    c = dict(DEFAULTS)
    c.update({k: v for k, v in cfg['align'].items() if not k.startswith('_')})
    for kv in A.align_set:
        k, v = kv.split('=', 1)
        c[k] = json.loads(v)
    if A.grasp_fidelity:
        cfg.setdefault('sim', {})['grasp_fidelity'] = A.grasp_fidelity
    if A.sim_phys:
        cfg.setdefault('sim', {})['phys_json'] = A.sim_phys
    pol = ROOT / 'deploy/bundle/models/insert_KJ.pt'
    a = EC.parse(chain_argv(cfg, pol, A.nworld, ['--enc-offset-deg', str(A.enc_offset_deg), '--pick', 'expert_privileged']))
    AlignChain.A = A
    ch = AlignChain(a)
    ch.acfg = c
    orig_tr = ch.stage_transport

    def tr(cmd, q_hold, ep, eR, log):        # eval_chain does not keep the port estimate
        ch.port_est = (ep, eR)
        return orig_tr(cmd, q_hold, ep, eR, log)
    ch.stage_transport = tr
    if A.friction != 1.:
        orig_a = ch.stage_approach

        def ap_(cmd, q_hold, ep, eR, log):
            rt = ch.rt
            d = rt.indices['right'][1]
            rt.frictionloss[:, [d[1], d[2]]] *= A.friction
            return orig_a(cmd, q_hold, ep, eR, log)
        ch.stage_approach = ap_
    out = Path(A.out)
    res = json.loads(out.read_text()) if out.exists() else []
    for b in range(len(res), A.batches):
        t0 = time.time()
        ch.seed = A.seed + b
        ch.run_batch(A.seed + b)
        res.append(ch.result)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res))
        print(json.dumps(summary(res, A.name)), f'{time.time() - t0:.0f} s', flush=True)


def summary(res, name=''):
    from blind_insertion.eval_insertion import wilson
    k = lambda f: [x for r in res for x in r[f]]  # noqa: E731
    h, post, rot, oc, reason, cyc, te = np.array(k('handed')), np.array(k('post')), np.array(k('rot')), k('outcome'), k('reason'), np.array(k('cycles')), np.array(k('t_end'))
    ih = np.array(k('in_hand0'))
    h = h & ih
    n, nh = int(ih.sum()), int(h.sum())
    out = dict(name=name, n_worlds=len(ih), reader_lost_before_stage=int((~ih).sum()), n=n, handed=nh, handed_ci=wilson(nh, n))
    rs = {}
    for r, hh, i_ in zip(reason, h, ih):
        if not hh and i_:
            key = r.split(':')[0].split(' (')[0][:40] or 'moved after the press'
            rs[key] = rs.get(key, 0) + 1
    out['stops'] = rs
    if nh:
        p = post[h]
        s = [o == 'seated' for o, hh in zip(oc, h) if hh]
        ns = int(sum(s))
        out.update(across_abs_p50_p95=np.percentile(np.abs(p[:, 0]), [50, 95]).round(2).tolist(), up_abs_p50_p95=np.percentile(np.abs(p[:, 1]), [50, 95]).round(2).tolist(), lateral_p50_p95=np.percentile(np.hypot(p[:, 0], p[:, 1]), [50, 95]).round(2).tolist(), gap_p5_p50_p95=np.percentile(p[:, 2], [5, 50, 95]).round(2).tolist(), gap_min_max=[float(p[:, 2].min()), float(p[:, 2].max())], tilt_abs_p95_deg=np.percentile(np.abs(rot[h][:, :2]), 95).round(2).tolist(), roll_abs_p95_deg=float(np.percentile(np.abs(rot[h][:, 2]), 95).round(2)), seconds_p50_p95=np.percentile(te[h], [50, 95]).round(1).tolist(), cycles_p50_p95=np.percentile(cyc[h], [50, 95]).tolist(), seated=ns, seated_of=nh, seated_ci=wilson(ns, nh))
        cl = {}
        for o, hh in zip(oc, h):
            if hh:
                cl[o] = cl.get(o, 0) + 1
        out['insert_classes'] = cl
    fr = k('frac')
    if fr:
        out['step_frac_p10_50_90'] = np.percentile(fr, [10, 50, 90]).round(2).tolist()
    return out


if __name__ == '__main__':
    main()
