"""Force budget of the final push: the right YAM as its own force sensor (docs/real/FORCE_BUDGET_20261115.md).

    PYTHONPATH=. python scripts/real/force_budget.py fit      # free-space torque model, leave-one-date-out residuals
    PYTHONPATH=. python scripts/real/force_budget.py contact  # force at the plug for every logged push and K4 insertion
    PYTHONPATH=. python scripts/real/force_budget.py all      # both, then the label check

Data (read-only): data/real_runs/2026-11-1{0,1,3}/full_* (not dry runs), ticks.npz at 50 Hz: q (motor frame), qd, eff (driver
effort = measured motor torque, N m), target_b, ff, stage. Believed angles = q + rig joint offsets (as the chain).

Free-space model of the right arm (per tick, per joint j), linear in its parameters:
  eff_j = sum_b [m_b gu.Jp_j(p_b) + (m l)_b . R_b^T (gu x w_j)]       gravity: 4 parameters per body (link_2 .. link_6 lumped with the
                                                                       gripper, camera and reader), shared by all joints
        + s_j tau_inert_j(q, qd, qdd)                                  nominal Menagerie M qdd + C qd, one scale per joint
        + Fc_j tanh(qd_j / 0.02) + Fv_j qd_j + b_j                     Coulomb + viscous friction, offset
Ridge prior toward the Menagerie masses. Contact-free ticks: stages 2 hold, 3 transport, 4 approach, align, 7 home (the plug is
at least about 5 mm from the port in the align stage; the left arm is not used). Held out by date.
External wrench at the TCP: J^T [F, M] = eff - model (least squares, as springback_timeline.F_eff); F along -port_z = into the port.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts/real'))
import insertion_log_analysis as L  # noqa: E402

RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
DATES = ('2026-11-10', '2026-11-11', '2026-11-13')
OUT = ROOT / 'results/real_align/force_budget'
YAM = '/home/yashchitambar/peg_test/third_party/mujoco_menagerie/i2rt_yam/yam_linear_4310.xml'
STAGES = ('0 capture', '1 pick', '2 hold', '3 transport', '4 approach', '5 insert', '6 release', '7 home', 'manual', 'align')
FREE = (2, 3, 4, 7, 9)
KP = np.array([80., 80., 80., 40., 10., 10.])
KD = np.array([5., 5., 5., 1.5, 1.5, 1.5])
OFF = np.asarray(L.OFF, float)
GU = np.array([0., 0., 9.81])
QD_EPS = 0.02
# operator labels (docs/real/DEPTH_SPRINGBACK_20261114.md section 4)
LABEL = {'demo_4': 'full', 'k4_5': 'full', 'k4_3': 'full', 'fastmove_1': 'full',
         'all9b_full_1': 'short', 'demo_1': 'short', 'all9_full_1': 'short', 'k4_1': 'short', 'kjrc_1': 'short',
         'kjrc_3_push': 'short', 'kjrc_2': 'short'}


# ------------------------------------------------------------------ regressors
class Reg:
    BODIES = ('link_2', 'link_3', 'link_4', 'link_5', 'link_6')

    def __init__(self):
        m = mujoco.MjModel.from_xml_path(YAM)
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT     # no joint-limit forces in mj_inverse
        m.body_gravcomp[:] = 0.                                             # Menagerie sets gravcomp 1: we want the real load
        self.m, self.d = m, mujoco.MjData(m)
        self.bid = [m.body(b).id for b in self.BODIES]
        # nominal parameters: link_6 lumped with its subtree (fingers, pads, camera), in link_6 coordinates
        self.d.qpos[:] = 0
        mujoco.mj_forward(m, self.d)
        th0 = []
        for b in self.bid:
            sub = [i for i in range(m.nbody) if self._anc(i, b)] if b == self.bid[-1] else [b]
            M = sum(m.body_mass[i] for i in sub)
            c = sum(m.body_mass[i] * self.d.xipos[i] for i in sub) / M
            ml = M * (self.d.xmat[b].reshape(3, 3).T @ (c - self.d.xpos[b]))
            th0 += [M, *ml]
        self.theta0 = np.array(th0)

    def _anc(self, i, b):
        while i > 0:
            if i == b:
                return True
            i = self.m.body_parentid[i]
        return False

    def __call__(self, q, qd, qdd):
        """q, qd, qdd (n, 6) believed frame -> gravity regressor (n, 6, 20), nominal inertial torque (n, 6), nominal gravity (n, 6)."""
        m, d = self.m, self.d
        n = len(q)
        G = np.zeros((n, 6, 4 * len(self.bid)))
        TI = np.zeros((n, 6))
        TG = np.zeros((n, 6))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for i in range(n):
            d.qpos[:6], d.qpos[6:] = q[i], 0.
            d.qvel[:6], d.qvel[6:] = qd[i], 0.
            d.qacc[:6], d.qacc[6:] = qdd[i], 0.
            mujoco.mj_inverse(m, d)
            full = d.qfrc_inverse[:6].copy()
            d.qvel[:], d.qacc[:] = 0., 0.
            mujoco.mj_inverse(m, d)
            TG[i] = d.qfrc_inverse[:6]
            TI[i] = full - TG[i]
            for k, b in enumerate(self.bid):
                mujoco.mj_jacBody(m, d, jp, jr, b)
                R = d.xmat[b].reshape(3, 3)
                G[i, :, 4 * k] = GU @ jp[:, :6]
                G[i, :, 4 * k + 1:4 * k + 4] = (R.T @ np.cross(GU[None], jr[:, :6].T).T).T
        return G, TI, TG


# ------------------------------------------------------------------ data
def runs():
    out = []
    for date in DATES:
        for d in sorted(glob.glob(str(RUNS / date / 'full_*'))):
            if not os.path.exists(os.path.join(d, 'ticks.npz')):
                continue
            meta = json.load(open(os.path.join(d, 'meta.json')))
            if meta.get('dry_run'):
                continue
            out.append(d)
    return out


def short(run):
    b = os.path.basename(run.rstrip('/'))
    return b.split('_', 2)[2]


def load_ticks(run, reg):
    t = np.load(os.path.join(run, 'ticks.npz'))
    q = t['q'][:, 0] + OFF
    qd = t['qd'][:, 0]
    qdd = np.gradient(qd, 0.02, axis=0)
    G, TI, TG = reg(q, qd, qdd)
    return dict(k=t['k'].astype(int), stage=t['stage'].astype(int), q=q, qd=qd, qdd=qdd, eff=t['eff'][:, 0], tgt=t['target_b'][:, 0],
                ff=t['ff'][:, 0], grip=t['grip'][:, 0], G=G, TI=TI, TG=TG)


def build(reg, cache=ROOT / 'runs/force_budget/ticks_cache.npz'):      # regenerable cache, not in results
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    cols = {k: [] for k in ('run', 'date', 'k', 'stage', 'q', 'qd', 'qdd', 'eff', 'tgt', 'ff', 'grip', 'G', 'TI', 'TG')}
    for r in runs():
        x = load_ticks(r, reg)
        n = len(x['k'])
        cols['run'].append(np.full(n, short(r), dtype=object))
        cols['date'].append(np.full(n, r.split('/')[-2], dtype=object))
        for k in x:
            cols[k].append(x[k])
        print(f'{short(r)}: {n} ticks', flush=True)
    data = {k: np.concatenate(v) for k, v in cols.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def free_mask(D):
    """Contact-free right-arm ticks: stages hold, transport, approach, align, home; finite effort; not the first 5 ticks of a stage."""
    st = D['stage']
    m = np.isin(st, FREE) & np.isfinite(D['eff']).all(1)
    edge = np.r_[True, (st[1:] != st[:-1]) | (D['run'][1:] != D['run'][:-1])]
    for s in range(1, 6):
        m &= ~np.r_[np.zeros(s, bool), edge[:-s]]
    m &= ~edge
    return m


# ------------------------------------------------------------------ model
def features(D, idx, model):
    """Design matrix rows (n*6, p) for model 'nominal' (fixed), 'lin' (gravity + friction + inertia)."""
    n = len(idx)
    G = D['G'][idx]                                  # (n, 6, 20)
    qd = D['qd'][idx]
    TI = D['TI'][idx]
    if model == 'lin':
        fr = np.zeros((n, 6, 6 * 4))
        for j in range(6):
            fr[:, j, 4 * j] = np.tanh(qd[:, j] / QD_EPS)
            fr[:, j, 4 * j + 1] = qd[:, j]
            fr[:, j, 4 * j + 2] = 1.
            fr[:, j, 4 * j + 3] = TI[:, j]
        return np.concatenate([G, fr], 2)
    raise ValueError(model)


def fit_lin(D, idx, reg, lam=50.):
    X = features(D, idx, 'lin')
    y = D['eff'][idx]
    w = 1. / np.array([0.3, 1.0, 1.0, 0.3, 0.2, 0.1])         # joint noise scale (N m), so every joint counts
    Xw = (X * w[None, :, None]).reshape(-1, X.shape[-1])
    yw = (y * w[None]).reshape(-1)
    p = X.shape[-1]
    th0 = np.zeros(p)
    th0[:20] = reg.theta0
    th0[20 + 3::4] = 1.                                        # inertial scale 1
    P = np.full(p, 1e-3)
    P[:20] = lam / 1e3                                        # weak pull of the masses toward the Menagerie values (per tick)
    A = Xw.T @ Xw + np.diag(P * len(idx))
    th = np.linalg.solve(A, Xw.T @ yw + P * len(idx) * th0)
    return th


def predict(D, idx, th, model='lin'):
    if model == 'nominal':
        return D['TG'][idx] + D['TI'][idx]
    return np.einsum('njp,p->nj', features(D, idx, 'lin'), th)


# ------------------------------------------------------------------ force mapping
def jac(qb):
    qt = torch.as_tensor(np.asarray(qb), dtype=torch.float32).reshape(-1, 6)
    _, _, J = L.FK.fk(qt, with_jac=True)
    return J.double().numpy()


SIG = np.array([0.21, 1.18, 0.65, 0.085, 0.085, 0.07])       # held-out free-space residual rms per joint (N m), fit of 2026-11-15


def axial_rows(qb, zin, sig=SIG):
    """Rows a with F_axial = a . tau at believed angles qb (one pose (6,) or (n, 6)):
    'wrench': full wrench at the TCP, J^T [F, M] = tau solved exactly (as springback_timeline.F_eff; moments free);
    'tip':    a pure force at the plug tip, tau = Jv_tip^T F, weighted least squares with the joint noise sig (no moment)."""
    qb = np.asarray(qb, float).reshape(-1, 6)
    J = jac(qb)                                                   # (n, 6, 6) [v; w] at the TCP
    qt = torch.as_tensor(qb, dtype=torch.float32)
    _, R = L.FK.fk(qt)
    r = -(R.double().numpy() @ L.TIP_S.double().numpy())          # TCP -> plug tip, world
    Jv = J[:, :3] + np.cross(J[:, 3:].transpose(0, 2, 1), r[:, None, :]).transpose(0, 2, 1)
    W6 = np.linalg.pinv(J.transpose(0, 2, 1))                     # [F, M] = W6 tau
    a6 = np.einsum('nij,i->nj', W6[:, :3], zin)
    A = Jv.transpose(0, 2, 1) / sig[None, :, None]                # (n, 6, 3)
    Wt = np.linalg.pinv(A) / sig[None, None, :]                   # F = Wt tau
    a3 = np.einsum('nij,i->nj', Wt, zin)
    return dict(wrench=a6, tip=a3)


def ref_pose(D):
    """Reference push pose: median believed angles at the K4 seats of the logged pushes; port axis of all9b_full_1."""
    qs, zs = [], []
    for r in runs():
        ev = L.events(r)
        st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
        seat = next((e for e in ev if e['kind'] == 'seated_detected'), None)
        if seat is None or '5 insert' not in st:
            continue
        k = st['5 insert'] + int(seat['step']) + 1
        m = (D['run'] == short(r)) & (D['k'] == k)
        if m.any():
            qs.append(D['q'][m][0])
            pc = [e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc'][-1]
            zs.append(-L.port_R(pc['yaw_deg'])[:, 2])
    q = np.median(qs, 0)
    z = np.mean(zs, 0)
    return q, z / np.linalg.norm(z), len(qs)


# ------------------------------------------------------------------ commands
def cmd_fit(A):
    reg = Reg()
    D = build(reg)
    fm = free_mask(D)
    qref, zref, nref = ref_pose(D)
    rows = {k: v[0] for k, v in axial_rows(qref, zref).items()}
    moving = (np.abs(D['qd']) > 0.01).any(1)
    res = dict(n_ticks_free=int(fm.sum()), ref_pose=dict(q_b=qref.round(4).tolist(), z_in=zref.round(4).tolist(), n_seats=nref),
               note='held-out residual of eff, N m per joint; axial = the residual mapped to a force along the port axis at the reference push pose', by_date={})
    for date in DATES:
        tr = np.where(fm & (D['date'] != date))[0]
        te = np.where(fm & (D['date'] == date))[0]
        th = fit_lin(D, tr, reg)
        out = {}
        for name, mdl in (('nominal', 'nominal'), ('lin', 'lin')):
            r = D['eff'][te] - predict(D, te, th, mdl)
            if name == 'nominal':        # nominal + per-joint offset from the training dates (a fair baseline)
                r = r - np.median(D['eff'][tr] - predict(D, tr, th, 'nominal'), 0)
            mv = moving[te]
            out[name] = dict(rms_nm=np.sqrt((r ** 2).mean(0)).round(3).tolist(), p95_abs_nm=np.percentile(np.abs(r), 95, 0).round(3).tolist(),
                             n_ticks=int(len(te)), n_moving=int(mv.sum()), n_runs=int(len(set(D['run'][te]))))
            for mp, a in rows.items():
                fa = r @ a
                out[name]['axial_n_' + mp] = dict(p50_abs=round(float(np.median(np.abs(fa))), 2), p95_abs=round(float(np.percentile(np.abs(fa), 95)), 2),
                                                  median=round(float(np.median(fa)), 2),
                                                  p95_abs_moving=round(float(np.percentile(np.abs(fa[mv]), 95)), 2) if mv.any() else None,
                                                  p95_abs_static=round(float(np.percentile(np.abs(fa[~mv]), 95)), 2) if (~mv).any() else None)
        res['by_date'][date] = out
        print(date, json.dumps(out), flush=True)
    th = fit_lin(D, np.where(fm)[0], reg)
    np.save(OUT / 'free_model_theta.npy', th)
    fr = th[20:].reshape(6, 4)
    res['all_dates'] = dict(coulomb_nm=fr[:, 0].round(3).tolist(), viscous_nm_s=fr[:, 1].round(3).tolist(), offset_nm=fr[:, 2].round(3).tolist(),
                            inertia_scale=fr[:, 3].round(3).tolist(), mass_kg=th[0:20:4].round(3).tolist(), mass0_kg=reg.theta0[0:20:4].round(3).tolist())
    # sensitivity of the axial force to 1 N m on each joint at the reference pose (N per N m)
    res['axial_per_nm'] = {k: v.round(3).tolist() for k, v in rows.items()}
    (OUT / 'free_model.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['all_dates']), res['axial_per_nm'])


def push_ticks(run):
    """Tick marks of the insertion and the final push (as springback_timeline.py)."""
    ev = L.events(run)
    st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
    seat = next((e for e in ev if e['kind'] == 'seated_detected'), None)
    push = next((e for e in ev if e['kind'] == 'insert_final_push'), None)
    skip = next((e for e in ev if e['kind'] == 'insert_final_push_skipped'), None)
    pc = [e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc'][-1]
    if '5 insert' not in st:
        return None
    m = dict(ev=ev, ins=st['5 insert'], rel=st.get('6 release'), zin=-L.port_R(pc['yaw_deg'])[:, 2], push=push, skip=skip, seat=seat)
    if seat is not None:
        m['seat'] = st['5 insert'] + int(seat['step']) + 1
        m['push0'] = m['seat'] + 10
    if push is not None:
        un = push.get('unload')
        m['drop'] = int(un['tick_start']) if un else int(push['tick']) - 15
    return m


def ramp_fit(adv, F, lo, hi):
    """Slope (mm/N) and intercept of advance against force for lo <= F <= hi."""
    s = (F >= lo) & (F <= hi)
    if s.sum() < 5:
        return None, None
    a, b = np.polyfit(F[s], adv[s], 1)
    return float(a), float(b)


def cmd_contact(A):
    reg = Reg()
    D = build(reg)
    fm = free_mask(D)
    th = {d: fit_lin(D, np.where(fm & (D['date'] != d))[0], reg) for d in DATES}        # held out by date
    out, traces = [], {}
    for run in runs():
        name = short(run)
        m = push_ticks(run)
        if m is None or m.get('seat') is None:
            continue
        date = run.split('/')[-2]
        sel = np.where(D['run'] == name)[0]
        kk = D['k'][sel]
        end = m['rel'] if m['rel'] is not None else kk[-1]
        i = sel[(kk >= m['ins']) & (kk < end + 40)]
        k = D['k'][i]
        q = D['q'][i]
        rows = axial_rows(q, m['zin'])
        r = D['eff'][i] - predict(D, i, th[date])
        tau_pd = KP * (D['tgt'][i] - q)
        F_tip = np.einsum('nj,nj->n', r, rows['tip'])
        F_wr = np.einsum('nj,nj->n', r, rows['wrench'])
        F_pd = np.einsum('nj,nj->n', tau_pd, rows['wrench'])
        fc = th[date][20:].reshape(6, 4)[:, 0]
        band = np.abs(rows['tip']) @ np.abs(fc)                                         # +/- static friction anywhere in its band
        p, _ = L.FK.fk(torch.as_tensor(q, dtype=torch.float32))
        p = p.double().numpy()
        j_seat = int(np.searchsorted(k, m['seat']))
        adv = (p - p[j_seat]) @ m['zin'] * 1e3                                           # into the port, mm, from the seat
        eR = L.port_R(float(np.degrees(np.arctan2(m["zin"][1], m["zin"][0]))))
        F_up = np.einsum('nj,nj->n', r, axial_rows(q, eR[:, 1])['tip'])                  # port y (up)
        F_ac = np.einsum('nj,nj->n', r, axial_rows(q, eR[:, 0])['tip'])                  # port x (across)
        tl = np.load(os.path.join(run, 'ticks.npz'))
        jl = np.searchsorted(tl['k'], k)
        ql = torch.as_tensor(tl['q'][jl, 1] + L.C.joint_offsets('left'), dtype=torch.float32)
        pl = L.SC.chain['left'].fk(ql)[0].double().numpy()
        grip = D['grip'][i]
        hs = slice(j_seat, j_seat + 10)
        F_loc = np.einsum('nj,nj->n', D['eff'][i] - D['eff'][i][hs].mean(0), rows['wrench'])
        ins = slice(0, j_seat)
        o = dict(run=name, date=date, label=LABEL.get(name, 'unlabelled'), push=None if m['push'] is None else m['push'].get('mode', 'position'),
                 force_n=None if m['push'] is None else m['push'].get('force_n'), skipped=m['skip'] is not None,
                 k4_insert=dict(F_tip_max_n=round(float(F_tip[ins].max()), 1), F_tip_p90_n=round(float(np.percentile(F_tip[ins], 90)), 1),
                                F_pd_max_n=round(float(F_pd[ins].max()), 1), seat_s=round((m['seat'] - m['ins']) * 0.02, 2),
                                F_tip_at_seat_n=round(float(F_tip[max(j_seat - 3, 0):j_seat + 1].mean()), 1)),
                 friction_band_n=round(float(np.median(band[hs])), 2))
        trace = dict(k=k, adv=adv, F_tip=F_tip, F_wr=F_wr, F_pd=F_pd, F_loc=F_loc, band=band, F_up=F_up, F_ac=F_ac, grip=grip)
        if m['push'] is not None:
            j0, jd = int(np.searchsorted(k, m['push0'])), int(np.searchsorted(k, m['drop']))
            jr = int(np.searchsorted(k, m['rel'])) if m['rel'] is not None else len(k) - 1
            a0 = adv[j0]
            pe = slice(max(jd - 10, j0), jd)                                             # last 0.2 s under load
            dr = slice(jr - 3, jr)                                                       # end of the drop / unload, jaws closed
            ad = adv - a0
            jpk = j0 + int(np.argmax(F_tip[j0:jd]))
            sl_up, _ = ramp_fit(ad[j0:jpk + 1], F_tip[j0:jpk + 1], 2.0, 0.95 * F_tip[jpk])
            sl_up_pd, _ = ramp_fit(ad[j0:jpk + 1], F_pd[j0:jpk + 1], 2.0, 0.95 * F_pd[j0:jd].max())
            Fpe, Fdr = float(F_tip[pe].mean()), float(F_tip[dr].mean())
            back = float(ad[pe].mean() - ad[dr].mean())
            tau_end = (KP * (D['tgt'][i] - q))[pe].mean(0)
            Jz = np.einsum('nij,i->nj', jac(q[pe.start:pe.start + 1])[:, :3], m['zin'])[0]  # tau per N of axial TCP force
            tmax = np.array([6., 6., 6., 2.5, 2., 2.])
            o['push_limits'] = dict(tau_pd_end_nm=tau_end.round(2).tolist(), tau_per_n=Jz.round(3).tolist(),
                                    F_at_tau_max_n=round(float(1. / np.max(np.abs(Jz) / tmax)), 1),
                                    joint_at_tau_max=int(np.argmax(np.abs(Jz) / tmax)) + 1,
                                    grip_change=round(float(np.abs(grip[j0:jd] - grip[j0]).max()), 4),
                                    left_mm=round(float(np.linalg.norm(pl[j0:jd] - pl[j0], axis=1).max() * 1e3), 2),
                                    F_up_end_n=round(float(F_up[pe].mean()), 1), F_across_end_n=round(float(F_ac[pe].mean()), 1))
            o['push_stats'] = dict(F_cmd_pd_end_n=round(float(F_pd[pe].mean()), 1), F_tip_end_n=round(Fpe, 1), F_wrench_end_n=round(float(F_wr[pe].mean()), 1),
                                   F_local_end_n=round(float(F_loc[pe].mean()), 1), F_tip_peak_n=round(float(F_tip[jpk]), 1),
                                   friction_band_end_n=round(float(band[pe].mean()), 2), adv_end_mm=round(float(ad[pe].mean()), 2),
                                   adv_rest_mm=round(float(ad[dr].mean()), 2), F_tip_rest_n=round(Fdr, 1),
                                   comp_load_mm_per_n=None if sl_up is None else round(sl_up, 3),
                                   comp_load_pd_mm_per_n=None if sl_up_pd is None else round(sl_up_pd, 3),
                                   comp_unload_mm_per_n=round(back / max(Fpe - Fdr, 1e-3), 3), push_s=round((jd - j0) * 0.02, 2))
            lo, _ = ramp_fit(ad[j0:jpk + 1], F_tip[j0:jpk + 1], 2.0, 0.5 * F_tip[jpk])
            hi, _ = ramp_fit(ad[j0:jpk + 1], F_tip[j0:jpk + 1], 0.5 * F_tip[jpk], 0.95 * F_tip[jpk])
            o['push_stats'].update(comp_low_half=None if lo is None else round(lo, 3), comp_high_half=None if hi is None else round(hi, 3),
                                   F_lat_end_n=round(float(np.hypot(F_up[pe].mean(), F_ac[pe].mean())), 1))
            trace.update(j0=j0, jd=jd, jr=jr)
        out.append(o)
        traces[name] = trace
        print(json.dumps(o), flush=True)
    (OUT / 'contact.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(OUT / 'contact_traces.npz', **{f'{n}__{k}': v for n, t in traces.items() for k, v in t.items()})


FEATS = [('k4_insert', 'F_tip_max_n'), ('k4_insert', 'F_tip_at_seat_n'), ('push_stats', 'F_tip_end_n'), ('push_limits', 'F_across_end_n'),
         ('push_limits', 'F_up_end_n'), ('push_stats', 'F_lat_end_n'), ('push_stats', 'comp_load_mm_per_n'), ('push_stats', 'comp_unload_mm_per_n'),
         ('push_stats', 'comp_low_half'), ('push_stats', 'comp_high_half'), ('push_stats', 'adv_end_mm'), ('push_stats', 'adv_rest_mm')]


def cmd_labels(A):
    C = json.loads((OUT / 'contact.json').read_text())
    res = []
    for blk, f in FEATS:
        v = {lab: [(c['run'], (c.get(blk) or {}).get(f)) for c in C if c['label'] == lab and (c.get(blk) or {}).get(f) is not None] for lab in ('full', 'short')}
        fu, sh = [x for _, x in v['full']], [x for _, x in v['short']]
        if not fu or not sh:
            continue
        inside = [n for n, x in v['short'] if min(fu) <= x <= max(fu)]
        res.append(dict(feature=f, full=v['full'], short=v['short'], full_range=[min(fu), max(fu)], short_range=[min(sh), max(sh)],
                        short_inside_full_range=inside, separates=len(inside) == 0))
        print(f'{f:22s} full {min(fu):6.2f}..{max(fu):6.2f} (n={len(fu)})  short {min(sh):6.2f}..{max(sh):6.2f} (n={len(sh)})  short inside: {inside}')
    (OUT / 'labels.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fit', 'contact', 'labels', 'all'])
    A = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if A.cmd in ('fit', 'all'):
        cmd_fit(A)
    if A.cmd in ('contact', 'all'):
        cmd_contact(A)
    if A.cmd in ('labels', 'all'):
        cmd_labels(A)
