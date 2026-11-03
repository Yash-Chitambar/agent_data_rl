"""Fit and test the classical port vision (deploy/bimanual/port_vision.py) on the real wrist data set. No motion.

    python scripts/real/port_vision_fit.py features        # blobs and plug edges per frame -> cache
    python scripts/real/port_vision_fit.py model           # triangulate the panel model on the TUNE blocks only

Split (never random frames): collect sweep cut into 6 contiguous blocks of 25 samples in visiting order; blocks 0, 2, 4 tune,
1, 3, 5 report. same_spot and self are report only (other runs).
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.bimanual import port_vision as PV   # noqa: E402

MAN = ROOT / 'data/real_training/port_v1/manifest.jsonl'
OUT = ROOT / 'results/real/port_vision'
TUNE_BLOCKS = (0, 2, 4)


def rows():
    out = [json.loads(x) for x in open(MAN)]
    i = 0
    for r in out:
        r['block'] = None
        if r['source'] == 'collect':
            r['block'], i = i // 25, i + 1
        r['split'] = 'tune' if r['block'] in TUNE_BLOCKS else 'test'
    return out


class Fk:
    """Camera pose in the right base from model-frame joints: grasp site FK times the fixed site -> camera transform."""

    def __init__(self):
        import mujoco
        import torch
        from deploy.bimanual.scene import SceneModel
        self.torch = torch
        self.sc = SceneModel('cpu')
        m = self.sc.mjm
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        site, link = m.site('grasp_site').id, m.body('link_6').id
        Rs, ps = d.site_xmat[site].reshape(3, 3).copy(), d.site_xpos[site].copy()
        Rl, pl = d.xmat[link].reshape(3, 3).copy(), d.xpos[link].copy()
        pose = json.loads(PV.SPEC.read_text())['wrist']['right']['pose_in_link6']
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, np.array(pose['quat'], float))
        Rc = Rc.reshape(3, 3)
        self.R_sc = Rs.T @ Rl @ Rc
        self.p_sc = Rs.T @ (Rl @ np.array(pose['pos']) + pl - ps)
        self.ch = self.sc.chain['right']

    def tcp(self, q):
        p, R = self.ch.fk(self.torch.as_tensor(np.asarray(q, float), dtype=self.torch.float32).reshape(1, 6))
        return np.asarray(p).reshape(3).astype(float), np.asarray(R).reshape(3, 3).astype(float)

    def cam(self, q):
        """(p, R): camera (OpenCV) in the base."""
        p, R = self.tcp(q)
        return p + R @ self.p_sc, R @ self.R_sc


def feats(r, panel):
    und, K = PV.undistort(cv2.imread(r['raw']))
    m = PV.load_model()
    pl, _ = PV.detect_plug(und)
    box = PV.plug_box(pl) if pl else None
    blobs = PV.bright_blobs(und, box)
    mt, _ = PV.match_row(blobs, panel)
    if not mt:
        return dict(K=K, row=None, lines={})
    row = {k: blobs[i][:2] + blobs[i][2:4] for k, i in mt.items()}
    R0 = np.array(m['R_view'], float)
    lp = PV.detect_lines(und, K, R0, PV.t_given_R(row, K, R0, panel), m['lines'], box)
    return dict(K=K, row=row, lines=lp)


def anchor_R():
    ev = [json.loads(x) for x in open(ROOT / 'data/real_runs/2026-11-02/collect_182100_collect1/events.jsonl')]
    a = next(e for e in ev if e['kind'] == 'collect_anchor')
    return a, np.array(a['port_R'], float)


def cmd_model(a):
    fk = Fk()
    panel = PV.load_model()['panel']
    _, eR = anchor_R()
    for it in range(a.iters):
        obs = {k: [] for k in panel}
        lobs = {k: [] for k in PV.load_model().get('lines', {})}
        nf, Rs = 0, []
        for r in rows():
            if r['source'] != 'collect' or r['split'] != 'tune':
                continue
            d = feats(r, panel)
            if d['row'] is None:
                continue
            nf += 1
            pc, Rc = fk.cam(r['q_b'])
            Rs.append(Rc.T @ eR)
            K = d['K']
            for ln, ps in d['lines'].items():
                for *_, u, v in ps:
                    ray = Rc @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
                    lobs[ln].append((pc, ray))
            for k, (u, v, w, h) in d['row'].items():
                ray = Rc @ np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
                obs[k].append((pc, ray / np.linalg.norm(ray), w, h))
        X = {}
        for k, o in obs.items():
            if len(o) < 5:
                print('too few views of', k, len(o))
                X[k] = None
                continue
            P = np.array([x[0] for x in o])
            D = np.array([x[1] for x in o])
            keep = np.ones(len(o), bool)
            for _ in range(3):
                A = np.zeros((3, 3))
                b = np.zeros(3)
                for p, u_ in zip(P[keep], D[keep]):
                    M = np.eye(3) - np.outer(u_, u_)
                    A += M
                    b += M @ p
                x = np.linalg.solve(A, b)
                res = np.linalg.norm(np.cross(x - P, D), axis=1)
                keep = res < max(0.8e-3, 2.5 * np.median(res))
            X[k] = (x, float(np.median(res[keep]) * 1e3), int(keep.sum()), float(np.median([q[2] for q in o])), float(np.median([q[3] for q in o])),
                    float(np.median(np.linalg.norm(x - P, axis=1))))
        # panel frame in eR axes, origin at P2 (or from P1 if P2 is missing)
        o = X['P2'][0]
        new = {}
        for k, v in X.items():
            if v is None:
                new[k] = panel[k]
                continue
            xyz = eR.T @ (v[0] - o) * 1e3
            f = 683.0 / (v[5] * 1e3)
            new[k] = dict(xyz=xyz.round(3).tolist(), wh=[round(v[3] / f, 2), round(v[4] / f, 2)], ray_res_mm=round(v[1], 3), n=v[2])
        newl = {}
        for ln, o_ in lobs.items():
            if len(o_) < 10:
                print('too few line samples', ln, len(o_))
                newl[ln] = PV.load_model()['lines'][ln]
                continue
            P = np.array([eR.T @ (p - o) * 1e3 for p, _ in o_])
            D = np.array([eR.T @ r_ for _, r_ in o_])
            isH = ln in ('band', 'bottom')
            ia, ib = (1, 2) if isH else (0, 2)     # plane of the fit: (y, z) or (x, z); running coordinate x or y
            irun = 0 if isH else 1
            lam = (0 - P[:, 2]) / D[:, 2]           # running coordinate from the ray at the panel plane
            c_ = P[:, irun] + lam * D[:, irun]
            nv = np.c_[-D[:, ib], D[:, ia]]
            nv /= np.linalg.norm(nv, axis=1, keepdims=True)
            b = (nv * P[:, [ia, ib]]).sum(1)
            A = np.c_[nv, nv[:, 0] * c_, nv[:, 1] * c_]
            keep = np.ones(len(nv), bool)
            for _ in range(4):
                sol = np.linalg.lstsq(A[keep], b[keep], rcond=None)[0]
                res = A @ sol - b
                keep = np.abs(res) < max(0.5, 2.5 * np.median(np.abs(res)))
            newl[ln] = {('y' if isH else 'x'): round(float(sol[0]), 3), 'z': round(float(sol[1]), 3), 'a': round(float(sol[2]), 5), 'b': round(float(sol[3]), 5),
                        'res_mm': round(float(np.median(np.abs(res[keep]))), 3), 'n': int(keep.sum())}
            print(ln, newl[ln])
        PV._MODEL['lines'] = newl
        Rs = np.array(Rs)
        U, _, Vt = np.linalg.svd(Rs.mean(0))
        PV._MODEL.update(panel=new, R_view=(U @ Vt).round(5).tolist())
        print(f'iter {it}, {nf} frames; R_view', PV._MODEL['R_view'])
        for k, v in new.items():
            print(k, v)
        panel = new
    base = eR @ np.zeros(3) + o
    out = dict(panel=panel, lines=PV._MODEL['lines'], R_view=PV._MODEL['R_view'], _about='Panel features (mm) in the ZED port axes of collect_182100_collect1, origin port 2 centre; triangulated from FK on the tune blocks.',
               port2_base_m=base.tolist(), port_R=eR.tolist(), tune_blocks=list(TUNE_BLOCKS))
    PV.MODEL.write_text(json.dumps(out, indent=1))
    print('wrote', PV.MODEL)


def cmd_probe(a):
    fk = Fk()
    _, eR = anchor_R()
    out = []
    for r in rows():
        if r['source'] != 'collect':
            continue
        m = PV.measure(cv2.imread(r['raw']))
        p = r['p_mm']
        rec = [r['block'], r['split'] == 'tune', *p, m.ok]
        if m.ok:
            pc, Rc = fk.cam(r['q_b'])
            R_fk = Rc.T @ eR
            dR, _ = cv2.Rodrigues(m.debug['R'] @ R_fk.T)
            rec += [m.across_mm, m.up_mm, m.gap_mm, *np.degrees(dR.ravel()), m.debug['rms'], m.debug['Z_w'], len(m.debug['used'])]
        out.append(rec)
    n = max(map(len, out))
    A = np.array([x + [np.nan] * (n - len(x)) for x in out], float)
    np.save(OUT / 'probe_collect.npy', A)
    for name, sel in (('tune', A[:, 1] > 0), ('test', A[:, 1] == 0)):
        B = A[sel & (A[:, 5] > 0)]
        print(name, 'n', sel.sum(), 'ok', len(B))
        for i, ax in enumerate('xyz'):
            sl, b = np.polyfit(B[:, 2 + i], B[:, 6 + i], 1)
            res = B[:, 6 + i] - sl * B[:, 2 + i] - b
            print(ax, 'slope', round(sl, 3), 'offset', round(b, 2), 'res SD', round(res.std(), 2), 'mean diff', round((B[:, 6 + i] - B[:, 2 + i]).mean(), 2), 'SD diff', round((B[:, 6 + i] - B[:, 2 + i]).std(), 2))
        print('PnP-FK rot deg mean', B[:, 9:12].mean(0).round(2), 'SD', B[:, 9:12].std(0).round(2), 'rms px', np.median(B[:, 12]).round(2), 'Z', np.median(B[:, 13]).round(2), B[:, 13].std().round(2), 'features', B[:, 14].mean().round(2))


def negatives(n=240):
    """Wrist frames without a plug in the hand: stream frames of the pick runs before the jaws close (stage 1 or earlier)."""
    out = []
    for run in sorted((ROOT / 'data/real_runs/2026-11-02').glob('pick_*')):
        fr = [json.loads(x) for x in open(run / 'frames.jsonl')]
        w = [x for x in fr if x['cam'] == 'wrist_right' and x['kind'] == 'raw' and x['file'].endswith('.jpg')]
        w = w[:len(w) // 2]    # before the reader is lifted: checked by eye on the contact sheet
        out += [dict(run=run.name, raw=str(run / x['file'])) for x in w[::2]]
    return out[:n]


def cmd_eval(a):
    import csv
    rs = [r for r in rows() if r.get('raw')] + [dict(r, source='negative', split='test', block=None, p_mm=None, moving=None) for r in negatives()]
    fields = ['run', 'source', 'split', 'block', 'moving', 'raw', 'lx', 'ly', 'lz', 'ok', 'reason', 'x', 'y', 'z', 'Zw', 'w', 've', 'u', 'emax', 'rms', 'nslots', 'nlines', 'r0', 'r1', 'r2', 't0', 't1', 't2', 'ms']
    with open(OUT / 'eval_frames.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fields)
        wr.writeheader()
        import time
        for r in rs:
            im = cv2.imread(r['raw'])
            t = time.perf_counter()
            m = PV.measure(im, max_reproj_px=a.gate)
            ms = (time.perf_counter() - t) * 1e3
            p = r.get('p_mm') or [None] * 3
            row = dict(run=r['run'], source=r['source'], split=r['split'], block=r['block'], moving=r.get('moving'), raw=r['raw'], lx=p[0], ly=p[1], lz=p[2], ok=int(m.ok), reason=m.reason, ms=round(ms, 1))
            d = m.debug
            if d.get('plug'):
                row.update(w=d['plug']['w'], ve=d['plug']['ve'], u=(d['plug']['ul'] + d['plug']['ur']) / 2, Zw=d.get('Z_w'))
            if m.ok:
                rv = cv2.Rodrigues(d['R'])[0].ravel()
                row.update(x=m.across_mm, y=m.up_mm, z=m.gap_mm, emax=d['emax'], rms=d['rms'], nslots=len(d['used']), nlines=len(d['lines']), r0=rv[0], r1=rv[1], r2=rv[2], t0=d['t'][0], t1=d['t'][1], t2=d['t'][2])
            wr.writerow(row)
    print('wrote', OUT / 'eval_frames.csv', len(rs))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['model','probe','eval'])
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--gate', type=float, default=PV.MAX_REPROJ_PX)
    a = ap.parse_args()
    {'model': cmd_model,'probe': cmd_probe,'eval': cmd_eval}[a.cmd](a)
