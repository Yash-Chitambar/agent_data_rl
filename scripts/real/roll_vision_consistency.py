"""How well does the align stage know the plug's roll? (docs/real/K2_PROGRESS.md, "Real-to-sim replay and roll")

The stage's plug angles (align.py PortVisionSource.measure) = panel pose fitted in the wrist image (R_panel_cam) combined
with a FIXED plug-in-camera rotation (mount + the mean real grasp). So rv_deg sees the hand's roll against the panel but
never a grasp that is rolled in the jaws.

Test of the vision part: the panel is fixed on the table, so the panel orientation in the right arm base,
R_panel_base = R_cam(FK of the logged believed joints q_b) @ R_panel_cam, should not change while the hand moves and turns.
Its spread across the cycles of one run = vision rotation error (+ relative FK error, small over a few cm).
Per run: frame-to-frame spread within a cycle (5 frames, still hand), cycle-to-cycle spread of R_panel_base (moving hand),
and the logged rz; across runs of one session: spread of the run means (absolute: includes encoder offsets and PC moves).

  python scripts/real/roll_vision_consistency.py --out results/real_align/roll/roll_consistency.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real2sim'))
DATA = pathlib.Path('/home/yashchitambar/peg_test/data/real_runs')


def parse_frames(e):
    fr = e.get('frames')
    if isinstance(fr, str):
        import ast
        fr = ast.literal_eval(fr)
    return fr or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT / 'results/real_align/roll/roll_consistency.json'))
    a = ap.parse_args()
    import inhand_measure as IM
    import torch
    from blind_insertion.envs import inhand_real as IR
    hand = IM.Hand()
    d = np.mean(np.asarray(IR.load('v2')['inhand']['rows'], float), 0)
    Rrs = IR.reader_in_site(torch.tensor(d[None], dtype=torch.float32), 'cpu')[0][0].numpy().astype(float)
    runs = sorted(glob.glob(str(DATA / '2026-11-0*' / '*')))
    out = []
    for r in runs:
        r = pathlib.Path(r)
        if 'dry' in r.name or not (r / 'events.jsonl').exists():
            continue
        ev = [json.loads(l) for l in open(r / 'events.jsonl')]
        ms = [e for e in ev if e['kind'] == 'align_measure' and e.get('q_b') is not None]
        cyc = []
        for e in ms:
            fr = [f for f in parse_frames(e) if f.get('ok') and f.get('panel_rv_deg') is not None]
            if len(fr) < 3:
                continue
            P = np.array([f['panel_rv_deg'] for f in fr], float)
            Rp = [Rot.from_rotvec(np.radians(p)).as_matrix() for p in P]
            # frame spread: rotvec of each frame against the first, in the panel frame (deg)
            fs = np.array([np.degrees(Rot.from_matrix(Rp[0].T @ x).as_rotvec()) for x in Rp])
            Rm = Rot.from_matrix(np.stack(Rp)).mean().as_matrix()
            hand.pose(np.asarray(e['q_b'], float))
            Rs, _ = hand.site()
            Rc, _ = hand.cam()
            Rpb = Rc @ Rm                                   # panel in the right base
            Rplug = Rs @ Rrs
            rv_chk = np.degrees(Rot.from_matrix(Rpb.T @ Rplug).as_rotvec())
            cyc.append(dict(cycle=e.get('cycle'), Rpb=Rpb, frame_spread=(fs.max(0) - fs.min(0)).tolist(), rv_log=e.get('rv_deg'), rv_chk=rv_chk.tolist(),
                            rot_applied=e.get('rot_applied'), q_b=e['q_b']))
        if len(cyc) < 3:
            continue
        Rmean = Rot.from_matrix(np.stack([c['Rpb'] for c in cyc])).mean().as_matrix()
        dev = np.array([np.degrees(Rot.from_matrix(Rmean.T @ c['Rpb']).as_rotvec()) for c in cyc])     # panel frame, deg
        q = np.array([c['q_b'] for c in cyc])
        rvl = np.array([c['rv_log'] for c in cyc if c['rv_log'] is not None], float)
        rvc = np.array([c['rv_chk'] for c in cyc], float)
        row = dict(run=r.name, day=r.parent.name, n_cycles=len(cyc),
                   frame_spread_deg_median=np.median([c['frame_spread'] for c in cyc], 0).round(3).tolist(),
                   panel_base_sd_deg=dev.std(0).round(3).tolist(), panel_base_range_deg=(dev.max(0) - dev.min(0)).round(3).tolist(),
                   hand_rot_range_deg=None, rz_log_mean=float(rvl[:, 2].mean()) if len(rvl) else None,
                   rz_log_sd=float(rvl[:, 2].std()) if len(rvl) else None,
                   rv_check_max_err_deg=float(np.abs(rvl - rvc[:len(rvl)]).max()) if len(rvl) == len(rvc) else None,
                   R_panel_base_mean=Rmean.tolist())
        # how much the hand itself turned over the run (so a constant panel means the vision followed it)
        Rss = []
        for qq in q:
            hand.pose(qq)
            Rss.append(hand.site()[0])
        hr = np.array([np.degrees(Rot.from_matrix(Rss[0].T @ x).as_rotvec()) for x in Rss])
        row['hand_rot_range_deg'] = (hr.max(0) - hr.min(0)).round(3).tolist()
        out.append(row)
        print(f"{row['day']} {row['run']:30s} n {row['n_cycles']:2d} frame spread {row['frame_spread_deg_median']} "
              f"panel-in-base sd {row['panel_base_sd_deg']} range {row['panel_base_range_deg']} hand turned {row['hand_rot_range_deg']} "
              f"rz {row['rz_log_mean']:.2f}±{row['rz_log_sd']:.2f} chk {row['rv_check_max_err_deg']}")
    # across runs of a session: panel in base of each run against the session mean
    sess = {}
    for row in out:
        sess.setdefault(row['day'], []).append(row)
    summ = {}
    for day, rows in sess.items():
        if len(rows) < 2:
            continue
        Rs_ = np.stack([np.array(r['R_panel_base_mean']) for r in rows])
        M = Rot.from_matrix(Rs_).mean().as_matrix()
        dv = np.array([np.degrees(Rot.from_matrix(M.T @ x).as_rotvec()) for x in Rs_])
        summ[day] = dict(n_runs=len(rows), runs=[r['run'] for r in rows], dev_deg=dv.round(3).tolist(), sd_deg=dv.std(0).round(3).tolist())
        print(day, 'across runs: panel-in-base sd (deg, panel axes x, y, z)', dv.std(0).round(3), 'n', len(rows))
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(dict(runs=out, sessions=summ), indent=1))


if __name__ == '__main__':
    main()
