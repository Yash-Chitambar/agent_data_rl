"""Clearance replay for a parallel two-arm schedule (docs/real/PARALLEL_SCHEDULE_20261115.md).

Logged real joint paths of both arms (moves_<side>.npz for the slow moves, ticks.npz for the pick and the hold) are
replayed on the bimanual scene model (deploy/bimanual/scene.SceneModel through scripts/real/arm_clearance.TwoArm:
measured mounts, links, grippers, wrist cameras, fingers) with the second arm's start shifted in time. At every
50 Hz sample: the smallest distance between any right-arm geom and any left-arm geom, and of each arm to the PC
(placed at the run's ZED estimate) and to the reader on the table (at the run's ZED reader estimate).

Overlaps:
  P1 capture: right start -> ZED pose, left start -> ZED pose. Offset = left start - right start (s).
  P2 home:    left -> start pose, right -> start pose. Offset = right start - left start (s).
  P3 swing:   the left park -> stand-off swing (first part of stage 2 hold, until the guarded descent) during the
              right arm's pick (stage 1). Offset = swing start - pick start (s).
Both the measured and the commanded (target) path are checked; the table reports the smaller distance.

P3 view check: is any left-arm geom (sphere bound) inside the right wrist camera's view cone during the pick
(the servo nets see the right wrist frame)?

    PYTHONPATH=. python scripts/real/parallel_clearance.py --out results/real_align/parallel
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / 'scripts/real'))
RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')
DATES = ('2026-11-10', '2026-11-11', '2026-11-13')
STAGES = ('0 capture', '1 pick', '2 hold', '3 transport', '4 approach', '5 insert', '6 release', '7 home', 'manual', 'align')
DT = 0.02


# ------------------------------------------------------------------ geometry
class Geo:
    def __init__(self):
        import mujoco
        from arm_clearance import TwoArm, deploy_package
        import importlib
        self.mj = mujoco
        self.ta = TwoArm()
        m = self.m = self.ta.m
        self.d = self.ta.d
        with deploy_package():
            self.S = importlib.import_module('blind_insertion.scenes.pc_bench')
            self.lay = importlib.import_module('deploy.bimanual.scene').SceneModel('cpu').lay
        body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])  # noqa: E731
        self.gPC = [g for g in range(m.ngeom) if body(g) == 'fixture']
        self.gRD = [g for g in range(m.ngeom) if body(g) == 'peg']
        self.g = {'right': self.ta.gR, 'left': self.ta.gL}
        self._ft = np.zeros(6)
        # yaw conventions: ZED nets report atan2(-R[1,2], -R[0,2]) of the PC body / the reader; place_* take a yaw
        # about +z from their own zero
        self.yaw0_pc = self._yaw_of('fixture', lambda: self.S.place_pc(m, self.d, port_xy=(0.2, 0.4), yaw_deg=0.))
        self.yaw0_rd = self._yaw_of('peg', lambda: self.S.place_reader(m, self.d, self.lay, xy=(0.1, 0.45), yaw_deg=0.), reader=True)
        self.cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_cam')

    def _yaw_of(self, bname, place, reader=False):
        place()
        self.mj.mj_kinematics(self.m, self.d)
        R = self.d.xmat[self.m.body(bname).id].reshape(3, 3)
        return math.degrees(math.atan2(-R[1, 2], -R[0, 2]))

    def place(self, pc=None, reader=None):
        m, d = self.m, self.d
        if pc is not None:
            self.S.place_pc(m, d, port_xy=pc[0], yaw_deg=pc[1] - self.yaw0_pc)
        else:
            self.S.place_pc(m, d, port_xy=(5., 5.), yaw_deg=0.)
        if reader is not None:
            self.S.place_reader(m, d, self.lay, xy=reader[0], yaw_deg=reader[1] - self.yaw0_rd)
        else:
            self.S.place_reader(m, d, self.lay, xy=(-5., 5.), yaw_deg=0.)

    def pose(self, qr, ql):
        """motor-frame joints of both arms; TwoArm adds the rig offsets."""
        d = self.d
        d.qpos[self.ta.qa['right']] = np.asarray(qr, float)[:6] + self.ta.off['right']
        d.qpos[self.ta.qa['left']] = np.asarray(ql, float)[:6] + self.ta.off['left']
        self.mj.mj_kinematics(self.m, d)

    def dmin(self, gA, gB, cap=0.3):
        """smallest geom distance between two sets at the current pose; sphere bounds prune the exact calls."""
        m, d = self.m, self.d
        ca, cb = d.geom_xpos[gA], d.geom_xpos[gB]
        ra, rb = m.geom_rbound[gA], m.geom_rbound[gB]
        lb = np.linalg.norm(ca[:, None] - cb[None], axis=2) - ra[:, None] - rb[None]
        order = np.argsort(lb, axis=None)
        best, pair = cap, None
        for f in order:
            i, j = divmod(int(f), len(gB))
            if lb[i, j] >= best:
                break
            x = self.mj.mj_geomDistance(m, d, gA[i], gB[j], best, self._ft)
            if x < best:
                best, pair = x, (gA[i], gB[j])
        return best, pair

    def bname(self, g):
        return self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_BODY, self.m.geom_bodyid[g])

    def in_right_view(self, half_deg=None):
        """left-arm geoms (sphere bound) inside the right wrist camera's view cone at the current pose: list of
        (body, angle off the axis deg, range m). Cone half angle = the camera's half diagonal FOV."""
        m, d = self.m, self.d
        fovy = float(m.cam_fovy[self.cam])
        if half_deg is None:      # square policy crop: half diagonal of a fovy x fovy image
            half_deg = math.degrees(math.atan(math.sqrt(2) * math.tan(math.radians(fovy / 2))))
        p, R = d.cam_xpos[self.cam], d.cam_xmat[self.cam].reshape(3, 3)
        out = []
        for g in self.g['left']:
            v = d.geom_xpos[g] - p
            z = -float(R[:, 2] @ v)                      # MuJoCo cameras look along -z
            r = float(np.linalg.norm(v))
            if z <= 0:
                continue
            ang = math.degrees(math.acos(min(1., z / r)))
            ang_edge = ang - math.degrees(math.asin(min(1., m.geom_rbound[g] / max(r, 1e-6))))
            if ang_edge < half_deg:
                out.append((self.bname(g), ang_edge, r))
        return out, half_deg


# ------------------------------------------------------------------ logs
def load_run(d):
    ev = [json.loads(x) for x in open(d / 'events.jsonl')]
    mv = {}
    for s in ('left', 'right'):
        f = d / f'moves_{s}.npz'
        if f.exists():
            z = np.load(f)
            mv[s] = dict(t=z['t'], meas=z['measured'][:, :6], tgt=z['target'][:, :6])
    tk = None
    if (d / 'ticks.npz').exists():
        z = np.load(d / 'ticks.npz')
        tk = dict(t=z['t'], q=z['q'], tgt=z['target_motor'], stage=z['stage'].astype(int))
    return ev, mv, tk


def move_segments(ev, mv):
    """[(stage, side, t0, t1, meas, tgt)] for each move_to event (until the next move_to/moved event)."""
    stage, out = 'manual', []
    for i, e in enumerate(ev):
        if e['kind'] == 'stage':
            stage = e['stage']
        if e['kind'] != 'move_to' or e['side'] not in mv:
            continue
        t1 = next((x['t'] for x in ev[i + 1:] if x['kind'] in ('move_to', 'moved', 'stopped')), None)
        if t1 is None:
            continue
        m = mv[e['side']]
        k = (m['t'] >= e['t']) & (m['t'] <= t1 + 1e-3)
        if k.sum() < 5:
            continue
        out.append(dict(stage=stage, side=e['side'], t0=float(m['t'][k][0]), t1=float(m['t'][k][-1]),
                        meas=m['meas'][k], tgt=m['tgt'][k], tt=m['t'][k] - m['t'][k][0]))
    return out


def resample(seg, T):
    """joint path on the 50 Hz grid of length T (s), endpoints held."""
    n = int(round(T / DT)) + 1
    t = np.arange(n) * DT
    out = {}
    for key in ('meas', 'tgt'):
        x = seg[key]
        out[key] = np.stack([np.interp(t, seg['tt'], x[:, j]) for j in range(6)], 1)
    return out


def at(path, k):
    return path[min(max(k, 0), len(path) - 1)]


def replay(G, A, B, offsets, sides=('right', 'left'), pc=None, reader=None, B_before=None, view=False, stride=1):
    """A starts at 0, B at each offset (s). A, B: segment dicts (A's side first in `sides`). Returns rows."""
    G.place(pc, reader)
    TA, TB = A['tt'][-1], B['tt'][-1]
    pA, pB = resample(A, TA), resample(B, TB)
    rows = []
    for off in offsets:
        T = max(TA, off + TB)
        n = int(round(T / DT)) + 1
        kB0 = int(round(off / DT))
        best = dict(arms=(1.0, None, None), pc={'right': 1.0, 'left': 1.0}, reader={'right': 1.0, 'left': 1.0})
        n_view, view_bodies, view_min = 0, set(), 99.
        for k in range(0, n, stride):
            for key in ('meas', 'tgt'):
                qa = at(pA[key], k)
                qb = at(pB[key], k - kB0) if (k >= kB0 or B_before is None) else B_before
                qr, ql = (qa, qb) if sides[0] == 'right' else (qb, qa)
                G.pose(qr, ql)
                x, pair = G.dmin(G.g['right'], G.g['left'])
                if x < best['arms'][0]:
                    best['arms'] = (x, None if pair is None else (G.bname(pair[0]), G.bname(pair[1])), k * DT)
                for s in ('right', 'left'):
                    if pc is not None:
                        best['pc'][s] = min(best['pc'][s], G.dmin(G.g[s], G.gPC)[0])
                    if reader is not None:
                        best['reader'][s] = min(best['reader'][s], G.dmin(G.g[s], G.gRD)[0])
                if view and key == 'meas' and kB0 <= k <= len(pA[key]) - 1:
                    v, half = G.in_right_view()
                    if v:
                        n_view += 1
                        view_bodies |= {b for b, _, _ in v}
                        view_min = min(view_min, min(r for _, _, r in v))
        rows.append(dict(offset_s=round(off, 3), makespan_s=round(T, 3), TA=round(TA, 3), TB=round(TB, 3),
                         arms_mm=round(best['arms'][0] * 1e3, 1), pair=best['arms'][1], t_min_s=round(best['arms'][2] or 0, 2),
                         pc_right_mm=round(best['pc']['right'] * 1e3, 1) if pc is not None else None,
                         pc_left_mm=round(best['pc']['left'] * 1e3, 1) if pc is not None else None,
                         reader_right_mm=round(best['reader']['right'] * 1e3, 1) if reader is not None else None,
                         reader_left_mm=round(best['reader']['left'] * 1e3, 1) if reader is not None else None,
                         **(dict(view_ticks=n_view, view_bodies=sorted(view_bodies), view_range_m=round(view_min, 3)) if view else {})))
    return rows


def tick_segment(tk, mask):
    idx = np.where(mask)[0]
    if len(idx) < 5:
        return None
    t = tk['t'][idx]
    return dict(t0=float(t[0]), t1=float(t[-1]), tt=t - t[0], idx=idx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/real_align/parallel')
    ap.add_argument('--runs', nargs='*', default=None, help='run folders (default: every real full_* run of DATES)')
    ap.add_argument('--stride', type=int, default=1, help='evaluate every n-th 50 Hz sample')
    ap.add_argument('--merge', nargs='*', default=None, help='merge the per-run output folders into --out (no replay)')
    a = ap.parse_args(argv)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.merge is not None:
        rows, timing = [], []
        for m in a.merge:
            rows += json.loads((Path(m) / 'clearance_rows.json').read_text())
            timing += json.loads((Path(m) / 'timeline.json').read_text())
        return write(out, rows, timing)
    runs = [Path(r) for r in a.runs] if a.runs else sorted(p for dt in DATES for p in (RUNS / dt).glob('full_*') if 'dry' not in p.name)
    G = Geo()
    rows, timing = [], []
    for d in runs:
        if not (d / 'events.jsonl').exists():
            continue
        ev, mv, tk = load_run(d)
        segs = move_segments(ev, mv)
        est = {e['est']: e for e in ev if e['kind'] == 'estimate' and e.get('est') in ('zed_pc', 'zed_reader')}
        pc = (est['zed_pc']['port_xy'], est['zed_pc']['yaw_deg']) if 'zed_pc' in est else None
        rd = (est['zed_reader']['p'][:2], est['zed_reader']['yaw_deg']) if 'zed_reader' in est else None
        name = f'{d.parent.name}/{d.name}'
        cap = [s for s in segs if s['stage'] == '0 capture']
        home = [s for s in segs if s['stage'] == '7 home']
        rec = dict(run=name, segs=[(s['stage'], s['side'], round(s['t1'] - s['t0'], 2)) for s in segs])
        # P1: first right and first left capture move (the second left move is the park)
        cr = next((s for s in cap if s['side'] == 'right'), None)
        cl = next((s for s in cap if s['side'] == 'left'), None)
        if cr is not None and cl is not None and cl['t0'] > cr['t0']:
            ser = round(cl['t0'] - cr['t0'], 2)
            offs = sorted({0.0, 1.0, 2.0, 4.0, -1.0, -2.0, ser})
            for r in replay(G, cr, cl, offs, ('right', 'left'), pc=None, reader=rd, stride=a.stride):
                rows.append(dict(run=name, overlap='P1 capture', serial=r['offset_s'] == ser, **r))
            rec['capture_serial_offset_s'] = ser
        # P2: home, left first today
        hl = next((s for s in home if s['side'] == 'left'), None)
        hr = next((s for s in home if s['side'] == 'right'), None)
        if hl is not None and hr is not None and hr['t0'] > hl['t0']:
            ser = round(hr['t0'] - hl['t0'], 2)
            offs = sorted({0.0, 1.0, 2.0, 4.0, -1.0, ser})
            for r in replay(G, hl, hr, offs, ('left', 'right'), pc=pc, stride=a.stride):
                rows.append(dict(run=name, overlap='P2 home', serial=r['offset_s'] == ser, **r))
            rec['home_serial_offset_s'] = ser
        # P3: left swing (stage 2 hold, until the left joint speed drops after the swing) during the pick
        if tk is not None:
            pick = tick_segment(tk, tk['stage'] == STAGES.index('1 pick'))
            hold = tick_segment(tk, tk['stage'] == STAGES.index('2 hold'))
            if pick is not None and hold is not None:
                hi = hold['idx']
                # swing = the first HoldCfg.move_s / --speed seconds of the hold stage (chain.stage_hold), then the
                # guarded descent at 1 cm/s starts (not part of P3: it ends in the press)
                meta = json.loads((d / 'meta.json').read_text())
                spd = float(meta.get('args', {}).get('speed', 1.0) or 1.0)
                n_sw = int(round(1.5 / spd / DT)) + 1
                sw_idx = hi[:n_sw]
                swing = dict(tt=tk['t'][sw_idx] - tk['t'][sw_idx[0]], meas=tk['q'][sw_idx, 1], tgt=tk['tgt'][sw_idx, 1])
                pk = dict(tt=pick['tt'], meas=tk['q'][pick['idx'], 0], tgt=tk['tgt'][pick['idx'], 0])
                Tp = float(pick['tt'][-1])
                park = tk['q'][pick['idx'][0], 1]
                offs = [0.0, max(Tp - 6., 0.), max(Tp - swing['tt'][-1], 0.)]
                for r in replay(G, pk, swing, offs, ('right', 'left'), pc=pc, reader=rd, B_before=park, view=True, stride=a.stride):
                    rows.append(dict(run=name, overlap='P3 swing in pick', serial=False, **r))
                # reference: today's sequence (left swings after the pick ends, right at its lift pose)
                ql = dict(tt=np.array([0., 0.02]), meas=np.stack([tk['q'][pick['idx'][-1], 0]] * 2), tgt=np.stack([tk['tgt'][pick['idx'][-1], 0]] * 2))
                for r in replay(G, ql, swing, [0.0], ('right', 'left'), pc=pc, reader=None, stride=a.stride):
                    rows.append(dict(run=name, overlap='P3 swing in pick', serial=True, **r))
                rec.update(pick_s=round(Tp, 2), swing_s=round(float(swing['tt'][-1]), 2),
                           hold_stage_s=round(float(hold['tt'][-1]), 2))
        # E-A3: press time: latch -> left unload (stage 6 release end), from ticks
        if tk is not None:
            hr_ = [e for e in ev if e['kind'] == 'hold_result']
            if hr_ and hr_[0].get('latched'):
                t_latch = hr_[0]['t']
                rel = tk['t'][tk['stage'] == STAGES.index('6 release')]
                if len(rel):
                    rec['press_s'] = round(float(rel[-1] - t_latch), 2)
                    rec['pick_to_latch_s'] = round(float(t_latch - tk['t'][tk['stage'] == STAGES.index('1 pick')][0]), 2) if (tk['stage'] == 1).any() else None
        # timeline: log open, capture prompt, last move end
        t_open = ev[0]['t']
        t_cap = next((e['t'] for e in ev if e['kind'] == 'stage' and e['stage'] == '0 capture'), None)
        t_end = max((s['t1'] for s in segs), default=None)
        rec.update(t_connect_s=round(next((e['t'] for e in ev if e['kind'] == 'arm_connected' and e.get('side') == 'left'), t_open) - t_open, 2),
                   capture_to_end_s=None if t_cap is None or t_end is None else round(t_end - t_cap, 2),
                   open_to_end_s=None if t_end is None else round(t_end - t_open, 2))
        timing.append(rec)
        print(name, json.dumps({k: v for k, v in rec.items() if k != 'segs'}), flush=True)
    write(out, rows, timing)


def write(out, rows, timing):
    (out / 'clearance_rows.json').write_text(json.dumps(rows, indent=1))
    (out / 'timeline.json').write_text(json.dumps(timing, indent=1))
    # summary per overlap and offset
    summ = {}
    for r in rows:
        key = (r['overlap'], 'serial' if r['serial'] else f"{r['offset_s']:+.1f}")
        summ.setdefault(key, []).append(r)
    lines = ['overlap,offset,n,arms_min_mm,arms_p10_mm,arms_median_mm,closest_pair,pc_min_mm,reader_min_mm,makespan_median_s,view_runs']
    for (ov, off), rs in sorted(summ.items()):
        x = np.array([r['arms_mm'] for r in rs])
        worst = min(rs, key=lambda r: r['arms_mm'])
        pcs = [v for r in rs for v in (r['pc_right_mm'], r['pc_left_mm']) if v is not None]
        rds = [v for r in rs for v in (r['reader_right_mm'], r['reader_left_mm']) if v is not None]
        nv = sum(1 for r in rs if r.get('view_ticks'))
        lines.append(f"{ov},{off},{len(rs)},{x.min():.1f},{np.percentile(x, 10):.1f},{np.median(x):.1f},{'/'.join(worst['pair'] or ['-'])},"
                     f"{min(pcs) if pcs else ''},{min(rds) if rds else ''},{np.median([r['makespan_s'] for r in rs]):.2f},"
                     f"{nv if 'view_ticks' in rs[0] else ''}")
    (out / 'clearance_summary.csv').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))




def planned(argv=None):
    """The plan check (deploy/bimanual/parallel.plan) on every logged capture and home: start = the first measured
    sample of each arm's move, goal = its move_to goal, today's speed (0.4 rad/s), a list of leads. Rows ->
    <out>/planned_rows.json. This is what the rig's plan check computes before the go."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/real_align/parallel')
    ap.add_argument('--vmax', type=float, default=0.4)
    ap.add_argument('--leads', default='0,1,2,3,4,5,6,7,8')
    ap.add_argument('--stride', type=int, default=1)
    a = ap.parse_args(argv)
    from deploy.bimanual import parallel as PS
    from deploy.bimanual.scene import SceneModel
    import rig
    sc = SceneModel('cpu')
    model = PS.ArmArmModel(sc)
    off = {s: np.asarray(rig.joint_offsets(s), float) for s in ('right', 'left')}
    leads = [float(x) for x in a.leads.split(',')]
    runs = sorted(p for dt in DATES for p in (RUNS / dt).glob('full_*') if 'dry' not in p.name)
    rows = []
    for d in runs:
        if not (d / 'events.jsonl').exists():
            continue
        ev, mv, _ = load_run(d)
        segs = move_segments(ev, mv)
        goals = {}
        for e in ev:          # move_to goals in order, per stage and side
            if e['kind'] == 'move_to':
                goals.setdefault((e.get('side'), round(e['t'], 3)), e['goal_motor'])
        for ov, stage, first in (('P1 capture', '0 capture', 'right'), ('P2 home', '7 home', 'left')):
            ss = [s for s in segs if s['stage'] == stage]
            sr = next((s for s in ss if s['side'] == 'right'), None)
            sl = next((s for s in ss if s['side'] == 'left'), None)
            if sr is None or sl is None:
                continue
            # start of the combined move = both arms' poses when the first arm starts: the second arm's pose then is its
            # own first sample (it has not moved yet)
            start = {s: seg['meas'][0] + off[s] for s, seg in (('right', sr), ('left', sl))}
            goal = {s: seg['tgt'][-1] + off[s] for s, seg in (('right', sr), ('left', sl))}
            for L in leads:
                r = PS.plan(model, start, goal, a.vmax, L, first, stride=a.stride)
                rows.append(dict(run=f'{d.parent.name}/{d.name}', overlap=ov, lead_s=L, **{k: r[k] for k in ('min_mm', 'pair', 't_min_s', 'serial_s', 'parallel_s', 'move_s')}))
            # serial reference on the same planned paths: the second arm starts when the first arm's path and settle end
            r = PS.plan(model, start, goal, a.vmax, 0., first, stride=a.stride)
            Ls = r['move_s'][first]
            r = PS.plan(model, start, goal, a.vmax, Ls, first, stride=a.stride)
            rows.append(dict(run=f'{d.parent.name}/{d.name}', overlap=ov, lead_s='serial', **{k: r[k] for k in ('min_mm', 'pair', 't_min_s', 'serial_s', 'parallel_s', 'move_s')}))
        print(d.name, flush=True)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'planned_rows.json').write_text(json.dumps(rows, indent=1))
    for ov in ('P1 capture', 'P2 home'):
        print(ov)
        ser = {r['run']: r['min_mm'] for r in rows if r['overlap'] == ov and r['lead_s'] == 'serial'}
        for L in leads + ['serial']:
            q = [r for r in rows if r['overlap'] == ov and r['lead_s'] == L]
            x = np.array([r['min_mm'] for r in q])
            ge = sum(r['min_mm'] >= min(50., ser[r['run']]) - 0.5 for r in q)
            print(f"  lead {L}: n {len(q)} min {x.min():.1f} p10 {np.percentile(x, 10):.1f} median {np.median(x):.1f} "
                  f">=50: {(x >= 50).sum()}  >=min(50, serial): {ge}  saved median {np.median([r['serial_s'] - r['parallel_s'] for r in q]):.2f} s")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'planned':
        planned(sys.argv[2:])
    else:
        main()
