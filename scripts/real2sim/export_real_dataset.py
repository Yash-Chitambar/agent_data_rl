"""Real frames + encoder-FK labels -> datasets to check and fine-tune the perception nets on real images.

Per run directory (deploy/run_bimanual.py logs; the policy images are the exact net inputs), labels wherever the
ground truth follows from the encoders and the task geometry:

  zed_pc       ZED frames at the capture pose; label = the operator-measured PC spot (--pc-spot / meta pc_spot:
               port xy, yaw) if given; else no label (estimate spread only).
  pick_servo   right-wrist frames at the hover and the descent waypoints; label = the grasp the hand actually
               closed on, relative to the hand at the frame: T_tcp(t)^-1 T_tcp(close) (m, grasp_site frame;
               rotation vector rad), the train_reader_servo.py target. Valid when the pick held the reader.
  zed_reader   ZED reader frames; label = reader body centre / yaw implied by the closed grasp and the nominal
               Part 1 in-hand pose (approximate: the in-hand pose spread, ~1-3 mm).
  approach     right-wrist approach frames; label (only if the plug seated: detector or operator label) = plug
               tip in the port frame back-propagated from the seated pose by encoder FK:
               plug(t) = T_port^-1 T_tcp(t) T_tcp(seat)^-1 T_port S, S = seated (0, 0, -depth); the port rotation
               from the ZED estimate. The wrist nets' target (x, y, gap mm; rotation vector deg).
  inhand       right-wrist frame at the stop; label = reader in the TCP from the same seated pose minus the
               nominal grasp (InsertionTask.inhand_error units: mm, deg), the inhand_ih1 target.
Each dataset: <out>/<name>.npz (X uint8 (N, 320, 320, 3) policy images, Y labels, Y_net = the logged net outputs,
tick, run) and <out>/report.json with the net error against the labels (mean, std, p50, p99 per output).

    PYTHONPATH=. .venv/bin/python scripts/real2sim/export_real_dataset.py data/real_runs/2026-11-02/* --out data/real_datasets/2026-11-02
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SIDES = ('right', 'left')


def T(p, R):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


def rotvec(R):
    from blind_insertion.envs import kinematics as K
    return K.rot_log(torch.as_tensor(R, dtype=torch.float32)[None])[0].numpy()


class RunData:
    def __init__(self, d, sc):
        from deploy.bimanual.runlog import load_run
        self.dir = Path(d)
        self.meta, self.st, self.ev, self.fr = load_run(d)
        tk = self.st.get('ticks')
        self.sc = sc
        self.off = np.asarray([self.meta['calibration']['joint_offsets_rad'][s] for s in SIDES])
        if tk is not None:
            o = np.argsort(tk['k'])
            self.k = tk['k'][o].astype(int)
            self.q = tk['q'][o]
            self.stage = tk['stage'][o].astype(int)
            self.grip = tk['grip'][o]
        self.outcome = (self.meta.get('outcome') or {}).get('label')

    def qb(self, tick):
        """Believed angles (2, 6) after tick `tick` (the reading logged with it)."""
        i = int(np.clip(np.searchsorted(self.k, tick), 0, len(self.k) - 1))
        return self.q[i] + self.off

    def tcp(self, qb6):
        p, R = self.sc.tcp_chain.fk(torch.tensor(qb6, dtype=torch.float32)[None])
        return T(p[0].numpy(), R[0].numpy())

    def frames(self, cam, tag_prefix, kind='policy'):
        return [f for f in self.fr if f['cam'] == cam and f['kind'] == kind and f['tag'].startswith(tag_prefix)]

    def image(self, f):
        import cv2
        return cv2.cvtColor(cv2.imread(str(self.dir / f['file'])), cv2.COLOR_BGR2RGB)

    def events(self, kind, **match):
        return [e for e in self.ev if e.get('kind') == kind and all(e.get(k) == v for k, v in match.items())]

    def port(self):
        e = next(iter(self.events('estimate', est='zed_pc')), None)
        if e is None:
            return None
        p, R = self.sc.port_frame(torch.tensor([e['port_xy']]), torch.tensor([math.radians(e['yaw_deg'])]))
        return T(p[0].numpy(), R[0].numpy())

    def close_tick(self):
        """First pick tick with the right gripper closing (the expert's phase 3 starts), from the gripper readings."""
        m = np.where(self.stage == 1)[0]
        if not len(m):
            return None
        g = self.grip[m, 0]
        drop = np.where(np.diff(g) < -0.01)[0]
        if not len(drop):
            return None
        # the hand is settled at the grasp when the fingers start closing; take the reading just before
        return int(self.k[m[drop[-1] if g[drop[-1]] < 0.5 else drop[0]]])

    def seat_tick(self):
        e = next(iter(self.events('seated_detected')), None)
        ins = np.where(self.stage == 5)[0]
        if e is not None and len(ins):
            return int(self.k[ins[0]] + e['step'])
        if self.outcome == 'seated' and len(ins):
            return int(self.k[ins[-1]])
        return None


def net_out(rd, frame, net):
    tag = f'{frame["tag"]}@{frame["tick"]}'
    e = [x for x in rd.ev if x.get('kind') == 'net' and x.get('net') == net and x.get('frame_tag') == tag]
    return np.asarray(e[-1]['out'][0]) if e else None


def stats(err):
    err = np.asarray(err, float)
    if not len(err):
        return None
    return {'n': int(len(err)), 'mean': err.mean(0).round(3).tolist(), 'std': err.std(0).round(3).tolist(),
            'p50_abs': np.percentile(np.abs(err), 50, axis=0).round(3).tolist(), 'p99_abs': np.percentile(np.abs(err), 99, axis=0).round(3).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pc-spot', default='', help='JSON {"port_xy": [x, y], "yaw_deg": d} for runs without meta pc_spot')
    a = ap.parse_args()
    from deploy.bimanual.scene import SceneModel
    from blind_insertion.envs.inhand import measured_mean
    from blind_insertion.envs.insertion_task import InsertionCfg, inhand_pose
    sc = SceneModel('cpu')
    cfg = json.loads((ROOT / 'deploy/config/bimanual.json').read_text())
    consts = json.loads((ROOT / cfg['bundle_dir'] / 'configs/chain_constants.json').read_text())
    R_sr, p_sr = (x.numpy() for x in measured_mean())
    R0, p0 = inhand_pose(1, InsertionCfg(), 'cpu', nominal=True)
    R_nom, p_nom = R0[0].numpy(), p0[0].numpy()
    ds = {k: {'X': [], 'Y': [], 'Y_net': [], 'tick': [], 'run': []} for k in ('zed_pc', 'pick_servo', 'zed_reader', 'approach', 'inhand')}
    err = {k: [] for k in ds}
    err.update(approach_label_vs_truth=[], inhand_label_vs_truth=[])
    label_src = {'zed_pc': 'measured PC spot', 'pick_servo': 'encoder FK at the close', 'zed_reader': 'FK at the close + nominal in-hand',
                 'approach': 'encoder FK from the seated pose (port rotation: ZED)', 'inhand': None}

    def add(name, rd, f, y, yn):
        d = ds[name]
        d['X'].append(rd.image(f))
        d['Y'].append(np.full(6, np.nan) if y is None else np.pad(np.asarray(y, float), (0, 6 - len(y)), constant_values=np.nan))
        d['Y_net'].append(np.full(6, np.nan) if yn is None else np.pad(np.asarray(yn, float), (0, 6 - len(yn)), constant_values=np.nan))
        d['tick'].append(f.get('tick', -1))
        d['run'].append(str(rd.dir))
        if y is not None and yn is not None:
            err[name].append(np.asarray(yn, float)[:len(y)] - np.asarray(y, float))

    for d in a.runs:
        if not (Path(d) / 'meta.json').exists():
            continue
        rd = RunData(d, sc)
        spot = rd.meta.get('pc_spot') or (json.loads(a.pc_spot) if a.pc_spot else None)
        # ZED PC
        for f in rd.frames('top', 'zed_pc'):
            e = next(iter(rd.events('estimate', est='zed_pc')), None)
            yn = None if e is None else [*e['port_xy'], e['yaw_deg']]
            y = None if spot is None else [*spot['port_xy'], spot['yaw_deg']]
            add('zed_pc', rd, f, y, yn)
        if not hasattr(rd, 'k'):
            continue
        port = rd.port()
        # pick servo + reader (from the closed grasp)
        tc = rd.close_tick()
        held = (rd.events('pick_result') or [{}])[-1].get('present')
        if tc is not None and held:
            Tc = rd.tcp(rd.qb(tc)[0])
            for f in rd.frames('wrist_right', '1_pick'):
                Tt = rd.tcp(rd.qb(f['tick'])[0])
                D = np.linalg.inv(Tt) @ Tc
                y = [*D[:3, 3], *rotvec(D[:3, :3])]
                sv = [e for e in rd.ev if e.get('kind') == 'servo_frame' and e.get('frame_tag') == f'{f["tag"]}@{f["tick"]}']
                add('pick_servo', rd, f, y, sv[-1]['delta'] if sv else None)
            Tr = Tc @ np.linalg.inv(T(p_sr, R_sr))               # reader (plug-tip frame) from the nominal in-hand pose
            yaw = math.atan2(-Tr[1, 2], -Tr[0, 2])               # the estimator event's convention
            for f in rd.frames('top', 'zed_reader'):
                e = next(iter(rd.events('estimate', est='zed_reader')), None)
                yn = None if e is None else [e['p'][0], e['p'][1], e['yaw_deg']]
                add('zed_reader', rd, f, [Tr[0, 3], Tr[1, 3], math.degrees(yaw)], yn)
        # approach + in-hand from the seated pose
        ts = rd.seat_tick()
        port_spot = None
        if spot is not None:      # measured PC spot: absolute port position for the in-hand label
            pp, RR = sc.port_frame(torch.tensor([spot['port_xy']]), torch.tensor([math.radians(spot['yaw_deg'])]))
            port_spot = T(pp[0].numpy(), RR[0].numpy())
        truth = {e['frame_tag']: e for e in rd.events('frame_truth')}
        if ts is not None and port is not None:
            Ts = rd.tcp(rd.qb(ts)[0])
            S = T(np.array([0., 0., -consts['seat_depth_m']]), np.eye(3))
            X = np.linalg.inv(Ts) @ port @ S                  # plug in the TCP frame (rigid grasp from the stop to the seat)
            for f in rd.frames('wrist_right', '4_approach'):
                Pt = np.linalg.inv(port) @ rd.tcp(rd.qb(f['tick'])[0]) @ X
                y = [*(Pt[:3, 3] * 1e3), *np.degrees(rotvec(Pt[:3, :3]))]
                tr = truth.get(f'{f["tag"]}@{f["tick"]}')
                if tr is not None:
                    err['approach_label_vs_truth'].append(np.asarray(y) - np.r_[tr['plug_in_port_mm'], tr['plug_rot_deg']])
                add('approach', rd, f, y, net_out(rd, f, cfg['chain_args']['fine_net']) if abs(y[2]) <= 7.5 else net_out(rd, f, cfg['chain_args']['wrist_net']))
            ih = [x for x in rd.ev if x.get('kind') == 'net' and x.get('net') == cfg['chain_args']['inhand_net']]
            if ih:
                tag = ih[-1]['frame_tag']
                f = next((g for g in rd.frames('wrist_right', '4_approach') if f'{g["tag"]}@{g["tick"]}' == tag), None)
                if f is not None:
                    Xi = X if port_spot is None else np.linalg.inv(Ts) @ port_spot @ S
                    y = [*((Xi[:3, 3] - p_nom) * 1e3), *np.degrees(rotvec(Xi[:3, :3] @ R_nom.T))]
                    label_src['inhand'] = 'measured PC spot' if port_spot is not None else 'ZED estimate (its error is in the label)'
                    add('inhand', rd, f, y, ih[-1]['out'][0])
                    tr = truth.get(tag)
                    if tr is not None:
                        yt = [*((np.asarray(tr['inhand_p']) - p_nom) * 1e3), *np.degrees(rotvec(np.asarray(tr['inhand_R']) @ R_nom.T))]
                        err['inhand_label_vs_truth'].append(np.asarray(y) - np.asarray(yt))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {'runs': a.runs, 'datasets': {}}
    for k, d in ds.items():
        if not d['X']:
            rep['datasets'][k] = {'n': 0}
            continue
        np.savez_compressed(out / f'{k}.npz', X=np.stack(d['X']).astype(np.uint8), Y=np.stack(d['Y']), Y_net=np.stack(d['Y_net']),
                            tick=np.asarray(d['tick']), run=np.asarray(d['run']))
        rep['datasets'][k] = {'n': len(d['X']), 'labelled': int(np.isfinite(np.stack(d['Y'])[:, 0]).sum()), 'label_source': label_src[k],
                              'net_error': stats(err[k])}
    for k in ('approach_label_vs_truth', 'inhand_label_vs_truth'):
        if err[k]:
            rep[k] = stats(err[k])           # dry runs only: label quality against the sim truth
    (out / 'report.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'runs'}, indent=1))


if __name__ == '__main__':
    main()
