"""Real port 2 position in the right base from the right wrist camera and the arm, independent of the ZED.

Per align_measure event of a real run (still arm, port vision ok): the raw right wrist frames of that measurement
(tick - 6 .. tick), the panel pose by PnP (deploy/bimanual/port_vision.measure -> debug R, t; camera frame, mm),
port 2 centre in the camera = t + R @ port_center_in_model, then through the hand-eye calibration
(real_cameras_duchamp3.json wrist.right.pose_in_link6, OpenCV optical frame) and the believed FK of link_6 at the
logged q_b (encoder + joint offsets) into the right base. No ZED, no in-hand estimate, no grasp model.
Yaw: the panel normal (PnP rotation) in the right base, the ZED nets' convention (insertion direction).
Per run: the median over the measurements before the hand-over (the press holds the PC; insertion not used).
Also logged per run: the v1 ZED estimate of the run and the first wrist-net reading at the look pose.

    PYTHONPATH=. python scripts/perception/zed_arm_port.py --dates 2026-11-10 2026-11-11 2026-11-13 --out results/placement/zed_map/arm_port.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')


def quat_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class ArmCam:
    def __init__(self):
        from deploy.bimanual.scene import SceneModel
        self.sm = SceneModel('cpu')
        self.m = self.sm.mjm
        self.d = mujoco.MjData(self.m)
        self.qa = self.sm.indices['right'][0]
        w = json.loads((ROOT / 'blind_insertion/configs/real_cameras_duchamp3.json').read_text())['wrist']['right']['pose_in_link6']
        self.p_cl, self.R_cl = np.asarray(w['pos'], float), quat_mat(w['quat'])
        self.b6 = self.m.body('link_6').id

    def cam_pose(self, q_b):
        self.d.qpos[self.qa] = q_b
        mujoco.mj_kinematics(self.m, self.d)
        R6, p6 = self.d.xmat[self.b6].reshape(3, 3), self.d.xpos[self.b6]
        return p6 + R6 @ self.p_cl, R6 @ self.R_cl


def events(run):
    out = []
    for line in (run / 'events.jsonl').read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def run_port(run, ac, max_meas=30):
    from deploy.bimanual import port_vision as PV
    ev = events(run)
    zed = next((e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc'), None)
    ho = next((e for e in ev if e.get('kind') == 'align_handover'), None)
    t_ho = ho['t'] if ho else float('inf')
    meas = [e for e in ev if e.get('kind') == 'align_measure' and e.get('ok') and e['t'] <= t_ho + 1e-6 and e.get('q_b')]
    if not meas:
        return None
    fr = [json.loads(x) for x in (run / 'frames.jsonl').read_text().splitlines() if x.strip()]
    fr = [f for f in fr if f['cam'] == 'wrist_right' and f['kind'] == 'raw' and f.get('tick') is not None and f['tag'].startswith('align')]
    pc = np.array(PV.load_model().get('port_center_in_model_mm', PV.PORT_CENTER_MM), float)
    rows = []
    for e in meas[-max_meas:]:
        fs = [f for f in fr if e['tick'] - 6 <= f['tick'] <= e['tick']]
        P = []
        for f in fs:
            img = cv2.imread(str(run / f['file']))
            if img is None:
                continue
            m = PV.measure(img)
            if 'R' not in m.debug:
                continue
            port_cam = (m.debug['t'] + m.debug['R'] @ pc) * 1e-3
            pcw, Rcw = ac.cam_pose(np.asarray(e['q_b'], float))
            zo = Rcw @ m.debug['R'][:, 2]              # model z (out of the panel) in the right base
            P.append([*(pcw + Rcw @ port_cam), math.atan2(-zo[1], -zo[0])])
        if P:
            P = np.array(P)
            rows.append(dict(cycle=e.get('cycle'), tick=e['tick'], n=len(P), port=np.median(P[:, :3], 0).tolist(), yaw_deg=math.degrees(float(np.median(P[:, 3]))),
                             spread_mm=float(np.ptp(P[:, 1]) * 1e3),
                             meas=[e.get('across'), e.get('up'), e.get('gap')]))
    if not rows:
        return None
    Pm = np.array([r['port'] for r in rows])
    first_net = next((e for e in ev if e.get('kind') == 'net' and 'approach' in str(e.get('frame_tag'))), None)
    return dict(run=run.name, n_meas=len(rows), port_mm=(np.median(Pm, 0) * 1e3).tolist(), yaw_deg=float(np.median([r['yaw_deg'] for r in rows])), port_mad_mm=(np.median(np.abs(Pm - np.median(Pm, 0)), 0) * 1e3).tolist(),
                port_range_y_mm=[float(Pm[:, 1].min() * 1e3), float(Pm[:, 1].max() * 1e3)],
                zed_v1=None if zed is None else dict(xy_mm=[v * 1e3 for v in zed['port_xy']], yaw_deg=zed['yaw_deg'], ok=zed['ok']),
                first_wrist_net=None if first_net is None else first_net['out'][0][:3], handover=bool(ho and ho.get('ok')), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dates', nargs='+', default=['2026-11-10', '2026-11-11', '2026-11-13'])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ac = ArmCam()
    res = []
    for d in a.dates:
        for run in sorted((RUNS / d).glob('full_*')):
            meta = run / 'meta.json'
            if not meta.exists() or json.loads(meta.read_text()).get('dry_run', True) or not (run / 'frames.jsonl').exists():
                continue
            r = run_port(run, ac)
            if r is None:
                continue
            r['date'] = d
            res.append(r)
            z = r['zed_v1']
            print(f"{d} {r['run'][:32]:32s} n {r['n_meas']:2d} arm port ({r['port_mm'][0]:.1f}, {r['port_mm'][1]:.1f}, {r['port_mm'][2]:.1f}) yaw {r['yaw_deg']:.2f} mad y {r['port_mad_mm'][1]:.2f} "
                  + (f"| v1 ({z['xy_mm'][0]:.1f}, {z['xy_mm'][1]:.1f}) dy(arm-v1) {r['port_mm'][1]-z['xy_mm'][1]:+.1f} dx {r['port_mm'][0]-z['xy_mm'][0]:+.1f} v1 yaw {z['yaw_deg']:.2f}" if z else ''), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(method=__doc__.strip(), runs=res), indent=1))


if __name__ == '__main__':
    main()
