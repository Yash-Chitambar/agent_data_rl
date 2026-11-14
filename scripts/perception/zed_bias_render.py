"""Sim twins of the real ZED capture frames (docs/real/PERCEPTION_BIAS_20261115.md, Part A).

For every real capture frame of results/placement/zed_map/frames.json (n = 89 run frames) the sim renders the same
scene: PC at the case-edge fit pose (geometry 'sim' with the sim PC, 'real' with the measured-geometry patch), both
arms at the logged believed joint angles, right fingers at the logged opening, left fingers open. Then zed_v5 (and v1)
run on the twin and on the real frame. The difference real - twin is the appearance gap at a matched pose.

Variants (all opt-in, nothing changes the default assets):
  --geom sim|real          real = scripts/perception/pc_geom_patch.py (103.5 x 103 mm, port 2 19.8 mm from the centre)
  --panel default|dark     dark = PC panel photo, case boxes and lid darkened to the real colour (load-time patch)
  --look nominal|dr        nominal = visual DR at its nominal draw; dr = one random DR draw per world
  --real-look DIR          BimanualCfg.pc_real_look (real panel and lid textures)
  --left logged|park       left arm at the logged capture angles or parked (render_data LEFT_PARK)
  --left-fingers M         left finger opening per finger (m); 0.0475 = fully open (the training value)

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl python scripts/perception/zed_bias_render.py --geom sim --out results/placement/perception_bias/twin_sim.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/perception'))
RUNS = Path('/home/yashchitambar/peg_test/data/real_runs')


def logged_arms(run_dir, k):
    ev = []
    for line in (Path(run_dir) / 'events.jsonl').read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc':
            ev.append(e)
    e = next((e for e in ev if e.get('frame_tag') == f'zed_pc@{k}'), ev[k] if k < len(ev) else None)
    return None if e is None else e.get('arms_b')


def dark_panel_patch(rgb=(0.06, 0.065, 0.07)):
    """Load-time patch: the PC panel photo texture is multiplied down to the real dark grey (material rgba), so the
    sim panel face is dark like the real one. Opt-in for this process only."""
    from blind_insertion.scenes import pc_rear as PR
    if getattr(PR.add, '_dark', None):
        return
    orig = PR.add

    def add(sp, profile):
        orig(sp, profile)
        for m in sp.materials:
            if m.name == 'pc_rear_photo':
                m.rgba = [*rgb, 1]
    add._dark = True
    PR.add = add


def load_frames(path, dates=None):
    d = json.loads(Path(path).read_text())
    out = []
    for f in d['frames']:
        if f['src'] != 'run' or (dates and f['date'] not in dates):
            continue
        run_dir = Path(f['png']).parents[2]
        k = sorted(p.name for p in (run_dir / 'frames/top').glob('*_zed_pc.png')).index(Path(f['png']).name)
        arms = logged_arms(run_dir, k)
        if arms is None:
            continue
        out.append(dict(f, arms_b=arms))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', default=str(ROOT / 'results/placement/zed_map/frames.json'))
    ap.add_argument('--geom', choices=('sim', 'real'), default='sim')
    ap.add_argument('--panel', choices=('default', 'dark'), default='default')
    ap.add_argument('--panel-rgb', type=float, nargs=3, default=(0.06, 0.065, 0.07))
    ap.add_argument('--look', choices=('nominal', 'dr'), default='nominal')
    ap.add_argument('--real-look', default=None)
    ap.add_argument('--left', default='logged', help="left arm: 'logged' (believed capture angles), 'park' (render_data LEFT_PARK)")
    ap.add_argument('--left-fingers', type=float, default=.0475, help='left finger joint (m per finger); .0475 = fully open')
    ap.add_argument('--reps', type=int, default=1, help='DR draws per frame (look dr)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save-images', default=None, help='directory: side-by-side real | twin pngs (every 8th frame)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.geom == 'real':
        import pc_geom_patch
        pc_geom_patch.apply()
    if a.panel == 'dark':
        dark_panel_patch(tuple(a.panel_rgb))
    from blind_insertion.perception.render_data import ApproachRenderer
    from blind_insertion.perception.zed_estimator import ZedEstimator
    import warp as wp
    import mujoco_warp as mjw
    frames = load_frames(a.frames)
    n = len(frames) * a.reps
    nw = int(math.ceil(n / 32) * 32)
    r = ApproachRenderer(nworld=nw, seed=a.seed, task_layout='results/placement/task_layout_real_v2.json',
                         pc_pose_range=(0., 0., 0.), pc_real_look=a.real_look)
    rt, h = r.rt, r.h
    dev = rt.device
    h.reset()
    if a.look == 'nominal':
        rt.visual.set_nominal(torch.arange(nw, device=dev))
    key = 'edge_' + a.geom
    pose = np.array([frames[i % len(frames)][key]['xy_mm'] + [frames[i % len(frames)][key]['yaw_deg']] for i in range(nw)])
    x, y, yaw = (torch.tensor(pose[:, j], device=dev, dtype=torch.float32) for j in range(3))
    x, y = x / 1e3, y / 1e3
    ref = rt._pc0
    t = torch.deg2rad(yaw)
    c, s_ = torch.cos(t), torch.sin(t)
    qa = rt.pc_qadr
    rt.q[:, qa:qa + 3] = torch.stack([x + c * ref[0] - s_ * ref[1], y + s_ * ref[0] + c * ref[1], ref[2].expand(nw)], -1)
    qz = torch.stack([torch.cos(t / 2), 0 * t, 0 * t, torch.sin(t / 2)], -1)
    w1, x1, y1, z1 = qz.unbind(-1); w2, x2, y2, z2 = ref[3:]
    rt.q[:, qa + 3:qa + 7] = torch.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2], -1)
    ra, la = rt.indices['right'][0], rt.indices['left'][0]
    rf = rt.finger_qadr_right
    for i in range(nw):
        ql, qr, fo = frames[i % len(frames)]['arms_b']
        rt.q[i, ra] = torch.tensor(qr, device=dev)
        rt.q[i, la] = torch.tensor(ql if a.left == 'logged' else r.LEFT_PARK, device=dev)
        rt.q[i, rf] = float(fo)
        rt.q[i, r.left_fingers] = a.left_fingers
    with wp.ScopedDevice(rt.cfg.device):
        mjw.forward(rt.m, rt.d)
    port_p, port_R = h.pc_pose()
    lab_yaw = torch.rad2deg(torch.atan2(-port_R[:, 1, 2], -port_R[:, 0, 2]))
    img = rt.render()['top']
    P = ROOT / 'runs/perception'
    nets = {'v1': ZedEstimator('cuda:0', pose=str(P / 'zed.pt'), presence=str(P / 'zed_v1_gate_arm.pt')),
            'v5': ZedEstimator('cuda:0', presence=str(P / 'zed_v5_gate.pt'))}
    rows = []
    with torch.no_grad():
        res = {k: e(img, arms=None) for k, e in nets.items()}
    for i in range(n):
        f = frames[i % len(frames)]
        row = dict(date=f['date'], run=f['run'], rep=i // len(frames), truth_key=key,
                   truth=[*(port_p[i, :2].cpu().numpy() * 1e3).tolist(), float(lab_yaw[i])],
                   edge=f[key]['xy_mm'] + [f[key]['yaw_deg']], real={k: f[k]['xy_mm'] + [f[k]['yaw_deg']] for k in ('v1', 'v5')})
        for k, (p, yw, ok, info) in res.items():
            row['twin_' + k] = [*(p[i].cpu().numpy() * 1e3).tolist(), math.degrees(float(yw[i])), bool(ok[i])]
        rows.append(row)
    if a.save_images:
        sd = Path(a.save_images); sd.mkdir(parents=True, exist_ok=True)
        tw = img.cpu().numpy()
        for i in range(0, len(frames), 8):
            real_bgr = cv2.imread(frames[i]['png'])
            twin = cv2.cvtColor(np.ascontiguousarray(tw[i, ..., :3]).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(sd / f'{i:02d}_{frames[i]["run"][:28]}.png'), np.hstack([real_bgr, twin]))
        np.save(sd / 'twin_rgb.npy', tw[:len(frames), ..., :3].astype(np.uint8))
    E = {k: np.array([r_['twin_' + k][:3] for r_ in rows]) for k in nets}
    T = np.array([r_['truth'] for r_ in rows]); EG = np.array([r_['edge'] for r_ in rows])
    R = {k: np.array([r_['real'][k] for r_ in rows]) for k in nets}
    summ = dict(n=n, geom=a.geom, panel=a.panel, look=a.look, real_look=a.real_look, left=a.left, left_fingers=a.left_fingers,
                label_minus_edge_med=np.median(T - EG, 0).round(3).tolist())
    for k in nets:
        summ[k] = dict(twin_minus_truth_med=np.median(E[k] - T, 0).round(2).tolist(),
                       real_minus_edge_med=np.median(R[k] - EG, 0).round(2).tolist(),
                       real_minus_twin_med=np.median(R[k] - E[k], 0).round(2).tolist(),
                       real_minus_twin_p10_p90=np.percentile(R[k] - E[k], [10, 90], 0).round(2).tolist(),
                       twin_gate_ok=int(sum(r_['twin_' + k][3] for r_ in rows)))
    print(json.dumps(summ, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(summary=summ, rows=rows), indent=0))


if __name__ == '__main__':
    main()
