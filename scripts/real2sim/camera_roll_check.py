"""Is there a camera roll against the sim at the same joint angles? Three probes, sim render against real frame:
  fingers   tips (rigid to the camera): image rotation of the sim mask that best fits the real mask (servo_roll_source.py, cam_fingers)
  panel     long line of the PC's panel or lid, port approach frames; the PC at the run's ZED pose
  rail      the long table-fixed rail edge in the held frames; independent of the ZED
and the wrist nets' output change when a real frame turns in the plane. Angles here: atan2(dy, dx) in the image (y down).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl python scripts/real2sim/camera_roll_check.py
"""
import json
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from inhand_compare import build, set_canvas  # noqa: E402
from inhand_measure import CANVAS, OFF, all_runs, held_frames, undist  # noqa: E402
from blind_insertion.scenes import pc_bench as S  # noqa: E402

OUT = ROOT / 'results/real/inhand_real'
FIT = json.loads((ROOT / 'results/usbc/gripper_sweep/sweep_20261030/fit.json').read_text())['arms']['right']['gripper_map']


def long_line(g, roi, lo, hi, minlen=.5):
    x0, x1, y0, y1 = roi
    c = cv2.Canny(cv2.GaussianBlur(g[y0:y1, x0:x1], (5, 5), 0), 30, 90)
    L = cv2.HoughLinesP(c, 1, np.pi / 1800, 100, minLineLength=int(minlen * max(x1 - x0, y1 - y0)), maxLineGap=25)
    if L is None:
        return None
    best = None
    for x_0, y_0, x_1, y_1 in L.reshape(-1, 4):
        a = np.degrees(np.arctan2(y_1 - y_0, x_1 - x_0))
        if a < lo:
            a += 180
        if a > hi:
            a -= 180
        ln = np.hypot(x_1 - x_0, y_1 - y_0)
        if lo <= a <= hi and (best is None or ln > best[0]):
            best = (ln, a)
    return best


def main():
    m, lay = build(None)
    set_canvas(m)
    d = mujoco.MjData(m)
    rd = mujoco.Renderer(m, CANVAS[1], CANVAS[0])

    def render(q, grip, pc=None):
        for j in range(6):
            d.qpos[m.joint(f'joint{j + 1}').qposadr] = q[j]
        for fn in ('left_finger', 'right_finger'):
            d.qpos[m.joint(fn).qposadr] = FIT['slope'] * grip + FIT['intercept']
        if pc:
            S.place_pc(m, d, port_xy=pc['port_xy'], yaw_deg=pc['yaw_deg'])
        d.qpos[lay.peg_qadr:lay.peg_qadr + 3] = [-.3, -.4, 0]
        mujoco.mj_forward(m, d)
        rd.update_scene(d, camera='wrist_cam')
        return cv2.cvtColor(rd.render(), cv2.COLOR_RGB2GRAY)

    rows = [json.loads(x) for x in open(ROOT / 'data/real_training/port_v1/manifest.jsonl')]
    panel, rail = [], []
    for run in ('full_174850_full_A', 'full_175742_full_A2', 'collect_182100_collect1'):
        rs = [r for r in rows if r['run'] == run and r.get('p_mm') and r.get('raw') and 9 < r['p_mm'][2] < 45][:24]
        ev = next(json.loads(x) for x in open(ROOT / f'data/real_runs/2026-11-02/{run}/events.jsonl') if '"est": "zed_pc"' in x)
        for r in rs:
            g = cv2.cvtColor(undist(cv2.imread(str(ROOT / r['raw']))), cv2.COLOR_BGR2GRAY)
            s = render(np.array(r['q_b']), .31, ev)
            a, b = long_line(g, (300, 1000, 250, 700), -15, 15), long_line(s, (300, 1000, 250, 700), -15, 15)
            if a and b:
                panel.append((run, a[1], b[1]))
    for run in all_runs():
        z = np.load(run / 'ticks.npz')
        for f in held_frames(run)[:6]:
            i = np.searchsorted(z['t'], f['t_capture'])
            g = cv2.cvtColor(undist(cv2.imread(str(run / f['file']))), cv2.COLOR_BGR2GRAY)
            s = render(z['q'][i, 0] + OFF, float(z['grip'][i, 0]))
            a, b = long_line(g, (0, 560, 0, 800), 92, 135), long_line(s, (0, 560, 0, 800), 92, 135)
            if a and b:
                rail.append((run.name, a[1], b[1]))
    rd.close()
    rep = {}
    for nm, x in (('panel', panel), ('rail', rail)):
        df = pd.DataFrame(x, columns=['run', 'real', 'sim'])
        df['diff'] = df.real - df.sim
        by = df.groupby('run')['diff'].median()
        rep[nm] = dict(n_frames=len(df), n_runs=int(len(by)), median_of_runs=float(by.median()), sd_runs=float(by.std()), by_run=by.round(2).to_dict())
        print(nm, rep[nm], flush=True)
    # wrist nets on real frames turned in the plane
    from deploy.bimanual.netcache import wrist
    from blind_insertion.perception.nets import PoseNet  # noqa: F401
    dev = 'cuda:0'
    c = [r for r in rows if r['source'] == 'collect' and r['p_mm'][2] <= 10 and r.get('image')]
    ims = [cv2.imread(str(ROOT / r['image']))[:, :, ::-1].copy() for r in c]
    rep['wrist'] = {}
    for name in ('wrist_v18ws', 'wrist_v14w'):
        ck = torch.load(ROOT / f'deploy/bundle/models/{name}.pt', map_location=dev, weights_only=False)
        from blind_insertion.perception.nets import PoseNet as PN
        net = PN(len(ck['scale']), ck['size'], width=ck.get('width', 32)).to(dev).eval()
        net.load_state_dict(ck['net'])
        sc = ck['scale'].to(dev)

        def out(th):
            X = []
            for i_ in ims:
                h, w = i_.shape[:2]
                X.append(cv2.warpAffine(i_, cv2.getRotationMatrix2D((w / 2 - .5, h / 2 - .5), th, 1.), (w, h), borderMode=cv2.BORDER_REPLICATE))
            x = torch.as_tensor(np.stack(X), device=dev)
            if ck.get('wrist_mode', 'resize') != 'resize':
                print('mode', ck.get('wrist_mode'))
            with torch.no_grad():
                if x.shape[1] != ck['size']:
                    from torch.nn.functional import interpolate
                    x = interpolate(x.permute(0, 3, 1, 2).float(), size=(ck['size'], ck['size']), mode='area').permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
                return (torch.cat([net(x[k:k + 64]) for k in range(0, len(x), 64)]) * sc).cpu().numpy()
        o0 = out(0.)
        rep['wrist'][name] = {'n': len(ims), 'scale_note': 'mm, deg: first 3 across, up, gap'}
        for th in (-3., -1.5, 1.5, 3.):
            d_ = out(th) - o0
            rep['wrist'][name][f'turn {th:+.1f}'] = np.round(np.mean(d_[:, :3], 0), 2).tolist() + np.round(np.mean(d_[:, 3:], 0), 2).tolist()
        print(name, rep['wrist'][name], flush=True)
    (OUT / 'camera_roll.json').write_text(json.dumps(rep, indent=1, default=float))


if __name__ == '__main__':
    main()
