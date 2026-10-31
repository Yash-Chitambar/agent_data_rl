"""Demo videos of the full chain (called by eval_chain.py --demo N).

Runs one batch of --nworld worlds (layouts, physics and visual DR as in the eval), records the full
state of every world at every control step, then re-renders the chosen episodes:
  top row: the policy cameras in the Warp renderer with the episode's visual DR (ZED, right wrist,
           left wrist; 320 px, shown at 426 px);
  bottom:  a free CPU MuJoCo camera (nominal look, 1280 x 720) that frames the workspace during the
           pick / hold / transport and closes in on port 2 for the approach and the insertion.
Real time (25 fps, every second 50 Hz control step), stage label and outcome overlaid.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/eval/eval_chain.py \
        --demo 6 --nworld 24 --seed 9100
Writes results/usbc/policy/chain/demos/*.mp4 and contact_sheet.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
import pandas as pd
import torch

OUT = Path(__file__).resolve().parents[2] / 'results/usbc/policy/chain/demos'
TOP = 426
W, H = 1280, 720


def _free_cam(m, d, label, t_stage, lay_port):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    port = lay_port
    over = dict(lookat=np.array([.15, .33, 0.]), distance=.7, azimuth=160., elevation=-55.)
    close = dict(lookat=port + np.array([-.035, 0., .005]), distance=.17, azimuth=35., elevation=-18.)
    stage = int(label[0])
    if stage <= 3:
        u = 0.
    else:
        u = min(1., t_stage / 1.5) if stage == 4 else 1.
    u = u * u * (3 - 2 * u)
    for k in ('lookat',):
        cam.lookat[:] = over[k] * (1 - u) + close[k] * u
    cam.distance = over['distance'] * (1 - u) + close['distance'] * u
    cam.azimuth = over['azimuth'] * (1 - u) + close['azimuth'] * u
    cam.elevation = over['elevation'] * (1 - u) + close['elevation'] * u
    return cam


def _text(img, s, org, scale=.8, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


STAGE_NAMES = {'0': 'ZED estimate', '1': 'pick (scripted expert)', '2': 'left hold (guarded 25 N press)',
               '3': 'transport to look pose', '4': 'approach (wrist RGB servo)', '5': 'insertion (blind PPO)'}


def render_demos(ch, a):
    from eval_chain import classify
    OUT.mkdir(parents=True, exist_ok=True)
    rt, n = ch.rt, ch.n
    out = ch.run_batch(a.seed, record_ids=list(range(n)))
    df = pd.DataFrame(out)
    df['e2e_class'] = classify(df)
    qs = np.stack(ch.rec['q'])          # (T, n, nq)
    labels = ch.rec['label']
    succ = [i for i in range(n) if df.e2e_class[i] == 'success'][:a.demo]
    fail = [i for i in range(n) if df.e2e_class[i] != 'success'][:2]
    chosen = succ + fail
    print('demo worlds', chosen, df.e2e_class[chosen].tolist(), flush=True)
    m = rt.mjm
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
    renderer = mujoco.Renderer(m, H, W)
    d = mujoco.MjData(m)
    fa = rt.pc_qadr
    stage_start = {}
    for t, lb in enumerate(labels):
        stage_start.setdefault(lb, t)
    writers = {w: imageio.get_writer(OUT / f'chain_w{w:02d}_{df.e2e_class[w]}.mp4', fps=25, codec='libx264', quality=8,
                                     macro_block_size=None) for w in chosen}
    sheet = {}
    T = len(labels)
    ids = torch.arange(n, device=rt.device)
    for t in range(0, T, 2):
        rt.q[ids] = torch.as_tensor(qs[t], device=rt.device)
        fr = ch.render()
        cams = [fr[c][chosen].cpu().numpy() for c in ('top', 'wrist_cam', 'left_wrist_cam')]
        lb = labels[t]
        t_stage = (t - stage_start[lb]) * ch.dt
        for j, w in enumerate(chosen):
            d.qpos[:] = qs[t, w]
            mujoco.mj_forward(m, d)
            port = d.qpos[fa:fa + 3].copy()
            renderer.update_scene(d, camera=_free_cam(m, d, lb, t_stage, port))
            bottom = renderer.render().copy()
            top = np.concatenate([cv2.resize(c[j], (TOP, TOP), interpolation=cv2.INTER_LINEAR) for c in cams], 1)
            top = np.pad(top, ((0, 0), (0, W - top.shape[1]), (0, 0)))
            img = np.concatenate([top, bottom], 0)
            for k, nm in enumerate(('ZED (policy view)', 'right wrist', 'left wrist')):
                _text(img, nm, (10 + k * TOP, 28), .7)
            _text(img, f'{lb[2:]}', (20, TOP + 45), 1.1, (255, 230, 120))
            _text(img, STAGE_NAMES[lb[0]], (20, TOP + 85), .8)
            _text(img, f't = {t * ch.dt:5.1f} s   world {w}  seed {a.seed}', (20, TOP + H - 20), .7)
            if lb[0] == '5' and t_stage > 0:
                res = df.insert_outcome[w]
                tt = df.insert_t_s[w]
                if np.isfinite(tt) and t_stage >= tt:
                    col = (120, 255, 120) if res == 'seated' else (255, 110, 110)
                    _text(img, f'{res.upper()}' + ('' if res == 'seated' else f'  ({df.e2e_class[w]})'), (W - 520, TOP + 45), 1.1, col)
            writers[w].append_data(img)
            if lb[0] == '5' and (w not in sheet or t_stage <= (df.insert_t_s[w] if np.isfinite(df.insert_t_s[w]) else 99)):
                sheet[w] = img
    for wr in writers.values():
        wr.close()
    renderer.close()
    tiles = [cv2.resize(sheet[w], (640, int(640 * sheet[w].shape[0] / W))) for w in chosen if w in sheet]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    imageio.imwrite(OUT / 'contact_sheet.png', grid)
    meta = {'seed': a.seed, 'nworld': n, 'videos': [f'chain_w{w:02d}_{df.e2e_class[w]}.mp4' for w in chosen],
            'classes': {int(w): df.e2e_class[w] for w in chosen},
            'batch_classes': df.e2e_class.value_counts().to_dict(),
            'layouts': {int(w): {'pc_xy': [float(df.pc_x[w]), float(df.pc_y[w])], 'pc_yaw_deg': float(df.pc_yaw_deg[w]),
                                 'reader_xy': [float(df.reader_x[w]), float(df.reader_y[w])]} for w in chosen},
            'note': 'top row: Warp policy cameras with the episode visual DR; bottom: CPU MuJoCo free camera, nominal look'}
    (OUT / 'demos.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta, indent=1))
    return meta
