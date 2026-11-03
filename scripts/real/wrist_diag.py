"""Why the wrist nets read compressed values on real frames: same poses, one change at a time. No motion.

    PYTHONPATH=. python scripts/real/wrist_diag.py --fit results/real/wrist_nets/geometry_fit.json --out results/real/wrist_nets/diagnosis.json

Every condition renders the logged q_b of the 150 collect frames, and every slope is against the same real label. Conditions:
  sim_old        old reader and PC look, measured mean grasp, port from the anchor: what the nets trained on, at these poses
  sim_old_dr     as sim_old, under the training visual DR
  sim_inhand     real in-hand preset and reader look, port from the geometry fit, old PC look
  sim_pc         old reader, real PC look (panel, lid from training blocks)
  sim_inhand_pc  real in-hand, reader look, port fit and real PC look: the closest sim to the real frame
  real           the real frames
Also: real frames under image scale and shift (preprocessing), channel order, and blur on sim. Subsets: all gaps, gap < 10, gap < 10 and lateral within the fine net's
training range. Slope = least squares net output per mm of label; interval = bootstrap over frames.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import wrist_net_real_eval as W  # noqa: E402
from wrist_sim_at_real import SimAtReal  # noqa: E402

LOOK = 'blind_insertion/assets/pc_real_look'


def render(fit, co, ql, an, reader_look=None, look=None, port_fit=False, real_grasp=False, dr=None):
    B = len(co)
    S = SimAtReal(B, reader_look=reader_look, look=look, visual_dr=dr)
    dev = S.dev
    eR = torch.tensor(an['port_R'], device=dev)
    q = torch.tensor([r['q_b'] for r in co], device=dev)
    Y = torch.tensor(np.array([r['p_mm'] for r in co]), device=dev).float() * 1e-3
    R_sr, p_sr = S.R_sr.reshape(3, 3), S.p_sr.reshape(3)
    sp, sR = S.h.chain['right'].fk(q)
    port = ((sp - sR @ (R_sr.T @ p_sr)) - (eR @ Y[..., None])[..., 0]).mean(0)
    if port_fit:
        port = torch.tensor(fit['port_p_m'], device=dev, dtype=torch.float32)
    kw = {}
    if real_grasp:
        from blind_insertion.envs.inhand_real import sample_real
        R, p, _ = sample_real(B, 'v1', device=dev, spread=0.)
        kw = dict(R_sr=R, p_sr=p)
    frames = []
    S.h.reset() if dr else None
    S.set_state(q, torch.tensor(ql, device=dev).expand(B, 6), port.expand(B, 3), eR.expand(B, 3, 3), **kw)
    frames = S.rt.render()['wrist_cam'].cpu().numpy()
    del S
    torch.cuda.empty_cache()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', default='results/real/wrist_nets/geometry_fit.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    fit = json.load(open(a.fit))
    co = W.collect_rows()
    ev = [json.loads(x) for x in open(f"data/real_runs/2026-11-02/{co[0]['run']}/events.jsonl")]
    an = [e for e in ev if e['kind'] == 'collect_anchor'][0]
    ql = [e for e in ev if e['kind'] == 'collect_press_removed'][0]['q_left']
    Y = np.array([r['p_mm'] for r in co])
    real = np.stack([W.read_rgb(r['image']) for r in co])
    C = dict(real=real)
    C['sim_old'] = render(fit, co, ql, an)
    C['sim_old_dr'] = render(fit, co, ql, an, dr=True)
    C['sim_inhand'] = render(fit, co, ql, an, reader_look='v1', port_fit=True, real_grasp=True)
    C['sim_pc'] = render(fit, co, ql, an, look=LOOK)
    C['sim_inhand_pc'] = render(fit, co, ql, an, reader_look='v1', look=LOOK, port_fit=True, real_grasp=True)
    np.savez_compressed(Path(a.out).with_suffix('.frames.npz'), **{k: v for k, v in C.items() if k != 'real'})

    def warp(im, s, dx, dy):
        M = np.array([[s, 0, 160 * (1 - s) + dx], [0, s, 160 * (1 - s) + dy]], float)
        return np.stack([cv2.warpAffine(i, M, (320, 320), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for i in im])
    C['real_bgr'] = real[..., ::-1].copy()
    for s in (.9, .95, 1.05, 1.1, 1.2):
        C[f'real_scale_{s}'] = warp(real, s, 0, 0)
    for dx, dy in ((0, -10), (0, 10), (-10, 0), (10, 0)):
        C[f'real_shift_{dx}_{dy}'] = warp(real, 1, dx, dy)
    for sg in (2., 3.):
        C[f'sim_old_blur_{sg}'] = np.stack([cv2.GaussianBlur(i, (0, 0), sg) for i in C['sim_old']])
    subsets = {'all': Y[:, 2] < 99, 'gap<10': Y[:, 2] < 10, 'gap<10,|x|,|y|<3.5': (Y[:, 2] < 10) & (np.abs(Y[:, :2]).max(1) < 3.5), 'gap<10,|x|,|y|<6': (Y[:, 2] < 10) & (np.abs(Y[:, :2]).max(1) < 6)}
    out = {}
    for name in ('wrist_v18ws', 'wrist_v14w'):
        net, sc, ck = W.load_net(f'runs/perception/{name}.pt', a.device)
        out[name] = {}
        for c, im in C.items():
            P = W.predict(net, sc, ck, im, a.device)
            out[name][c] = {}
            for sn, m in subsets.items():
                if not (c in ('real', 'sim_old', 'sim_old_dr', 'sim_inhand', 'sim_pc', 'sim_inhand_pc') or sn in ('all', 'gap<10,|x|,|y|<6')):
                    continue
                out[name][c][sn] = dict(n=int(m.sum()), **{ax: dict(slope=round(W.slope(Y[m, i], P[m, i]), 3), ci=W.boot(Y[m, i], P[m, i], n=1000), resid_sd=round(float((P[m, i] - Y[m, i]).std()), 2),
                                                                     mean_err=round(float((P[m, i] - Y[m, i]).mean()), 2)) for i, ax in enumerate(W.AXES)})
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    for n in out:
        print(n)
        for c in out[n]:
            for sn, v in out[n][c].items():
                print(f'  {c:20s} {sn:20s} n={v["n"]:3d} ' + ' '.join(f"{ax} {v[ax]['slope']:+.2f} [{v[ax]['ci'][0]:+.2f},{v[ax]['ci'][1]:+.2f}] sd {v[ax]['resid_sd']:.1f}" for ax in W.AXES))


if __name__ == '__main__':
    main()
