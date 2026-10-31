"""Train the approach estimators on DR renders (policy/v1 visual DR defaults; RGB only) and report
held-out errors.

  zed   : PC port pose (x, y, yaw) in the right-base frame from the ZED frame taken before the arms
          move (crop around the nominal PC). Labels: deviation from the nominal port pose.
  wrist : plug tip pose relative to port 2 (x across, y up, z gap; rotation vector) from the right
          wrist frame at an approach pose (left arm holding the PC, reader in the 60-deg grasp).

    PYTHONPATH=. .venv/bin/python scripts/perception/train_approach_estimators.py --which wrist --n 24576
Writes runs/perception/<which>.pt and results/usbc/policy/approach/estimator_<which>.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.perception.nets import PoseNet, TwoViewPoseNet
from blind_insertion.perception.render_data import ApproachRenderer, SampleCfg

NOMINAL = {'port_xy': (.22, .36), 'yaw': 0.}
ZED_CROP = (110, 70, 160)       # top-left x, y and size (px) of the PC crop in the 320-px ZED frame
WRIST_SIZE = 192
SCALE = {'zed': torch.tensor([30., 30., 10.]), 'wrist': torch.tensor([12., 12., 25., 6., 6., 3.])}
SCALE['wrist2'] = SCALE['wrist']


def wrist_img(img):
    return F.interpolate(img.permute(0, 3, 1, 2).float(), size=(WRIST_SIZE, WRIST_SIZE), mode='area').round().byte().permute(0, 2, 3, 1)


def zed_crop(img):
    x, y, s = ZED_CROP
    return img[:, y:y + s, x:x + s]


def targets(which, lab):
    if which == 'zed':
        yaw = torch.rad2deg(torch.atan2(torch.sin(lab['port_yaw']), torch.cos(lab['port_yaw'])))
        t = torch.cat([(lab['port_xy'] - torch.tensor(NOMINAL['port_xy'], device=yaw.device)) * 1e3, yaw[:, None]], 1)
    else:
        t = torch.cat([lab['rel_p'] * 1e3 - torch.tensor([0., 0., 24.], device=lab['rel_p'].device),
                       torch.rad2deg(lab['rel_rv'])], 1)
        t[:, 2] += 24.
    return t


def generate(r, which, n, cfg=None, nominal=False):
    imgs, ts = [], []
    while sum(len(x) for x in imgs) < n:
        if which == 'zed':
            img, lab = r.sample_top(nominal=nominal)
            img, ok = zed_crop(img), torch.ones(len(img), dtype=torch.bool, device=img.device)
        else:
            fr, lab = r.sample(cfg, nominal=nominal)
            ok = lab['ok']
            img = wrist_img(fr['wrist_cam'])
            if which == 'wrist2':
                img = torch.stack([img, wrist_img(fr['left_wrist_cam'])], 1)
        imgs.append(img[ok].contiguous()); ts.append(targets(which, lab)[ok])
    return torch.cat(imgs)[:n], torch.cat(ts)[:n]


def errors(which, pred, t):
    e = (pred - t).abs().cpu().numpy()
    names = ['x_mm', 'y_mm', 'yaw_deg'] if which == 'zed' else ['x_mm', 'y_mm', 'gap_mm', 'rx_deg', 'ry_deg', 'rz_deg']
    out = {k: {f'p{p}': float(np.percentile(e[:, i], p)) for p in (50, 90, 99, 99.5)} for i, k in enumerate(names)}
    if which != 'zed':
        lat = np.linalg.norm(e[:, :2], axis=1)
        tilt = np.linalg.norm(e[:, 3:5], axis=1)
        out['lateral_mm'] = {f'p{p}': float(np.percentile(lat, p)) for p in (50, 90, 99, 99.5)}
        out['tilt_deg'] = {f'p{p}': float(np.percentile(tilt, p)) for p in (50, 90, 99, 99.5)}
    else:
        xy = np.linalg.norm(e[:, :2], axis=1)
        out['xy_mm'] = {f'p{p}': float(np.percentile(xy, p)) for p in (50, 90, 99, 99.5)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--which', choices=('zed', 'wrist', 'wrist2'), required=True)
    ap.add_argument('--n', type=int, default=24576)
    ap.add_argument('--n-test', type=int, default=2048)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--lateral', type=float, default=.012)
    ap.add_argument('--gap', type=float, nargs=2, default=(.003, .045))
    ap.add_argument('--rot', type=float, default=6.)
    ap.add_argument('--inhand-rot', type=float, default=8., help='+/- in-hand rotation about the closing axis (deg)')
    ap.add_argument('--tag', default='', help='suffix of the saved estimator name (e.g. _fine)')
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    t0 = time.time()
    r = ApproachRenderer(nworld=a.nworld, seed=0)
    r.ins.cfg.inhand_rot_deg = a.inhand_rot
    cfg = SampleCfg(lateral_m=a.lateral, gap_m=tuple(a.gap), rot_deg=a.rot)
    X, Y = generate(r, a.which, a.n, cfg)
    r.gen.manual_seed(12345); torch.manual_seed(12345)
    Xt, Yt = generate(r, a.which, a.n_test, cfg)
    Xn, Yn = generate(r, a.which, 512, cfg, nominal=True)
    t_gen = time.time() - t0
    print(f'generated {len(X)}+{len(Xt)} in {t_gen:.0f} s', flush=True)
    del r
    torch.cuda.empty_cache()
    sc = SCALE[a.which].to(dev)
    size = ZED_CROP[2] if a.which == 'zed' else WRIST_SIZE
    net = (TwoViewPoseNet if a.which == 'wrist2' else PoseNet)(len(sc), size).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    bs = 128
    steps = a.epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=.1)
    k = 0
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(len(X) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            img = X[idx]
            # photometric augmentation on top of the render DR
            loss = F.smooth_l1_loss(net(img), Y[idx] / sc, beta=.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); k += 1
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                pred = torch.cat([net(Xt[i:i + 256]) for i in range(0, len(Xt), 256)]) * sc
            e = errors(a.which, pred, Yt)
            print(ep, f'loss {loss.item():.4f}', {k2: round(v['p50'], 3) for k2, v in e.items()}, flush=True)
    net.eval()
    with torch.no_grad():
        pred = torch.cat([net(Xt[i:i + 256]) for i in range(0, len(Xt), 256)]) * sc
        pn = torch.cat([net(Xn[i:i + 256]) for i in range(0, len(Xn), 256)]) * sc
    Path('runs/perception').mkdir(parents=True, exist_ok=True)
    torch.save({'net': net.state_dict(), 'which': a.which, 'scale': sc.cpu(), 'size': size, 'zed_crop': ZED_CROP,
                'wrist_size': WRIST_SIZE, 'sample_cfg': vars(cfg)}, f'runs/perception/{a.which}{a.tag}.pt')
    rep = {'estimator': a.which + a.tag, 'method': __doc__.strip(), 'n_train': len(X), 'n_test_dr': len(Xt), 'n_test_nominal_look': len(Xn),
           'sample_cfg': vars(cfg), 'held_out_dr_abs_error': errors(a.which, pred, Yt),
           'held_out_nominal_look_abs_error': errors(a.which, pn, Yn), 'generate_s': t_gen, 'total_s': time.time() - t0,
           'note': 'visual DR = policy/v1 defaults incl. camera pose/intrinsics error at real scale; labels exact (sim)'}
    out = Path(f'results/usbc/policy/approach/estimator_{a.which}{a.tag}.json')
    out.write_text(json.dumps(rep, indent=1) + '\n')
    np.savez_compressed(out.with_suffix('.npz'), pred=pred.cpu().numpy(), target=Yt.cpu().numpy())
    print(json.dumps(rep['held_out_dr_abs_error'], indent=1))


if __name__ == '__main__':
    main()
