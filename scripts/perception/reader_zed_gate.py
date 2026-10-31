"""Validity gate for the ZED reader estimator (as zed_estimator.py does for the PC): Mahalanobis distance
of the pose CNN's pooled features to their distribution on DR renders; threshold = 99.9th percentile on
held-out DR renders x OOD_SCALE. Also a two-crop consistency check is not used (single frame).

Tested on: held-out DR renders (rejection rate and the error of accepted vs rejected frames), renders
with the reader removed (must be rejected), and real ZED frames from the curated 2026-10-28 sessions
(none has the reader on the strip: all must be rejected).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/perception/reader_zed_gate.py --ckpt runs/perception/reader_zed_r320.pt
Adds feat_mean/feat_prec/ood_thresh to the checkpoint; writes results/usbc/policy/pick/reader_zed_gate.json
"""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import cv2
import numpy as np
import torch

from blind_insertion.envs.bimanual_pick import PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.perception.reader_zed import ReaderZedEstimator
from blind_insertion.perception.zed_estimator import pooled_features
from blind_insertion.scenes.physical_rig import real_top_policy_image
import train_reader_zed as T

REAL = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/cameras')
OOD_SCALE = 1.25


def feats(net, X):
    with torch.no_grad():
        return torch.cat([pooled_features(net, X[i:i + 256]) for i in range(0, len(X), 256)])


def maha(F, mu, prec):
    d = F - mu
    return torch.sqrt(((d @ prec) * d).sum(1).clamp(min=0))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='runs/perception/reader_zed_r320.pt')
    ap.add_argument('--n', type=int, default=16384); a = ap.parse_args()
    dev = torch.device('cuda:0')
    est = ReaderZedEstimator(a.ckpt)
    x0, y0, s = est.crop
    task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=256, device='cuda:0', render_rgb=True, visual_dr=True, camera_resolution=est.res,
                                                    measured_camera_cadence=False, task_layout='results/usbc/policy/pick/task_layout.json')))
    torch.manual_seed(7)
    X, Y = T.generate(task, est.crop, a.n)
    torch.manual_seed(8)
    Xt, Yt = T.generate(task, est.crop, 4096)
    # renders without the reader (moved off the table)
    Xa = []
    for _ in range(4):
        task.reset(); pa = task.lay.peg_qadr
        task.runtime.q[:, pa:pa + 3] = torch.tensor([3., 3., -2.], device=dev)
        Xa.append(task.runtime.render()['top'][:, y0:y0 + s, x0:x0 + s].contiguous())
    Xa = torch.cat(Xa)
    F = feats(est.net, X).double()
    mu = F.mean(0); C = torch.cov(F.T) + 1e-3 * torch.eye(F.shape[1], device=dev, dtype=F.dtype)
    prec = torch.linalg.inv(C)
    dt = maha(feats(est.net, Xt).double(), mu, prec)
    thr = float(torch.quantile(dt, .999)) * OOD_SCALE
    with torch.no_grad():
        pt = torch.cat([est.net(Xt[i:i + 256]) for i in range(0, len(Xt), 256)]) * est.scale
    err = torch.linalg.norm(pt[:, :2] - Yt[:, :2], dim=1) * 1e3
    acc = dt <= thr
    da = maha(feats(est.net, Xa).double(), mu, prec)
    real = {}
    for sess in sorted(p.name for p in REAL.iterdir()):
        fs = sorted(glob.glob(str(REAL / sess / 'zed' / 'raw_*.png')))
        if not fs: continue
        im = torch.tensor(real_top_policy_image(cv2.imread(fs[0]), est.res), device=dev)[None, y0:y0 + s, x0:x0 + s].contiguous()
        dr = float(maha(feats(est.net, im).double(), mu, prec)[0])
        real[sess] = {'maha': dr, 'accepted': dr <= thr}
    rep = {'ckpt': a.ckpt, 'ood_thresh': thr, 'ood_scale': OOD_SCALE, 'n_fit': len(X),
           'heldout_dr': {'n': len(Xt), 'accepted_frac': float(acc.float().mean()), 'maha_p50_p999': [float(torch.quantile(dt, .5)), float(torch.quantile(dt, .999))],
                          'xy_err_accepted_p99_max_mm': [float(torch.quantile(err[acc], .99)), float(err[acc].max())],
                          'xy_err_rejected_mm': err[~acc].tolist()[:20], 'n_err_gt_5mm': int((err > 5).sum()), 'n_err_gt_5mm_rejected': int(((err > 5) & ~acc).sum())},
           'no_reader_renders': {'n': len(Xa), 'rejected_frac': float((da > thr).float().mean()), 'maha_p1': float(torch.quantile(da, .01))},
           'real_zed_no_reader_on_strip': real, 'real_rejected': f"{sum(not v['accepted'] for v in real.values())}/{len(real)}"}
    ck = torch.load(a.ckpt, map_location='cpu'); ck.update(feat_mean=mu.float().cpu(), feat_prec=prec.float().cpu(), ood_thresh=thr)
    torch.save(ck, a.ckpt)
    Path('results/usbc/policy/pick/reader_zed_gate.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'real_zed_no_reader_on_strip'}, indent=1)); print(rep['real_rejected'])


if __name__ == '__main__':
    main()
