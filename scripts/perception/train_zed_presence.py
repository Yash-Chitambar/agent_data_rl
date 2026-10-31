"""Presence and validity gate for the ZED PC-pose CNN (blind_insertion/perception/zed_estimator.py).

Presence: a small CNN (same body as the pose net) classifies the ZED PC crop as PC / no PC. Training
renders use the policy/v1 visual DR; negatives move the PC out of view (the rest of the scene, arms,
reader, patterns and lighting unchanged). Validity: Mahalanobis distance of the pose CNN's pooled
features to their distribution on DR renders; the threshold is the 99.9th percentile on held-out DR
renders with a PC. Tested on held-out sim renders and on real ZED frames (first frame of each curated
2026-10-28 session, labelled by eye: PC in view or not).

    PYTHONPATH=. .venv/bin/python scripts/perception/train_zed_presence.py
Writes runs/perception/zed_presence.pt, results/usbc/policy/approach/zed_presence.json
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import cv2
import mujoco_warp as mjw
import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.perception.nets import PoseNet
from blind_insertion.perception.render_data import ApproachRenderer
from blind_insertion.perception.zed_estimator import OOD_SCALE, ZedEstimator, pooled_features, zed_crop
from blind_insertion.scenes.physical_rig import real_top_policy_image

REAL = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/cameras')
# first frame of each session, by eye from results/usbc/policy/approach/views/real_zed_crops.png
NO_PC = {'board_rgb_001', 'left_elevated_j3_001', 'left_free_settle_001', 'left_free_settle_002',
         'left_free_settle_003', 'left_reader_002', 'left_rest_sweep_001'}


def render(r, n, present):
    out = []
    while sum(len(x) for x in out) < n:
        h = r.h
        h.reset()
        if not present:
            fa = r.rt.pc_qadr
            r.rt.q[:, fa:fa + 3] = torch.tensor([3., 3., -2.], device=r.dev)   # out of view
        with wp.ScopedDevice(r.rt.cfg.device):
            mjw.forward(r.rt.m, r.rt.d)
        out.append(zed_crop(r.rt.render()['top']).contiguous())
    return torch.cat(out)[:n]


def main():
    dev = torch.device('cuda:0')
    r = ApproachRenderer(nworld=256, seed=3)
    Xp, Xn = render(r, 8192, True), render(r, 8192, False)
    r.gen.manual_seed(77); torch.manual_seed(77)
    Tp, Tn = render(r, 1024, True), render(r, 1024, False)
    del r
    X = torch.cat([Xp, Xn]); Y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).to(dev)
    net = PoseNet(1, 160, width=16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(8):
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X) - 127, 128):
            idx = perm[i:i + 128]
            loss = F.binary_cross_entropy_with_logits(net(X[idx])[:, 0], Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pp = torch.cat([torch.sigmoid(net(Tp[i:i + 256])[:, 0]) for i in range(0, len(Tp), 256)])
        pn = torch.cat([torch.sigmoid(net(Tn[i:i + 256])[:, 0]) for i in range(0, len(Tn), 256)])
    # feature statistics of the pose CNN on DR renders with a PC
    ck = torch.load('runs/perception/zed.pt', map_location=dev)
    pose = PoseNet(len(ck['scale']), ck['size']).to(dev).eval(); pose.load_state_dict(ck['net'])
    with torch.no_grad():
        f = torch.cat([pooled_features(pose, Xp[i:i + 256]) for i in range(0, len(Xp), 256)])
        ft = torch.cat([pooled_features(pose, Tp[i:i + 256]) for i in range(0, len(Tp), 256)])
        fn = torch.cat([pooled_features(pose, Tn[i:i + 256]) for i in range(0, len(Tn), 256)])
    mu = f.mean(0); C = torch.cov((f - mu).T) + 1e-3 * torch.eye(f.shape[1], device=dev)
    prec = torch.linalg.inv(C)
    md = lambda g: (((g - mu) @ prec) * (g - mu)).sum(1).clamp(min=0).sqrt()   # noqa: E731
    thr = float(torch.quantile(md(ft), .999))
    Path('runs/perception').mkdir(exist_ok=True, parents=True)
    torch.save({'net': net.state_dict(), 'size': 160, 'width': 16, 'feat_mean': mu.cpu(), 'feat_prec': prec.cpu(),
                'ood_thresh': thr, 'presence_thresh': .5}, 'runs/perception/zed_presence.pt')
    # PoseNet(width=16) for presence: ZedEstimator builds width 32 by default -> rebuild with the stored width
    est = ZedEstimator.__new__(ZedEstimator)
    est.pose, est.scale = pose, ck['scale'].to(dev)
    est.presence, est.mu, est.prec, est.ood_thresh, est.presence_thresh = net, mu, prec, thr, .5
    sim = {'n_pos': len(Tp), 'n_neg': len(Tn),
           'presence_true_pos_rate': float((pp >= .5).float().mean()), 'presence_false_pos_rate': float((pn >= .5).float().mean()),
           'ood_thresh_mahalanobis': thr, 'pos_flagged_ood_rate': float((md(ft) > thr).float().mean()),
           'neg_flagged_ood_rate': float((md(fn) > thr).float().mean()),
           'ood_scale': OOD_SCALE,
           'gate_accept_pos_rate': float(((pp >= .5) & (md(ft) <= thr * OOD_SCALE)).float().mean()),
           'gate_accept_neg_rate': float(((pn >= .5) & (md(fn) <= thr * OOD_SCALE)).float().mean())}
    real = {}
    for sess in sorted(p.name for p in REAL.iterdir() if (p / 'zed').is_dir()):
        files = sorted(glob.glob(str(REAL / sess / 'zed' / 'raw_*.png')))
        if not files:
            continue
        img = torch.tensor(np.stack([real_top_policy_image(cv2.imread(f), 320) for f in files[:1]]), device=dev)
        port, yaw, ok, info = ZedEstimator.__call__(est, img)
        real[sess] = {'pc_in_view_by_eye': sess not in NO_PC, 'pose_returned': bool(ok[0]),
                      'presence': float(info['presence'][0]), 'mahalanobis': float(info['mahalanobis'][0])}
    lab = np.array([v['pc_in_view_by_eye'] for v in real.values()]); got = np.array([v['pose_returned'] for v in real.values()])
    rep = {'method': __doc__.strip(), 'sim_heldout': sim,
           'real_first_frames': {'n': len(real), 'correct': int((lab == got).sum()),
                                 'pc_frames_accepted': f'{int((got & lab).sum())}/{int(lab.sum())}',
                                 'no_pc_frames_rejected': f'{int((~got & ~lab).sum())}/{int((~lab).sum())}', 'per_session': real}}
    Path('results/usbc/policy/approach/zed_presence.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'method'}, indent=1)[:3000])


if __name__ == '__main__':
    main()
