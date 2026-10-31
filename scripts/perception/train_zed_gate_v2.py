"""ZED v2 gate: presence CNN + Mahalanobis OOD + predicted-uncertainty check for the arm-pose-robust ZED
pose net (zed_v2: trained with varied arm poses and gripper openings, heteroscedastic output).

Training / calibration renders (policy/v1 visual DR, reader on the Part 1 strip) use
render_data.vary_arms(scale 1): right arm at rest or perturbed, right fingers closed..fully open, left arm
rest / parked / pressing (2 or 3.5 cm) / perturbed. Negatives move the PC out of view. Thresholds:
Mahalanobis 1.25 x the 99.9th percentile, predicted std at the 99.5th percentile of held-out positives.
Tests: held-out positives per arm category; positives with arm poses beyond the training spread
(vary_arms scale 2.5) -- the gate should reject the frames it cannot handle; negatives; real ZED first
frames of the 24 curated sessions (by-eye labels) and the stationary session vs its manual landmark.

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv/bin/python scripts/perception/train_zed_gate_v2.py
Writes runs/perception/zed_v2_gate.pt, results/usbc/policy/approach/zed_v2_gate.json
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
from blind_insertion.perception.zed_estimator import ZedEstimator, pooled_features, zed_crop
from blind_insertion.scenes.physical_rig import real_top_policy_image

REAL = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/cameras')
NO_PC = {'board_rgb_001', 'left_elevated_j3_001', 'left_free_settle_001', 'left_free_settle_002',
         'left_free_settle_003', 'left_reader_002', 'left_rest_sweep_001'}
REF = json.loads(Path('results/usbc/bimanual_20261029/overlays/manifest.json').read_text())['scene_annotation']
LAYOUT = 'results/usbc/policy/pick/task_layout.json'


def render(r, n, present, scale):
    imgs, labs = [], []
    while sum(len(x) for x in imgs) < n:
        r.h.reset()
        mode = r.vary_arms(scale) if scale else torch.zeros(r.n, dtype=torch.long, device=r.dev)
        fing = r.rt.q[:, [int(r.rt.mjm.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]][:, 0].clone()
        p, R = r.h.pc_pose()
        if not present:
            fa = r.rt.pc_qadr
            r.rt.q[:, fa:fa + 3] = torch.tensor([3., 3., -2.], device=r.dev)
        with wp.ScopedDevice(r.rt.cfg.device):
            mjw.forward(r.rt.m, r.rt.d)
        imgs.append(r.rt.render()['top'].contiguous())
        labs.append(torch.cat([p[:, :2], torch.atan2(-R[:, 1, 2], -R[:, 0, 2])[:, None], mode[:, None].float(), fing[:, None]], 1))
    return torch.cat(imgs)[:n], torch.cat(labs)[:n]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--pose', default='zed_v4s0'); ap.add_argument('--pose2', default='zed_v4s1')
    ap.add_argument('--tag', default='zed_v4_gate'); ap.add_argument('--pc-range', type=float, nargs=3, default=(.03, .03, 10.))
    a = ap.parse_args()
    dev = torch.device('cuda:0')
    r = ApproachRenderer(nworld=256, seed=21, task_layout=LAYOUT, pc_pose_range=a.pc_range)
    Xp, _ = render(r, 8192, True, 1.); Xn, _ = render(r, 8192, False, 1.)
    r.gen.manual_seed(99); torch.manual_seed(99)
    Tp, Lp = render(r, 2048, True, 1.); Tn, _ = render(r, 1024, False, 1.)
    To, Lo = render(r, 1024, True, 2.5)                           # arm poses beyond the training spread
    T0, L0 = render(r, 1024, True, 0.)                            # reset pose (rounds 1-4 condition)
    del r
    torch.cuda.empty_cache()
    crop = lambda X: zed_crop(X).contiguous()                    # noqa: E731
    Xc, Yc = torch.cat([crop(Xp), crop(Xn)]), torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).to(dev)
    net = PoseNet(1, 160, width=16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(8):
        perm = torch.randperm(len(Xc), device=dev)
        for i in range(0, len(Xc) - 127, 128):
            idx = perm[i:i + 128]
            loss = F.binary_cross_entropy_with_logits(net(Xc[idx])[:, 0], Yc[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    ck = torch.load(f'runs/perception/{a.pose}.pt', map_location=dev)
    pose = PoseNet(ck['n_out'], ck['size'], width=ck.get('width', 32)).to(dev).eval(); pose.load_state_dict(ck['net'])
    k, sc = len(ck['scale']), ck['scale'].to(dev)
    with torch.no_grad():
        f = torch.cat([pooled_features(pose, crop(Xp[i:i + 256])) for i in range(0, len(Xp), 256)])
        out = torch.cat([pose(crop(Tp[i:i + 256])) for i in range(0, len(Tp), 256)])
    mu = f.mean(0); C = torch.cov((f - mu).T) + 1e-3 * torch.eye(f.shape[1], device=dev); prec = torch.linalg.inv(C)
    md = lambda g: (((g - mu) @ prec) * (g - mu)).sum(1).clamp(min=0).sqrt()   # noqa: E731
    with torch.no_grad():
        thr = float(torch.quantile(md(torch.cat([pooled_features(pose, crop(Tp[i:i + 256])) for i in range(0, len(Tp), 256)])), .999))
    gate = {'net': net.state_dict(), 'size': 160, 'width': 16, 'feat_mean': mu.cpu(), 'feat_prec': prec.cpu(),
            'ood_thresh': thr, 'ood_scale': 1.25, 'presence_thresh': .5}
    if ck.get('hetero'):
        sig = torch.exp(out[:, k:].clamp(-7, 3)) * sc
        gate['sigma_xy_thresh_mm'] = float(torch.quantile(sig[:, :2].norm(dim=1), .995)); gate['sigma_yaw_thresh_deg'] = float(torch.quantile(sig[:, 2], .995))
    if a.pose2:
        c2 = torch.load(f'runs/perception/{a.pose2}.pt', map_location=dev)
        p2 = PoseNet(c2.get('n_out', len(c2['scale'])), c2['size'], width=c2.get('width', 32)).to(dev).eval(); p2.load_state_dict(c2['net'])
        with torch.no_grad():
            o2 = torch.cat([p2(crop(Tp[i:i + 256]))[:, :k] for i in range(0, len(Tp), 256)])
        dis = (out[:, :k] - o2).abs() * sc
        gate['disagree_xy_thresh_mm'] = float(torch.quantile(dis[:, :2].norm(dim=1), .995)); gate['disagree_yaw_thresh_deg'] = float(torch.quantile(dis[:, 2], .995))
    torch.save(gate, f'runs/perception/{a.tag}.pt')
    est = ZedEstimator(str(dev), pose=f'runs/perception/{a.pose}.pt', presence=f'runs/perception/{a.tag}.pt',
                       pose2=f'runs/perception/{a.pose2}.pt' if a.pose2 else None)

    def evaluate(X, L=None):
        ports, yaws, oks = [], [], []
        for i in range(0, len(X), 256):
            p, y, ok, _ = est(X[i:i + 256]); ports.append(p); yaws.append(y); oks.append(ok)
        p, y, ok = torch.cat(ports), torch.cat(yaws), torch.cat(oks)
        res = {'n': len(X), 'accept': float(ok.float().mean())}
        if L is not None:
            e = (p - L[:, :2]).norm(dim=1) * 1e3
            ey = torch.rad2deg(torch.atan2(torch.sin(y - L[:, 2]), torch.cos(y - L[:, 2]))).abs()
            q = lambda x, m: float(torch.quantile(x[m], .5)) if m.any() else None   # noqa: E731
            q99 = lambda x, m: float(torch.quantile(x[m], .99)) if m.any() else None   # noqa: E731
            res.update(xy_mm_accepted={'p50': q(e, ok), 'p99': q99(e, ok)}, xy_mm_rejected={'p50': q(e, ~ok), 'p99': q99(e, ~ok)},
                       xy_mm_all={'p50': q(e, ok | ~ok), 'p99': q99(e, ok | ~ok)}, yaw_deg_accepted_p99=q99(ey, ok))
            res['_e'], res['_ok'] = e, ok
        return res

    out = {'method': __doc__.strip(), 'pose_nets': [a.pose, a.pose2], 'thresholds': {k_: v for k_, v in gate.items() if k_ not in ('net', 'feat_mean', 'feat_prec', 'size', 'width')}}
    ind = evaluate(Tp, Lp)
    e, ok = ind.pop('_e'), ind.pop('_ok')
    cats = {}
    names = {0: 'left_rest', 1: 'left_parked', 2: 'left_pressing', 3: 'left_perturbed'}
    for c, nm in names.items():
        m = Lp[:, 3] == c
        if not (m & ok).any():
            continue
        cats[nm] = {'n': int(m.sum()), 'accept': float(ok[m].float().mean()), 'xy_mm_p50': float(torch.quantile(e[m & ok], .5)),
                    'xy_mm_p99': float(torch.quantile(e[m & ok], .99))}
    for lo, hi in ((0, .012), (.012, .03), (.03, .0475)):
        m = (Lp[:, 4] >= lo) & (Lp[:, 4] < hi + 1e-9)
        cats[f'right_fingers_{lo * 1e3:.0f}-{hi * 1e3:.0f}mm'] = {'n': int(m.sum()), 'accept': float(ok[m].float().mean()),
                                                                    'xy_mm_p50': float(torch.quantile(e[m & ok], .5)), 'xy_mm_p99': float(torch.quantile(e[m & ok], .99))}
    out['sim_in_distribution'] = ind; out['sim_by_arm_category'] = cats
    rest = evaluate(T0, L0); rest.pop('_e'); rest.pop('_ok'); out['sim_reset_pose'] = rest
    ood = evaluate(To, Lo); ood.pop('_e'); ood.pop('_ok'); out['sim_arms_beyond_training_x2.5'] = ood
    out['sim_no_pc'] = evaluate(Tn)
    real = {}
    for sess in sorted(p.name for p in REAL.iterdir() if (p / 'zed').is_dir()):
        files = sorted(glob.glob(str(REAL / sess / 'zed' / 'raw_*.png')))
        if not files:
            continue
        img = torch.tensor(np.stack([real_top_policy_image(cv2.imread(f_), 320) for f_ in files[:1]]), device=dev)
        port, yaw, okr, info = est(img)
        real[sess] = {'pc_in_view_by_eye': sess not in NO_PC, 'pose_returned': bool(okr[0]), 'presence': float(info['presence'][0]),
                      'mahalanobis': float(info['mahalanobis'][0]), 'sigma_xy_mm': float(info['sigma_xy_mm'][0])}
    lab = np.array([v['pc_in_view_by_eye'] for v in real.values()]); got = np.array([v['pose_returned'] for v in real.values()])
    files = sorted(glob.glob(str(REAL / REF['trial'] / 'zed' / 'raw_*.png')))
    img = torch.tensor(np.stack([real_top_policy_image(cv2.imread(f_), 320) for f_ in files]), device=dev)
    port, yaw, okr, _ = est(img)
    d = (port[okr].mean(0).cpu().numpy() - np.array(REF['derived_port_xy_m'])) * 1e3 if okr.any() else None
    out['real'] = {'first_frames': {'n': len(real), 'pc_frames_accepted': f'{int((got & lab).sum())}/{int(lab.sum())}',
                                    'no_pc_frames_rejected': f'{int((~got & ~lab).sum())}/{int((~lab).sum())}', 'per_session': real},
                   'stationary_vs_landmark': {'n': len(files), 'accepted': int(okr.sum()), 'mean_offset_mm': None if d is None else float(np.linalg.norm(d)),
                                              'std_mm': (port[okr].std(0).cpu().numpy() * 1e3).tolist() if okr.any() else None}}
    Path(f'results/usbc/policy/approach/{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k_: v for k_, v in out.items() if k_ not in ('method', 'real')}, indent=1)); print(json.dumps({k_: v for k_, v in out['real'].items() if k_ != 'first_frames'})); print(out['real']['first_frames']['pc_frames_accepted'], out['real']['first_frames']['no_pc_frames_rejected'])


if __name__ == '__main__':
    main()
