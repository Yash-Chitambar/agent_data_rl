"""Reader presence classifier for the ZED reader estimator (the Mahalanobis feature gate does not
separate frames without the reader: results/usbc/policy/pick/reader_zed_gate.json).

Positives: DR renders with the reader on the Part 1 strip. Negatives: the same scenes with the reader
removed, or lying elsewhere on the table outside the strip. Tested on held-out renders and on real ZED
frames of the curated 2026-10-28 sessions (no reader on the strip in any of them: all must be rejected).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.:scripts/perception MUJOCO_GL=egl .venv/bin/python scripts/perception/train_reader_presence.py
Adds the presence net to runs/perception/reader_zed_r320.pt ('presence'); writes results/usbc/policy/pick/reader_presence.json
"""
import argparse, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import cv2
import torch
import torch.nn.functional as F

from blind_insertion.envs.bimanual_pick import PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.perception.nets import PoseNet
from blind_insertion.perception.reader_zed import ReaderZedEstimator
from blind_insertion.scenes.physical_rig import real_top_policy_image

REAL = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/cameras')


def render(task, crop, n, mode):
    x0, y0, s = crop
    out = []
    while sum(len(o) for o in out) < n:
        task.reset()
        if mode != 'present':
            pa, N, dev = task.lay.peg_qadr, task.cfg.physics.nworld, task.device
            off = torch.rand(N, device=dev) < .5
            far = torch.tensor([3., 3., -2.], device=dev).expand(N, 3)
            # elsewhere on the table: x 0.20-0.40, y -0.15-0.15 (beside the right arm, outside the crop's strip)
            alt = torch.stack([.2 + .2 * torch.rand(N, device=dev), -.15 + .3 * torch.rand(N, device=dev), task.runtime.q[:, pa + 2]], 1)
            task.runtime.q[:, pa:pa + 3] = torch.where(off[:, None], far, alt)
        out.append(task.runtime.render()['top'][:, y0:y0 + s, x0:x0 + s].contiguous())
    return torch.cat(out)[:n]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', default='runs/perception/reader_zed_r320.pt')
    ap.add_argument('--n', type=int, default=12288); ap.add_argument('--epochs', type=int, default=15); a = ap.parse_args()
    dev = torch.device('cuda:0')
    est = ReaderZedEstimator(a.ckpt)
    task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=256, device='cuda:0', render_rgb=True, visual_dr=True, camera_resolution=est.res,
                                                    measured_camera_cadence=False, task_layout='results/usbc/policy/pick/task_layout.json')))
    torch.manual_seed(21)
    Xp, Xn = render(task, est.crop, a.n, 'present'), render(task, est.crop, a.n, 'absent')
    torch.manual_seed(22)
    Tp, Tn = render(task, est.crop, 2048, 'present'), render(task, est.crop, 2048, 'absent')
    del task
    X = torch.cat([Xp, Xn]); Y = torch.cat([torch.ones(len(Xp)), torch.zeros(len(Xn))]).to(dev)
    net = PoseNet(1, est.crop[2], width=16).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    bs = 128; steps = a.epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=.1)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(len(X) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.binary_cross_entropy_with_logits(net(X[idx])[:, 0], Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        pp = torch.cat([torch.sigmoid(net(Tp[i:i + 256])[:, 0]) for i in range(0, len(Tp), 256)])
        pn = torch.cat([torch.sigmoid(net(Tn[i:i + 256])[:, 0]) for i in range(0, len(Tn), 256)])
        x0, y0, s = est.crop
        real = {}
        for sess in sorted(p.name for p in REAL.iterdir()):
            fs = sorted(glob.glob(str(REAL / sess / 'zed' / 'raw_*.png')))
            if not fs: continue
            im = torch.tensor(real_top_policy_image(cv2.imread(fs[0]), est.res), device=dev)[None, y0:y0 + s, x0:x0 + s].contiguous()
            real[sess] = float(torch.sigmoid(net(im)[0, 0]))
    rep = {'n_train': len(X), 'heldout_present_accepted': float((pp > .5).float().mean()), 'heldout_absent_rejected': float((pn <= .5).float().mean()),
           'real_no_reader_rejected': f"{sum(v <= .5 for v in real.values())}/{len(real)}", 'real_p_present': real}
    ck = torch.load(a.ckpt, map_location='cpu'); ck.update(presence=net.state_dict(), presence_width=16, presence_thresh=.5)
    torch.save(ck, a.ckpt)
    Path('results/usbc/policy/pick/reader_presence.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'real_p_present'}, indent=1))


if __name__ == '__main__':
    main()
