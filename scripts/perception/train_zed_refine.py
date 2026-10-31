"""Two-stage ZED PC pose: a refinement CNN on a tight 64-px crop centred on the coarse estimate's
projected PC centre. Context (arms, wall, reader) mostly falls outside the crop, so the refiner must
use the case itself. Trained on DR renders with varied arm poses / reader placement (render_data
.vary_arms), coarse estimates = truth + N(0, 4 mm) xy, N(0, 2 deg) yaw; target = truth - coarse.
Tested in sim (by arm condition, coarse from the noise model) and on the 123 real stationary frames with
coarse estimates from several nets (does the refiner pull them to one answer?).

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. .venv/bin/python scripts/perception/train_zed_refine.py
Writes runs/perception/zed_refine.pt, results/usbc/policy/approach/zed_refine.json
"""
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import cv2
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.left_hold import pc_box
from blind_insertion.perception.nets import PoseNet
from blind_insertion.perception.render_data import ApproachRenderer
from blind_insertion.perception.zed_estimator import NOMINAL_PORT_XY, zed_crop
from blind_insertion.scenes.physical_rig import real_top_policy_image

S = 64
REAL = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/cameras')
REF = json.loads(Path('results/usbc/bimanual_20261029/overlays/manifest.json').read_text())['scene_annotation']


class Projector:
    """Nominal top camera (no DR error): world point -> pixel in the 320-px policy frame."""
    def __init__(self, m, dev, res=320):
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        c = m.camera('top').id
        self.p = torch.tensor(d.cam_xpos[c], device=dev, dtype=torch.float32)
        self.R = torch.tensor(d.cam_xmat[c].reshape(3, 3), device=dev, dtype=torch.float32)
        self.f = .5 * res / math.tan(math.radians(m.cam_fovy[c]) / 2); self.c = res / 2
        self.box = pc_box()
        self.top_z = None

    def centre_px(self, port_xy, yaw, port_z):
        """PC lid centre (port pose + box centre at the lid height) -> (u, v) pixels."""
        n = len(port_xy)
        a = yaw
        # port frame: z = -(cos a, sin a, 0) (out of the panel), y up, x = y cross z
        z = -torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], 1)
        y = torch.tensor([0., 0., 1.], device=a.device).expand(n, 3)
        x = torch.cross(y, z, dim=1)
        R = torch.stack([x, y, z], 2)
        cl = torch.tensor([self.box['centre'][0], self.box['top_y'], self.box['centre'][2]], device=a.device, dtype=torch.float32)
        p = torch.cat([port_xy, port_z[:, None]], 1) + (R @ cl[:, None])[..., 0]
        pc = (p - self.p) @ self.R                       # camera frame (x right, y up, z back)
        return self.f * pc[:, 0] / -pc[:, 2] + self.c, self.c - self.f * pc[:, 1] / -pc[:, 2]


def crops(frames, u, v):
    out = []
    for f, a, b in zip(frames, u.round().long(), v.round().long()):
        x0 = int(a.clamp(S // 2, 320 - S // 2)) - S // 2; y0 = int(b.clamp(S // 2, 320 - S // 2)) - S // 2
        out.append(f[y0:y0 + S, x0:x0 + S])
    return torch.stack(out)


def make(r, proj, n, scale, gen):
    X, Y = [], []
    while sum(len(x) for x in X) < n:
        img, lab = r.sample_top(vary=scale)
        p, R = r.h.pc_pose()
        cxy = lab['port_xy'] + torch.randn(r.n, 2, device=r.dev, generator=gen) * .004
        cyaw = lab['port_yaw'] + torch.deg2rad(torch.randn(r.n, device=r.dev, generator=gen) * 2.)
        u, v = proj.centre_px(cxy, cyaw, p[:, 2])
        X.append(crops(img, u, v))
        dy = torch.atan2(torch.sin(lab['port_yaw'] - cyaw), torch.cos(lab['port_yaw'] - cyaw))
        Y.append(torch.cat([(lab['port_xy'] - cxy) * 1e3, torch.rad2deg(dy)[:, None], cxy * 1e3, torch.rad2deg(cyaw)[:, None], p[:, 2:3]], 1))
    return torch.cat(X)[:n], torch.cat(Y)[:n]


def main():
    dev = torch.device('cuda:0')
    r = ApproachRenderer(nworld=256, seed=31, task_layout='results/usbc/policy/pick/task_layout.json', pc_pose_range=(.03, .03, 10.))
    proj = Projector(r.rt.mjm, dev)
    gen = torch.Generator(device=dev); gen.manual_seed(5)
    X, Y = make(r, proj, 32768, .7, gen)
    Xt, Yt = make(r, proj, 2048, .7, gen)
    sc = torch.tensor([6., 6., 3.], device=dev)
    net = PoseNet(3, S, width=32).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    ep_, bs = 40, 128
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=ep_ * (len(X) // bs), pct_start=.1)
    for ep in range(ep_):
        perm = torch.randperm(len(X), device=dev)
        for i in range(len(X) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.smooth_l1_loss(net(X[idx]), Y[idx, :3] / sc, beta=.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        pr = torch.cat([net(Xt[i:i + 256]) for i in range(0, len(Xt), 256)]) * sc
    e = (pr[:, :2] - Yt[:, :2]).norm(dim=1)
    out = {'method': __doc__.strip(), 'sim_heldout_vary0.7': {'n': len(e), 'xy_mm_p50': float(e.quantile(.5)), 'xy_mm_p99': float(e.quantile(.99)),
                                                             'coarse_xy_mm_p50': float(Yt[:, :2].norm(dim=1).quantile(.5))}}
    torch.save({'net': net.state_dict(), 'scale': sc.cpu(), 'size': S}, 'runs/perception/zed_refine.pt')
    # real: coarse estimates from several nets, refine twice
    files = sorted(glob.glob(str(REAL / REF['trial'] / 'zed' / 'raw_*.png')))
    imgs = torch.tensor(np.stack([real_top_policy_image(cv2.imread(f), 320) for f in files]), device=dev)
    pz = torch.full((len(imgs),), float(r.h.pc_pose()[0][0, 2]), device=dev)
    real = {}
    for name in ('zed', 'zed_abl_e2_arms', 'zed_v4s0'):
        ck = torch.load(f'runs/perception/{name}.pt', map_location=dev)
        pn = PoseNet(ck.get('n_out', len(ck['scale'])), ck['size'], width=ck.get('width', 32)).to(dev).eval(); pn.load_state_dict(ck['net'])
        with torch.no_grad():
            est = pn(zed_crop(imgs))[:, :3] * ck['scale'].to(dev)
        cxy = torch.tensor(NOMINAL_PORT_XY, device=dev) + est[:, :2] * 1e-3
        cyaw = torch.deg2rad(est[:, 2])
        traj = [cxy.mean(0).tolist()]
        for it in range(3):
            u, v = proj.centre_px(cxy, cyaw, pz)
            with torch.no_grad():
                d = net(crops(imgs, u, v)) * sc
            cxy = cxy + d[:, :2] * 1e-3; cyaw = cyaw + torch.deg2rad(d[:, 2])
            traj.append(cxy.mean(0).tolist())
        ref = np.array(REF['derived_port_xy_m'])
        real[name] = {'coarse_vs_landmark_mm': float(np.linalg.norm(np.array(traj[0]) - ref) * 1e3),
                      'refined_vs_landmark_mm': [float(np.linalg.norm(np.array(t) - ref) * 1e3) for t in traj[1:]],
                      'refined_xy_m': traj[-1]}
    out['real_stationary'] = real
    Path('results/usbc/policy/approach/zed_refine.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'method'}, indent=1))


if __name__ == '__main__':
    main()
