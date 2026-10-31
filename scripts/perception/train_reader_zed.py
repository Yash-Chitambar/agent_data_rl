"""ZED reader-pose CNN for the Part 1 pick (RGB only, Part 1 strip in front of the PC).

Renders with the pick task's reset: right arm at rest with the gripper open (24 mm), left arm parked,
PC pose DR (x +/-1 cm, y +/-1.5 cm, +/-5 deg), reader uniform on the strip (yaw = bearing + 15 +/- 10
deg), policy/v1 visual DR (appearance, lighting, image noise, camera pose/intrinsics error at real
scale). The input is a fixed square crop of the ZED frame around the strip (projected with the nominal
camera). Labels: reader body-centre xy (m) and plug yaw (rad), relative to the strip centre / 1.5 rad.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/perception/train_reader_zed.py --n 49152
Writes runs/perception/reader_zed{tag}.pt, results/usbc/policy/pick/reader_zed{tag}.json (+ .npz)
"""
import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.envs.bimanual_pick import PickTask, PickTaskCfg
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.perception.nets import PoseNet
from blind_insertion.scenes import physical_rig as Rig

STRIP_CENTRE = (.12, .385)      # body-centre strip centre (results/usbc/policy/pick/reader_region.json)
YAW0 = 1.5                      # rad, about the middle of the strip's plug yaws (73-102 deg)
SCALE = torch.tensor([.04, .06, .3])   # label normalisation (m, m, rad)


def crop_box(rt, res, half_extent=(.16, .16)):
    """Square pixel box (x0, y0, size) around the strip + reader extent in the nominal ZED view."""
    m = rt.mjm
    d = mujoco.MjData(m); Rig.reference_pose(m, d, rt.lay); mujoco.mj_forward(m, d)
    c = m.camera('top').id
    top = float(m.geom('table').pos[2] + m.geom('table').size[2])
    cx, cy = STRIP_CENTRE
    pts = np.array([[cx + sx * half_extent[0], cy + sy * half_extent[1], top] for sx in (-1, 1) for sy in (-1, 1)])
    rel = (pts - d.cam_xpos[c]) @ d.cam_xmat[c].reshape(3, 3)
    f = res / 2 / math.tan(math.radians(m.cam_fovy[c]) / 2); z = -rel[:, 2]
    uv = np.stack([f * rel[:, 0] / z + res / 2, -f * rel[:, 1] / z + res / 2], -1)
    lo, hi = uv.min(0), uv.max(0)
    size = int(math.ceil(max(hi - lo) / 32) * 32)
    ctr = (lo + hi) / 2
    x0 = int(np.clip(round(ctr[0] - size / 2), 0, res - size)); y0 = int(np.clip(round(ctr[1] - size / 2), 0, res - size))
    return x0, y0, size


def labels(task):
    p, R = task.reader_pose()
    bz = float(task.lay.variants[0].peg_boxes[0][0][2])
    c = p + R[:, :, 2] * bz
    plug = -R[:, :2, 2]
    yaw = torch.atan2(plug[:, 1], plug[:, 0])
    return torch.stack([c[:, 0] - STRIP_CENTRE[0], c[:, 1] - STRIP_CENTRE[1], yaw - YAW0], 1)


def generate(task, box, n, nominal=False):
    X, Y = [], []
    x0, y0, s = box
    while sum(len(x) for x in X) < n:
        task.reset()
        if nominal:
            task.runtime.visual.set_nominal(torch.arange(task.cfg.physics.nworld, device=task.device))
        fr = task.runtime.render()['top']
        X.append(fr[:, y0:y0 + s, x0:x0 + s].contiguous()); Y.append(labels(task))
    return torch.cat(X)[:n], torch.cat(Y)[:n]


def errors(pred, y):
    e = (pred - y).abs()
    out = {}
    for k, name, mul in ((0, 'x_mm', 1e3), (1, 'y_mm', 1e3), (2, 'yaw_deg', 180 / math.pi)):
        v = e[:, k].cpu().numpy() * mul
        out[name] = {q: float(np.percentile(v, float(q[1:]))) for q in ('p50', 'p90', 'p99', 'p99.5')}
    xy = torch.linalg.norm(pred[:, :2] - y[:, :2], dim=1).cpu().numpy() * 1e3
    out['xy_mm'] = {q: float(np.percentile(xy, float(q[1:]))) for q in ('p50', 'p90', 'p99', 'p99.5')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=49152)
    ap.add_argument('--n_test', type=int, default=4096)
    ap.add_argument('--res', type=int, default=320)
    ap.add_argument('--size', type=int, default=128, help='CNN input size (the crop is resized)')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--nworld', type=int, default=256)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    t0 = time.time(); dev = torch.device('cuda:0')
    torch.manual_seed(a.seed)
    task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=a.nworld, device='cuda:0', render_rgb=True, visual_dr=True,
                                                    camera_resolution=a.res, measured_camera_cadence=False, task_layout='results/usbc/policy/pick/task_layout.json')))
    box = crop_box(task.runtime, a.res)
    print('crop', box, flush=True)
    X, Y = generate(task, box, a.n)
    torch.manual_seed(12345)
    Xt, Yt = generate(task, box, a.n_test)
    Xn, Yn = generate(task, box, 512, nominal=True)
    t_gen = time.time() - t0
    print(f'generated {len(X)}+{len(Xt)} in {t_gen:.0f} s', flush=True)
    del task; torch.cuda.empty_cache()
    sc = SCALE.to(dev)
    net = PoseNet(3, a.size, width=a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    bs = 128; steps = a.epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=.1)
    pred_of = lambda XX: torch.cat([net(XX[i:i + 256]) for i in range(0, len(XX), 256)]) * sc   # noqa: E731
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(X), device=dev)
        for i in range(len(X) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.smooth_l1_loss(net(X[idx]), Y[idx] / sc, beta=.02)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                e = errors(pred_of(Xt), Yt)
            print(ep, f'loss {loss.item():.4f}', {k: (round(v['p50'], 2), round(v['p99'], 2)) for k, v in e.items()}, flush=True)
    net.eval()
    with torch.no_grad():
        pt, pn = pred_of(Xt), pred_of(Xn)
    Path('runs/perception').mkdir(parents=True, exist_ok=True)
    torch.save({'net': net.state_dict(), 'size': a.size, 'width': a.width, 'n_out': 3, 'scale': sc.cpu(), 'crop': box, 'res': a.res,
                'strip_centre': STRIP_CENTRE, 'yaw0': YAW0}, f'runs/perception/reader_zed{a.tag}.pt')
    rep = {'estimator': 'reader_zed' + a.tag, 'method': __doc__.strip(), 'args': vars(a), 'crop_xy_size_px': box, 'n_train': len(X), 'n_test': len(Xt),
           'held_out_dr_abs_error': errors(pt, Yt), 'held_out_nominal_look_abs_error': errors(pn, Yn), 'generate_s': t_gen, 'total_s': time.time() - t0}
    out = Path(f'results/usbc/policy/pick/reader_zed{a.tag}.json')
    out.write_text(json.dumps(rep, indent=1) + '\n')
    np.savez_compressed(out.with_suffix('.npz'), pred=pt.cpu().numpy(), target=Yt.cpu().numpy())
    print(json.dumps(rep['held_out_dr_abs_error'], indent=1))


if __name__ == '__main__':
    main()
