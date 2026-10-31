"""Wrist-camera refinement of the reader pose at the pick's pre-grasp hover (RGB only).

The ZED estimate puts the hand at the pre-grasp hover (6 cm back along the tool axis, 10 cm up). From
there the right wrist camera sees the reader close up. This CNN predicts the correction of the ZED
estimate: true minus estimated body centre (m) in the estimated reader's horizontal frame (along the
plug, across) and the yaw error (rad). The hand is off the hover by +/-10 mm / +/-3 deg (tracking); the estimate is re-expressed relative to
the hand's measured (encoder FK, offset errors included) pose, horizontally (rebase_estimate). Training hovers come from estimates perturbed like the ZED net
(70 %: xy N(0, 2.5 mm), yaw N(0, 2 deg); 30 % gross: xy uniform within 40 mm, yaw U(+/-8 deg)).
Full visual DR (incl. wrist-camera pose error), left arm parked, gripper at the 24 mm approach opening.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=.:scripts/perception MUJOCO_GL=egl .venv/bin/python scripts/perception/train_reader_wrist.py
Writes runs/perception/reader_wrist{tag}.pt, results/usbc/policy/pick/reader_wrist{tag}.json
"""
import argparse, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import mujoco_warp as mjw
import numpy as np
import torch
import torch.nn.functional as F

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_pick import PickTask, PickTaskCfg, PickExpert
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.perception.nets import PoseNet
from blind_insertion.scenes import reader_grasp as G

SCALE = torch.tensor([.02, .02, .08])


def yaw_of(R):
    plug = -R[:, :2, 2]
    return torch.atan2(plug[:, 1], plug[:, 0])


def body_centre(task, p, R):
    return p + R[:, :, 2] * float(task.lay.variants[0].peg_boxes[0][0][2])


def sample_errors(n, dev, gen):
    gross = torch.rand(n, device=dev, generator=gen) < .3
    r = torch.where(gross, .04 * torch.sqrt(torch.rand(n, device=dev, generator=gen)), (torch.randn(n, device=dev, generator=gen) * .0025).abs())
    ang = torch.rand(n, device=dev, generator=gen) * 2 * math.pi
    dxy = torch.stack([r * torch.cos(ang), r * torch.sin(ang)], 1)
    dyaw = torch.where(gross, (torch.rand(n, device=dev, generator=gen) * 2 - 1) * math.radians(8), torch.randn(n, device=dev, generator=gen) * math.radians(2))
    return dxy, dyaw


def estimated_pose(task, dxy, dyaw):
    """Estimated reader pose = true pose moved by dxy and turned by dyaw about the body centre."""
    p, R = task.reader_pose()
    c = body_centre(task, p, R)
    n = len(p); z, o = torch.zeros(n, device=p.device), torch.ones(n, device=p.device)
    Rz = torch.stack([torch.stack([torch.cos(dyaw), -torch.sin(dyaw), z], -1), torch.stack([torch.sin(dyaw), torch.cos(dyaw), z], -1), torch.stack([z, z, o], -1)], -2)
    Re = Rz @ R
    ce = c.clone(); ce[:, :2] += dxy
    return ce - Re[:, :, 2] * float(task.lay.variants[0].peg_boxes[0][0][2]), Re


def correction_label(task, p_est, R_est):
    """True minus estimated body centre in the estimate's frame (along the plug, across) and yaw error."""
    p, R = task.reader_pose()
    d = (body_centre(task, p, R) - body_centre(task, p_est, R_est))[:, :2]
    ye = yaw_of(R_est)
    along = torch.stack([torch.cos(ye), torch.sin(ye)], 1); across = torch.stack([-torch.sin(ye), torch.cos(ye)], 1)
    dyaw = torch.atan2(torch.sin(yaw_of(R) - ye), torch.cos(yaw_of(R) - ye))
    return torch.stack([(d * along).sum(1), (d * across).sum(1), dyaw], 1)


def apply_correction(task, p_est, R_est, corr):
    """Inverse of correction_label: corrected reader pose."""
    ye = yaw_of(R_est)
    along = torch.stack([torch.cos(ye), torch.sin(ye)], 1); across = torch.stack([-torch.sin(ye), torch.cos(ye)], 1)
    dxy = corr[:, :1] * along + corr[:, 1:2] * across
    p_new = p_est.clone()
    n = len(p_est); z, o = torch.zeros(n, device=p_est.device), torch.ones(n, device=p_est.device)
    a = corr[:, 2]
    Rz = torch.stack([torch.stack([torch.cos(a), -torch.sin(a), z], -1), torch.stack([torch.sin(a), torch.cos(a), z], -1), torch.stack([z, z, o], -1)], -2)
    bz = float(task.lay.variants[0].peg_boxes[0][0][2])
    c = body_centre(task, p_est, R_est); c[:, :2] += dxy
    R_new = Rz @ R_est
    return c - R_new[:, :, 2] * bz, R_new


def heading(R):
    """Horizontal heading of the tool axis (site z) of a hand orientation."""
    return torch.atan2(R[:, 1, 2], R[:, 0, 2])


def rebase_estimate(task, p_est, R_est, p_hover, R_hover, p_hand, R_hand):
    """Re-express an estimate made for the PLANNED hover relative to the hand's MEASURED pose, in the
    horizontal plane only (shift and turn about the vertical; the reader stays flat): what the wrist
    camera sees is the reader relative to where the hand actually is."""
    dyaw = heading(R_hand) - heading(R_hover)
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    n = len(p_est); z, o = torch.zeros(n, device=p_est.device), torch.ones(n, device=p_est.device)
    Rz = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
    bz = float(task.lay.variants[0].peg_boxes[0][0][2])
    cen = body_centre(task, p_est, R_est)
    rel = cen - p_hover; rel[:, 2] = 0
    cen_new = p_hand + (Rz @ rel[..., None])[..., 0]; cen_new[:, 2] = cen[:, 2]
    R_new = Rz @ R_est
    return cen_new - R_new[:, :, 2] * bz, R_new


def measured_tcp(task):
    """Hand pose from the encoders (believed joint angles incl. the encoder-offset error)."""
    return task.chain.fk(task._measured_model_angles()[:, 0])


def hover_q(task, p_est, R_est):
    """IK of the pre-grasp hover for the estimated pose (as PickExpert.reset)."""
    ex = PickExpert(task, reader_pose_fn=lambda: (p_est, R_est)); ex.reset()
    return ex.q_pre, ex.ik_ok


def generate(task, n, gen, size=192, hover_err=(.010, 3.)):
    """hover_err: the hand is off the planned hover by U(+/-m) per axis and U(+/-deg) about a random
    axis (PD tracking error); labels are relative to the hand's measured (encoder FK) pose."""
    X, Y = [], []
    rt = task.runtime
    while sum(len(x) for x in X) < n:
        task.reset()
        N = task.cfg.physics.nworld
        dxy, dyaw = sample_errors(N, task.device, gen)
        p_est, R_est = estimated_pose(task, dxy, dyaw)
        ex = PickExpert(task, reader_pose_fn=lambda: (p_est, R_est)); ex.reset()
        p_h, R_h = ex.p_pre, ex.R_g
        dp = (torch.rand(N, 3, device=task.device, generator=gen) * 2 - 1) * hover_err[0]
        ax = torch.randn(N, 3, device=task.device, generator=gen); ax = ax / ax.norm(dim=1, keepdim=True)
        ang = (torch.rand(N, device=task.device, generator=gen) * 2 - 1) * math.radians(hover_err[1])
        R_t = K.axis_angle_mat(ax, ang) @ R_h
        q, pe, re = task.chain.ik(p_h + dp, R_t, ex.q_pre, iters=60)
        ok = ex.ik_ok & (pe < 1e-3)
        rt.q[:, task.qadr] = q
        with wp.ScopedDevice(task.cfg.physics.device):
            mjw.kinematics(rt.m, rt.d)
        p_m, R_m = measured_tcp(task)
        p_eb, R_eb = rebase_estimate(task, p_est, R_est, p_h, R_h, p_m, R_m)
        fr = rt.render()['wrist_cam'][ok]
        fr = F.interpolate(fr.permute(0, 3, 1, 2).float(), size=(size, size), mode='area').round().clamp(0, 255).byte().permute(0, 2, 3, 1)
        X.append(fr.contiguous().cpu()); Y.append(correction_label(task, p_eb, R_eb)[ok].cpu())
    return torch.cat(X)[:n], torch.cat(Y)[:n]


def errors(pred, y):
    out = {}
    for k, name, mul in ((0, 'along_mm', 1e3), (1, 'across_mm', 1e3), (2, 'yaw_deg', 180 / math.pi)):
        v = (pred[:, k] - y[:, k]).abs().cpu().numpy() * mul
        out[name] = {q: float(np.percentile(v, float(q[1:]))) for q in ('p50', 'p90', 'p99', 'p99.5')}
    v = torch.linalg.norm(pred[:, :2] - y[:, :2], dim=1).cpu().numpy() * 1e3
    out['xy_mm'] = {q: float(np.percentile(v, float(q[1:]))) for q in ('p50', 'p90', 'p99', 'p99.5')}
    out['xy_max_mm'] = float(v.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40960); ap.add_argument('--n_test', type=int, default=4096)
    ap.add_argument('--res', type=int, default=320); ap.add_argument('--size', type=int, default=192)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--width', type=int, default=32)
    ap.add_argument('--tag', default=''); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time(); dev = torch.device('cuda:0')
    torch.manual_seed(a.seed); gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=256, device='cuda:0', render_rgb=True, visual_dr=True, camera_resolution=a.res,
                                                    measured_camera_cadence=False, task_layout='results/usbc/policy/pick/task_layout.json')))
    X, Y = generate(task, a.n, gen, a.size)
    gen.manual_seed(999); torch.manual_seed(999)
    Xt, Yt = generate(task, a.n_test, gen, a.size)
    Y, Yt = Y.to(dev), Yt.to(dev)
    t_gen = time.time() - t0
    print(f'generated {len(X)}+{len(Xt)} in {t_gen:.0f} s', flush=True)
    del task; torch.cuda.empty_cache()
    sc = SCALE.to(dev)
    net = PoseNet(3, a.size, width=a.width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    bs = 128; steps = a.epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=.1)
    pred_of = lambda XX: torch.cat([net(XX[i:i + 128].to(dev)) for i in range(0, len(XX), 128)]) * sc   # noqa: E731
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(X))
        for i in range(len(X) // bs):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.smooth_l1_loss(net(X[idx].to(dev, non_blocking=True)), Y[idx.to(dev)] / sc, beta=.02)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == a.epochs - 1:
            net.eval()
            with torch.no_grad():
                e = errors(pred_of(Xt), Yt)
            print(ep, f'loss {loss.item():.4f}', {k: (round(v['p50'], 2), round(v['p99'], 2)) for k, v in e.items() if isinstance(v, dict)}, flush=True)
    net.eval()
    with torch.no_grad():
        pt = pred_of(Xt)
    torch.save({'net': net.state_dict(), 'size': a.size, 'width': a.width, 'n_out': 3, 'scale': sc.cpu(), 'res': a.res,
                'camera': 'wrist_cam', 'hover': {'back_m': G.APPROACH_BACK, 'up_m': G.APPROACH_UP}}, f'runs/perception/reader_wrist{a.tag}.pt')
    gross = (torch.linalg.norm(Yt[:, :2], dim=1) > .005)
    rep = {'estimator': 'reader_wrist' + a.tag, 'method': __doc__.strip(), 'args': vars(a), 'n_train': len(X), 'n_test': len(Xt),
           'heldout_abs_error_of_correction': errors(pt, Yt), 'heldout_gross_start_errors': errors(pt[gross], Yt[gross]),
           'n_gross': int(gross.sum()), 'generate_s': t_gen, 'total_s': time.time() - t0}
    out = Path(f'results/usbc/policy/pick/reader_wrist{a.tag}.json')
    out.write_text(json.dumps(rep, indent=1) + '\n')
    np.savez_compressed(out.with_suffix('.npz'), pred=pt.cpu().numpy(), target=Yt.cpu().numpy())
    print(json.dumps(rep['heldout_abs_error_of_correction'], indent=1))


if __name__ == '__main__':
    main()
