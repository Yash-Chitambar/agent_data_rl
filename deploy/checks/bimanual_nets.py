"""Every net of the bimanual bundle, loaded the way the runner loads it: input/output shapes and dtypes, finite
outputs, the expected input size against the checkpoint's own metadata (training script values), and latency on
CPU and GPU (p50 / p99 over --reps calls, batch 1, after warm-up). Optional --real: one real frame pair of the
capture pose (data/sim_ref/2026-10-25_scene/pointed_*: ZED + right wrist at the ZED net's capture pose) through
the real preprocessing into the ZED estimator and its gates.

    PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_nets [--devices cpu cuda:0] [--real] [--out results/...json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
P = ROOT / 'runs/perception'


def timeit(fn, dev, reps):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {'p50_ms': float(np.percentile(ts, 50)), 'p99_ms': float(np.percentile(ts, 99)), 'max_ms': float(np.max(ts))}


def check_device(dev, reps, cfg):
    from deploy.bimanual.io_sim import eval_chain
    from blind_insertion.perception.zed_estimator import ZedEstimator
    from blind_insertion.perception.reader_zed import ReaderZedEstimator
    from blind_insertion.pipeline.pick_servo import load_servo_net, run_net
    from blind_insertion.eval_insertion import load_insertion_policy
    from deploy.bimanual.insert import contact_summary
    EC = eval_chain()
    out = {}
    g = torch.Generator().manual_seed(0)
    img = torch.randint(0, 256, (1, 320, 320, 3), dtype=torch.uint8, generator=g).to(dev)
    cap = json.loads((ROOT / 'blind_insertion/configs/zed_capture_pose.json').read_text())['model_frame_rad']
    arms = (torch.tensor([cap['left']], device=dev), torch.tensor([cap['right']], device=dev), torch.tensor([.0475], device=dev))
    ca = cfg['chain_args']
    z = ZedEstimator(str(dev), pose=str(P / 'zed.pt'), presence=str(P / f'{ca["zed_gate"]}.pt'))
    xy, yaw, ok, info = z(img, arms=arms)
    ck = torch.load(P / 'zed.pt', map_location='cpu')
    out['zed+gate'] = dict(input=list(img.shape), dtype=str(img.dtype), out_shapes=[list(xy.shape), list(yaw.shape), list(ok.shape)],
                           finite=bool(torch.isfinite(xy).all() and torch.isfinite(yaw).all()), net_size=ck['size'],
                           crop=list(ck.get('zed_crop', ())), **timeit(lambda: z(img, arms=arms), dev, reps))
    rz = ReaderZedEstimator(str(P / f'{cfg["pick"]["reader_estimator"]}.pt'), str(dev))
    xy, yaw, ok = rz.estimate(img, arms[0], arms[1], arms[2])
    out['reader_zed_cap+gate'] = dict(input=list(img.shape), res_expected=rz.res, crop=list(rz.crop), out_shapes=[list(xy.shape), list(yaw.shape)],
                                     finite=bool(torch.isfinite(xy).all()), **timeit(lambda: rz.estimate(img, arms[0], arms[1], arms[2]), dev, reps))
    pc = cfg['pick']
    for k in ('hover_net', 'near_net', 'rot_net'):
        net, sc, n_out = load_servo_net(P / f'{pc[k]}.pt', dev)
        ck = torch.load(P / f'{pc[k]}.pt', map_location='cpu')
        y = run_net(net, sc, img)
        out[pc[k]] = dict(input=list(img.shape), res_expected=ck.get('res'), net_size=ck['size'], n_out=n_out, out_shape=list(y.shape),
                          finite=bool(torch.isfinite(y[:, :n_out]).all()), **timeit(lambda: run_net(net, sc, img), dev, reps))
    for name in (ca['wrist_net'], ca['fine_net'], ca['inhand_net']):
        net, sc, mode = EC.wrist_loader(name, dev)
        ck = torch.load(P / f'{name}.pt', map_location='cpu')

        def f():
            x = F.interpolate(img.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
            with torch.no_grad():
                return net(x) * sc
        y = f()
        out[name] = dict(input='320 -> area 192', mode=mode, net_size=ck['size'], wrist_size=ck.get('wrist_size'), out_shape=list(y.shape),
                         finite=bool(torch.isfinite(y).all()), **timeit(f, dev, reps))
    pol = load_insertion_policy(str(ROOT / 'runs/chain/P4-insert-best_KJ.pt'), str(dev))
    hist = torch.randn(1, pol.meta['history'], pol.obs_dim, device=dev)
    a = pol(hist)
    out['insert_KJ'] = dict(input=list(hist.shape), out_shape=list(a.shape), finite=bool(torch.isfinite(a).all()),
                            in_range=bool((a.abs() <= 1).all()), mask_idx=list(pol.mask_idx), **timeit(lambda: pol(hist), dev, reps))
    out['insert_KJ+contact_head'] = timeit(lambda: (pol(hist), contact_summary(pol, hist)), dev, reps)
    return out


def real_capture(dev, cfg, snap=ROOT / 'data/sim_ref/2026-10-25_scene', tag='pointed'):
    """A real ZED + right-wrist frame at the capture pose through the real preprocessing and the ZED estimator."""
    import cv2
    from blind_insertion.assets.real_camera import load_spec, real_policy_image
    from blind_insertion.perception.zed_estimator import ZedEstimator
    from blind_insertion.scenes.physical_rig import real_top_policy_image
    sys.path.append(str(ROOT / 'scripts/real'))
    import rig
    meta = json.loads((snap / f'{tag}.json').read_text())
    top = real_top_policy_image(cv2.imread(str(snap / f'{tag}_zed_left.png')), 320)
    wr = real_policy_image(cv2.imread(str(snap / f'{tag}_wrist_right.png')), load_spec(rig.SPEC), 320, 'right')
    jm = meta['joints_measured']
    qb = {s: np.asarray(jm[s][:6]) + rig.joint_offsets(s) for s in ('right', 'left')}
    from deploy import conventions as C
    gm = C.GripperMap(**{k: cfg['gripper']['right'][k] for k in ('open', 'close', 'm_open', 'g_reader', 'm_reader')})
    finger = gm.finger_m(jm['right'][6])
    z = ZedEstimator(str(dev), pose=str(P / 'zed.pt'), presence=str(P / f'{cfg["chain_args"]["zed_gate"]}.pt'))
    t = lambda x: torch.tensor(np.asarray(x)[None], dtype=torch.float32, device=dev)  # noqa: E731
    xy, yaw, ok, info = z(torch.as_tensor(top, device=dev)[None], arms=(t(qb['left']), t(qb['right']), t([finger])))
    return dict(snapshot=str(snap.relative_to(ROOT)), tag=tag, top_shape=list(top.shape), wrist_shape=list(wr.shape),
                port_xy=xy[0].tolist(), yaw_deg=float(np.degrees(float(yaw[0]))), ok=bool(ok[0]),
                **{k: float(v[0]) for k, v in info.items()}, right_finger_m=finger)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--devices', nargs='+', default=['cpu'] + (['cuda:0'] if torch.cuda.is_available() else []))
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--real', action='store_true')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    cfg = json.loads((ROOT / 'deploy/config/bimanual.json').read_text())
    torch.set_num_threads(4)
    res = {'host': __import__('socket').gethostname(), 'torch_threads': torch.get_num_threads()}
    for d in a.devices:
        dev = torch.device(d)
        res[d] = check_device(dev, a.reps, cfg)
        if d.startswith('cuda'):
            res[d]['gpu'] = torch.cuda.get_device_name(dev)
    if a.real:
        res['real_capture'] = real_capture(torch.device(a.devices[-1]), cfg)
    ok = all(v.get('finite', True) for d in a.devices for v in res[d].values() if isinstance(v, dict))
    res['all_finite'] = ok
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
