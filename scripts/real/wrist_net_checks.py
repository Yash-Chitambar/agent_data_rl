"""Second checks for the r10 nets: sim, rotation, port vision agreement, timing, loader. No motion.

    PYTHONPATH=. python scripts/real/wrist_net_checks.py --frames <wrist_diag frames npz> --port-vision <dir of a real/port-vision checkout> --out results/real/wrist_nets/checks.json
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/real'))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import wrist_net_real_eval as W  # noqa: E402

NETS = {'r10f_ft_abs': 'runs/perception_real/wrist_r10f_ft_abs.pt', 'r10w_ft_abs': 'runs/perception_real/wrist_r10w_ft_abs.pt'}
CACHE = {'r10f_ft_abs': 'runs/perception_real/cache/sim_f.npz', 'r10w_ft_abs': 'runs/perception_real/cache/sim_w.npz'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--port-vision', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dev = 'cuda:0'
    rs = W.rows()
    co = W.collect_rows(rs)
    Y = np.array([r['p_mm'] + r['rv_deg'] for r in co])
    tr, ho, _ = W.split_collect(len(co))
    real = np.stack([W.read_rgb(r['image']) for r in co])
    F = np.load(a.frames)
    # port vision, read from the other agent's checkout; one depth per run as its doc recommends
    spec = importlib.util.spec_from_file_location('port_vision', Path(a.port_vision) / 'deploy/bimanual/port_vision.py')
    pv = importlib.util.module_from_spec(spec)
    sys.modules['port_vision'] = pv
    spec.loader.exec_module(pv)
    def vis(sel):
        raws = [cv2.imread(str(ROOT / r['raw'])) if r.get('raw') else None for r in sel]
        z = pv.plug_depth([f for f in raws if f is not None])
        ms = [pv.measure(f, plug_z_mm=z) if f is not None else None for f in raws]
        return np.array([[m.across_mm, m.up_mm, m.gap_mm] if m is not None and m.ok else [np.nan] * 3 for m in ms], float)
    Vh = vis([co[i] for i in ho])
    st = W.stops(rs)
    Vs = {k: vis(v) for k, v in st.items()}
    out = {}
    for name, path in NETS.items():
        net, sc, ck = W.load_net(path, dev)
        r = {}
        # sim: frames never trained on (last 2048 of the cache), and sim at the real poses
        d = np.load(CACHE[name])
        Xt, Yt = d['X'][-2048:], d['Y'][-2048:]
        with torch.no_grad():
            P = torch.cat([net(torch.as_tensor(Xt[i:i + 256], device=dev)) * sc for i in range(0, 2048, 256)]).cpu().numpy()
        e = np.abs(P - Yt)
        r['sim_heldout_dr'] = dict(n=2048, p50=np.percentile(e, 50, 0).round(2).tolist(), p99=np.percentile(e, 99, 0).round(2).tolist(),
                                   slope=[round(W.slope(Yt[:, i], P[:, i]), 3) for i in range(6)], order='across, up, gap mm, rx, ry, rz deg')
        for c in ('sim_inhand_pc', 'sim_old', 'sim_inhand', 'sim_pc'):
            Ps = W.predict(net, sc, ck, F[c], dev)
            r[c + '_at_real_poses'] = dict(n=150, **{ax: dict(slope=round(W.slope(Y[:, i], Ps[:, i]), 3), ci=W.boot(Y[:, i], Ps[:, i], n=1000), resid_sd=round(float((Ps[:, i] - Y[:, i]).std()), 2)) for i, ax in enumerate(W.AXES)})
        # rotation on the hold-out (differences within the run are what the labels support)
        Pr = W.predict(net, sc, ck, real[ho], dev)
        r['rotation_holdout'] = {k: dict(slope=round(W.slope(Y[ho, i], Pr[:, i]), 3), ci=W.boot(Y[ho, i], Pr[:, i], n=1000), resid_sd_deg=round(float((Pr[:, i] - Y[ho, i]).std()), 2)) for k, i in (('rx', 3), ('ry', 4), ('rz', 5))}
        # against port vision, frame by frame
        ok = np.isfinite(Vh).all(1)
        diff = Pr[ok, :3] - Vh[ok]
        near = Y[ho][ok, 2] < 15
        r['vs_vision_holdout'] = dict(n=int(ok.sum()), of=len(ho), mean=diff.mean(0).round(2).tolist(), sd=diff.std(0).round(2).tolist(), p95_abs=np.percentile(np.abs(diff - diff.mean(0)), 95, 0).round(2).tolist(),
                                      near_n=int(near.sum()), near_mean=diff[near].mean(0).round(2).tolist(), near_sd=diff[near].std(0).round(2).tolist(),
                                      resid_corr_with_label=[round(float(np.corrcoef(Pr[ok, i] - Y[ho][ok, i], Vh[ok, i] - Y[ho][ok, i])[0, 1]), 2) for i in range(3)])
        for k, sel in st.items():
            Pst = W.predict(net, sc, ck, np.stack([W.read_rgb(x['image']) for x in sel]), dev)
            m = np.isfinite(Vs[k]).all(1)
            r.setdefault('stops_vs_vision', {})[k] = dict(n=int(m.sum()), net=Pst[m, :3].mean(0).round(2).tolist(), vision=Vs[k][m].mean(0).round(2).tolist())
        # timing, one frame, as the chain calls it: 320 px frame -> area resize -> net
        one = torch.as_tensor(real[:1], device=dev)
        ts = []
        for k in range(120):
            t = time.perf_counter()
            with torch.no_grad():
                y = net(W.net_input(one, ck)) * sc
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
        r['ms_gpu_one_frame'] = dict(device=torch.cuda.get_device_name(0), p50=round(float(np.percentile(ts[20:], 50) * 1e3), 2), p99=round(float(np.percentile(ts[20:], 99) * 1e3), 2))
        ncpu = W.load_net(path, 'cpu')
        ts = []
        one_c = torch.as_tensor(real[:1])
        for k in range(20):
            t = time.perf_counter()
            with torch.no_grad():
                ncpu[0](W.net_input(one_c, ncpu[2]))
            ts.append(time.perf_counter() - t)
        r['ms_cpu_one_frame_p50'] = round(float(np.percentile(ts[3:], 50) * 1e3), 1)
        out[name] = r
    # the chain's own loader, through the name the config gives
    sys.path.insert(0, str(ROOT))
    from scripts.eval import eval_chain as EC
    for nm in ('../perception_real/wrist_r10f_ft_abs', '../perception_real/wrist_r10w_ft_abs'):
        n, s, mode = EC.wrist_loader(nm, torch.device(dev))
        im = torch.nn.functional.interpolate(torch.as_tensor(real[:2], device=dev).permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
        with torch.no_grad():
            y = n(im) * s
        out.setdefault('chain_loader', {})[nm] = dict(mode=mode, out_frame0=y[0].cpu().numpy().round(2).tolist(), label_frame0=Y[0].round(2).tolist())
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
