"""Capture-stage edge fit on the REAL IO path with mock hardware (no CAN, no cameras): docs/real/ZED_EDGEFIT_20261115.md.

RealIO + RealChain.capture_pc on the fake arms and fake cameras of deploy/checks/bimanual_realio_mock.py, with a real
HD1080 capture frame as the ZED raw frame (the dry run has no raw frame; this is the code path the rig runs).
Per config (shadow, use): the 'estimate' event must carry net / start / edge_fit, the fit must use the raw frame
('raw_hd1080', geometry 'real'), and in mode 'use' port_xy must equal the fit (x, y) and yaw the gate net's yaw;
in mode 'shadow' port_xy must equal the net's. A frame without the PC (left-hand-only crop: black frame) must fail the
fit gate in mode 'use' (no fall-back to the net).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. python -m deploy.checks.zed_edgefit_realio [--frame <run>/frames/top/000001_zed_pc.jpg]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
FRAME = '/home/yashchitambar/peg_test/data/real_runs/2026-11-10/full_152717_k4_1/frames/top/000001_zed_pc.jpg'


def one(cfg_path, raw_top, net_device):
    import cv2
    import deploy.cameras as DC
    from deploy.backends import real_loop
    from deploy.bimanual import io_real
    from deploy.bimanual.chain import RealChain
    from deploy.bimanual.runlog import RunLogger, load_run
    from deploy.bimanual.scene import SceneModel
    from deploy.checks.bimanual_realio_mock import FakeArm, FakeCam
    from deploy.run_bimanual import gripper_maps, load_cfg
    cfg = load_cfg(cfg_path)
    snap = ROOT / 'data/sim_ref/2026-10-25_scene'
    raws = {'top': raw_top, 'wrist_right': cv2.imread(str(snap / 'pointed_wrist_right.png')),
            'wrist_left': cv2.imread(str(snap / 'pointed_wrist_left.png'))}
    jm = json.loads((snap / 'pointed.json').read_text())['joints_measured']
    D = real_loop()
    D.RealArm = lambda side, kp, kd, **kw: FakeArm(side, kp, kd, jm[side], **kw)
    DC.ZedTop = lambda: FakeCam(raws['top'])
    DC.WristCam = lambda side: FakeCam(raws['wrist_' + side])
    sc = SceneModel('cpu')
    tmp = Path(tempfile.mkdtemp())
    log = RunLogger(tmp, {'check': 'zed_edgefit_realio'})
    io = io_real.RealIO(cfg, sc.kp.numpy(), sc.kd.numpy(), log=log)
    consts = json.loads((ROOT / cfg['bundle_dir'] / 'configs/chain_constants.json').read_text())
    ch = RealChain(cfg, io, sc, consts, gripper_maps(cfg), log=log, device='cpu', net_device=net_device,
                   policy_path=ROOT / cfg['bundle_dir'] / 'models/insert_KJ.pt', auto='')
    lg = {}
    ep, eR, ok = ch.capture_pc(lg)
    io.hold()
    log.close()
    _, _, ev, _ = load_run(tmp)
    est = [e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc']
    return lg['zed'], ok, ep, est


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame', default=FRAME)
    ap.add_argument('--net-device', default='cuda:0')
    a = ap.parse_args()
    import cv2
    raw = cv2.imread(a.frame)
    fails, out = [], {}

    def check(name, cond, **info):
        print(('ok    ' if cond else 'FAILED') + f' {name} {info}', flush=True)
        if not cond:
            fails.append(name)

    for mode, cfgp in (('shadow', 'deploy/config/bimanual_align10_all_edgefit_capture.json'), ('use', 'deploy/config/bimanual_align10_all_edgefit.json')):
        z, ok, ep, est = one(ROOT / cfgp, raw, a.net_device)
        f = z.get('edge_fit', {})
        out[mode] = {k: z[k] for k in ('port_xy', 'yaw_deg', 'ok', 'est_source')} | dict(net=z.get('net'), fit=f.get('fit_xy'), why=f.get('why'))
        check(f'{mode}: estimate event has net/start/edge_fit', len(est) == 1 and all(k in est[0] for k in ('net', 'start', 'edge_fit')))
        check(f'{mode}: fit on the raw HD1080 frame, geometry real', f.get('frame_source') == 'raw_hd1080' and f.get('geom') == 'real', src=f.get('frame_source'))
        check(f'{mode}: fit gate ok on a real frame', bool(f.get('ok')), why=f.get('why'), rms=f.get('inlier_rms_px'))
        if mode == 'use':
            check('use: port_xy = fit x, y', np.allclose(z['port_xy'], f['fit_xy'], atol=1e-6), port=z['port_xy'], fit=f['fit_xy'])
            check('use: yaw = gate net yaw', abs(z['yaw_deg'] - z['net']['yaw_deg']) < 1e-4)
            check('use: port pose = port_frame(fit)', np.allclose(ep[0, :2].cpu().numpy(), f['fit_xy'], atol=1e-6))
            check('use: ok = net gate and fit gate', ok == (z['net']['ok'] and f['ok']))
        else:
            check('shadow: port_xy = net', np.allclose(z['port_xy'], z['net']['port_xy'], atol=1e-9))
    # no PC in view: the fit gate must reject (the net gate may or may not; the result must not be ok)
    blank = (raw * 0 + 40).astype(np.uint8)
    z, ok, ep, est = one(ROOT / 'deploy/config/bimanual_align10_all_edgefit.json', blank, a.net_device)
    out['blank'] = dict(ok=ok, why=z['edge_fit'].get('why'), net_ok=z['net']['ok'])
    check('use, uniform frame: capture rejected, fit gate rejects', (not ok) and (not z['edge_fit']['ok']), why=z['edge_fit'].get('why'))
    print(json.dumps(out, indent=1, default=str))
    print('FAILED:', fails or 'none', flush=True)
    raise SystemExit(1 if fails else 0)


if __name__ == '__main__':
    main()
