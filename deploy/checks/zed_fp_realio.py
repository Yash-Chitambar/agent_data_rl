"""Capture-stage zed_fp block on the REAL IO path with mock hardware (no CAN, no ZED): docs/real/FOUNDATIONPOSE_20261116.md.

RealIO + RealChain.capture_pc with the fake arms of deploy/checks/bimanual_realio_mock.py. The ZED is a _Grabber with
the real deploy.cameras.ZedTop._grab code on a fake SDK (sl.Camera / sl.Mat / VIEW.LEFT / VIEW.RIGHT) that serves a real
rectified HD1080 stereo pair (data/zed_gt/spot1/capture_reader, arm joints of that capture). This exercises the
right-view hook of deploy/bimanual/zed_fp.py exactly as on the rig.

Checks:
  shadow (bimanual_align12_wide_k7_fp_shadow.json): the saved left.png / right.png equal the served pair pixel for pixel,
    request.json has prior / net / edge_fit, the chain pose equals the edge fit (= bimanual_align11_wide_k7), and
    FoundationPose (fp env, server.py --once) gives an ok result within 4 mm of the edge fit;
  use (bimanual_align12_wide_k7_fp_use.json) with a live server.py on the run folder: port_xy = FP x, y, yaw = gate yaw,
    ok = gates;
  use without a server (timeout 3 s): the capture is rejected (no fall-back).
  block off (bimanual_align11_wide_k7.json): no zed_fp folder, no hook installed.

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -m deploy.checks.zed_fp_realio [--fp-python /home/yashchitambar/fp_work/fp_python]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
PAIR = Path('/home/yashchitambar/peg_test/data/zed_gt/spot1/capture_reader')
FP_PY = '/home/yashchitambar/fp_work/fp_python'


class _Mat:
    def __init__(self):
        self.data = None

    def get_data(self):
        return self.data


class _FakeZedCam:
    def __init__(self, left, right):
        a = np.full(left.shape[:2] + (1,), 255, np.uint8)
        self.views = {0: np.concatenate([left, a], 2), 1: np.concatenate([right, a], 2)}
        self.n = 0

    def grab(self):
        time.sleep(1 / 30)
        self.n += 1
        return 0

    def retrieve_image(self, mat, view):
        mat.data = self.views[view].copy()

    def close(self):
        pass


def fake_zed(left, right):
    import deploy.cameras as DC

    class FakeZed(DC._Grabber):
        _grab = DC.ZedTop._grab                  # the rig code

        def __init__(self):
            super().__init__('top')
            self.sl = SimpleNamespace(VIEW=SimpleNamespace(LEFT=0, RIGHT=1), ERROR_CODE=SimpleNamespace(SUCCESS=0), Mat=_Mat)
            self.cam = _FakeZedCam(left, right)
            self.mat = _Mat()
            self.th.start()

        def close(self):
            super().close()
    return FakeZed


def one(cfg, left, right, joints, net_device, out_dir, reader=False):
    import cv2
    import deploy.cameras as DC
    from deploy.backends import real_loop
    from deploy.bimanual import io_real
    from deploy.bimanual.chain import RealChain
    from deploy.bimanual.runlog import RunLogger, load_run
    from deploy.bimanual.scene import SceneModel
    from deploy.checks.bimanual_realio_mock import FakeArm, FakeCam
    from deploy.run_bimanual import gripper_maps
    snap = ROOT / 'data/sim_ref/2026-10-25_scene'
    D = real_loop()
    D.RealArm = lambda side, kp, kd, **kw: FakeArm(side, kp, kd, joints[side], **kw)
    DC.ZedTop = fake_zed(left, right)
    DC.WristCam = lambda side: FakeCam(cv2.imread(str(snap / f'pointed_wrist_{side}.png')))
    sc = SceneModel('cpu')
    log = RunLogger(out_dir, {'check': 'zed_fp_realio'})
    io = io_real.RealIO(cfg, sc.kp.numpy(), sc.kd.numpy(), log=log)
    consts = json.loads((ROOT / cfg['bundle_dir'] / 'configs/chain_constants.json').read_text())
    ch = RealChain(cfg, io, sc, consts, gripper_maps(cfg), log=log, device='cpu', net_device=net_device,
                   policy_path=ROOT / cfg['bundle_dir'] / 'models/insert_KJ.pt', auto='')
    lg = {}
    try:
        ep, eR, ok = ch.capture_pc(lg)
        if reader:
            ch.reader_frame()
    finally:
        io.hold()
        hooked = bool(getattr(io.cams.get('top'), 'fp_hooked', False))
        for c in io.cams.values():
            c.close()
        log.close()
    return lg['zed'], ok, ep, hooked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net-device', default='cuda:0')
    ap.add_argument('--fp-python', default=FP_PY)
    a = ap.parse_args()
    import cv2
    from deploy.run_bimanual import load_cfg
    left, right = cv2.imread(str(PAIR) + '_zed_left.png'), cv2.imread(str(PAIR) + '_zed_right.png')
    meta = json.loads(PAIR.with_suffix('.json').read_text())
    jm = {s: meta['joints_measured'][s] for s in ('right', 'left')}
    fails, out = [], {}
    env = dict(os.environ)

    def check(name, cond, **info):
        print(('ok    ' if cond else 'FAILED') + f' {name} {info}', flush=True)
        if not cond:
            fails.append(name)

    # 0. block off: nothing saved, no hook
    t0 = Path(tempfile.mkdtemp())
    z, ok, ep, hooked = one(load_cfg(ROOT / 'deploy/config/bimanual_align11_wide_k7.json'), left, right, jm, a.net_device, t0)
    check('off: no zed_fp folder, no hook', not (t0 / 'zed_fp').exists() and not hooked and 'zed_fp' not in z)
    base_xy = z['port_xy']

    # 1. shadow
    t1 = Path(tempfile.mkdtemp())
    z, ok, ep, hooked = one(load_cfg(ROOT / 'deploy/config/bimanual_align12_wide_k7_fp_shadow.json'), left, right, jm, a.net_device, t1,
                            reader=True)
    rd = sorted((t1 / 'zed_fp').glob('zed_reader*'))
    rq = json.loads((rd[0] / 'request.json').read_text()) if rd else {}
    check('shadow: reader frame pairs saved (save_reader)', len(rd) == 1 and rq.get('object') == 'reader' and rq.get('n_pairs') == 5
          and (rd[0] / 'left_4.png').exists(), dirs=[x.name for x in rd], n_pairs=rq.get('n_pairs'))
    f = z.get('zed_fp', {})
    d = t1 / f.get('dir', 'missing')
    check('shadow: hook installed, pair saved', hooked and (d / 'left.png').exists() and (d / 'right.png').exists(), dir=f.get('dir'))
    if (d / 'left.png').exists():
        check('shadow: saved left/right = served pair (lossless)', np.array_equal(cv2.imread(str(d / 'left.png')), left)
              and np.array_equal(cv2.imread(str(d / 'right.png')), right))
    req = json.loads((d / 'request.json').read_text()) if (d / 'request.json').exists() else {}
    check('shadow: n_pairs pairs saved (config 5)', req.get('n_pairs') == 5 and (d / 'left_4.png').exists() and (d / 'right_4.png').exists(),
          n_pairs=req.get('n_pairs'))
    check('shadow: request has prior, net, edge_fit, geom real', all(k in req for k in ('prior', 'net', 'edge_fit')) and req.get('geom') == 'real')
    check('shadow: chain pose = edge fit (= wide_k7)', np.allclose(z['port_xy'], base_xy, atol=1e-9) and np.allclose(z['port_xy'], z['edge_fit']['fit_xy'], atol=1e-9),
          port=z['port_xy'], base=base_xy)
    p = subprocess.run([a.fp_python, str(ROOT / 'scripts/perception/zed_fp/server.py'), '--root', str(t1), '--once', '--geoms', 'real', '--all-pairs'],
                       capture_output=True, text=True, env=env)
    res = json.loads((d / 'result.json').read_text()) if (d / 'result.json').exists() else {}
    dxy = float(np.hypot(*((np.array(res['pose'][:2]) - z['edge_fit']['fit_xy']) * 1e3))) if res.get('ok') else float('nan')
    check('shadow: FP on the saved pair ok, |FP - edge fit| <= 4 mm', bool(res.get('ok')) and dxy <= 4.0, dxy_mm=dxy,
          score=res.get('score'), iou=res.get('iou_fp_sam'), stderr=p.stderr[-300:] if not res else '')
    check('shadow: median over the 5 pairs', len(res.get('pairs', [])) == 5, n=len(res.get('pairs', [])))
    out['shadow'] = dict(port=z['port_xy'], fp=res.get('pose'), fp_median=res.get('pose_median'), dxy_edge_mm=dxy, fp_time_s=res.get('time_s'))

    # 2. use, live server
    t2 = Path(tempfile.mkdtemp())
    srv = subprocess.Popen([a.fp_python, str(ROOT / 'scripts/perception/zed_fp/server.py'), '--root', str(t2), '--geoms', 'real'],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    ready = threading.Event()

    def watch():
        for line in srv.stdout:
            if 'models loaded' in line:
                ready.set()
    threading.Thread(target=watch, daemon=True).start()
    ready.wait(300)
    try:
        z, ok, ep, hooked = one(load_cfg(ROOT / 'deploy/config/bimanual_align12_wide_k7_fp_use.json'), left, right, jm, a.net_device, t2)
    finally:
        srv.terminate()
    f = z.get('zed_fp', {})
    r = f.get('result') or {}
    check('use: FP result received, gate ok', bool(f.get('ok')), why=f.get('why'), status=f.get('status'))
    if r.get('ok'):
        check('use: port_xy = FP x, y', np.allclose(z['port_xy'], r['pose'][:2], atol=1e-6), port=z['port_xy'], fp=r['pose'][:2])
        check('use: yaw = gate net yaw', abs(z['yaw_deg'] - z['net']['yaw_deg']) < 1e-4, yaw=z['yaw_deg'], net=z['net']['yaw_deg'])
        check('use: port pose = port_frame(FP)', np.allclose(ep[0, :2].cpu().numpy(), r['pose'][:2], atol=1e-6))
        check('use: ok = net gate and edge-fit gate and FP gate', ok == bool(z['net']['ok'] and z['edge_fit']['ok'] and f['ok']), ok=ok)
    out['use'] = dict(port=z['port_xy'], ok=ok, d_edge_mm=f.get('d_edge_mm'), d_gate_mm=f.get('d_gate_mm'), wait_s=f.get('seconds'))

    # 3. use, no server: timeout -> rejected
    cfg = load_cfg(ROOT / 'deploy/config/bimanual_align12_wide_k7_fp_use.json')
    cfg['zed_fp']['timeout_s'] = 3.0
    t3 = Path(tempfile.mkdtemp())
    z, ok, ep, hooked = one(cfg, left, right, jm, a.net_device, t3)
    check('use, no server: capture rejected (timeout)', (not ok) and not z['zed_fp'].get('ok'), why=z['zed_fp'].get('why'))
    print(json.dumps(out, indent=1, default=str))
    print('FAILED:', fails or 'none', flush=True)
    raise SystemExit(1 if fails else 0)


if __name__ == '__main__':
    main()
