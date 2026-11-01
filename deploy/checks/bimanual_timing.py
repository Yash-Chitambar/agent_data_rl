"""Per-tick compute of the real runner (read -> controller -> send), without arms, cameras or a simulator.

FakeIO tracks every target perfectly (the reading = the last sent target) and serves fixed real frames (data/sim_ref
pointed_*: ZED + right wrist at the capture pose, through the real preprocessing). The chain runs capture, servo
pick, hold, transport, approach and 10 s of insertion with logging on, as on the rig; the net outputs on the fixed
frames are meaningless, so a stage that stops (guard / IK) is restarted from the next one. Reports the compute time
per 50 Hz tick per stage (budget: the tick is 20 ms; the CAN read/send and the wait take the rest).

    PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_timing [--device cpu --net-device cuda:0] [--out ...json]
Run it on duchamp3 before any motion: p99 compute per stage must stay below ~15 ms.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
SIDES = ('right', 'left')


class FakeIO:
    is_sim = False

    def __init__(self, cfg, off, frames):
        self.cfg, self.off, self.dt = cfg, off, cfg['control_dt']
        self.fr = frames
        self.cams = {n: None for n in frames}
        cap = json.loads((ROOT / 'blind_insertion/configs/zed_capture_pose.json').read_text())['model_frame_rad']
        self.q = np.stack([np.asarray(cap[s]) for s in SIDES]) - off
        self.g = np.ones(2)
        self.k = 0

    def read(self):
        return dict(q=self.q.copy(), qd=np.zeros((2, 6)), grip=self.g.copy(), eff=np.full((2, 6), np.nan),
                    t_obs=np.full(2, time.monotonic()), t=time.monotonic())

    def send(self, q_motor, grip, ff_right=None):
        self.q, self.g = np.asarray(q_motor, float).copy(), np.asarray(grip, float).copy()
        self.k += 1
        return np.concatenate([self.q, self.g[:, None]], 1), np.zeros((2, 6))

    def start_clock(self):
        pass

    def wait_tick(self):
        return 0.0

    def move_to(self, q_motor, grip=None, **kw):
        for s, q in q_motor.items():
            self.q[SIDES.index(s)] = q
        for s, g in (grip or {}).items():
            self.g[SIDES.index(s)] = g

    def set_grip(self, side, value, wait_s=0):
        self.g[SIDES.index(side)] = value

    def hold(self):
        pass

    def stopped(self):
        return False

    def drain_can(self):
        return {}

    def frames(self, names=('top', 'wrist_right')):
        return {n: (self.fr[n], None, time.monotonic(), self.k) for n in names}

    fresh_frames = frames

    def truth(self):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--net-device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    import cv2
    torch.set_num_threads(a.threads)
    from blind_insertion.assets.real_camera import load_spec, real_policy_image
    from blind_insertion.scenes.physical_rig import real_top_policy_image
    import rig
    from deploy.bimanual.chain import RealChain
    from deploy.bimanual.guards import Guards
    from deploy.bimanual.runlog import RunLogger
    from deploy.bimanual.scene import SceneModel
    from deploy.run_bimanual import gripper_maps, load_cfg
    cfg = load_cfg(ROOT / 'deploy/config/bimanual.json')
    snap = ROOT / 'data/sim_ref/2026-10-25_scene'
    frames = {'top': real_top_policy_image(cv2.imread(str(snap / 'pointed_zed_left.png')), 320),
              'wrist_right': real_policy_image(cv2.imread(str(snap / 'pointed_wrist_right.png')), load_spec(rig.SPEC), 320, 'right')}
    sc = SceneModel(a.device)
    consts = json.loads((ROOT / cfg['bundle_dir'] / 'configs/chain_constants.json').read_text())
    io = FakeIO(cfg, sc.encoder_offsets.cpu().numpy().astype(float), frames)
    tmp = Path(tempfile.mkdtemp())
    log = RunLogger(tmp, {'check': 'timing'})
    g = Guards(cfg, sc.kp.cpu().numpy(), cfg['control_dt'])
    ch = RealChain(cfg, io, sc, consts, gripper_maps(cfg), log=log, guards=None, device=a.device, net_device=a.net_device,
                   policy_path=ROOT / cfg['bundle_dir'] / 'models/insert_KJ.pt', auto='')
    del g
    res, notes = {}, {}
    lg = {}

    def stage(name, fn):
        k0 = ch.k
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001  (meaningless net outputs on fixed frames can stop a stage)
            notes[name] = f'{type(e).__name__}: {str(e)[:120]}'
            out = None
        notes[name + '_ticks'] = ch.k - k0
        notes[name + '_wall_s'] = time.perf_counter() - t0
        return out
    ep, eR, _ = ch.capture_pc(lg)
    ch.park_left()
    r = stage('pick', lambda: ch.stage_pick(lg, 'servo'))
    qr = ch.q_b[0][None].clone()
    q_hold = stage('hold', lambda: ch.stage_hold(qr, ep, eR, lg))
    if q_hold is None:
        q_hold = ch.q_b[1][None].clone()
    ch.rt.comp['right'] = ch.fric
    q_look = stage('transport', lambda: ch.stage_transport(qr, q_hold, ep, eR, lg))
    q_look = q_look if q_look is not None else ch.q_b[0][None].clone()
    cmd = stage('approach', lambda: ch.stage_approach(q_look, q_hold, ep, eR, lg))
    if not hasattr(ch, 'inhand_est6'):
        ch.inhand_est6 = torch.zeros(1, 6)
    cmd = cmd if cmd is not None else ch.q_b[0][None].clone()
    stage('insert', lambda: ch.stage_insert(cmd, q_hold, lg))
    log.close()
    t = dict(np.load(tmp / 'ticks.npz'))
    st, c = t['stage'].astype(int), t['compute_ms']
    from deploy.bimanual.chain import STAGES
    for s in sorted(set(st.tolist())):
        m = c[st == s][1:]
        if len(m):
            res[STAGES[s]] = {'ticks': int(len(m)), 'p50_ms': float(np.percentile(m, 50)), 'p99_ms': float(np.percentile(m, 99)),
                              'max_ms': float(m.max()), 'over_20ms': float((m > 20).mean())}
    out = {'host': __import__('socket').gethostname(), 'device': a.device, 'net_device': a.net_device, 'threads': a.threads,
           'per_stage': res, 'notes': notes}
    print(json.dumps(out, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
