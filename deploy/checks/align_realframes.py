"""The align stage on the robot code path with a real logged stop frame, no hardware: RealChain + fake arm + PortVisionSource. Run A's stop frames as the camera; the arm does what it is told.

  measure: 10 still measurements, values as the offline replay, stage ends "measure mode finished", no align motion before the back-off.
  align: the frame cannot follow the hand, so a gate must stop the stage (measurement against FK, or scale). It must not reach the hand-over.
  missing plug: a frame with no shell (pick-run frame): no depth, stops without any move.

    PYTHONPATH=. python -m deploy.checks.align_realframes
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
RUN = ROOT / 'data/real_runs/2026-11-02/full_174850_full_A'
EMPTY = ROOT / 'data/real_runs/2026-11-02/pick_155557_batchA'


def frames(run, want_tag=None):
    rows = [json.loads(x) for x in open(run / 'frames.jsonl')]
    return [run / r['file'] for r in rows if r['cam'] == 'wrist_right' and r['kind'] == 'raw']


def main():
    from deploy.bimanual.align import align_cfg, run_align
    from deploy.bimanual.chain import RealChain
    from deploy.bimanual.runlog import RunLogger, load_run
    from deploy.bimanual.scene import SceneModel
    from deploy.checks.bimanual_timing import FakeIO
    from deploy.run_bimanual import gripper_maps, load_cfg
    cfg = load_cfg(ROOT / 'deploy/config/bimanual_align1.json')
    ac = align_cfg(cfg)
    sc = SceneModel('cpu')
    consts = json.loads((ROOT / cfg['bundle_dir'] / 'configs/chain_constants.json').read_text())
    tk = np.load(RUN / 'ticks.npz')
    ev = [json.loads(x) for x in open(RUN / 'events.jsonl')]
    k1 = next(e['tick'] for e in ev if e['kind'] == 'stage' and e['stage'] == '5 insert')
    i1 = int(np.searchsorted(tk['k'].astype(int), k1)) - 1
    off = sc.encoder_offsets.cpu().numpy().astype(float)
    stop = [json.loads(x) for x in open(RUN / 'frames.jsonl')]
    stop = [RUN / r['file'] for r in stop if r['cam'] == 'wrist_right' and r['kind'] == 'raw' and r.get('tick') is not None and k1 - 300 <= r['tick'] <= k1]
    raws = [cv2.imread(str(f)) for f in stop]
    empty = cv2.imread(str(frames(EMPTY)[0]))
    pol = np.zeros((320, 320, 3), np.uint8)
    q_stop = tk['q'][i1].astype(float)

    class IO(FakeIO):
        n = 0
        raws = None

        def fresh_frames(self, names=('wrist_right',)):
            self.n += 1
            f = self.raws[self.n % len(self.raws)]
            return {n: (pol, f, 0., self.k) for n in names}
        frames = fresh_frames

    def run(mode, raws_, **over):
        tmp = Path(tempfile.mkdtemp())
        log = RunLogger(tmp, {'check': 'align_realframes'})
        io = IO(cfg, off, {'top': pol, 'wrist_right': pol})
        io.raws = raws_
        io.q = q_stop.copy()
        io.g = np.array([.31, .3])
        ch = RealChain(cfg, io, sc, consts, gripper_maps(cfg), log=log, guards=None, device='cpu', net_device='cuda:0' if torch.cuda.is_available() else 'cpu',
                       policy_path=ROOT / cfg['bundle_dir'] / 'models/insert_KJ.pt', auto='')
        ch.inhand_est6 = torch.zeros(1, 6)
        ch.pressing = True
        yaw = next(e['yaw_deg'] for e in ev if e['kind'] == 'estimate' and e.get('est') == 'zed_pc')
        sys.path.insert(0, str(ROOT / 'scripts/real'))
        import insertion_log_analysis as L
        eR = torch.tensor(L.port_R(yaw), dtype=torch.float32)
        cmd = ch.q_b[0][None].clone()
        q_hold = ch.q_b[1].clone()
        res = run_align(ch, cmd, q_hold, None, eR, {}, dict(ac, mode=mode, **over))
        log.close()
        _, st, evs, _ = load_run(tmp)
        return res, evs
    ok = True
    raws = raws[-2:]        # the earlier stop frames are other servo poses
    r, evs = run('measure', raws)
    ms = [e for e in evs if e['kind'] == 'align_measure' and e.get('across') is not None]
    v = np.array([[e['across'], e['up'], e['gap']] for e in ms], float)
    steps = sum(1 for e in evs if e['kind'] == 'align_step')
    print('measure:', r['reason'], 'cycles', r['cycles'], 'steps', steps, 'median', np.median(v, 0).round(2).tolist() if len(v) else None, 'spread', (v.max(0) - v.min(0)).round(2).tolist() if len(v) else None)
    ok &= steps == 0 and r['reason'] == 'measure mode finished' and r['cmd'] is None and len(v) >= 5
    r, evs = run('align', raws)
    print('align with a frozen frame:', r['ok'], r['reason'], 'cycles', r['cycles'], 'steps', sum(1 for e in evs if e['kind'] == 'align_step'))
    ok &= not r['ok'] and r['cmd'] is None
    r, evs = run('full', [empty])
    print('no shell in view:', r['ok'], r['reason'], 'steps', sum(1 for e in evs if e['kind'] == 'align_step'), [e for e in evs if e['kind'] == 'align_prepare'][:1])
    ok &= not r['ok'] and r['cmd'] is None and sum(1 for e in evs if e['kind'] == 'align_step') == 0
    print('ok' if ok else 'FAILED')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
