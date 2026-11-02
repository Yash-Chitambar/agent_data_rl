"""Perception nets, loaded once per process BEFORE the arms connect (run_bimanual.Runner.open_io -> preload).

Why: torch.load + .to(cuda) and the first CUDA use hold the Python GIL for 20-230 ms per net, cold or warm
(measured 2026-11-01 with a 2 ms ticker thread). With the arms connected in this process, i2rt's motor thread
cannot send meanwhile; pyzed's Camera.open (1.7 s) did trip the DM motors' CAN watchdog (0xD "loss
communication": torque off until i2rt re-enables them). Every loader here is memoized: the chain and the
stages get the objects preload() made. All are eval-mode nets without per-episode state.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'runs/perception'
_cache = {}


def _get(key, make):
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def _d(dev):
    return str(torch.device(dev))


def zed(ndev, gate):
    from blind_insertion.perception.zed_estimator import ZedEstimator
    return _get(('zed', _d(ndev), gate),
                lambda: ZedEstimator(_d(ndev), pose=str(P / 'zed.pt'), presence=str(P / f'{gate}.pt')))


def wrist(name, ndev):
    from scripts.eval import eval_chain as EC
    return _get(('wrist', name, _d(ndev)), lambda: EC.wrist_loader(name, torch.device(ndev)))


def zed_reader(name, dev, retries):
    from blind_insertion.pipeline.pick_stage import ZedReader
    return _get(('zed_reader', name, _d(dev), int(retries)),
                lambda: ZedReader(str(P / f'{name}.pt'), torch.device(dev), retries=retries))


def servo(name, ndev):
    from blind_insertion.pipeline.pick_servo import load_servo_net
    return _get(('servo', name, _d(ndev)), lambda: load_servo_net(P / f'{name}.pt', torch.device(ndev)))


def preload(cfg, dev, ndev):
    """Every net a run can use (RealChain.__init__, make_pick, the capture stage). Returns seconds taken."""
    t0 = time.monotonic()
    ca, pc = cfg['chain_args'], cfg['pick']
    zed(ndev, ca['zed_gate'])
    for k in ('wrist_net', 'fine_net', 'inhand_net'):
        wrist(ca[k], ndev)
    zed_reader(pc['reader_estimator'], dev, pc['reader_retries'])
    for k in ('hover_net', 'near_net', 'rot_net'):
        if pc.get(k):
            servo(pc[k], ndev)
    if torch.device(ndev).type == 'cuda':
        torch.cuda.synchronize(ndev)
    return time.monotonic() - t0
