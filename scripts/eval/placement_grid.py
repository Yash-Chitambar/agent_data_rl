"""Full real-path dry run (deploy.run_bimanual --dry-run --stage full, SimIO) at forced PC and reader poses: where each
stage fails as the placement moves away from the real spot. No hardware.

Each grid point is one dry-run process: BimanualRuntime._sample_layout is patched so the PC port and the reader sit
exactly at the asked pose (the rest of the episode's DR stays as the seed draws it). Poses use the rig's numbers:
  pc     = [port x, port y (m, right-base frame), yaw (deg, ZED convention atan2(-R12, -R02))]
  reader = [body x, body y (m, the ZED reader estimate's p), yaw (deg, the estimate's yaw)]
The runtime's own layout point for the reader is body - Rz(yaw) [0.0698, 0] (BimanualRuntime._reader0).

    # one point (worker)
    PYTHONPATH=. MUJOCO_GL=egl python scripts/eval/placement_grid.py one --pc 0.231 0.394 4 --reader 0.131 0.448 92 \
        --seed 7041 --config deploy/config/bimanual_align8_nojump.json --out results/placement/grid --tag p0
    # a grid (JSON list of {"tag", "pc", "reader", "seed"}), N parallel processes
    PYTHONPATH=. MUJOCO_GL=egl python scripts/eval/placement_grid.py grid --points pts.json --workers 8 --config ... --out ...
    # table of stage outcomes from the run folders
    PYTHONPATH=. python scripts/eval/placement_grid.py table --out results/placement/grid
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def patch_layout(pc, reader):
    import torch
    from blind_insertion.envs import bimanual_runtime as BR
    orig = BR.BimanualRuntime._sample_layout

    def forced(self, ids):
        orig(self, ids)
        n, dev = len(ids), self.device
        for qadr, ref, (x, y, t), body in ((self.pc_qadr, self._pc0, pc, False), (self.lay.peg_qadr, self._reader0, reader, True)):
            a = torch.full((n,), math.radians(t), device=dev)
            c, s_ = torch.cos(a), torch.sin(a)
            p0 = ref[:3]
            if body:            # reader: the asked xy is the body position
                pos = torch.stack([torch.full((n,), x, device=dev), torch.full((n,), y, device=dev), p0[2].expand(n)], -1)
            else:               # PC: the asked xy is the port (the PC reference point)
                pos = torch.stack([x + c * p0[0] - s_ * p0[1], y + s_ * p0[0] + c * p0[1], p0[2].expand(n)], -1)
            qz = torch.stack([torch.cos(a / 2), torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2)], -1)
            w1, x1, y1, z1 = qz.unbind(-1)
            w2, x2, y2, z2 = ref[3:]
            quat = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
            self.q[ids, qadr:qadr + 3] = pos
            self.q[ids, qadr + 3:qadr + 7] = quat
    BR.BimanualRuntime._sample_layout = forced


def one(a):
    patch_layout(a.pc, a.reader)
    if a.zed_ood_scale:     # sim study only: open the ZED OOD gate to see the stages behind it (the estimate is still the net's)
        from blind_insertion.perception import zed_estimator as ZE
        ZE.OOD_SCALE = float(a.zed_ood_scale)
    from deploy import run_bimanual as RB
    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    (out / 'point.json').write_text(json.dumps(dict(pc=a.pc, reader=a.reader, seed=a.seed, config=a.config, speed=a.speed, zed_ood_scale=a.zed_ood_scale)))
    argv = ['--dry-run', '--stage', 'full', '--seed', str(a.seed), '--config', a.config, '--align-mode', 'full', '--speed', str(a.speed),
            '--yes', '--name', a.tag, '--out', str(out), '--no-home'] + list(a.extra or [])
    try:
        res = RB.main(argv)
        (out / 'result.json').write_text(json.dumps(res, default=str))
    except BaseException as e:  # noqa: BLE001
        (out / 'result.json').write_text(json.dumps({'exception': repr(e)}))
        raise


def grid(a):
    pts = json.loads(Path(a.points).read_text())
    procs, todo = [], list(pts)
    logd = Path(a.out) / '_logs'
    logd.mkdir(parents=True, exist_ok=True)
    while todo or procs:
        while todo and len(procs) < a.workers:
            p = todo.pop(0)
            if (Path(a.out) / p['tag'] / 'result.json').exists():
                continue
            cmd = [sys.executable, __file__, 'one', '--pc', *map(str, p['pc']), '--reader', *map(str, p['reader']), '--seed', str(p.get('seed', a.seed)),
                   '--config', p.get('config', a.config), '--out', a.out, '--tag', p['tag'], '--speed', str(a.speed)] + \
                (['--zed-ood-scale', str(p['zed_ood_scale'])] if p.get('zed_ood_scale') else [])
            f = open(logd / f"{p['tag']}.log", 'w')
            procs.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=os.environ.copy()), f, p['tag']))
        time.sleep(5)
        for pr in list(procs):
            if pr[0].poll() is not None:
                pr[1].close()
                procs.remove(pr)
                print(f'{pr[2]} exit {pr[0].returncode}', flush=True)


def classify(run_dir):
    """First failing stage of one dry run from its events.jsonl (and the sim truth the dry run logs)."""
    evs = []
    for f in Path(run_dir).rglob('events.jsonl'):
        evs = [json.loads(x) for x in open(f)]
        break
    k = {}
    for e in evs:
        k.setdefault(e['kind'], []).append(e)
    r = dict(stage='done', why='')
    zed = [e for e in k.get('estimate', []) if e.get('est') == 'zed_pc']
    rd = [e for e in k.get('estimate', []) if e.get('est') == 'zed_reader']
    if zed:
        r.update(zed_ok=zed[0].get('ok'), zed_err_xy_mm=zed[0].get('err_xy_mm'), zed_yaw=zed[0].get('yaw_deg'), zed_true_yaw=zed[0].get('true_yaw_deg'))
    if rd:
        r.update(reader_ok=rd[-1].get('ok'), reader_err_mm=rd[-1].get('err_mm'), reader_tries=len(rd))
    po = k.get('pick_outcome', [{}])[-1].get('outcome')
    pr = k.get('pick_result', [{}])[-1]
    hr = k.get('hold_result', [{}])[-1]
    al = k.get('align_result', [{}])[-1]
    ir = k.get('insert_result', [{}])[-1]
    stops = [e for e in evs if e['kind'] in ('guard_stop', 'stop', 'safety_stop', 'error')]
    r.update(pick_outcome=po, pick_ok=pr.get('ok'), hold_latched=hr.get('latched'), align_ok=al.get('ok'), align_reason=al.get('reason'),
             align_s=al.get('seconds'), align_cycles=al.get('cycles'), insert_outcome=ir.get('outcome'),
             stop=[{kk: e.get(kk) for kk in ('kind', 'msg', 'reason', 'stage', 'message') if kk in e} for e in stops][:2])
    stages = [e['stage'] for e in k.get('stage', [])]
    r['last_stage'] = stages[-1] if stages else None
    if zed and not zed[0].get('ok'):
        r.update(stage='zed', why='gate')
    elif rd and not rd[-1].get('ok'):
        r.update(stage='reader_zed', why='gate')
    elif pr and not pr.get('ok', False):
        r.update(stage='pick', why=str(po))
    elif not pr:
        r.update(stage='pick', why='no pick result')
    elif hr and not hr.get('latched', False):
        r.update(stage='hold', why='not latched')
    elif not hr:
        r.update(stage='hold', why='no hold result')
    elif not al:
        r.update(stage='transport/approach', why=str(r['stop'])[:200])
    elif not al.get('ok', False):
        r.update(stage='align', why=str(al.get('reason'))[:120])
    elif ir.get('outcome') not in ('seated', 'seated_detected'):
        r.update(stage='insert', why=str(ir.get('outcome')))
    res = Path(run_dir) / 'result.json'
    if res.exists():
        j = json.loads(res.read_text())
        if isinstance(j, dict) and 'exception' in j:
            r['exception'] = j['exception'][:200]
        elif isinstance(j, list) and j:
            r['completed'] = j[0].get('completed')
            r['stop_msg'] = str(j[0].get('stop'))[:200]
            r['true_seated'] = (j[0].get('results') or {}).get('true_seated_after_release')
            if r['stage'] == 'done' and not j[0].get('completed'):
                r.update(stage=r['last_stage'] or 'unknown', why=r['stop_msg'])
    return r


def table(a):
    rows = []
    for d in sorted(Path(a.out).iterdir()):
        if not (d / 'point.json').exists():
            continue
        p = json.loads((d / 'point.json').read_text())
        rows.append(dict(tag=d.name, **p, **classify(d)))
    Path(a.out, 'table.json').write_text(json.dumps(rows, indent=1, default=str))
    for r in rows:
        print(r['tag'], r['pc'], r['reader'], '->', r['stage'], r['why'][:80] if r['why'] else '', 'zed_err', r.get('zed_err_xy_mm'), 'align', r.get('align_s'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('one', 'grid', 'table'))
    ap.add_argument('--pc', type=float, nargs=3)
    ap.add_argument('--reader', type=float, nargs=3)
    ap.add_argument('--seed', type=int, default=7041)
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual_align8_nojump.json'))
    ap.add_argument('--out', default=str(ROOT / 'results/placement/grid'))
    ap.add_argument('--tag', default='p')
    ap.add_argument('--speed', type=float, default=1.0)
    ap.add_argument('--points')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--extra', nargs='*')
    ap.add_argument('--zed-ood-scale', type=float, default=None, help='sim study only: ZED OOD gate scale (default the estimator\'s 1.25)')
    a = ap.parse_args()
    {'one': one, 'grid': grid, 'table': table}[a.mode](a)


if __name__ == '__main__':
    main()
