"""Self-describing run directory for real (and dry) runs, written off the control loop.

    data/real_runs/<YYYY-MM-DD>/<run_id>/
      meta.json          git commit (+ dirty files), bundle manifest + verify result, config, args, rig calibration
                         (camera spec / joint offsets / capture pose hashes), operator notes, outcome label,
                         failure labels, dropped-sample counts
      events.jsonl       stage transitions, prompts and answers, estimator inputs/outputs and gate decisions,
                         guard stops, timings (one JSON object per line, host time.monotonic in "t")
      <stream>.npz       numeric streams, one row per sample, every key stacked (written as chunks under
                         chunks/<stream>/ while running, merged at close):
                           ticks       50 Hz: q/qd/effort/gripper both arms, targets, gripper commands, torque
                                       estimate, feed-forward, stage, loop timing
                           can_right   every encoder reading at CAN rate (arrival time, i2rt q7)
                           can_left
                           policy      insertion: observation (68), action (6), joint targets, contact-head
                                       prediction, seated-detector state
                           hold        left press: force estimate (3), normal force, latch, torques
      frames/<cam>/<n>_<tag>.{png,jpg}  every frame a net used (policy image as PNG = the exact net input, raw
                         frame as JPEG q95) and a continuous low-rate stream (raw JPEG); frames.jsonl indexes
                         them with capture (arrival) and use timestamps

Every call only enqueues (put_nowait); a writer thread encodes and writes. A full queue drops the sample and
counts it per stream (meta.json "dropped"), so logging can never stall the 50 Hz loop.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def git_state():
    def run(*a):
        try:
            return subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    return {'commit': run('rev-parse', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': [ln for ln in (run('status', '--porcelain', '--untracked-files=no') or '').splitlines()]}


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


class RunLogger:
    def __init__(self, out_dir, meta, chunk_s=5.0, maxsize=20000):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / 'frames').mkdir(exist_ok=True)
        self.meta = dict(meta)
        self.meta.setdefault('created', time.strftime('%Y-%m-%d %H:%M:%S %Z'))
        self.meta['t0_monotonic'] = time.monotonic()
        self.q = queue.Queue(maxsize=maxsize)
        self.dropped, self.counts = {}, {}
        self.chunk_s = chunk_s
        self.buf, self.nchunk = {}, {}
        self.frame_n = {}
        self.closed = False
        self._write_meta()
        self.f_events = open(self.dir / 'events.jsonl', 'a')
        self.f_frames = open(self.dir / 'frames.jsonl', 'a')
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    # ------------------------------------------------------------------ producer side (control loop)
    def _put(self, stream, item):
        self.counts[stream] = self.counts.get(stream, 0) + 1
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped[stream] = self.dropped.get(stream, 0) + 1

    def event(self, kind, **fields):
        self._put('events', ('event', dict(kind=kind, t=time.monotonic(), **fields)))

    def row(self, stream, **values):
        """One numeric sample (scalars / arrays / tensors) of a stream; keys must stay the same per stream."""
        vals = {k: (v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)) for k, v in values.items()}
        self._put(stream, ('row', stream, vals))

    def frame(self, cam, image, tag, t_capture=None, t_use=None, kind='raw', bgr=False, **extra):
        """image: HxWx3 uint8 (RGB unless bgr). kind 'policy' (exact net input) -> PNG; 'raw' -> JPEG q95."""
        img = image.detach().cpu().numpy() if hasattr(image, 'detach') else np.asarray(image)
        if img.ndim == 4:
            img = img[0]
        self._put('frames', ('frame', cam, img.copy(), tag, t_capture, t_use, kind, bgr, extra))

    # ------------------------------------------------------------------ writer thread
    def _loop(self):
        last_flush = time.monotonic()
        while True:
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                item = None
            if item is not None:
                if item[0] == 'close':
                    break
                try:
                    self._handle(item)
                except Exception as e:  # noqa: BLE001  (logging must not kill the run)
                    self.dropped['writer_errors'] = self.dropped.get('writer_errors', 0) + 1
                    self.meta.setdefault('writer_errors', []).append(repr(e)[:300])
            if time.monotonic() - last_flush > self.chunk_s:
                self._flush_rows()
                self.f_events.flush()
                self.f_frames.flush()
                last_flush = time.monotonic()
        while not self.q.empty():
            item = self.q.get_nowait()
            if item[0] != 'close':
                self._handle(item)
        self._flush_rows()

    def _handle(self, item):
        kind = item[0]
        if kind == 'event':
            self.f_events.write(json.dumps(_jsonable(item[1])) + '\n')
        elif kind == 'row':
            _, stream, vals = item
            self.buf.setdefault(stream, []).append(vals)
        elif kind == 'frame':
            import cv2
            _, cam, img, tag, t_cap, t_use, fk, bgr, extra = item
            n = self.frame_n.get(cam, 0)
            self.frame_n[cam] = n + 1
            d = self.dir / 'frames' / cam
            d.mkdir(parents=True, exist_ok=True)
            ext = 'png' if fk == 'policy' else 'jpg'
            name = f'{n:06d}_{tag}.{ext}'
            im = img if bgr else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(d / name), im, [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == 'jpg' else [])
            self.f_frames.write(json.dumps(_jsonable(dict(cam=cam, file=f'frames/{cam}/{name}', tag=tag, kind=fk,
                                                          t_capture=t_cap, t_use=t_use, shape=list(img.shape), **extra))) + '\n')

    def _flush_rows(self):
        for stream, rows in list(self.buf.items()):
            if not rows:
                continue
            k = self.nchunk.get(stream, 0)
            self.nchunk[stream] = k + 1
            d = self.dir / 'chunks' / stream
            d.mkdir(parents=True, exist_ok=True)
            keys = sorted({kk for r in rows for kk in r})
            out = {}
            for kk in keys:
                vs = [r.get(kk) for r in rows]
                try:
                    out[kk] = np.stack([np.asarray(v, float) if v is not None else np.full(np.shape(vs[0]), np.nan) for v in vs])
                except Exception:  # noqa: BLE001  (ragged: keep as an object array)
                    out[kk] = np.array(vs, dtype=object)
            np.savez(d / f'{k:05d}.npz', **out)
            self.buf[stream] = []

    def _write_meta(self):
        m = dict(self.meta, dropped=self.dropped, counts=self.counts)
        (self.dir / 'meta.json').write_text(json.dumps(_jsonable(m), indent=1, default=str))

    # ------------------------------------------------------------------ close
    def update_meta(self, **kw):
        self.meta.update(kw)
        self._write_meta()

    def close(self, **meta):
        if self.closed:
            return
        self.closed = True
        self.meta.update(meta)
        self.q.put(('close',))
        self.th.join(timeout=60)
        self.f_events.close()
        self.f_frames.close()
        merged = {}
        for d in sorted((self.dir / 'chunks').glob('*')) if (self.dir / 'chunks').exists() else []:
            parts = [dict(np.load(p, allow_pickle=True)) for p in sorted(d.glob('*.npz'))]
            keys = sorted({k for p in parts for k in p})
            out = {}
            for k in keys:
                try:
                    out[k] = np.concatenate([p[k] for p in parts if k in p])
                except ValueError:
                    out[k] = np.array([x for p in parts if k in p for x in p[k]], dtype=object)
            np.savez(self.dir / f'{d.name}.npz', **out)
            merged[d.name] = int(len(next(iter(out.values())))) if out else 0
        self.meta['streams'] = merged
        self.meta['closed'] = time.strftime('%Y-%m-%d %H:%M:%S %Z')
        self._write_meta()


def load_run(run_dir):
    """(meta, {stream: dict of arrays}, events list, frames list) of a run directory."""
    d = Path(run_dir)
    meta = json.loads((d / 'meta.json').read_text())
    streams = {p.stem: dict(np.load(p, allow_pickle=True)) for p in d.glob('*.npz')}
    ev = [json.loads(x) for x in (d / 'events.jsonl').read_text().splitlines() if x.strip()] if (d / 'events.jsonl').exists() else []
    fr = [json.loads(x) for x in (d / 'frames.jsonl').read_text().splitlines() if x.strip()] if (d / 'frames.jsonl').exists() else []
    return meta, streams, ev, fr
