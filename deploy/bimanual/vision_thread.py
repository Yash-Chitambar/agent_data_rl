"""Port vision off the 50 Hz loop (servo align mode, docs/real/K2_PROGRESS.md "Servo deploy").

port_vision.measure takes about 82 ms on the CPU (docs/real/PORT_VISION.md); the step stage calls it inline, so the arm gets one control
tick between frames. The servo needs a fresh measurement while it moves, without stopping the ticks. The main loop owns the camera,
the logger and the arm; the worker only runs the pure measurement function on one raw image at a time.

Each result carries the control tick at which its frame was captured, so the servo can add the FK motion since that tick.

Two backends with one interface (submit / poll / busy):
  ThreadWorker: a daemon thread runs fn(raw) -> PortMeasure (robot).
  DeferredWorker: the sim. The measurement is taken at submit time (the sim source reads the truth at that tick) and handed back
    `latency_ticks` later, as a real frame would be.
"""
from __future__ import annotations

import queue
import threading
import time


class Result:
    __slots__ = ('m', 'tick', 't_capture', 'ms', 'tag')

    def __init__(self, m, tick, t_capture, ms, tag):
        self.m, self.tick, self.t_capture, self.ms, self.tag = m, tick, t_capture, ms, tag


class ThreadWorker:
    def __init__(self, fn, on_error):
        """fn(raw) -> measurement; on_error(exc) -> measurement for a failed call (never raises into the loop)."""
        self.fn, self.on_error = fn, on_error
        self.q_in, self.q_out = queue.Queue(maxsize=1), queue.Queue()
        self._busy = False
        self._stop = False
        self.th = threading.Thread(target=self._run, name='port_vision_worker', daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop:
            try:
                job = self.q_in.get(timeout=.1)
            except queue.Empty:
                continue
            raw, tick, t_cap, tag = job
            t0 = time.perf_counter()
            try:
                m = self.fn(raw)
            except Exception as e:  # noqa: BLE001  (a vision failure is a failed measurement)
                m = self.on_error(e)
            self.q_out.put(Result(m, tick, t_cap, (time.perf_counter() - t0) * 1e3, tag))

    @property
    def busy(self):
        return self._busy

    def submit(self, raw, tick, t_capture, tag, now_tick=None):
        if self._busy:
            return False
        self._busy = True
        self.q_in.put((raw, tick, t_capture, tag))
        return True

    def poll(self, now_tick=None):
        """Newest finished result or None. Non-blocking."""
        out = None
        while True:
            try:
                out = self.q_out.get_nowait()
            except queue.Empty:
                break
            self._busy = False
        return out

    def close(self):
        self._stop = True
        self.th.join(timeout=1.)


class DeferredWorker:
    def __init__(self, measure_now, latency_ticks=5, period_ticks=5):
        """measure_now(tag) -> measurement of the current state (sim truth source)."""
        self.measure_now, self.lat, self.period = measure_now, int(latency_ticks), int(period_ticks)
        self.pending = None
        self.last_submit = -10 ** 9

    @property
    def busy(self):
        return self.pending is not None

    def submit(self, raw, tick, t_capture, tag, now_tick=None):
        if self.pending is not None or tick - self.last_submit < self.period:
            return False
        t0 = time.perf_counter()
        m = self.measure_now(tag)
        self.pending = Result(m, tick, t_capture, (time.perf_counter() - t0) * 1e3, tag)
        self.last_submit = tick
        return True

    def poll(self, now_tick=None):
        if self.pending is not None and now_tick is not None and now_tick - self.pending.tick >= self.lat:
            out, self.pending = self.pending, None
            return out
        return None

    def close(self):
        pass


def _proc_main(conn, shm_name, shape, kw):
    """Child process: port_vision.measure on the image in shared memory; replies (status, measurement or error text, ms)."""
    import numpy as np
    from multiprocessing import shared_memory
    from deploy.bimanual import port_vision as PV
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    try:
        PV.load_model()
    except Exception:  # noqa: BLE001
        pass
    conn.send(('ready', None, 0.))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        t0 = time.perf_counter()
        try:
            m = PV.measure(buf.copy(), side='right', **dict(kw, plug_z_mm=msg))
            conn.send(('ok', m, (time.perf_counter() - t0) * 1e3))
        except Exception as e:  # noqa: BLE001
            conn.send(('err', f'{type(e).__name__}: {e}', (time.perf_counter() - t0) * 1e3))
    shm.close()


class ProcessWorker:
    """port_vision.measure in a child process (no shared interpreter lock with the 50 Hz loop). The image goes through shared memory; the
    child returns port_vision's own PortMeasure, and convert() turns it into the align stage's measurement on the main side."""

    def __init__(self, kw, plug_z_fn, convert, on_error, shape=(1024, 1280, 3), ready_timeout_s=60.):
        import multiprocessing as mp
        from multiprocessing import shared_memory
        import numpy as np
        ctx = mp.get_context('spawn')
        self.shape = shape
        self.shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        self.buf = np.ndarray(shape, dtype=np.uint8, buffer=self.shm.buf)
        self.conn, child = ctx.Pipe()
        self.p = ctx.Process(target=_proc_main, args=(child, self.shm.name, shape, dict(kw)), daemon=True)
        self.p.start()
        self.plug_z_fn, self.convert, self.on_error = plug_z_fn, convert, on_error
        self.pending = None
        if not self.conn.poll(ready_timeout_s):
            raise RuntimeError('port_vision worker process did not start')
        self.conn.recv()

    @property
    def busy(self):
        return self.pending is not None

    def submit(self, raw, tick, t_capture, tag, now_tick=None):
        if self.pending is not None or raw is None or raw.shape != self.shape:
            return False
        self.buf[...] = raw
        self.conn.send(self.plug_z_fn())
        self.pending = (tick, t_capture, tag)
        return True

    def poll(self, now_tick=None):
        if self.pending is None or not self.conn.poll():
            return None
        status, payload, ms = self.conn.recv()
        tick, t_cap, tag = self.pending
        self.pending = None
        try:
            m = self.convert(payload) if status == 'ok' else self.on_error(RuntimeError(payload))
        except Exception as e:  # noqa: BLE001
            m = self.on_error(e)
        return Result(m, tick, t_cap, ms, tag)

    def close(self):
        try:
            self.conn.send(None)
            self.p.join(timeout=2.)
        except Exception:  # noqa: BLE001
            pass
        if self.p.is_alive():
            self.p.terminate()
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass
