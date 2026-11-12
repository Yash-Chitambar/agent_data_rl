"""Live run state for the camera viewer (scripts/real/camview.py): which GaP stage the chain is in, what the stage just did,
and the outcome. RealChain.log_event feeds every event here; the state is written to /dev/shm/yam_cams/run_state.json
(atomic replace, at most every 0.25 s except on a stage change or a result). Costs well under 1 ms per write.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

DIR = Path('/dev/shm/yam_cams')
PATH = DIR / 'run_state.json'

# chain stage label -> graph node (scripts/real/camview.py draws NODES in this order)
NODES = ('capture', 'pick', 'hold', 'transport', 'approach', 'align', 'insert', 'release', 'home')
STAGE_NODE = {'0 capture': 'capture', '1 pick': 'pick', '2 hold': 'hold', '3 transport': 'transport', '4 approach': 'approach',
              'align': 'align', '5 insert': 'insert', '6 release': 'release', '7 home': 'home'}


def _f(x, n=1):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


class LiveState:
    def __init__(self, run, config, policies):
        """policies: {node: short text}, what runs in each node (shown under the node)."""
        DIR.mkdir(exist_ok=True)
        self.s = dict(run=run, config=config, policies=policies, nodes=list(NODES), node=None, done=[], failed=None,
                      status='connecting', outcome=None, t0=time.time(), t=time.time(), pid=os.getpid(), finished=False)
        self.last = 0.
        self.write(True)

    def write(self, force=False):
        now = time.time()
        if not force and now - self.last < .25:
            return
        self.last, self.s['t'] = now, now
        tmp = DIR / f'.run_state.{os.getpid()}.json'
        try:
            tmp.write_text(json.dumps(self.s))
            tmp.replace(PATH)
        except OSError:
            pass

    def event(self, kind, kw):
        s, force = self.s, False
        if kind == 'stage':
            node = STAGE_NODE.get(kw.get('stage'))
            prev = STAGE_NODE.get(kw.get('prev'))
            if prev is not None and prev != node and prev not in s['done']:      # a stage entered without its own event (pick)
                s['done'].append(prev)
            if node is not None and node != s['node']:
                if s['node'] is not None and s['node'] not in s['done']:
                    s['done'].append(s['node'])
                s['node'], s['status'], force = node, '', True
        elif kind == 'pick_result':
            s['status'] = 'picked' if kw.get('ok') else 'pick failed'
        elif kind == 'hold_result':
            s['status'] = f"PC held (latched {kw.get('latched')})"
        elif kind == 'align_measure' and kw.get('ok'):
            s['status'] = f"cycle {kw.get('cycle')}: across {_f(kw.get('across'))}, up {_f(kw.get('up'))}, gap {_f(kw.get('gap'))} mm"
        elif kind == 'align_servo_end':
            s['status'] = f"servo {'OK' if kw.get('ok') else 'STOP'} in {_f(kw.get('seconds'))} s"
        elif kind == 'align_press_on':
            s['status'] = 'PC pressed again: re-checking'
        elif kind == 'align_handover':
            if kw.get('ok'):
                lat = kw.get('lat_mm') or [None, None]
                s['status'] = f"hand-over: gap {_f(kw.get('gap_mm'))} mm, lateral {_f(lat[0])}, {_f(lat[1])} mm"
            else:
                s['status'], s['failed'] = f"align stop: {kw.get('reason')}", 'align'
            force = True
        elif kind == 'seated_detected':
            s['status'] = f"seated: depth {_f(kw.get('depth_mm'))} mm"
            force = True
        elif kind == 'insert_final_push':
            s['status'] = f"final push: +{_f(kw.get('advance_mm'), 2)} mm" + (f" (stopped: {kw.get('stopped')})" if kw.get('stopped') else '')
            force = True
        elif kind == 'insert_final_push_skipped':
            s['status'] = f"push skipped: {kw.get('reason')}"
        elif kind == 'insert_stop':
            s['status'] = f"insert stop: {kw.get('outcome')}"
        elif kind == 'insert_result':
            s['outcome'] = kw.get('outcome')
            if kw.get('outcome') != 'seated_detected':
                s['failed'] = 'insert'
            force = True
        elif kind == 'stopped':
            s['status'], s['failed'], force = f"STOPPED: {kw.get('reason')}", s['node'] or 'capture', True
        self.write(force)

    def finish(self, out):
        s = self.s
        if s['node'] is not None and s['node'] not in s['done'] and not s['failed']:
            s['done'].append(s['node'])
        s['finished'], s['node'] = True, None
        if out.get('stop'):
            s['outcome'] = s['outcome'] or 'stopped'
        self.write(True)
