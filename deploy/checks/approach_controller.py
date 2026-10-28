"""Supervised 5-mrad joint-3 baseline/feedforward comparison at folded rest."""
import json
import argparse
import time
from pathlib import Path

import numpy as np

from deploy import conventions as C
from deploy.backends import RealBackend, real_loop
from deploy.runner import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--amp', type=float, default=.005)
    ap.add_argument('--name', default='approach_controller_001')
    args = ap.parse_args()
    if not 0 < args.amp <= .015:
        raise SystemExit('amplitude must be in (0, .015] rad')
    import estop
    if estop.stopped():
        raise SystemExit('software stop is set')
    out = C.REPO / 'data/task_h' / (args.name + '.json')
    if out.exists():
        raise SystemExit(f'refusing overwrite: {out}')
    cfg = load_config()
    be = RealBackend(cfg, C.joint_offsets('right'))
    rows, result, completed = [], 'incomplete', False
    try:
        q0 = be.read()[0].copy()
        if np.max(np.abs(q0[1:3])) > .05:
            raise RuntimeError('not at folded rest')
        for enabled in (False, True):
            start = time.monotonic()
            for k in range(301):
                if be.stopped():
                    raise RuntimeError('software stop')
                t = k * .02
                offset = args.amp * min(t, 1., max(0., 4. - t))
                target = np.r_[q0, 1.0]
                target[2] += offset
                be.arm.safe.command(target, real_loop().FRIC_COMP if enabled else None)
                q, qd, g, _ = be.read()
                _, ages, counts = be.frames()
                rows.append(dict(host_monotonic=time.monotonic(), feedforward=enabled,
                    target=target.tolist(), q_encoder=q.tolist(), qd=qd.tolist(), gripper=g,
                    camera_age_s=ages, camera_counts=counts,
                    computed_feedforward=getattr(be.arm.safe, 'last_ff', None)))
                if np.max(np.abs(q - q0)) > .02:
                    raise RuntimeError('deviation exceeds 0.020 rad')
                time.sleep(max(0., start + (k + 1) * .02 - time.monotonic()))
        result, completed = 'completed', True
    except Exception as exc:
        result = repr(exc)
        estop.STOP_FILE.write_text('stop: controller comparison failed\n')
        be.hold()
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(result=result, controller=be.arm.info(), samples=rows),
                                  indent=1, default=lambda v: np.asarray(v).tolist()) + '\n')
        print(result, out, flush=True)
        if completed:
            be.close()
        else:
            print('Holding powered; secure support before shutdown. Do not terminate.', flush=True)
            while True:
                time.sleep(.1)


if __name__ == '__main__':
    main()
