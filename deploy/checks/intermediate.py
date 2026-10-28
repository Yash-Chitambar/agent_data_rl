"""Supervised intermediate-pose acquisition with existing configured feedforward."""
import json
from pathlib import Path
import numpy as np
from deploy import conventions as C
from deploy.runner import load_config
from deploy.backends import RealBackend
from deploy.recovery import supported_shutdown


def main():
    import estop
    if estop.stopped(): raise SystemExit('software stop is set')
    out=C.REPO/'data/task_h/intermediate_feedforward_001.json'
    if out.exists(): raise SystemExit(f'refusing overwrite {out}')
    cfg=load_config();be=RealBackend(cfg,C.joint_offsets('right'))
    q0=be.read()[0].copy();result='incomplete';ok=False
    try:
        if np.max(np.abs(q0[1:3]))>.05: raise RuntimeError('not at folded rest')
        goal=np.array([.15,.3,.3,-.3,0.,.15])
        be.arm.move_to(goal,max_vel=.05,approach_fric_comp=True)
        reached=be.read()[0].copy()
        print('Intermediate reached; returning to recorded rest.',flush=True)
        be.arm.move_to(q0,max_vel=.05,approach_fric_comp=True)
        result='completed';ok=True
    except Exception as exc:
        result=repr(exc);estop.STOP_FILE.write_text('stop: intermediate diagnostic failed\n')
    finally:
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(result=result,start=q0.tolist(),controller=be.arm.info(),
            motion_trace=be.arm.motion_trace),indent=1)+'\n')
        print(result,flush=True)
        if ok:be.close()
        else:supported_shutdown(be)


if __name__=='__main__':main()
