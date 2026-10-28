"""Supervised left/right stationary gripper readback test; no arm trajectories."""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--side',choices=['left','right'],required=True)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--rig-code-root',type=Path,required=True)
    a=p.parse_args()
    if Path(a.name).name!=a.name or a.name in ('.','..'):raise SystemExit('invalid trial name')
    out=a.session/'grasp'/(a.name+'.json')
    if out.exists():raise SystemExit('refusing overwrite')
    sys.path.insert(0,str(a.rig_code_root/'scripts/real'))
    import numpy as np
    import safety,estop
    if estop.stopped():raise SystemExit('software stop is set')
    r=safety.connect(a.side);safe_disconnect=False;rows=[];result='incomplete'
    def prompt(text):
        print(text,flush=True)
        answer=input()
        if estop.stopped():raise RuntimeError('stop set; no further jaw command')
        return answer
    def jaw(g):
        if estop.stopped():raise RuntimeError('software stop')
        q=np.asarray(r.get_joint_pos(),float);q[6]=g;arm.command(q)
        for _ in range(75):
            if estop.stopped():raise RuntimeError('software stop')
            rows.append(dict(host_monotonic_ns=time.monotonic_ns(),target_gripper=g,q_encoder=np.asarray(r.get_joint_pos()).tolist()))
            time.sleep(.02)
        return float(r.get_joint_pos()[6])
    readings={}
    try:
        arm=safety.SafeArm(r);q=np.asarray(r.get_joint_pos(),float)
        if np.max(np.abs(q[1:3]))>.05:raise RuntimeError('arm not at folded rest')
        safe_disconnect=True
        readings['open']=jaw(1.)
        readings['empty']=jaw(.02)
        jaw(1.)
        safe_disconnect=False
        prompt('Place and SUPPORT the reader between the jaws; press Enter to close.')
        readings['reader']=jaw(.02)
        readings['operator_retention']=prompt('Does the reader remain held without slipping? Enter yes/no; keep ready to retrieve it.')
        prompt('Secure the reader by hand; press Enter to open.')
        jaw(1.);safe_disconnect=True;result='completed'
    except Exception as exc:
        result=repr(exc);estop.STOP_FILE.write_text('stop: grasp trial interrupted\n');print(result,flush=True)
    finally:
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(dict(side=a.side,result=result,readings=readings,samples=rows,force_measured=False),indent=1)+'\n')
        if not safe_disconnect:
            print('Power retained. Secure the arm and object. Type SECURED to disable torque.',flush=True)
            while True:
                try: answer=input()
                except (EOFError,KeyboardInterrupt):time.sleep(.5);continue
                if answer.strip()=='SECURED':break
        safety.close_robot(r)

if __name__=='__main__':main()
