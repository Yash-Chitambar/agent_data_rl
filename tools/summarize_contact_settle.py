"""Summarize immutable free-space controller records, without connecting hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('refusing to overwrite report')
    runs=[]
    for path in sorted((a.session/'contact').glob('free_settle_*/trial.json')):
        d=json.loads(path.read_text());summary={'name':path.parent.name,'source':str(path),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'result':d['result'],
            'candidate':d.get('settle_candidate'),'phase_results':d.get('settle_results'),
            'controller':d['controller'],'camera_run':d['camera_run'],'phases':{}}
        for phase in ['baseline_hold','candidate_trim','candidate_frozen_hold']:
            rows=[r for r in d['rows'] if r['phase']==phase]
            if not rows:continue
            errors=np.array([r['model_fk_error_m'] for r in rows]);last=rows[-1]
            q=np.array(last['observations']['left']['q_encoder'][:6]);goal=np.array(last['nominal_left_encoder'])
            summary['phases'][phase]={'ticks':len(rows),'model_fk_error_p50_m':float(np.median(errors)),
                'model_fk_error_p95_m':float(np.quantile(errors,.95)),
                'final_joint_error_rad':(q-goal).tolist(),
                'can_send_rows':sum(r.get('left_first_can_send_host_s') is not None for r in rows)}
        runs.append(summary)
    report={'schema':'left_free_space_comparison_v1','units':'SI; encoder radians; model-FK metres',
        'limitations':['No external tool metrology','No measured force or output torque',
                      'Single trials, not held-out statistical acceptance','No contact or slide commanded',
                      'Candidate results do not authorize controller adoption'], 'runs':runs}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(a.output)

if __name__=='__main__':main()
