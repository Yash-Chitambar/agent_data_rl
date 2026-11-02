# servo net outputs on real vs dry frames, from the logged deltas: the roll and tilt of the target each correction commands.
import json, glob, numpy as np, torch, sys
from deploy.bimanual.scene import SceneModel
from blind_insertion.envs import kinematics as K
sc=SceneModel('cpu'); off=sc.encoder_offsets.numpy(); ch=sc.chain['right']
rows=[]
for run in sorted(glob.glob('data/real_runs/2026-11-01/pick_*')+glob.glob('data/real_runs/2026-11-01/full_*dry*')):
    try: z=np.load(run+'/ticks.npz')
    except Exception: continue
    ev=[json.loads(l) for l in open(run+'/events.jsonl')]
    q=z['q'][:,0]+off[0]; k=z['k'].astype(int); tt=z['t']
    meta=json.load(open(run+'/meta.json')); dry=bool(meta.get('dry_run'))
    fr={}
    for e in ev:
        if e['kind']=='servo_frame':
            # the frame's tick: frames.jsonl by tag
            fr[e.get('frame_tag')]=e
    ticks={t:int(t.split('@')[1]) for t in fr if t and '@' in t}
    for tag,e in fr.items():
        d=np.array(e['delta']);kk=ticks.get(tag)
        if kk is None or np.isnan(d[3:]).any(): continue
        i=np.searchsorted(k,kk); i=min(i,len(q)-1)
        _,Rh=ch.fk(torch.tensor(q[i:i+1],dtype=torch.float32)); rv=torch.tensor(d[3:][None],dtype=torch.float32)
        a=rv.norm(dim=1); Rt=Rh@K.axis_angle_mat(rv/a,a)
        rh=np.degrees(np.arcsin(float(Rh[0,2,1]))); rt=np.degrees(np.arcsin(float(Rt[0,2,1])))
        rows.append(dict(run=run.split('/')[-1],dry=dry,kind=e['frame_kind'],k=e.get('servo_k'),hand_roll=rh,target_roll=rt,rv=np.degrees(d[3:]).round(2).tolist()))
import pandas as pd
df=pd.DataFrame(rows); pd.set_option('display.width',200,'display.max_rows',200,'display.max_colwidth',40)
print(df.round(2).to_string())
print(df.groupby(['dry','k'])[['hand_roll','target_roll']].agg(['mean','std','count']).round(2))
df.to_json(sys.argv[1])
