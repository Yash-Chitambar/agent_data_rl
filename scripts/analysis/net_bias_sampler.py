import sys,json,torch,numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'scripts/perception')
np.set_printoptions(precision=2,suppress=True)
import train_approach_estimators as T
from blind_insertion.perception.render_data import ApproachRenderer, SampleCfg
from deploy.bimanual.io_sim import eval_chain
EC=eval_chain(); net,sc,mode=EC.wrist_loader('../perception_real/wrist_r10f_ft_abs','cuda:0'); T.MODE.update(wrist=mode)
r=ApproachRenderer(nworld=256,seed=5,task_layout='results/usbc/policy/pick/task_layout.json',inhand_real='v2',pc_real_look='blind_insertion/assets/pc_real_look')
r.ins.cfg.inhand='real'; from blind_insertion.envs import inhand_real as IR; cc=dict(IR.load('v2')['inhand'],render_lateral_mm=float(sys.argv[1])); r.ins.cfg.inhand_real_preset=cc; r.ins.cfg.inhand_real_spread=1.
Y,O=[],[]
for k in range(8):
    fr,lab=r.sample(SampleCfg(lateral_m=.003,gap_m=(.003,.009),rot_deg=2.))
    with torch.no_grad(): o=net(T.wrist_img(fr['wrist_cam']))*sc
    y=T.targets('wrist',lab); m=lab['ok']; Y.append(y[m].cpu().numpy()); O.append(o[m].cpu().numpy())
Y=np.concatenate(Y);O=np.concatenate(O);E=O-Y
print('n',len(Y),'err mean',E.mean(0),'sd',E.std(0))
C=np.linalg.lstsq(np.c_[Y,np.ones(len(Y))],O,rcond=None)[0]
print('O = Y C + c: rows Y axes, cols O axes\n',C)
for lo,hi in [(-7,-4),(-4,-2),(-2,2),(2,4),(4,7)]:
    m=(Y[:,3]>=lo)&(Y[:,3]<hi); print('rx in',lo,hi,'n',m.sum(),'net rx',O[m,3].mean().round(2),'err all',E[m].mean(0))
# conditional: the chain's state class
m=(Y[:,3]<-3)&(Y[:,4]<-.5)&(np.abs(Y[:,0])<4)&(Y[:,2]<6.5); print('chain-like states n',m.sum(),'net',O[m].mean(0),'truth',Y[m].mean(0))
print('chain truth state: net predicted by linear fit at truth (-1.8,-1.4,4.5,-4.8,-2,-.8):',np.array([-1.8,-1.4,4.5,-4.8,-2,-.8,1.])@C)
