import sys,json,torch,numpy as np,cv2
sys.path.insert(0,'.'); sys.path.insert(0,'scripts/perception')
np.set_printoptions(precision=2,suppress=True)
import train_approach_estimators as T
from blind_insertion.perception.render_data import ApproachRenderer
from blind_insertion.envs import kinematics as K, inhand_real as IR
from deploy.bimanual.io_sim import eval_chain
EC=eval_chain(); net,sc,mode=EC.wrist_loader('../perception_real/wrist_r10f_ft_abs','cuda:0'); T.MODE.update(wrist=mode)
z=np.load(sys.argv[1]); q=torch.tensor(z['q'],device='cuda:0'); n=len(q)
r=ApproachRenderer(nworld=n,seed=3,task_layout='results/usbc/policy/pick/task_layout.json',inhand_real='v2',pc_real_look='blind_insertion/assets/pc_real_look')
rt=r.rt;h=r.h
fr0,lab0=r.sample_states(q)
y0=T.targets('wrist',lab0); ih0=lab0['inhand6']
print('state labels',y0[:3].cpu().numpy()); print('state inhand6',ih0.cpu().numpy()[:4])
with torch.no_grad(): o0=net(T.wrist_img(fr0['wrist_cam']))*sc
print('net on state',(o0-y0).mean(0).cpu().numpy())
# place the same relative pose from the sampler code path, in-hand = the state's own
R0,p0=None,None
r.ins.cfg.inhand='real'; r.ins.cfg.inhand_real_preset='v2'; r.ins.cfg.inhand_real_spread=0.
rel_p=y0[:,:3]*1e-3
rv=torch.deg2rad(y0[:,3:]); rel_R=K.axis_angle_mat(rv/rv.norm(dim=1,keepdim=True).clamp(min=1e-9),rv.norm(dim=1))
ok=r.ins.place(rel_p,rel_R,resample_inhand=True)
fr1=rt.render()
print('place ok',ok.cpu().numpy(),'place inhand6',r.inhand6().cpu().numpy()[:2])
with torch.no_grad(): o1=net(T.wrist_img(fr1['wrist_cam']))*sc
print('net on place',(o1-y0).mean(0).cpu().numpy())
q1=rt.q.clone()
ra=rt.indices['right'][0];la=rt.indices['left'][0]
print('right q diff mean',(q1[:,ra]-q[:,ra]).abs().mean(0).cpu().numpy())
print('left q state',q[0,la].cpu().numpy(),'place',q1[0,la].cpu().numpy())
print('pc state',q[0,rt.pc_qadr:rt.pc_qadr+7].cpu().numpy(),'place',q1[0,rt.pc_qadr:rt.pc_qadr+7].cpu().numpy())
a=np.hstack([fr0['wrist_cam'][0].cpu().numpy(),fr1['wrist_cam'][0].cpu().numpy()]); cv2.imwrite(sys.argv[2],cv2.cvtColor(a,cv2.COLOR_RGB2BGR))
# one change at a time from the placed pose
def ev(tag):
    with wp.ScopedDevice(rt.cfg.device): mjw.forward(rt.m,rt.d)
    fr=rt.render()
    with torch.no_grad(): o=net(T.wrist_img(fr['wrist_cam']))*sc
    y=torch.cat([h_rel()[0]*1e3,torch.rad2deg(h_rel()[1])],1)
    print(tag,'err',(o-y).mean(0).cpu().numpy(),'sd',(o-y).std(0).cpu().numpy(),flush=True); return fr
import warp as wp, mujoco_warp as mjw
def h_rel():
    pp,pR=h.pc_pose(); tip,rR=h.reader_pose()
    return (pR.transpose(1,2)@(tip-pp)[...,None])[...,0],K.rot_log(pR.transpose(1,2)@rR)
ev('placed, chain left arm')
port_p,port_R=h.pc_pose(); plan=r.hold.solve(port_p,port_R)
rt.q[:,rt.indices['left'][0]]=plan['q_contact']; rt.q[:,r.left_fingers]=0.
fr2=ev('placed, generator left arm')
fing=[int(rt.mjm.joint(f+'_finger').qposadr[0]) for f in('left','right')]
print('right fingers state',q[0,fing].cpu().numpy(),'place',rt.q[0,fing].cpu().numpy())
rt.q[:,fing]=q[:,fing]; ev('+ state right fingers')
print('left fingers state',q[0,r.left_fingers].cpu().numpy())
