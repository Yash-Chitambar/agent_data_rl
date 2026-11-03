import sys,json,numpy as np,torch
sys.path.insert(0,'.')
np.set_printoptions(precision=2,suppress=True)
from deploy.run_bimanual import parse, Runner
from deploy.bimanual.chain import RealChain
LOOK={"pc_real_look":"blind_insertion/assets/pc_real_look","inhand_real":"v2"}
a=parse(['--dry-run','--stage','approach_insert','--seed','7600','--config','deploy/config/bimanual_align2.json','--sim-phys',json.dumps(LOOK),'--sim-inhand','v2','--align-mode','measure','--align-set','measure_n=1'])
r=Runner(a); r.open_io(None); io=r.io
class Done(Exception): pass
Q,Fr,Nt,Tr=[],[],[],[]
for seed in range(7600,7608):
    io.reset(seed); log=r.new_logger(seed); ch=r.make_chain(log); ch.logger=None; io.start_clock(); ch.seed=seed
    orig=ch.stage_approach
    def ap(cmd,qh,ep,eR,lg):
        c=orig(cmd,qh,ep,eR,lg)
        fr=ch.render(('wrist_right',),tag='x')['wrist_cam']
        rq=io.rt.q.clone()
        import torch.nn.functional as F
        net,scale,_=ch.fine
        img=F.interpolate(fr.permute(0,3,1,2).float(),size=(192,192),mode='area').round().byte().permute(0,2,3,1)
        with torch.no_grad(): o=(net(img)*scale).to('cpu').numpy()[0]
        tr=io.truth()
        Q.append(rq[0].cpu().numpy()); Fr.append(fr[0].cpu().numpy()); Nt.append(o); Tr.append(np.r_[tr['plug_in_port_mm'],tr['plug_rot_deg']])
        print(seed,'chain net',o,'truth',Tr[-1],flush=True)
        raise Done()
    ch.stage_approach=ap
    try: r.run_stages(ch,{})
    except Done: pass
    except Exception as e: print('stop',e)
    log.close()
np.savez(sys.argv[1],q=np.array(Q),fr=np.array(Fr),net=np.array(Nt),truth=np.array(Tr))
