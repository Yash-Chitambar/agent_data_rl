"""Check held frames, delayed/out-of-order publication and reset isolation without a GPU."""
import json,tempfile
from pathlib import Path
import torch
from blind_insertion.envs.camera_cadence import CameraCadence
count=[0]
def render():
 count[0]+=1
 return {'c':torch.full((2,1,1,3),count[0],dtype=torch.uint8)}
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'timing.json';p.write_text(json.dumps({'schema':'host-camera-cadence-v1','cameras':{'c':{'capture_s':[0.,.01],'ready_s':[.02,.01],'cycle_s':.1}}}))
 s=CameraCadence(render,2,'cpu',p);s.reset(torch.tensor([0,1]),0)
 initial=s.observation(0)[0]['c'].clone()
 s.advance(0);assert torch.equal(s.observation(0)[0]['c'],initial)
 s.advance(.01);latest=s.observation(.01)[0]['c'].clone()
 assert not torch.equal(latest,initial)
 s.advance(.02);assert torch.equal(s.observation(.02)[0]['c'],latest)
 assert torch.allclose(s.observation(.02)[1]['c'],torch.full((2,),.01))
 # Next cycle queues a delayed old-generation frame, then world 0 resets.
 s.advance(.1);s.reset(torch.tensor([0]),.105);reset=s.observation(.105)[0]['c'].clone()
 s.advance(.12)
 assert s.observation(.12)[0]['c'][0].flatten()[0]>=reset[0].flatten()[0]
 # Returned frame copies must not modify internal camera history.
 frames,_=s.observation(.12);frames['c'].zero_();assert s.observation(.12)[0]['c'].any()
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'timing.json';p.write_text(json.dumps({'schema':'host-camera-cadence-v1','cameras':{'c':{'capture_s':[0.],'ready_s':[.02],'cycle_s':1.}}}))
 s=CameraCadence(render,2,'cpu',p);s.reset(torch.tensor([0,1]),0);s.advance(0)
 s.reset(torch.tensor([0]),.01);fresh=s.observation(.01)[0]['c'][0].clone()
 s.advance(.02)
 assert torch.equal(s.observation(.02)[0]['c'][0],fresh),'pre-reset frame leaked into reset world'
 assert s.observation(.02)[0]['c'][1].flatten()[0]<fresh.flatten()[0],'other world publication suppressed'
report={'checks' :['hold before publication','newest ready frame wins','late old frame discarded','capture age retained','reset generation isolation','returned frames owned'],'scope':'scheduler semantics; exposure latency is unknown'}
Path('results/usbc/bimanual_20261029/camera_cadence_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
