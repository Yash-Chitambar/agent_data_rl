"""Episode interface for bimanual USB-C insertion (physics validation in progress).

Actions are normalized right-arm six joint increments and gripper, then left.
Rewards/hold duration are task design choices, not measurements of hardware.
Resets currently use the runtime's fixed reference layout, not a measured pose distribution.
"""
from dataclasses import dataclass, field
import math
import torch
from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime


@dataclass
class BimanualTaskCfg:
    physics: BimanualCfg = field(default_factory=BimanualCfg)
    episode_s: float = 30.
    success_hold_s: float = .2
    action_scale_rad: float = .04
    seat_tolerance_m: float = .00025
    tilt_limit_deg: float = 2.


class BimanualTask:
    def __init__(self, cfg=None):
        self.cfg=cfg or BimanualTaskCfg()
        c=self.cfg
        if any(not math.isfinite(x) or x<=0 for x in (c.episode_s,c.success_hold_s,c.action_scale_rad)):
            raise ValueError('episode duration, success hold and action scale must be positive finite values')
        self.runtime=BimanualRuntime(c.physics)
        self.device=self.runtime.device
        self.dt=self.runtime.mjm.opt.timestep*c.physics.substeps
        self.max_steps=math.ceil(c.episode_s/self.dt)
        self.hold_steps=math.ceil(c.success_hold_s/self.dt)
        n=c.physics.nworld
        self.steps=torch.zeros(n,device=self.device,dtype=torch.long)
        self.seated_steps=torch.zeros_like(self.steps)
        self.done=torch.zeros(n,device=self.device,dtype=torch.bool)
        self.previous_action=torch.zeros((n,14),device=self.device)

    def observation(self):
        obs=self.runtime.proprioception()
        obs['previous_action']=self.previous_action.clone()
        if self.cfg.physics.render_rgb:obs['rgb']=self.runtime.render()
        return obs

    def reset(self, ids=None):
        if ids is None:ids=torch.arange(self.cfg.physics.nworld,device=self.device)
        self.runtime.reset(ids)
        self.steps[ids]=0;self.seated_steps[ids]=0;self.done[ids]=False;self.previous_action[ids]=0
        return self.observation()

    def step(self, action):
        if self.done.any():
            raise RuntimeError('reset terminated/truncated worlds before stepping again')
        a=torch.as_tensor(action,device=self.device,dtype=self.previous_action.dtype)
        if a.shape!=self.previous_action.shape or not torch.isfinite(a).all():
            raise ValueError('normalized action must be finite with shape (nworld,14)')
        a=a.clamp(-1,1)
        targets=a.clone()
        for side,offset in (('right',0),('left',7)):
            q=self.runtime.indices[side][0]
            targets[:,offset:offset+6]=self.runtime.q[:,q]+self.cfg.action_scale_rad*a[:,offset:offset+6]
        self.runtime.step(targets)
        self.previous_action.copy_(a)
        reward,terminated,truncated,info=self._transition()
        return self.observation(),reward,terminated,truncated,info

    def _transition(self):
        """Update episode status after physics; do not reset before returning terminal observations."""
        e=self.runtime;c=self.cfg
        metrics=e.insertion_metrics(c.seat_tolerance_m,c.tilt_limit_deg)
        self.steps+=1
        table=e.mjm.geom('table')
        top=float(table.pos[2]+table.size[2])
        reader=e.q[:,e.lay.peg_qadr:e.lay.peg_qadr+3]
        pc=e.q[:,e.pc_qadr:e.pc_qadr+3]
        bad=~metrics['finite'] | ~torch.isfinite(e.v).all(1)
        dropped=(reader[:,2]<top-.02) | (pc[:,2]<top-.02)
        failure=bad | dropped
        seated=metrics['geometric_seated'] & ~failure
        self.seated_steps=torch.where(seated,self.seated_steps+1,torch.zeros_like(self.seated_steps))
        success=(self.seated_steps>=self.hold_steps) & ~failure
        terminated=success | failure
        truncated=(self.steps>=self.max_steps) & ~terminated
        self.done=terminated | truncated
        info={'success':success.clone(),'dropped':dropped,'nonfinite':bad,
              'geometric_seated':metrics['geometric_seated'],'steps':self.steps.clone(),
              'privileged':e.state()}
        return success.float(),terminated,truncated,info
