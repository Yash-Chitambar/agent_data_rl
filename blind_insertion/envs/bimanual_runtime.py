"""Measured bimanual GPU runtime for training integration.

Absolute joint targets in the model frame, right arm then left arm, each followed
by an open-positive/close-negative gripper command (14 values). This is a physics
runtime, not yet a validated task reward/reset distribution or deployment policy.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from blind_insertion.envs import friction_comp as FC, kinematics as K
from blind_insertion.scenes import pc_bench as S, physical_rig as R


@dataclass
class BimanualCfg:
    nworld: int = 2
    device: str = 'cuda:0'
    substeps: int = 20
    controller_profile: str = 'native_free_settle_20261028'
    right_fric_comp: dict | None = None
    left_fric_comp: dict | None = None
    grip_open: float = 4.
    grip_close: float = -12.


class BimanualRuntime:
    def __init__(self, cfg=None):
        self.cfg = cfg or BimanualCfg()
        c = self.cfg
        if c.nworld < 1 or c.substeps < 1:
            raise ValueError('nworld and substeps must be positive')
        self.device = torch.device(c.device)
        self.comp = {side: FC.parse(getattr(c, side+'_fric_comp'), side=side) for side in ('right','left')}
        profile_path = Path(__file__).resolve().parents[2] / 'data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json'
        self.mjm, self.lay = S.build_scene(detailed=True, bimanual=True, measured_support=True,
            support_mode='free', profile=json.loads(profile_path.read_text()), controller_profile=c.controller_profile)
        m = self.mjm
        d = mujoco.MjData(m)
        R.reference_pose(m,d,self.lay);S.place_pc(m,d);S.place_reader(m,d,self.lay)
        self.indices = {}
        for side, prefix in (('right',''),('left','left_')):
            q = [int(m.joint(prefix+f'joint{i}').qposadr[0]) for i in range(1,7)]
            v = [int(m.joint(prefix+f'joint{i}').dofadr[0]) for i in range(1,7)]
            act = [m.actuator(prefix+f'joint{i}').id for i in range(1,7)]
            grip = [m.actuator(prefix+'grip_'+finger).id for finger in ('left','right')]
            limits = np.array([m.joint(prefix+f'joint{i}').range for i in range(1,7)])
            self.indices[side] = (q,v,act,grip,torch.tensor(limits,device=self.device,dtype=torch.float32))
            d.ctrl[act] = d.qpos[q]
        mujoco.mj_forward(m,d)
        self.initial_q = torch.tensor(d.qpos.copy(),device=self.device,dtype=torch.float32)
        self.initial_ctrl = torch.tensor(d.ctrl.copy(),device=self.device,dtype=torch.float32)
        self.pc_qadr = int(m.joint('pc_free_joint').qposadr[0])
        wp.init()
        with wp.ScopedDevice(c.device):
            self.m = mjw.put_model(m)
            self.d = mjw.put_data(m,d,nworld=c.nworld,nconmax=512,njmax=3000)
        self.q = wp.to_torch(self.d.qpos)
        self.v = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.force = wp.to_torch(self.d.qfrc_applied)

    def reset(self, ids=None):
        """Reset selected worlds; randomization and task-specific reset poses are separate."""
        if ids is None: ids = torch.arange(self.cfg.nworld,device=self.device)
        self.q[ids] = self.initial_q
        self.v[ids] = 0
        self.ctrl[ids] = self.initial_ctrl
        self.force[ids] = 0
        for name in ('qacc_warmstart','qacc','time'):
            wp.to_torch(getattr(self.d,name))[ids] = 0
        with wp.ScopedDevice(self.cfg.device): mjw.forward(self.m,self.d)
        return self.state()

    def step(self, action):
        action = torch.as_tensor(action,device=self.device,dtype=self.q.dtype)
        if action.shape != (self.cfg.nworld,14) or not torch.isfinite(action).all():
            raise ValueError('action must be finite with shape (nworld,14)')
        self.force.zero_()
        for side, offset in (('right',0),('left',7)):
            q,v,act,grip,limits = self.indices[side]
            target = action[:,offset:offset+6].clamp(limits[:,0],limits[:,1])
            self.ctrl[:,act] = target
            squeeze = torch.where(action[:,offset+6]>0,self.cfg.grip_open,self.cfg.grip_close)
            self.ctrl[:,grip] = squeeze[:,None]
            if self.comp[side] is not None:
                tau,ramp = self.comp[side]
                self.force[:,v] = FC.feedforward(target-self.q[:,q],tau,ramp)
        # Explicit compensation is computed once and held, as in the recorded driver law.
        with wp.ScopedDevice(self.cfg.device):
            for _ in range(self.cfg.substeps): mjw.step(self.m,self.d)
        return self.state()

    def state(self):
        """Privileged state; actor camera/encoder observations need their own sensor pipeline."""
        return {'qpos':self.q.clone(),'qvel':self.v.clone(),'plug_in_port':self.plug_in_port()[0]}

    def plug_in_port(self):
        """Live reader tip relative to the moving PC port, never a cached reset pose."""
        pa,fa = self.lay.peg_qadr,self.pc_qadr
        port_R = K.quat_to_mat(self.q[:,fa+3:fa+7])
        peg_R = K.quat_to_mat(self.q[:,pa+3:pa+7])
        inv = port_R.transpose(1,2)
        return (inv@(self.q[:,pa:pa+3]-self.q[:,fa:fa+3])[...,None])[...,0], inv@peg_R
