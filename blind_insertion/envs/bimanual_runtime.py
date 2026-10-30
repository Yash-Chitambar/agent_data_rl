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
from blind_insertion.envs import arm_residual as AR, friction_comp as FC, kinematics as K
from blind_insertion.scenes import pc_bench as S, physical_rig as R


@dataclass
class BimanualCfg:
    nworld: int = 2
    device: str = 'cuda:0'
    substeps: int = 20
    controller_profile: str = 'native_free_settle_20261028'
    # Identified real-arm dynamics (scripts/tools/identify_arm_dynamics.py, per arm): the
    # driver's own gravity feedforward replaces perfect gravity cancellation, the plant
    # gravity is the scene's scaled per joint, and joint friction is the measured Coulomb
    # with a stiff (non-creeping) impedance. None restores the legacy ideal-gravcomp plant.
    dynamics_profile: str | None = 'results/usbc/bimanual_20261029/dynamics_id/profile.json'
    # Per-world, per-joint randomization around the profile, resampled on reset. The Coulomb
    # range covers the day-to-day spread of the fits (e.g. left j2 0.40-1.03 N m); the gravity
    # jitter covers the by-day scale differences (<= 0.07).
    friction_scale_range: tuple = (0.5, 1.3)
    gravity_scale_jitter: float = 0.04
    # Opt-in fingertip offset fitted to wrist silhouettes (scripts/tools/fit_gripper_tips.py):
    # 'fitted' reads results/usbc/bimanual_20261029/gripper_tips/fit.json. Off by default: the
    # silhouettes cannot separate tip geometry from a hand-eye translation error (measure first).
    tip_offset: str | None = None
    # Legacy plant only (dynamics_profile=None): right joint-2 friction from the older replays.
    right_joint2_friction_nm: float = 1.0
    right_fric_comp: dict | None = None
    left_fric_comp: dict | None = None
    grip_open: float = 4.
    grip_close: float = -12.
    render_rgb: bool = False
    measured_camera_cadence: bool = True
    quantize_encoders: bool = True
    camera_resolution: int = 320


class BimanualRuntime:
    def __init__(self, cfg=None):
        self.cfg = cfg or BimanualCfg()
        c = self.cfg
        if not 16 <= c.camera_resolution <= 2048:
            raise ValueError('camera_resolution must be between 16 and 2048')
        if c.nworld < 1 or c.substeps < 1:
            raise ValueError('nworld and substeps must be positive')
        if not np.isfinite(c.right_joint2_friction_nm) or c.right_joint2_friction_nm < 0:
            raise ValueError('right_joint2_friction_nm must be finite and nonnegative')
        self.device = torch.device(c.device)
        self.comp = {side: FC.parse(getattr(c, side+'_fric_comp'), side=side) for side in ('right','left')}
        profile_path = Path(__file__).resolve().parents[2] / 'data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json'
        self.mjm, self.lay = S.build_scene(detailed=True, bimanual=True, measured_support=True,
            support_mode='free', profile=json.loads(profile_path.read_text()), controller_profile=c.controller_profile)
        m = self.mjm
        self.profile = None
        if c.dynamics_profile:
            lo, hi = c.friction_scale_range
            if not 0 <= lo <= hi or not np.isfinite(hi) or not 0 <= c.gravity_scale_jitter < .5:
                raise ValueError('friction_scale_range must be 0 <= lo <= hi and gravity_scale_jitter in [0, 0.5)')
            self.profile = json.loads((Path(__file__).resolve().parents[2] / c.dynamics_profile).read_text())
            for side, prefix in (('right', ''), ('left', 'left_')):
                dofs = [int(m.joint(prefix + f'joint{i}').dofadr[0]) for i in range(1, 7)]
                m.dof_frictionloss[dofs] = self.profile['arms'][side]['coulomb_nm']
                m.dof_solref[dofs] = self.profile['friction_solref']
                m.dof_solimp[dofs] = self.profile['friction_solimp']
        else:
            m.dof_frictionloss[int(m.joint("joint2").dofadr[0])] = c.right_joint2_friction_nm
        if c.tip_offset == 'fitted':
            fit = json.loads((Path(__file__).resolve().parents[2] / 'results/usbc/bimanual_20261029/gripper_tips/fit.json').read_text())
            for side, prefix in (('right', ''), ('left', 'left_')):
                off = np.array(list(fit['arms'][side]['fitted_offset_mm'].values())) * 1e-3   # gripper-frame x, y, z
                m.body_pos[[m.body(prefix + f + '_finger').id for f in ('left', 'right')]] += off
        elif c.tip_offset is not None:
            raise ValueError("tip_offset must be None or 'fitted'")
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
            self.m = mjw.put_model(m, batch_sizes={'dof_frictionloss': c.nworld} if self.profile else None)
            self.d = mjw.put_data(m,d,nworld=c.nworld,nconmax=512,njmax=3000)
        self.q = wp.to_torch(self.d.qpos)
        self.v = wp.to_torch(self.d.qvel)
        self.ctrl = wp.to_torch(self.d.ctrl)
        self.force = wp.to_torch(self.d.qfrc_applied)
        self.sensor_time = 0.0
        self.camera_sensor = None
        self.camera_names = ('top','wrist_cam','left_wrist_cam')
        calibration = R.load()
        self.encoder_offsets = torch.tensor(
            [calibration['arms'][side]['joint_offsets_rad'] for side in ('right','left')],
            device=self.device,dtype=self.q.dtype)
        if self.profile:
            arms = [self.profile['arms'][side] for side in ('right','left')]
            f = lambda key: torch.tensor([arm[key] for arm in arms],device=self.device,dtype=self.q.dtype)
            self.coulomb, self.gravity_scale_base = f('coulomb_nm'), f('gravity_scale')
            self.driver_factor = f('driver_gravity_comp_factor')
            self.frictionloss = wp.to_torch(self.m.dof_frictionloss)
            self.gravcomp = wp.to_torch(self.d.qfrc_gravcomp)
            self.gravity_scale = self.gravity_scale_base.expand(c.nworld,2,6).clone()
            self.driver_gravity = AR.DriverGravity(self.device)
            self._sample_dynamics(torch.arange(c.nworld,device=self.device))
        if c.render_rgb:
            ids = [m.camera(name).id for name in self.camera_names]
            active = [i in ids for i in range(m.ncam)]
            order = [i for i in range(m.ncam) if active[i]]
            self.camera_indices = [order.index(i) for i in ids]
            res = c.camera_resolution
            with wp.ScopedDevice(c.device):
                self.rc = mjw.create_render_context(m,nworld=c.nworld,cam_res=[(res,res)]*len(ids),
                    render_rgb=True,render_depth=False,use_textures=True,use_shadows=False,cam_active=active)
                self.rgb_buffer = wp.zeros((c.nworld,res,res),dtype=wp.vec3)
            if c.measured_camera_cadence:
                from blind_insertion.envs.camera_cadence import CameraCadence
                self.camera_sensor=CameraCadence(self.render,c.nworld,self.device)
                self.camera_sensor.reset(torch.arange(c.nworld,device=self.device),self.sensor_time)


    def reset(self, ids=None):
        """Reset selected worlds; randomization and task-specific reset poses are separate."""
        if ids is None: ids = torch.arange(self.cfg.nworld,device=self.device)
        self.q[ids] = self.initial_q
        self.v[ids] = 0
        self.ctrl[ids] = self.initial_ctrl
        self.force[ids] = 0
        for name in ('qacc_warmstart','qacc','time'):
            wp.to_torch(getattr(self.d,name))[ids] = 0
        if self.profile: self._sample_dynamics(ids)
        with wp.ScopedDevice(self.cfg.device): mjw.forward(self.m,self.d)
        if self.camera_sensor is not None:self.camera_sensor.reset(ids,self.sensor_time)
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
        if self.profile:
            for k,side in enumerate(('right','left')):
                self.force[:,self.indices[side][1]] += self.gravity_residual(k)
        # Explicit compensation is computed once and held, as in the recorded driver law.
        with wp.ScopedDevice(self.cfg.device):
            for _ in range(self.cfg.substeps):
                mjw.step(self.m,self.d)
                self.sensor_time+=self.mjm.opt.timestep
                if self.camera_sensor is not None:self.camera_sensor.advance(self.sensor_time)
        return self.state()

    def _sample_dynamics(self, ids):
        """Per-world Coulomb friction and gravity scales around the identified profile."""
        lo,hi = self.cfg.friction_scale_range
        n = len(ids)
        for k,side in enumerate(('right','left')):
            u = torch.rand(n,6,device=self.device)*(hi-lo)+lo
            self.frictionloss[ids[:,None],torch.tensor(self.indices[side][1],device=self.device)] = self.coulomb[k]*u
        jitter = (torch.rand(n,2,6,device=self.device)*2-1)*self.cfg.gravity_scale_jitter
        jitter[...,0] = 0                                       # joint 1 carries no gravity
        self.gravity_scale[ids] = self.gravity_scale_base+jitter

    def gravity_residual(self, k):
        """Driver gravity feedforward at the encoder angles minus the scaled plant gravity that
        the scene's gravcomp cancels; held for the control step like the driver's update."""
        side = ('right','left')[k]
        q,v = self.indices[side][0],self.indices[side][1]
        q_enc = self.q[:,q]-self.encoder_offsets[k]
        return self.driver_factor[k]*self.driver_gravity(q_enc)-self.gravity_scale[:,k]*self.gravcomp[:,v]

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

    def proprioception(self):
        """Twelve encoder angles/velocities, right then left, without privileged object poses.

        Gripper state is deliberately not labeled as hardware-normalized position:
        that sensor-to-opening map has not yet been identified.
        """
        q = torch.stack([self.q[:,self.indices[side][0]] for side in ('right','left')],dim=1)
        v = torch.stack([self.v[:,self.indices[side][1]] for side in ('right','left')],dim=1)
        enc=q-self.encoder_offsets
        if self.cfg.quantize_encoders:
            # DM native position telemetry: unsigned 16-bit encoding over +/-12.5 rad.
            enc=((enc+12.5)*(65535./25.)).round().clamp(0,65535)*(25./65535.)-12.5
        return {'encoder_rad':enc.flatten(1),'encoder_velocity_rad_s':v.flatten(1)}

    def render(self):
        """Instantaneous calibrated RGB frames; exposure/frame-age modeling remains separate."""
        if not self.cfg.render_rgb:
            raise RuntimeError('construct with render_rgb=True to request camera observations')
        with wp.ScopedDevice(self.cfg.device):
            mjw.kinematics(self.m,self.d)
            mjw.camlight(self.m,self.d)
            mjw.refit_bvh(self.m,self.d,self.rc)
            mjw.render(self.m,self.d,self.rc)
            frames = {}
            for name,index in zip(self.camera_names,self.camera_indices):
                mjw.get_rgb(self.rc,index,self.rgb_buffer)
                frames[name]=(wp.to_torch(self.rgb_buffer).clamp(0,1)*255).to(torch.uint8)
        return frames

    def insertion_metrics(self, seat_tolerance_m=.00025, tilt_limit_deg=2.):
        """Geometric insertion checks in the live port frame, not electrical continuity.

        Every sampled plug-edge point must fit the aperture. Both depth bounds
        exclude a part teleported through the computer from counting as seated.
        """
        if not 0 < seat_tolerance_m < float(self.lay.depth[0]):
            raise ValueError('seat tolerance must be positive and less than socket depth')
        if not 0 < tilt_limit_deg < 90:
            raise ValueError('tilt limit must be between 0 and 90 degrees')
        p,rotation = self.plug_in_port()
        xy=torch.as_tensor(self.lay.edge_points[0],device=self.device,dtype=self.q.dtype)
        points=torch.cat((xy,torch.zeros((len(xy),1),device=self.device,dtype=self.q.dtype)),dim=1)
        points=points[None]@rotation.transpose(1,2)+p[:,None]
        polygon=torch.as_tensor(self.lay.hole_polys[0],device=self.device,dtype=self.q.dtype)
        edge=torch.roll(polygon,-1,0)-polygon
        delta=points[:,:,:2,None].transpose(2,3)-polygon[None,None]
        cross=edge[None,None,:,0]*delta[:,:,:,1]-edge[None,None,:,1]*delta[:,:,:,0]
        inside=(cross>=-1e-9).all(dim=2).all(dim=1)
        depth=float((self.lay.seating_depth if self.lay.seating_depth is not None else self.lay.depth)[0])
        axial=points[:,:,2]
        tilt=torch.rad2deg(torch.acos(rotation[:,2,2].clamp(-1,1)))
        seated=inside & (axial<=-depth+seat_tolerance_m).all(1) & (axial>=-depth-seat_tolerance_m).all(1)
        finite=torch.isfinite(self.q).all(1)
        return {'geometric_seated':seated & (tilt<=tilt_limit_deg) & finite,
                'aperture_fit':inside,'tip_depth_m':-p[:,2],'tilt_deg':tilt,'finite':finite}
