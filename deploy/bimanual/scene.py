"""The bimanual sim scene on the CPU, for the encoder-only real path: kinematic chains, gains, joint limits,
reader/PC reference poses and the Part 1 grasp offsets, built exactly as BimanualRuntime / PickTask / Harness build
them (no MuJoCo Warp, no GPU physics). deploy/checks/bimanual_conventions.py compares every field with a live
BimanualRuntime so the real path cannot drift from the sim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.scenes import pc_bench as S, physical_rig as Rig, reader_grasp as G

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / 'data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json'
LAYOUT = 'results/usbc/policy/pick/task_layout.json'          # eval_chain.PICK_LAYOUT
CONTROLLER_PROFILE = 'native_free_settle_20261028'           # BimanualCfg.controller_profile
SIDES = ('right', 'left')


class SceneModel:
    def __init__(self, device='cpu', task_layout=LAYOUT):
        self.device = dev = torch.device(device)
        m, lay = S.build_scene(detailed=True, bimanual=True, measured_support=True, support_mode='free',
                               profile=json.loads(PROFILE.read_text()), controller_profile=CONTROLLER_PROFILE)
        self.mjm, self.lay = m, lay
        d = mujoco.MjData(m)
        rep = json.loads((ROOT / task_layout).read_text())
        self.layout = {'port_xy': rep['nominal']['port_xy_m'], 'yaw': rep['nominal']['yaw_deg'], 'reader': rep['reader_region']}
        # reference pose and layout placement (BimanualRuntime.__init__)
        Rig.reference_pose(m, d, lay)
        S.place_pc(m, d, port_xy=self.layout['port_xy'], yaw_deg=self.layout['yaw'])
        S.place_reader(m, d, lay, xy=self.layout['reader']['centre_xy_m'], yaw_deg=0.)
        d0 = mujoco.MjData(m)
        S.place_pc(m, d0, port_xy=(0., 0.), yaw_deg=0.)
        S.place_reader(m, d0, lay, xy=(0., 0.), yaw_deg=0.)
        fa0, pa0 = int(m.joint('pc_free_joint').qposadr[0]), lay.peg_qadr
        self._pc0 = torch.tensor(d0.qpos[fa0:fa0 + 7], device=dev, dtype=torch.float32)
        self._reader0 = torch.tensor(d0.qpos[pa0:pa0 + 7], device=dev, dtype=torch.float32)
        self.indices = {}
        for side, pre in (('right', ''), ('left', 'left_')):
            q = [int(m.joint(pre + f'joint{i}').qposadr[0]) for i in range(1, 7)]
            v = [int(m.joint(pre + f'joint{i}').dofadr[0]) for i in range(1, 7)]
            act = [m.actuator(pre + f'joint{i}').id for i in range(1, 7)]
            grip = [m.actuator(pre + 'grip_' + f).id for f in ('left', 'right')]
            lim = np.array([m.joint(pre + f'joint{i}').range for i in range(1, 7)])
            self.indices[side] = (q, v, act, grip, torch.tensor(lim, device=dev, dtype=torch.float32))
            d.ctrl[act] = d.qpos[q]
        mujoco.mj_forward(m, d)
        self.initial_q = torch.tensor(d.qpos.copy(), device=dev, dtype=torch.float32)
        self.pc_qadr = fa0
        cal = Rig.load()
        self.encoder_offsets = torch.tensor([cal['arms'][s]['joint_offsets_rad'] for s in SIDES], device=dev, dtype=torch.float32)
        # Harness: grasp_site chains and PD gains; PickTask: right tcp chain (lay.tcp_site). On the CPU: FastChain
        # (same IK, NumPy FK: one world in a 50 Hz loop); checked against K.Chain in deploy/checks/bimanual_conventions.py
        from deploy.bimanual.fastchain import FastChain
        Ch = (lambda *a_: FastChain(*a_)) if dev.type == 'cpu' else (lambda *a_: K.Chain(*a_, device=dev))
        self.chain = {s: Ch(m, m.site(pre + 'grasp_site').id, np.array(self.indices[s][0]))
                      for s, pre in (('right', ''), ('left', 'left_'))}
        self.tcp_chain = Ch(m, lay.tcp_site, np.array(self.indices['right'][0]))
        self.kp = torch.tensor([[m.actuator_gainprm[a, 0] for a in self.indices[s][2]] for s in SIDES], device=dev, dtype=torch.float32)
        self.kd = torch.tensor([[-m.actuator_biasprm[a, 2] for a in self.indices[s][2]] for s in SIDES], device=dev, dtype=torch.float32)
        self.finger_qadr = [int(m.joint(f + '_finger').qposadr[0]) for f in ('left', 'right')]
        table = m.geom('table')
        self.table_top = float(table.pos[2] + table.size[2])
        # PickTask grasp geometry: hand at the reference pose, grasp offsets per (tilt, flip)
        dr = mujoco.MjData(m)
        Rig.reference_pose(m, dr, lay)
        mujoco.mj_forward(m, dr)
        self.hand = G.hand_geometry(m, dr)
        self.offsets, self.pad_contact_height = {}, {}
        for tilt in (G.TILT_DEG, *G.TILT_FALLBACK_DEG):
            for flip in (False, True):
                self.offsets[(tilt, flip)], self.pad_contact_height[(tilt, flip)] = G.grasp_offset(lay, self.hand, tilt, flip=flip)

    # ------------------------------------------------------------------ PC port frame from the ZED estimate
    def port_frame(self, port_xy, yaw):
        """Port pose (p (N,3), R (N,3,3)) of a flat PC whose port is at port_xy (m) with yaw (rad, the ZED net's
        convention: atan2(-R[1,2], -R[0,2])); height and tilt from the scene's reference PC (on its feet)."""
        port_xy = torch.as_tensor(port_xy, dtype=torch.float32, device=self.device).reshape(-1, 2)
        yaw = torch.as_tensor(yaw, dtype=torch.float32, device=self.device).reshape(-1)
        n = len(yaw)
        R0 = K.quat_to_mat(self._pc0[3:][None])[0]
        yaw0 = math.atan2(-float(R0[1, 2]), -float(R0[0, 2]))
        Rz = K.axis_angle_mat(torch.tensor([0., 0., 1.], device=self.device).expand(n, 3), yaw - yaw0)
        p = torch.cat([port_xy, torch.full((n, 1), float(self._pc0[2]), device=self.device)], 1)
        return p, Rz @ R0

    def tcp_in_table(self, side, q):
        """TCP height above the tabletop (m) at believed angles q (N, 6), right-base frame."""
        p, _ = self.chain[side].fk(torch.as_tensor(q, dtype=torch.float32, device=self.device).reshape(-1, 6))
        return p[:, 2] - self.table_top
