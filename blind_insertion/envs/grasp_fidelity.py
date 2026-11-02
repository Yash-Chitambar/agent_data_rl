"""Realistic reader-grasp contact model (BimanualCfg.grasp_fidelity; off by default).

The baseline sim grasps 100 % where the real robot held 2 of 8 closes (docs/real/PICK_GRASP_SIM_HANDOFF_20261101.md).
This module holds what the baseline lacks, as one opt-in block shared by the GPU runtime and everything built on it
(pick_expert_eval, eval_chain, the deploy dry run deploy/bimanual/io_sim.py). Guide: docs/real/GRASP_FIDELITY.md.

Fields (per-world ranges are uniform (lo, hi), resampled on reset; None keeps the baseline):
  tip_solref, tip_solimp  stiff right fingertip contact (the baseline's 20 ms default lets the tips sink under the press)
  tip_extend_m            right tip collision hulls moved along the approach (m, per world): the hand stops higher above
                          the table than the model FK says, so the pads sit higher on the reader. The real logs show the
                          tips blocked at FK +2..+3 mm; this does not say whether the error is tip, pad or FK geometry.
  tip_friction            tip/table mu (per world); the tip geoms get contact priority so this value sets the pair
  wrist_kp_scale          scale on right joints 4-6 kp and kd (per world): weak DM4310 wrist
  wrist_friction_nm       right joints 4-6 Coulomb friction (per world, replaces the identified value)
  reader_fillet_m         radius of the reader body's four long edges (per world, m): the body box stops colliding and
                          2 boxes + 4 capsules take over (invisible; the render keeps its visual shell)
  reader_body_length_m    silver body length (m), shortened from the back end. Changes the reader's look and mass
                          distribution, so vision nets trained on the old length see a different part.
  pad_friction            pad/reader mu (per world), replaces PickTaskCfg.pad_friction_range
  pad_timeconst           pad contact time constant (per world, s): pad compliance
  servo_err               [[roll_deg, tilt_deg], ...]: per world one row is drawn (+ servo_err_jitter_deg) and composed
                          on the rotation of every servo correction. Roll tips the closing axis from horizontal, so one
                          finger stands on the table and the other's pads ride high. Bootstrap of the real target poses.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ROOT / 'blind_insertion/configs'
PRIMS = tuple(f'reader_fillet_{n}' for n in ('box_w', 'box_t', 'cap_pp', 'cap_pm', 'cap_mp', 'cap_mm'))


@dataclass
class GraspFidelityCfg:
    tip_solref: tuple | None = None
    tip_solimp: tuple | None = None
    tip_extend_m: tuple | None = None
    tip_friction: tuple | None = None
    wrist_kp_scale: tuple | None = None
    wrist_friction_nm: tuple | None = None
    reader_fillet_m: tuple | None = None
    reader_body_length_m: float | None = None
    pad_friction: tuple | None = None
    pad_timeconst: tuple | None = None
    servo_err: list | None = None
    servo_err_jitter_deg: float = 0.3

    @staticmethod
    def make(x):
        """None, a cfg, a dict, a preset name (configs/grasp_fidelity_<name>.json) or a JSON path."""
        if x is None or isinstance(x, GraspFidelityCfg):
            return x
        if isinstance(x, str):
            p = PRESETS / f'grasp_fidelity_{x}.json'
            x = json.loads((p if p.exists() else ROOT / x).read_text())
        known = {f.name for f in fields(GraspFidelityCfg)}
        x = {k: v for k, v in x.items() if not k.startswith('_')}
        if set(x) - known:
            raise ValueError(f'unknown grasp_fidelity keys {sorted(set(x) - known)}')
        return GraspFidelityCfg(**x)

    def to_dict(self):
        return asdict(self)


def add_spec(spec, c):
    """pc_bench.build_scene(extra_spec=...): collision primitives of the rounded reader body. Sizes are placeholders,
    set by apply_model (the layout's body box) and per world."""
    if c.reader_fillet_m is None:
        return
    body = next(b for b in spec.bodies if b.name == 'peg')
    for n in PRIMS:
        cap = 'cap' in n
        body.add_geom(name=n, type=mujoco.mjtGeom.mjGEOM_CAPSULE if cap else mujoco.mjtGeom.mjGEOM_BOX,
                      size=[.001, .001, 0] if cap else [.001, .001, .001], pos=[0, 0, .06], mass=0,
                      rgba=[0, 0, 0, 0], group=3)


def _prim_values(box, r):
    """(pos (6,3), size (6,3), aabb half (6,3), rbound (6,)) of the primitives for fillet radii r (N,); box = the
    layout's ((cx, cy, cz), (hx, hy, hz))."""
    (cx, cy, cz), (hx, hy, hz) = box
    N, dev = len(r), r.device
    pos = torch.tensor([cx, cy, cz], device=dev, dtype=r.dtype).expand(N, 6, 3).clone()
    size = torch.zeros(N, 6, 3, device=dev, dtype=r.dtype)
    size[:, 0] = torch.stack([hx - r, torch.full_like(r, hy), torch.full_like(r, hz)], 1)    # full thickness, inner width
    size[:, 1] = torch.stack([torch.full_like(r, hx), hy - r, torch.full_like(r, hz)], 1)    # full width, flat side band
    half = size.clone()
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1)), start=2):
        pos[:, i, 0] = cx + sx * (hx - r)
        pos[:, i, 1] = cy + sy * (hy - r)
        size[:, i, 0], size[:, i, 1] = r, hz - r
        half[:, i] = torch.stack([r, r, torch.full_like(r, hz)], 1)
    rbound = half.norm(dim=2)
    rbound[:, 2:] = hz
    return pos, size, half, rbound


def apply_model(m, lay, c):
    """CPU-model edits before the GPU model is built. Returns the geom index sets."""
    out = {'tips': [m.geom(f + '_tip_hull').id for f in ('left', 'right')], 'prims': []}
    if c.tip_solref is not None or c.tip_solimp is not None or c.tip_friction is not None:
        m.geom_priority[out['tips']] = 2     # above the table: these values alone set the tip/table pair
    if c.tip_solref is not None:
        m.geom_solref[out['tips'], :2] = c.tip_solref
    if c.tip_solimp is not None:
        m.geom_solimp[out['tips'], :3] = c.tip_solimp
    if c.reader_fillet_m is not None:
        slots = list(lay.peg_slots)
        body_slot = slots[int(np.argmax(m.geom_rbound[slots]))]
        out['prims'] = [m.geom(n).id for n in PRIMS]
        for g in out['prims']:
            m.geom_contype[g], m.geom_conaffinity[g] = m.geom_contype[body_slot], m.geom_conaffinity[body_slot]
            m.geom_condim[g], m.geom_priority[g] = m.geom_condim[body_slot], m.geom_priority[body_slot]
            m.geom_solref[g], m.geom_solimp[g] = m.geom_solref[body_slot], m.geom_solimp[body_slot]
            m.geom_friction[g] = m.geom_friction[body_slot]
            m.geom_sameframe[g] = 0
        m.geom_contype[body_slot] = m.geom_conaffinity[body_slot] = 0
        r = torch.tensor([float(np.mean(c.reader_fillet_m))], dtype=torch.float64)
        pos, size, half, rb = _prim_values(lay.variants[0].peg_boxes[0], r)
        m.geom_pos[out['prims']] = pos[0].numpy()
        m.geom_size[out['prims']] = size[0].numpy()
        m.geom_aabb[out['prims']] = np.concatenate([np.zeros((6, 3)), half[0].numpy()], 1)
        m.geom_rbound[out['prims']] = rb[0].numpy()
    return out


def batch_fields(c):
    f = set()
    if c.tip_extend_m is not None or c.reader_fillet_m is not None:
        f.add('geom_pos')
    if c.reader_fillet_m is not None:
        f |= {'geom_size', 'geom_aabb', 'geom_rbound'}
    if c.tip_friction is not None or c.pad_friction is not None:
        f.add('geom_friction')
    if c.pad_timeconst is not None:
        f.add('geom_solref')
    if c.wrist_kp_scale is not None:
        f |= {'actuator_gainprm', 'actuator_biasprm'}
    if c.wrist_friction_nm is not None:
        f.add('dof_frictionloss')
    return f


class GraspFidelity:
    """Per-world sampling on the Warp model; `drawn` holds this episode's values for logging."""
    KEYS = ('tip_extend_m', 'tip_friction', 'wrist_kp_scale', 'wrist_friction_nm', 'reader_fillet_m', 'pad_timeconst',
            'servo_roll_deg', 'servo_tilt_deg')

    def __init__(self, rt, c, geoms):
        self.rt, self.c, self.dev = rt, c, rt.device
        m, wm, dev = rt.mjm, rt.m, rt.device
        self.nworld = rt.cfg.nworld
        self.drawn = {k: torch.full((self.nworld,), float('nan'), device=dev) for k in self.KEYS}
        self.tips = torch.tensor([m.geom('right_tip_hull').id, m.geom('left_tip_hull').id], device=dev)
        self.prims = torch.tensor(geoms['prims'], device=dev, dtype=torch.long)
        self.box = rt.lay.variants[0].peg_boxes[0]
        self.pads = torch.tensor([g for g in range(m.ngeom) if m.geom(g).name.startswith(('left_pad_s', 'right_pad_s'))], device=dev)
        arr = lambda n: wp.to_torch(getattr(wm, n)) if n in batch_fields(c) else None   # noqa: E731
        self.geom_pos, self.geom_size, self.geom_aabb, self.geom_rbound = (arr(n) for n in ('geom_pos', 'geom_size', 'geom_aabb', 'geom_rbound'))
        self.geom_friction, self.geom_solref = arr('geom_friction'), arr('geom_solref')
        self.gain, self.bias = arr('actuator_gainprm'), arr('actuator_biasprm')
        self.frictionloss = arr('dof_frictionloss')
        if self.geom_pos is not None:
            self.tip_pos0 = self.geom_pos[0, self.tips].clone()
            # the approach (site +z) in each finger body's frame, for the hull shift
            d = mujoco.MjData(m)
            mujoco.mj_kinematics(m, d)
            s = m.site('grasp_site').id
            a = d.site_xmat[s].reshape(3, 3)[:, 2]
            dirs = []
            for f in ('right_finger', 'left_finger'):
                dirs.append(d.xmat[m.body(f).id].reshape(3, 3).T @ a)
            self.tip_dir = torch.tensor(np.array(dirs), device=dev, dtype=self.geom_pos.dtype)
        if self.gain is not None:
            self.acts = torch.tensor(rt.indices['right'][2][3:], device=dev)
            self.gain0, self.bias0 = self.gain[0, self.acts].clone(), self.bias[0, self.acts].clone()
        self.wrist_dofs = torch.tensor(rt.indices['right'][1][3:], device=dev)
        self.sample(torch.arange(self.nworld, device=dev))

    def apply_servo_err(self, R):
        """R (N, 3, 3) grasp_site orientations with this episode's error: tilt about the closing axis y (more tilt: the
        reader_grasp convention), then roll about the horizontal axis normal to y so that asin(R[2, 1]) changes by the
        drawn roll (the analysis' closing_axis_from_horizontal_deg)."""
        from blind_insertion.envs import kinematics as K
        tilt = torch.deg2rad(self.drawn['servo_tilt_deg']).to(R.dtype)
        y = torch.tensor([0., 1., 0.], device=R.device, dtype=R.dtype).expand(len(R), 3)
        R = R @ K.axis_angle_mat(y, -tilt)
        Y = R[:, :, 1]
        ax = torch.linalg.cross(Y, torch.tensor([0., 0., 1.], device=R.device, dtype=R.dtype).expand(len(R), 3))
        ax = ax / ax.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return K.axis_angle_mat(ax, torch.deg2rad(self.drawn['servo_roll_deg']).to(R.dtype)) @ R

    def _u(self, r, n):
        return torch.rand(n, device=self.dev) * (r[1] - r[0]) + r[0]

    def sample(self, ids):
        c, n = self.c, len(ids)
        if c.servo_err is not None:
            rows = torch.as_tensor(c.servo_err, device=self.dev, dtype=torch.float32)
            d = rows[torch.randint(len(rows), (n,), device=self.dev)]
            j = torch.randn(n, 2, device=self.dev) * c.servo_err_jitter_deg
            self.drawn['servo_roll_deg'][ids] = d[:, 0] + j[:, 0]
            self.drawn['servo_tilt_deg'][ids] = d[:, 1] + j[:, 1]
        if c.tip_extend_m is not None:
            x = self._u(c.tip_extend_m, n)
            self.drawn['tip_extend_m'][ids] = x
            # the two fingers of the right hand ('left_tip_hull', 'right_tip_hull'); the left arm's stay nominal
            self.geom_pos[ids[:, None], self.tips] = self.tip_pos0 + x[:, None, None] * self.tip_dir
        if c.tip_friction is not None:
            x = self._u(c.tip_friction, n)
            self.drawn['tip_friction'][ids] = x
            self.geom_friction[ids[:, None], self.tips, 0] = x[:, None]
        if c.wrist_kp_scale is not None:
            x = self._u(c.wrist_kp_scale, n)
            self.drawn['wrist_kp_scale'][ids] = x
            self.gain[ids[:, None], self.acts, 0] = x[:, None] * self.gain0[:, 0]
            self.bias[ids[:, None], self.acts, 1] = x[:, None] * self.bias0[:, 1]
            self.bias[ids[:, None], self.acts, 2] = x[:, None] * self.bias0[:, 2]
        if c.wrist_friction_nm is not None:
            x = self._u(c.wrist_friction_nm, n)
            self.drawn['wrist_friction_nm'][ids] = x
            self.frictionloss[ids[:, None], self.wrist_dofs] = x[:, None]
        if c.reader_fillet_m is not None:
            x = self._u(c.reader_fillet_m, n)
            self.drawn['reader_fillet_m'][ids] = x
            pos, size, half, rb = _prim_values(self.box, x)
            P = self.prims
            self.geom_pos[ids[:, None], P] = pos.to(self.geom_pos.dtype)
            self.geom_size[ids[:, None], P] = size.to(self.geom_size.dtype)
            self.geom_aabb[ids[:, None], P, 0] = 0.
            self.geom_aabb[ids[:, None], P, 1] = half.to(self.geom_aabb.dtype)
            self.geom_rbound[ids[:, None], P] = rb.to(self.geom_rbound.dtype)
        if c.pad_timeconst is not None:
            x = self._u(c.pad_timeconst, n)
            self.drawn['pad_timeconst'][ids] = x
            self.geom_solref[ids[:, None], self.pads, 0] = x[:, None].expand(-1, len(self.pads)).to(self.geom_solref.dtype)
