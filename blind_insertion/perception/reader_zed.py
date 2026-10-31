"""ZED reader-pose estimator for the Part 1 pick (RGB only; scripts/perception/train_reader_zed.py).

`ReaderZedEstimator(ckpt)(top_frames)` -> body-centre xy (m, right-base frame) and plug yaw (rad) from
the ZED frame taken at the rest pose with the gripper open. `reader_pose(task, xy, yaw)` turns that
into the reader frame pose (plug-tip origin) the pick expert plans with (reader flat on the table).
"""
import math

import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.perception.nets import PoseNet


class ReaderZedEstimator:
    def __init__(self, ckpt='runs/perception/reader_zed.pt', dev='cuda:0'):
        ck = torch.load(ckpt, map_location=dev)
        self.net = PoseNet(ck['n_out'], ck['size'], width=ck['width']).to(dev).eval()
        self.net.load_state_dict(ck['net'])
        self.scale, self.crop, self.res = ck['scale'].to(dev), ck['crop'], ck['res']
        self.centre, self.yaw0 = ck['strip_centre'], ck['yaw0']

    @torch.no_grad()
    def __call__(self, top):
        x0, y0, s = self.crop
        if top.shape[1] != self.res:
            raise ValueError(f'ZED frames must be {self.res} px (got {top.shape[1]})')
        out = torch.cat([self.net(top[i:i + 256, y0:y0 + s, x0:x0 + s].contiguous()) for i in range(0, len(top), 256)]) * self.scale
        xy = out[:, :2] + torch.tensor(self.centre, device=out.device)
        return xy, out[:, 2] + self.yaw0


def reader_pose(task, xy, yaw):
    """Reader pose (plug-tip origin) for a flat reader with body centre xy and plug yaw."""
    rt = task.runtime
    R0 = K.quat_to_mat(rt._reader0[3:][None])[0]          # place_reader at yaw 0: plug along +x
    n = len(xy)
    c, s, z, o = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    Rz = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
    R = Rz @ R0
    bz = float(task.lay.variants[0].peg_boxes[0][0][2])
    p = torch.cat([xy, torch.full((n, 1), float(rt._reader0[2]), device=xy.device)], 1) - R[:, :, 2] * bz
    p[:, 2] = float(rt._reader0[2])
    return p, R
