"""Robot-arm masks for the ZED frame from the joint angles (no image processing): render the scene's
segmentation from the calibrated camera and keep the pixels that belong to either arm (bases, links,
grippers, fingers, wrist cameras). The ZED pose net sees the masked pixels as flat grey, so it cannot use
(or be confused by) arm appearance, which differs between the sim renders and the real arms.

In sim the mask comes from the same (DR-perturbed) camera as the image; training shifts it by a few
pixels to mimic the real case, where the mask is rendered from the encoders and the nominal calibration.
"""
import torch
import torch.nn.functional as F
import warp as wp
import mujoco_warp as mjw

from blind_insertion.envs import visual_dr as VD

FILL = 128


def arm_geoms(m):
    """Bool (ngeom,): geoms on the bodies of both arm trees (rooted at 'arm' and 'left_arm')."""
    roots = {m.body('arm').id, m.body('left_arm').id}
    inside = []
    for b in range(m.nbody):
        x = b
        while x != 0 and x not in roots:
            x = int(m.body_parentid[x])
        inside.append(x in roots)
    return torch.tensor([inside[int(m.geom_bodyid[g])] for g in range(m.ngeom)])


class ArmMasker:
    def __init__(self, rt, camera='top', dilate_px=3):
        m, c = rt.mjm, rt.cfg
        self.rt, self.dilate = rt, dilate_px
        cid = m.camera(camera).id
        active = [i == cid for i in range(m.ncam)]
        res = c.camera_resolution
        with wp.ScopedDevice(c.device), VD.render_groups(m):
            self.rc = mjw.create_render_context(m, nworld=c.nworld, cam_res=[(res, res)], render_rgb=False, render_seg=True,
                                                cam_active=active, **VD.render_context_kwargs(getattr(rt, 'visual_dr_cfg', None)))
            self.seg = wp.zeros((c.nworld, res, res), dtype=wp.vec2i)
        self.arm = arm_geoms(m).to(rt.device)

    def __call__(self, shift_px=0, generator=None):
        """(nworld, res, res) bool arm mask at the current state; shift_px: random per-world shift (+/-)."""
        rt = self.rt
        with wp.ScopedDevice(rt.cfg.device):
            mjw.kinematics(rt.m, rt.d)
            mjw.camlight(rt.m, rt.d)
            mjw.refit_bvh(rt.m, rt.d, self.rc)
            mjw.render(rt.m, rt.d, self.rc)
            mjw.get_segmentation(self.rc, 0, self.seg)
        s = wp.to_torch(self.seg)
        g = s[..., 0].long()
        mask = (g >= 0) & self.arm[g.clamp(min=0)]
        mask = mask.float()[:, None]
        if self.dilate:
            mask = F.max_pool2d(mask, 2 * self.dilate + 1, 1, self.dilate)
        mask = mask[:, 0] > .5
        if shift_px:
            n = len(mask)
            dx = torch.randint(-shift_px, shift_px + 1, (n,), device=mask.device, generator=generator)
            dy = torch.randint(-shift_px, shift_px + 1, (n,), device=mask.device, generator=generator)
            mask = torch.stack([torch.roll(mk, (int(a), int(b)), (0, 1)) for mk, a, b in zip(mask, dy, dx)])
        return mask


def apply_mask(frames, mask):
    """frames (B,H,W,3) uint8, mask (B,H,W) bool -> masked copy (arm pixels = FILL grey)."""
    out = frames.clone()
    out[mask] = FILL
    return out
