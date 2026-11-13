"""ZED PC-pose estimator with a presence and validity gate (RGB only).

`ZedEstimator(frame)` returns (port_xy m, yaw rad, ok) per image: ok is False when
  * the presence classifier says there is no PC in the crop (p < presence_thresh), or
  * the frame is unlike the training data: Mahalanobis distance of the pose CNN's pooled features
    to the DR-render feature distribution above OOD_SCALE x the 99.9th percentile of held-out DR renders.
    OOD_SCALE = 1.25 covers the sim-to-real shift of the distance (real frames with a bare PC: 14.4-18.0;
    sim p99.9: 17.1; real frames without a PC: 21.9-24.2), results/usbc/policy/approach/zed_presence.json.
Weights: runs/perception/zed.pt (pose), runs/perception/zed_presence.pt (presence + feature stats),
trained by scripts/perception/train_zed_presence.py.

Estimator key (zed_v5, docs/real/ZED_V5_20261114.md): a gate checkpoint may name its own pose nets ('pose_nets':
[name, name2], files next to the gate). Then ZedEstimator loads those nets instead of the `pose` / `pose2` arguments:
a gate's Mahalanobis statistics are only valid for the net they were fit on. The deploy configs select the estimator
by chain_args.zed_gate alone (e.g. 'zed_v5_gate'). A pose checkpoint may carry 'nominal_port_xy' (m) and 'zed_crop'
(x, y, size px); without them NOMINAL_PORT_XY and ZED_CROP apply. Old gates and nets have neither key, so they
behave as before.
"""
from pathlib import Path

import torch

from blind_insertion.perception.nets import PoseNet

NOMINAL_PORT_XY = (.22, .36)
OOD_SCALE = 1.25
ZED_CROP = (110, 70, 160)


def zed_crop(img, crop=None):
    x, y, s = ZED_CROP if crop is None else crop
    return img[:, y:y + s, x:x + s]


def pooled_features(net, x):
    """Global-average-pooled last conv features (B, C) of a PoseNet on uint8 crops."""
    return net.body(net.prep(x)).mean((2, 3))


class ZedEstimator:
    def __init__(self, dev='cuda:0', pose='runs/perception/zed.pt', presence='runs/perception/zed_presence.pt', pose2=None):
        cp = torch.load(presence, map_location=dev)
        if cp.get('pose_nets'):    # estimator key: the gate names its pose nets (files next to the gate)
            names = list(cp['pose_nets'])
            pose = str(Path(presence).parent / f'{names[0]}.pt')
            pose2 = str(Path(presence).parent / f'{names[1]}.pt') if len(names) > 1 and names[1] else None
        self.pose_files = (str(pose), None if pose2 is None else str(pose2))
        self.pose2 = None
        ck = torch.load(pose, map_location=dev)
        self.nominal = tuple(float(v) for v in ck.get('nominal_port_xy', NOMINAL_PORT_XY))
        self.crop = tuple(int(v) for v in ck.get('zed_crop', ZED_CROP))
        if pose2 is not None:      # two-net ensemble: mean estimate; the gate also rejects large disagreement
            c2 = torch.load(pose2, map_location=dev)
            if tuple(float(v) for v in c2.get('nominal_port_xy', NOMINAL_PORT_XY)) != self.nominal or \
                    tuple(int(v) for v in c2.get('zed_crop', ZED_CROP)) != self.crop:
                raise ValueError(f'ZED pose nets {pose} and {pose2} differ in nominal port or crop')
            self.pose2 = PoseNet(c2.get('n_out', len(c2['scale'])), c2['size'], width=c2.get('width', 32)).to(dev).eval()
            self.pose2.load_state_dict(c2['net'])
        for k_, v_ in (('nominal_port_xy', self.nominal), ('zed_crop', self.crop)):
            if k_ in cp and tuple(cp[k_]) != tuple(v_):
                raise ValueError(f'ZED gate {presence}: {k_} {tuple(cp[k_])} differs from the pose net ({tuple(v_)})')
        self.hetero = bool(ck.get('hetero', False))
        self.pose = PoseNet(ck.get('n_out', len(ck['scale'])), ck['size'], width=ck.get('width', 32)).to(dev).eval()
        self.pose.load_state_dict(ck['net'])
        self.scale = ck['scale'].to(dev)
        self.presence = PoseNet(1, cp['size'], width=cp.get('width', 32)).to(dev).eval()
        self.presence.load_state_dict(cp['net'])
        self.mu, self.prec = cp['feat_mean'].to(dev), cp['feat_prec'].to(dev)
        self.ood_thresh, self.presence_thresh = float(cp['ood_thresh']), float(cp.get('presence_thresh', .5))
        self.ood_scale = float(cp.get('ood_scale', OOD_SCALE))
        # predicted-uncertainty gate (hetero pose nets): reject when the predicted std is large
        self.sigma_xy_thresh = float(cp.get('sigma_xy_thresh_mm', float('inf')))
        self.sigma_yaw_thresh = float(cp.get('sigma_yaw_thresh_deg', float('inf')))
        self.dis_xy_thresh = float(cp.get('disagree_xy_thresh_mm', float('inf')))
        self.dis_yaw_thresh = float(cp.get('disagree_yaw_thresh_deg', float('inf')))
        # Arm-pose gate (encoders): the pose CNN is only valid for the arm configurations it was trained on.
        # envelope = {'left_refs': [[6 rad], ...], 'right_refs': [[6 rad], ...], 'tol_rad': t,
        #             'right_finger_min_m': f}. A frame is accepted only when each arm is within tol_rad
        # (max over joints) of one of its reference poses and the right fingers are open >= f.
        self.envelope = cp.get('arm_envelope')

    def arm_ok(self, q_left, q_right, right_finger_m):
        """(B,6), (B,6) believed joint angles (rad, model frame), (B,) right finger opening per finger (m)."""
        e = self.envelope
        if e is None:
            return torch.ones(len(q_left), dtype=torch.bool, device=q_left.device)
        def near(q, refs):
            refs = torch.as_tensor(refs, device=q.device, dtype=q.dtype)
            return ((q[:, None] - refs[None]).abs().max(-1).values <= e['tol_rad']).any(1)
        return near(q_left, e['left_refs']) & near(q_right, e['right_refs']) & (right_finger_m >= e['right_finger_min_m'])

    @torch.no_grad()
    def __call__(self, frames, arms=None):
        """frames: (B, 320, 320, 3) uint8 policy-crop ZED frames; arms: optional (q_left, q_right,
        right_finger_m) from the encoders for the arm-pose gate."""
        x = zed_crop(frames, self.crop)
        out = self.pose(x)
        k = len(self.scale)
        est = out[:, :k] * self.scale
        sig = torch.exp(out[:, k:].clamp(-7, 3)) * self.scale if self.hetero else torch.zeros_like(est)
        dis = torch.zeros_like(est)
        if self.pose2 is not None:
            est2 = self.pose2(x)[:, :k] * self.scale
            dis = (est - est2).abs()
            est = (est + est2) / 2
        p = torch.sigmoid(self.presence(x)[:, 0])
        f = pooled_features(self.pose, x) - self.mu
        d = ((f @ self.prec) * f).sum(1).clamp(min=0).sqrt()
        port = torch.tensor(self.nominal, device=x.device) + est[:, :2] * 1e-3
        sxy = sig[:, :2].norm(dim=1)
        ok = (p >= self.presence_thresh) & (d <= self.ood_thresh * self.ood_scale) & \
             (sxy <= self.sigma_xy_thresh) & (sig[:, 2] <= self.sigma_yaw_thresh) & \
             (dis[:, :2].norm(dim=1) <= self.dis_xy_thresh) & (dis[:, 2] <= self.dis_yaw_thresh)
        arm = self.arm_ok(*arms) if arms is not None else torch.ones_like(ok)
        ok = ok & arm
        return port, torch.deg2rad(est[:, 2]), ok, {'arm_ok': arm, 'presence': p, 'mahalanobis': d, 'sigma_xy_mm': sxy, 'sigma_yaw_deg': sig[:, 2],
                                                    'disagree_xy_mm': dis[:, :2].norm(dim=1), 'disagree_yaw_deg': dis[:, 2]}
