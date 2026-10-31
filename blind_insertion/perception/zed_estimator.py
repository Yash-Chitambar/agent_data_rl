"""ZED PC-pose estimator with a presence and validity gate (RGB only).

`ZedEstimator(frame)` returns (port_xy m, yaw rad, ok) per image: ok is False when
  * the presence classifier says there is no PC in the crop (p < presence_thresh), or
  * the frame is unlike the training data: Mahalanobis distance of the pose CNN's pooled features
    to the DR-render feature distribution above OOD_SCALE x the 99.9th percentile of held-out DR renders.
    OOD_SCALE = 1.25 covers the sim-to-real shift of the distance (real frames with a bare PC: 14.4-18.0;
    sim p99.9: 17.1; real frames without a PC: 21.9-24.2), results/usbc/policy/approach/zed_presence.json.
Weights: runs/perception/zed.pt (pose), runs/perception/zed_presence.pt (presence + feature stats),
trained by scripts/perception/train_zed_presence.py.
"""
import torch

from blind_insertion.perception.nets import PoseNet

NOMINAL_PORT_XY = (.22, .36)
OOD_SCALE = 1.25
ZED_CROP = (110, 70, 160)


def zed_crop(img):
    x, y, s = ZED_CROP
    return img[:, y:y + s, x:x + s]


def pooled_features(net, x):
    """Global-average-pooled last conv features (B, C) of a PoseNet on uint8 crops."""
    return net.body(net.prep(x)).mean((2, 3))


class ZedEstimator:
    def __init__(self, dev='cuda:0', pose='runs/perception/zed.pt', presence='runs/perception/zed_presence.pt'):
        ck = torch.load(pose, map_location=dev)
        self.pose = PoseNet(len(ck['scale']), ck['size']).to(dev).eval()
        self.pose.load_state_dict(ck['net'])
        self.scale = ck['scale'].to(dev)
        cp = torch.load(presence, map_location=dev)
        self.presence = PoseNet(1, cp['size'], width=cp.get('width', 32)).to(dev).eval()
        self.presence.load_state_dict(cp['net'])
        self.mu, self.prec = cp['feat_mean'].to(dev), cp['feat_prec'].to(dev)
        self.ood_thresh, self.presence_thresh = float(cp['ood_thresh']), float(cp.get('presence_thresh', .5))

    @torch.no_grad()
    def __call__(self, frames):
        """frames: (B, 320, 320, 3) uint8 policy-crop ZED frames."""
        x = zed_crop(frames)
        est = self.pose(x) * self.scale
        p = torch.sigmoid(self.presence(x)[:, 0])
        f = pooled_features(self.pose, x) - self.mu
        d = ((f @ self.prec) * f).sum(1).clamp(min=0).sqrt()
        port = torch.tensor(NOMINAL_PORT_XY, device=x.device) + est[:, :2] * 1e-3
        ok = (p >= self.presence_thresh) & (d <= self.ood_thresh * OOD_SCALE)
        return port, torch.deg2rad(est[:, 2]), ok, {'presence': p, 'mahalanobis': d}
