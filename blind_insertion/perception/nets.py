"""Small CNN pose regressors (RGB only) with CoordConv input and a spatial (non-pooled) head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, stride):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class PoseNet(nn.Module):
    """uint8 (B,H,W,3) -> (B, n_out) normalized targets."""

    def __init__(self, n_out, size, width=32):
        super().__init__()
        self.size = size
        w = width
        self.body = nn.Sequential(_block(5, w, 2), _block(w, 2 * w, 2), _block(2 * w, 4 * w, 2),
                                  _block(4 * w, 8 * w, 2), _block(8 * w, 8 * w, 2))
        s = size // 32
        self.head = nn.Sequential(nn.Conv2d(8 * w, 64, 1), nn.ReLU(inplace=True), nn.Flatten(),
                                  nn.Linear(64 * s * s, 256), nn.ReLU(inplace=True), nn.Linear(256, n_out))

    def prep(self, img):
        x = img.permute(0, 3, 1, 2).float() / 255.
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='area')
        x = (x - .45) / .25
        b = x.shape[0]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, self.size, device=x.device),
                                torch.linspace(-1, 1, self.size, device=x.device), indexing='ij')
        return torch.cat([x, xx.expand(b, 1, -1, -1), yy.expand(b, 1, -1, -1)], 1)

    def forward(self, img):
        return self.head(self.body(self.prep(img)))


class TwoViewPoseNet(nn.Module):
    """Two RGB views (e.g. right + left wrist), separate CNN bodies, features fused in the head."""

    def __init__(self, n_out, size, width=32):
        super().__init__()
        self.a, self.b = PoseNet(n_out, size, width), PoseNet(n_out, size, width)
        s = size // 32
        w = width
        self.head = nn.Sequential(nn.Linear(2 * 64 * s * s, 256), nn.ReLU(inplace=True), nn.Linear(256, n_out))
        self.size = size

    def feat(self, net, img):
        return net.head[:3](net.body(net.prep(img)))

    def forward(self, img):
        """img: (B, 2, H, W, 3) uint8."""
        return self.head(torch.cat([self.feat(self.a, img[:, 0]), self.feat(self.b, img[:, 1])], 1))
