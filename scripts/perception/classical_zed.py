"""Classical (fiducial-free) ZED PC-pose estimator: bright-case silhouette in the PC crop (Otsu
threshold of the grey image, largest blob), minimum-area rectangle -> blob centre ray intersected with
the case mid-height plane through the nominal (calibrated) camera; yaw from the rectangle's long-edge
angle in the table plane. A constant offset (centre -> port) is fitted on 256 nominal-look sim
renders. Evaluated on held-out DR renders (policy/v1 visual DR) and on the real stationary frames.

    PYTHONPATH=. .venv/bin/python scripts/perception/classical_zed.py
"""
import glob
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warp as wp
wp.config.quiet = True
import cv2
import mujoco
import numpy as np
import torch

from blind_insertion.perception.render_data import ApproachRenderer
from blind_insertion.scenes.physical_rig import real_top_policy_image
from train_approach_estimators import ZED_CROP

REAL = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/cameras/bimanual_stationary_001/zed')
REF = json.loads(Path('results/usbc/bimanual_20261029/overlays/manifest.json').read_text())['scene_annotation']


class Classical:
    def __init__(self, m, res=320):
        d = mujoco.MjData(m); mujoco.mj_forward(m, d)
        c = m.camera('top').id
        self.p, self.R = d.cam_xpos[c].copy(), d.cam_xmat[c].reshape(3, 3).copy()
        self.f = .5 * res / math.tan(math.radians(m.cam_fovy[c]) / 2)
        self.c = res / 2
        self.z = float(m.geom('table').pos[2] + m.geom('table').size[2]) + .0445 * .75
        self.offset = np.zeros(2)

    def ray_plane(self, u, v):
        d = self.R @ np.array([(u - self.c) / self.f, -(v - self.c) / self.f, -1.])
        t = (self.z - self.p[2]) / d[2]
        return self.p[:2] + t * d[:2]

    def __call__(self, img):
        """img: (320,320,3) uint8 RGB full frame -> port xy (m), yaw (rad) or None."""
        x0, y0, s = ZED_CROP
        g = cv2.cvtColor(np.ascontiguousarray(img[y0:y0 + s, x0:x0 + s]), cv2.COLOR_RGB2GRAY)
        _, bw = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(bw)
        if n < 2:
            return None, None
        # the PC: the largest blob that does not touch the crop border
        best, area = None, 0
        for i in range(1, n):
            x, y, w, h, a = stats[i]
            if x > 0 and y > 0 and x + w < s and y + h < s and a > area:
                best, area = i, a
        if best is None or area < 100:
            return None, None
        pts = np.column_stack(np.nonzero(lab == best))[:, ::-1].astype(np.float32)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
        centre = self.ray_plane(cx + x0, cy + y0)
        # yaw: project a rectangle edge direction to the plane
        a = math.radians(ang)
        e = self.ray_plane(cx + x0 + 10 * math.cos(a), cy + y0 + 10 * math.sin(a)) - centre
        yaw = math.atan2(e[1], e[0])
        yaw = (yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4      # square-ish case: mod 90 deg
        return centre + self.offset, yaw


def main():
    r = ApproachRenderer(nworld=256, seed=7)
    est = Classical(r.rt.mjm)
    # fit centre -> port offset on nominal-look renders
    img, lab = r.sample_top(nominal=True)
    img = img.cpu().numpy()
    res = [est(im) for im in img]
    ok = np.array([p is not None for p, _ in res])
    pred = np.array([p for p, _ in res if p is not None])
    est.offset = (lab['port_xy'].cpu().numpy()[ok] - pred).mean(0)
    errs, yerr, fail = [], [], 0
    r.gen.manual_seed(999)
    for b in range(8):
        img, lab = r.sample_top()
        for im, t, ty in zip(img.cpu().numpy(), lab['port_xy'].cpu().numpy(), lab['port_yaw'].cpu().numpy()):
            p, y = est(im)
            if p is None:
                fail += 1; continue
            errs.append(np.linalg.norm(p - t) * 1e3)
            yerr.append(abs(math.degrees(math.atan2(math.sin(y - ty), math.cos(y - ty)))))
    errs, yerr = np.array(errs), np.array(yerr)
    pc = lambda x: {f'p{q}': float(np.percentile(x, q)) for q in (50, 90, 99, 99.5)}
    real = []
    for f in sorted(glob.glob(str(REAL / 'raw_*.png'))):
        p, y = est(real_top_policy_image(cv2.imread(f), 320))
        if p is not None:
            real.append([*p, math.degrees(y)])
    real = np.array(real)
    d = (real[:, :2].mean(0) - np.array(REF['derived_port_xy_m'])) * 1e3
    rep = {'method': __doc__.strip(), 'offset_fit_m': est.offset.tolist(),
           'dr_heldout': {'n': len(errs) + fail, 'failed': fail, 'xy_mm': pc(errs), 'yaw_deg': pc(yerr)},
           'real_stationary': {'n_frames': len(real), 'port_xy_m_mean': real[:, :2].mean(0).tolist(),
                               'xy_std_mm': (real[:, :2].std(0) * 1e3).tolist(), 'yaw_deg_mean': float(real[:, 2].mean()),
                               'vs_manual_reference_mm': float(np.linalg.norm(d)), 'dxdy_mm': d.tolist()}}
    Path('results/usbc/policy/approach/estimator_zed_classical.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k != 'method'}, indent=1))


if __name__ == '__main__':
    main()
