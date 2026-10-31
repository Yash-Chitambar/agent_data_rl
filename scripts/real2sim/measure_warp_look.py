"""Real vs MuJoCo Warp (training renderer) look, on the recorded duchamp3 frames.

The CPU-renderer tools (`render_sim_real_alignment.py`, `fit_backdrop.py`) score the OpenGL
render; policies see the Warp render of `BimanualRuntime`. This script renders the same 21
views as `fit_backdrop.py` (stationary_001 + three free_settle trials: ZED top + both wrists, logged
joint angles and jaw openings, PC and reader moved out of view as in that fit) through the
runtime, one world per pose, and scores each camera against the real frame:

  tip_iou       fingertip IoU (yellow mask on both images; wrist views, rows >= 150 of 320)
  edge_px       symmetric mean truncated (10 px) distance between sim and real Canny edges
                (320 px images, fingertips masked)
  regions       median real vs median sim colour per appearance group (visual_dr.GROUPS, from a Warp
                segmentation pass); abs_delta = mean |real - sim| over RGB, 0-255
  l1            mean absolute RGB difference over the image (0-255)

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/real2sim/measure_warp_look.py [--cfg '{...}'] [--label L]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch

RES, TAU = 320, 10.0
OUT = Path('results/usbc/policy/visual_dr/nominal')


def load_views():
    from blind_insertion.scenes import physical_rig as R
    from fit_backdrop import views
    from render_sim_real_alignment import FIT
    fit = json.loads(FIT.read_text())
    return views(R.load(), fit), fit


def edge_distance(real, sim, mask):
    def edges(img):
        return cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (3, 3), 0), 25, 75) > 0
    er, es = edges(real) & ~mask, edges(sim) & ~mask
    if er.sum() < 5 or es.sum() < 5:
        return float('nan')
    dr = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    ds = cv2.distanceTransform((~es).astype(np.uint8), cv2.DIST_L2, 3)
    return float((np.minimum(dr[es], TAU).mean() + np.minimum(ds[er], TAU).mean()) / 2)


class WarpLook:
    """Runtime posed at the recorded views (world k = joint configuration k) + segmentation."""

    def __init__(self, cfg_kw=None):
        from blind_insertion.envs import visual_dr as VD
        from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
        self.views, self.fit = load_views()
        self.keys, poses = [], []
        for v in self.views:                    # one world per distinct joint configuration
            key = self.key(v[0])
            if key not in self.keys:
                self.keys.append(key); poses.append(v[4])
        kw = dict(nworld=len(poses), render_rgb=True, camera_resolution=RES, measured_camera_cadence=False)
        kw.update(cfg_kw or {})
        self.e = e = BimanualRuntime(BimanualCfg(**kw))
        e.reset()
        self.pose(poses)
        self.groups = {g: v for g, v in VD.group_geoms(e.mjm).items() if len(v)}
        self.names = list(self.groups)
        lut = np.full(e.mjm.ngeom + 1, -1)
        for i, g in enumerate(self.names):
            lut[self.groups[g]] = i
        self.seg = {k: lut[v] for k, v in self.segmentation().items()}

    @staticmethod
    def key(label):
        trial, _, frame = label.split('/')
        return trial + '/' + frame

    def pose(self, poses):
        from blind_insertion.scenes import physical_rig as R
        e, spec, m = self.e, R.load(), self.e.mjm
        for k, q in enumerate(poses):
            for arm, prefix in (('right', ''), ('left', 'left_')):
                for j in range(6):
                    e.q[k, int(m.joint(prefix + f'joint{j + 1}').qposadr[0])] = float(q[arm][j] + spec['arms'][arm]['joint_offsets_rad'][j])
                gm = self.fit['arms'][arm]['gripper_map']
                for f in ('left', 'right'):
                    e.q[k, int(m.joint(prefix + f + '_finger').qposadr[0])] = float(np.clip(gm['slope'] * q[arm][6] + gm['intercept'], 0, .0475))
            e.q[k, e.pc_qadr:e.pc_qadr + 3] = torch.tensor([5., 5., .05])
            e.q[k, e.lay.peg_qadr:e.lay.peg_qadr + 3] = torch.tensor([0., 0., -10.])

    def segmentation(self):
        """Per camera (nworld, RES, RES) geom ids (-1 = none) from a Warp segmentation pass."""
        import mujoco_warp as mjw
        import warp as wp
        from blind_insertion.envs import visual_dr as VD
        e, m = self.e, self.e.mjm
        ids = [m.camera(n).id for n in e.camera_names]
        active = [i in ids for i in range(m.ncam)]
        order = [i for i in range(m.ncam) if active[i]]
        with wp.ScopedDevice(e.cfg.device), VD.render_groups(m):
            rc = mjw.create_render_context(m, nworld=e.cfg.nworld, cam_res=[(RES, RES)] * len(ids), render_rgb=False,
                                           render_depth=False, render_seg=True, cam_active=active)
        with wp.ScopedDevice(e.cfg.device):
            mjw.kinematics(e.m, e.d); mjw.camlight(e.m, e.d); mjw.refit_bvh(e.m, e.d, rc); mjw.render(e.m, e.d, rc)
            out, buf = {}, wp.zeros((e.cfg.nworld, RES, RES), dtype=wp.vec2i)
            for name, i in zip(e.camera_names, ids):
                mjw.get_segmentation(rc, order.index(i), buf)
                s = wp.to_torch(buf).cpu().numpy()
                out[name] = np.where(s[..., 1] == 5, s[..., 0], -1)       # mjOBJ_GEOM = 5
        return out

    def render(self):
        return {k: v.cpu().numpy() for k, v in self.e.render().items()}

    def score(self, frames, label='warp'):
        from fit_gripper_tips import yellow_mask
        rows, tiles = [], []
        for name, camera, side, real, q in self.views:
            k = self.keys.index(self.key(name))
            sim, reg = frames[camera][k], self.seg[camera][k]
            tips_r, tips_s = yellow_mask(real), yellow_mask(sim)
            tipmask = cv2.dilate((tips_r | tips_s).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            row = {'view': name, 'camera': camera, 'edge_px': edge_distance(real, sim, tipmask),
                   'l1': float(np.abs(real.astype(float) - sim.astype(float)).mean())}
            if side is not None:
                row['tip_iou'] = float((tips_r & tips_s).sum() / max(1, (tips_r | tips_s).sum()))
            for i, g in enumerate(self.names):
                mk = cv2.erode((reg == i).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~tipmask
                if mk.sum() > 40:
                    row[g] = {'real': np.median(real[mk], 0).tolist(), 'sim': np.median(sim[mk], 0).tolist(), 'n': int(mk.sum())}
            rows.append(row)
            tiles.append(np.hstack([real, sim]))
        s = {'label': label, 'n_views': len(rows)}
        s['tip_iou'] = float(np.mean([r['tip_iou'] for r in rows if 'tip_iou' in r]))
        s['edge_px'] = float(np.nanmean([r['edge_px'] for r in rows]))
        s['l1'] = float(np.mean([r['l1'] for r in rows]))
        for cam in self.e.camera_names:
            cr = [r for r in rows if r['camera'] == cam]
            s[f'edge_px/{cam}'] = float(np.nanmean([r['edge_px'] for r in cr]))
            s[f'l1/{cam}'] = float(np.mean([r['l1'] for r in cr]))
            if any('tip_iou' in r for r in cr):
                s[f'tip_iou/{cam}'] = float(np.mean([r['tip_iou'] for r in cr]))
        s['regions'] = {}
        for g in self.names:
            for cam in self.e.camera_names:
                rr = [r[g] for r in rows if g in r and r['camera'] == cam]
                if rr:
                    real = np.median([x['real'] for x in rr], 0); sim = np.median([x['sim'] for x in rr], 0)
                    s['regions'][f'{g}/{cam}'] = {'real_rgb': real.round(1).tolist(), 'sim_rgb': sim.round(1).tolist(),
                                                 'abs_delta': float(np.abs(real - sim).mean().round(1)), 'views': len(rr)}
        d = [v['abs_delta'] for v in s['regions'].values()]
        s['region_abs_delta_mean'] = float(np.mean(d))
        return s, rows, tiles

    def save(self, out, label, s, rows, tiles):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles) - len(tiles) % 3, 3)])
        sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out / f'{label}_real_vs_sim.jpg'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        (out / f'{label}.json').write_text(json.dumps({'summary': s, 'views': rows}, indent=1) + '\n')


def measure(cfg_kw=None, out=OUT, label='warp'):
    w = WarpLook(cfg_kw)
    s, rows, tiles = w.score(w.render(), label)
    s['cfg'] = cfg_kw or {}
    if out is not None:
        w.save(out, label, s, rows, tiles)
    del w
    torch.cuda.empty_cache()
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cfg', default='{}', help='BimanualCfg overrides (JSON)')
    p.add_argument('--label', default='warp')
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--legacy', action='store_true', help='render as before visual_dr: alpha-0 geoms and lenses '
                   'rendered (combine with --cfg \'{"visual_look": null}\')')
    a = p.parse_args()
    if a.legacy:
        from blind_insertion.envs import visual_dr as VD
        VD.render_hidden = lambda m: []
    s = measure(json.loads(a.cfg), a.out, a.label)
    print(json.dumps({k: v for k, v in s.items() if k != 'regions'}, indent=1))
    for k, v in s['regions'].items():
        print(f"{k:32s} real {v['real_rgb']} sim {v['sim_rgb']} |d| {v['abs_delta']}")


if __name__ == '__main__':
    main()
