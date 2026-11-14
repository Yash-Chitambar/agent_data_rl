"""ZED PC pose net zed_v6 (docs/real/PERCEPTION_BIAS_20261115.md, Part A): the zed_v5 recipe (train_zed_v5.py: layout
task_layout_real_v2, box +/-40 mm / +/-20 deg about (231, 394) mm, crop (96, 72, 160), 49152 renders, 40 epochs) with
  * gripper DR at the capture pose: each world draws its own left and right finger opening, uniform 0..47.5 mm per
    finger, and a small joint jitter of both arms about the capture (reset) pose, N(0, 0.03 rad) left, N(0, 0.02 rad)
    right. Reason: zed_v5 saw the left gripper at one fixed opening and uses its pad as a landmark; its y estimate moves
    -0.3 / +4.0 / +8.9 / +12.8 / +8.8 / +4.6 mm at left openings 47.5 / 40 / 30 / 20 / 10 / 0 mm (sim twins of the 89 real
    capture frames), and the real left gripper does not show that pad (real - twin(open) y +4.4 mm).
  * the measured PC geometry (scripts/perception/pc_geom_patch.py: 103.5 x 103 mm, port 2 19.8 mm from the case
    centre); labels = the physical port 2 (the frame of the edge fit 'real' and of the arm).
New file names only: runs/perception/zed_v6s<k>.pt (+ runs/perception/zed_v6/). zed_v5 and every older net are untouched.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl python scripts/perception/train_zed_v6.py --seed 20 --tag _v6s0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
LAYOUT = 'results/placement/task_layout_real_v2.json'
NOMINAL_V6 = (.231, .394)
ZED_CROP_V6 = (96, 72, 160)
BOX = ('0.04', '0.04', '20')
FINGER_MAX = .0475
JITTER = {'left': .03, 'right': .02}


def patch_renderer():
    import torch
    from blind_insertion.perception import render_data as RD
    orig = RD.ApproachRenderer.sample_top

    def sample_top(self, *a, **kw):
        h = self.h
        if not getattr(h, '_v6', False):
            reset0 = h.reset

            def reset(*ra, **rkw):
                out = reset0(*ra, **rkw)
                rt, n, dev, g = self.rt, self.n, self.dev, self.gen
                for side in ('left', 'right'):
                    q = rt.indices[side][0]
                    rt.q[:, q] = rt.q[:, q] + torch.randn(n, 6, generator=g, device=dev) * JITTER[side]
                lf = torch.rand(n, 1, generator=g, device=dev) * FINGER_MAX
                rf = torch.rand(n, 1, generator=g, device=dev) * FINGER_MAX
                rt.q[:, self.left_fingers] = lf.expand(n, 2)
                rt.q[:, rt.finger_qadr_right] = rf.expand(n, 2)
                return out
            h.reset = reset
            h._v6 = True
        return orig(self, *a, **kw)
    RD.ApproachRenderer.sample_top = sample_top


def main():
    import torch
    import pc_geom_patch
    pc_geom_patch.apply()
    patch_renderer()
    import train_approach_estimators as T
    T.NOMINAL['port_xy'] = NOMINAL_V6
    T.ZED_CROP = ZED_CROP_V6
    argv = sys.argv[1:]
    defaults = {'--which': ['zed'], '--n': ['49152'], '--epochs': ['40'], '--vary-arms': ['0'], '--pc-range': list(BOX),
                '--task-layout': [LAYOUT], '--out-dir': ['runs/perception/zed_v6']}
    for k, v in defaults.items():
        if k not in argv:
            argv += [k, *v]
    sys.argv = [sys.argv[0]] + argv
    import argparse
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--tag', default=''); p.add_argument('--out-dir'); p.add_argument('--pc-range', type=float, nargs=3)
    a, _ = p.parse_known_args(argv)
    T.main()
    src = Path(a.out_dir) / f'zed{a.tag}.pt'
    ck = torch.load(src, map_location='cpu', weights_only=False)
    ck.update(nominal_port_xy=NOMINAL_V6, zed_crop=ZED_CROP_V6, layout=LAYOUT, pc_pose_range=list(a.pc_range), argv=argv,
              pc_geom=pc_geom_patch.REAL, gripper_dr=dict(finger_max_m=FINGER_MAX, jitter_rad=JITTER))
    dst = Path('runs/perception') / f'zed{a.tag}.pt'
    if dst.exists():
        raise SystemExit(f'{dst} exists: new names only')
    torch.save(ck, dst)
    print('wrote', dst)


if __name__ == '__main__':
    main()
