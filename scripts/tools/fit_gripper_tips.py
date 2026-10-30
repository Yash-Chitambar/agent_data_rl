"""Fit the FlexPoint tip offset and per-state opening to recorded wrist silhouettes.

The yellow tips are rigid relative to the wrist camera apart from the opening, so their
silhouette in the calibrated policy crop constrains the tip geometry in the gripper frame.
Per arm, a rigid offset of both finger bodies in the gripper frame (x: finger thickness,
toward the wrist camera; y: lateral/opening; z: along the fingers, -z is the approach) and
the opening per frame are fitted by IoU. Left fit frames: `left_reader_002` open and closed
empty (tips touching: absolute opening zero). Right: open frames only (no recorded closed
frame), so its offset trades off against the opening and is weaker. Held-out frames come
from other runs (other arm poses, backgrounds, lighting). The fitted offset cannot by
itself separate tip geometry from a hand-eye translation error: both move the near-field
tips while barely moving the far-field calibration board. Masks: HSV yellow in the lower
part of the crop (assumption).
Output: `results/usbc/bimanual_20261029/gripper_tips/fit.json` and overlays.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cv2
import mujoco
import numpy as np

from blind_insertion.assets.real_camera import real_policy_image
from blind_insertion.scenes import pc_bench as S, physical_rig as R

BUNDLE = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2')
RES, ROI = 320, 150                       # render size; rows above ROI are ignored (background)
# (arm, camera run, frame, gripper state, grasp log for joint angles or None for bimanual_stationary)
VIEWS = [('left', 'left_reader_002', 'raw_000510.png', 'open', 'fit'),
         ('left', 'left_reader_002', 'raw_000270.png', 'closed', 'fit'),
         ('left', 'left_free_settle_001', 'raw_000600.png', 'closed', 'held_out'),
         ('left', 'left_free_settle_001', 'raw_001800.png', 'closed', 'held_out'),
         ('right', 'bimanual_stationary_001', 'raw_000890.png', 'open', 'fit'),
         ('right', 'left_reader_002', 'raw_000510.png', 'open', 'fit'),
         ('right', 'left_free_settle_001', 'raw_000600.png', 'open', 'held_out'),
         ('right', 'left_free_settle_001', 'raw_001800.png', 'open', 'held_out')]
AXES = {'x_thickness_toward_camera': 0, 'y_lateral_opening': 1, 'z_along_fingers': 2}


def yellow_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 40) & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 80)
    m[:ROI] = False
    return m


class Scene:
    def __init__(self):
        spec = R.load()
        prof = json.loads((BUNDLE / 'reference/right_arm_20261027/rig_profile_provisional_v0.json').read_text())
        self.m, lay = S.build_scene(detailed=True, bimanual=True, support_mode='free', profile=prof, measured_support=True)
        self.d = mujoco.MjData(self.m)
        S.place_pc(self.m, self.d); S.place_reader(self.m, self.d, lay)
        self.spec, self.r = spec, mujoco.Renderer(self.m, RES, RES)
        self.r.enable_segmentation_rendering()
        self.base = {}
        for side, pre in (('right', ''), ('left', 'left_')):
            bodies = [self.m.body(pre + f + '_finger').id for f in ('left', 'right')]
            self.base[side] = (bodies, self.m.body_pos[bodies].copy())

    def set_arm(self, side, q_enc):
        pre = 'left_' if side == 'left' else ''
        for j in range(6):
            self.d.qpos[self.m.joint(pre + f'joint{j + 1}').qposadr] = q_enc[j] + self.spec['arms'][side]['joint_offsets_rad'][j]

    def tip_geoms(self, side):
        bodies = self.base[side][0]
        return [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] in bodies and self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                and self.m.mesh(self.m.geom_dataid[g]).name.endswith('soft_tips')]

    def render_tips(self, side, opening, offset):
        """Tip silhouette with both finger bodies shifted by `offset` (m) in the gripper frame."""
        pre = 'left_' if side == 'left' else ''
        bodies, pos0 = self.base[side]
        self.m.body_pos[bodies] = pos0 + offset
        for f in ('left', 'right'):
            self.d.qpos[self.m.joint(pre + f + '_finger').qposadr] = opening
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=pre + 'wrist_cam')
        seg = self.r.render()
        self.m.body_pos[bodies] = pos0
        sim = np.isin(seg[:, :, 0], self.tip_geoms(side)) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        sim[:ROI] = False
        return sim


def joint_angles(run, side, frame):
    """Encoder angles nearest the frame's host receive time (both logs share the host clock)."""
    import csv
    with (BUNDLE / 'cameras' / run / f'wrist_{side}' / 'frames.csv').open() as f:
        t = next(int(r['host_receive_ns']) for r in csv.DictReader(f) if r['raw_png'] == frame)
    # Camera and tips ride on the same link, so the silhouette does not depend on the arm
    # angles; they only set the background. Use the logged pose where one is at hand.
    if run == 'left_reader_002' and side == 'left':
        s = json.loads((BUNDLE / 'grasp/left_reader_002.json').read_text())['samples']
        return min(s, key=lambda x: abs(x['host_monotonic_ns'] - t))['q_encoder'][:6]
    if run.startswith('left_free_settle') and side == 'left':
        rows = json.loads((BUNDLE / 'contact' / run.replace('left_', '') / 'trial.json').read_text())['rows']
        r = min(rows, key=lambda x: abs(x['host_monotonic_ns'] - t))
        return r['observations']['left']['q_encoder'][:6]
    return [0.0] * 6


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('results/usbc/bimanual_20261029/gripper_tips'))
    p.add_argument('--span-mm', type=float, default=10.0)
    p.add_argument('--step-mm', type=float, default=0.5)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sc = Scene()
    views = []
    for side, run, frame, state, role in VIEWS:
        raw = cv2.imread(str(BUNDLE / 'cameras' / run / f'wrist_{side}' / frame))
        real = real_policy_image(raw, sc.spec, RES, side)
        views.append({'side': side, 'run': run, 'frame': frame, 'state': state, 'role': role, 'real': real,
                      'mask': yellow_mask(real), 'q': joint_angles(run, side, frame)})
    openings = {'open': np.arange(0.042, 0.0481, 0.001), 'closed': np.arange(-0.004, 0.0061, 0.001)}

    def score(offset, vs):
        out = []
        for v in vs:
            sc.set_arm(v['side'], v['q'])
            out.append(max((iou(sc.render_tips(v['side'], o, offset), v['mask']), o) for o in openings[v['state']]))
        return out

    def search(vs, center, half, step):
        best = None
        ax = np.arange(-half, half + 1e-9, step)
        for d in itertools.product(ax, ax, ax):
            offset = center + np.array(d)
            mean = float(np.mean([x[0] for x in score(offset, vs)]))
            if best is None or mean > best[0]:
                best = (mean, offset)
        return best

    report = {'method': __doc__.strip(), 'axes_gripper_frame': list(AXES), 'arms': {}, 'views': []}
    offsets, results = {}, []
    for side in ('left', 'right'):
        fit_views = [v for v in views if v['side'] == side and v['role'] == 'fit']
        coarse = search(fit_views, np.zeros(3), a.span_mm * 1e-3, 2e-3)
        fine = search(fit_views, coarse[1], 2e-3, a.step_mm * 1e-3)
        offsets[side] = fine[1]
        on_edge = bool(np.any(np.abs(fine[1]) >= a.span_mm * 1e-3 - 1e-9))
        report['arms'][side] = {'fitted_offset_mm': dict(zip(AXES, (fine[1] * 1e3).round(2).tolist())),
                                'fit_mean_iou': round(fine[0], 3), 'offset_on_search_edge': on_edge}
    for v in views:
        results.append((v, score(np.zeros(3), [v])[0], score(offsets[v['side']], [v])[0]))
    for v, b0, b1 in results:
        report['views'].append({k: v[k] for k in ('side', 'run', 'frame', 'state', 'role')} |
                               {'iou_current': round(b0[0], 3), 'opening_current_m': round(float(b0[1]), 4),
                                'iou_fitted_offset': round(b1[0], 3), 'opening_fitted_m': round(float(b1[1]), 4)})
        sc.set_arm(v['side'], v['q'])
        for tag, off, o in (('current', np.zeros(3), b0[1]), ('fitted', offsets[v['side']], b1[1])):
            sim = sc.render_tips(v['side'], o, off)
            ov = v['real'].copy()
            ov[v['mask']] = (.5 * ov[v['mask']] + [0, 127, 0]).astype('uint8')
            ov[sim] = (.5 * ov[sim] + [127, 0, 127]).astype('uint8')
            cv2.imwrite(str(a.out / f"{v['side']}_{v['run']}_{v['frame'][:-4]}_{tag}.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    sc.r.close()
    (a.out / 'fit.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report['arms'], indent=1))
    for v in report['views']:
        print(v)


if __name__ == '__main__':
    main()
