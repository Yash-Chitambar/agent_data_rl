"""Fit the wall-side backdrop (`blind_insertion/scenes/backdrop.py`) to calibrated real frames.

Views: ZED top crop and both wrist crops from `bimanual_stationary_001` and the three
`free_settle` trials (both arms' encoder angles logged; wrist cameras use the runtime's
fitted mount). Geometry is fitted by edge alignment: the silhouette boundaries of each
backdrop object in the sim segmentation must land on Canny edges of the real frame
(truncated distance transform), robots/tips masked out. Objects are fitted one group at a
time (Nelder-Mead), then the colours are matched to the real median colour per object.
Writes `blind_insertion/configs/backdrop_duchamp3.json` and `report.json` + before/after.jpg (real | blend).

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/real2sim/fit_backdrop.py
"""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize

from blind_insertion.assets.real_camera import real_policy_image
from blind_insertion.scenes import backdrop as BD, pc_bench as S, physical_rig as R
from blind_insertion.scenes.measured_support import dimensions
from fit_gripper_tips import BUNDLE, yellow_mask
from render_sim_real_alignment import FIT, shift_cameras

RES, TAU = 320, 10.0
CAMS = (('zed', 'top', None), ('wrist_right', 'wrist_cam', 'right'), ('wrist_left', 'left_wrist_cam', 'left'))
GROUPS = {   # parameter group -> (scalar parameter paths, step, geom-name prefixes scored)
    # The raceway front sits on the measured table back edge (ZED junction ray: 0.517 m vs measured
    # 0.515 m); fitting it too let the top-down ZED trade depth for height (3 cm back, 2 cm tall).
    'raceway': ([('wall_x',), ('raceway_top_z',)], .01, ('bd_raceway',)),
    'plugs': ([('plug_y', 0), ('plug_y', 1), ('plug_y', 2), ('plug_z',), ('plug_spacing',), ('plug_length',)], .006,
              ('bd_plug', 'bd_plate')),
    'strip': ([('strip_x',), ('strip_y',), ('strip_yaw',), ('strip_size', 0), ('strip_size', 1), ('strip_size', 2)], .01,
              ('bd_strip',)),
    'device': ([('device_x',), ('device_y',), ('device_z',), ('device_tilt',), ('device_length',)], .01, ('bd_device',)),
    'cables': ([('cable_start', 0), ('cable_start', 1), ('cable_lift', 0), ('cable_lift', 1), ('cable_end', 0, 0),
                ('cable_end', 0, 1)], .02, ('bd_cable',)),
    'ryobi': ([('ryobi_y',), ('ryobi_z',), ('ryobi_size', 0), ('ryobi_size', 1)], .01, ('bd_ryobi',)),
}
COLOR_OF = {'wall': ('bd_wall',), 'raceway': ('bd_raceway',), 'black': ('bd_strip', 'bd_device'),
            'ryobi': ('bd_ryobi',)}


def get(p, path):
    for k in path:
        p = p[k]
    return p


def put(p, path, v):
    for k in path[:-1]:
        p = p[k]
    p[path[-1]] = float(v)


def views(spec, fit):
    """[(label, camera, side, real RGB, {arm: q7 encoder})]"""
    out = []

    def add(trial, frame_sel, q_at):
        cap = BUNDLE / 'cameras' / trial
        for label, camera, side in CAMS:
            with (cap / label / 'frames.csv').open() as f:
                rows = [r for r in csv.DictReader(f) if r['raw_png']]
            for r in frame_sel(rows):
                q = q_at(int(r['host_receive_ns']))
                if q is None:
                    continue
                raw = cv2.imread(str(cap / label / r['raw_png']))
                real = R.real_top_policy_image(raw, RES) if side is None else real_policy_image(raw, spec, RES, side)
                out.append((f'{trial}/{label}/{r["raw_png"]}', camera, side, real, q))

    enc = {}
    for arm in ('right', 'left'):
        with (BUNDLE / 'motion/bimanual_stationary_001' / f'encoder_{arm}.csv').open() as f:
            rows = list(csv.DictReader(f))
        enc[arm] = (np.array([float(r['host_arrival_s']) for r in rows]), np.array([[float(r[f'q_{j}_rad']) for j in range(1, 8)] for r in rows]))
    manifest = json.loads(Path('results/usbc/bimanual_20261029/overlays_pad_correction/manifest.json').read_text())['views']

    def stationary_q(t_ns):
        t = t_ns / 1e9
        return {arm: enc[arm][1][int(np.argmin(abs(enc[arm][0] - t)))] for arm in enc}
    add('bimanual_stationary_001', lambda rows: [r for r in rows if r['raw_png'] in {v['frame'] for v in manifest.values()}][:1],
        stationary_q)
    for k in (1, 2, 3):
        trial = json.loads((BUNDLE / f'contact/free_settle_00{k}/trial.json').read_text())['rows']
        ts = np.array([r['host_monotonic_ns'] for r in trial])

        def q_at(t_ns, trial=trial, ts=ts):
            i = int(np.argmin(abs(ts - t_ns)))
            if abs(ts[i] - t_ns) > 2e8:
                return None
            obs = trial[i]['observations']
            return {arm: np.r_[obs[arm]['q_encoder'][:6], obs[arm]['q_encoder'][6] if len(obs[arm]['q_encoder']) > 6 else 1.]
                    for arm in ('right', 'left')}
        add(f'left_free_settle_00{k}', lambda rows: [rows[len(rows) // 4], rows[3 * len(rows) // 4]], q_at)
    return out


class Fitter:
    def __init__(self, params):
        self.spec, self.fit = R.load(), json.loads(FIT.read_text())
        profile = json.loads((BUNDLE / 'reference/right_arm_20261027/rig_profile_provisional_v0.json').read_text())
        self.table_top = dimensions()['table_top_z_m']
        self.m, lay = S.build_scene(detailed=True, bimanual=True, support_mode='free', profile=profile, measured_support=True)
        shift_cameras(self.m, self.fit)
        self.d = mujoco.MjData(self.m)
        S.place_pc(self.m, self.d, port_xy=(5., 5.)); S.place_reader(self.m, self.d, lay)
        self.d.qpos[lay.peg_qadr:lay.peg_qadr + 3] = [0, 0, -10]
        self.r = mujoco.Renderer(self.m, RES, RES)
        self.p = copy.deepcopy(params)
        self.names = {g: self.m.geom(g).name for g in range(self.m.ngeom)}
        self.bd = np.array([self.names[g].startswith('bd_') for g in range(self.m.ngeom)] + [False])
        self.robot = np.array([self.m.geom_bodyid[g] != 0 and not self.names[g].startswith('bd_') for g in range(self.m.ngeom)] + [False])
        self.views = []
        for label, camera, side, real, q in views(self.spec, self.fit):
            gray = cv2.GaussianBlur(cv2.cvtColor(real, cv2.COLOR_RGB2GRAY), (3, 3), 0)
            edges = cv2.Canny(gray, 25, 75)
            dt = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
            tips = cv2.dilate(yellow_mask(real).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            self.views.append({'label': label, 'camera': camera, 'side': side, 'real': real, 'dt': dt, 'tips': tips, 'q': q})
        self.cache = {}

    def pose(self, v):
        for arm, prefix in (('right', ''), ('left', 'left_')):
            q = v['q'][arm]
            for j in range(6):
                self.d.qpos[self.m.joint(prefix + f'joint{j + 1}').qposadr] = q[j] + self.spec['arms'][arm]['joint_offsets_rad'][j]
            gm = self.fit['arms'][arm]['gripper_map']
            for f in ('left', 'right'):
                self.d.qpos[self.m.joint(prefix + f + '_finger').qposadr] = float(np.clip(gm['slope'] * q[6] + gm['intercept'], 0, .0475))
        mujoco.mj_forward(self.m, self.d)

    def seg(self, v):
        self.pose(v)
        self.r.enable_segmentation_rendering(); self.r.update_scene(self.d, camera=v['camera'])
        s = self.r.render(); self.r.disable_segmentation_rendering()
        ids = np.where(s[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[:, :, 0], -1)
        return ids

    def boundary_cost(self, prefixes):
        BD.update(self.m, self.p, self.table_top)
        total, n = 0., 0
        for v in self.views:
            ids = self.seg(v)
            mine = np.isin(ids, [g for g, nm in self.names.items() if nm.startswith(prefixes)])
            if not mine.any():
                continue
            robot = self.robot[ids] | v['tips']
            robot = cv2.dilate(robot.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            edge = np.zeros_like(mine)
            edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]; edge[1:] |= ids[1:] != ids[:-1]
            edge &= cv2.dilate(mine.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~robot
            edge[:2] = edge[-2:] = False; edge[:, :2] = edge[:, -2:] = False
            if edge.sum() < 5:
                continue
            total += float(np.minimum(v['dt'][edge], TAU).mean()); n += 1
        return total / max(n, 1) if n else TAU

    def fit_group(self, name, iters):
        paths, step, prefixes = GROUPS[name]
        x0 = np.array([get(self.p, pth) for pth in paths])

        def f(x):
            for pth, val in zip(paths, x0 + x * step):
                put(self.p, pth, val)
            # Physical prior: the plugs sit in outlets on the raceway's vertical face. Without it the
            # top-down ZED prefers a low, deep shelf whose two edges land on the real face's edges.
            if self.p['raceway_top_z'] < self.p['plug_z'] + .025 or self.p['wall_x'] < self.p['raceway_front_x'] + .005:
                return TAU
            return self.boundary_cost(prefixes)
        c0 = f(np.zeros(len(paths)))
        res = minimize(f, np.zeros(len(paths)), method='Nelder-Mead',
                       options={'maxfev': iters, 'xatol': .05, 'fatol': .01, 'initial_simplex': np.vstack([np.zeros(len(paths)), np.eye(len(paths))])})
        f(res.x)
        return c0, float(res.fun)

    def fit_colors(self):
        for _ in range(3):
            BD.update(self.m, self.p, self.table_top)
            ratios = {k: [] for k in COLOR_OF}
            for v in self.views:
                ids = self.seg(v)
                self.r.update_scene(self.d, camera=v['camera']); sim = self.r.render().astype(float)
                for key, prefixes in COLOR_OF.items():
                    mask = np.isin(ids, [g for g, nm in self.names.items() if nm.startswith(prefixes)]) & ~v['tips']
                    mask = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                    if mask.sum() > 30:
                        ratios[key].append(np.median(v['real'][mask].astype(float), 0) / np.maximum(np.median(sim[mask], 0), 1.))
            emission = self.p.setdefault('emission', {})
            for key, rs in ratios.items():
                if rs:
                    c = np.array(self.p['colors'][key][:3]) * np.median(rs, 0)
                    if c.max() > 1:       # brighter than the lit material can get: add emission
                        emission[key] = round(float((1 + emission.get(key, 0.)) * c.max() - 1), 3)
                        c /= c.max()
                    self.p['colors'][key] = [*np.clip(c, 0, 1).round(3).tolist(), 1]

    def overlays(self, out):
        BD.update(self.m, self.p, self.table_top)
        tiles = []
        for v in self.views:
            self.pose(v); self.r.update_scene(self.d, camera=v['camera']); sim = self.r.render()
            ids = self.seg(v)
            img = cv2.addWeighted(v['real'], .5, sim, .5, 0)
            for prefixes, col in ((('bd_raceway', 'bd_wall'), (0, 200, 255)), (('bd_plug', 'bd_plate'), (255, 0, 255)),
                                  (('bd_strip', 'bd_cable', 'bd_device'), (255, 140, 0)), (('bd_ryobi',), (0, 255, 0))):
                mask = np.isin(ids, [g for g, nm in self.names.items() if nm.startswith(prefixes)]).astype(np.uint8)
                cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 1)
            tiles.append(np.hstack([v['real'], img]))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) if i + 1 < len(tiles) else np.hstack([tiles[i], np.zeros_like(tiles[i])])
                           for i in range(0, len(tiles), 2)])
        sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('results/usbc/bimanual_20261029/backdrop_fit'))
    ap.add_argument('--iters', type=int, default=250)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--groups', nargs='+', default=list(GROUPS), choices=list(GROUPS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    params = BD.load()
    params['raceway_front_x'] = round(dimensions()['wall_edge_x_m'], 4)
    fz = Fitter(params)
    print(f'{len(fz.views)} views: ' + ', '.join(v['label'] for v in fz.views), flush=True)
    fz.overlays(a.out / 'before.jpg')
    report = {'views': [v['label'] for v in fz.views], 'tau_px': TAU, 'res': RES, 'groups': {}}
    for rnd in range(a.rounds):
        for name in a.groups:
            c0, c1 = fz.fit_group(name, a.iters)
            report['groups'].setdefault(name, []).append({'round': rnd, 'edge_px_before': round(c0, 2), 'edge_px_after': round(c1, 2)})
            print(f'round {rnd} {name}: mean edge distance {c0:.2f} -> {c1:.2f} px (of {RES})', flush=True)
    fz.fit_colors()
    fz.overlays(a.out / 'after.jpg')
    p = fz.p
    for k, v in list(p.items()):
        if isinstance(v, float):
            p[k] = round(v, 4)
        elif isinstance(v, list) and v and isinstance(v[0], float):
            p[k] = [round(x, 4) for x in v]
    p['cable_end'] = [[round(x, 4) for x in e] for e in p['cable_end']]
    BD.PARAMS.write_text(json.dumps(p, indent=1) + '\n')
    report['params'] = p
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report['groups'], indent=1))


if __name__ == '__main__':
    main()
