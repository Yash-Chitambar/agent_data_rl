"""Current sim vs current real, side by side and blended, with the PC placement space drawn in.

Frame: `bimanual_stationary_001` (ZED top + both wrists), sim at the logged joint angles with the
training scene (fitted backdrop, fitted wrist cameras, encoder jaw map) and the training layout
(PC at the `BimanualCfg.task_layout` nominal, reader at the centre of its region). The real PC
is wherever it stood in that session.

Placement space (from `placement/maps.npz`, PC square to the table, drawn on the table plane as
PC-centre positions):
  green  : every PC centre where a single placement works (both arms, see explore_pc_placement);
  yellow : centres whose whole +/-3 cm / +/-10 deg training box works (robust nominals);
  cyan   : the training randomization box of the PC centre, PC outline at the nominal;
  magenta: the reader's pick region (training randomization).

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/real2sim/render_layout_views.py
Writes results/usbc/bimanual_20261029/layout_views/{zed,wrist_right,wrist_left}.png, summary.png.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import mujoco
import numpy as np

from blind_insertion.assets.real_camera import real_policy_image
from blind_insertion.envs.bimanual_runtime import BimanualCfg
from blind_insertion.scenes import pc_bench as S, physical_rig as R
from fit_gripper_tips import BUNDLE
from render_sim_real_alignment import FIT, TRIAL, VIEWS, label, shift_cameras

RES = 640
COL = {'any': (60, 220, 60), 'robust': (255, 220, 0), 'box': (0, 220, 255), 'reader': (255, 0, 255)}


def project(m, d, cam, pts):
    """World points (N,3) -> pixel (N,2) in the RES x RES render of camera `cam`; nan if behind."""
    c = m.camera(cam).id
    rel = (pts - d.cam_xpos[c]) @ d.cam_xmat[c].reshape(3, 3)
    f = RES / 2 / np.tan(np.radians(m.cam_fovy[c]) / 2)
    z = -rel[:, 2]
    uv = np.stack([f * rel[:, 0] / z + RES / 2, -f * rel[:, 1] / z + RES / 2], -1)
    uv[z <= 1e-3] = np.nan
    return uv


def region_polys(mask, xs, ys, shift):
    """Outlines of a boolean (x, y) grid mask as world xy polygons (cell edges), shifted by `shift`."""
    up = 8
    big = cv2.resize(mask.astype(np.uint8), (mask.shape[1] * up, mask.shape[0] * up), interpolation=cv2.INTER_NEAREST)
    big = np.pad(big, 1)
    cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    out = []
    for c in cs:
        c = c[:, 0].astype(float) - 1                      # (col=y, row=x) in upsampled cells
        x = xs[0] - dx / 2 + c[:, 1] / up * dx
        y = ys[0] - dy / 2 + c[:, 0] / up * dy
        out.append(np.stack([x + shift[0], y + shift[1]], -1))
    return out


def rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def draw_layer(img, m, d, cam, polys, z, colour, fill, thick=2):
    over = img.copy()
    for poly in polys:
        dense = np.concatenate([np.linspace(a, b, 12, endpoint=False) for a, b in zip(poly, np.roll(poly, -1, 0))])
        uv = project(m, d, cam, np.c_[dense, np.full(len(dense), z)])
        if np.isnan(uv).any():
            continue
        pts = np.round(uv).astype(np.int32)
        if fill:
            cv2.fillPoly(over, [pts], colour)
        cv2.polylines(img, [pts], True, colour, thick, cv2.LINE_AA)
    if fill:
        cv2.addWeighted(over, .35, img, .65, 0, dst=img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('results/usbc/bimanual_20261029/layout_views'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec, fit = R.load(), json.loads(FIT.read_text())
    profile = json.loads((BUNDLE / 'reference/right_arm_20261027/rig_profile_provisional_v0.json').read_text())
    layout = json.loads(Path(BimanualCfg.task_layout).read_text())
    nom, reader = layout['nominal'], layout['reader_region']
    maps = np.load(Path(BimanualCfg.task_layout).parent / 'maps.npz')
    xs, ys, yaws = maps['xs'], maps['ys'], maps['yaws']
    k0 = int(np.argmin(abs(yaws)))

    m, lay = S.build_scene(detailed=True, bimanual=True, support_mode='free', profile=profile, measured_support=True)
    shift_cameras(m, fit)
    d = mujoco.MjData(m)
    S.place_pc(m, d, port_xy=nom['port_xy_m'], yaw_deg=nom['yaw_deg'])
    S.place_reader(m, d, lay, xy=reader['centre_xy_m'], yaw_deg=0.)
    table_top = float(m.geom('table').pos[2] + m.geom('table').size[2])
    pc = profile['pc']['measurements']
    from blind_insertion.scenes.pc_rear import layout as rear
    centre_shift = np.array([pc['depth']['value'] / 2, -(pc['width']['value'] / 2 - rear(profile)['target_from_left'])])
    cx, cy = np.array(nom['port_xy_m']) + centre_shift
    (bx, by, _) = BimanualCfg.pc_pose_range
    half = np.array([pc['depth']['value'], pc['width']['value']]) / 2
    layers = [
        ('any', region_polys(maps['score'][:, :, k0] > 0, xs, ys, centre_shift), table_top + .001, True),
        ('robust', region_polys(maps['robust'][:, :, k0] > 0, xs, ys, centre_shift), table_top + .002, True),
        ('box', [rect(cx - bx, cx + bx, cy - by, cy + by)], table_top + .003, False),
        ('box', [rect(cx - half[0], cx + half[0], cy - half[1], cy + half[1])], table_top + pc['height_with_feet']['value'], False),
        ('reader', region_polys(maps['reader_free'], maps['rx'], maps['ry'], (0., 0.)), table_top + .003, False),
    ]

    enc = {}
    for side in ('right', 'left'):
        with (BUNDLE / 'motion' / TRIAL / f'encoder_{side}.csv').open() as f:
            rows = list(csv.DictReader(f))
        enc[side] = (np.array([float(r['host_arrival_s']) for r in rows]), np.array([[float(r[f'q_{j}_rad']) for j in range(1, 8)] for r in rows]))
    frames = json.loads(Path('results/usbc/bimanual_20261029/overlays_pad_correction/manifest.json').read_text())['views']
    r = mujoco.Renderer(m, RES, RES)
    rows_out = []
    try:
        for name, camera, side in VIEWS:
            fr = frames[name]; t = int(fr['host_receive_ns']) / 1e9
            for arm, prefix in (('right', ''), ('left', 'left_')):
                times, qs = enc[arm]; q = qs[int(np.argmin(abs(times - t)))]
                for j in range(6):
                    d.qpos[m.joint(prefix + f'joint{j + 1}').qposadr] = q[j] + spec['arms'][arm]['joint_offsets_rad'][j]
                gm = fit['arms'][arm]['gripper_map']
                for f in ('left', 'right'):
                    d.qpos[m.joint(prefix + f + '_finger').qposadr] = float(np.clip(gm['slope'] * q[6] + gm['intercept'], 0, .0475))
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=camera); sim = r.render().copy()
            raw = cv2.imread(str(BUNDLE / 'cameras' / TRIAL / name / fr['frame']))
            real = R.real_top_policy_image(raw, RES) if side is None else real_policy_image(raw, spec, RES, side)
            blend = cv2.addWeighted(real, .5, sim, .5, 0)
            panels = [real.copy(), sim.copy(), blend]
            for img, txt in zip(panels, ('REAL (2026-10-28 session)', 'SIM (training scene + layout)', 'OVERLAY 50/50')):
                label(img, [txt, name])
            row = [np.hstack(panels)]
            if side is None:                      # placement space on the top view
                spaced = []
                for img, txt in ((real.copy(), 'REAL + PC placement space'), (sim.copy(), 'SIM + PC placement space')):
                    for key, polys, z, fill in layers:
                        draw_layer(img, m, d, camera, polys, z, COL[key], fill)
                    label(img, [txt])
                    spaced.append(img)
                legend = np.zeros((RES, RES, 3), np.uint8)
                for i, (key, txt) in enumerate((('any', 'PC centre: single placement works'),
                                                ('robust', 'PC centre: whole +/-3cm/10deg box works'),
                                                ('box', 'training box (+/-3 cm) + PC at nominal'),
                                                ('reader', 'reader pick region (training)'))):
                    cv2.rectangle(legend, (20, 40 + 50 * i), (50, 66 + 50 * i), COL[key], -1)
                    cv2.putText(legend, txt, (62, 60 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
                notes = [f'nominal port ({nom["port_xy_m"][0]:.2f}, {nom["port_xy_m"][1]:.2f}) m, yaw {nom["yaw_deg"]:.0f} deg',
                         'right arm base frame: x toward wall, y toward left arm',
                         'regions drawn for a PC square to the table (yaw 0)',
                         'real PC = 2026-10-28 placement, not moved yet']
                for i, t_ in enumerate(notes):
                    cv2.putText(legend, t_, (20, 300 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, .55, (200, 200, 200), 1, cv2.LINE_AA)
                row.append(np.hstack([*spaced, legend]))
            img = np.vstack(row)
            cv2.imwrite(str(a.out / f'{name}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            rows_out.append(img)
    finally:
        r.close()
    sheet = np.vstack(rows_out)
    sheet = cv2.resize(sheet, (sheet.shape[1] // 2, sheet.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(a.out / 'summary.png'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
