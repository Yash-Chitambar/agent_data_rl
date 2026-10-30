"""Real vs sim camera overlays, before and after the 2026-10-30 alignment work.

For one recorded stationary frame (ZED top + both wrists, `bimanual_stationary_001`), renders
the sim at the recorded joint angles twice:
  OLD: the scene as before (nominal wrist camera, jaws at a fixed 47.5 mm);
  NEW: the training runtime's alignment (wrist camera shifted by the caliper-held sweep fit,
       BimanualCfg.tip_offset_target='camera'; jaw opening from the recorded gripper encoder
       through the fitted encoder -> opening map).
Each panel blends sim over real and draws outlines: sim PC (orange), sim fingertips (magenta),
real yellow fingertips (green, wrist views). The PC sits at the manual top-image landmark
(`overlays/scene_annotation.json`); the cyan outline is the runtime's nominal PC pose (the
centre of the reset distribution). Tip IoU against the real yellow mask is printed per wrist.

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/tools/render_sim_real_alignment.py
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
from blind_insertion.scenes import pc_bench as S, physical_rig as R
from fit_gripper_tips import BUNDLE, yellow_mask

TRIAL = 'bimanual_stationary_001'
RES = 640
FIT = Path('results/usbc/gripper_sweep/sweep_20261030/fit.json')
ANNOTATION = Path('results/usbc/bimanual_20261029/overlays/scene_annotation.json')
VIEWS = [('zed', 'top', None), ('wrist_right', 'wrist_cam', 'right'), ('wrist_left', 'left_wrist_cam', 'left')]
COL = {'pc': (255, 140, 0), 'pc_nominal': (0, 220, 255), 'tips': (255, 0, 255), 'real_tips': (0, 255, 0)}


def annotated_port(spec, profile):
    """PC port xy from the manual lid-centre landmark in the ZED crop (as render_bimanual_capture)."""
    from blind_insertion.scenes.measured_support import dimensions
    from blind_insertion.scenes.pc_rear import layout
    ann = json.loads(ANNOTATION.read_text())
    z = dimensions()['table_top_z_m'] + float(profile['pc']['measurements']['height_with_feet']['value'])
    top = spec['top']; k = top['intrinsics_rectified']['HD1080']; pose = top['extrinsics_cam_in_right_base']
    cx, cy = round(k['cx']), round(k['cy']); crop = 2 * min(cx, 1920 - cx, cy, 1080 - cy)
    u, v = np.array(ann['pc_lid_center_px']) * crop / 640 + np.array([cx - crop / 2, cy - crop / 2])
    q = np.array(pose['quat']); q /= np.linalg.norm(q); rot = np.zeros(9); mujoco.mju_quat2Mat(rot, q)
    ray = rot.reshape(3, 3) @ np.array([(u - k['cx']) / k['fx'], (v - k['cy']) / k['fy'], 1.])
    centre = np.array(pose['pos']) + ray * ((z - pose['pos'][2]) / ray[2])
    depth = float(profile['pc']['measurements']['depth']['value']); width = float(profile['pc']['measurements']['width']['value'])
    return centre[:2] + [-depth / 2, width / 2 - layout(profile)['target_from_left']]


def shift_cameras(m, fit):
    """Same as BimanualRuntime with tip_offset_target='camera'."""
    for side, prefix in (('right', ''), ('left', 'left_')):
        off = np.array(list(fit['arms'][side]['fitted_offset_mm'].values())) * 1e-3
        g, cam = m.body(prefix + 'gripper'), m.camera(prefix + 'wrist_cam')
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, g.quat)
        m.cam_pos[cam.id] -= rot.reshape(3, 3) @ off


def masks(m, d, r, camera):
    r.enable_segmentation_rendering(); r.update_scene(d, camera=camera); seg = r.render(); r.disable_segmentation_rendering()
    geom = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    ids = seg[:, :, 0]
    pc = np.isin(ids, [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body('fixture').id]) & geom
    tips = np.isin(ids, [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                         and m.mesh(m.geom_dataid[g]).name.endswith('soft_tips')]) & geom
    return pc, tips


def outline(img, mask, colour, thick=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, colour, thick)


def label(img, lines):
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, default=Path('results/usbc/bimanual_20261029/alignment_20261030'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    spec = R.load(); fit = json.loads(FIT.read_text())
    profile = json.loads((BUNDLE / 'reference/right_arm_20261027/rig_profile_provisional_v0.json').read_text())
    capture = BUNDLE / 'cameras' / TRIAL
    enc = {}
    for side in ('right', 'left'):
        with (BUNDLE / 'motion' / TRIAL / f'encoder_{side}.csv').open() as f:
            rows = list(csv.DictReader(f))
        enc[side] = (np.array([float(r['host_arrival_s']) for r in rows]), np.array([[float(r[f'q_{j}_rad']) for j in range(1, 8)] for r in rows]))
    frames = json.loads(Path('results/usbc/bimanual_20261029/overlays_pad_correction/manifest.json').read_text())['views']
    port = annotated_port(spec, profile)

    scenes = {}
    for tag in ('old', 'new'):
        m, lay = S.build_scene(detailed=True, bimanual=True, support_mode='free', profile=profile, measured_support=True)
        if tag == 'new':
            shift_cameras(m, fit)
        d = mujoco.MjData(m); S.place_reader(m, d, lay)
        d.qpos[lay.peg_qadr:lay.peg_qadr + 3] = [0, 0, -10]          # reader absent in this capture
        scenes[tag] = (m, d, mujoco.Renderer(m, RES, RES))
    report = {'trial': TRIAL, 'pc_port_xy_annotated_m': port.tolist(), 'pc_port_xy_runtime_nominal_m': list(S.PORT_XY), 'views': {}}
    rows_out = []
    try:
        for name, camera, side in VIEWS:
            fr = frames[name]; t = int(fr['host_receive_ns']) / 1e9
            raw = cv2.imread(str(capture / name / fr['frame']))
            real = R.real_top_policy_image(raw, RES) if side is None else real_policy_image(raw, spec, RES, side)
            real_tips = yellow_mask(cv2.resize(real, (320, 320)))
            real_tips = cv2.resize(real_tips.astype(np.uint8), (RES, RES), interpolation=cv2.INTER_NEAREST).astype(bool)
            panels = [real.copy()]; label(panels[0], ['REAL', f'{name} {fr["frame"]}'])
            report['views'][name] = {}
            for tag, (m, d, r) in scenes.items():
                for arm, prefix in (('right', ''), ('left', 'left_')):
                    times, qs = enc[arm]; q = qs[int(np.argmin(abs(times - t)))]
                    for j in range(6):
                        d.qpos[m.joint(prefix + f'joint{j + 1}').qposadr] = q[j] + spec['arms'][arm]['joint_offsets_rad'][j]
                    gm = fit['arms'][arm]['gripper_map']
                    opening = .0475 if tag == 'old' else float(np.clip(gm['slope'] * q[6] + gm['intercept'], 0, .0475))
                    for f in ('left', 'right'):
                        d.qpos[m.joint(prefix + f + '_finger').qposadr] = opening
                S.place_pc(m, d); mujoco.mj_forward(m, d)
                pc_nominal, _ = masks(m, d, r, camera)
                S.place_pc(m, d, port_xy=port); mujoco.mj_forward(m, d)
                pc, tips = masks(m, d, r, camera)
                r.update_scene(d, camera=camera); sim = r.render()
                img = cv2.addWeighted(real, .5, sim, .5, 0)
                outline(img, pc_nominal, COL['pc_nominal'], 1); outline(img, pc, COL['pc'])
                outline(img, tips, COL['tips'])
                lines = [tag.upper() + (' (nominal wrist cam, jaws 47.5 mm)' if tag == 'old' else ' (fitted wrist cam, encoder jaw map)')]
                if side is not None:
                    roi = np.zeros_like(tips); roi[RES * 150 // 320:] = True       # fit_gripper_tips ROI
                    sim_t, real_t = tips & roi, real_tips & roi
                    iou = float((sim_t & real_t).sum() / max(1, (sim_t | real_t).sum()))
                    outline(img, real_t, COL['real_tips'], 1)
                    lines.append(f'fingertip IoU vs real: {iou:.2f}')
                    report['views'][name][tag] = {'tip_iou': round(iou, 3)}
                label(img, lines)
                panels.append(img)
            row = np.hstack(panels)
            cv2.imwrite(str(a.out / f'{name}.png'), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
            rows_out.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    finally:
        for _, _, r in scenes.values():
            r.close()
    sheet = np.vstack(rows_out)
    legend = np.zeros((40, sheet.shape[1], 3), np.uint8)
    x = 10
    for key, txt in (('pc', 'sim PC (landmark pose)'), ('pc_nominal', 'sim PC (runtime nominal)'), ('tips', 'sim fingertips'), ('real_tips', 'real fingertips')):
        cv2.rectangle(legend, (x, 12), (x + 20, 28), COL[key], -1); cv2.putText(legend, txt, (x + 26, 27), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        x += 250
    cv2.imwrite(str(a.out / 'summary.png'), cv2.cvtColor(np.vstack([legend, sheet]), cv2.COLOR_RGB2BGR))
    (a.out / 'report.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
