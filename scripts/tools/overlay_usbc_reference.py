"""Render the recorded YAM pose beside matching real policy crops; offline only.

Example: MUJOCO_GL=egl .venv/bin/python scripts/tools/overlay_usbc_reference.py \
    --ref data/sim_ref/usbc_lab_20261027_port --tag pointed \
    --out results/real/usbc_overlay_20261027 --port-xy 0.321370 0.236917
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import mujoco
import numpy as np
from PIL import Image

from blind_insertion.assets.real_camera import real_policy_image
from blind_insertion.scenes import pc_bench as S, physical_rig as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=Path, required=True)
    ap.add_argument('--tag', default='pointed')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--port-xy', nargs=2, type=float, default=S.PORT_XY)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((args.ref / f'{args.tag}.json').read_text())
    spec = R.load()
    m, lay = S.build_scene()
    d = mujoco.MjData(m)
    for side, prefix in [('right', ''), ('left', 'left_')]:
        q = np.asarray(snapshot['joints_measured'][side][:6]) + spec['arms'][side]['joint_offsets_rad']
        for i, value in enumerate(q, 1):
            d.qpos[m.joint(prefix + f'joint{i}').qposadr] = value
        for finger in ('left', 'right'):
            d.qpos[m.joint(prefix + finger + '_finger').qposadr] = .0475
    S.place_pc(m, d, port_xy=args.port_xy)
    mujoco.mj_forward(m, d)
    renderer = mujoco.Renderer(m, height=640, width=640)
    try:
        for camera, file, side in [('top', f'{args.tag}_zed_left.png', None),
                                   ('wrist_cam', f'{args.tag}_wrist_right.png', 'right'),
                                   ('left_wrist_cam', f'{args.tag}_wrist_left.png', 'left')]:
            renderer.update_scene(d, camera=camera)
            sim = Image.fromarray(renderer.render().copy())
            raw = cv2.imread(str(args.ref / file))
            if raw is None:
                raise FileNotFoundError(args.ref / file)
            real = R.real_top_policy_image(raw, 640) if side is None else real_policy_image(raw, spec, 640, side)
            real = Image.fromarray(real)
            pair = Image.new('RGB', (1280, 640))
            pair.paste(real, (0, 0)); pair.paste(sim, (640, 0))
            pair.save(args.out / f'{camera}_side_by_side.png')
            Image.blend(real, sim, .5).save(args.out / f'{camera}_blend.png')
    finally:
        renderer.close()
    (args.out / 'manifest.json').write_text(json.dumps({'reference': str(args.ref), 'tag': args.tag,
        'port_xy_m': list(args.port_xy), 'camera_spec': str(R.SPEC),
        'note': 'Baseline fixed-fixture geometry; visual comparison only. PC pose is assumed at port_xy.'}, indent=2))


if __name__ == '__main__':
    main()
