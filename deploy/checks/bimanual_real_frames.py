"""Score the bundle's ZED nets on real frames before any task motion (docs/campaign/PHYSICAL_TODO_20261030.md
section 6; captures from scripts/real/snapshot.py pose).

For every <tag>.json in the given capture directories (snapshot.py writes <tag>_zed_left.png, <tag>_wrist_<side>.png
and the joint readings): real preprocessing (real_top_policy_image, 320 px) -> ZedEstimator (PC port pose +
presence / OOD / arm-pose gate, arms = readings + rig offsets, right finger from the gripper reading) and, at the
reader capture pose, ReaderZedEstimator (reader + presence / arm gate). With data/zed_gt/spots.json (or --spots) the
PC error against the measured spot is reported per capture. The arm gate SHOULD reject frames away from the
capture pose (folded, parked): those rows show the estimate anyway.

    PYTHONPATH=. .venv/bin/python -m deploy.checks.bimanual_real_frames data/zed_gt/spot1 data/zed_gt/spot2 ... [--spots data/zed_gt/spots.json]
spots.json: {"spot1": {"port_xy": [x, y], "yaw_deg": d, "method": "...", "unc_mm": 1.0}, ...} (key = capture dir name)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))
P = ROOT / 'runs/perception'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--spots', default=str(ROOT / 'data/zed_gt/spots.json'))
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    import cv2
    import rig
    from blind_insertion.perception.reader_zed import ReaderZedEstimator
    from blind_insertion.perception.zed_estimator import ZedEstimator
    from blind_insertion.scenes.physical_rig import real_top_policy_image
    from deploy.run_bimanual import gripper_maps, load_cfg
    cfg = load_cfg(ROOT / 'deploy/config/bimanual.json')
    dev = torch.device(a.device)
    z = ZedEstimator(str(dev), pose=str(P / 'zed.pt'), presence=str(P / f'{cfg["chain_args"]["zed_gate"]}.pt'))
    rz = ReaderZedEstimator(str(P / f'{cfg["pick"]["reader_estimator"]}.pt'), str(dev))
    gm = gripper_maps(cfg)['right']
    spots = json.loads(Path(a.spots).read_text()) if Path(a.spots).exists() else {}
    rows = []
    for d in a.dirs:
        d = Path(d)
        for js in sorted(d.glob('*.json')):
            meta = json.loads(js.read_text())
            tag = js.stem
            img = d / f'{tag}_zed_left.png'
            if not img.exists() or 'joints_measured' not in meta:
                continue
            top = torch.as_tensor(real_top_policy_image(cv2.imread(str(img)), 320), device=dev)[None]
            jm = meta['joints_measured']
            if 'right' not in jm or 'left' not in jm:
                continue
            t = lambda s: torch.tensor((np.asarray(jm[s][:6]) + rig.joint_offsets(s))[None], dtype=torch.float32, device=dev)  # noqa: E731
            fo = torch.tensor([gm.finger_m(jm['right'][6]) if len(jm['right']) > 6 else 0.0], device=dev)
            xy, yaw, ok, info = z(top, arms=(t('left'), t('right'), fo))
            rxy, ryaw, rok = rz.estimate(top, t('left'), t('right'), fo)
            row = dict(dir=str(d), tag=tag, pose=meta.get('pose'), port_xy=xy[0].tolist(), yaw_deg=math.degrees(float(yaw[0])),
                       pc_gate=bool(ok[0]), **{f'pc_{k}': float(v[0]) for k, v in info.items()},
                       reader_xy=rxy[0].tolist(), reader_yaw_deg=math.degrees(float(ryaw[0])), reader_gate=bool(rok[0]),
                       reader_presence=None if rz.present(top) is None else float(rz.present(top)[0]))
            sp = spots.get(d.name)
            if sp is not None:
                row['pc_err_xy_mm'] = float(np.linalg.norm(np.asarray(row['port_xy']) - np.asarray(sp['port_xy'])) * 1e3)
                row['pc_err_yaw_deg'] = row['yaw_deg'] - sp['yaw_deg']
            rows.append(row)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
