"""ZED PC-pose estimator on real frames (duchamp3, 2026-10-28 curated session).

  bimanual_stationary_001: PC still; reference pose from the manual lid landmark
      (results/usbc/bimanual_20261029/overlays/scene_annotation.json: port (0.2142, 0.3308) m, yaw ~0,
      +/-4 px at 640 px, i.e. roughly +/-5 mm; yaw 'approximate, not fitted'). Reports agreement and the
      frame-to-frame spread.
  pc_manual_motion_001, other sessions: no reference; estimate spread and trajectories only.
Real frames go through scenes.physical_rig.real_top_policy_image (same crop as the sim camera).

    PYTHONPATH=. .venv/bin/python scripts/perception/check_real_zed.py
"""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch

from blind_insertion.perception.nets import PoseNet
from blind_insertion.scenes.physical_rig import real_top_policy_image
from train_approach_estimators import NOMINAL, zed_crop

ROOT = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2/cameras')
REF = json.loads(Path('results/usbc/bimanual_20261029/overlays/manifest.json').read_text())['scene_annotation']


def main():
    dev = torch.device('cuda:0')
    ck = torch.load('runs/perception/zed.pt', map_location=dev)
    net = PoseNet(len(ck['scale']), ck['size']).to(dev); net.load_state_dict(ck['net']); net.eval()
    sc = ck['scale'].to(dev)
    out = {'method': __doc__.strip(), 'reference': REF, 'sessions': {}}
    crops = []
    for sess in sorted(p.name for p in ROOT.iterdir() if (p / 'zed').is_dir()):
        files = sorted(glob.glob(str(ROOT / sess / 'zed' / 'raw_*.png')))
        if not files:
            continue
        imgs = np.stack([real_top_policy_image(cv2.imread(f), 320) for f in files])
        x = zed_crop(torch.tensor(imgs, device=dev))
        with torch.no_grad():
            e = torch.cat([net(x[i:i + 64]) for i in range(0, len(x), 64)]) * sc
        e = e.cpu().numpy()
        port = np.array(NOMINAL['port_xy']) + e[:, :2] * 1e-3
        s = {'n_frames': len(files), 'port_xy_m_mean': port.mean(0).tolist(), 'port_xy_mm_std': (port.std(0) * 1e3).tolist(),
             'yaw_deg_mean': float(e[:, 2].mean()), 'yaw_deg_std': float(e[:, 2].std()),
             'port_xy_mm_range': (np.ptp(port, 0) * 1e3).tolist()}
        if sess == REF['trial']:
            d = (port - np.array(REF['derived_port_xy_m'])) * 1e3
            s['vs_reference_mm'] = {'mean_dx': float(d[:, 0].mean()), 'mean_dy': float(d[:, 1].mean()),
                                    'mean_dist': float(np.linalg.norm(d.mean(0))), 'yaw_vs_reference_deg': float(e[:, 2].mean() - REF['pc_yaw_deg'])}
        out['sessions'][sess] = s
        crops.append(x[0].cpu().numpy())
        print(sess, json.dumps(s))
    Path('results/usbc/policy/approach/real_zed_check.json').write_text(json.dumps(out, indent=1) + '\n')
    cv2.imwrite('results/usbc/policy/approach/views/real_zed_crops.png',
                cv2.cvtColor(np.concatenate([np.concatenate(crops[i:i + 6], 1) for i in range(0, len(crops) - len(crops) % 6, 6)], 0), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
