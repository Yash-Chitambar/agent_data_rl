"""Real-photo texture patches for the visual domain randomization (`envs/visual_dr.py`).

Crops from the rectified ZED frame of `bimanual_stationary_001` (raw_001220.png, 1920x1080): two
clean tabletop patches, the wall, the paper sign, the Ryobi case and the black tower. Each patch is
mirror-tiled 2x2 (seamless when repeated) and resized to 256x256 RGB PNG in
`blind_insertion/assets/visual_dr/`. The DR mixes them with procedural patterns on the table and wall.

    PYTHONPATH=. .venv/bin/python scripts/real2sim/extract_real_textures.py
"""
from pathlib import Path

import cv2
import numpy as np

BUNDLE = Path('/home/yashchitambar/usbc_artifacts/left_bimanual_session_20261028_curated_v2')
FRAME = BUNDLE / 'cameras/bimanual_stationary_001/zed/raw_001220.png'
OUT = Path(__file__).resolve().parents[2] / 'blind_insertion/assets/visual_dr'
CROPS = {   # name: (x0, y0, x1, y1) in the 1920x1080 frame
    'real_table_a': (760, 460, 940, 600),
    'real_table_b': (1100, 760, 1280, 880),
    'real_wall': (900, 40, 1400, 300),
    'real_sign': (0, 390, 330, 510),
    'real_ryobi': (580, 0, 810, 140),
    'real_tower': (1680, 460, 1900, 860),
}


def main():
    img = cv2.imread(str(FRAME))
    assert img is not None, FRAME
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (x0, y0, x1, y1) in CROPS.items():
        p = img[y0:y1, x0:x1]
        p = np.vstack([np.hstack([p, p[:, ::-1]]), np.hstack([p[::-1], p[::-1, ::-1]])])
        cv2.imwrite(str(OUT / f'{name}.png'), cv2.resize(p, (256, 256), interpolation=cv2.INTER_AREA))
        print(name, p.shape)


if __name__ == '__main__':
    main()
