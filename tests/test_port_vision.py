"""Port vision: fail-safe on bad input, sign and scale on real frames when the data set is present."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from deploy.bimanual.port_vision import measure

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'data/real_training/port_v1/manifest.jsonl'


def test_bad_input_refused():
    assert not measure(np.zeros((480, 640, 3), np.uint8)).ok
    assert not measure(None).ok
    assert not measure(np.zeros((1024, 1280, 3), np.uint8)).ok
    assert not measure(np.zeros((1024, 1280, 3), np.uint8), side='left').ok
    n = np.random.default_rng(0).integers(0, 255, (1024, 1280, 3), dtype=np.uint8)
    m = measure(n)
    assert not m.ok and m.reason


@pytest.mark.skipif(not MAN.exists(), reason='real data set not here')
def test_real_frames_follow_labels():
    rows = [json.loads(x) for x in open(MAN)]
    rows = [r for r in rows if r['source'] == 'collect'][25:50]     # a held-out block
    e = []
    for r in rows:
        if not Path(ROOT / r['raw']).exists():
            pytest.skip('frames not here')
        m = measure(cv2.imread(str(ROOT / r['raw'])))
        if m.ok:
            e.append([m.across_mm - r['p_mm'][0], m.up_mm - r['p_mm'][1], m.gap_mm - r['p_mm'][2]])
    assert len(e) >= 15
    assert (np.std(e, axis=0) < [1.5, 2.5, 1.5]).all()
