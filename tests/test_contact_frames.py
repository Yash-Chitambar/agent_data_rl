import os
import cv2
import numpy as np
import pytest
from deploy.contact_frames import latest_complete_frame


def test_incomplete_latest_uses_recent_complete_then_rejects_stale(tmp_path):
    good=tmp_path/'raw_001.png'
    cv2.imwrite(str(good),np.zeros((8,8,3),dtype=np.uint8))
    (tmp_path/'raw_002.png').write_bytes(b'incomplete')
    path,frame=latest_complete_frame(tmp_path)
    assert path==good and frame.shape==(8,8,3)
    os.utime(good,(1,1))
    with pytest.raises(RuntimeError,match='no recent complete'):
        latest_complete_frame(tmp_path)


def test_empty_directory_refuses(tmp_path):
    with pytest.raises(RuntimeError):latest_complete_frame(tmp_path)
