from deploy.grasp import reader_present


def test_real_calibration_rejects_empty_and_open():
    assert not reader_present(.038, .038, .3371)
    assert not reader_present(1., .038, .3371)
    assert reader_present(.3371, .038, .3371)
    assert not reader_present(float('nan'), .038, .3371)
    assert not reader_present(.1, .038, .04)
