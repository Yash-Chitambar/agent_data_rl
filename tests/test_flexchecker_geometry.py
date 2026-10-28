"""Table-clearance geometry regressions; no hardware or GPU required."""
from types import SimpleNamespace

import mujoco
import numpy as np

from deploy.safety_task import FlexChecker


def test_oriented_capsule_bottom():
    checker = object.__new__(FlexChecker)
    checker.m = SimpleNamespace(geom_type=np.array([mujoco.mjtGeom.mjGEOM_CAPSULE]),
                                geom_size=np.array([[0.03, 0.10, 0.0]]), geom_rbound=np.array([0.13]))
    checker.d = SimpleNamespace(geom_xpos=np.array([[0.0, 0.0, 0.2]]),
                                geom_xmat=np.array([np.eye(3).reshape(-1)]))
    assert abs(checker._bottom(0, None) - 0.07) < 1e-12  # vertical
    checker.d.geom_xmat[0] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).reshape(-1)
    assert abs(checker._bottom(0, None) - 0.17) < 1e-12  # horizontal
