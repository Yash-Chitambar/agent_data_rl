"""Depth for the ZED left image from the rectified HD1080 stereo pair (OpenCV SGBM, CPU).

The ZED SDK depth is not available on duchamp3 (no neural depth model; DEPTH_MODE.NONE in deploy/cameras.py), so the rig
saves the rectified right image and depth is computed here. z = fx * baseline / disparity.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

# ZED 2 at about 0.88 m above the table: the table and the PC lie at z 0.75-1.2 m -> disparity 105-170 px (fx 1055, b 120 mm).
MIN_DISP, NUM_DISP = 48, 224          # z 0.47 .. 2.6 m


def sgbm_depth(left_bgr, right_bgr, fx, baseline, roi=None, block=5, lr_check=1, uniq=10, speckle=100):
    """Depth (m, float32, 0 = invalid) for the left image. roi = (u0, v0, u1, v1) computes only that box
    (the right image gets the extra disparity margin); outside the box depth is 0."""
    t0 = time.perf_counter()
    H, W = left_bgr.shape[:2]
    if roi is None:
        roi = (0, 0, W, H)
    u0, v0, u1, v1 = [int(v) for v in roi]
    u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
    m = MIN_DISP + NUM_DISP
    ue = max(u0 - m, 0)                                  # extend to the left so the matcher sees the full range
    L = cv2.cvtColor(left_bgr[v0:v1, ue:u1], cv2.COLOR_BGR2GRAY)
    R = cv2.cvtColor(right_bgr[v0:v1, ue:u1], cv2.COLOR_BGR2GRAY)
    sg = cv2.StereoSGBM_create(minDisparity=MIN_DISP, numDisparities=NUM_DISP, blockSize=block,
                               P1=8 * block * block, P2=32 * block * block, disp12MaxDiff=lr_check,
                               uniquenessRatio=uniq, speckleWindowSize=speckle, speckleRange=2,
                               mode=cv2.STEREO_SGBM_MODE_HH)
    d = sg.compute(L, R).astype(np.float32) / 16.0
    d = d[:, u0 - ue:]
    z = np.zeros((H, W), np.float32)
    ok = d > MIN_DISP
    zz = np.zeros_like(d)
    zz[ok] = fx * baseline / d[ok]
    z[v0:v1, u0:u1] = zz
    return z, time.perf_counter() - t0
