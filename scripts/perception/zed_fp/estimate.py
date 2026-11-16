"""FoundationPose (NVlabs, model-based) on one ZED left frame + stereo depth + first-frame mask.

Runs in the separate `fp` env (FoundationPose code at FP_ROOT, weights in FP_ROOT/weights:
refiner 2023-10-28-18-33-37, scorer 2024-01-11-20-02-45). Never imported by the rig venv.

Hypotheses:
  * 'upright' (default): the object stands on the table: R = Rz(yaw) in the base, yaw every `yaw_step` deg over 360
    deg (12 at 30 deg). Uses only the known table plane, no pose prior.
  * 'full': FoundationPose's own grid (icosphere views x in-plane rotations; no mycpp clustering).
Translation start: FoundationPose's guess (mask box centre, median mask depth).
Then `iterations` refiner passes on every hypothesis, the scorer ranks them. The near-square PC (103.5 x 103 mm) has a
4-fold near-symmetry: the panel colour is the cue. With a prior yaw, `resolve_yaw` reports the 90-deg multiple
nearest to the prior (and flags a flip); the raw FP pose is logged too.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import geometry as G

FP_ROOT = Path(os.environ.get('FP_ROOT', '/home/yashchitambar/fp_work/FoundationPose'))


class FPEstimator:
    def __init__(self, mesh, hyp='upright', yaw_step=30.0, iterations=5, T_cam_in_base=None, debug_dir=None, R_up=None):
        """R_up: rotation of the mesh frame in the base at yaw 0 (identity for the port-frame meshes of geometry.py)."""
        sys.path.insert(0, str(FP_ROOT))
        import torch
        import nvdiffrast.torch as dr
        import estimater as E   # FoundationPose
        logging.getLogger().setLevel(logging.WARNING)
        t0 = time.perf_counter()
        self.torch, self.E = torch, E
        self.mesh = mesh
        self.T_cam_in_base = T_cam_in_base if T_cam_in_base is not None else G.zed_camera()[1]
        self.hyp, self.yaw_step, self.iterations = hyp, yaw_step, iterations
        self.R_up = np.eye(3) if R_up is None else np.asarray(R_up, float)
        E.FoundationPose.make_rotation_grid = _rotation_grid_nocluster
        self.scorer = E.ScorePredictor()
        self.refiner = E.PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.fp = E.FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                   scorer=self.scorer, refiner=self.refiner, glctx=self.glctx,
                                   debug=0, debug_dir=str(debug_dir or '/tmp/fp_debug'))
        if hyp == 'upright':
            self.fp.rot_grid = self._upright_grid()
        self.load_s = time.perf_counter() - t0

    def _upright_grid(self, yaws=None):
        torch = self.torch
        Rcb = self.T_cam_in_base[:3, :3]
        out = []
        for yaw in (np.arange(0, 360, self.yaw_step) if yaws is None else yaws):
            T = np.eye(4)
            T[:3, :3] = Rcb.T @ G.pose_T(0, 0, yaw)[:3, :3] @ self.R_up     # object in cam (rotation only)
            out.append(T)
        # FoundationPose's grid is for the centred mesh; a rotation about the mesh centre is the same rotation.
        return torch.as_tensor(np.array(out), device='cuda', dtype=torch.float)

    def canon(self, T_obj_in_base):
        """The pose with the mesh's up-alignment removed: its x axis is the yaw direction, z is up when upright."""
        T = T_obj_in_base.copy(); T[:3, :3] = T[:3, :3] @ self.R_up.T
        return T

    def register(self, rgb, depth, mask, K, prior_yaw=None):
        """rgb HxWx3 uint8 RGB, depth HxW float m (0 = invalid), mask HxW 0/1. Returns dict (pose in cam and base).
        hyp 'prior': hypotheses at prior_yaw + {-30, -15, 0, 15, 30} deg (upright)."""
        torch = self.torch
        if self.hyp == 'prior':
            if prior_yaw is None:
                raise ValueError("hyp 'prior' needs prior_yaw")
            self.fp.rot_grid = self._upright_grid(prior_yaw + np.arange(-30, 31, 15))
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            T = self.fp.register(K=K.astype(np.float64), rgb=rgb, depth=depth.astype(np.float32), ob_mask=mask.astype(bool),
                                 iteration=self.iterations)
        finally:
            torch.set_default_tensor_type(torch.FloatTensor)     # FoundationPose sets CUDA tensors as the global default
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        scores = self.fp.scores.float().cpu().numpy()
        Tb = self.T_cam_in_base @ T
        x, y, yaw, z, tilt = G.T_to_xyyaw(self.canon(Tb))
        # second-best hypothesis that is not the same pose (yaw differs > 20 deg), for an ambiguity margin
        poses = self.fp.poses.float().cpu().numpy() @ self.fp.get_tf_to_centered_mesh().cpu().numpy()
        yaws = np.array([G.T_to_xyyaw(self.canon(self.T_cam_in_base @ p))[2] for p in poses])
        other = np.abs(G.wrap180(yaws - yaw)) > 20
        s2 = float(scores[other][0]) if other.any() else float('nan')
        return dict(T_obj_in_cam=T, T_obj_in_base=Tb, x=x, y=y, yaw_deg=yaw, z=z, tilt_deg=tilt,
                    score=float(scores[0]), score_2nd_other_yaw=s2, n_hyp=int(len(scores)), time_s=dt)


def resolve_yaw(T_obj_in_base, prior_yaw_deg, center_local, yaw_of=None):
    """Turn a pose by the 90-deg multiple about the base vertical axis through the mesh point `center_local` that brings
    its yaw nearest to the prior (the 4-fold near-symmetry of the case). Returns (T, k quarter turns applied)."""
    yaw_of = yaw_of or (lambda T: G.T_to_xyyaw(T)[2])
    k = int(round(G.wrap180(prior_yaw_deg - yaw_of(T_obj_in_base)) / 90.0)) % 4
    if k == 0:
        return T_obj_in_base, 0
    c = (T_obj_in_base @ np.append(np.asarray(center_local, float), 1.0))[:3]
    A = np.eye(4); A[:3, 3] = c
    B = np.eye(4); B[:3, 3] = -c
    return A @ G.pose_T(0, 0, 90.0 * k, z=0.0) @ B @ T_obj_in_base, k


def _rotation_grid_nocluster(self, min_n_views=40, inplane_step=60):
    """FoundationPose.make_rotation_grid without mycpp.cluster_poses (no symmetry given, so clustering only merges
    near-duplicates); the grid is the same icosphere x in-plane set."""
    import torch
    from Utils import sample_views_icosphere, euler_matrix
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    rot = []
    for c in cam_in_obs:
        for a in np.deg2rad(np.arange(0, 360, inplane_step)):
            rot.append(np.linalg.inv(c @ euler_matrix(0, 0, a)))
    self.rot_grid = torch.as_tensor(np.asarray(rot), device='cuda', dtype=torch.float)


def depth_residual(fp_est, T_obj_in_cam, depth, mask, K):
    """Median |rendered - observed| depth (mm) over mask pixels where both exist (nvdiffrast render of the mesh)."""
    import torch
    E = fp_est.E
    H, W = depth.shape
    A = np.eye(4); A[:3, 3] = fp_est.fp.model_center          # mesh_tensors are of the centred mesh
    ren = E.nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=torch.as_tensor((T_obj_in_cam @ A)[None], device='cuda', dtype=torch.float),
                              glctx=fp_est.glctx, mesh_tensors=fp_est.fp.mesh_tensors, output_size=(H, W), use_light=False)
    rd = ren[1][0].cpu().numpy()
    sil = rd > 0
    sel = sil & (mask > 0) & (depth > 0)
    res = float(np.median(np.abs(rd[sel] - depth[sel])) * 1e3) if sel.any() else float('nan')
    return res, sil.astype(np.uint8)
