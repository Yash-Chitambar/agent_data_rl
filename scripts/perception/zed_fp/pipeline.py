"""ZED stereo pair -> PC pose: SAM 3 mask (first frame) -> SGBM depth -> FoundationPose register -> checks.

Runs in the `fp` env (docs/real/FOUNDATIONPOSE_20261116.md). One record per frame (JSON-safe dict):
  pose_raw / pose: port 2 (x m, y m), yaw deg, z m, tilt deg in the right base (pose = after the 90-deg resolution
  against the prior yaw, when a prior is given), fp score, iou_fp_sam (FP silhouette vs SAM 3 mask),
  depth_res_mm (median |render - stereo| on the mask), dz_mm (z - known top height), timings, GPU memory.

    from scripts.perception.zed_fp.pipeline import Pipeline
    P = Pipeline(mesh='real'); rec = P.run(left_bgr, right_bgr, prior=(x, y, yaw_deg))
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from . import geometry as G
from . import stereo as ST

TABLE_ROI = (560, 300, 1360, 1000)        # HD1080 left: the table area around the PC and the reader at the capture pose


def roi_around(box, W=1920, H=1080, half=320):
    cu, cv_ = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    u0, v0 = int(max(0, min(W - 2 * half, cu - half))), int(max(0, min(H - 2 * half, cv_ - half)))
    return (u0, v0, u0 + 2 * half, v0 + 2 * half)


class Pipeline:
    def __init__(self, mesh='real', hyp='upright', yaw_step=30.0, iterations=5, R_up=None, center_local=None,
                 sam_threshold=0.05, spec=None, z_top=None, plane_fix=True):
        """plane_fix: the PC stands on the table, so its mesh origin height is known (z_ref). The pose is moved along the
        camera ray through the mesh origin to that height and levelled (yaw kept). Reason: on real frames the SGBM depth of
        the silver lid reads 10-14 mm high (panel face depth agrees with the edge fit; FOUNDATIONPOSE_20261116.md)."""
        self.plane_fix = plane_fix
        import torch
        from .estimate import FPEstimator
        from .segment import Sam3Segmenter
        self.torch = torch
        self.K, self.T_cam, self.baseline = G.zed_camera(spec)
        if isinstance(mesh, str) and mesh == 'real':
            self.mesh = G.pc_real_mesh()
            g = G.PC_REAL
            center_local = (g['depth'] / 2, g['port_from_left'] - g['width'] / 2, 0.0)
            self.z_ref = g['z_top']                          # port frame origin = top face, z 11.5 mm in the base
        else:
            import trimesh
            self.mesh = trimesh.load(mesh, process=False) if isinstance(mesh, str) else mesh
            self.z_ref = z_top                               # z of the mesh origin in the base, if known
        self.center_local = center_local if center_local is not None else self.mesh.bounds.mean(0)
        t0 = time.perf_counter()
        self.seg = Sam3Segmenter(threshold=sam_threshold)
        self.fp = FPEstimator(self.mesh, hyp=hyp, yaw_step=yaw_step, iterations=iterations, T_cam_in_base=self.T_cam,
                              R_up=R_up)
        self.load_s = time.perf_counter() - t0

    def prior_T(self, prior):
        x, y, yaw = prior
        T = G.pose_T(x, y, yaw, z=self.z_ref if self.z_ref is not None else G.PC_REAL['z_top'])
        T[:3, :3] = T[:3, :3] @ self.fp.R_up
        return T

    def run(self, left_bgr, right_bgr=None, prior=None, depth=None, mask=None, sam_mode='text', keep=False):
        """left/right: rectified HD1080 BGR. prior: (x m, y m, yaw deg) of port 2 (net / edge fit) or None.
        depth: optional depth (m) instead of stereo (sim truth check). mask: optional mask instead of SAM 3."""
        torch = self.torch
        torch.cuda.reset_peak_memory_stats()
        t_all = time.perf_counter()
        rec = dict(prior=None if prior is None else [float(v) for v in prior])
        rgb = np.ascontiguousarray(left_bgr[:, :, ::-1])
        pb = None if prior is None else G.prior_box(self.mesh, self.prior_T(prior), self.K, self.T_cam)
        roi = TABLE_ROI if pb is None else roi_around(pb)
        if mask is None:
            mask, info = self.seg.segment(rgb, obj='pc', mode=sam_mode, roi=roi, prior_box=pb)
            if mask is not None:
                mask = G.largest_cc(mask)
            rec['sam'] = info
        else:
            rec['sam'] = dict(mode='given')
        if mask is None or mask.sum() < 200:
            rec.update(ok=False, reason='no_mask', time_s=time.perf_counter() - t_all)
            return rec
        ys, xs = np.nonzero(mask)
        if depth is None:
            box = (xs.min() - 80, ys.min() - 80, xs.max() + 80, ys.max() + 80)
            depth, rec['stereo_s'] = ST.sgbm_depth(left_bgr, right_bgr, self.K[0, 0], self.baseline, roi=box)
        sel = (mask > 0)
        rec['depth_valid_frac'] = float((depth[sel] > 0).mean())
        r = self.fp.register(rgb, depth, mask, self.K, prior_yaw=None if prior is None else float(prior[2]))
        from .estimate import depth_residual, resolve_yaw
        T = r['T_obj_in_base']
        k = 0
        if prior is not None:
            T, k = resolve_yaw(T, prior[2], self.center_local, yaw_of=lambda X: G.T_to_xyyaw(self.fp.canon(X))[2])
        x, y, yaw, z_raw, tilt = G.T_to_xyyaw(self.fp.canon(T))
        T_unfixed = T
        if self.plane_fix and self.z_ref is not None:
            # slide the case centre (not the port corner: FP leaves a 1-4 deg tilt) along its camera ray to the known
            # height, then level the case about that point with FP's yaw
            Tc_ = self.fp.canon(T)
            c_off = self.fp.R_up @ np.asarray(self.center_local, float)        # centre offset in the canonical frame
            q = Tc_[:3, :3] @ c_off + Tc_[:3, 3]
            cam = self.T_cam[:3, 3]
            d = q - cam
            q2 = cam + d * (self.z_ref + c_off[2] - cam[2]) / d[2]
            Rz = G.pose_T(0, 0, yaw)[:3, :3]
            T = np.eye(4); T[:3, :3] = Rz @ self.fp.R_up; T[:3, 3] = q2 - Rz @ c_off
        x, y, yaw, z, _ = G.T_to_xyyaw(self.fp.canon(T))
        T_cam = np.linalg.inv(self.T_cam) @ T
        dres, sil = depth_residual(self.fp, T_cam, depth, mask, self.K)
        rec.update(ok=True, pose_raw=[r['x'], r['y'], r['yaw_deg'], r['z'], r['tilt_deg']],
                   pose=[x, y, yaw, z, tilt], quarter_turns=int(k), score=r['score'], plane_fix=bool(self.plane_fix),
                   pose_unfixed=list(G.T_to_xyyaw(self.fp.canon(T_unfixed))),
                   score_2nd_other_yaw=r['score_2nd_other_yaw'], n_hyp=r['n_hyp'], fp_s=r['time_s'],
                   iou_fp_sam=G.iou(sil, mask), depth_res_mm=dres,
                   dz_mm=None if self.z_ref is None else (z_raw - self.z_ref) * 1e3,
                   T_obj_in_base=T.tolist(), mask_area_px=int(mask.sum()),
                   gpu_peak_gb=torch.cuda.max_memory_allocated() / 1e9, time_s=time.perf_counter() - t_all)
        if keep:
            rec['_mask'], rec['_depth'], rec['_sil'] = mask, depth, sil
        return rec


def overlay(left_bgr, rec, mesh, K, T_cam_in_base, crop_half=170):
    """Debug image: SAM mask (red), FP silhouette outline (green), prior box (blue)."""
    im = left_bgr.copy()
    if '_mask' in rec:
        m = rec['_mask'] > 0
        im[m] = (0.6 * im[m] + [0, 0, 100]).astype(np.uint8)
    if rec.get('ok'):
        T = np.linalg.inv(T_cam_in_base) @ np.array(rec['T_obj_in_base'])
        sil = G.mesh_mask(mesh, T, K, im.shape[:2])
        cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cs, -1, (0, 255, 0), 1)
        port = G.project(np.array(rec['T_obj_in_base'])[:3, 3], K, T_cam_in_base)[0]
        cv2.drawMarker(im, tuple(int(v) for v in port), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
        ys, xs = np.nonzero(sil)
        c = (int(xs.mean()), int(ys.mean())) if len(xs) else (960, 600)
    else:
        c = (960, 600)
    u0, v0 = max(c[0] - crop_half, 0), max(c[1] - crop_half, 0)
    return im[v0:v0 + 2 * crop_half, u0:u0 + 2 * crop_half]
