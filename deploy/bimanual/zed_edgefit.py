"""PC pose from the ZED case-edge fit in the capture stage (config block `zed_edge_refine`, OFF by default).

docs/real/ZED_EDGEFIT_20261115.md. The fit is scripts/perception/fit_pc_edges.py (docs/real/ZED_MAP_20261114.md):
the PC case top-face outline in the raw HD1080 rectified-left ZED frame, robust least squares on (x, y, yaw).

Chain use (deploy/bimanual/chain.RealChain.capture_pc):
  1. The gate net (chain_args.zed_gate, today zed_v1_gate_arm) runs as today. Its gate (presence, OOD, arm
     envelope) stays a hard requirement. Its yaw is the yaw the chain uses (yaw_source 'gate', see the doc).
  2. The start net (block key start_gate, zed_v5_gate) gives the start pose of the fit. zed_v5 follows PC moves
     within 1.2 mm (n = 5) with a near constant offset (+4..+6 mm y); v1 saturates near y 395 mm (16 mm error at one
     real pose). Its own gate result is logged, not required (the gate net's gate is).
  3. The fit refines x, y (and yaw, logged). It is used only if its quality gate passes:
       * panel, right and back edges each >= min_inliers of n_points inliers (|residual| < 3 px);
       * the left edge is used when it has >= min_inliers inliers, else the fit runs again without it ('noL',
         the left gripper hides it at the capture pose);
       * inlier rms <= max_rms_px;
       * |fit - start| <= max_dxy_start_mm and |yaw fit - yaw start| <= max_dyaw_start_deg;
       * |fit - gate net| <= max_shift_mm (the shift of the approach start against today's chain).
     A failed fit gate makes the capture fail like a net gate reject (the run stops at the gate). No silent
     fall-back to the net.
Mode 'shadow' (the capture-only config): the fit runs and is logged with its gate result, the chain keeps the gate
net's pose and gate exactly as today.
On the rig the fit reads the raw frame (BGR, HD1080). In the sim dry run there is no raw frame: the 320 px policy
render is scaled back into its HD1080 crop (3.27x upsampled; frame_source 'sim_policy_upsampled'), and the fit uses
the sim PC geometry (geom_sim).
"""
from __future__ import annotations

import importlib.util
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = dict(
    enabled=False,
    mode='use',                   # 'use' = the chain takes the fit (gated); 'shadow' = fit and log only, the chain uses the net as today
    start_gate='zed_v5_gate',     # estimator key of the start net (netcache.zed); '' = the gate net's own pose
    geom='real',                  # fit_pc_edges geometry on the rig (physical PC: 103.5 x 103 mm, physical port 2)
    geom_sim='sim',               # geometry in the dry run (the sim PC: 102.5 x 97 mm, sim port label)
    yaw_source='gate',            # 'gate' = the gate net's yaw (today), 'start' = the start net's, 'fit' = the fit's
    n_points=40,
    min_inliers=30,
    max_rms_px=1.2,
    max_dxy_start_mm=10.0,
    max_dyaw_start_deg=4.0,
    max_shift_mm=8.0,
    # DRY RUN ONLY (ignored on the rig): 'render' = fit the upsampled sim render (the sim PC panel face is light grey,
    # the real one dark: the panel edge is not found and the gate stops the run, docs/real/ZED_EDGEFIT_20261115.md);
    # 'truth_model' = the fit result is replaced by the sim truth + N(0, sd) noise (the real fit-vs-arm spread), so a
    # dry run can exercise the downstream chain with a fit-quality pose. The distance gates still apply; the image
    # quality gates are skipped in this mode (no image fit). The render fit is still run and logged.
    dry_run_fit='render',
    dry_run_fit_sd_mm=0.5,
    dry_run_fit_sd_deg=0.5,
)
REQUIRED_EDGES = ('panel', 'right', 'back')
_F = None


def fitmod():
    """scripts/perception/fit_pc_edges.py (loaded by path: scripts/ is not a package)."""
    global _F
    if _F is None:
        spec = importlib.util.spec_from_file_location('fit_pc_edges', ROOT / 'scripts/perception/fit_pc_edges.py')
        _F = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_F)
    return _F


def parse(cfg):
    """The config block with defaults, or None when absent or enabled is false."""
    b = cfg.get('zed_edge_refine') or {}
    if not b.get('enabled', False):
        return None
    unknown = set(k for k in b if not k.startswith('_')) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'zed_edge_refine: unknown keys {sorted(unknown)}')
    c = dict(DEFAULTS, **{k: v for k, v in b.items() if not k.startswith('_')})
    if c['mode'] not in ('use', 'shadow'):
        raise ValueError(f"zed_edge_refine.mode {c['mode']!r}: 'use' or 'shadow'")
    if c['dry_run_fit'] not in ('render', 'truth_model'):
        raise ValueError(f"zed_edge_refine.dry_run_fit {c['dry_run_fit']!r}: 'render' or 'truth_model'")
    if c['yaw_source'] not in ('gate', 'start', 'fit'):
        raise ValueError(f"zed_edge_refine.yaw_source {c['yaw_source']!r}: 'gate', 'start' or 'fit'")
    for g in (c['geom'], c['geom_sim']):
        if g not in fitmod().GEOMS:
            raise ValueError(f'zed_edge_refine geometry {g!r} not in {sorted(fitmod().GEOMS)}')
    return c


def canvas_from_policy(pol_rgb):
    """Sim dry run: the 320 px top policy image (RGB) back into its HD1080 crop (BGR, zeros outside). Inverse of
    scenes/physical_rig.real_top_policy_image."""
    import cv2
    from blind_insertion.scenes.physical_rig import load
    K = load()['top']['intrinsics_rectified']['HD1080']
    w, h = K['resolution']
    cx, cy = round(K['cx']), round(K['cy'])
    half = min(cx, w - cx, cy, h - cy)
    out = np.zeros((h, w, 3), np.uint8)
    up = cv2.resize(np.ascontiguousarray(pol_rgb), (2 * half, 2 * half), interpolation=cv2.INTER_CUBIC)
    out[cy - half:cy + half, cx - half:cx + half] = cv2.cvtColor(up, cv2.COLOR_RGB2BGR)
    return out


def _edges(r):
    return {k: dict(inl=v['inl'], n=v['n'], rms_px=None if v['rms_px'] is None else round(v['rms_px'], 3))
            for k, v in r['edges'].items()}


def run_fit(img_bgr, start, c, geom):
    """Fit from start (x m, y m, yaw deg): all four edges, or 'noL' when the left edge has too few inliers."""
    F = fitmod()
    g = F.GEOMS[geom]
    r = F.fit(img_bgr, start, geom=g, n=c['n_points'])
    used = 'all'
    if r['edges']['left']['inl'] < c['min_inliers']:
        r = F.fit(img_bgr, start, geom=g, n=c['n_points'], use=REQUIRED_EDGES)
        used = 'noL'
    return r, used


def refine(img_bgr, frame_source, start_xy, start_yaw_deg, gate_xy, gate_yaw_deg, c, geom, model_pose=None):
    """-> (xy (2,) m, yaw deg, ok, rec). xy and yaw are the values the chain uses when ok (yaw per yaw_source).
    start_*: the start net; gate_*: the gate net (today's estimate). model_pose (x m, y m, yaw deg): DRY RUN ONLY
    (dry_run_fit 'truth_model'): replaces the fit result; the image fit still runs and is logged as render_fit."""
    t0 = time.monotonic()
    why = []
    rec = dict(frame_source=frame_source, geom=geom, start_gate=c['start_gate'], yaw_source=c['yaw_source'],
               start_xy=[float(v) for v in start_xy], start_yaw_deg=float(start_yaw_deg))
    if img_bgr is None or img_bgr.shape[:2] != (1080, 1920):
        rec.update(ok=False, why=['no raw HD1080 frame'], seconds=time.monotonic() - t0)
        return np.asarray(gate_xy, float), float(gate_yaw_deg), False, rec
    r, used = run_fit(img_bgr, (float(start_xy[0]), float(start_xy[1]), float(start_yaw_deg)), c, geom)
    fx, fy, fyaw = r['pose']
    if model_pose is not None:
        rec['render_fit'] = dict(fit_xy=[float(fx), float(fy)], fit_yaw_deg=float(fyaw), edges_used=used,
                                 inlier_rms_px=r['inlier_rms_px'], edges=_edges(r))
        rec['frame_source'] = 'sim_truth_model (dry run only)'
        fx, fy, fyaw = (float(v) for v in model_pose)
    fit_xy = np.array([fx, fy])
    d_start = (fit_xy - np.asarray(start_xy, float)) * 1e3
    d_gate = (fit_xy - np.asarray(gate_xy, float)) * 1e3
    dyaw_start = (fyaw - float(start_yaw_deg) + 180.) % 360. - 180.
    rms = r['inlier_rms_px']
    if not all(np.isfinite([fx, fy, fyaw])):
        why.append('fit not finite')
    if model_pose is None:
        for e in REQUIRED_EDGES:
            if r['edges'][e]['inl'] < c['min_inliers']:
                why.append(f"{e} edge {r['edges'][e]['inl']} < {c['min_inliers']} inliers")
        if rms is None or not rms <= c['max_rms_px']:
            why.append(f'inlier rms {rms} > {c["max_rms_px"]} px')
    if not np.hypot(*d_start) <= c['max_dxy_start_mm']:
        why.append(f'|fit - start| {np.hypot(*d_start):.1f} > {c["max_dxy_start_mm"]} mm')
    if not abs(dyaw_start) <= c['max_dyaw_start_deg']:
        why.append(f'|yaw fit - start| {abs(dyaw_start):.2f} > {c["max_dyaw_start_deg"]} deg')
    if not np.hypot(*d_gate) <= c['max_shift_mm']:
        why.append(f'|fit - gate net| {np.hypot(*d_gate):.1f} > {c["max_shift_mm"]} mm')
    ok = not why
    yaw = {'gate': float(gate_yaw_deg), 'start': float(start_yaw_deg), 'fit': float(fyaw)}[c['yaw_source']]
    rec.update(ok=ok, why=why, edges_used=used, fit_xy=[float(fx), float(fy)], fit_yaw_deg=float(fyaw),
               sd_stat=[float(r['sd_stat'][0]) * 1e3, float(r['sd_stat'][1]) * 1e3, float(r['sd_stat'][2])],
               inlier_rms_px=None if rms is None else float(rms), edges=_edges(r),
               d_start_mm=d_start.tolist(), d_start_yaw_deg=float(dyaw_start), d_gate_mm=d_gate.tolist(),
               d_gate_yaw_deg=float((fyaw - float(gate_yaw_deg) + 180.) % 360. - 180.),
               shift_mm=float(np.hypot(*d_gate)), seconds=time.monotonic() - t0)
    return fit_xy, yaw, ok, rec


def summary_line(rec):
    if 'fit_xy' not in rec:
        return f"edge fit: FAILED ({'; '.join(rec['why'])})"
    s = (f"edge fit ({rec['edges_used']}): port ({rec['fit_xy'][0] * 1e3:.1f}, {rec['fit_xy'][1] * 1e3:.1f}) mm, "
         f"yaw {rec['fit_yaw_deg']:.1f} deg, rms {rec['inlier_rms_px']:.2f} px, shift from the gate net "
         f"({rec['d_gate_mm'][0]:+.1f}, {rec['d_gate_mm'][1]:+.1f}) mm, from the start ({rec['d_start_mm'][0]:+.1f}, "
         f"{rec['d_start_mm'][1]:+.1f}) mm, {rec['seconds']:.2f} s")
    return s + (' -> ok' if rec['ok'] else f" -> REJECTED ({'; '.join(rec['why'])})")
