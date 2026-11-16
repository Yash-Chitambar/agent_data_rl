"""ZED stereo capture + FoundationPose PC pose in the capture stage (config block `zed_fp`, OFF by default).

docs/real/FOUNDATIONPOSE_20261116.md. FoundationPose (NVlabs) and SAM 3 (facebook/sam3) run in a separate env (`fp`),
never in the rig venv: scripts/perception/zed_fp/ (pipeline, server, batch). This module only
  1. grabs a rectified HD1080 stereo pair from the ZED (left + right image of ONE grab) at the PC capture, or renders
     one in the dry run (CPU MuJoCo, calibrated ZED pose and intrinsics, right camera = left + baseline along x);
  2. saves it losslessly with a request.json (prior = the gate net, the edge fit, the arms) under
     <run>/zed_fp/<tag>/ ;
  3. mode 'shadow': with server 'none' (default) nothing else happens: the server (mason) runs FP on the saved pairs
     after the session (scripts/perception/zed_fp/batch.py). With server 'wait', the chain waits up to timeout_s for
     result.json from a running scripts/perception/zed_fp/server.py and logs it. The chain's pose is never changed.
     mode 'use': waits for result.json; the chain takes FP's x, y (yaw per yaw_source) only when the FP gate passes.
     A failed gate or a timeout fails the capture like a net reject (no silent fall-back).
The ZED right image: deploy/cameras.ZedTop retrieves the LEFT view only. When the block is on, this module wraps the
instance's _grab (not the class, not the file) so that the capture thread also retrieves the RIGHT view of the same
grab while `want` is set (about 2 ms per frame, only during the capture). With the block off nothing is patched.
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = dict(
    enabled=False,
    mode='shadow',            # 'shadow' = save the stereo pair (+ log FP if a server answers), chain unchanged; 'use' = gated FP x, y
    server='none',            # 'none' = save only (FP runs on mason after the session); 'wait' = wait for a running server.py
    timeout_s=30.0,           # wait for result.json (server 'wait' or mode 'use'); the arms hold at the capture pose meanwhile
    pair_timeout_s=1.5,       # wait for a stereo pair grabbed after the request
    n_pairs=1,                # stereo pairs saved per capture (left.png/right.png, then left_1.png/right_1.png, ...): the
                              # server can take the median (FP per-frame sd 0.4-1.2 mm on real frames, n = 18)
    yaw_source='gate',        # 'gate' = the chain's yaw before FP (today), 'fp' = FP's yaw
    save_reader=False,        # also save the stereo pair(s) of the reader frame (left arm parked; zed_fp/zed_reader_<k>/,
                              # request.json object 'reader'); save only, never used by the chain
    prior='gate',             # FP prior (SAM 3 detection choice, 4-fold symmetry): 'gate' = the gate net pose
    geom_sim='sim',           # dry run: the sim PC mesh (server --geom sim)
    # FP gate (mode 'use'), from real zed_gt frames (n = 18) and sim (n = 40), FOUNDATIONPOSE_20261116.md:
    min_score=80.0,           # FP scorer logit of the best hypothesis (real 109.9-124.0; sim mesh scale differs: min_score_sim)
    min_score_sim=30.0,
    min_iou=0.85,             # FP silhouette vs SAM 3 mask
    min_sam_score=0.3,
    max_dz_raw_mm=25.0,       # raw FP height vs the known top before the plane fix (real +6.5..+17.6 mm: stereo on the lid)
    max_d_edge_mm=4.0,        # |FP - edge fit| when the edge fit ran and passed (real max 3.4 mm)
    max_dyaw_edge_deg=2.0,
    max_shift_mm=8.0,         # |FP - gate net| (the approach start shift against the net, as zed_edge_refine)
    max_dyaw_gate_deg=4.0,
)
_RIGHT_LOCK = threading.Lock()


def parse(cfg):
    b = cfg.get('zed_fp') or {}
    if not b.get('enabled', False):
        return None
    unknown = set(k for k in b if not k.startswith('_')) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'zed_fp: unknown keys {sorted(unknown)}')
    c = dict(DEFAULTS, **{k: v for k, v in b.items() if not k.startswith('_')})
    if c['mode'] not in ('shadow', 'use'):
        raise ValueError("zed_fp.mode must be 'shadow' or 'use'")
    if c['server'] not in ('none', 'wait'):
        raise ValueError("zed_fp.server must be 'none' or 'wait'")
    if c['yaw_source'] not in ('gate', 'fp'):
        raise ValueError("zed_fp.yaw_source must be 'gate' or 'fp'")
    if c['mode'] == 'use' and c['server'] != 'wait':
        raise ValueError("zed_fp mode 'use' needs server 'wait' (a running scripts/perception/zed_fp/server.py)")
    return c


# ---------------------------------------------------------------- stereo pair, rig
def _install_right_hook(cam):
    """Wrap one ZedTop instance's _grab: while cam.fp_want is set, also retrieve the RIGHT view of the same grab and
    store (left, right, count) in cam.fp_pair. Idempotent."""
    if getattr(cam, 'fp_hooked', False):
        return
    sl = cam.sl
    cam.fp_mat_r, cam.fp_want, cam.fp_pair, cam.fp_n = sl.Mat(), False, None, 0
    orig = cam._grab

    def _grab():
        f = orig()
        if f is not None and cam.fp_want:
            cam.cam.retrieve_image(cam.fp_mat_r, sl.VIEW.RIGHT)
            r = cam.fp_mat_r.get_data()[:, :, :3].copy()
            with _RIGHT_LOCK:
                cam.fp_n += 1
                cam.fp_pair = (f, r, time.monotonic(), cam.fp_n)
        return f

    cam._grab = _grab
    cam.fp_hooked = True


def grab_stereo_real(io, timeout_s=1.5):
    """(left BGR, right BGR, t) of one ZED grab made after this call, or (None, None, reason)."""
    cam = getattr(io, 'cams', {}).get('top')
    if cam is None or not hasattr(cam, 'sl'):
        return None, None, 'no ZedTop camera in io.cams'
    _install_right_hook(cam)
    with _RIGHT_LOCK:
        n0 = cam.fp_n
    cam.fp_want = True
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < timeout_s:
            with _RIGHT_LOCK:
                p = cam.fp_pair
            if p is not None and p[3] > n0 + 1:       # skip a grab that may have started before the request
                return p[0], p[1], p[2]
            time.sleep(0.01)
    finally:
        cam.fp_want = False
    return None, None, f'no stereo pair within {timeout_s} s'


# ---------------------------------------------------------------- stereo pair, dry run
_SIM = {}


def grab_stereo_sim(io):
    """Dry run: HD1080 stereo pair of the current sim state (world 0) from the calibrated ZED, CPU MuJoCo (EGL)."""
    import importlib.util
    import mujoco
    if 'mod' not in _SIM:
        spec = importlib.util.spec_from_file_location('fp_sim_render', ROOT / 'scripts/perception/fp_sim_render.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        from blind_insertion.scenes import physical_rig as R
        m = copy.deepcopy(io.rt.mjm)
        m.vis.global_.offwidth, m.vis.global_.offheight = 1920, 1080
        _SIM.update(mod=mod, m=m, d=mujoco.MjData(m), spec=R.load(), r=mujoco.Renderer(m, height=1080, width=1920))
    mod, m, d, r = _SIM['mod'], _SIM['m'], _SIM['d'], _SIM['r']
    q = io.rt.q[0].detach().cpu().numpy()
    d.qpos[:len(q)] = q[:m.nq]
    mujoco.mj_forward(m, d)
    out = []
    for right in (False, True):
        mod.set_camera(m, _SIM['spec'], right=right)
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera='top')
        out.append(np.ascontiguousarray(r.render()[:, :, ::-1]))
    return out[0], out[1], time.monotonic()


# ---------------------------------------------------------------- request / result
def save_request(run_dir, tag, left, right, req, extra=()):
    import cv2
    d = Path(run_dir) / 'zed_fp' / tag
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / 'left.png'), left)
    cv2.imwrite(str(d / 'right.png'), right)
    for i, (le, ri) in enumerate(extra, 1):
        cv2.imwrite(str(d / f'left_{i}.png'), le)
        cv2.imwrite(str(d / f'right_{i}.png'), ri)
    (d / 'request.json').write_text(json.dumps(req, indent=1, default=float))    # last: the server keys on it
    return d


def wait_result(d, timeout_s):
    p = Path(d) / 'result.json'
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError:
                pass
        time.sleep(0.05)
    return None


def gate(res, c, net_xy, net_yaw, edge=None, sim=False):
    """FP gate -> (ok, why list, distances)."""
    why, dist = [], {}
    if res is None:
        return False, ['no FP result (timeout)'], dist
    if not res.get('ok'):
        return False, [f"FP failed: {res.get('reason')}"], dist
    x, y, yaw = res['pose'][:3]
    smin = c['min_score_sim'] if sim else c['min_score']
    if not res['score'] >= smin:
        why.append(f"score {res['score']:.1f} < {smin}")
    if not res['iou_fp_sam'] >= c['min_iou']:
        why.append(f"IoU FP/SAM {res['iou_fp_sam']:.3f} < {c['min_iou']}")
    ss = (res.get('sam') or {}).get('score')
    if ss is not None and not ss >= c['min_sam_score']:
        why.append(f'SAM 3 score {ss:.2f} < {c["min_sam_score"]}')
    if res.get('dz_mm') is not None and not abs(res['dz_mm']) <= c['max_dz_raw_mm']:
        why.append(f"raw FP height {res['dz_mm']:+.1f} mm beyond {c['max_dz_raw_mm']} mm")
    d_net = (np.array([x, y]) - np.asarray(net_xy, float)) * 1e3
    dyaw_net = (yaw - net_yaw + 180.) % 360. - 180.
    dist.update(d_gate_mm=d_net.tolist(), shift_mm=float(np.hypot(*d_net)), d_gate_yaw_deg=float(dyaw_net))
    if not np.hypot(*d_net) <= c['max_shift_mm']:
        why.append(f'|FP - gate net| {np.hypot(*d_net):.1f} > {c["max_shift_mm"]} mm')
    if not abs(dyaw_net) <= c['max_dyaw_gate_deg']:
        why.append(f'|yaw FP - gate net| {abs(dyaw_net):.2f} > {c["max_dyaw_gate_deg"]} deg')
    if edge is not None and edge.get('ok') and 'fit_xy' in edge:
        d_e = (np.array([x, y]) - np.asarray(edge['fit_xy'], float)) * 1e3
        dyaw_e = (yaw - edge['fit_yaw_deg'] + 180.) % 360. - 180.
        dist.update(d_edge_mm=d_e.tolist(), d_edge_yaw_deg=float(dyaw_e))
        if not np.hypot(*d_e) <= c['max_d_edge_mm']:
            why.append(f'|FP - edge fit| {np.hypot(*d_e):.1f} > {c["max_d_edge_mm"]} mm')
        if not abs(dyaw_e) <= c['max_dyaw_edge_deg']:
            why.append(f'|yaw FP - edge fit| {abs(dyaw_e):.2f} > {c["max_dyaw_edge_deg"]} deg')
    return not why, why, dist


def summary_line(frec):
    if frec.get('result') is None:
        return f"zed_fp: stereo pair saved ({frec.get('dir')}), {frec.get('status')}"
    r = frec['result']
    if not r.get('ok'):
        return f"zed_fp: FP FAILED ({r.get('reason')}) -> {'ok' if frec.get('ok') else 'REJECTED'}"
    x, y, yaw = r['pose'][:3]
    s = (f"zed_fp: port ({x * 1e3:.1f}, {y * 1e3:.1f}) mm, yaw {yaw:.1f} deg, score {r['score']:.1f}, "
         f"IoU {r['iou_fp_sam']:.3f}, raw dz {r.get('dz_mm', float('nan')):+.1f} mm, {r.get('time_s', float('nan')):.1f} s")
    return s + (' -> ok' if frec.get('ok') else f" -> REJECTED ({'; '.join(frec.get('why', []))})")


def save_reader(chain):
    """Reader frame (chain.reader_frame): save the stereo pair(s) for an offline reader fit. Logs a 'zed_fp_reader'
    event; never changes the chain."""
    c = chain.zfp
    is_sim = getattr(chain.io, 'is_sim', False)
    t0 = time.monotonic()
    pairs = []
    for _ in range(1 if is_sim else int(c['n_pairs'])):
        le, ri, t = grab_stereo_sim(chain.io) if is_sim else grab_stereo_real(chain.io, c['pair_timeout_s'])
        if le is None:
            break
        pairs.append((le, ri))
    run_dir = getattr(getattr(chain, 'logger', None), 'dir', None)
    ev = dict(n_pairs=len(pairs))
    if pairs and run_dir is not None:
        tag = chain.last_frame_tag.replace('@', '_')
        req = dict(tag=tag, frame_tag=chain.last_frame_tag, object='reader', geom=c['geom_sim'] if is_sim else 'real',
                   prior=None, n_pairs=len(pairs), arms_b=[x[0].tolist() for x in chain.arms_believed()[:2]])
        d = save_request(run_dir, tag, pairs[0][0], pairs[0][1], req, pairs[1:])
        ev['dir'] = str(d.relative_to(run_dir))
    ev['seconds'] = time.monotonic() - t0
    chain.log_event('zed_fp_reader', **ev)
    print(f"zed_fp: reader frame stereo pairs saved: {ev.get('n_pairs')} ({ev.get('dir')}), {ev['seconds']:.1f} s", flush=True)


def apply(chain, xy, yaw, ok, ep, eR, rec, tr):
    """Capture stage hook (chain.capture_pc, after the edge fit). Returns (xy, yaw, ok, ep, eR); unchanged in shadow."""
    import torch
    c = chain.zfp
    is_sim = getattr(chain.io, 'is_sim', False)
    t0 = time.monotonic()
    net = rec.get('net') or {k: rec[k] for k in ('port_xy', 'yaw_deg', 'ok')}
    edge = rec.get('edge_fit')
    frec = dict(mode=c['mode'], server=c['server'])
    if is_sim:
        left, right, tinfo = grab_stereo_sim(chain.io)
        frec['frame_source'] = 'sim_cpu_render_hd1080'
    else:
        left, right, tinfo = grab_stereo_real(chain.io, c['pair_timeout_s'])
        frec['frame_source'] = 'zed_hd1080_rectified_pair'
    extra = []
    for _ in range(int(c['n_pairs']) - 1):
        if left is None or is_sim:
            break
        le, ri, _t = grab_stereo_real(chain.io, c['pair_timeout_s'])
        if le is not None:
            extra.append((le, ri))
    run_dir = getattr(getattr(chain, 'logger', None), 'dir', None)
    if left is None or run_dir is None:
        frec.update(status=f'no stereo pair ({tinfo})' if left is None else 'no run logger', result=None, ok=False,
                    why=['no stereo pair' if left is None else 'no run dir'])
    else:
        tag = f"{chain.last_frame_tag.replace('@', '_')}"
        req = dict(tag=tag, frame_tag=chain.last_frame_tag, geom=c['geom_sim'] if is_sim else 'real',
                   prior=[float(net['port_xy'][0]), float(net['port_xy'][1]), float(net['yaw_deg'])], prior_source='gate net',
                   net=net, edge_fit=None if edge is None else {k: edge.get(k) for k in ('ok', 'fit_xy', 'fit_yaw_deg', 'edges_used')},
                   arms_b=rec.get('arms_b'), t_grab_monotonic=float(tinfo) if not isinstance(tinfo, str) else None,
                   true_port_xy=rec.get('true_port_xy'), true_yaw_deg=rec.get('true_yaw_deg'), seed=getattr(chain, 'seed', None),
                   n_pairs=1 + len(extra))
        d = save_request(run_dir, tag, left, right, req, extra)
        frec.update(dir=str(d.relative_to(run_dir)), save_s=time.monotonic() - t0)
        res = wait_result(d, c['timeout_s']) if (c['server'] == 'wait') else None
        frec['result'] = None if res is None else {k: res.get(k) for k in (
            'ok', 'reason', 'pose', 'pose_raw', 'quarter_turns', 'score', 'iou_fp_sam', 'dz_mm', 'time_s', 'sam')}
        frec['status'] = 'deferred (server none: run batch.py after the session)' if c['server'] == 'none' else \
            ('result' if res is not None else f"timeout {c['timeout_s']} s")
        if res is not None or c['mode'] == 'use':
            gok, why, dist = gate(res, c, net['port_xy'], net['yaw_deg'], edge, sim=is_sim)
            frec.update(ok=gok, why=why, **dist)
        if tr is not None and res is not None and res.get('ok'):
            frec['err_mm'] = ((np.asarray(res['pose'][:2]) - tr['pc_p'][:2]) * 1e3).tolist()
    frec['seconds'] = time.monotonic() - t0
    print(summary_line(frec), flush=True)
    rec.update(zed_fp=frec)
    if c['mode'] != 'use':
        return xy, yaw, ok, ep, eR
    gok = bool(frec.get('ok'))
    ok_all = bool(ok[0]) and gok
    if gok:
        r = frec['result']
        fx, fy, fyaw = r['pose'][:3]
        new_yaw = math.radians(fyaw) if c['yaw_source'] == 'fp' else float(yaw[0])
        xy = torch.tensor([[float(fx), float(fy)]], dtype=xy.dtype, device=chain.dev)
        yaw = torch.tensor([new_yaw], dtype=yaw.dtype, device=chain.dev)
        ep, eR = chain.scene.port_frame(xy.cpu(), yaw.cpu())
        ep, eR = ep.to(chain.dev), eR.to(chain.dev)
        rec.update(port_xy=xy[0].tolist(), yaw_deg=math.degrees(float(yaw[0])),
                   est_source=f"FoundationPose x, y + {c['yaw_source']} yaw")
    ok = torch.tensor([ok_all], device=chain.dev)
    rec['ok'] = ok_all
    if tr is not None:
        rec['err_xy_mm'] = float(np.linalg.norm(np.asarray(rec['port_xy']) - tr['pc_p'][:2]) * 1e3)
    return xy, yaw, ok, ep, eR
