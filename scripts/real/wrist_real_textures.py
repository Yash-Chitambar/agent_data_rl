"""PC panel and lid textures from real wrist frames, for the sim's PC (BimanualCfg.pc_real_look). No motion.

    PYTHONPATH=. python scripts/real/wrist_real_textures.py --fit results/real/wrist_nets/geometry_fit.json --out blind_insertion/assets/pc_real_look

Each texture pixel is a point on the PC (panel plane, lid plane) in the port frame. With the PC pose of the geometry fit, the
logged q_b, the sim wrist camera and the calibrated intrinsics, that point falls on a raw-frame pixel. The texture is the median
over frames of the sweep's TRAINING blocks only (wrist_net_real_eval.split_collect). The left hand shows in part of the lid: the
median removes its moving parallax and what remains is filled in. Writes panel.png (same UV layout as the rear photo: pc_rear.py),
top.png, top.json.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
import mujoco_warp as mjw  # noqa: E402
wp.config.quiet = True

import wrist_net_real_eval as W  # noqa: E402
from wrist_sim_at_real import SimAtReal  # noqa: E402
from blind_insertion.assets.real_camera import load_spec  # noqa: E402
from blind_insertion.scenes import pc_rear  # noqa: E402

SPEC = 'blind_insertion/configs/real_cameras_duchamp3.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--min-gap', type=float, default=8.)
    a = ap.parse_args()
    fit = json.load(open(a.fit))
    rs = W.rows()
    co = W.collect_rows(rs)
    tr, _, _ = W.split_collect(len(co))
    sel = [co[i] for i in tr if co[i]['p_mm'][2] >= a.min_gap]
    B = len(sel)
    print('frames', B)
    S = SimAtReal(B)
    dev = S.dev
    ev = [json.loads(x) for x in open(f"data/real_runs/2026-11-02/{co[0]['run']}/events.jsonl")]
    ql = [e for e in ev if e['kind'] == 'collect_press_removed'][0]['q_left']
    eR = torch.tensor(fit['port_R'], device=dev, dtype=torch.float32)
    port = torch.tensor(fit['port_p_m'], device=dev, dtype=torch.float32)
    q = torch.tensor([r['q_b'] for r in sel], device=dev)
    S.set_state(q, torch.tensor(ql, device=dev).expand(B, 6), port.expand(B, 3), eR.expand(B, 3, 3))
    with wp.ScopedDevice(S.rt.cfg.device):
        mjw.camlight(S.rt.m, S.rt.d)
    cid = S.rt.mjm.camera('wrist_cam').id
    cp = wp.to_torch(S.rt.d.cam_xpos)[:, cid].cpu().numpy().astype(float)
    cR = wp.to_torch(S.rt.d.cam_xmat)[:, cid].reshape(-1, 3, 3).cpu().numpy().astype(float) @ np.diag([1., -1., -1.])
    w = load_spec(SPEC)['wrist']['right']
    K = np.array([[w['intrinsics']['fx'], 0, w['intrinsics']['cx']], [0, w['intrinsics']['fy'], w['intrinsics']['cy']], [0, 0, 1.]])
    dist = np.asarray(w['distortion'], float)
    und = [cv2.undistort(cv2.imread(str(W.ROOT / r['raw'])), K, dist) for r in sel]
    Rp, pp = eR.cpu().numpy().astype(float), port.cpu().numpy().astype(float)

    def sample(P):
        """P (H, W, 3) points in the port frame -> median BGR over frames, with the visible-frame count."""
        out = []
        for i in range(B):
            X = (P @ Rp.T + pp - cp[i]) @ cR[i]          # camera frame
            u = K[0, 0] * X[..., 0] / X[..., 2] + K[0, 2]
            v = K[1, 1] * X[..., 1] / X[..., 2] + K[1, 2]
            bad = (u < 0) | (v < 0) | (u > 1279) | (v > 1023)
            im = cv2.remap(und[i], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
            im[bad] = np.nan
            out.append(im)
        out = np.stack(out)
        return np.nanmedian(out, 0), (~np.isnan(out[..., 0])).sum(0)

    prof = json.load(open('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json'))
    lay = pc_rear.layout(prof)
    wd, hd = lay['width'], lay['height']
    tx, ty = lay['centers']['usbc_2_target']
    depth = float(prof['pc']['measurements']['depth']['value'])
    # panel: UV layout of the rear photo
    src = cv2.imread(str(pc_rear.ASSET / 'source.png'))
    H, Wd = src.shape[:2]
    H_im2m = np.array(lay['image_to_metric'])
    uu, vv = np.meshgrid(np.arange(Wd), np.arange(H))
    m = np.stack([uu, vv, np.ones_like(uu)], -1).astype(float) @ H_im2m.T
    mx, my = m[..., 0] / m[..., 2], m[..., 1] / m[..., 2]
    inside = (mx >= 0) & (mx <= wd) & (my >= 0) & (my <= hd)
    P = np.stack([mx - tx, my - ty, np.full_like(mx, .0009)], -1)
    pan, cnt = sample(P)
    panel = np.where((inside & (cnt > 3))[..., None], np.nan_to_num(pan), 16.).astype(np.uint8)
    # lid: 1 px = 0.1 mm
    Hz, Wx = int(depth * 1e4), int(wd * 1e4)
    x0, x1, z0, z1, y = -tx, wd - tx, 0., -depth, hd - ty + .0004
    xx, zz = np.meshgrid(np.linspace(x0, x1, Wx), np.linspace(z0, z1, Hz))
    P = np.stack([xx, np.full_like(xx, y), zz], -1)
    top, cnt = sample(P)
    top = np.nan_to_num(top).astype(np.uint8)
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    yellow = (hsv[..., 0] > 15) & (hsv[..., 0] < 40) & (hsv[..., 1] > 90)
    far = np.zeros(top.shape[:2], bool)
    far[int(.55 * Hz):] = True                        # stickers sit near the panel edge; beyond them dark = the hand
    dark = far & (hsv[..., 2] < 90)
    hole = cv2.dilate(((yellow | dark | (cnt <= 3)) * 255).astype(np.uint8), np.ones((15, 15), np.uint8))
    top = cv2.inpaint(top, hole, 5, cv2.INPAINT_TELEA)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f'{a.out}/panel.png', panel)
    cv2.imwrite(f'{a.out}/top.png', top)
    json.dump(dict(x0=x0, x1=x1, y=y, z0=z0, z1=z1, frames=B, source='training blocks of collect_182100_collect1', fit=fit), open(f'{a.out}/top.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
