"""Fit the nominal Warp look (`blind_insertion/configs/warp_look_duchamp3.json`) to the real frames.

The scene colours were matched to the real frames in the CPU (OpenGL) renderer; the Warp renderer
that policies see lights them differently (white wall saturated, arm black grey, gripper white).
Model per colour channel, region r (appearance group), camera c:

    real_rc = gain_c * sim_rc(colour_r) + offset_c     (gain 1 for the ZED: the reference)

The real wrist webcams differ from the ZED (right: darker; left: low contrast, lifted blacks), so
each camera gets an RGB gain and black level. Each round renders the 21 recorded views
(`measure_warp_look.WarpLook`) without the camera affine, takes the median real and sim colour per
(group, camera), fits (gain_c, offset_c) per wrist camera by least squares over the groups, then
moves each group colour by 0.8 of the mean log ratio of camera-corrected real to sim. Saturated sim
pixels push down; saturated real pixels (the white arm shells) are skipped. Colours above 1 become
emission. Writes the JSON and
before/after scores to results/usbc/policy/visual_dr/nominal/.

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/real2sim/fit_warp_look.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
import warp as wp

from measure_warp_look import OUT, WarpLook

FIT = ('wall', 'raceway', 'table', 'arm_dark', 'gripper', 'backdrop_black', 'mount', 'accents')
LOOK = Path('blind_insertion/configs/warp_look_duchamp3.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rounds', type=int, default=8)
    ap.add_argument('--out', type=Path, default=OUT)
    a = ap.parse_args()
    w = WarpLook({'visual_look': None})
    e, mjm = w.e, w.e.mjm
    cams = list(e.camera_names)
    mat_rgba, mat_emis, geom_rgba = (wp.to_torch(getattr(e.m, k)) for k in ('mat_rgba', 'mat_emission', 'geom_rgba'))
    groups = [g for g in FIT if g in w.groups]
    colour, emission = {}, {}
    for g in groups:
        ids = w.groups[g]
        cols = [mjm.mat_rgba[mjm.geom_matid[i], :3] if mjm.geom_matid[i] >= 0 else mjm.geom_rgba[i, :3] for i in ids]
        colour[g] = np.median(np.array(cols), 0).astype(float)
        mids = [mjm.geom_matid[i] for i in ids if mjm.geom_matid[i] >= 0]
        emission[g] = float(mjm.mat_emission[mids].max()) if mids else 0.
    gain = {c: np.ones(3) for c in cams}

    def write():
        """Group colours/emission into the device model (same rule as visual_dr.apply_look)."""
        for g in groups:
            for i in w.groups[g]:
                mid = mjm.geom_matid[i]
                if mid >= 0:
                    mat_rgba[:, mid, :3] = torch.tensor(colour[g], dtype=torch.float32)
                    mat_emis[:, mid] = emission[g]
                else:
                    geom_rgba[:, i, :3] = torch.tensor(colour[g], dtype=torch.float32)

    history = []
    offset = {c: np.zeros(3) for c in cams}
    for c in cams:                                          # fit on the raw render; affine applied below
        e.visual.cam_gain[c] = torch.ones(3, device=e.device)
        e.visual.cam_offset[c] = torch.zeros(3, device=e.device)
    s0 = None
    for rnd in range(a.rounds + 1):
        write()
        s, rows, tiles = w.score(w.render(), 'fit')
        if s0 is None:
            s0 = (s, rows, tiles)
        obs = []                                            # (group, camera, real (3,), raw sim (3,), weight)
        for key, v in s['regions'].items():
            g, c = key.split('/')
            if g in groups:
                obs.append((g, c, np.array(v['real_rgb']), np.array(v['sim_rgb']), np.sqrt(v['views'])))
        pred = [np.abs(R - np.clip(gain[c] * S + offset[c], 0, 255)).mean() for g, c, R, S, _ in obs]
        history.append({'round': rnd, 'fitted_region_abs_delta_mean': round(float(np.mean(pred)), 2)})
        print(history[-1], flush=True)
        if rnd == a.rounds:
            break
        # A: per-camera affine (gain, black level) per channel; the ZED is the reference (1, 0)
        for c in cams:
            oc = [o for o in obs if o[1] == c and o[2].max() < 250 and o[3].max() < 252]
            if len(oc) < 3:
                continue
            Wt = np.array([o[4] for o in oc])
            for ch in range(3):
                S, y = np.array([o[3][ch] for o in oc]), np.array([o[2][ch] for o in oc])
                if c == cams[0]:                           # reference camera: black level only
                    gk, bk = 1., float(np.sum(Wt * (y - S)) / Wt.sum())
                else:
                    X = np.stack([S, np.ones(len(oc))], 1)
                    gk, bk = np.linalg.lstsq(X * Wt[:, None], y * Wt, rcond=None)[0]
                gain[c][ch], offset[c][ch] = np.clip(gk, .5, 2.), np.clip(bk, -20., 60.)
        # B: colours toward the camera-corrected real colour, averaged over cameras (log space)
        step = {g: [] for g in groups}
        for g, c, R, S, wt in obs:
            if R.max() >= 250:
                continue
            T = np.maximum((R - offset[c]) / gain[c], 1.)
            rho = np.log(T + 1) - np.log(S + 1)
            sat = S >= 252
            rho[sat] = np.minimum(rho[sat], 0) - .3
            step[g].append((rho, wt))
        for g, st in step.items():
            if st:
                rho = sum(r * wt for r, wt in st) / sum(wt for _, wt in st)
                colour[g] = colour[g] * np.exp(.8 * rho)
        for g in groups:
            k = colour[g].max()
            if k > 1:                                      # brighter than a lit material: add emission
                emission[g] = float((1 + emission[g]) * k - 1)
                colour[g] = colour[g] / k
            colour[g] = np.clip(colour[g], .003, 1)
    for c in cams:
        e.visual.cam_gain[c] = torch.tensor(gain[c], device=e.device, dtype=torch.float32)
        e.visual.cam_offset[c] = torch.tensor(offset[c], device=e.device, dtype=torch.float32) / 255
    look = {'_note': 'Nominal Warp look fitted by scripts/real2sim/fit_warp_look.py to the 21 recorded views of '
                     'fit_backdrop.py (colour per appearance group, emission, per-camera RGB gain; gain_top = 1).',
            'colors': {g: [round(float(x), 4) for x in colour[g]] for g in groups},
            'emission': {g: round(emission[g], 4) for g in groups if emission[g] > 0},
            'camera_gain': {c: [round(float(x), 4) for x in gain[c]] for c in cams},
            'camera_offset': {c: [round(float(x), 2) for x in offset[c]] for c in cams},
            'fit_history': history}
    LOOK.write_text(json.dumps(look, indent=1) + '\n')
    s1, rows1, tiles1 = w.score(w.render(), 'after_fit')
    w.save(a.out, 'raw_colours', *s0)
    w.save(a.out, 'fitted', s1, rows1, tiles1)
    print(json.dumps(look, indent=1))


if __name__ == '__main__':
    main()
