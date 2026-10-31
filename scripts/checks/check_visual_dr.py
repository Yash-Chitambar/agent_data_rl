"""Visual domain randomization checks (envs/visual_dr.py) on the bimanual Warp runtime.

  ranges     sampled episode parameters and the per-world device fields stay inside VisualDRCfg
  per world  randomized worlds render different images and different materials/lights/cameras
  reset      reset(ids) resamples exactly the worlds ids
  nominal    a nominal world under DR renders the same frame as the DR-off runtime (seed 0 = real
             look), and the DR-off look still matches the real frames (measure_warp_look: fingertip
             IoU, edge distance, L1)
  finite     float render buffers finite, frames not empty or uniform
  sheet      contact sheet: REAL | SIM nominal | 6 randomized worlds, all at the recorded
             bimanual_stationary_001 joint angles (results/usbc/policy/visual_dr/contact_sheet.jpg)
  --bench    frames/s (3 cameras, 320 px) at 512 and 2048 worlds: legacy render (alpha-0 geoms drawn,
             raw colours), nominal look (DR off), DR on

    PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/checks/check_visual_dr.py [--bench] [--no-real]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'real2sim'))
import cv2
import numpy as np
import torch
import warp as wp

from blind_insertion.envs import visual_dr as VD
from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime

OUT = Path('results/usbc/policy/visual_dr')
RES = 320


def runtime(n, dr, **kw):
    return BimanualRuntime(BimanualCfg(nworld=n, render_rgb=True, camera_resolution=RES, measured_camera_cadence=False,
                                       visual_dr=dr, **kw))


def inside(x, lo, hi, tol=1e-4):
    return bool(((x >= lo - tol) & (x <= hi + tol)).all())


def check_ranges(e, report):
    c, v, p = e.visual_dr_cfg, e.visual, e.visual.params
    r = ~v.nominal
    assert r.sum() > 0.7 * e.cfg.nworld, 'too many nominal worlds'
    assert inside(p['light_elevation_deg'][r], *c.light_elevation_deg, tol=1e-2)
    assert inside(p['light_intensity'][r], *c.light_intensity)
    assert inside(p['light_kelvin'][r], *c.light_kelvin, tol=1e-1)
    sh = float(p['shadow'][r].mean())
    assert .3 < sh < .7, f'shadow fraction {sh}'
    for k, cam in enumerate(e.camera_names):
        pe, re = p['cam_pos_err_mm'][r, k], p['cam_rot_err_deg'][r, k]
        assert inside(pe, 0, c.cam_err_scale * c.cam_pos_mm[cam]) and inside(re, 0, c.cam_err_scale * c.cam_rot_deg[cam])
        # sim error at least the real one on average (deck: sim sensors >= real)
        assert float(pe.mean()) > .8 * c.cam_pos_mm[cam] and float(re.mean()) > .8 * c.cam_rot_deg[cam]
        cid = e.mjm.camera(cam).id
        dpos = (v.f['cam_pos'][r, cid] - v.f0['cam_pos'][cid]).norm(dim=-1) * 1e3
        assert torch.allclose(dpos, pe, atol=1e-2), 'cam_pos field does not match the sampled error'
        intr0, intr = v.f0['cam_intrinsic'][cid], v.f['cam_intrinsic'][r, cid]
        assert inside(intr[:, :2] / intr0[:2] - 1, -c.focal_jitter, c.focal_jitter)
        assert inside((intr[:, 2:] - intr0[2:]) / v.sensor[k], -c.principal_jitter, c.principal_jitter)
    for g, mat in v.group_mat.items():
        assert inside(v.f['mat_specular'][r, mat], *c.specular_range) and inside(v.f['mat_shininess'][r, mat], *c.shininess_range)
        assert inside(v.f['mat_rgba'][r, mat, :3], 0, 1)
        geoms = v.groups[g][~torch.isin(v.groups[g], torch.tensor([d[0] for d in v.decal.values()], device=e.device))]
        assert (v.f['geom_matid'][r][:, geoms] == mat).all(), f'{g} geoms not on their DR material'
    tex = v.f['mat_texid'][r][:, [v.decal[g][1] for g in v.decal], 1]
    valid = torch.cat([v.tex_proc, v.tex_real, torch.tensor([-1], device=e.device)])
    assert torch.isin(tex, valid).all()
    assert inside(v.img['gain'][r], *c.gain) and inside(v.img['exposure'][r], 1 - c.exposure, 1 + c.exposure)
    assert inside(v.img['brightness'][r], -c.brightness, c.brightness)
    assert inside(v.noise[r], *c.noise_std) and inside(v.blur[r], *c.blur_sigma_px) and inside(v.k1[r], -c.radial_k1, c.radial_k1)
    report['ranges'] = {'worlds': e.cfg.nworld, 'nominal_worlds': int(v.nominal.sum()), 'shadow_fraction': round(sh, 3),
                        'textured_fraction': p['textured'][r].mean(0).cpu().round(decimals=3).tolist(),
                        'real_texture_fraction': p['real_texture'][r].mean(0).cpu().round(decimals=3).tolist(),
                        'cam_pos_err_mm_mean': p['cam_pos_err_mm'][r].mean(0).cpu().round(decimals=2).tolist(),
                        'cam_rot_err_deg_mean': p['cam_rot_err_deg'][r].mean(0).cpu().round(decimals=3).tolist()}


def check_frames(e, frames, report):
    rgb = wp.to_torch(e.rgb_buffer)
    assert torch.isfinite(rgb).all(), 'non-finite render buffer'
    r = ~e.visual.nominal
    for name, f in frames.items():
        assert f.dtype == torch.uint8 and f.shape == (e.cfg.nworld, RES, RES, 3)
        std = f.float().flatten(1).std(1)
        assert (std > 5).all(), f'{name}: empty or uniform image'
        fr = f[r].flatten(1).float()
        d = torch.cdist(fr[:32] / 255, fr[:32] / 255) / fr.shape[1] ** .5
        off = d[~torch.eye(len(d), dtype=torch.bool, device=d.device)]
        assert (off > .01).all(), f'{name}: two randomized worlds rendered the same image'
    tm = e.visual.f['mat_rgba'][r][:, e.visual.group_mat['table'], :3]
    assert tm.std(0).min() > .05, 'table colour does not vary across worlds'
    report['per_world'] = {'min_pairwise_rms_diff': round(float(off.min()), 4)}


def check_reset(e, report):
    v = e.visual
    before = {k: x.clone() for k, x in v.params.items()}
    fields = {k: x.clone() for k, x in v.f.items()}
    ids = torch.arange(8, device=e.device)
    e.reset(ids)
    rest = torch.ones(e.cfg.nworld, dtype=torch.bool, device=e.device); rest[ids] = False
    for k in before:
        assert torch.equal(before[k][rest], v.params[k][rest]), f'reset changed other worlds ({k})'
    for k in fields:
        assert torch.equal(fields[k][rest], v.f[k][rest]), f'reset changed other worlds ({k})'
    changed = (before['light_kelvin'][ids] != v.params['light_kelvin'][ids]) | v.nominal[ids]
    assert changed.all(), 'reset did not resample'
    report['reset'] = 'reset(ids) resamples ids only'


def check_nominal(report):
    torch.manual_seed(0)
    a = runtime(4, None)
    a.reset()
    b = runtime(4, {'nominal_prob': 1., 'nominal_clean': True})
    b.q[:] = a.q
    assert b.visual.nominal.all()
    fa, fb = a.render(), b.render()
    stats = {}
    for name in fa:
        d = (fa[name].float() - fb[name].float()).abs()
        stats[name] = {'mean_abs': round(float(d.mean()), 4), 'p99': float(d.flatten().kthvalue(int(.99 * d.numel())).values)}
        assert d.mean() < .5 and stats[name]['p99'] <= 3, f'{name}: nominal world under DR differs from DR off {stats[name]}'
    # set_nominal on a randomized runtime gives the same frame too
    c = runtime(4, {'nominal_prob': 0.})
    c.q[:] = a.q
    c.visual.set_nominal(torch.arange(4, device=c.device))
    fc = c.render()
    for name in fa:
        assert (fa[name].float() - fc[name].float()).abs().mean() < .5
    report['nominal_vs_dr_off'] = stats
    del a, b, c
    torch.cuda.empty_cache()


def check_real(report):
    from measure_warp_look import measure
    s = measure({}, out=None, label='check')
    report['real_vs_sim_nominal'] = {k: round(v, 3) for k, v in s.items() if isinstance(v, float)}
    assert s['tip_iou'] > .75 and s['l1'] < 25 and s['edge_px'] < 4.5, s


def contact_sheet(report):
    from measure_warp_look import load_views
    from blind_insertion.scenes import physical_rig as R
    from blind_insertion.assets.real_camera import real_policy_image
    from render_sim_real_alignment import annotated_port
    from fit_gripper_tips import BUNDLE
    from blind_insertion.scenes import pc_bench as S
    views, fit = load_views()
    st = [v for v in views if v[0].startswith('bimanual_stationary_001')]
    n = 7
    e = runtime(n, {'seed': 3, 'nominal_prob': 0.})
    e.reset()
    e.visual.set_nominal([0])
    spec = R.load()
    profile = json.loads((BUNDLE / 'reference/right_arm_20261027/rig_profile_provisional_v0.json').read_text())
    port = annotated_port(spec, profile)
    import mujoco
    d = mujoco.MjData(e.mjm)
    S.place_pc(e.mjm, d, port_xy=port)
    q = st[0][4]
    m = e.mjm
    for k in range(n):
        for arm, prefix in (('right', ''), ('left', 'left_')):
            for j in range(6):
                e.q[k, int(m.joint(prefix + f'joint{j + 1}').qposadr[0])] = float(q[arm][j] + spec['arms'][arm]['joint_offsets_rad'][j])
            gm = fit['arms'][arm]['gripper_map']
            for f in ('left', 'right'):
                e.q[k, int(m.joint(prefix + f + '_finger').qposadr[0])] = float(np.clip(gm['slope'] * q[arm][6] + gm['intercept'], 0, .0475))
        e.q[k, e.pc_qadr:e.pc_qadr + 7] = torch.tensor(d.qpos[e.pc_qadr:e.pc_qadr + 7], dtype=torch.float32)
    frames = {k: v.cpu().numpy() for k, v in e.render().items()}
    real = {v[1]: v[3] for v in st}
    T = 240

    def tile(img, text):
        img = cv2.resize(img, (T, T), interpolation=cv2.INTER_AREA).copy()
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        return img
    cams = e.camera_names
    cols = [np.vstack([tile(real[c], f'REAL {c}') for c in cams])]
    cols.append(np.vstack([tile(frames[c][0], f'SIM seed 0 (nominal) {c}') for c in cams]))
    for k in range(1, n):
        p = e.visual.params
        txt = f"seed {k}: {p['light_kelvin'][k]:.0f}K el {p['light_elevation_deg'][k]:.0f}" + (' sh' if p['shadow'][k] else '')
        cols.append(np.vstack([tile(frames[c][k], txt if i == 0 else f'seed {k} {c}') for i, c in enumerate(cams)]))
    sheet = np.hstack(cols)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / 'contact_sheet.jpg'), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    report['contact_sheet'] = str(OUT / 'contact_sheet.jpg')
    del e
    torch.cuda.empty_cache()


def bench(report, sizes=(512, 2048)):
    out = {}
    legacy_hidden = VD.render_hidden
    for label, dr, kw in (('legacy', None, {'visual_look': None}), ('nominal', None, {}), ('dr', True, {})):
        if label == 'legacy':
            VD.render_hidden = lambda m: []
        for n in sizes:
            e = runtime(n, dr, **kw)
            e.reset(); e.render(); torch.cuda.synchronize()
            t, K = time.time(), 3
            for _ in range(K):
                e.render()
            torch.cuda.synchronize()
            dt = (time.time() - t) / K
            out[f'{label}/{n}'] = {'ms_per_render': round(dt * 1e3, 1), 'frames_per_s': round(3 * n / dt)}
            print(label, n, out[f'{label}/{n}'], flush=True)
            del e
            torch.cuda.empty_cache()
        VD.render_hidden = legacy_hidden
    report['throughput_3cam_320px'] = out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--no-real', action='store_true', help='skip the steps that need the recorded real frames')
    a = ap.parse_args()
    report = {}
    torch.manual_seed(0)
    e = runtime(64, VD.VisualDRCfg(seed=0))
    e.reset()
    frames = e.render()
    check_ranges(e, report)
    check_frames(e, frames, report)
    check_reset(e, report)
    f2 = e.render()                                            # per-frame camera noise: new frame differs
    assert not torch.equal(f2['top'], frames['top'])
    del e
    torch.cuda.empty_cache()
    check_nominal(report)
    if not a.no_real:
        check_real(report)
        contact_sheet(report)
    if a.bench:
        bench(report)
    OUT.mkdir(parents=True, exist_ok=True)
    name = 'checks_bench.json' if a.bench else 'checks.json'
    (OUT / name).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
    print('visual DR checks passed')


if __name__ == '__main__':
    main()
