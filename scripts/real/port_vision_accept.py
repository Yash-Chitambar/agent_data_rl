"""Key acceptance of the classical port vision: A and A2 stop frames, across runs, and the per-run label offset. No motion.

    python scripts/real/port_vision_accept.py     -> results/real/port_vision/{acceptance,label_offset}.json

Needs eval_frames_full.csv (scripts/real/port_vision_report.py). label_offset.json holds one scalar per run per axis:
true (vision) = manifest p_mm + offset. The manifest stays as it is.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'scripts/real'))
import importlib.util   # noqa: E402
spec = importlib.util.spec_from_file_location('pv_fit', ROOT / 'scripts/real/port_vision_fit.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

OUT = ROOT / 'results/real/port_vision'


def mouth_frames(fk, eR):
    """Frames of the collect run with the plug in the port mouth: mouth line, tip ray, shell width. Returns one record per window.
    The plug's top at the mouth lies on the plane z = 0; the plug is level, so the tip ray meets the same height: tip depth without any width.
    Windows: before the first seat and the second entry (later frames of the second entry lose the mouth line behind glare)."""
    PV, R0 = F.PV, ROOT / 'data/real_runs/2026-11-02/collect_182100_collect1'
    fr = [json.loads(x) for x in open(R0 / 'frames.jsonl')]
    w = [x for x in fr if x['cam'] == 'wrist_right' and x['kind'] == 'raw' and x['file'].endswith('.jpg')]
    a, _ = F.anchor_R()
    ta = a['t']
    T = np.load(R0 / 'ticks.npz')
    from deploy import conventions as C
    OFF = C.joint_offsets('right')
    s_, tip_a = np.array(a['tip_in_hand_s']), np.array(a['tip'])
    K = PV._camera('right')[0]
    f = 0.5 * (K[0, 0] + K[1, 1])
    ray = lambda u, v: np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])   # noqa: E731
    out = {}
    for name, (lo, hi) in {'first_entry': (-151.5, -132.2), 'second_entry': (646.0, 649.2)}.items():
        recs = []
        for x in w:
            if not lo <= x['t_capture'] - ta < hi:
                continue
            und, _ = PV.undistort(cv2.imread(str(R0 / x['file'])))
            g = cv2.GaussianBlur(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
            pose, _ = PV.panel_pose(und, K, box=(605, 490, 690, 560), max_reproj_px=6.0)
            if pose is None:
                continue
            p = g[:, 632:656].mean(1)
            gr = np.convolve(p, [1, 1, 0, -1, -1], 'same') / 2
            ve = 510 + int(np.argmin(gr[510:548]))
            q = g[:, 612:626].mean(1)
            vm = ve - 26 + int(np.argmin(q[ve - 26:ve - 6]))
            if not 508 <= vm <= 516:          # the notch: any other row is glare, not the mouth (checked on the overlays)
                continue
            side = g[ve - 9:ve - 3].mean(0)
            ul = PV._edge(side[600:640], 2, 38, 1)[0] + 600
            ur = PV._edge(side[645:690], 2, 43, -1)[0] + 645
            w_, u = ur - ul, (ul + ur) / 2
            R, t = pose['R'], pose['t']
            o = -R.T @ t
            dM = R.T @ ray(u, vm)
            M = o + (0 - o[2]) / dM[2] * dM
            vt = ve - PV.TIP_ABOVE_END_PX * PV.Z0_TIP / (f * PV.SHELL_W_EFF_MM / w_)
            dT = R.T @ ray(u, vt)
            lt = (M[1] - o[1]) / dT[1]
            Tp = o + lt * dT
            qq = T['q'][int(np.searchsorted(T['t'], x['t_capture'])) - 1, 0] + OFF
            pp, Rm = fk.tcp(qq)
            lab = eR.T @ (pp - Rm @ s_ - tip_a) * 1e3 + np.array(a['rel_mm'])
            tipc = np.array([(u - K[0, 2]) / K[0, 0] * lt, (vt - K[1, 2]) / K[1, 1] * lt, lt])
            recs.append([M[0], M[1], Tp[2], lt, w_, ve, vm, *lab, *tipc])
        A = np.array(recs)
        med = np.median(A, 0)
        out[name] = dict(n=len(A), plug_top_at_mouth_x_y_mm=med[:2].round(2).tolist(), tip_z_mm=round(float(med[2]), 2), tip_depth_mm=round(float(med[3]), 2), shell_w_px=round(float(med[4]), 2),
                         shell_end_v=float(med[5]), mouth_v=float(med[6]), label_mm=med[7:10].round(2).tolist(), width_eff_mm=round(float(med[4] * (med[3] - PV.TIP_DEEPER_MM) / f), 3), tip_cam=med[10:13].tolist())
    return out


def main():
    d = pd.read_csv(OUT / 'eval_frames_full.csv')
    man = {json.loads(x)['raw']: json.loads(x) for x in open(F.MAN) if json.loads(x).get('raw')}
    fk = F.Fk()
    mod = F.PV.load_model()
    o = np.array(mod['port2_base_m'])
    _, eR = F.anchor_R()
    res = {'runs': {}}
    stops = {}
    for run, g in d[d.source.isin(['same_spot', 'self', 'collect'])].groupby('run'):
        s = g[(g.ok == 1)]
        if run.endswith(('full_A', 'full_A2', 'ai4')):
            # the stop: the last settled frames before the insertion
            last = s.iloc[-1]
            s = s[(s.lz - last.lz).abs().lt(1.0) & (s.ly - last.ly).abs().lt(1.0) & (s.lx - last.lx).abs().lt(1.0)]
            stops[run] = s
        rows = []
        for _, r in s.iterrows():
            R = cv2.Rodrigues(np.array([r.r0, r.r1, r.r2]))[0]
            cv_ = -R.T @ np.array([r.t0, r.t1, r.t2])
            pc, Rc = fk.cam(man[r.raw]['q_b'])
            rows.append([*cv_, *(eR.T @ (pc - o) * 1e3), r.xf, r.yf, r.zf, r.lx, r.ly, r.lz, r.ve, r.w, r.Zw])
        A = np.array(rows)
        med = np.median(A, 0)
        res['runs'][run] = dict(
            n=len(A), cam_vision_mm=med[:3].round(2).tolist(), cam_fk_mm=med[3:6].round(2).tolist(), cam_vision_minus_fk_mm=(med[:3] - med[3:6]).round(2).tolist(),
            tip_vision_mm=med[6:9].round(2).tolist(), label_mm=med[9:12].round(2).tolist(), vision_minus_label_mm=np.median(A[:, 6:9] - A[:, 9:12], 0).round(2).tolist(),
            vision_minus_label_sd_mm=(A[:, 6:9] - A[:, 9:12]).std(0, ddof=1).round(2).tolist(), shell_end_v_px=round(float(med[12]), 2), shell_w_px=round(float(med[13]), 2), shell_depth_mm=round(float(med[14]), 2))
    a, b = res['runs']['full_174850_full_A'], res['runs']['full_175742_full_A2']
    d_tip = np.subtract(b['tip_vision_mm'], a['tip_vision_mm'])
    d_cam_v = np.subtract(b['cam_vision_mm'], a['cam_vision_mm'])
    d_cam_f = np.subtract(b['cam_fk_mm'], a['cam_fk_mm'])
    d_lab = np.subtract(b['label_mm'], a['label_mm'])
    res['A2_minus_A'] = dict(n=[a['n'], b['n']], label_fk_tip_mm=d_lab.round(2).tolist(), camera_fk_mm=d_cam_f.round(2).tolist(), camera_vision_mm=d_cam_v.round(2).tolist(), plug_tip_vision_mm=d_tip.round(2).tolist(),
                             plug_in_hand_mm=(d_tip - d_lab).round(2).tolist())
    # label offset of the collect run, three ways
    res['mouth'] = mouth_frames(fk, eR)
    R_nom = np.array(mod['R_view'], float)
    K = F.PV._camera('right')[0]
    f_ = 0.5 * (K[0, 0] + K[1, 1])
    c = d[(d.source == 'collect') & (d.ok == 1)]
    Zs = float(c.Zw.median())
    us, ves = float(c.u.median()), float(c.ve.median())
    vt = ves - F.PV.TIP_ABOVE_END_PX * F.PV.Z0_TIP / Zs
    Zt = Zs + F.PV.TIP_DEEPER_MM
    tip_sweep = np.array([(us - K[0, 2]) / K[0, 0] * Zt, (vt - K[1, 2]) / K[1, 1] * Zt, Zt])
    pc = np.array(F.PV.PORT_CENTER_MM)
    chain = {}
    for name, m in res['mouth'].items():
        # plug in the mouth: true y = 0 up to clearance, true z = tip_z. Label there + what the plug moved in the hand since then.
        dk = R_nom.T @ (tip_sweep - np.array(m['tip_cam']))
        true_there = np.array([m['plug_top_at_mouth_x_y_mm'][0] - pc[0], 0.0, m['tip_z_mm']])
        chain[name] = (true_there - np.array(m['label_mm']) + dk).round(2).tolist()
    run = res['runs']['collect_182100_collect1']
    res['collect_offset_estimates'] = dict(vision_against_sweep_labels=run['vision_minus_label_mm'], label_at_mouth_plus_in_hand_change=chain,
                                           recheck_anchor_would_mean=[0.38, 4.49, -0.92], first_anchor_would_mean=[0.0, 0.0, 0.0])
    offs = {r: dict(offset_mm=v['vision_minus_label_mm'], sd_mm=v['vision_minus_label_sd_mm'], n=v['n'], frames='stop frames' if r.endswith(('full_A', 'full_A2', 'ai4')) else 'all detected') for r, v in res['runs'].items()}
    (OUT / 'label_offset.json').write_text(json.dumps(dict(
        _about='true plug tip (vision) = manifest p_mm + offset_mm, one scalar per run per axis; x across, y up, z gap. The manifest is not changed. See docs/real/PORT_VISION.md for the evidence and limits.',
        _uncertainty_mm_1sigma_guess=dict(x=0.5, y=0.6, z=0.5), _same_spot_note='A and A2 borrow the collect anchor: their offset also holds the in-hand and PC differences of that run.', runs=offs), indent=1))
    print(json.dumps(res, indent=1))
    (OUT / 'acceptance.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
