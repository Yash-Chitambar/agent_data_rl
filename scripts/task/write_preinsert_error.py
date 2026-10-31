"""Collect the closed-loop approach evaluations into the pre-insert error deliverable for Part 4.

Reads results/usbc/policy/approach/<config>_s1.npz, _s2.npz (scripts/task/sweep_approach.sh) and
writes results/usbc/policy/approach/preinsert_error.json (+ preinsert_error_samples.npz: the per-episode
errors of the recommended configuration, for sampling RL start poses).

    PYTHONPATH=. .venv/bin/python scripts/task/write_preinsert_error.py --recommended servo_wrist_fine_fc
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

D = Path('results/usbc/policy/approach')
KEYS = ('x_mm', 'y_mm', 'gap_mm', 'rx_deg', 'ry_deg', 'roll_deg', 'lateral_mm', 'tilt_deg')


def load(cfg):
    fs = sorted(glob.glob(str(D / f'{cfg}_s[0-9].npz')))
    if not fs:
        return None, None
    ws = [np.load(f) for f in fs]
    w = {k: np.concatenate([x[k] for x in ws]) for k in KEYS + ('finite',)}
    args = json.loads(Path(fs[0][:-4] + '.json').read_text())['args']
    return w, args


def stats(w, stop):
    ok = w['finite'].astype(bool)
    out = {'n': int(len(ok)), 'n_finite': int(ok.sum())}
    pct = (50, 90, 99, 99.5, 100)
    for k in KEYS:
        x = w[k][ok]
        if k == 'gap_mm':
            x = x - stop
        out[k if k != 'gap_mm' else 'gap_error_mm'] = {'mean': float(x.mean()), 'std': float(x.std()),
                                                       **{f'abs_p{p}': float(np.percentile(np.abs(x), p)) for p in pct}}
    out['within_1mm_2deg'] = float(((w['lateral_mm'] <= 1) & (w['tilt_deg'] <= 2))[ok].mean())
    out['within_1mm_2deg_incl_nonfinite'] = float(((w['lateral_mm'] <= 1) & (w['tilt_deg'] <= 2) & ok).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recommended', required=True)
    a = ap.parse_args()
    configs = sorted({Path(f).name.rsplit('_s', 1)[0] for f in glob.glob(str(D / '*_s[0-9].npz'))})
    table = {}
    for c in configs:
        w, args = load(c)
        if w is not None:
            table[c] = {'args': {k: args[k] for k in ('pc', 'servo', 'stop', 'fric_comp', 'fine_below_mm', 'frames', 'final_iters', 'iters') if k in args},
                        **stats(w, args['stop'])}
    w, args = load(a.recommended)
    rec = table[a.recommended]
    out = {
        'what': 'Plug-tip pose relative to port 2 at the end of the approach (pre-insert), per episode, in sim with full DR. '
                'This is the start distribution Part 4 (insertion RL) must absorb.',
        'frame_and_units': 'port-2 frame: x across the panel (m->mm), y up, z out of the panel (gap); rx, ry = rotation of the '
                           'reader about port x / y (deg; tilt = |(rx, ry)|), roll about the port axis (deg). Signed mean/std and '
                           'percentiles of |error|. gap_error = gap - stop distance.',
        'how_measured': 'scripts/task/eval_approach.py (2 seeds x 512 worlds per configuration, GPU runtime): ZED RGB estimate of the '
                        'PC pose (learned CNN on DR renders) -> left press hold (guarded) -> right arm IK move to 30 mm in front of the '
                        'estimated port with the nominal in-hand pose (true joints carry the encoder-offset error, true reader the '
                        'randomized in-hand pose) -> visual servo on the right wrist RGB (learned plug-to-port estimator, coarse net, '
                        'fine net below 12 mm) with smooth moves and Coulomb feed-forward -> privileged measurement of the live '
                        'plug-tip pose in the live port frame. DR: PC layout +/-3 cm/+/-10 deg, PC feet mu U(0.3, 0.9), arm '
                        'Coulomb x U(0.5, 1.3), gravity jitter, encoder offsets +/-1.5 deg (joints 2-5), latency 0-1 step, in-hand '
                        '+/-3 mm / +/-8 deg (confirmed 60-deg grasp, provisional numbers), policy/v1 visual DR (appearance, lighting, '
                        'image noise, camera pose up to 2x real: ZED 8.6 mm / 1.2 deg, right wrist 10.2 mm / 2.4 deg, intrinsics).',
        'recommended_configuration': a.recommended, 'recommended': rec,
        'all_configurations': table,
        'target': 'lateral <= 1 mm and tilt <= 2 deg at 99.5 %',
        'samples_file': 'preinsert_error_samples.npz (recommended configuration, finite episodes: x, y, gap, rx, ry, roll)',
    }
    ok = w['finite'].astype(bool)
    np.savez_compressed(D / 'preinsert_error_samples.npz', **{k: w[k][ok] for k in ('x_mm', 'y_mm', 'gap_mm', 'rx_deg', 'ry_deg', 'roll_deg')})
    (D / 'preinsert_error.json').write_text(json.dumps(out, indent=1) + '\n')
    for c, v in table.items():
        print(f"{c:40s} n={v['n_finite']:5d} lat p50 {v['lateral_mm']['abs_p50']:.2f} p99 {v['lateral_mm']['abs_p99']:.2f} p99.5 {v['lateral_mm']['abs_p99.5']:.2f} | "
              f"tilt p50 {v['tilt_deg']['abs_p50']:.2f} p99 {v['tilt_deg']['abs_p99']:.2f} | gap p99 {v['gap_error_mm']['abs_p99']:.2f} | in 1mm/2deg {v['within_1mm_2deg']:.3f}")


if __name__ == '__main__':
    main()
