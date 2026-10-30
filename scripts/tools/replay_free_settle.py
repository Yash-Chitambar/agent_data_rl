"""Replay executed free-settle commands, including recorded trim, without refitting.

Normalizes archived JSON into the baseline replay's CSV schema in a temporary
directory. Source captures remain immutable. This does not enable integral control.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--recorded-state-feedforward', action='store_true')
    p.add_argument('--driver-xml',type=Path)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    trial = json.loads((a.run / 'trial.json').read_text())
    native = json.loads((a.run / 'left_native_feedback.json').read_text())
    calibration = root / 'blind_insertion/configs/real_cameras_duchamp3.json'
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if digest(calibration) != trial['calibration_sha256']:
        raise ValueError('capture calibration differs; resolve before replay')
    offsets = json.loads(calibration.read_text())['arms']['left']['joint_offsets_rad']
    rows = trial['rows']
    if not rows or any(r.get('left_first_can_send_host_s') is None for r in rows):
        raise ValueError('missing executed-command send timestamps')
    stem = a.run.name + ('_driver_reconstruction' if a.recorded_state_feedforward else '_baseline')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='free-settle-replay-') as tmp:
        dest = Path(tmp)
        (dest / 'run.json').write_text(json.dumps({'arm': {'left': {
            'joint_offset_rad': offsets, 'gains': trial['controllers']['left']}}}))
        with (dest / 'steps.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['side', 't_send', 't_obs', 'q_cmd', 'q', 'qd'])
            for r in rows:
                obs = r['observations']['left']
                w.writerow(['left', r['left_first_can_send_host_s'], obs['observation_time'],
                            json.dumps(r['executed_left_encoder'][:6]),
                            json.dumps(obs['q_encoder'][:6]), json.dumps(obs['qd_encoder'][:6])])
        with (dest / 'encoder_left.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['host_arrival_s', *[f'q_{i}_rad' for i in range(1, 7)]])
            for t, sequence, q in native:
                w.writerow([t, *q[:6]])
        cmd = [sys.executable, str(root / 'scripts/tools/replay_motion_cpu.py'), str(dest),
               '--side', 'left', '--source', 'steps', '--out-dir', str(a.out_dir.resolve()),
               '--output-stem', stem]
        if a.recorded_state_feedforward:
            cmd.append('--recorded-state-feedforward')
        if a.driver_xml:
            cmd.extend(['--driver-xml',str(a.driver_xml.resolve())])
        subprocess.run(cmd, check=True, cwd=root)
    report_path = a.out_dir / (stem + '.json')
    report = json.loads(report_path.read_text())
    report['run'] = str(a.run.resolve())
    report['normalized_input_sha256'] = report.pop('input_sha256')
    report['input_sha256'] = {n: digest(a.run / n) for n in ('trial.json', 'left_native_feedback.json')}
    report['calibration_sha256'] = digest(calibration)
    report['controller'] = 'recorded native PD gains; executed commands already include candidate trim; no additional integral'
    report['recorded_gains'] = trial['controllers']['left']
    data = np.genfromtxt(a.out_dir / (stem + '.csv'), delimiter=',', names=True)
    origin = rows[0]['observations']['left']['observation_time']
    sends = np.asarray([r['left_first_can_send_host_s'] - origin for r in rows])
    phase_idx = np.clip(np.searchsorted(sends, data[data.dtype.names[0]], side='right') - 1, 0, len(rows)-1)
    phases = np.asarray([r['phase'] for r in rows])[phase_idx]
    residual = np.column_stack([data[f'sim_q{i}_rad'] - data[f'real_q{i}_rad'] for i in range(1, 7)])
    report['phase_joint_rmse_rad'] = {phase: np.sqrt(np.mean(residual[phases == phase]**2, axis=0)).tolist()
                                     for phase in sorted(set(phases))}
    report['limitations'].append('phase boundaries use first CAN-send host time; native feedback is not synchronized exposure or torque')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['phase_joint_rmse_rad'], indent=2))


if __name__ == '__main__':
    main()
