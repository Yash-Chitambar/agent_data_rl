"""Replay camera availability against virtual control ticks; no policy or robot is run."""
import argparse
import csv
import json
import math
from pathlib import Path

CAMERAS = ('zed', 'wrist_right', 'wrist_left')


def read_camera(root, camera):
    folder = root / camera
    with (folder / 'frames.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    frames = []
    seen = set()
    for row in rows:
        if row['error'] or not row['host_receive_ns'] or not row['policy_ready_ns']:
            raise ValueError(f'{camera}: incomplete acquisition or preprocessing row')
        ident, received, ready = int(row['local_frame_id']), int(row['host_receive_ns']), int(row['policy_ready_ns'])
        if ident in seen or ready < received:
            raise ValueError(f'{camera}: duplicate frame ID or preprocessing before arrival')
        if frames and (received < frames[-1]['received_ns'] or ready < frames[-1]['ready_ns']):
            raise ValueError(f'{camera}: frame timestamps are not ordered')
        seen.add(ident)
        rel = row['policy_png']
        if rel and (Path(rel).name != rel or not (folder / rel).is_file()):
            raise ValueError(f'{camera}: saved policy crop is missing or path is invalid')
        frames.append({'frame_id': ident, 'received_ns': received, 'ready_ns': ready,
                       'policy_png': f'{camera}/{rel}' if rel else None,
                       'exact_repeat': bool(int(row['exact_repeat_of_previous']))})
    if not frames:
        raise ValueError(f'{camera}: no frames')
    return frames


def select(frames, cutoff_ns):
    """Newest arrived frame whose preprocessing finished by the observation cutoff."""
    result = None
    for frame in frames:
        if frame['ready_ns'] > cutoff_ns:
            break
        if frame['received_ns'] <= cutoff_ns:
            result = frame
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--hz', type=float, default=50.0)
    p.add_argument('--max-age-ms', type=float, required=True, help='diagnostic threshold; not assumed validated')
    p.add_argument('--require-images', action='store_true', help='fail if any selected crop was not archived')
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    if not math.isfinite(a.hz) or a.hz <= 0 or not math.isfinite(a.max_age_ms) or a.max_age_ms <= 0:
        raise SystemExit('hz and max-age-ms must be positive and finite')
    capture = json.loads((a.capture / 'capture.json').read_text())
    if capture.get('startup_failed') or any('error' in capture.get('results', {}).get(c, {}) for c in CAMERAS):
        raise SystemExit('capture has a camera failure; cannot replay as complete')
    frames = {camera: read_camera(a.capture, camera) for camera in CAMERAS}
    start = max(frames[c][0]['ready_ns'] for c in CAMERAS)
    end = min(frames[c][-1]['ready_ns'] for c in CAMERAS)
    if end < start:
        raise SystemExit('camera timing windows do not overlap')
    step = round(1e9 / a.hz)
    count = (end - start) // step + 1
    selections = []
    for tick in range(count):
        cutoff = start + tick * step
        selected, reasons = {}, []
        for camera in CAMERAS:
            frame = select(frames[camera], cutoff)
            if frame is None:
                selected[camera] = None; reasons.append(f'{camera}:no_frame'); continue
            age_ns = cutoff - frame['received_ns']
            selected[camera] = {**frame, 'arrival_age_ns': age_ns,
                                'policy_crop_saved': frame['policy_png'] is not None}
            if age_ns > a.max_age_ms * 1e6:
                reasons.append(f'{camera}:stale')
            if a.require_images and not frame['policy_png']:
                reasons.append(f'{camera}:crop_not_saved')
        selections.append({'tick_id': tick, 'observation_cutoff_ns': cutoff,
                           'clock_domain': 'host_monotonic_ns', 'selected': selected,
                           'decision': 'hold' if reasons else 'ready', 'reasons': reasons,
                           'policy_consumption_claimed': False})
    if a.require_images and any(any(x.endswith(':crop_not_saved') for x in r['reasons']) for r in selections):
        raise SystemExit('selected policy crops are missing; no replay output written')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        for row in selections:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    print(json.dumps({'ticks': len(selections), 'holds': sum(r['decision']=='hold' for r in selections),
                      'max_age_ms_diagnostic': a.max_age_ms, 'policy_consumption_claimed': False,
                      'output': str(a.out)}, indent=2))


if __name__ == '__main__':
    main()
