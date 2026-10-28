"""Record code, calibration, devices and last measured controller settings without motion."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import rig  # noqa: E402


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--controller-run', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit(f'refusing to overwrite {a.out}')
    status = git('status', '--porcelain=v1', '-z', '--untracked-files=all')
    diff = git('diff', '--binary', 'HEAD', '--')
    untracked_paths = [x.decode() for x in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if x]
    untracked = []
    for rel in untracked_paths:
        path = ROOT / rel
        if path.is_symlink():
            untracked.append({'path': rel, 'kind': 'symlink', 'target': str(path.readlink())})
        elif path.is_file():
            untracked.append({'path': rel, 'kind': 'file', 'size_bytes': path.stat().st_size, 'sha256': file_hash(path)})
    versions = {}
    for name in ('mujoco', 'numpy', 'torch', 'opencv-python', 'i2rt', 'warp-lang', 'mjfit'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    run = json.loads(a.controller_run.read_text())
    data = {'schema': 'usbc-baseline-v1', 'captured_utc': datetime.now(timezone.utc).isoformat(),
            'hostname': platform.node(), 'platform': platform.platform(), 'python': sys.version,
            'git': {'head': git('rev-parse', 'HEAD').decode().strip(),
                    'branch': git('branch', '--show-current').decode().strip(),
                    'status_sha256': sha(status), 'tracked_diff_sha256': sha(diff),
                    'untracked_files': untracked,
                    'untracked_listing_sha256': sha(json.dumps(untracked, sort_keys=True).encode())},
            'dependencies': versions,
            'calibration': {'path': str(rig.SPEC.relative_to(ROOT)), 'sha256': file_hash(rig.SPEC)},
            'devices': {'can_interfaces': rig.can_interfaces(), 'can_adapter_serials': rig.CAN_SERIAL,
                        'wrist_devices': rig.WRIST_CAM, 'zed_serial': rig.ZED_SERIAL},
            'world_frame': 'right_arm_base', 'joint_frame': 'raw encoder; model = encoder + offsets',
            'support_condition': 'free_pc',
            'controller': {'source_run': str(a.controller_run), 'settings': run['arm']['right']['gains'],
                           'status': 'last recorded settings; live controller not connected or checked'},
            'camera_mode': 'ZED rectified-left HD1080 and wrist 1280x1024 MJPG, last recorded capture',
            'limitations': ['capture is a point-in-time worktree fingerprint, not a frozen session',
                            'camera exposure time and policy consumption timestamps remain unknown',
                            'controller settings reflect the referenced run, not a live read']}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f: json.dump(data, f, indent=2); f.write('\n')
    print(a.out)


if __name__ == '__main__':
    main()
