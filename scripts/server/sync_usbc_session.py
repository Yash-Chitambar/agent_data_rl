"""Create, verify and resumably sync a frozen USB-C session without size filtering.

The source must not change after `manifest`. Existing destination data is never deleted.
Remote verification needs rsync, ssh and Python 3 on the destination host.
"""
import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

MANIFEST = 'usbc_session_manifest.json'
SUMS = 'usbc_session_sha256sums.txt'
COMPLETE = 'usbc_session_complete.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def entries(root):
    result = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'symlink not allowed in frozen bundle: {p}')
        if not p.is_file() or p.relative_to(root).as_posix() in (MANIFEST, SUMS, COMPLETE):
            continue
        rel = p.relative_to(root).as_posix()
        if '\n' in rel or '\r' in rel or '\\' in rel:
            raise ValueError(f'unsupported filename: {rel!r}')
        result.append({'path': rel, 'size_bytes': p.stat().st_size, 'sha256': digest(p)})
    return result


def create(root):
    root = root.resolve(strict=True)
    if not (root / 'session.json').is_file():
        raise ValueError('session.json is required')
    if any((root / x).exists() for x in (MANIFEST, SUMS, COMPLETE)):
        raise FileExistsError('manifest already exists; create a new frozen bundle version')
    files = entries(root)
    session = json.loads((root / 'session.json').read_text())
    # Older bimanual sessions identify themselves by schema but predate the
    # session_id field required by the first version of this artifact tool.
    source_session_id = session.get('session_id') or f"{session.get('schema', 'session')}:{root.name}"
    data = {'schema': 'usbc-session-artifact-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'source_session_id': source_session_id, 'files': files}
    with (root / MANIFEST).open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')
    with (root / SUMS).open('x') as f:
        for e in files:
            f.write(f"{e['sha256']}  {e['path']}\n")
        f.write(f'{digest(root / MANIFEST)}  {MANIFEST}\n')
    print(f'created {MANIFEST} and {SUMS} for {len(files)} files')


def stage(session, out, references, metadata):
    """Copy finalized collection inputs into a new bundle before making its manifest."""
    session = session.resolve(strict=True)
    out = out.resolve()
    if out.exists() or out == session or session in out.parents:
        raise ValueError('bundle destination must be new and outside the source session')
    if not (session / 'session.json').is_file():
        raise ValueError('session.json is required')
    if any((session / x).exists() for x in (MANIFEST, SUMS, COMPLETE)):
        raise ValueError('stage from the active collection rather than an already frozen bundle')
    all_sources = [session, *references, *metadata]
    for source in all_sources:
        source = source.resolve(strict=True)
        if source.is_dir():
            entries(source)  # refuses symlinks and unsupported filenames before copying
        elif not source.is_file():
            raise ValueError(f'unsupported source: {source}')
    ref_names = [p.name for p in references]
    file_names = [p.name for p in metadata]
    if len(set(ref_names)) != len(ref_names) or len(set(file_names)) != len(file_names):
        raise ValueError('reference and metadata basenames must be unique within each group')
    shutil.copytree(session, out)
    for source in references:
        shutil.copytree(source, out / 'reference' / source.name)
    for source in metadata:
        target = out / 'metadata' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(f'staged {out}; freeze with the manifest command once collection is complete')


def verify(root):
    root = root.resolve(strict=True)
    data = json.loads((root / MANIFEST).read_text())
    if data.get('schema') != 'usbc-session-artifact-v1':
        raise ValueError('unsupported manifest schema')
    actual = entries(root)
    if actual != data['files']:
        expected = {e['path']: e for e in data['files']}
        found = {e['path']: e for e in actual}
        bad = sorted(k for k in expected.keys() | found.keys() if expected.get(k) != found.get(k))
        raise ValueError(f'bundle differs from manifest: {bad[:20]}')
    subprocess.run(['sha256sum', '--check', '--status', SUMS], cwd=root, check=True)
    if (root / COMPLETE).exists():
        marker = json.loads((root / COMPLETE).read_text())
        if marker.get('schema') != 'usbc-session-complete-v1' or marker.get('manifest_sha256') != digest(root / MANIFEST):
            raise ValueError('invalid completion marker')
    print(f'verified {len(actual)} files in {root}')


def sync(source, dest, host):
    source = source.resolve(strict=True)
    verify(source)
    manifest_hash = digest(source / MANIFEST)
    if host:
        if not re.fullmatch(r'[A-Za-z0-9_@.:-]+', host) or host.startswith('-'):
            raise ValueError('invalid SSH host')
        destq = shlex.quote(str(dest))
        # An existing destination must identify the same immutable manifest to resume.
        check = (f'if [ -f {destq}/{COMPLETE} ]; then echo "destination already marked complete; verify separately" >&2; exit 3; '
                 f'elif [ -f {destq}/{MANIFEST} ]; then sha256sum -- {destq}/{MANIFEST}; '
                 f'elif [ -d {destq} ] && [ -n "$(ls -A -- {destq})" ]; then exit 2; '
                 f'else mkdir -p -- {destq}; fi')
        r = subprocess.run(['ssh', host, check], text=True, capture_output=True, check=True)
        if r.stdout and r.stdout.split()[0] != manifest_hash:
            raise ValueError('destination already has a different manifest')
        destination = f'{host}:{dest}'
    else:
        dest = Path(dest).resolve()
        if dest == source or source in dest.parents:
            raise ValueError('destination must be outside source')
        if dest.exists() and any(dest.iterdir()):
            if not (dest / MANIFEST).is_file() or digest(dest / MANIFEST) != manifest_hash:
                raise ValueError('destination is nonempty and lacks the same manifest')
            if (dest / COMPLETE).exists():
                verify(dest)
                print('destination already complete and verified')
                return
        dest.mkdir(parents=True, exist_ok=True)
        destination = str(dest) + '/'

    data = json.loads((source / MANIFEST).read_text())
    names = [e['path'] for e in data['files']] + [MANIFEST, SUMS]
    # Copy the small immutable manifest first, atomically, so an interrupted bulk copy can resume.
    subprocess.run(['rsync', '-a', '--checksum', '--protect-args', str(source / MANIFEST), destination], check=True)
    with tempfile.NamedTemporaryFile() as filelist:
        filelist.write(b'\0'.join(x.encode() for x in names) + b'\0'); filelist.flush()
        subprocess.run(['rsync', '-a', '--partial', '--checksum', '--no-whole-file', '--protect-args',
                        '--from0', f'--files-from={filelist.name}', str(source) + '/', destination], check=True)
    if host:
        remote_check = r'''
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(strict=True); expected_manifest=sys.argv[2]; expected_sums=sys.argv[3]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
manifest=root/'usbc_session_manifest.json'; sums=root/'usbc_session_sha256sums.txt'
if sha(manifest)!=expected_manifest or sha(sums)!=expected_sums: raise SystemExit('manifest or checksum list differs')
data=json.loads(manifest.read_text()); expected={e['path']:e for e in data['files']}; actual={}
for p in root.rglob('*'):
 if p.is_symlink(): raise SystemExit('symlink in destination: '+str(p))
 if p.is_file():
  rel=p.relative_to(root).as_posix()
  if rel not in ('usbc_session_manifest.json','usbc_session_sha256sums.txt','usbc_session_complete.json'):
   actual[rel]={'path':rel,'size_bytes':p.stat().st_size,'sha256':sha(p)}
if actual!=expected: raise SystemExit('destination contents differ from manifest')
marker=root/'usbc_session_complete.json'
if marker.exists():
 existing=json.loads(marker.read_text())
 if existing.get('schema')!='usbc-session-complete-v1' or existing.get('manifest_sha256')!=expected_manifest: raise SystemExit('invalid completion marker')
else:
 with marker.open('x') as f: json.dump({'schema':'usbc-session-complete-v1','manifest_sha256':expected_manifest},f)
print('verified',len(actual),'files')
'''
        command = 'python3 -c ' + shlex.quote(remote_check) + ' ' + shlex.quote(str(dest)) + ' ' + manifest_hash + ' ' + digest(source / SUMS)
        subprocess.run(['ssh', host, command], check=True)
        print(f'verified remote checksums at {host}:{dest}')
    else:
        verify(dest)
        with (dest / COMPLETE).open('x') as f:
            json.dump({'schema': 'usbc-session-complete-v1', 'manifest_sha256': manifest_hash}, f)
            f.write('\n')
        verify(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('manifest', 'verify'):
        s = sub.add_parser(name); s.add_argument('session', type=Path)
    s = sub.add_parser('stage'); s.add_argument('session', type=Path); s.add_argument('destination', type=Path)
    s.add_argument('--reference', type=Path, action='append', default=[], help='additional reference directory')
    s.add_argument('--metadata', type=Path, action='append', default=[], help='additional metadata file')
    s = sub.add_parser('sync'); s.add_argument('session', type=Path); s.add_argument('destination')
    s.add_argument('--host', help='SSH host; omit for local destination')
    a = parser.parse_args()
    if a.command == 'stage': stage(a.session, a.destination, a.reference, a.metadata)
    elif a.command == 'manifest': create(a.session)
    elif a.command == 'verify': verify(a.session)
    else: sync(a.session, a.destination, a.host)


if __name__ == '__main__':
    main()
