"""Replay October 25 native-feedback CSVs with their own recorded offsets and gains."""
import argparse,csv,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('csv',type=Path);p.add_argument('--out-dir',type=Path,required=True)
a=p.parse_args();root=Path(__file__).resolve().parents[2]
meta_path=a.csv.with_name(a.csv.stem+'_meta.json');meta=json.loads(meta_path.read_text())
side=meta['side'];stem=a.csv.stem+'_legacy_baseline'
with a.csv.open() as f:rows=list(csv.DictReader(f))
if not rows or any(not r.get('t_cmd') for r in rows):raise ValueError('missing actual send timestamps')
with tempfile.TemporaryDirectory(prefix='legacy-motion-replay-') as tmp:
 dest=Path(tmp)
 (dest/'run.json').write_text(json.dumps({'arm':{side:{'joint_offset_rad':meta['joint_offsets_rad'],'gains':meta['arm']}}}))
 with (dest/'steps.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['side','t_send','t_obs','q_cmd','q','qd'])
  for r in rows:
   w.writerow([side,r['t_cmd'],r['t'],json.dumps([float(r[f'q_cmd_{i}']) for i in range(1,7)]),json.dumps([float(r[f'q_{i}']) for i in range(1,7)]),json.dumps([0.]*6)])
 with (dest/f'encoder_{side}.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['host_arrival_s',*[f'q_{i}_rad' for i in range(1,7)]])
  for r in rows:w.writerow([r['t'],*[r[f'q_{i}'] for i in range(1,7)]])
 subprocess.run([sys.executable,str(root/'scripts/tools/replay_motion_cpu.py'),str(dest),'--side',side,'--joint',str(meta['args'].get('joint',3)),'--source','steps','--out-dir',str(a.out_dir.resolve()),'--output-stem',stem],check=True,cwd=root)
out=a.out_dir/(stem+'.json');report=json.loads(out.read_text())
report['run']=str(a.csv.resolve());report['normalized_input_sha256']=report.pop('input_sha256')
report['input_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.csv,meta_path)}
report['recorded_gains']=meta['arm'];report['recorded_offsets_rad']=meta['joint_offsets_rad']
report['limitations'].append('legacy CSV lacks velocity: initial simulated velocity assumed zero; no driver feedforward reconstructed')
out.write_text(json.dumps(report,indent=2)+'\n')
