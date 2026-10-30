"""Measure host-side frame cadence and preprocessing delay, never exposure latency."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--trial',default='bimanual_stationary_001');p.add_argument('--out',type=Path,required=True);p.add_argument('--export-profile',type=Path);a=p.parse_args()
base=a.bundle/'cameras'/a.trial;meta=json.loads((base/'capture.json').read_text());report={'trial':a.trial,'clock':'host monotonic','startup_exclusion_frames':5,'cameras':{},'limitations':['host arrival is not exposure time','camera clocks are not synchronized to host','runtime capture load may differ from policy deployment','no measured policy-consumption timestamps in these captures']}
timelines={}
def stats(values):
 return {'count':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),'p05':float(np.percentile(values,5)),'p95':float(np.percentile(values,95)),'max':float(np.max(values))}
for name in ('zed','wrist_left','wrist_right'):
 source=base/name/'frames.csv'
 with source.open() as f:rows=list(csv.DictReader(f))[5:]
 timelines[name]=rows
 received=np.array([int(r['host_receive_ns']) for r in rows],dtype=np.int64)
 delta=np.diff(received)/1e9
 if not len(delta) or (delta<=0).any():raise ValueError('frame receive times not strictly ordered')
 delay=np.array([(int(r['policy_ready_ns'])-int(r['host_receive_ns']))/1e9 for r in rows if r['policy_ready_ns']])
 if (delay<0).any():raise ValueError('negative preprocessing duration')
 report['cameras'][name]={'configured_fps':meta['results'][name]['settings']['fps'],'delivered_fps':float(1/delta.mean()),'arrival_interval_s':stats(delta),'preprocessing_delay_s':stats(delay),'exact_repeats':sum(int(r['exact_repeat_of_previous']) for r in rows),'policy_consumed_timestamps':sum(bool(r['policy_consumed_ns']) for r in rows),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if a.export_profile:
 origin=min(int(rows[0]['host_receive_ns']) for rows in timelines.values())
 cadence={'schema':'host-camera-cadence-v1','source_trial':a.trial,'meaning':'host receive is capture-time proxy; exposure age unknown; inter-camera phase retained','cameras':{}}
 for raw,name in [('zed','top'),('wrist_right','wrist_cam'),('wrist_left','left_wrist_cam')]:
  rows=timelines[raw]
  capture=[(int(r['host_receive_ns'])-origin)/1e9 for r in rows]
  ready=[(int(r['policy_ready_ns'])-origin)/1e9 for r in rows]
  cadence['cameras'][name]={'capture_s':capture,'ready_s':ready,'cycle_s':capture[-1]-capture[0]+(capture[-1]-capture[0])/(len(capture)-1),'source_sha256':report['cameras'][raw]['source_sha256']}
 # A common cycle keeps the measured phase relationships when replaying the trace.
 duration=max(s['capture_s'][-1] for s in cadence['cameras'].values())+.05
 for schedule in cadence['cameras'].values():schedule['cycle_s']=duration
 a.export_profile.write_text(json.dumps(cadence,separators=(',',':'))+'\n')
