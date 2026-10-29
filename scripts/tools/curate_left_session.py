"""Build a smaller immutable-ready left-arm evidence bundle from a completed session.

Structured logs and calibration boards are retained in full. Camera CSVs retain
original host timestamps and frame IDs, while redundant PNGs are sampled by ID.
The hand-pushed PC overhead sequence is kept at every saved frame.
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

IMAGE_SUFFIXES={'.png','.jpg','.jpeg'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    a=p.parse_args();source=a.source.resolve(strict=True);dest=a.destination.resolve()
    if dest.exists() or dest==source or source in dest.parents:raise SystemExit('destination must be new and outside source')
    dest.mkdir(parents=True)
    # Preserve all numeric, textual, controller and protocol data; camera rasters
    # are handled separately below so long RGB sessions do not dominate transfer.
    transfer_files={'usbc_session_manifest.json','usbc_session_sha256sums.txt','usbc_session_complete.json'}
    for item in source.iterdir():
        if item.name=='cameras' or item.name in transfer_files:continue
        target=dest/item.name
        if item.is_symlink():raise SystemExit(f'refusing symlink: {item}')
        if item.is_dir():shutil.copytree(item,target,symlinks=False)
        else:shutil.copy2(item,target)
    stats=[]
    for run in sorted((source/'cameras').iterdir()):
        if run.is_symlink():raise SystemExit(f'refusing symlink: {run}')
        run_target=dest/'cameras'/run.name;run_target.mkdir(parents=True,exist_ok=True)
        for camera in sorted(run.iterdir()):
            if camera.is_symlink():raise SystemExit(f'refusing symlink: {camera}')
            if camera.is_file():
                shutil.copy2(camera,run_target/camera.name)
                continue
            target=dest/'cameras'/run.name/camera.name;target.mkdir(parents=True)
            files={x.name:x for x in camera.iterdir() if x.is_file()}
            raster=[name for name,path in files.items() if path.suffix.lower() in IMAGE_SUFFIXES]
            small= sum(files[n].stat().st_size for n in raster)<=100*1024*1024
            full=(run.name in {'initial_rgb_001','board_rgb_001','bimanual_stationary_001'} or
                  (run.name=='pc_manual_motion_001' and camera.name=='zed') or small)
            if run.name=='contact_interactive_001':stride=180
            elif run.name in {'left_reader_002','contact_final_002','contact_interactive_002'}:stride=30
            elif run.name=='pc_manual_motion_001':stride=5
            else:stride=120
            if full:stride=1
            selected=set()
            csvpath=files.get('frames.csv')
            if csvpath:
                with csvpath.open(newline='') as f:
                    for row in csv.DictReader(f):
                        name=row.get('raw_png') or ''
                        if not name:continue
                        m=re.search(r'(\d+)(?=\.[^.]+$)',name)
                        frame_id=int(row.get('local_frame_id') or (m.group(1) if m else -1))
                        if full or (frame_id>=0 and frame_id%stride==0):selected.add(name)
                        # Keep paired policy image only when it belongs to a kept raw frame.
                        policy=row.get('policy_png') or ''
                        if policy and (full or (frame_id>=0 and frame_id%stride==0)):selected.add(policy)
            if not csvpath:
                for name in raster:
                    m=re.search(r'(\d+)(?=\.[^.]+$)',name)
                    if full or (m and int(m.group(1))%stride==0):selected.add(name)
            for name,path in files.items():
                if path.suffix.lower() not in IMAGE_SUFFIXES or name in selected:
                    shutil.copy2(path,target/name)
            # Explicitly subset CSV rows to the retained frames, keeping original IDs/times.
            if csvpath:
                with csvpath.open(newline='') as fi, (target/'frames.csv').open('w',newline='') as fo:
                    reader=csv.DictReader(fi);writer=csv.DictWriter(fo,fieldnames=reader.fieldnames);writer.writeheader()
                    for row in reader:
                        if not row.get('raw_png') or row['raw_png'] in selected:writer.writerow(row)
            saved=sum(1 for name in selected if name in files)
            total=len(raster)
            stats.append({'run':run.name,'camera':camera.name,'source_rasters':total,
                          'kept_rasters':saved,'stride_frames':stride,'full_rate':full})
    report={'schema':'curated_left_session_v1','source':str(source),
            'selection':'all structured logs and calibration images; small/static camera runs retained; long camera runs sampled every 120 local frames (about 6-8 s at measured acquisition rates); task, grasp and contact debug runs sampled every 30 frames; manual PC motion keeps every overhead frame and every fifth wrist frame; redundant failed interactive run sampled every 180 frames.',
            'limitations':['Do not use sampled camera sequences as continuous video or assume aligned exposure clocks.',
                           'Original full session remains the source of record; raw camera images omitted by this curation remain local.'],
            'camera_selection':stats}
    (dest/'curation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'destination':str(dest),'camera_rasters_kept':sum(r['kept_rasters'] for r in stats),
                      'camera_rasters_source':sum(r['source_rasters'] for r in stats),'runs':len(stats)},indent=2))

if __name__=='__main__':main()
