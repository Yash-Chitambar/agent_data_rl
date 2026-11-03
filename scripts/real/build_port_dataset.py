"""Real wrist frames with port-relative labels, for fine-tuning and testing the wrist nets. No motion.

    .venv/bin/python scripts/real/build_port_dataset.py --out data/real_training/port_v1 --eval wrist_v14w wrist_v18ws

Sources (data/real_runs/<day>/<run>):
  collect   : deploy/bimanual/collect.py sweep: settled frames, labels from the operator's anchor. Best quality.
  self      : a run whose plug really seated (--seated, default ai4): its own seated pose is the anchor, same grasp.
              Labels hold for the frames before the insertion. Frames while the arm moves are dropped (--max-qd).
  same_spot : a run with the same picked-grasp type and the same ZED port estimate (within --spot-mm) as a collect run of
              that day: its anchor is borrowed. Extra error: the grasp difference, about 1 mm in the in-hand estimates.
  unlabeled : every other run with wrist frames at the approach: listed, images only.
Label: plug tip in the port frame (x across, y up, z gap mm), rotation vector deg: deploy/bimanual/collect.py header.

Output: <out>/manifest.jsonl (one frame per line: image path, labels, run, source), <out>/images/ (policy images made from raw
frames of the stream), <out>/summary.json. Images already logged as the exact net input are referenced, not copied.
With --eval: each net on the labeled frames, error by gap bin.
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from blind_insertion.envs import kinematics as K   # noqa: E402
from blind_insertion.envs.inhand import measured_mean   # noqa: E402
from deploy import conventions as C   # noqa: E402
from deploy.bimanual.collect import rot   # noqa: E402
from deploy.bimanual.scene import SceneModel   # noqa: E402

SC = SceneModel('cpu')
FK = SC.chain['right']
OFF = C.joint_offsets('right')
Rs, ps = measured_mean(device='cpu')
TIP_S = (Rs.reshape(3, 3).float().T @ ps.reshape(3).float())


def events(run):
    return [json.loads(x) for x in open(os.path.join(run, 'events.jsonl'))]


def port_R(yaw_deg):
    """Port axes from the ZED yaw: z out of the port, y up, x across (right in the wrist view)."""
    y = math.radians(yaw_deg)
    z = np.array([-math.cos(y), -math.sin(y), 0.])
    up = np.array([0., 0., 1.])
    return torch.tensor(np.c_[np.cross(up, z), up, z], dtype=torch.float32)


def pose(q_b):
    p, R = FK.fk(torch.as_tensor(q_b, dtype=torch.float32).reshape(1, 6))
    return p.reshape(3), R.reshape(3, 3)


def label(q_b, a):
    p, R = pose(q_b)
    tip = p - R @ TIP_S
    lp = a['rel'] + a['eR'].T @ (tip - a['tip']) * 1e3
    return lp, torch.rad2deg(K.rot_log(a['eR'].T @ R @ a['R'].T @ a['eR'] @ rot(a['rv'])))


def anchor_of_collect(run):
    e = next((x for x in events(run) if x['kind'] == 'collect_anchor'), None)
    if e is None:
        return None
    return dict(rel=torch.tensor(e['rel_mm']), rv=torch.tensor(e['rv_deg']), R=torch.tensor(e['tcp_R']), tip=torch.tensor(e['tip']), eR=torch.tensor(e['port_R']),
                kind=e['anchor_kind'], inhand6=e.get('inhand6'))


def run_info(run):
    ev = events(run)
    st = {e['stage']: e['tick'] for e in ev if e.get('kind') == 'stage'}
    pc = [e for e in ev if e.get('kind') == 'estimate' and e.get('est') == 'zed_pc']
    meta = json.load(open(os.path.join(run, 'meta.json')))
    return ev, st, (pc[-1] if pc else None), meta


def anchor_of_seated(run):
    """Seated run: the measured pose at the end of the insertion is the anchor (plug at full depth)."""
    ev, st, pc, meta = run_info(run)
    t = np.load(os.path.join(run, 'ticks.npz'))
    i = np.searchsorted(t['k'], st['6 release']) - 1
    p, R = pose(t['q'][i, 0] + OFF)
    d = float(meta['chain_constants']['seat_depth_m']) * 1e3
    return dict(rel=torch.tensor([0., 0., -d]), rv=torch.zeros(3), R=R, tip=p - R @ TIP_S, eR=port_R(pc['yaw_deg']), kind='seated_run')


def rows_of_collect(run, a, day):
    fr = {}
    for x in map(json.loads, open(os.path.join(run, 'frames.jsonl'))):
        if x['cam'] == 'wrist_right':
            fr.setdefault(x['tag'], {})[x['kind']] = x['file']
    out = []
    for e in events(run):
        if e['kind'] != 'collect_sample':
            continue
        tag = e['frame_tag'].split('@')[0]
        out.append(dict(run=os.path.basename(run), day=day, source='collect', image=os.path.join(run, fr[tag]['policy']), raw=os.path.join(run, fr[tag].get('raw', '')) if 'raw' in fr[tag] else None,
                        p_mm=e['label_p_mm'], rv_deg=e['label_rv_deg'], q_b=e['q_b'], anchor=a['kind'], qd_max=e['qd_max'], moving=False))
    return out


def rows_of_run(run, a, day, source, max_qd, out_dir):
    """Wrist frames of a chain run up to the insertion: logged net inputs and raw stream frames, with q at the capture time."""
    ev, st, _, _ = run_info(run)
    t = np.load(os.path.join(run, 'ticks.npz'))
    k1 = st.get('5 insert', t['k'][-1])
    k0 = st.get('3 transport', k1)
    i0, i1 = np.searchsorted(t['k'], k0), np.searchsorted(t['k'], k1)
    tt = t['t'][i0:i1]
    if not len(tt):
        return []
    from blind_insertion.assets.real_camera import load_spec, real_policy_image
    import rig
    spec, res = load_spec(rig.SPEC), 320
    rows, last_pol = [], {}
    for x in map(json.loads, open(os.path.join(run, 'frames.jsonl'))):
        if x['cam'] != 'wrist_right' or x.get('t_capture') is None or not (tt[0] <= x['t_capture'] <= tt[-1]):
            continue
        j = i0 + int(np.searchsorted(tt, x['t_capture']))
        j = min(j, len(t['t']) - 1)
        q, qd = t['q'][j, 0] + OFF, float(np.abs(t['qd'][j, 0]).max())
        if x['kind'] == 'policy':
            img, raw = os.path.join(run, x['file']), None
        elif x['tag'] == 'stream':
            if qd > max_qd:
                continue
            raw = os.path.join(run, x['file'])
            img = os.path.join(out_dir, 'images', os.path.basename(run) + '_' + os.path.basename(x['file']).replace('.jpg', '.png'))
            if not os.path.exists(img):
                f = cv2.imread(raw)
                os.makedirs(os.path.dirname(img), exist_ok=True)
                cv2.imwrite(img, cv2.cvtColor(real_policy_image(f, spec, res, 'right'), cv2.COLOR_RGB2BGR))
        else:
            continue
        r = dict(run=os.path.basename(run), day=day, source=source, image=img, raw=raw, tag=x['tag'], q_b=q.tolist(), qd_max=qd, moving=qd > .02)
        if a is not None:
            lp, lr = label(q, a)
            r.update(p_mm=lp.tolist(), rv_deg=lr.tolist(), anchor=a['kind'])
        rows.append(r)
    return rows


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def eval_nets(rows, names, dev):
    from deploy.bimanual import netcache as NC
    import torch.nn.functional as F
    res = {}
    lab = [r for r in rows if 'p_mm' in r]
    for name in names:
        net, scale, mode = NC.wrist(name, dev)
        pred = []
        for i in range(0, len(lab), 64):
            im = torch.as_tensor(np.stack([load_rgb(r['image']) for r in lab[i:i + 64]]), device=dev)
            if mode == 'crop':
                im = im[:, 40:232, 64:256].contiguous()
            else:
                im = F.interpolate(im.permute(0, 3, 1, 2).float(), size=(192, 192), mode='area').round().byte().permute(0, 2, 3, 1)
            with torch.no_grad():
                pred.append((net(im) * scale).cpu().numpy())
        pred = np.concatenate(pred)
        for r, p in zip(lab, pred):
            r.setdefault('pred', {})[name] = p.round(3).tolist()
        y = np.array([r['p_mm'] + r['rv_deg'] for r in lab])
        res[name] = {}
        for src in sorted({r['source'] for r in lab}):
            ms = np.array([r['source'] == src for r in lab])
            for lo, hi in ((0, 10), (10, 25), (25, 70)):
                m = ms & (y[:, 2] >= lo) & (y[:, 2] < hi)
                if m.sum() < 2:
                    continue
                e = pred[m] - y[m]
                sl = np.polyfit(y[m, 1], pred[m, 1], 1)[0] if np.ptp(y[m, 1]) > 1 else None
                res[name][f'{src} gap {lo}-{hi}'] = dict(n=int(m.sum()), mean=e.mean(0).round(2).tolist(), rms=np.sqrt((e ** 2).mean(0)).round(2).tolist(),
                                                         y_slope_pred_vs_label=None if sl is None else round(float(sl), 2))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', default='data/real_runs')
    ap.add_argument('--out', default='data/real_training/port_v1')
    ap.add_argument('--seated', nargs='*', default=['2026-11-01/approach_insert_171123_ai4'], help='runs whose plug really seated (operator)')
    ap.add_argument('--max-qd', type=float, default=.1, help='rad/s: stream frames faster than this are dropped')
    ap.add_argument('--spot-mm', type=float, default=2.5)
    ap.add_argument('--eval', nargs='*', default=[])
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, used = [], set()
    anchors = {}
    for run in sorted(glob.glob(os.path.join(a.runs, '*/collect_*'))):
        if '_dry' in run or not os.path.exists(os.path.join(run, 'events.jsonl')):
            continue
        an = anchor_of_collect(run)
        if an is None:
            continue
        day = Path(run).parent.name
        rows += rows_of_collect(run, an, day)
        pc = run_info(run)[2]
        anchors.setdefault(day, []).append((an, pc))
        used.add(run)
    for s in a.seated:
        run = os.path.join(a.runs, s)
        rows += rows_of_run(run, anchor_of_seated(run), Path(run).parent.name, 'self', a.max_qd, a.out)
        used.add(run)
    for run in sorted(glob.glob(os.path.join(a.runs, '*/full_*')) + glob.glob(os.path.join(a.runs, '*/approach_insert_*'))):
        if '_dry' in run or run in used or not os.path.exists(os.path.join(run, 'ticks.npz')):
            continue
        try:
            ev, st, pc, _ = run_info(run)
        except (OSError, ValueError, KeyError):
            continue
        if '4 approach' not in st or pc is None:
            continue
        day, an = Path(run).parent.name, None
        if os.path.basename(run).startswith('full_'):        # a real pick: the anchor of a collect run at the same spot
            for b, pcb in anchors.get(day, []):
                if pcb is not None and np.linalg.norm((np.array(pc['port_xy']) - pcb['port_xy'])) * 1e3 <= a.spot_mm:
                    an = b
        rows += rows_of_run(run, an, day, 'same_spot' if an is not None else 'unlabeled', a.max_qd, a.out)
    recheck = {}
    for run in sorted(glob.glob(os.path.join(a.runs, '*/collect_*'))):
        if '_dry' in run or not os.path.exists(os.path.join(run, 'events.jsonl')):
            continue
        for e in events(run):
            if e['kind'] == 'collect_recheck':     # labels back in the port: the anchor's own error, or the grasp moved
                recheck[os.path.basename(run)] = dict(label_p_mm=e['label_p_mm'], label_rv_deg=e['label_rv_deg'], depth_mm=e['depth_mm'])
    res = dict(anchor_recheck=recheck, n=len(rows), by_source={s: sum(r['source'] == s for r in rows) for s in sorted({r['source'] for r in rows})},
               runs=sorted({r['run'] for r in rows}))
    if a.eval:
        res['nets'] = eval_nets(rows, a.eval, a.device)
    with open(os.path.join(a.out, 'manifest.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    json.dump(res, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
