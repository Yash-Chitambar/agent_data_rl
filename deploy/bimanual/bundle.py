"""Deploy bundle for the bimanual chain: every checkpoint, the configs, the sim constants the encoder-only path
needs, and a manifest (sha256, source path, training commit, expected input preprocessing). Weights are not in git.

    # mason (GPU; builds one sim world for the constants)
    .venv/bin/python -m deploy.bimanual.bundle stage            # -> deploy/bundle/ (+ deploy/bundle.tar)
    .venv/bin/python -m deploy.bimanual.bundle copy-cmd         # prints the rsync line for the rig PC
    # rig (duchamp3), from the repo root
    .venv/bin/python -m deploy.bimanual.bundle verify           # sha256 of every file against MANIFEST.json
    .venv/bin/python -m deploy.bimanual.bundle link             # runs/perception/*.pt and runs/chain/ -> the bundle

`link` makes runs/perception/<name>.pt (the paths eval_chain.py and the runner load) symlinks into the bundle; the
runner refuses to start when a link does not resolve to a verified bundle file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'deploy/bundle'
WT = Path('/home/yashchitambar/wt')
APPROACH = WT / 'policy-approach-hold/runs/perception'
PICK = WT / 'policy-pick/runs/perception'
INSERT = WT / 'policy-insertion/runs/P4-insert-best'
CHAIN_PERCEPTION = WT / 'policy-chain/runs/perception'     # the copies the 99.56-99.66 % chain eval loaded

FRAME_320 = 'policy frame 320x320 uint8 RGB'
WRIST = (f'right wrist {FRAME_320}: real = assets/real_camera.real_policy_image(raw 1280x1024 BGR, spec, 320, "right") '
         '(undistort, policy_crop_px square on the principal point, INTER_AREA resize, BGR->RGB); sim = Warp render of '
         'wrist_cam at camera_resolution 320')
TOP = (f'ZED {FRAME_320}: real = scenes/physical_rig.real_top_policy_image(rectified HD1080 LEFT BGR, 320) (square '
       'crop on the principal point, INTER_AREA, BGR->RGB); sim = Warp render of top at 320')

NETS = {
    'zed': dict(src=APPROACH / 'zed.pt', script='scripts/perception/train_zed.py (approach-hold)',
                input=TOP + '; ZedEstimator crops ZED_CROP (x 110, y 70, 160 px) -> PoseNet 160 px',
                output='port xy offset from NOMINAL_PORT_XY (mm, x scale 30) and yaw (deg) -> port_xy m, yaw rad',
                capture='both arms at configs/zed_capture_pose.json, grippers fully open'),
    'zed_v1_gate_arm': dict(src=APPROACH / 'zed_v1_gate_arm.pt', script='scripts/perception/train_zed_presence.py',
                            input='same crop as zed (presence net + Mahalanobis OOD on zed features + arm-pose envelope on '
                                  'believed joint angles and the right finger opening)',
                            output='gate ok (presence >= 0.5, Mahalanobis <= 1.25 x ood_thresh, arms within tol of the capture pose)'),
    'reader_zed_cap': dict(src=PICK / 'reader_zed_cap.pt', script='scripts/perception/train_reader_zed.py (pick)',
                           input=TOP + '; crop (x 80, y 173, 128 px) -> PoseNet 128 px; presence net on the same crop',
                           output='reader body-centre xy (m, right base) and plug yaw (rad) + presence/arm-envelope gate',
                           capture='left arm parked (PickTaskCfg.left_park_q), right at the capture pose, right gripper fully open'),
    'reader_servo_hover': dict(src=PICK / 'reader_servo_hover.pt', script='scripts/perception/train_reader_servo.py',
                               input=WRIST + '; PoseNet area-resizes to 224 px', output='grasp position relative to the hand (m, grasp_site frame); 3 outputs'),
    'reader_servo_near': dict(src=PICK / 'reader_servo_near.pt', script='scripts/perception/train_reader_servo.py',
                              input=WRIST + '; PoseNet area-resizes to 224 px', output='grasp position relative to the hand (m); 3 outputs; waypoint frames'),
    'reader_servo_near6': dict(src=PICK / 'reader_servo_near6.pt', script='scripts/perception/train_reader_servo.py',
                               input=WRIST + '; PoseNet area-resizes to 224 px', output='grasp pose relative to the hand (m, rotation vector rad); rotation used'),
    'wrist_v14w': dict(src=APPROACH / 'wrist_v14w.pt', script='scripts/perception/train_wrist*.py (approach-hold round 8)',
                       input=WRIST + '; eval_chain area-resizes 320 -> 192 (F.interpolate mode area, rounded to uint8)',
                       output='plug tip in the port frame (mm) and rotation vector (deg): x, y, gap, rx, ry, roll'),
    'wrist_v18ws': dict(src=APPROACH / 'wrist_v18ws.pt', script='scripts/perception/train_approach_estimators.py (round 9 stop-gap specialist, servo-pick in-hand)',
                        input=WRIST + '; area-resize 320 -> 192', output='as wrist_v14w; used at waypoints <= 7 mm'),
    'inhand_ih5': dict(src=APPROACH / 'inhand_ih5.pt', script='scripts/perception/train_approach_estimators.py (round 9 in-hand head, servo-pick in-hand)',
                       input=WRIST + '; area-resize 320 -> 192; frame at the 6 mm stop',
                       output='reader in the right TCP frame minus the nominal grasp: dx dy dz mm, rotation vector deg (InsertionTask.inhand_error)'),
}
POLICY = dict(src=INSERT / 'best.pt', readme=INSERT / 'README.json', script='blind_insertion/train_bimanual.py (policy/insertion, K-J 5M)',
              input='68-D actor observation history (30) of InsertionTask._actor_obs (deploy/bimanual/insert.py); left joint-6 '
                    'entries masked in the loader', output='6 normalized right-arm joint increments (x 0.04 rad, accumulate, cap 0.05)')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def train_commit(path):
    """Commit of the source worktree at the time the checkpoint was written (HEAD before its mtime)."""
    p = Path(path).resolve()
    wt = next((q for q in p.parents if (q / '.git').exists()), None)
    if wt is None:
        return None
    mt = int(p.stat().st_mtime)
    try:
        c = subprocess.run(['git', '-C', str(wt), 'log', '-1', '--format=%H %cd %s', f'--before=@{mt}', '--all'],
                           capture_output=True, text=True, timeout=20).stdout.strip()
        br = subprocess.run(['git', '-C', str(wt), 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    return {'worktree': str(wt), 'branch': br, 'commit_before_mtime': c,
            'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mt))}


def constants(policy_path, device='cuda:0'):
    """Sim values the encoder-only path needs, read off one eval_chain Chain world (InsertionTask + LeftHold)."""
    from deploy.bimanual.io_sim import chain_argv, eval_chain
    cfg = json.loads((ROOT / 'deploy/config/bimanual.json').read_text())
    EC = eval_chain()
    ch = EC.Chain(EC.parse(chain_argv(cfg, policy_path, 1)))
    env = ch.env
    R, p = env._R_obs, env._p_obs
    mask = list(ch.pol.mask_idx)
    from blind_insertion.envs.insertion_task import InsertionCfg
    out = dict(R_obs=R.cpu().tolist(), p_obs=p.cpu().tolist(), hold_yaws_deg=list(env.hold.cfg.yaws_deg),
               seat_depth_m=float(env.seat_depth), warmup_steps=int(env.icfg.warmup_steps), history=int(env.icfg.history),
               obs_dim=int(env.obs_dim), mask_idx=mask, mask_left_obs=env.icfg.mask_left_obs,
               env_mask_idx=list(InsertionCfg(mask_left_obs=env.icfg.mask_left_obs).left_mask_idx()),
               max_tcp_speed_m_s=float(env.icfg.max_tcp_speed_m_s), target_cap_rad=float(env.cfg.target_cap_rad),
               action_scale_rad=float(env.cfg.action_scale_rad), q_right_nominal=env.q_right_nominal.cpu().tolist(),
               pc_port_z_sim_reset=float(ch.rt.q[0, ch.rt.pc_qadr + 2]),
               policy_meta={k: v for k, v in ch.pol.meta.items() if k != 'paths'})
    return out


def stage(a):
    out = Path(a.out)
    (out / 'models').mkdir(parents=True, exist_ok=True)
    (out / 'configs').mkdir(exist_ok=True)
    man = {'staged': time.strftime('%Y-%m-%d %H:%M:%S %Z'), 'staged_from': subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(), 'files': {}}
    for name, d in NETS.items():
        dst = out / 'models' / f'{name}.pt'
        shutil.copy2(d['src'], dst)
        chain_copy = CHAIN_PERCEPTION / f'{name}.pt'
        man['files'][f'models/{name}.pt'] = dict(
            kind='perception', sha256=sha256(dst), bytes=dst.stat().st_size, source=str(d['src']),
            training=train_commit(d['src']), training_script=d['script'], input=d['input'], output=d['output'],
            capture=d.get('capture'),
            same_as_chain_eval_copy=(sha256(chain_copy) == sha256(dst)) if chain_copy.exists() else None)
    dst = out / 'models/insert_KJ.pt'
    shutil.copy2(POLICY['src'], dst)
    shutil.copy2(POLICY['readme'], out / 'models/insert_KJ.README.json')
    man['files']['models/insert_KJ.pt'] = dict(kind='policy', sha256=sha256(dst), bytes=dst.stat().st_size,
                                               source=str(POLICY['src']), training=train_commit(POLICY['src']),
                                               training_script=POLICY['script'], input=POLICY['input'], output=POLICY['output'])
    man['files']['models/insert_KJ.README.json'] = dict(kind='doc', sha256=sha256(out / 'models/insert_KJ.README.json'),
                                                         source=str(POLICY['readme']))
    for rel in ('deploy/config/bimanual.json', 'blind_insertion/configs/zed_capture_pose.json',
                'blind_insertion/configs/real_cameras_duchamp3.json', 'results/usbc/policy/pick/task_layout.json'):
        dst = out / 'configs' / Path(rel).name
        shutil.copy2(ROOT / rel, dst)
        man['files'][f'configs/{dst.name}'] = dict(kind='config', sha256=sha256(dst), source=rel)
    link(a)                                     # eval_chain loads runs/perception/<name>.pt
    print('building one sim world for the constants ...', flush=True)
    c = constants(out / 'models/insert_KJ.pt', a.device)
    (out / 'configs/chain_constants.json').write_text(json.dumps(c, indent=1))
    man['files']['configs/chain_constants.json'] = dict(kind='config', sha256=sha256(out / 'configs/chain_constants.json'),
                                                        source='deploy.bimanual.bundle.constants (eval_chain Chain, nworld 1)')
    (out / 'MANIFEST.json').write_text(json.dumps(man, indent=1))
    if a.tar:
        subprocess.run(['tar', '-cf', str(out.parent / 'bundle.tar'), '-C', str(out.parent), out.name], check=True)
    print(json.dumps({k: {'sha256': v['sha256'][:12], 'source': v.get('source')} for k, v in man['files'].items()}, indent=1))
    bad = [k for k, v in man['files'].items() if v.get('same_as_chain_eval_copy') is False]
    if bad:
        print('WARNING: differ from the policy-chain copies the chain eval used:', bad)


def verify(a=None, quiet=False):
    out = Path(getattr(a, 'out', None) or BUNDLE)
    man = json.loads((out / 'MANIFEST.json').read_text())
    bad = [k for k, v in man['files'].items() if not (out / k).exists() or sha256(out / k) != v['sha256']]
    if not quiet:
        print('bundle OK' if not bad else f'bundle MISSING or CHANGED: {bad}')
    return bad


def link(a):
    out = Path(a.out).resolve()
    for d in ('runs/perception', 'runs/chain'):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    pairs = [(ROOT / f'runs/perception/{n}.pt', out / f'models/{n}.pt') for n in NETS]
    pairs.append((ROOT / 'runs/chain/P4-insert-best_KJ.pt', out / 'models/insert_KJ.pt'))
    for dst, src in pairs:
        if dst.is_symlink() or dst.exists():
            if dst.resolve() == src:
                continue
            bak = dst.with_suffix(dst.suffix + f'.bak{int(time.time())}')
            dst.rename(bak)
            print(f'moved existing {dst} -> {bak}')
        os.symlink(src, dst)
        print(f'{dst.relative_to(ROOT)} -> {src}')


def check_links(out=BUNDLE):
    """Names whose runs/ path does not resolve to the bundle file."""
    out = Path(out).resolve()
    bad = [n for n in NETS if (ROOT / f'runs/perception/{n}.pt').resolve() != out / f'models/{n}.pt']
    if (ROOT / 'runs/chain/P4-insert-best_KJ.pt').resolve() != out / 'models/insert_KJ.pt':
        bad.append('insert_KJ')
    return bad


def copy_cmd(a):
    print('# from this repo root on mason; <rig-repo> = the rig checkout on duchamp3 (e.g. ~/peg_test)')
    print(f'rsync -av --progress {BUNDLE.relative_to(ROOT)}/ duchamp3:<rig-repo>/deploy/bundle/')
    print('# then on duchamp3:  .venv/bin/python -m deploy.bimanual.bundle verify && .venv/bin/python -m deploy.bimanual.bundle link')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('stage')
    s.add_argument('--device', default='cuda:0')
    s.add_argument('--tar', action='store_true')
    for nm in ('verify', 'link', 'copy-cmd'):
        sub.add_parser(nm)
    for p in sub.choices.values():
        p.add_argument('--out', default=str(BUNDLE))
    a = ap.parse_args()
    r = {'stage': stage, 'verify': verify, 'link': link, 'copy-cmd': copy_cmd}[a.cmd](a)
    raise SystemExit(1 if r else 0)
