"""Bimanual USB-C chain on the real YAMs (duchamp3), staged, with a dry run through the same code.

Stages (--stage), each from the arms' rest pose and back:
  capture          both arms to the ZED capture pose -> PC estimate; park left -> reader estimate (one re-take)
  hold             capture -> park left -> left guarded press (--hold-force N, default 25) for --hold-seconds -> unload
  approach_insert  capture -> right arm to the post-pick lift pose, operator puts the reader in the open jaws, close
                   -> hold -> transport -> wrist-servo approach -> in-hand estimate -> torque hand-over -> K-J
                   insertion -> seated detector / time limit -> release, retreat, unload
  pick             capture -> park left -> reader estimate -> servo-descent pick -> lift; operator takes the reader
  full             capture -> pick -> hold -> transport -> approach -> insertion -> release

    # dry run (no CAN, no cameras): one eval_chain sim world through this exact code path
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m deploy.run_bimanual --dry-run --stage full --seed 7000
    # real (duchamp3): bases clamped, scripts/real/estop.py running in another terminal
    .venv/bin/python -m deploy.run_bimanual --stage capture --name cap1 --notes "PC at spot 1"

Every run writes data/real_runs/<date>/<run_id>/ (deploy/bimanual/runlog.py). Safety: SafeArm (joint box, table
clearance, step cap) on every command, deploy/bimanual/guards.py (torque estimate, speed, TCP box, press time,
software stop = Enter in this terminal or estop.py, --max-steps). A stop holds both arms at their measured pose
(no press force, no push) and asks what to do.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'scripts/real'))

from deploy import conventions as C  # noqa: E402
from deploy.bimanual import bundle as BU  # noqa: E402
from deploy.bimanual.guards import GuardStop, Guards, StopKey  # noqa: E402
from deploy.bimanual.runlog import RunLogger, git_state  # noqa: E402

SIDES = ('right', 'left')
STAGE_CHOICES = ('capture', 'hold', 'approach_insert', 'pick', 'full', 'collect')


def load_cfg(path):
    cfg = json.loads(Path(path).read_text())
    return {k: v for k, v in cfg.items() if not k.startswith('_')}


def apply_overrides(cfg, batch=None, open_mm=None, grasp_set=()):
    """--batch (config pick.batches.<name>: grasp_v3 keys, body_clearance), --open-mm, --grasp-set key=json. Logged in meta.json."""
    pc = cfg['pick']
    bs = pc.pop('batches', {})
    if batch is not None:
        if batch not in bs:
            raise SystemExit(f'batch {batch}: not in pick.batches ({sorted(bs)})')
        pc['batch'] = batch
        pc.setdefault('grasp_v3', {}).update({k: v for k, v in bs[batch].get('grasp_v3', {}).items()})
        cfg['safety']['body_clearance'].update(bs[batch].get('body_clearance', {}))
    if open_mm is not None:
        cfg['gripper']['right_approach_open_m'] = float(open_mm) * 1e-3
    for kv in grasp_set:
        k, v = kv.split('=', 1)
        pc.setdefault('grasp_v3', {})[k] = json.loads(v)
    return cfg


def gripper_maps(cfg):
    out = {}
    for s in SIDES:
        g = cfg['gripper'][s]
        out[s] = C.GripperMap(g['open'], g['close'], g['m_open'], g.get('g_reader'), g.get('m_reader', 0.019))
    return out


def calib_versions():
    out = {}
    for rel in ('blind_insertion/configs/real_cameras_duchamp3.json', 'blind_insertion/configs/zed_capture_pose.json',
                'deploy/config/bimanual.json', 'results/usbc/policy/pick/task_layout.json'):
        out[rel] = BU.sha256(ROOT / rel)
    spec = json.loads((ROOT / 'blind_insertion/configs/real_cameras_duchamp3.json').read_text())
    out['joint_offsets_rad'] = {s: spec['arms'][s]['joint_offsets_rad'] for s in SIDES}
    out['calibration_version'] = spec.get('calibration_version', 'pre-handeye.py (2026-10-25 wrist calibration)')
    out['wrist_handeye'] = {s: {'pose_in_link6': spec['wrist'][s].get('pose_in_link6'), 'calibrated': spec['wrist'][s].get('calibrated'),
                                'quality': spec['wrist'][s].get('quality')} for s in SIDES}
    out['wrist_sim_pose_in_link6'] = spec['wrist']['sim'].get('pose_in_link6')
    return out


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', choices=STAGE_CHOICES, required=True)
    ap.add_argument('--dry-run', action='store_true', help='SimIO (eval_chain world) instead of the arms and cameras')
    ap.add_argument('--grasp-fidelity', default=None, help='dry run only: realistic grasp model preset or JSON (docs/real/GRASP_FIDELITY.md)')
    ap.add_argument('--seed', type=int, default=7000, help='dry run: first episode seed (eval_chain seeds)')
    ap.add_argument('--episodes', type=int, default=1, help='dry run: episodes (seed, seed + 1, ...)')
    ap.add_argument('--name', default='', help='run id suffix')
    ap.add_argument('--notes', default='', help='operator notes (meta.json)')
    ap.add_argument('--pc-spot', default='', help='measured PC spot, JSON {"port_xy": [x, y], "yaw_deg": d, "method": ..., "unc_mm": ...} (meta, exporter labels)')
    ap.add_argument('--max-steps', type=int, default=None, help='stop after this many 50 Hz ticks')
    ap.add_argument('--speed', type=float, default=1.0, help='scripted motions x speed (pick, transport, approach ramps, retreat); 1 = sim')
    ap.add_argument('--hold-force', type=float, default=None, help='left press force (N); config default 25; 15 allowed')
    ap.add_argument('--hold-seconds', type=float, default=5.0, help='stage hold: keep the press this long')
    ap.add_argument('--pick-mode', choices=('servo', 'expert'), default='servo', help="'expert' = eval_chain's default pick (no servo): dry-run equivalence only")
    ap.add_argument('--batch', default=None, help='real test batch from pick.batches in the config (docs/real/PICK_GRASP_V3.md)')
    ap.add_argument('--open-mm', type=float, default=None, help='descent opening per finger (mm): overrides gripper.right_approach_open_m (24, 25, 26)')
    ap.add_argument('--grasp-set', action='append', default=[], metavar='KEY=JSON', help='override one pick.grasp_v3 key, e.g. --grasp-set max_attempts=3')
    ap.add_argument('--pick-test', choices=('touchdown', 'air'), default=None, help="commissioning, stage pick with pick.grasp_v3, jaws stay open, the hand rises: 'air' = slow free descent above the plan (no contact expected), 'touchdown' = guarded touch-down and hold on the table")
    ap.add_argument('--collect-n', type=int, default=150, help='stage collect: poses in the sweep (deploy/bimanual/collect.py)')
    ap.add_argument('--collect-seed', type=int, default=0, help='stage collect: seed of the sweep poses; use a new one per run')
    ap.add_argument('--align-mode', choices=('measure', 'align', 'full'), default=None, help='config align on: measure = camera measurements at the stop, no align motion, no insertion; align = move, no insertion; full = align then insertion (default: config align.mode)')
    ap.add_argument('--align-set', action='append', default=[], metavar='KEY=JSON', help='override one align key (dry-run experiments and rig commissioning; logged in meta.json)')
    ap.add_argument('--sim-phys', default='', help='dry run only, JSON of runtime fields, e.g. {"pc_real_look": "blind_insertion/assets/pc_real_look", "inhand_real": "v2"}')
    ap.add_argument('--sim-inhand', default='', help='dry run only, stage approach_insert: in-hand pose from this real-grasp preset (v2), one draw per episode')
    ap.add_argument('--enc-offset-deg', type=float, default=None, help='dry run only: encoder-offset error DR, +/- deg (default eval_chain: 1.5)')
    ap.add_argument('--yes', action='store_true', help='answer every prompt with Enter (dry runs default to this)')
    ap.add_argument('--pause-in-press', action='store_true', help='also prompt between hold, transport, approach and insertion (the press keeps running; a watchdog unloads at hold_max_s)')
    ap.add_argument('--device', default='cpu', help='kinematics, controllers and the insertion policy (one world: CPU is faster than GPU launches)')
    ap.add_argument('--net-device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='perception nets')
    ap.add_argument('--config', default=str(ROOT / 'deploy/config/bimanual.json'))
    ap.add_argument('--out', default=str(ROOT / 'data/real_runs'))
    ap.add_argument('--no-left-wrist', action='store_true', help='do not open the left wrist camera (logging only)')
    ap.add_argument('--check-obs', action='store_true', help='dry run: compare the insertion observation with InsertionTask._actor_obs every tick')
    ap.add_argument('--no-home', action='store_true', help='stay at the end pose (holding) instead of returning to the start pose')
    return ap.parse_args(argv)


class Runner:
    def __init__(self, a):
        self.a = a
        self.cfg = apply_overrides(load_cfg(a.config), a.batch, a.open_mm, a.grasp_set)
        if a.align_mode or a.align_set:
            if not self.cfg.get('align', {}).get('enabled'):
                raise SystemExit('--align-mode / --align-set need a config with align.enabled')
            if a.align_mode:
                self.cfg['align']['mode'] = a.align_mode
            for kv in a.align_set:
                k, v = kv.split('=', 1)
                self.cfg['align'][k] = json.loads(v)
        self.dev = torch.device(a.device)
        bundle = ROOT / self.cfg['bundle_dir']
        bad = BU.verify(argparse.Namespace(out=str(bundle)), quiet=True)
        links = BU.check_links(bundle)
        if bad or links:
            raise SystemExit(f'bundle not verified: changed/missing {bad}, runs/ links not into the bundle {links}: '
                             f'run python -m deploy.bimanual.bundle verify / link')
        self.bundle_manifest = json.loads((bundle / 'MANIFEST.json').read_text())
        self.config_matches_bundle = BU.sha256(a.config) == self.bundle_manifest['files']['configs/bimanual.json']['sha256']
        if not self.config_matches_bundle:
            print(f'NOTE: {a.config} differs from the bundle copy (configs/bimanual.json); the run uses {a.config}', flush=True)
        self.consts = json.loads((bundle / 'configs/chain_constants.json').read_text())
        self.policy_path = bundle / 'models/insert_KJ.pt'
        self.gmaps = gripper_maps(self.cfg)
        self.keys = None if a.dry_run else StopKey()
        self.auto = '' if (a.dry_run or a.yes) else None
        from deploy.bimanual.scene import SceneModel
        self.scene = SceneModel(self.dev)
        self.io = None

    # ------------------------------------------------------------------ setup
    def open_io(self, log):
        # every perception net BEFORE the arms connect: loading holds the GIL up to ~0.2 s (deploy/bimanual/netcache.py)
        from deploy.bimanual import netcache as NC
        dt = NC.preload(self.cfg, self.dev, self.a.net_device or str(self.dev))
        print(f'nets loaded in {dt:.1f} s (before the arms connect)', flush=True)
        if log is not None:
            log.event('nets_preloaded', seconds=dt)
        if self.a.dry_run:
            from deploy.bimanual.io_sim import SimIO
            if self.a.grasp_fidelity:
                self.cfg.setdefault('sim', {})['grasp_fidelity'] = self.a.grasp_fidelity
            if self.a.sim_phys:
                self.cfg.setdefault('sim', {})['phys_json'] = self.a.sim_phys
            if self.a.sim_inhand:
                self.cfg.setdefault('sim', {})['inhand_preset'] = self.a.sim_inhand
            extra = ['--enc-offset-deg', str(self.a.enc_offset_deg)] if self.a.enc_offset_deg is not None else []
            self.io = SimIO(self.cfg, self.gmaps, self.policy_path, device=str(self.dev), extra_args=extra)
            return
        import estop
        import rig
        if estop.stopped():
            raise SystemExit(f'{estop.STOP_FILE} is set: clear it in estop.py first')
        for s in SIDES:
            rig.can_channel(s)
        if self.cfg['gripper']['left'].get('g_reader') is None:
            print('NOTE: left gripper map uncalibrated (proportional); it only opens and closes.', flush=True)
        print('Rig checklist: both arm bases CLAMPED; scripts/real/estop.py running in another terminal; workspace clear;'
              ' PC on its spot; reader in the strip (pick/full) or in your hand (approach_insert).', flush=True)
        if self.keys.ask('Type GO to connect both arms (PD hold from connect):', auto='GO' if self.a.yes else None) != 'GO':
            raise SystemExit('not confirmed')
        from deploy.bimanual.io_real import RealIO
        cams = ('top', 'wrist_right') if self.a.no_left_wrist else ('top', 'wrist_right', 'wrist_left')
        kp, kd = self.scene.kp.cpu().numpy(), self.scene.kd.cpu().numpy()
        self.io = RealIO(self.cfg, kp, kd, cams=cams, log=log)

    def new_logger(self, seed=None):
        a = self.a
        name = a.name or (f'batch{a.batch}' if a.batch else '')
        rid = time.strftime('%H%M%S') + ('_dry' if a.dry_run else '') + (f'_s{seed}' if seed is not None else '') + (f'_{name}' if name else '')
        d = Path(a.out) / time.strftime('%Y-%m-%d') / f'{a.stage}_{rid}'
        meta = dict(stage=a.stage, dry_run=a.dry_run, seed=seed, args=vars(a), git=git_state(), config=self.cfg,
                    bundle_manifest=self.bundle_manifest, bundle_verified=True, config_matches_bundle=self.config_matches_bundle, chain_constants=self.consts,
                    calibration=calib_versions(), operator_notes=a.notes,
                    pc_spot=json.loads(a.pc_spot) if a.pc_spot else None,
                    host=__import__('socket').gethostname())
        return RunLogger(d, meta)

    def make_chain(self, log):
        from deploy.bimanual.chain import RealChain
        g = Guards(self.cfg, self.scene.kp.cpu().numpy(), self.cfg['control_dt'], self.a.max_steps,
                   estop_fn=self.io.stopped, stop_key=self.keys)
        return RealChain(self.cfg, self.io, self.scene, self.consts, self.gmaps, log=log, keys=self.keys, guards=g,
                         speed=self.a.speed, hold_force=self.a.hold_force, device=str(self.dev),
                         policy_path=self.policy_path, auto=self.auto, net_device=self.a.net_device)

    # ------------------------------------------------------------------ prompts
    def confirm(self, ch, what):
        ch.log_event('prompt', what=what)
        ans = ch.ask(f'\n[{what}] Enter = go, s = skip the rest, q = abort >', '')
        ch.log_event('answer', what=what, answer=ans)
        if ans.lower().startswith('q'):
            raise GuardStop(f'operator abort before {what}')
        return not ans.lower().startswith('s')

    def confirm_in_press(self, ch, what):
        """Prompt while the left arm presses: a watchdog holds both arms (press released) at hold_max_s."""
        if not self.a.pause_in_press or self.a.dry_run:
            return True
        left = self.cfg['safety']['hold_max_s'] - ch.guards.hold_ticks * ch.dt
        timer = threading.Timer(max(left, 1.0), lambda: (print('\nwatchdog: press time over, holding both arms', flush=True), self.io.hold()))
        timer.start()
        try:
            return self.confirm(ch, what)
        finally:
            timer.cancel()

    # ------------------------------------------------------------------ one episode
    def episode(self, seed=None):
        a = self.a
        log = self.new_logger(seed)
        if self.io is None:
            self.open_io(log)
        if a.dry_run:
            truth0 = self.io.reset(seed)
            log.update_meta(sim_truth0=truth0)
            if not a.pc_spot:      # the dry run's "measured spot" is the sim truth (exporter labels)
                R = truth0['pc_R']
                log.update_meta(pc_spot={'port_xy': truth0['pc_p'][:2].tolist(), 'yaw_deg': math.degrees(math.atan2(-R[1, 2], -R[0, 2])),
                                         'method': 'sim truth (dry run)', 'unc_mm': 0.0})
        ch = self.make_chain(log)
        ch.seed = seed or 0
        ch.pick_test = a.pick_test
        res = {}
        out = dict(stage=a.stage, completed=False)
        try:
            self.io.start_clock()
            res = self.run_stages(ch, res)
            out['completed'] = True
        except Exception as e:  # noqa: BLE001  (GuardStop, SafetyStop, KeyboardInterrupt via StopKey, bugs)
            out['stop'] = f'{type(e).__name__}: {e}'
            print(f'\n*** STOPPED: {out["stop"]}', flush=True)
            ch.log_event('stopped', reason=out['stop'], stage=ch.label, tick=ch.k)
            self.io.hold()
            if not a.dry_run:
                self.recover(ch)
            elif not isinstance(e, GuardStop):
                import traceback
                traceback.print_exc()
        finally:
            out.update(results=res, ticks=ch.k)
            tr = ch.truth()
            if tr is not None:
                out['sim_truth_end'] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in tr.items() if k in ('seated', 'tip_depth_mm', 'finite', 'plug_in_port_mm')}
            label = self.outcome_prompt(ch, out)
            log.close(result=out, outcome=label)
            print(f'log: {log.dir}', flush=True)
        return out

    def recover(self, ch):
        if self.keys is not None:
            self.keys.clear()
        while True:
            ans = ch.ask('Both arms hold their measured pose. u = unload left press, h = return home (slow, checked), '
                         'r = unload + home, x = stay holding >', 'x')
            try:
                if ans in ('u', 'r'):
                    ch.guards.reset_counts()
                    ch.unload_left()
                if ans in ('h', 'r'):
                    ch.guards.reset_counts()
                    ch.home()
                return
            except Exception as e:  # noqa: BLE001
                print(f'recovery step stopped: {type(e).__name__}: {e}', flush=True)
                self.io.hold()
                if self.keys is not None:
                    self.keys.clear()

    def outcome_prompt(self, ch, out):
        if self.a.dry_run:
            tr = out.get('sim_truth_end') or {}
            return {'label': 'seated' if tr.get('seated') else 'not_seated', 'source': 'sim truth', 'failure_labels': ''}
        lab = ch.ask('Outcome? s = seated / inserted, p = partly, f = failed, a = aborted (+ Enter) >', '')
        txt = ch.ask('Failure labels / what you saw (free text, Enter to skip) >', '')
        return {'label': {'s': 'seated', 'p': 'partial', 'f': 'failed', 'a': 'aborted'}.get(lab.strip()[:1], lab), 'failure_labels': txt,
                'source': 'operator'}

    # ------------------------------------------------------------------ stages
    def run_stages(self, ch, res):
        a, st = self.a, self.a.stage
        log = {}
        res['chain_log'] = log
        if not self.confirm(ch, 'capture: both arms to the ZED capture pose, grippers fully open'):
            return res
        ep, eR, ok = ch.capture_pc(log)
        print(f'ZED PC estimate: port {np.round(log["zed"]["port_xy"], 4)} m, yaw {log["zed"]["yaw_deg"]:.1f} deg, gate {ok}', flush=True)
        if not ok and not a.dry_run:
            if ch.ask('PC estimate REJECTED by the gate. r = retake, c = continue anyway (logged), Enter = stop >', '') == 'r':
                ep, eR, ok = ch.capture_pc(log)
        if not ok and st != 'capture':
            if a.dry_run or ch.ask('still rejected: c = continue anyway >', '') != 'c':
                raise GuardStop('ZED PC estimate rejected')
        res['zed_ok'] = ok
        if st == 'capture':
            ch.park_left()
            from deploy.bimanual import netcache as NC
            zr = NC.zed_reader(self.cfg['pick']['reader_estimator'], ch.dev, self.cfg['pick']['reader_retries'])
            from deploy.bimanual.chain import RealPickTask
            ch.pick = RealPickTask(ch, 1.0)
            p, R, okr = zr(ch)
            res['reader'] = dict(p=p[0].tolist(), yaw_deg=math.degrees(float(torch.atan2(-R[0, 1, 2], -R[0, 0, 2]))), ok=bool(okr[0]))
            ch.log_event('estimate', est='zed_reader', **res['reader'])
            print(f'reader estimate: {res["reader"]}', flush=True)
            self.finish(ch)
            return res
        ch.park_left()
        if st in ('pick', 'full', 'collect'):
            if not self.confirm(ch, 'pick: servo-descent pick of the reader'):
                return res
            q_lift, ok = ch.stage_pick(log, a.pick_mode)
            res['pick_ok'] = ok
            print(f'pick: done {log.get("pick_done")}, reader present (gripper) {log.get("pick_reader_present")}', flush=True)
            if a.pick_test:
                self.finish(ch)
                return res
            if not ok:
                raise GuardStop('pick failed (no grasp or no reader between the fingers)')
            if st == 'pick':
                self.release_by_hand(ch)
                self.finish(ch)
                return res
        elif st in ('hold',):
            q_lift = ch.q_b[0].clone()
        else:   # approach_insert: reader placed by hand
            q_lift = self.hand_place(ch)
        if st == 'hold':
            if not self.confirm(ch, f'hold: left guarded press, {ch.a.hold_force:.0f} N'):
                return res
            q_hold = ch.stage_hold(q_lift[None] if q_lift.dim() == 1 else q_lift, ep, eR, log)
            for _ in range(int(round(a.hold_seconds / ch.dt))):
                ch.hstep(q_lift.reshape(6), q_hold.reshape(6), '2 hold')
            res['hold'] = {k: log[k] for k in log if k.startswith('hold')}
            ch.unload_left()
            self.finish(ch)
            return res
        if st == 'collect':
            from deploy.bimanual.collect import run_collect
            res['collect'] = run_collect(self, ch, q_lift.reshape(1, 6), ep, eR, log)
            print(f'collect: {res["collect"]}', flush=True)
            self.finish(ch)
            return res
        if not self.confirm(ch, f'hold -> transport -> approach -> insertion -> release (runs through; left press {ch.a.hold_force:.0f} N)'):
            return res
        qr = q_lift.reshape(1, 6)
        q_hold = ch.stage_hold(qr, ep, eR, log)
        if not bool(ch.hold_obj.latched[0]):
            raise GuardStop('left hold did not latch (no contact force within the search depth)')
        if not self.confirm_in_press(ch, 'transport + approach'):
            raise GuardStop('operator stop in the press')
        from deploy.bimanual.align import align_cfg, run_align
        ac = align_cfg(self.cfg)
        q_arm = q_hold
        if ac is not None and ac['press_release']:     # no press force and no press clock until the plug is lined up (as stage collect): the hand stays on the PC
            for _ in range(50):
                q_arm = ch.q_b[1].clone()[None]
                ch.hstep(qr.reshape(6), q_arm.reshape(6), '2 hold')
            ch.pressing = False
            ch.log_event('align_press_off', q_left=q_arm.tolist(), tick=ch.k, at='after the latch')
        ch.rt.comp['right'] = ch.fric
        q_look = ch.stage_transport(qr, q_arm, ep, eR, log)
        cmd = ch.stage_approach(q_look, q_arm, ep, eR, log)
        print(f'in-hand estimate (mm, deg): {np.round(ch.inhand_est6[0].cpu().numpy(), 2)}', flush=True)
        lat = ch.preinsert_lateral_mm(cmd)
        res['preinsert_fk_lateral_mm'] = lat
        lim = self.cfg['insert'].get('wrong_port_mm')
        ch.log_event('preinsert_check', fk_lateral_mm=lat, limit_mm=lim)
        print(f'pre-insert: plug {lat:.1f} mm across the ZED port axis (encoder FK; log only unless wrong_port_mm is set)', flush=True)
        if lim is not None and lat > lim:
            raise GuardStop(f'approach ended {lat:.1f} mm across the ZED port axis (port pitch 18.6 mm): neighbouring port?')
        if ac is not None:
            ar = run_align(ch, cmd, q_hold, ep, eR, log, ac)
            res['align'] = {k: v for k, v in ar.items() if k not in ('cmd', 'q_left')}
            if ar['cmd'] is None:          # stop or no insertion wanted: the plug is out, jaws closed, never handed to RL
                self.end_no_insert(ch, ar['q_left'], q_hold)
                return res
            cmd = ar['cmd']
        if not self.confirm_in_press(ch, 'insertion'):
            raise GuardStop('operator stop in the press')
        sim_obs = self.sim_obs_fn() if (a.dry_run and a.check_obs) else None
        from deploy.bimanual.insert import run_insertion
        r = run_insertion(ch, cmd, q_hold, log, sim_obs=sim_obs)
        res['insert'] = r
        print(f'insertion: {r}', flush=True)
        if self.cfg['insert'].get('keep_reader_unless_seated') and r['outcome'] != 'seated_detected':
            # real run 2026-11-02 A: the time limit opened the jaws in front of the port and the reader fell. Not seated: out with closed jaws
            from deploy.bimanual.align import back_off_closed
            ic = self.cfg['insert']
            back_off_closed(ch, eR[0].cpu().numpy(), float(ic.get('abort_lift_mm', 20.)), float(ic.get('abort_v_mm_s', 10.)), q_hold.reshape(6))
            self.end_no_insert(ch, None, q_hold)
            return res
        ch.retreat_right(eR)
        ch.unload_left()
        res['released'] = True
        tr = ch.truth()
        if tr is not None:
            res['true_seated_after_release'] = bool(tr['seated'])
        self.finish(ch)
        return res

    def end_no_insert(self, ch, q_left, q_hold):
        """No insertion or no seat: left press off (if on) and lifted, reader to the operator's hand, home."""
        ch.unload_left()
        res = ch.truth()
        if res is not None:
            ch.log_event('no_insert_truth', plug_in_port_mm=res['plug_in_port_mm'], seated=res['seated'])
        if not self.a.dry_run:
            self.release_by_hand(ch)
        self.finish(ch)

    def sim_obs_fn(self):
        env, h = self.io.ch.env, self.io.h

        def f(jt, prev, inhand):
            env.enc_error.copy_(h.enc_error)
            env.joint_target[0] = jt.to(env.joint_target.device)
            env.previous_action[0] = prev.to(env.previous_action.device)
            env.inhand_est[0] = inhand.to(env.inhand_est.device)
            return env._actor_obs().to(jt.device)
        return f

    def hand_place(self, ch):
        """approach_insert: right arm to the nominal post-pick lift pose, jaws at the 24 mm approach opening; the
        operator puts the reader in (Part 1 grasp: fingers on the two side faces near the body front, reader flat,
        plug pointing away from the right base like a picked reader); close; return the lift pose."""
        q_lift = ch_nominal_lift(ch)
        g_open = ch.grip_cmd('right', 1.)
        ch.move({'right': q_lift}, grip={'right': g_open}, label='1 pick')
        if self.a.dry_run:
            self.io.place_reader_in_hand() if hasattr(self.io, 'place_reader_in_hand') else None
        ch.ask('Put the reader between the open jaws (plug out, flat, fingers on the side faces near the body front). '
               'Keep fingers clear. Enter = close the gripper >', '')
        for _ in range(int(round(1.5 / ch.dt))):
            ch.hstep(torch.as_tensor(q_lift, dtype=torch.float32), ch.q_b[1].clone(), '1 pick', grip_r=-1., grip_l=1.)
        present = ch.reader_in_hand()
        ch.log_event('hand_placed', present=present, grip=float(ch.state['grip'][0]))
        if present is False and ch.ask('gripper reading says NO reader in hand. c = continue anyway >', '') != 'c':
            raise GuardStop('no reader in hand')
        return ch.q_b[0].clone()

    def release_by_hand(self, ch):
        ch.ask('Reader lifted. Hold it, then Enter opens the gripper >', '')
        for _ in range(int(round(1.0 / ch.dt))):
            ch.hstep(ch.q_b[0].clone(), ch.q_b[1].clone(), '6 release', grip_r=1., grip_l=1.)

    def finish(self, ch):
        if not self.a.no_home:
            if self.a.dry_run:
                return
            if self.confirm(ch, 'home: return both arms to the start pose (left first)'):
                ch.home()


def ch_nominal_lift(ch):
    """Believed right-arm lift pose of a nominal pick (reader at the ZED net's strip centre and mean yaw)."""
    from blind_insertion.envs.bimanual_pick import PickExpert
    from blind_insertion.perception.reader_zed import reader_pose
    from deploy.bimanual.chain import RealPickTask
    ck = torch.load(ROOT / 'runs/perception' / f'{ch.cfg["pick"]["reader_estimator"]}.pt', map_location='cpu')
    task = RealPickTask(ch, 1.0)
    ch.pick = task
    p, R = reader_pose(task, torch.tensor([ck['strip_centre']], dtype=torch.float32, device=ch.dev),
                       torch.tensor([ck['yaw0']], dtype=torch.float32, device=ch.dev))
    task.est = (p, R)
    ex = PickExpert(task, reader_pose_fn=lambda: (p, R))
    ex.believed_frame = True
    ex.reset()
    q, pe, re = ch.scene.tcp_chain.ik(ex.p_lift, ex.R_g, ex.q_pre, iters=150)
    if float(pe[0]) > 1e-3:
        raise GuardStop(f'nominal lift pose IK failed ({float(pe[0]) * 1e3:.1f} mm)')
    return q[0].cpu().numpy().astype(float)


def main(argv=None):
    a = parse(argv)
    torch.set_num_threads(2)       # one-world kinematics: thread fan-out costs more than it saves
    r = Runner(a)
    outs = []
    try:
        seeds = [a.seed + i for i in range(a.episodes)] if a.dry_run else [None]
        for s in seeds:
            outs.append(r.episode(s))
            if a.dry_run:
                o = outs[-1]
                print(f'seed {s}: completed {o["completed"]} stop {o.get("stop")} insert {o["results"].get("insert", {}).get("outcome")} '
                      f'true seated after release {o["results"].get("true_seated_after_release")}', flush=True)
    finally:
        if r.io is not None:
            r.io.close(read_line=(lambda p: r.keys.ask(p)) if r.keys is not None else input)
    return outs


if __name__ == '__main__':
    main()
