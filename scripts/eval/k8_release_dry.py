"""Release test in the sim dry run (docs/real/K8_20261115.md, task B). Sim only; the deployed chain does not change.

deploy.run_bimanual --dry-run with sim-side options that exist only in this process (all off by default):
  --plug tight|loose    the sim plug outline (contact_sens.PLUGS: tight 20 / 50 um, loose 90 / 110 um per side; default nominal 45 / 80)
  --port-mu MU          the sim plug-port friction (contact_dr port_friction = [MU, MU]; default: InsertionCfg U(0.10, 0.20))
  --release-open-m X    the sim right jaws open to X (finger joint limit) from the release on (default 0.024, PickTaskCfg.finger_open)
and a truth trace of every control tick from the seat detection to the end (tip depth, plug lateral, sim seat test, jaw command):
<out>/<run>/release_trace.json. All other arguments go to deploy.run_bimanual.

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl PYTHONPATH=. python scripts/eval/k8_release_dry.py --plug tight --port-mu 0.3 --tag tight_mu03 -- \
        --config deploy/config/bimanual_align10_all.json --stage full --align-mode full --dry-run --seed 7041 --out runs/k8_eval/dry
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TRACE = []
TRACE_STATE = {}


def _fingers(ch):
    """Right finger joint positions (mm) of the sim world (opening per finger)."""
    rt = getattr(ch.io, 'rt', None)
    if rt is None:
        return None
    return [round(float(rt.q[0, a]) * 1e3, 3) for a in rt.finger_qadr_right]


_HAND = {}


def _hand_contacts(ch):
    """Names of the right-hand geoms touching the reader (sim world 0): fingers, pads, gripper body (descendants of link_6)."""
    import mujoco
    import warp as wp
    rt = getattr(ch.io, 'rt', None)
    if rt is None:
        return None
    m = rt.mjm
    if not _HAND:
        root = m.body('link_6').id
        def under(b):
            while b > 0:
                if b == root:
                    return True
                b = int(m.body_parentid[b])
            return False
        _HAND['hand'] = {g for g in range(m.ngeom) if under(int(m.geom_bodyid[g]))}
        _HAND['peg'] = set(int(g) for g in rt.lay.peg_slots)
    nc = int(rt.d.nacon.numpy()[0])
    if not nc:
        return []
    g = wp.to_torch(rt.d.contact.geom)[:nc].cpu().numpy()
    out = set()
    nm = lambda x: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(x)) or f'g{x}') + '@' + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[x])) or '')   # noqa: E731
    for a, b in g:
        if (a in _HAND['hand']) != (b in _HAND['hand']):      # hand geom against any geom outside the hand
            h_, o_ = (a, b) if a in _HAND['hand'] else (b, a)
            out.add(nm(h_) + ' | ' + nm(o_))
    return sorted(out)


def main():
    argv = sys.argv[1:]
    cut = argv.index('--') if '--' in argv else len(argv)
    ap = argparse.ArgumentParser()
    ap.add_argument('--plug', default='nom', choices=['nom', 'tight', 'loose'])
    ap.add_argument('--port-mu', type=float, default=None)
    ap.add_argument('--release-open-m', type=float, default=None,
                    help='sim right finger upper joint limit (m) from the first release tick (default: PickTaskCfg.finger_open 0.024)')
    ap.add_argument('--tag', default='')
    ap.add_argument('--summary', default=str(ROOT / 'results/real_align/k8/release_dry.json'))
    A = ap.parse_args(argv[:cut])
    rest = argv[cut + 1:]
    if A.plug != 'nom':
        from scripts.eval.contact_sens import patch_scene
        patch_scene('4x1.2', A.plug, batch_solimp=False)
    if A.port_mu is not None:
        from blind_insertion.envs.insertion_task import DEFAULT_CONTACT_DR
        cdr = dict(DEFAULT_CONTACT_DR, port_friction=[A.port_mu, A.port_mu])
        cdr = {k: list(v) if isinstance(v, tuple) else v for k, v in cdr.items()}
        cfg = json.loads(Path(rest[rest.index('--config') + 1]).read_text())
        phys = json.loads((cfg.get('sim') or {}).get('phys_json') or '{}')
        rest += ['--sim-phys', json.dumps(dict(phys, contact_dr=cdr))]
    from deploy.bimanual import chain as CH
    orig = CH.RealChain.hstep

    def hstep(self, qr, ql, label, grip_r=-1., grip_l=-1.):
        if A.release_open_m is not None and label.startswith('6 release') and not TRACE_STATE.get('widened'):
            import warp as wp
            rt = self.io.rt
            jr = wp.to_torch(rt.m.jnt_range)
            for f in ('left', 'right'):
                jid = rt.mjm.joint(f + '_finger').id
                (jr[:, jid, 1] if jr.dim() == 3 else jr[jid, 1:2]).fill_(A.release_open_m)
            TRACE_STATE['widened'] = True
        out = orig(self, qr, ql, label, grip_r=grip_r, grip_l=grip_l)
        if label.startswith(('5 insert', '6 release')) or label.startswith('push'):
            tr = self.truth()
            if tr is not None:
                TRACE.append(dict(k=int(self.k), label=label, grip_r=float(grip_r), depth_mm=round(tr['tip_depth_mm'], 4),
                                  lat_mm=[round(float(x), 4) for x in tr['plug_in_port_mm'][:2]], seated=tr['seated'], fingers_mm=_fingers(self), hand_contacts=_hand_contacts(self)))
        return out
    CH.RealChain.hstep = hstep
    from deploy import run_bimanual as RB
    outs = RB.main(rest)
    summ = []
    for o in outs:
        res = o['results']
        ins = res.get('insert', {})
        rel = [t for t in TRACE if t['label'].startswith('6 release')]
        before = [t for t in TRACE if t['label'].startswith('5 insert')]
        row = dict(tag=A.tag, plug=A.plug, port_mu=A.port_mu, release_open_m=A.release_open_m, argv=rest, completed=o.get('completed'), stop=o.get('stop'),
                   insert=ins.get('outcome'), t_seat_s=ins.get('t_seat_s'),
                   true_seated_after_release=res.get('true_seated_after_release'))
        if before:
            row['depth_end_insert_mm'] = before[-1]['depth_mm']; row['seated_end_insert'] = before[-1]['seated']
            row['depth_max_insert_mm'] = max(t['depth_mm'] for t in before)
        if rel:
            opn = [t for t in rel if t['grip_r'] > 0]
            row['depth_release_start_mm'] = rel[0]['depth_mm']
            row['depth_after_open_mm'] = opn[min(len(opn) - 1, 19)]['depth_mm'] if opn else None   # 0.4 s open (release_open_s)
            row['depth_after_retreat_mm'] = opn[-1]['depth_mm'] if opn else None
            row['depth_end_mm'] = rel[-1]['depth_mm']
            row['lat_end_mm'] = rel[-1]['lat_mm']
            row['seated_end'] = rel[-1]['seated']
            row['first_unseated_k'] = next((t['k'] - rel[0]['k'] for t in rel if not t['seated']), None)
        summ.append(row)
        print('[k8_release_dry]', json.dumps(row), flush=True)
    run_dir = Path(rest[rest.index('--out') + 1]) if '--out' in rest else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f'release_trace_{A.tag}.json').write_text(json.dumps(TRACE))
    p = Path(A.summary); p.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(p.read_text()) if p.exists() else []
    p.write_text(json.dumps(done + summ, indent=1))


if __name__ == '__main__':
    main()
