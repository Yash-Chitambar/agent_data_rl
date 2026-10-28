"""Task H on one arm: ACT pick -> handoff -> blind insertion -> release, one 50 Hz loop.

The tick order mirrors blind_insertion/pipeline/pick_insert.run (sim) step for step:

  1. read encoders, gripper, (pick) camera frames; q_model = q_motor + joint offsets
  2. insert phase: push the previous action + this observation; handoff.ReleaseTrigger (5 s, or
     60 mm of tcp travel) -> release
  3. pick phase: ACT (temporal ensembling) -> absolute target, gripper; handoff.HandoffGate or the pick
     time limit -> insert (history filled with this observation, the same tick acts)
  4. insert phase: target = q_motor + scale * a (offsets cancel), gripper keeps squeezing
  5. release phase: hold, open, back the tcp out along its approach axis
  6. send (SafeArm: joint box, table clearances, step cap, friction feed-forward)

ACT targets are absolute: each tick's change is capped at act.max_step_rad from the last sent target,
which also feeds ACT's cmd_err state. The switch to the relative insertion actions needs no special
case: the last ACT target sits ~20 mrad ahead of the arm (cmd_err), the first insertion target within
scale * clip of it.

    .venv/bin/python -m deploy.runner sim --episodes 16 --name dry          # full dry run (GPU)
    .venv/bin/python -m deploy.runner real --name pick1 --until pick        # real arm, pick only (estop.py first)
    .venv/bin/python -m deploy.runner real --name try1                      # real arm, full task
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from blind_insertion.pipeline import handoff as H
from deploy import conventions as C
from deploy.stages import ActStage, InsertStage, ReleaseStage

HERE = Path(__file__).resolve().parent
PICK, INSERT, RELEASE, DONE = 0, 1, 2, 3
PHASES = ("pick", "insert", "release", "done")


def _strip(d):
    """Drop the "_comment" keys of the config file."""
    return {k: _strip(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")} if isinstance(d, dict) else d


def load_config(path=HERE / "config/task_h.json"):
    cfg = _strip(json.loads(Path(path).read_text()))
    for k, v in cfg["models"].items():
        if isinstance(v, str) and not Path(v).is_absolute():
            cfg["models"][k] = str(C.REPO / v)
    return cfg


class TaskH:
    def __init__(self, cfg, device):
        self.cfg, self.dt = cfg, cfg["control_dt"]
        self.robot = C.Robot("cpu")
        self.off = C.joint_offsets(cfg["side"])
        g = cfg["gripper"]
        self.gmap = C.GripperMap(g["open"], g["close"], g["m_open"], g.get("g_reader"), g.get("m_reader", 0.019))
        self.act = ActStage(cfg["models"]["act"], device, cfg["act"]["ensemble"])
        self.ins = InsertStage(cfg["models"]["insert"], cfg["models"]["insert_config"], self.robot)
        self.rel = ReleaseStage(self.robot, self.dt, **cfg["release"])
        h = cfg["handoff"]
        self.gate = H.HandoffGate(h["p_nom"], self.dt, **{k: v for k, v in h.items() if k != "p_nom"})
        self.seat = H.ReleaseTrigger(self.dt, **cfg["release_trigger"])
        self.q_ready = np.asarray(cfg["act"]["start_q_model"], float)

    def episode(self, be, log=None, until="all"):
        """One attempt on backend `be`. Returns a summary; per-tick rows go to `log` (list).
        until="pick": stop and hold at the handoff (grasp + pre-insert pose, no insertion)."""
        cfg = self.cfg
        T_pick, T_ins = int(cfg["act"]["pick_s"] / self.dt), int(cfg["insert"]["insert_s"] / self.dt)
        max_step = cfg["act"]["max_step_rad"]
        be.move_to(self.q_ready - self.off, grip01=self.gmap.open, settle_trim=True)
        self.act.reset()
        self.gate.reset(1, "cpu")
        self.seat.reset(1, "cpu")
        phase, t_hand, t_rel, t_ins, a_ins, stop = PICK, None, None, 0, None, None
        seated_at_release = None
        last = None
        t_next = be.clock()
        k = 0
        for k in range(T_pick + T_ins + self.rel.T + 2):
            if be.stopped():
                stop = "software stop (estop.py)"
                break
            t_tick = be.clock()
            q_mot, qd, g, t_obs = be.read()
            q = q_mot + self.off
            finger = self.gmap.finger_m(g)
            p, _, tw = self.robot.fk(q, qd)
            row = dict(k=k, t=t_tick, phase=phase, q=q.tolist(), qd=qd.tolist(), grip=g, finger_mm=finger * 1e3,
                       tcp_mm=(p.numpy() * 1e3).round(2).tolist())
            last = last if last is not None else q.copy()
            if phase == INSERT:
                self.ins.push(a_ins, q, qd)
                t_ins += 1
                on = torch.ones(1, dtype=torch.bool)
                if bool(self.seat(on, p[None])[0]) or t_ins >= T_ins:
                    phase, t_rel = RELEASE, k
                    row["release"] = "time" if t_ins * self.dt >= self.seat.after_s - 1e-9 else "travel"
                    tr = be.truth()
                    seated_at_release = None if tr is None else tr["seated"]
                    self.rel.start(q)
            if phase == PICK:
                fr, age, _ = be.frames()
                t0 = time.perf_counter()
                a = self.act.step(q, finger, fr["top"], fr["wrist"])
                row.update(act_ms=(time.perf_counter() - t0) * 1e3, age_ms={c: x * 1e3 for c, x in age.items()},
                           done_out=float(a[7]))
                closed_with_reader = not a[6] > 0
                if be.truth() is None:
                    from deploy.grasp import reader_present
                    gc = cfg["gripper"]
                    closed_with_reader &= reader_present(g, gc["g_empty"], gc["g_reader"])
                    row["reader_present_encoder"] = bool(closed_with_reader)
                go = bool(self.gate(torch.tensor([a[7]]), torch.tensor([closed_with_reader]), torch.tensor([finger]),
                                    torch.as_tensor(qd[None], dtype=torch.float32), p[None])[0])
                if not go and k >= T_pick - 1 and be.truth() is None:
                    stop = "pickup timeout without a valid grasp handoff"
                    if log is not None:
                        log.append(row)
                    break
                if go or k >= T_pick - 1:
                    phase, t_hand = INSERT, k
                    row["handoff"] = "gate" if go else "time limit"
                    if until == "pick":
                        stop = f"until pick: holding at the handoff ({row['handoff']})"
                        if log is not None:
                            log.append(row)
                        break
                    self.ins.start(q, qd)
                    self.seat.start(torch.zeros(1, dtype=torch.long), p[None])
                else:
                    tgt = last + np.clip(a[:6] - last, -max_step, max_step)
                    grip = self.gmap.command(a[6])
                    self.act.sent(tgt)
            if phase == INSERT:
                a_ins = self.ins.act()
                tgt, grip = q + self.ins.scale * a_ins, self.gmap.close
                row["a_ins"] = a_ins.tolist()
            if phase == RELEASE:
                tgt, rel_done = self.rel.act()
                grip = self.gmap.open
                if rel_done:
                    phase = DONE
            if phase == DONE:
                break
            row["target"] = tgt.tolist()
            try:
                be.send(tgt - self.off, grip)
            except Exception as e:  # noqa: BLE001   (safety.SafetyStop: the arm holds)
                stop = f"{type(e).__name__}: {e}"
                break
            last = tgt
            if log is not None:
                tr = be.truth()
                if tr is not None:
                    row["truth"] = tr
                log.append(row)
            t_next += self.dt
            be.wait(t_next)
        be.hold()
        out = dict(phase=PHASES[phase], ticks=k, stop=stop, t_hand_s=None if t_hand is None else t_hand * self.dt,
                   t_release_s=None if t_rel is None else t_rel * self.dt)
        tr = be.truth()
        if tr is not None:
            out["truth_end"] = tr
            out["seated_at_release"] = seated_at_release
        return out


def cmd_sim(a):
    cfg = load_config(a.config)
    from deploy.backends import SimBackend
    from deploy.safety_task import FlexChecker
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    task = TaskH(cfg, dev)
    out_dir = C.REPO / "data/task_h" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    res = []
    for e in range(a.episodes):
        sc = cfg["safety"]
        chk = FlexChecker(task.off, sc["clearance"], sc["tip_clearance"], sc.get("body_clearance"), sc.get("table_z", 0.0))
        be = SimBackend(cfg, task.off, task.gmap, seed=a.seed + e, device=dev, phys_json=json.loads(a.phys_json), checker=chk)
        # the sim episode starts at the ready pose + noise (PickCfg.start_q_noise), as in training
        log = []
        r = task.episode(be, log, until=a.until)
        r["seed"] = a.seed + e
        r["safety_violations"] = be.violations[:20]
        r["n_safety_violations"] = len(be.violations)
        low = [x["truth"]["low_mm"] for x in log if "truth" in x]
        r["low_mm"] = {b: min(x[b] for x in low) for b in low[0]} if low else None
        res.append(r)
        np.save(out_dir / f"ep{e:03d}.npy", np.array(log, dtype=object), allow_pickle=True)
        te = r.get("truth_end", {})
        print(f"episode {e}: {r['phase']}, handoff {r['t_hand_s']} s, release {r['t_release_s']} s, "
              f"seated at end {te.get('seated')}, in hand {te.get('in_hand')}, safety violations {len(be.violations)}"
              f"{' (first: ' + be.violations[0][1] + ')' if be.violations else ''}, lowest fingers "
              f"{min(r['low_mm']['left_finger'], r['low_mm']['right_finger']) if r['low_mm'] else None} mm", flush=True)
        be.close()
    ok = [bool(r.get("truth_end", {}).get("seated")) and not r["truth_end"]["in_hand"] for r in res]
    summ = dict(episodes=len(res), seated_after_release=float(np.mean(ok)), config=cfg, results=res)
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(f"seated after release: {sum(ok)}/{len(ok)}")


def cmd_real(a):
    cfg = load_config(a.config)
    from deploy import models
    from deploy.backends import RealBackend
    import estop
    if models.verify(None):
        raise SystemExit("models do not match deploy/models/MANIFEST.json")
    if cfg["gripper"].get("g_reader") is None and not a.uncal_gripper:
        raise SystemExit("gripper map is uncalibrated (gripper.g_reader null): run deploy.checks.gripper_cal, "
                         "or pass --uncal_gripper")
    if estop.stopped():
        raise SystemExit(f"{estop.STOP_FILE} is set: clear it in estop.py first")
    g = cfg["gripper"]
    from deploy.grasp import reader_present
    if g.get("g_empty") is None or not reader_present(g["g_reader"], g["g_empty"], g["g_reader"]):
        raise SystemExit("invalid real grasp calibration")
    if not g.get("verified_reader_gate", False):
        raise SystemExit("real empty/held-reader grasp gate validation is pending")
    if not cfg.get("verified_start_envelope", False):
        raise SystemExit("measured starting-envelope validation is pending")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    task = TaskH(cfg, dev)
    torch.set_num_threads(2)
    be = RealBackend(cfg, task.off)
    out_dir = C.REPO / "data/task_h" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    try:
        r = task.episode(be, log, until=a.until)
    finally:
        be.hold()
        np.save(out_dir / "ticks.npy", np.array(log, dtype=object), allow_pickle=True)
        from deploy.recovery import supported_shutdown
        supported_shutdown(be)
    ms = [x["act_ms"] for x in log if "act_ms" in x]
    ages = [x["age_ms"] for x in log if "age_ms" in x]
    r["timing"] = dict(act_ms_p50=float(np.median(ms)) if ms else None, act_ms_max=float(np.max(ms)) if ms else None,
                       age_top_ms_p50=float(np.median([x["top"] for x in ages])) if ages else None,
                       age_wrist_ms_p50=float(np.median([x["wrist"] for x in ages])) if ages else None,
                       period_ms_p99=float(np.percentile(np.diff([x["t"] for x in log]), 99) * 1e3) if len(log) > 2 else None)
    (out_dir / "summary.json").write_text(json.dumps(dict(config=cfg, result=r), indent=1, default=str))
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim")
    s.add_argument("--episodes", type=int, default=16)
    s.add_argument("--seed", type=int, default=700)
    s.add_argument("--phys_json", default='{"enc_offset_deg": [0, 0.3, 0.3, 0.3, 0.3, 0]}')
    r = sub.add_parser("real")
    r.add_argument("--uncal_gripper", action="store_true", help="run with the proportional gripper map")
    for p in (s, r):
        p.add_argument("--until", choices=["pick", "all"], default="all", help="pick: stop and hold at the handoff")
        p.add_argument("--config", default=str(HERE / "config/task_h.json"))
        p.add_argument("--name", required=True)
    a = ap.parse_args()
    {"sim": cmd_sim, "real": cmd_real}[a.cmd](a)
