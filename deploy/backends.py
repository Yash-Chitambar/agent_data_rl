"""Arm + camera backends for deploy/runner.py. Both speak the motor joint frame and i2rt's gripper
range (0 closed .. 1 open), so the runner code is the same on the real arm and in the dry run.

  RealBackend  scripts/real/deploy.RealArm (sim gains, kd capped at 5, CAN timing hooks, SafeArm with the
               Task H clearances (deploy/safety_task.FlexChecker) and the friction feed-forward) + deploy/cameras.Cameras.
  SimBackend   one PickEnv world (MuJoCo Warp, the pipeline's randomization). Its "motor" reading is the
               sim encoder reading - rig.joint_offsets, its gripper reading the inverse finger map, its
               frames follow the demos' camera clock (as pipeline/pick_insert.ActPick._frames).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "scripts/real"))   # after REPO: its deploy.py must not shadow this package


def real_loop():
    """scripts/real/deploy.py (the 50 Hz loop: RealArm, gains, timing hooks), loaded by path: the name
    `deploy` is this package."""
    import importlib.util
    if "real_deploy" not in sys.modules:
        spec = importlib.util.spec_from_file_location("real_deploy", REPO / "scripts/real/deploy.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["real_deploy"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["real_deploy"]


class RealBackend:
    def __init__(self, cfg, offsets):
        import safety
        from blind_insertion.envs import friction_comp as FC
        from deploy.cameras import Cameras
        from deploy.safety_task import FlexChecker
        D = real_loop()
        self.D, self.safety = D, safety
        D.FRIC_COMP = FC.parse(cfg["fric_comp"])
        kd = np.minimum(D.SIM_KD, D.DM_KD_MAX)
        # Complete potentially blocking SDK initialization before motor control starts.
        self.cams = Cameras(cfg["side"], cfg["cameras"]["res"], cfg["cameras"]["max_age_ms"] / 1e3)
        self.arm = D.RealArm(cfg["side"], D.SIM_KP, kd)
        sc = cfg["safety"]
        chk = FlexChecker(offsets, clearance=sc["clearance"], tip_clearance=sc["tip_clearance"], body_clearance=sc.get("body_clearance"))
        self.arm.safe = safety.SafeArm(self.arm.robot, chk)    # the move_to path checks use it too
        self.arm.safe._hold(np.asarray(self.arm.robot.get_joint_pos(), float))
        self.off = offsets
        self.control_dt = D.CONTROL_DT
        self.k = 0

    def read(self):
        q, qd, t_obs, _ = self.arm.read()
        return q[:6].copy(), qd[:6].copy(), float(q[6]), t_obs

    def frames(self):
        return self.cams.frames()

    def send(self, q6_motor, grip01):
        self.arm.send(np.append(q6_motor, grip01), self.k)
        self.k += 1

    def clock(self):
        return time.monotonic()

    def wait(self, t_next):
        self.arm.wait(t_next)

    def move_to(self, q6_motor, grip01=1.0, settle_trim=False):
        """Slow move; settle_trim: settle onto the goal with the bounded integral trim (the ready pose,
        which the arm's static friction otherwise misses by 20+ mrad on joint 3)."""
        q7 = np.asarray(self.arm.robot.get_joint_pos(), float)
        q7[6] = grip01
        self.arm.safe.command(q7)
        self.arm.move_to(q6_motor, max_vel=0.1, settle_trim=settle_trim)

    def stopped(self):
        import estop
        return estop.stopped()

    def hold(self):
        self.arm.hold()

    def truth(self):
        return None

    def close(self):
        try:
            self.cams.close()
        finally:
            self.arm.close()


class SimBackend:
    def __init__(self, cfg, offsets, gmap, seed=0, device="cuda:0", phys_json=None, checker=None):
        """checker: a FlexChecker; every sent target then also goes through SafeArm's rules (table
        clearances, joint box, step cap, deviation limit) and violations are logged, not enforced."""
        from blind_insertion.assets import builder as B
        from blind_insertion.envs.pick_env import PickCfg, PickEnv
        phys = {**PickCfg().phys, **(phys_json or {})}
        v = B.sample_reader(1, seed)[0]
        self.env = PickEnv(PickCfg(nworld=1, seed=seed, phys=phys, fric_comp=cfg["fric_comp"], device=device), variant=v)
        self.env.reset_all()
        self.off, self.gmap, self.control_dt = offsets, gmap, self.env.control_dt
        self.depth = v.depth
        # camera clock as in the demos and ActPick: 30 fps, frame age per camera uniform 30-100 ms
        rng = np.random.default_rng(seed)
        self.fps, self.age_lim = 30.0, (0.030, 0.100)
        self.age = {c: rng.uniform(*self.age_lim) for c in ("top", "wrist_cam")}
        self.phase = {c: rng.uniform(0, 1 / self.fps) for c in ("top", "wrist_cam")}
        self.hist, self.step = [], 0
        self.t = 0.0
        self.checker, self.last_cmd, self.violations = checker, None, []
        self.true_chk = None
        if checker is not None:
            from deploy.safety_task import FlexChecker
            self.true_chk = FlexChecker(np.zeros(6))

    def read(self):
        st = self.env.state()[0].cpu().numpy().astype(float)
        qd = self.env.t_qvel[0, self.env.arm_dofs].cpu().numpy().astype(float)
        return st[:6] - self.off, qd, self.gmap.reading(st[6]), self.t

    def frames(self):
        """Latest frame captured at or before now - age on each camera's 30 fps clock."""
        fr = {k: x[0].cpu().numpy() for k, x in self.env.render().items()}
        dt = self.control_dt
        K = int(np.ceil((self.age_lim[1] + 1 / self.fps) / dt)) + 2
        self.hist = (self.hist + [fr])[-K:]
        t, self.step = self.step, self.step + 1
        out, age = {}, {}
        for c in ("top", "wrist_cam"):
            tau = t * dt - self.age[c]
            cap = np.floor((tau - self.phase[c]) / (1 / self.fps)) * (1 / self.fps) + self.phase[c]
            back = int(np.clip(t - np.floor(max(cap, 0) / dt + 1e-6), 0, len(self.hist) - 1))
            out["top" if c == "top" else "wrist"] = self.hist[len(self.hist) - 1 - back][c]
            age["top" if c == "top" else "wrist"] = back * dt
        return out, age, {"top": t, "wrist": t}

    def _safearm_rules(self, q6_motor):
        import safety
        q_meas = self.read()[0]
        why = self.checker.violation(q6_motor)
        if why is None and self.last_cmd is not None and np.abs(q6_motor - self.last_cmd).max() > safety.MAX_STEP:
            why = f"step {np.abs(q6_motor - self.last_cmd).max():.3f} rad > MAX_STEP (capped on the arm)"
        if why is None and np.abs(q6_motor - q_meas).max() > safety.MAX_DEVIATION:
            why = f"{np.abs(q6_motor - q_meas).max():.2f} rad from the measured pose > MAX_DEVIATION (stop)"
        if why is None and (w := self.checker.violation(q_meas)):
            why = f"measured pose: {w}"
        if why is not None:
            self.violations.append((self.step, why))
        self.last_cmd = np.asarray(q6_motor, float).copy()

    def send(self, q6_motor, grip01):
        if self.checker is not None:
            self._safearm_rules(np.asarray(q6_motor, float))
        q = torch.as_tensor(np.asarray(q6_motor, np.float32) + self.off.astype(np.float32), device=self.env.device)
        g = 1.0 if grip01 > 0.5 * (self.gmap.open + self.gmap.close) else -1.0
        self.env.step(torch.cat([q, torch.tensor([g], device=self.env.device)])[None])
        self.t += self.control_dt

    def clock(self):
        return self.t

    def wait(self, t_next):
        pass

    def move_to(self, q6_motor, grip01=1.0, settle_trim=False):
        """Teleport (the sim episode starts at the ready pose + noise, as in training)."""

    def stopped(self):
        return False

    def hold(self):
        pass

    def truth(self):
        p, R = self.env.plug_in_port()
        tilt = float(torch.rad2deg(torch.acos(R[0, 2, 2].clamp(-1, 1))))
        out = dict(seated=bool(p[0, 2] < -(self.depth - 0.0002) and tilt <= 2.0), in_hand=bool(self.env.in_hand()[0]),
                   plug_mm=(p[0] * 1e3).cpu().numpy().round(2).tolist(), tilt_deg=round(tilt, 2))
        if self.true_chk is not None:     # true joints (no encoder error): lowest point of each body
            h = self.true_chk.heights(self.env.t_qpos[0, self.env.arm_q].cpu().numpy())
            out["low_mm"] = {b: round(z * 1e3, 1) for b, z in h.items()}
        return out

    def close(self):
        del self.env
