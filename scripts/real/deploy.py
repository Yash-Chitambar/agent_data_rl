"""Deployment control loop for the real YAM: the sim's 50 Hz relative-action scheme, with timing logs.

One loop for the real-arm tests (docs/REAL_YAM_LATENCY_BACKLASH.md) and for policies:

  * 50 Hz (sim control_dt). Each step: read the encoders, build the observation in the model's
    joint frame (q_model = q_measured + rig.joint_offsets(side)), compute the action, send
    q_target = q_measured + action_scale * action through safety.SafeArm. The offsets cancel in
    the target (both sides are motor readings); they matter in the observation (joints and FK).
  * Gains set at start (i2rt update_kp_kd). Default: the sim's YAM gains (builder "yam"), with kd
    capped at the DM motors' MIT limit (5.0: joints 1-3 get 5 instead of 6).
  * Timing: hooks in the i2rt driver threads stamp (time.monotonic) every encoder reading when it
    arrives (end of a CAN burst) and the first CAN burst that carries each command. Per step:
    t_obs = arrival of the reading the step used, t_act = action ready, t_send = command on the bus.

Sources: hold (keeps the start pose: a = (q_start - q) / scale), step (joint step test),
policy (a checkpoint; proprioceptive TCN policies), random (offline check only). --infer adds a
random-weight policy forward pass to hold, to time inference without trusting a policy.

    .venv/bin/python scripts/real/deploy.py check                      # offline: loop vs direct sim
    .venv/bin/python scripts/real/deploy.py run --side right --source hold --seconds 60 --infer --name hold_60s
    .venv/bin/python scripts/real/deploy.py run --side right --source step --joint 4 --amp 0.01 --reps 3 --name steps_joint4

Run scripts/real/estop.py first (Enter holds the arm). Logs go to data/real_yam/<name>*.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import estop  # noqa: E402
import rig  # noqa: E402
import safety  # noqa: E402
from blind_insertion.assets import builder as B  # noqa: E402
from blind_insertion.envs import friction_comp as FC, kinematics as K  # noqa: E402

SIM_KP = np.array([80, 80, 80, 30, 30, 30], float)    # builder._load_robot "yam"
SIM_KD = np.array([6, 6, 6, 2, 2, 2], float)
I2RT_KP = np.array([80, 80, 80, 40, 10, 10], float)   # i2rt yam.yml (what safety.connect holds with)
I2RT_KD = np.array([5, 5, 5, 1.5, 1.5, 1.5], float)
DM_KD_MAX = 5.0          # i2rt motor_drivers/utils.py: MIT kd range [0, 5] on all DM motors
CONTROL_DT = 0.020
ACTION_SCALE = 0.04      # RobotInfo("yam").action_scale
ACTION_CLIP = 3.0        # EnvCfg.action_clip
FRIC_COMP = None         # (tau, ramp): Coulomb friction feed-forward (envs/friction_comp.py), --fric_comp
HISTORY = 30             # EnvCfg.history
OUT = REPO / "data/real_yam"


# ----------------------------------------------------------------------------- observation
class ObsBuilder:
    """The actor observation of PegEnv._actor_obs_now for the YAM, from model-frame joints:
    [q, qd, tcp pos, tcp euler xyz in R_obs, tcp twist, prev action], plus the 30-step history."""

    def __init__(self, grasp_mode="axial", fix_base_rpy_deg=(0.0, 0.0, 0.0), history=HISTORY):
        spec, info = B.load_robot("yam")
        m = spec.compile()
        site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, info.tcp_site)
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in info.arm_joints]
        self.chain = K.Chain(m, site, np.array(qadr), device="cpu")
        if grasp_mode == "top":
            R_nom = torch.tensor([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
            self.R_obs = K.rpy_mat(torch.tensor(fix_base_rpy_deg, dtype=torch.float32) * math.pi / 180) @ R_nom
        else:
            yaw = math.radians(info.nom_yaw_deg)
            self.R_obs = K.rpy_mat(torch.tensor([0.0, 0.0, yaw])) @ K.rpy_mat(torch.tensor([math.pi, 0.0, 0.0]))
        self.history = history
        self.prev = torch.zeros(1, 6)
        self.hist = None

    def now(self, q_model, qd):
        q = torch.as_tensor(q_model, dtype=torch.float32)[None]
        qd = torch.as_tensor(qd, dtype=torch.float32)[None]
        p, R, J = self.chain.fk(q, with_jac=True)
        tw = (J @ qd[..., None])[..., 0]
        eul = K.euler_xyz(self.R_obs.T[None] @ R)
        return torch.cat([q, qd, p, eul, tw, self.prev], 1)

    def push(self, q_model, qd):
        """Observation for this step, appended to the history (filled with it on the first call,
        as PegEnv.reset does). Returns the (1, H, D) history."""
        o = self.now(q_model, qd)
        if self.hist is None:
            self.hist = o[:, None].expand(-1, self.history, -1).clone()
        else:
            self.hist = torch.cat([self.hist[:, 1:], o[:, None]], 1)
        return self.hist

    def set_prev(self, a):
        self.prev = torch.as_tensor(a, dtype=torch.float32).reshape(1, 6)


def load_actor(path, history=HISTORY):
    from blind_insertion.policies.tcn_actor_critic import ActorCritic
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("modality") in ("rgb", "depth", "rgbd"):
        raise SystemExit("visual policies are not supported by deploy.py yet")
    ac = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["critic_dim"], history=history,
                     encoder=ck.get("encoder", "tcn"), n_points=ck.get("n_points", 64))
    ac.load_state_dict(ck["model"])
    ac.eval()
    ac.obs_norm.freeze()
    return ac


def dummy_actor():
    from blind_insertion.policies.tcn_actor_critic import ActorCritic
    torch.manual_seed(0)
    return ActorCritic(30, 6, 401, history=HISTORY).eval()


# ----------------------------------------------------------------------------- sources
class Hold:
    """Keep the start pose: the target is the start reading every step."""

    def __init__(self, actor=None):
        self.q0, self.actor = None, actor

    def __call__(self, k, q_raw, q_model, qd, obs):
        if self.actor is not None:   # timing only: the result is not used
            with torch.no_grad():
                self.actor.act_mean(obs.push(q_model, qd))
        if self.q0 is None:
            self.q0 = q_raw.copy()
        return (self.q0 - q_raw) / ACTION_SCALE


class Step:
    """Joint step test: hold 1 s, +amp on one joint for 1 s, back for 1 s; `reps` times."""

    def __init__(self, joint, amp, reps, hold_s=1.0):
        self.j, self.amp, self.n = joint - 1, amp, int(round(hold_s / CONTROL_DT))
        self.steps = self.n * (1 + 2 * reps)
        self.q0 = None

    def __call__(self, k, q_raw, q_model, qd, obs):
        if self.q0 is None:
            self.q0 = q_raw.copy()
        tgt = self.q0.copy()
        if k >= self.n and ((k - self.n) // self.n) % 2 == 0:
            tgt[self.j] += self.amp
        return (tgt - q_raw) / ACTION_SCALE


class Policy:
    def __init__(self, path, history=HISTORY):
        self.actor = load_actor(path, history)

    def __call__(self, k, q_raw, q_model, qd, obs):
        with torch.no_grad():
            mu, _ = self.actor.act_mean(obs.push(q_model, qd))
        return mu[0].numpy().astype(float)


class Random:
    """Smooth random actions (offline check)."""

    def __init__(self, seed=0, n=500):
        rng = np.random.default_rng(seed)
        w = rng.normal(0, 1.0, (n + 20, 6))
        k = np.ones(10) / 10
        self.a = np.stack([np.convolve(w[:, j], k, "same") for j in range(6)], 1)[:n] * 1.5

    def __call__(self, k, q_raw, q_model, qd, obs):
        return self.a[k]


# ----------------------------------------------------------------------------- arms
class BusTiming:
    """Hooks into the i2rt threads (DM chain and robot server) of one arm. Stamps every encoder
    reading at its arrival and the first CAN burst of every labelled command. Wrappers only record:
    each calls the original method unchanged."""

    def __init__(self, robot):
        self.robot, mc = robot, robot.motor_chain
        self.enc = []                     # (t, iteration, q7 in command joint space)
        self.it = 0
        self.ts_it = collections.OrderedDict()   # read_states timestamp -> (it before, it after)
        self.sent = {}                    # label -> t of the first CAN burst carrying it
        self.cmd_label = collections.OrderedDict()   # id(command list) -> (list, label)
        self.pending = None               # (label, expected arm pos in robot joint space)
        self.cur = None

        upd0, read0, set0, send0 = (mc._update_absolute_positions, mc.read_states, mc.set_commands,
                                    mc._set_commands)
        n = len(mc.motor_list)

        def upd(fb):
            r = upd0(fb)
            t = time.monotonic()
            self.it += 1
            q = np.array([mc._joint_position_real_to_sim_idx(mc.absolute_positions[i], i) for i in range(n)])
            self.enc.append((t, self.it, robot.remapper.to_command_joint_pos_space(q)))
            return r

        def read(torques=None):
            before = self.it
            out = read0(torques)
            self.ts_it[out[0].timestamp] = (before, self.it)
            while len(self.ts_it) > 5000:
                self.ts_it.popitem(last=False)
            return out

        def setc(torques, pos=None, vel=None, kp=None, kd=None, get_state=True):
            p = self.pending
            if p is not None and pos is not None and np.array_equal(np.asarray(pos)[:6], p[1]):
                self.cur = p[0]
            with mc.command_lock:    # RLock: the DM thread cannot send the list before it is labelled
                out = set0(torques, pos=pos, vel=vel, kp=kp, kd=kd, get_state=False)
                self.cmd_label[id(mc.commands)] = (mc.commands, self.cur)
                while len(self.cmd_label) > 64:
                    self.cmd_label.popitem(last=False)
            return read(torques) if get_state else out

        def send(commands):
            ent = self.cmd_label.get(id(commands))
            if ent is not None and ent[0] is commands and ent[1] is not None and ent[1] not in self.sent:
                self.sent[ent[1]] = time.monotonic()
            return send0(commands)

        mc._update_absolute_positions, mc.read_states, mc.set_commands, mc._set_commands = upd, read, setc, send

        cmd0 = robot.command_joint_pos

        def command(q7):
            q7 = np.asarray(q7, float)
            exp = robot.remapper.to_robot_joint_pos_space(robot._clip_robot_joint_pos_command(q7))[:6]
            self.pending = (self.label, exp)
            return cmd0(q7)

        self.label = None
        robot.command_joint_pos = command

    def t_obs(self, timestamp):
        """Arrival time of the reading behind a robot joint state (older bound if a DM iteration
        finished during the read), and whether it was exact."""
        before, after = self.ts_it.get(timestamp, (None, None))
        if before is None or before == 0:
            return math.nan, False
        return self.enc[before - 1][0], before == after


class RealArm:
    def __init__(self, side, kp, kd):
        self.side = side
        self.motion_trace = []
        self.robot = safety.connect(side)
        self.timing = BusTiming(self.robot)
        r = self.robot
        kp7, kd7 = r._kp.copy(), r._kd.copy()
        self.gains_before = (kp7[:6].tolist(), kd7[:6].tolist())
        kp7[:6], kd7[:6] = kp, kd
        r.update_kp_kd(kp7, kd7)   # used from the next command on
        self.safe = safety.SafeArm(r)
        self.safe._hold(np.asarray(r.get_joint_pos(), float))   # re-send the hold with the new gains

    def read(self):
        r = self.robot
        with r._state_lock:
            q, qd, ts = r._joint_state.pos.copy(), r._joint_state.vel.copy(), r._joint_state.timestamp
        t_obs, exact = self.timing.t_obs(ts)
        return q, qd, t_obs, exact

    def send(self, q7, label):
        self.timing.label = label
        out = self.safe.command(q7, FRIC_COMP)
        self.last_ff = getattr(self.safe, "last_ff", None)
        return out

    def hold(self):
        self.safe._hold(np.asarray(self.robot.get_joint_pos(), float))

    def move_to(self, q6_goal, max_vel=0.3):
        """Slow smoothstep move (no logging), the straight joint path checked first. Raises
        SafetyStop (holding) on an unsafe path, the software stop or a safety check."""
        q0 = np.asarray(self.robot.get_joint_pos(), float)
        q6_goal = np.asarray(q6_goal, float)
        for a in np.linspace(0, 1, 41):
            why = self.safe.check.violation((1 - a) * q0[:6] + a * q6_goal)
            if why:
                raise safety.SafetyStop(f"path to {q6_goal.round(2)} unsafe at {a:.2f}: {why}")
        n = max(1, int(np.ceil(np.abs(q6_goal - q0[:6]).max() / (max_vel * CONTROL_DT) * 1.5)))
        t_next = time.monotonic()
        for i in range(1, n + 1):
            if estop.stopped():
                self.hold()
                raise safety.SafetyStop("software stop (estop.py)")
            s = i / n
            s = s * s * (3 - 2 * s)
            tgt = q0.copy()
            tgt[:6] = (1 - s) * q0[:6] + s * q6_goal
            sent = self.safe.command(tgt)
            measured = np.asarray(self.robot.get_joint_pos(), float)
            self.motion_trace.append(dict(host_monotonic=time.monotonic(), target=sent.tolist(),
                                          measured=measured.tolist(), mode="move_to"))
            t_next += CONTROL_DT
            self.wait(t_next)
        time.sleep(1.0)   # settle
        actual = np.asarray(self.robot.get_joint_pos(), float)[:6]
        if np.max(np.abs(actual - q6_goal)) > 0.02:
            self.hold()
            raise safety.SafetyStop(f"move_to goal not attained within 0.02 rad; error_rad={(actual - q6_goal).tolist()}; holding")

    def wait(self, t_next):
        while True:
            dt = t_next - time.monotonic()
            if dt <= 0:
                return
            time.sleep(dt - 0.0015 if dt > 0.002 else 0)   # sleep, then spin the last 1.5 ms

    def info(self):
        r = self.robot
        return {"gains_before": self.gains_before, "kp": r._kp[:6].tolist(), "kd": r._kd[:6].tolist(),
                "gravity_comp_factor": np.asarray(r.gravity_comp_factor).tolist(),
                "coulomb_friction": np.asarray(r._coulomb_friction).tolist()}

    def close(self):
        safety.close_robot(self.robot)


class SimArm:
    """CPU MuJoCo YAM with the sim's actuators, in lockstep with the loop. Its encoders read
    q_true - offsets, as a real arm whose motor zeros sit off the model's."""

    def __init__(self, kp, kd, offsets, q0):
        spec, info = B.load_robot("yam", gains=(kp, kd))
        spec.option.timestep = 0.001
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        J = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
        self.qadr = np.array([self.m.jnt_qposadr[J(j)] for j in info.arm_joints])
        self.dadr = np.array([self.m.jnt_dofadr[J(j)] for j in info.arm_joints])
        self.act = np.array([mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in info.arm_joints])
        self.off = np.asarray(offsets, float)
        self.d.qpos[self.qadr] = q0
        self.d.ctrl[self.act] = q0
        mujoco.mj_forward(self.m, self.d)
        self.t = 0.0

    def read(self):
        q = np.append(self.d.qpos[self.qadr] - self.off, 0.0)
        return q, np.append(self.d.qvel[self.dadr], 0.0), self.t, True

    def send(self, q7, label):
        q7 = np.asarray(q7, float)
        if FRIC_COMP is not None:     # as SafeArm.command: from the target and the encoder reading
            self.last_ff = FC.feedforward(q7[:6] - (self.d.qpos[self.qadr] - self.off), *FRIC_COMP)
            self.d.qfrc_applied[self.dadr] = self.last_ff
        self.d.ctrl[self.act] = q7[:6] + self.off
        return q7

    def hold(self):
        pass

    def wait(self, t_next):
        while self.t < t_next - 1e-9:
            mujoco.mj_step(self.m, self.d)
            self.t += self.m.opt.timestep

    def info(self):
        return {"sim": True}

    def close(self):
        pass


# ----------------------------------------------------------------------------- loop
def run_loop(arm, source, n_steps, offsets, obs=None, clock=time.monotonic):
    """Returns per-step rows. Stops (holding) on the software stop or a safety stop."""
    obs = obs or ObsBuilder()
    rows, grip = [], None
    t_next = clock()
    stop = None
    for k in range(n_steps):
        if estop.stopped():
            arm.hold()
            stop = "software stop (estop.py)"
            break
        t_tick = clock()
        q, qd, t_obs, exact = arm.read()
        q_model = q[:6] + offsets
        a = np.clip(np.asarray(source(k, q[:6], q_model, qd[:6], obs), float), -ACTION_CLIP, ACTION_CLIP)
        t_act = clock()
        grip = q[6] if grip is None else grip
        tgt = np.append(q[:6] + ACTION_SCALE * a, grip)
        try:
            sent = arm.send(tgt, k)
        except safety.SafetyStop as e:
            stop = str(e)
            break
        t_ret = clock()
        obs.set_prev(a)
        rows.append(dict(k=k, t_tick=t_tick, t_obs=t_obs, obs_exact=exact, t_act=t_act, t_ret=t_ret,
                         q=q[:6].tolist(), qd=qd[:6].tolist(), q_model=q_model.tolist(),
                         a=a.tolist(), q_cmd=np.asarray(sent)[:6].tolist(),
                         tau_ff=None if FRIC_COMP is None else np.asarray(arm.last_ff).tolist()))
        t_next += CONTROL_DT
        arm.wait(t_next)
    return rows, stop


# ----------------------------------------------------------------------------- offline check
def cmd_check(a):
    """Loop + SimArm (offsets on) against a direct rollout of the sim's action scheme."""
    off = rig.joint_offsets("right")
    q0 = np.array([0.0, 1.2, 1.0, -0.6, 0.0, 0.0])
    n = 300
    src = Random(seed=1, n=n)

    # reference: PegEnv's scheme on the model directly, q_t = q + scale * a held 20 ms
    ref = SimArm(SIM_KP, SIM_KD, np.zeros(6), q0)
    ref_q = []
    for k in range(n):
        q = ref.d.qpos[ref.qadr].copy()
        ref_q.append(q)
        ref.d.ctrl[ref.act] = q + ACTION_SCALE * np.clip(src.a[k], -ACTION_CLIP, ACTION_CLIP)
        ref.wait((k + 1) * CONTROL_DT)
    ref_q = np.array(ref_q)

    sim = SimArm(SIM_KP, SIM_KD, off, q0)
    est = estop.STOP_FILE
    estop.STOP_FILE = Path("/nonexistent/yam_estop_check")   # never stopped in the check
    try:
        rows, stop = run_loop(sim, src, n, off, clock=lambda: sim.t)
    finally:
        estop.STOP_FILE = est
    assert stop is None, stop
    loop_q = np.array([r["q_model"] for r in rows])
    err = np.abs(loop_q - ref_q).max()
    print(f"loop vs direct sim, {n} steps of random actions, offsets {np.degrees(off).round(2)} deg: "
          f"max joint difference {err:.2e} rad")
    no_off = np.abs(np.array([r["q"] for r in rows]) - ref_q).max()
    print(f"  (the raw encoder readings differ from the model by up to {no_off:.4f} rad: the offsets)")
    travel = np.abs(np.diff(ref_q, axis=0)).sum(0)
    print(f"  joint travel in the check (rad): {travel.round(2)}")
    ok = err < 1e-9

    # observation against PegEnv (GPU), same joints and velocities
    try:
        ok &= check_obs_vs_env(a)
    except Exception as e:   # noqa: BLE001
        print(f"  PegEnv observation check failed to run: {e!r}")
        ok = False
    ok &= check_fric_comp()
    print("CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def check_fric_comp():
    """Step test (as on the real arm, FINDINGS 2026-10-25) on SimArm with the measured joint friction:
    fraction of a 0.01 / 0.04 rad step reached after 1 s on joints 2-3, without and with the
    friction feed-forward. Passes if 1.0x brings every step to >= 85% (0.7x: >= 65%; the push fades
    within `ramp` of the target, so some error stays at < 1.0x)."""
    global FRIC_COMP
    q0 = np.array([0.0, 1.2, 1.0, -0.6, 0.0, 0.0])
    est, keep = estop.STOP_FILE, FRIC_COMP
    estop.STOP_FILE = Path("/nonexistent/yam_estop_check")
    ok = True
    try:
        for name, fc in (("off", None), ("0.7x measured", FC.parse({"scale": 0.7, "ramp": 0.01})),
                         ("1.0x measured", FC.parse({"scale": 1.0, "ramp": 0.01}))):
            FRIC_COMP = fc
            res = []
            for j in (2, 3):
                for amp in (0.01, 0.04):
                    sim = SimArm(SIM_KP, np.minimum(SIM_KD, DM_KD_MAX), np.zeros(6), q0)
                    sim.m.dof_frictionloss[sim.dadr] = FC.MEASURED_TAU
                    src = Step(j, amp, reps=2)
                    rows, stop = run_loop(sim, src, src.steps, np.zeros(6), clock=lambda: sim.t)
                    assert stop is None, stop
                    q = np.array([r["q"] for r in rows])[:, j - 1]
                    n = src.n
                    fr = [(q[n * (2 * i + 2) - 1] - q[n * (2 * i + 1) - 1]) / amp for i in range(2)]
                    res.append(f"j{j} {amp:.2f}: {100 * np.mean(fr):4.0f}%")
                    if fc is not None:   # MuJoCo's soft frictionloss moves partly even without it
                        ok &= np.mean(fr) >= (0.85 if fc[0][j - 1] >= FC.MEASURED_TAU[j - 1] else 0.65)
            print(f"  friction feed-forward {name:>14}: step reached after 1 s  " + ", ".join(res))
    finally:
        estop.STOP_FILE, FRIC_COMP = est, keep
    return ok


def check_obs_vs_env(a):
    from blind_insertion.envs.peg_env import EnvCfg, PegEnv
    cfgs = {"axial (phase_c_yam)": "blind_insertion/configs/phase_c_yam.json",
            "top (task_h_reader)": "blind_insertion/configs/task_h_reader.json"}
    ok = True
    for name, path in cfgs.items():
        kw = json.load(open(REPO / path))
        env = PegEnv(EnvCfg(**kw, nworld=4, auto_reset=False))
        env.reset(torch.arange(4, device=env.device))
        rng = np.random.default_rng(0)
        q = env.home[None].repeat(4, 1) + torch.as_tensor(rng.normal(0, 0.2, (4, 6)), dtype=torch.float32,
                                                            device=env.device)
        qd = torch.as_tensor(rng.normal(0, 0.5, (4, 6)), dtype=torch.float32, device=env.device)
        env.t_qpos[:, torch.as_tensor(env.lay.arm_qadr, device=env.device)] = q
        env.t_qvel[:, torch.as_tensor(env.lay.arm_dofadr, device=env.device)] = qd
        env.prev_action = torch.as_tensor(rng.normal(0, 1, (4, 6)), dtype=torch.float32, device=env.device)
        saved = env.cfg.q_noise, env.cfg.qd_noise
        o_env = env._actor_obs_now().cpu().numpy()
        ob = ObsBuilder(kw.get("grasp_mode", "axial"), kw.get("fix_base_rpy_deg", (0.0, 0.0, 0.0)))
        o_me = []
        for i in range(4):
            ob.set_prev(env.prev_action[i].cpu().numpy())
            o_me.append(ob.now(o_env[i, :6], o_env[i, 6:12]).numpy()[0])
        err = np.abs(np.array(o_me) - o_env).max()
        print(f"  observation vs PegEnv._actor_obs_now, {name}: max difference {err:.2e} (noise {saved})")
        ok &= err < 1e-4
        del env
    return ok


# ----------------------------------------------------------------------------- real run
def resolve_and_save(arm, rows, stop, a, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    tm = arm.timing
    for r in rows:
        r["t_send"] = tm.sent.get(r["k"], math.nan)
    import pandas as pd
    df = pd.DataFrame([{**{k: r[k] for k in ("k", "t_tick", "t_obs", "obs_exact", "t_act", "t_ret", "t_send")},
                        **{f"q_{j + 1}": r["q"][j] for j in range(6)},
                        **{f"q_cmd_{j + 1}": r["q_cmd"][j] for j in range(6)},
                        **{f"a_{j + 1}": r["a"][j] for j in range(6)}} for r in rows])
    df.to_csv(OUT / f"timing_{a.name}.csv", index=False)   # not <name>_*.csv: the fit globs steps_joint*.csv

    # every encoder reading, with the command on the bus at that time (fit script format)
    sends = sorted((t, k) for k, t in tm.sent.items() if np.isfinite(t))
    cmd = {r["k"]: r["q_cmd"] for r in rows}
    t0 = rows[0]["t_tick"] if rows else 0
    enc = [(t, q) for t, _, q in tm.enc if t >= t0 - 0.5]
    out, si, cur, cur_t = [], 0, None, math.nan
    for t, q in enc:
        while si < len(sends) and sends[si][0] <= t:
            cur, cur_t = cmd.get(sends[si][1]), sends[si][0]
            si += 1
        if cur is None:
            continue
        out.append({"t": t, **{f"q_cmd_{j + 1}": cur[j] for j in range(6)},
                    **{f"q_{j + 1}": q[j] for j in range(6)}, "t_cmd": cur_t})
    pd.DataFrame(out).to_csv(OUT / f"{a.name}.csv", index=False)

    lat = df.t_send - df.t_obs
    enc_t = np.array([t for t, _ in enc])
    summ = {"steps": len(df), "stop": stop,
            "loop_period_ms": {"mean": float(np.diff(df.t_tick).mean() * 1e3), "std": float(np.diff(df.t_tick).std() * 1e3),
                               "max": float(np.diff(df.t_tick).max() * 1e3)} if len(df) > 1 else None,
            "obs_age_at_tick_ms": _stats((df.t_tick - df.t_obs) * 1e3),
            "compute_ms (t_act - t_tick)": _stats((df.t_act - df.t_tick) * 1e3),
            "act_to_send_ms": _stats((df.t_send - df.t_act) * 1e3),
            "loop_latency_ms (t_send - t_obs)": _stats(lat * 1e3),
            "obs_exact_fraction": float(df.obs_exact.mean()) if len(df) else None,
            "unsent_steps": int(df.t_send.isna().sum()),
            "encoder_rate_hz": float(1 / np.diff(enc_t).mean()) if len(enc_t) > 1 else None,
            "encoder_period_ms_p99": float(np.percentile(np.diff(enc_t), 99) * 1e3) if len(enc_t) > 1 else None,
            "hold_error_rad (q_cmd - q, mean / std)": {
                "mean": (df[[f"q_cmd_{j}" for j in range(1, 7)]].to_numpy() - df[[f"q_{j}" for j in range(1, 7)]].to_numpy()).mean(0).round(5).tolist(),
                "std": df[[f"q_{j}" for j in range(1, 7)]].to_numpy().std(0).round(6).tolist()} if len(df) else None}
    meta.update(summary=summ)
    (OUT / f"{a.name}_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(summ, indent=1))


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return {"mean": round(float(x.mean()), 3), "std": round(float(x.std()), 3), "p50": round(float(np.median(x)), 3),
            "p99": round(float(np.percentile(x, 99)), 3), "max": round(float(x.max()), 3)}


def cmd_run(a):
    if estop.stopped():
        raise SystemExit(f"{estop.STOP_FILE} is set: clear it in estop.py first")
    if a.gains == "sim":
        kp, kd = SIM_KP, np.minimum(SIM_KD, DM_KD_MAX)
        print(f"sim gains: kp {kp.tolist()}, kd {kd.tolist()} (sim kd {SIM_KD.tolist()}, DM MIT limit {DM_KD_MAX})")
    else:
        kp, kd = I2RT_KP, I2RT_KD
    global ACTION_SCALE, ACTION_CLIP, FRIC_COMP
    env = json.load(open(a.env_file)) if a.env_file else {}
    FRIC_COMP = FC.parse(json.loads(a.fric_comp) if a.fric_comp else env.get("fric_comp"))
    print(f"friction feed-forward: {FRIC_COMP}")
    if a.source == "policy" and not a.env_file:
        raise SystemExit("--source policy needs --env_file (the policy's training config: action scale, history, grasp)")
    ACTION_SCALE = env.get("action_scale") or ACTION_SCALE
    ACTION_CLIP = env.get("action_clip") or ACTION_CLIP   # as trained (EnvCfg.action_clip)
    obs = ObsBuilder(env.get("grasp_mode", "axial"), env.get("fix_base_rpy_deg", (0.0, 0.0, 0.0)),
                     env.get("history", HISTORY))
    if a.source == "step":
        src = Step(a.joint, a.amp, a.reps)
        n = src.steps
    elif a.source == "policy":
        src, n = Policy(a.policy, obs.history), int(a.seconds / CONTROL_DT)
    else:
        src, n = Hold(dummy_actor() if a.infer else None), int(a.seconds / CONTROL_DT)
    off = rig.joint_offsets(a.side)
    arm = RealArm(a.side, kp, kd)
    torch.set_num_threads(1)
    meta = {"args": vars(a), "side": a.side, "joint_offsets_rad": off.tolist(), "arm": arm.info(),
            "control_dt": CONTROL_DT, "action_scale": ACTION_SCALE, "fric_comp": FRIC_COMP, "history": obs.history, "clock": "time.monotonic",
            "started": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    rows, stop = [], None
    q_rest = np.asarray(arm.robot.get_joint_pos(), float)[:6]
    try:
        time.sleep(1.0)    # settle at the new gains
        if a.start:
            arm.move_to(a.start)
        rows, stop = run_loop(arm, src, n, off, obs)
    except safety.SafetyStop as e:
        stop = str(e)
    except KeyboardInterrupt:
        stop = "interrupted"
    finally:
        arm.hold()
        time.sleep(0.3)
        try:
            if rows:
                resolve_and_save(arm, rows, stop, a, meta)
        finally:
            if stop:
                print(f"STOPPED: {stop}. The arm holds; clear the stop in estop.py to continue.", flush=True)
                estop.STOP_FILE.write_text("stop")
                while estop.stopped():
                    time.sleep(0.1)
            if a.start:
                for _ in range(3):   # back to where the run began; a stop on the way holds and waits
                    try:
                        arm.move_to(q_rest, max_vel=0.2)
                        break
                    except safety.SafetyStop as e:
                        print(f"STOPPED on the way back: {e}. Clear the stop in estop.py to retry.", flush=True)
                        estop.STOP_FILE.write_text("stop")
                        while estop.stopped():
                            time.sleep(0.1)
            arm.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    r = sub.add_parser("run")
    r.add_argument("--side", default="right", choices=["right", "left"])
    r.add_argument("--source", default="hold", choices=["hold", "step", "policy"])
    r.add_argument("--seconds", type=float, default=60.0)
    r.add_argument("--gains", default="sim", choices=["sim", "i2rt"])
    r.add_argument("--infer", action="store_true", help="hold + a policy-sized forward pass per step (timing)")
    r.add_argument("--policy", help="checkpoint for --source policy")
    r.add_argument("--env_file", help="the policy's training config (action_scale, history, grasp_mode, fric_comp)")
    r.add_argument("--fric_comp", help='friction feed-forward, e.g. \'{"scale": 0.7, "ramp": 0.01}\' (default: env_file fric_comp, else off)')
    r.add_argument("--joint", type=int, default=1)
    r.add_argument("--amp", type=float, default=0.04)
    r.add_argument("--reps", type=int, default=10)
    r.add_argument("--start", type=float, nargs=6, help="move here first (slowly), back to rest after")
    r.add_argument("--name", required=True)
    a = ap.parse_args()
    sys.exit({"check": cmd_check, "run": cmd_run}[a.cmd](a) or 0)
