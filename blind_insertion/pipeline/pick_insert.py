"""Task H end to end in MuJoCo Warp: pick (scripted expert or ACT) -> handoff -> blind insertion
(the proprio TCN policy trained in PegEnv), in the full scene of `envs.pick_env`.

Handoff: when the pick policy raises `done` (ACT: done output > 0.5 for 5 steps; expert: its
done flag) or at the pick time limit, the insertion policy takes over with its history filled
with the current observation; the fingers keep squeezing with the pick's constant force. The
insertion policy sees exactly PegEnv's actor observation (joints, velocities, tcp pose and twist,
previous action) and commands q_target = q_meas + scale * a.

Release: once the plug is seated, the arm holds its joints, opens the gripper for `open_s`, then
backs the tcp out along its approach axis by `lift` (IK, fixed orientation). `seated_after` checks that
the plug is still seated when the hand is clear (strict: tilt <= 2 deg; a released reader sags ~1.7-2.3
deg in the port under its weight, so a few fully inserted plugs, z ~ -6.27 mm, fail it).

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$PWD MUJOCO_GL=egl .venv/bin/python -m blind_insertion.pipeline.pick_insert \
        --pick expert --insert runs/H-reader-proprio-top-s0/last.pt --episodes 256 --out results/H/pipe_expert.json
    ... --pick act --act runs/H-act-pick-v1/act_traced.pt
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from blind_insertion.analysis.stats import wilson
from blind_insertion.assets import builder as B
from blind_insertion.envs import kinematics as K
from blind_insertion.envs.pick_env import Expert, PickCfg, PickEnv
from blind_insertion.policies.tcn_actor_critic import ActorCritic
from blind_insertion.scenes import pc_bench as S


class ActPick:
    """TorchScript ACT (scripts/act/export_act.py): (state, top, wrist) -> action chunk, with the
    dataset normalization inside. ensemble=k > 0: query every step and average the predictions
    for the current step from all chunks, weights exp(-k * age) (ACT temporal ensembling);
    otherwise execute `n_exec` actions of each chunk open loop."""

    def __init__(self, path, device, n_exec=10, ensemble=0.01, settle=True, rest_qd=0.03, rest_s=0.3,
                 near=0.06, min_finger=0.005, cam_fps=30.0, cam_age_ms=(30.0, 100.0), seed=0, gate=True):
        """settle: also hand off when the arm has rested `rest_s` (all |qd| < rest_qd rad/s) with the
        gripper closed on the reader (finger opening > min_finger) and the tcp within `near` of
        the nominal pre-insert pose (configured PC position, not the randomized one), for when ACT
        does not raise `done`. Uses only joint encoders, so it runs on the real arm unchanged."""
        self.m = torch.jit.load(path, map_location=device).eval()
        # state = joints + finger (7), or + last sent command - joints (13; gen_demos --cmd_err)
        self.cmd_err = self.m.s_mean.shape[0] == 13
        self.n_exec, self.ens = n_exec, ensemble
        self.settle, self.rest_qd, self.rest_s, self.near, self.min_finger = settle, rest_qd, rest_s, near, min_finger
        self.gate = gate
        # camera timing as in the demos (scripts/act/gen_demos.py): 30 fps frame clock, frame age per
        # episode and camera; cam_fps 0 = a fresh frame every step
        self.cam_fps, self.cam_age, self.rng = cam_fps, cam_age_ms, np.random.default_rng(seed)

    def reset(self, n, device):
        self.chunk, self.k, self.t = None, 0, 0
        self.buf = None
        self.done_run = torch.zeros(n, dtype=torch.long, device=device)
        self.rest_run = torch.zeros(n, dtype=torch.long, device=device)
        self.hist, self.step = [], 0
        self.last_cmd = None
        period = 1 / self.cam_fps if self.cam_fps > 0 else 0.0
        self.age = {c: torch.as_tensor(self.rng.uniform(*self.cam_age, n) / 1e3, device=device) for c in ("top", "wrist_cam")}
        self.phase = {c: torch.as_tensor(self.rng.uniform(0, period, n), device=device) for c in ("top", "wrist_cam")}

    def _frames(self, env):
        """Latest frame captured at or before now - age on each camera's frame clock (per world)."""
        fr = env.render()
        dt = env.control_dt
        K = int(np.ceil((self.cam_age[1] / 1e3 + (1 / self.cam_fps if self.cam_fps else 0)) / dt)) + 2
        self.hist = (self.hist + [fr])[-K:]
        t, self.step = self.step, self.step + 1
        if self.cam_fps <= 0:
            return fr
        period, out = 1 / self.cam_fps, {}
        idx = torch.arange(env.N, device=env.device)
        for c in ("top", "wrist_cam"):
            tau = t * dt - self.age[c]
            cap = torch.floor((tau - self.phase[c]) / period) * period + self.phase[c]
            back = (t - torch.floor(cap.clamp(min=0) / dt + 1e-6).long()).clamp(0, len(self.hist) - 1)
            stack = torch.stack([h[c] for h in self.hist])            # (K, N, H, W, 3), oldest first
            out[c] = stack[len(self.hist) - 1 - back, idx]
        return out

    def _settled(self, env, closed):
        qd = env.t_qvel[:, torch.as_tensor(env.lay.arm_dofadr, device=env.device)]
        p_nom = torch.tensor([env.cfg.port_xy[0] - env.body_z - 0.006, env.cfg.port_xy[1],
                              S.PORT_HEIGHT + float(np.mean(S.GRASP_Y[env.gripper]))], device=env.device)
        rest = ((qd.abs().max(1).values < self.rest_qd) & closed & (env.state()[:, 6] > self.min_finger)
                & ((env.tcp_pose()[0] - p_nom).norm(dim=1) < self.near))
        self.rest_run = torch.where(rest, self.rest_run + 1, torch.zeros_like(self.rest_run))
        return self.rest_run >= int(self.rest_s / env.control_dt)

    def _query(self, env):
        fr = self._frames(env)
        top = fr["top"].permute(0, 3, 1, 2).float() / 255
        wrist = fr["wrist_cam"].permute(0, 3, 1, 2).float() / 255
        st = env.state()
        if self.cmd_err:
            last = st[:, :6] if self.last_cmd is None else self.last_cmd
            st = torch.cat([st, last - st[:, :6]], 1)
        with torch.no_grad():
            return self.m(st, top, wrist)                           # (N, C, 8)

    def act(self, env: PickEnv):
        if self.ens > 0:
            c = self._query(env)
            N, C, D = c.shape
            if self.buf is None:
                self.buf = torch.zeros(N, C, C, D, device=c.device)    # [age, offset] predictions
                self.valid = torch.zeros(C, dtype=torch.bool, device=c.device)
            self.buf = torch.roll(self.buf, 1, dims=1)
            self.buf[:, 0] = c
            self.valid = torch.roll(self.valid, 1)
            self.valid[0] = True
            ages = torch.arange(C, device=c.device)
            preds = self.buf[:, ages, ages]                             # chunk of age i, entry i = now
            w = torch.exp(-self.ens * ages.float()) * self.valid
            a = (preds * w[None, :, None]).sum(1) / w.sum()
        else:
            if self.chunk is None or self.k >= self.n_exec:
                self.chunk, self.k = self._query(env), 0
            a = self.chunk[:, self.k]
            self.k += 1
        cmd = torch.cat([a[:, :6], torch.where(a[:, 6:7] > 0, 1.0, -1.0)], 1)
        self.last_cmd = cmd[:, :6].clone()
        self.done_run = torch.where(a[:, 7] > 0.5, self.done_run + 1, torch.zeros_like(self.done_run))
        # grasp gate: no handoff with the gripper open or closed on nothing (fingers ~19 mm on the
        # reader, ~-6 mm empty); ACT keeps control and can regrasp
        grasped = (cmd[:, 6] < 0) & (env.state()[:, 6] > self.min_finger)
        done = (self.done_run >= 5) & (grasped if self.gate else True)
        if self.settle:
            done |= self._settled(env, cmd[:, 6] < 0)
        return cmd, done


class ExpertPick:
    def __init__(self, env):
        self.ex = Expert(env, seed=env.cfg.seed)

    def reset(self, n, device):
        self.ex.reset()

    def act(self, env):
        return self.ex.act()


class Inserter:
    """PegEnv actor observation and action on the pick env's state."""

    def __init__(self, path, env: PickEnv, history=10, scale=0.01):
        ck = torch.load(path, map_location=env.device, weights_only=False)
        self.ac = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["critic_dim"], history=history,
                              encoder=ck.get("encoder", "tcn"), n_points=ck.get("n_points", 64)).to(env.device)
        self.ac.load_state_dict(ck["model"])
        self.ac.eval()
        self.ac.obs_norm.freeze()
        self.env, self.H, self.scale = env, history, scale
        cfg = Path(path).parent / "config.json"            # the run's training config: action clip
        self.clip = float(json.loads(cfg.read_text())["env"].get("action_clip", 3.0)) if cfg.exists() else 3.0
        R_nom = torch.tensor(S.R_TOP_TCP, device=env.device)
        self.R_obs = K.rpy_mat(torch.tensor(S.PORT_BASE_RPY_DEG, device=env.device) * math.pi / 180) @ R_nom
        self.prev = torch.zeros(env.N, 6, device=env.device)
        self.hist = None

    def obs(self):
        env = self.env
        q = env.t_qpos[:, env.arm_q]
        qd = env.t_qvel[:, torch.as_tensor(env.lay.arm_dofadr, device=env.device)]
        p, R, J = env.chain.fk(q, with_jac=True)
        tw = (J @ qd[..., None])[..., 0]
        eul = K.euler_xyz(self.R_obs.expand(env.N, 3, 3).transpose(1, 2) @ R)
        return torch.cat([q, qd, p, eul, tw, self.prev], 1)

    def start(self, ids):
        """Fill the history of worlds `ids` with their current observation (as PegEnv resets)."""
        o = self.obs()
        if self.hist is None:
            self.hist = o[:, None].repeat(1, self.H, 1)
        self.prev[ids] = 0
        o = self.obs()
        self.hist[ids] = o[ids, None].expand(-1, self.H, -1)

    def act(self):
        with torch.no_grad():
            a, _ = self.ac.act_mean(self.hist)
        a = a.clamp(-self.clip, self.clip)
        q = self.env.t_qpos[:, self.env.arm_q]
        return a, q + self.scale * a

    def push(self, a, active):
        self.prev = torch.where(active[:, None], a, self.prev)
        o = self.obs()
        self.hist = torch.where(active[:, None, None], torch.cat([self.hist[:, 1:], o[:, None]], 1), self.hist)


class Release:
    """Scripted release after a successful insertion: hold, open, lift straight up."""

    def __init__(self, env: PickEnv, open_s=0.4, lift=0.08, v=0.08):
        self.env, self.n_open, self.lift, self.v = env, int(open_s / env.control_dt), lift, v
        N, dev = env.N, env.device
        self.q = torch.zeros(N, 6, device=dev)
        self.p0 = torch.zeros(N, 3, device=dev)
        self.R0 = torch.eye(3, device=dev).repeat(N, 1, 1)
        self.t = torch.zeros(N, dtype=torch.long, device=dev)
        self.T = self.n_open + int(lift / v / env.control_dt) + 10

    def start(self, ids):
        env = self.env
        q = env.t_qpos[:, env.arm_q]
        p, R = env.chain.fk(q)
        self.q[ids], self.p0[ids], self.R0[ids], self.t[ids] = q[ids], p[ids], R[ids], 0

    def act(self, active):
        env = self.env
        h = ((self.t - self.n_open).clamp(min=0).float() * self.v * env.control_dt).clamp(max=self.lift)
        # back out along the tcp approach axis (tcp -z): straight up dragged the part out of the port with
        # the tilted grasp (seated after release 68% vs 76% inserted)
        p = self.p0 - h[:, None] * self.R0[:, :, 2]
        q, _, _ = env.chain.ik(p, self.R0, self.q, iters=6, q_rest=env.home)
        self.q = torch.where((active & (self.t >= self.n_open))[:, None], q, self.q)
        self.t += active.long()
        return self.q, self.t >= self.T


def run(env: PickEnv, picker, inserter: Inserter, pick_s=15.0, insert_s=10.0, video=None, release=True):
    N, dev = env.N, env.device
    env.reset_all()
    picker.reset(N, dev)
    T_pick = int(pick_s / env.control_dt)
    T_ins = int(insert_s / env.control_dt)
    v = env.lay.variants[0]
    depth, tol = v.depth, 0.0002
    rel = Release(env)
    T_rel = rel.T if release else 0
    phase = torch.zeros(N, dtype=torch.long, device=dev)        # 0 pick, 1 insert, 2 release, 3 finished
    t_hand = torch.full((N,), -1, device=dev)
    hand_err = torch.zeros(N, 3, device=dev)
    hand_tilt = torch.zeros(N, device=dev)
    success = torch.zeros(N, dtype=torch.bool, device=dev)
    t_ins = torch.zeros(N, dtype=torch.long, device=dev)
    frames = []
    for t in range(T_pick + T_ins + T_rel):
        a_pick, done = picker.act(env) if (phase == 0).any() else (None, None)
        go = (phase == 0) & (done if done is not None else False)
        go |= (phase == 0) & (t >= T_pick - 1)
        if go.any():
            p, R = env.plug_in_port()
            hand_err[go] = p[go] * 1e3
            hand_tilt[go] = torch.rad2deg(torch.acos(R[go][:, 2, 2].clamp(-1, 1)))
            t_hand[go] = t
            inserter.start(go.nonzero()[:, 0])
            phase[go] = 1
        a_ins, q_ins = inserter.act() if inserter.hist is not None else (None, None)
        # env.step takes encoder-frame targets (PickCfg.phys enc_offset_deg); the insertion policy and
        # the release work on the true joints here, so their targets get the offset added
        cmd = env.t_ctrl[:, env.arm_act] + env.enc_off
        grip = torch.where(phase == 0, a_pick[:, 6] if a_pick is not None else -1.0, torch.full((N,), -1.0, device=dev))
        grip = torch.where(phase >= 2, torch.ones_like(grip), grip)
        if a_pick is not None:
            cmd[phase == 0] = a_pick[phase == 0, :6]
        if q_ins is not None:
            cmd[phase == 1] = (q_ins + env.enc_off)[phase == 1]
        if (phase == 2).any():
            q_rel, rel_done = rel.act(phase == 2)
            cmd[phase == 2] = (q_rel + env.enc_off)[phase == 2]
            phase = torch.where((phase == 2) & rel_done, torch.full_like(phase, 3), phase)
        env.step(torch.cat([cmd, grip[:, None]], 1))
        if q_ins is not None:
            inserter.push(a_ins, phase == 1)
        t_ins += (phase == 1).long()
        p, R = env.plug_in_port()
        tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
        seated = (p[:, 2] < -(depth - tol)) & (tilt <= 2.0)
        new = (phase == 1) & seated
        success |= new
        if release and new.any():
            rel.start(new.nonzero()[:, 0])
        phase = torch.where(new, torch.full_like(phase, 2 if release else 3), phase)
        phase = torch.where((phase == 1) & (t_ins >= T_ins), torch.full_like(phase, 3), phase)
        if video is not None and t % 2 == 0:
            frames.append({k: x[:video].cpu().numpy() for k, x in env.render().items()})
        if (phase == 3).all():
            break
    near = env.in_hand()
    held = near | success                     # still in hand at the end, or inserted
    released = success & ~near
    p, R = env.plug_in_port()
    tilt = torch.rad2deg(torch.acos(R[:, 2, 2].clamp(-1, 1)))
    seated_after = success & (p[:, 2] < -(depth - tol)) & (tilt <= 2.0) & (released if release else True)
    return dict(success=success, handoff=t_hand >= 0, hand_err_mm=hand_err, hand_tilt=hand_tilt,
                t_hand_s=t_hand * env.control_dt, t_insert_s=t_ins * env.control_dt, held=held,
                released=released, seated_after=seated_after), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", choices=["expert", "act"], default="expert")
    ap.add_argument("--act", default="")
    ap.add_argument("--ensemble", type=float, default=0.01, help="ACT temporal ensembling weight; 0 = chunks of 10")
    ap.add_argument("--no_settle", action="store_true", help="hand off on ACT's done flag only")
    ap.add_argument("--no_gate", action="store_true", help="hand off on done even with an empty gripper")
    ap.add_argument("--insert", required=True)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=128)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--spec_parts", action="store_true", help="draw a USB-C part within tolerance per batch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cam_fps", type=float, default=30.0, help="ACT camera frame clock (0: fresh frames)")
    ap.add_argument("--phys_json", default="{}", help="override PickCfg.phys entries, e.g. enc_offset_deg")
    ap.add_argument("--fric_comp", default="", help='friction feed-forward, e.g. {"scale":0.7,"ramp":0.01}')
    a = ap.parse_args()
    phys = {**PickCfg().phys, **json.loads(a.phys_json)}
    res = []
    for b in range((a.episodes + a.nworld - 1) // a.nworld):
        v = B.sample_reader(1, a.seed + b)[0] if a.spec_parts else None
        env = PickEnv(PickCfg(nworld=a.nworld, seed=a.seed + b, render=a.pick == "act", phys=phys,
                                fric_comp=json.loads(a.fric_comp) if a.fric_comp else None), variant=v)
        picker = (ActPick(a.act, env.device, ensemble=a.ensemble, settle=not a.no_settle, cam_fps=a.cam_fps, seed=a.seed + b,
                          gate=not a.no_gate)
                  if a.pick == "act" else ExpertPick(env))
        ins = Inserter(a.insert, env)
        r, _ = run(env, picker, ins)
        res.append({k: x.cpu().numpy() for k, x in r.items()})
        print(f"batch {b}: success {r['success'].float().mean():.3f}", flush=True)
        del env
    R = {k: np.concatenate([x[k] for x in res]) for k in res[0]}
    n = len(R["success"])
    k = int(R["success"].sum())
    lo, hi = wilson(k, n)
    he = R["hand_err_mm"]
    out = dict(pick=a.pick, insert=a.insert, act=a.act, phys=json.loads(a.phys_json), fric_comp=a.fric_comp, n=n, success=k / n, ci=[lo, hi],
               released=float(R["released"].mean()), seated_after_release=float(R["seated_after"].mean()),
               handoff=float(R["handoff"].mean()), held=float(R["held"].mean()),
               hand_lat_mm=dict(median=float(np.median(np.linalg.norm(he[:, :2], axis=1))),
                                p90=float(np.quantile(np.linalg.norm(he[:, :2], axis=1), 0.9))),
               hand_gap_mm=dict(median=float(np.median(he[:, 2])), p10=float(np.quantile(he[:, 2], 0.1)),
                                p90=float(np.quantile(he[:, 2], 0.9))),
               hand_tilt_deg=dict(median=float(np.median(R["hand_tilt"])), p90=float(np.quantile(R["hand_tilt"], 0.9))),
               t_insert_s=dict(median=float(np.median(R["t_insert_s"][R["success"]])) if k else None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    np.savez(Path(a.out).with_suffix(".npz"), **R)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
