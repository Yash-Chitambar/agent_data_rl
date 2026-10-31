"""ACT on real frames against the sim render of the same pose: the first real-image test, with no
arm motion beyond the slow move to the ready pose.

  capture (rig)  move to the ready pose (gripper open), save the raw ZED / wrist frames, the
                 policy crops and the joints to data/task_h/frames/<name>/; the reader and PC
                 must be in place (README, scene placement).
  eval (GPU)     render the nominal sim scene (no layout or render randomization) at the same model
                 joints; run ACT on real and on sim images with the same state; print the first
                 action chunk of both (targets - q, gripper, done) and save a side-by-side image.

A large difference is expected while the image gap is open; it says which way ACT goes on real
frames before any motion. Fixes go into the demos' domain randomization, not real data.

    python -m deploy.checks.act_frames capture --name ready1      (rig; estop.py running)
    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl python -m deploy.checks.act_frames eval --name ready1
"""
import argparse
import json

import cv2
import numpy as np
import torch

from deploy import conventions as C
from deploy.runner import load_config

OUT = C.REPO / "data/task_h/frames"


def capture(a):
    from deploy.backends import RealBackend
    cfg = load_config()
    off = C.joint_offsets(cfg["side"])
    d = OUT / a.name
    if d.exists():
        raise SystemExit(f"refusing to overwrite {d}")
    be = RealBackend(cfg, off)
    q_start = be.read()[0].copy()
    try:
        be.move_to(np.asarray(cfg["act"]["start_q_model"]) - off, grip01=cfg["gripper"]["open"], settle_trim=True)
        q, qd, g, _ = be.read()
        raw = {k: c.latest()[0] for k, c in be.cams.cams.items()}
        pol, age, _ = be.frames()
        # Persist observations before attempting the return, including on return failure.
        d.mkdir(parents=True, exist_ok=False)
        for k in raw:
            cv2.imwrite(str(d / f"{k}_raw.png"), raw[k])
            cv2.imwrite(str(d / f"{k}_policy.png"), cv2.cvtColor(pol[k], cv2.COLOR_RGB2BGR))
        (d / "state.json").write_text(json.dumps(dict(q_motor=q.tolist(), q_model=(q + off).tolist(), grip=g,
            age_ms={k: v * 1e3 for k, v in age.items()}), indent=1))
        be.move_to(q_start, grip01=cfg["gripper"]["open"])
        if np.max(np.abs(be.read()[0] - q_start)) > 0.02:
            raise RuntimeError("return pose not attained within 0.02 rad")
    except BaseException as exc:
        be.hold()
        d.mkdir(parents=True, exist_ok=True)
        (d / "failure.json").write_text(json.dumps(dict(error=repr(exc),
            q_start_motor=q_start.tolist(), motion_trace=be.arm.motion_trace,
            controller=be.arm.info(), note="Failed attempt, not a ready-pose observation"), indent=1))
        for name, camera in be.cams.cams.items():
            frame, stamp, count = camera.latest()
            if frame is not None:
                cv2.imwrite(str(d / f"{name}_failure_raw.png"), frame)
        print(f"Capture/return stopped: {exc}. Arm remains powered. Do not terminate this process.", flush=True)
        # A failed return must never disable a raised arm automatically.
        from deploy.recovery import supported_shutdown
        supported_shutdown(be)
        return
    be.close()
    d = OUT / a.name
    d.mkdir(parents=True, exist_ok=True)
    for k in raw:
        cv2.imwrite(str(d / f"{k}_raw.png"), raw[k])
        cv2.imwrite(str(d / f"{k}_policy.png"), cv2.cvtColor(pol[k], cv2.COLOR_RGB2BGR))
    (d / "state.json").write_text(json.dumps(dict(q_motor=q.tolist(), q_model=(q + off).tolist(), grip=g,
                                                  age_ms={k: v * 1e3 for k, v in age.items()}), indent=1))
    print(f"saved {d}")


def evaluate(a):
    from blind_insertion.envs.pick_env import PickCfg, PickEnv
    cfg = load_config()
    d = OUT / a.name
    st = json.loads((d / "state.json").read_text())
    gm = C.GripperMap(**{k: cfg["gripper"][k] for k in ("open", "close", "m_open", "g_reader", "m_reader")})
    q, finger = np.array(st["q_model"]), gm.finger_m(st["grip"])
    real = {k: cv2.cvtColor(cv2.imread(str(d / f"{k}_policy.png")), cv2.COLOR_BGR2RGB) for k in ("top", "wrist")}
    env = PickEnv(PickCfg(nworld=1, seed=0, phys={}, rand={}, port_xy_range=(0.0, 0.0), pc_yaw_deg=0.0,
                          reader_xy_range=(0.0, 0.0), reader_yaw_deg=0.0, start_q_noise=0.0))
    env.reset_all()
    env.t_qpos[0, env.arm_q] = torch.as_tensor(q, dtype=torch.float32, device=env.device)
    fr = env.render()
    sim = {"top": fr["top"][0].cpu().numpy(), "wrist": fr["wrist_cam"][0].cpu().numpy()}
    m = torch.jit.load(cfg["models"]["act"], map_location=env.device).eval()
    s = torch.as_tensor(np.r_[q, finger, np.zeros(6)], dtype=torch.float32, device=env.device)[None]
    img = lambda x: torch.as_tensor(x, device=env.device).permute(2, 0, 1)[None].float() / 255  # noqa: E731
    np.set_printoptions(precision=3, suppress=True)
    out = {}
    with torch.no_grad():
        for name, im in (("sim", sim), ("real", real)):
            c = m(s, img(im["top"]), img(im["wrist"]))[0].cpu().numpy()
            out[name] = c
            print(f"{name:4s}: chunk[0] target - q {c[0, :6] - q}, gripper {c[0, 6]:+.2f}, done {c[0, 7]:+.2f}; "
                  f"chunk[-1] target - q {c[-1, :6] - q}")
    print(f"|real - sim| over the chunk, joints (rad): mean {np.abs(out['real'][:, :6] - out['sim'][:, :6]).mean(0)}")
    grid = np.concatenate([np.concatenate([sim[k], real[k]], 1) for k in ("top", "wrist")], 0)
    cv2.imwrite(str(d / "sim_vs_real.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"side by side (left sim, right real; top row ZED, bottom wrist): {d / 'sim_vs_real.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("capture", "eval"):
        sub.add_parser(n).add_argument("--name", required=True)
    a = ap.parse_args()
    {"capture": capture, "eval": evaluate}[a.cmd](a)
