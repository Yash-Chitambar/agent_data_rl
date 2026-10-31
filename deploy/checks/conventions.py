"""The deploy code against the sim it mirrors (GPU, no hardware):

  1. Robot (FlexPoint YAM, model frame) forward kinematics = the PickEnv scene chain
  2. ready pose = PickEnv.q_ready; config act.start_q_model = it
  3. config handoff.p_nom = handoff.nominal_preinsert for the scene (the gate's pre-insert point)
  4. InsertStage observation = pipeline Inserter observation on the same PickEnv state
  5. config models: insertion scale / history / clip from its run config; ACT state 13 (cmd_err)
  6. joint offsets round trip through SimBackend (read motor -> + offsets = the sim encoder reading)
  7. gripper map round trip; FlexChecker accepts the ready pose and refuses a pose in the table

    CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl .venv/bin/python -m deploy.checks.conventions
"""
import json

import numpy as np
import torch

from blind_insertion.envs.pick_env import PickCfg, PickEnv
from blind_insertion.pipeline import handoff as H
from blind_insertion.pipeline.pick_insert import Inserter
from blind_insertion.scenes import pc_bench as S
from deploy import conventions as C
from deploy.backends import SimBackend
from deploy.runner import load_config
from deploy.safety_task import FlexChecker
from deploy.stages import InsertStage


def main():
    cfg = load_config()
    ok = True

    def check(name, err, tol):
        nonlocal ok
        good = err <= tol
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {name}: {err:.2e} (tol {tol:.0e})")

    env = PickEnv(PickCfg(nworld=8, seed=0, render=False, phys={}))
    env.reset_all()
    dev = env.device
    robot = C.Robot("cpu")
    g = torch.Generator(device=dev).manual_seed(0)
    q = env.q_ready + 0.4 * (torch.rand(8, 6, generator=g, device=dev) - 0.5)
    p1, R1 = env.chain.fk(q)
    p2, R2 = robot.chain.fk(q.cpu())
    check("1 fk position (m)", (p1.cpu() - p2).abs().max().item(), 1e-6)
    check("1 fk rotation", (R1.cpu() - R2).abs().max().item(), 1e-6)

    qr = robot.ready()
    check("2 ready pose vs PickEnv.q_ready (rad)", np.abs(qr - env.q_ready.cpu().numpy()).max(), 1e-4)
    check("2 config act.start_q_model (rad)", np.abs(qr - np.array(cfg["act"]["start_q_model"])).max(), 1e-4)

    p_nom = np.array(H.nominal_preinsert(S.PORT_XY, env.body_z, env.gripper))
    check("3 config handoff.p_nom (m)", np.abs(p_nom - np.array(cfg["handoff"]["p_nom"])).max(), 1e-5)
    print(f"     p_nom = {p_nom.round(6).tolist()}, ready = {qr.round(5).tolist()}")

    ins = Inserter(cfg["models"]["insert"], env)
    qd = 0.3 * (torch.rand(8, 6, generator=g, device=dev) - 0.5)
    env.t_qpos[:, env.arm_q] = q
    env.t_qvel[:, env.arm_dofs] = qd
    ins.prev = torch.randn(8, 6, generator=g, device=dev)
    o_sim = ins.obs().cpu()
    st = InsertStage(cfg["models"]["insert"], cfg["models"]["insert_config"], robot)
    err = 0.0
    for i in range(8):
        st.prev = ins.prev[i].cpu()
        err = max(err, (st.obs(q[i].cpu().numpy(), qd[i].cpu().numpy()) - o_sim[i]).abs().max().item())
    check("4 insertion observation vs Inserter.obs", err, 1e-4)

    rc = json.load(open(cfg["models"]["insert_config"]))["env"]
    check("5 insertion scale = run config", abs(st.scale - rc["action_scale"]) + abs(ins.scale - rc["action_scale"]), 0)
    # (the sim Inserter reads the clip from config.json beside the checkpoint: the staged copy has none)
    check("5 insertion history / clip = run config", abs(st.H - rc["history"]) + abs(st.clip - rc["action_clip"]), 0)
    m = torch.jit.load(cfg["models"]["act"], map_location="cpu")
    check("5 ACT state dim 13 (cmd_err)", abs(m.s_mean.shape[0] - 13), 0)
    del env

    off = C.joint_offsets(cfg["side"])
    gc = cfg["gripper"]
    gm = C.GripperMap(gc["open"], gc["close"], gc["m_open"], gc.get("g_reader"), gc.get("m_reader", 0.019))
    be = SimBackend(cfg, off, gm, seed=3, device=str(dev), phys_json={"q_noise": 0.0, "q_bits": 0, "enc_offset_deg": [0] * 6})
    q_mot, _, gr, _ = be.read()
    st7 = be.env.state()[0].cpu().numpy()
    check("6 SimBackend motor + offsets = sim encoder reading (rad)", np.abs(q_mot + off - st7[:6]).max(), 1e-6)
    check("7 gripper map round trip (m)", abs(gm.finger_m(gr) - st7[6]), 1e-6)
    print(f"     offsets (deg) {np.degrees(off).round(2).tolist()}, gripper uncalibrated: {not gm.calibrated}")
    be.close()

    sc = cfg["safety"]
    fc = FlexChecker(off, sc["clearance"], sc["tip_clearance"], sc.get("body_clearance"), sc.get("table_z", 0.0))
    why = fc.violation(qr - off)
    ok &= why is None
    print(f"{'ok  ' if why is None else 'FAIL'} 7 FlexChecker: ready pose -> {why or 'safe'}")
    low = robot.chain.ik(torch.tensor([[0.25, 0.2, -0.02]]), (robot.R_base @ robot.R_top)[None], torch.as_tensor(qr[None], dtype=torch.float32),
                         iters=200, q_rest=robot.home)[0][0].numpy()
    why = fc.violation(low - off)
    ok &= why is not None
    print(f"{'ok  ' if why else 'FAIL'} 7 FlexChecker: tcp 2 cm below the table -> {why}")
    print("CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
