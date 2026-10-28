"""Real rig: where is the table in the model frame? Hover the fingertips over the table, measure the gap.

pick1 (2026-10-28) stopped on the -15 mm fingertip limit while the operator saw a gap under the
fingertips, the model putting them at -3.8 mm: the real table lies below the model's z = 0. Here
the arm hovers its open fingertips (FlexPoint tcp = the tip ends) HOVER_Z above model z = 0 at three
table spots around the reader (clear of the reader and the PC), each settled with the trim; you
type the gap from the fingertip ends to the table (mm, ruler). Table plane in the model frame:
z = tcp z - gap at each spot, fitted as z0 + a x + b y. Path: ready -> above each spot at
ABOVE_Z -> down to HOVER_Z -> back up -> ... -> ready -> rest.

    python -m deploy.checks.table_probe --name table_probe_001     (estop.py running, scene placed)
"""
import argparse
import json

import numpy as np
import torch

from deploy import conventions as C
from deploy.runner import load_config

SPOTS = ((0.204, 0.165), (0.150, 0.225), (0.265, 0.190))   # model frame xy (m), around the reader
ABOVE_Z, HOVER_Z = 0.080, 0.015


def poses(robot, q_seed):
    """IK (top-down, plug +x) for each spot at ABOVE_Z and HOVER_Z, warm-started along the way."""
    R = (robot.R_base @ robot.R_top)[None]
    out, q = [], torch.as_tensor(q_seed, dtype=torch.float32)[None]
    for x, y in SPOTS:
        pair = []
        for z in (ABOVE_Z, HOVER_Z):
            q, pe, *_ = robot.chain.ik(torch.tensor([[x, y, z]]), R, q, iters=200, q_rest=robot.home[None])
            assert pe.item() < 1e-3, f"IK failed at {(x, y, z)}"
            pair.append(q[0].numpy().astype(float))
        out.append(pair)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    import estop
    from deploy.backends import RealBackend
    from deploy.checks.gripper_cal import wait_input
    from deploy.recovery import supported_shutdown
    out = C.REPO / "data/task_h" / f"{a.name}.json"
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if estop.stopped():
        raise SystemExit(f"{estop.STOP_FILE} is set: clear it in estop.py first")
    cfg = load_config()
    off = C.joint_offsets(cfg["side"])
    robot = C.Robot()
    q_ready = np.asarray(cfg["act"]["start_q_model"], float)
    targets = poses(robot, q_ready)
    be = RealBackend(cfg, off)
    q0 = be.read()[0].copy()
    rep, ok = dict(spots=[], hover_z=HOVER_Z), False
    try:
        if np.max(np.abs(q0[1:3])) > 0.05:
            raise RuntimeError(f"not at the folded rest: joints 2-3 {q0[1:3].round(3).tolist()}")
        g_open = cfg["gripper"]["open"]
        be.move_to(q_ready - off, grip01=g_open, settle_trim=True)
        for (x, y), (q_above, q_hover) in zip(SPOTS, targets):
            be.move_to(q_above - off, grip01=g_open, settle_trim=True)
            be.move_to(q_hover - off, grip01=g_open, settle_trim=True)
            q = be.read()[0] + off
            tcp = robot.fk(q)[0].numpy().astype(float)
            print(f"hovering at model x {tcp[0] * 1e3:.0f}, y {tcp[1] * 1e3:.0f} mm, fingertips at model z {tcp[2] * 1e3:.1f} mm", flush=True)
            gap = None
            while gap is None:
                s = wait_input("Gap from the fingertip ends to the table, mm (ruler; 'x' if they touch): ").strip()
                try:
                    gap = 0.0 if s.lower() == "x" else float(s)
                except ValueError:
                    print("a number in mm, please", flush=True)
            rep["spots"].append(dict(tcp_model=tcp.tolist(), gap_mm=gap, table_z_model=float(tcp[2] - gap / 1e3),
                                     q_model=q.tolist()))
            print(f"-> table at model z {(tcp[2] - gap / 1e3) * 1e3:+.1f} mm here", flush=True)
            be.move_to(q_above - off, grip01=g_open, settle_trim=True)
        be.move_to(q_ready - off, grip01=g_open, settle_trim=True)
        be.move_to(q0, grip01=g_open)
        ok = True
        P = np.array([[1.0, s["tcp_model"][0], s["tcp_model"][1]] for s in rep["spots"]])
        z = np.array([s["table_z_model"] for s in rep["spots"]])
        coef = np.linalg.lstsq(P, z, rcond=None)[0]
        rx, ry = C.S.READER_XY if hasattr(C, "S") else (0.204, 0.221)
        rep["plane"] = dict(z0=float(coef[0]), a=float(coef[1]), b=float(coef[2]),
                            z_at_reader=float(coef @ [1, rx, ry]), z_at_port=float(coef @ [1, 0.321, 0.237]))
        print(f"table plane (model frame): z = {coef[0] * 1e3:+.1f} mm {coef[1] * 1e3:+.1f} mm/m x {coef[2] * 1e3:+.1f} mm/m y; "
              f"at the reader {rep['plane']['z_at_reader'] * 1e3:+.1f} mm, at the port {rep['plane']['z_at_port'] * 1e3:+.1f} mm", flush=True)
        rep["result"] = "completed"
    except BaseException as exc:      # noqa: BLE001  (SafetyStop, KeyboardInterrupt: the arm holds)
        rep["result"] = repr(exc)
        if not ok:
            estop.STOP_FILE.write_text("stop: table_probe failed\n")
    finally:
        rep["motion_trace"] = be.arm.motion_trace
        out.write_text(json.dumps(rep, indent=1) + "\n")
        print(rep["result"], "->", out, flush=True)
        if ok:
            be.close()
        else:
            print("Arm holds powered. Do not terminate this process.", flush=True)
            supported_shutdown(be)


if __name__ == "__main__":
    main()
