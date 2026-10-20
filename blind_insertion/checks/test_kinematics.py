"""FK/Jacobian/IK parity between torch Chain and MuJoCo (CPU)."""
import mujoco
import numpy as np
import torch

from blind_insertion.assets import builder as B
from blind_insertion.envs.kinematics import Chain


def check(robot):
    m, lay = B.build(robot, B.sample_study_pegs(2, 0))
    d = mujoco.MjData(m)
    ch = Chain(m, lay.tcp_site, lay.arm_qadr, device="cuda")
    rng = np.random.default_rng(0)
    n = len(lay.arm_qadr)
    lo, hi = ch.lo.cpu().numpy(), ch.hi.cpu().numpy()
    qs = rng.uniform(lo, hi, size=(64, n))
    p, R, J = ch.fk(torch.tensor(qs, dtype=torch.float32, device="cuda"), with_jac=True)
    perr, rerr, jerr = 0, 0, 0
    for i, q in enumerate(qs):
        d.qpos[lay.arm_qadr] = q
        mujoco.mj_forward(m, d)
        perr = max(perr, np.abs(d.site_xpos[lay.tcp_site] - p[i].cpu().numpy()).max())
        rerr = max(rerr, np.abs(d.site_xmat[lay.tcp_site].reshape(3, 3) - R[i].cpu().numpy()).max())
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, lay.tcp_site)
        Jm = np.vstack([jp, jr])[:, lay.arm_dofadr]
        jerr = max(jerr, np.abs(Jm - J[i].cpu().numpy()).max())
    # IK round trip
    qt = torch.tensor(qs[:32], dtype=torch.float32, device="cuda")
    pt, Rt = ch.fk(qt)
    q0 = torch.tensor(lay.robot.home, dtype=torch.float32, device="cuda").expand(32, n)
    qi, pe, re = ch.ik(pt, Rt, q0 + 0.3 * (qt - q0), iters=100, q_rest=q0)
    print(f"{robot}: FK pos err {perr:.2e} m, rot err {rerr:.2e}, jac err {jerr:.2e}; "
          f"IK median pos err {pe.median():.2e} m, rot {re.median():.2e} rad")
    assert perr < 1e-5 and rerr < 1e-5 and jerr < 1e-4


if __name__ == "__main__":
    check("flexiv")
    check("yam")
