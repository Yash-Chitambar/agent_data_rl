"""Smoke check for the scripted left hold and the pre-insert placement (policy Parts 2/3), 64 worlds.

Passes when: every hold plan solves, every guarded engage latches, no arm-arm contact, the reader is
placed at the pre-insert pose with IK converged, and with no external load the PC moves < 0.5 mm in
>= 95 % of worlds over 1 s (the full gate runs live in scripts/task/eval_left_hold.py).

    PYTHONPATH=. .venv/bin/python scripts/checks/check_left_hold.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import warp as wp
wp.config.quiet = True
import torch

from blind_insertion.envs import kinematics as K
from blind_insertion.envs.bimanual_runtime import BimanualCfg, BimanualRuntime
from blind_insertion.envs.left_hold import Harness, HoldCfg, LeftHold, RightInserter

torch.manual_seed(0)
rt = BimanualRuntime(BimanualCfg(nworld=64, device='cuda:0'))
h = Harness(rt); h.reset()
hold = LeftHold(h, HoldCfg(mode='press', force_n=35., along_m=-.02, across_m=-.0208))
p, R = h.pc_pose()
plan = hold.solve(p, R)
ql, qr = h.believed('left').clone(), h.believed('right').clone()
for k in range(int(hold.duration / .02)):
    h.step(qr, 1., hold.target(k * .02, ql), -1.)
ins = RightInserter(h)
ok = ins.place()
p0, R0 = h.pc_pose()
arm_arm = torch.zeros(h.n, dtype=torch.bool, device=h.dev)
for k in range(50):
    h.step(ins.target, -1., hold.hold_target(), -1.)
    if k % 5 == 0:
        arm_arm |= h.contacts()['arm_arm']
p1, R1 = h.pc_pose()
dp = (p1 - p0).norm(dim=1) * 1e3
fin = torch.isfinite(dp)
res = {'plans_ok': int(plan['ok'].sum()), 'latched': int(hold.latched.sum()), 'ik_ok': int(ok.sum()),
       'arm_arm': int(arm_arm.sum()), 'still_frac': float((dp[fin] < .5).float().mean()), 'finite': int(fin.sum())}
print(res)
assert res['plans_ok'] == 64 and res['latched'] == 64 and res['ik_ok'] == 64, res
assert res['arm_arm'] == 0, res
assert res['finite'] >= 63 and res['still_frac'] >= .95, res
print('PASS')
