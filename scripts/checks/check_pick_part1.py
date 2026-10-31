"""Part 1 (pick) checks: grasp frame, reader region file, and a short GPU expert run.

1. reader_grasp at reader yaw 0: grasp_site y = world y, z = (sin 60, 0, -cos 60); fingertips
   TIP_CLEARANCE above the table; some pad spheres reach the side face.
2. results/usbc/policy/pick/reader_region.json has the task_layout reader_region keys, samples
   inside the stated ranges, |yaw| within the stated range; task_layout.json embeds it.
3. GPU: 64 worlds of PickTask + PickExpert with all DR on the Part 1 layout: >= 90 % success
   (the gate itself, n >= 1000, is scripts/policy/pick_expert_eval.py), no IK failures.

    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python scripts/checks/check_pick_part1.py [--no_gpu]
"""
import argparse, json, math
from pathlib import Path
import mujoco, numpy as np, torch
from blind_insertion.scenes import pc_bench as S, physical_rig as R, reader_grasp as G

ap = argparse.ArgumentParser(); ap.add_argument('--no_gpu', action='store_true'); a = ap.parse_args()
prof = json.loads(Path('data/real_yam/usbc_lab_20261027/rig_profile_provisional_v0.json').read_text())
m, l = S.build_scene(detailed=False, bimanual=True, support_mode='free', profile=prof, measured_support=True)
d = mujoco.MjData(m); R.reference_pose(m, d, l); S.place_reader(m, d, l, xy=(.36, .1), yaw_deg=0.); mujoco.mj_forward(m, d)
hand = G.hand_geometry(m, d)
off, pad_h = G.grasp_offset(l, hand, G.TILT_DEG)
q = d.qpos[l.peg_qadr:l.peg_qadr + 7]; Rr = np.zeros(9); mujoco.mju_quat2Mat(Rr, q[3:]); Rr = Rr.reshape(3, 3)
W = G.grasp_poses(torch.tensor(q[None, :3]), torch.tensor(Rr[None]), off, G.TILT_DEG)
Rs = W['grasp'][1][0].numpy(); t = math.radians(G.TILT_DEG)
assert np.allclose(Rs[:, 1], [0, 1, 0], atol=1e-6), Rs
assert np.allclose(Rs[:, 2], [math.sin(t), 0, -math.cos(t)], atol=1e-6), Rs
top = float(m.geom('table').pos[2] + m.geom('table').size[2])
tips = (hand['tip_vertices'] @ Rs.T + W['grasp'][0][0].numpy())[:, 2].min() - top
assert abs(tips - G.TIP_CLEARANCE) < 1e-5, tips
assert 0 < pad_h[0] < pad_h[1] <= 2 * G.reader_dims(l)[3] + 1e-3, pad_h
print(f'grasp frame ok; fingertips {tips*1e3:.2f} mm above the table; pads touch {pad_h[0]*1e3:.1f}-{pad_h[1]*1e3:.1f} mm')

reg = json.loads(Path('results/usbc/policy/pick/reader_region.json').read_text())
for k in ('samples_xy_yaw', 'cell_m', 'yaw_bin_deg', 'x_range_m', 'y_range_m', 'yaw_range_deg', 'centre_xy_m'):
    assert k in reg, k
s = np.array(reg['samples_xy_yaw'])
assert (s[:, 0] >= reg['x_range_m'][0]).all() and (s[:, 0] <= reg['x_range_m'][1]).all()
assert (s[:, 1] >= reg['y_range_m'][0]).all() and (s[:, 1] <= reg['y_range_m'][1]).all()
assert (np.abs(s[:, 2]) + reg['yaw_bin_deg'] / 2 <= max(map(abs, reg['yaw_range_deg'])) + 1e-6).all()
lay = json.loads(Path('results/usbc/policy/pick/task_layout.json').read_text())
assert lay['reader_region']['samples_xy_yaw'] == reg['samples_xy_yaw']
print(f'reader region ok: {len(s)} (cell, yaw bin) samples, x {reg["x_range_m"]}, y {reg["y_range_m"]}, yaw {reg["yaw_range_deg"]}')

if not a.no_gpu:
    import warp as wp
    wp.config.quiet = True
    from blind_insertion.envs.bimanual_pick import PickExpert, PickTask, PickTaskCfg
    from blind_insertion.envs.bimanual_runtime import BimanualCfg
    torch.manual_seed(7)
    task = PickTask(PickTaskCfg(physics=BimanualCfg(nworld=64, device='cuda:0', task_layout='results/usbc/policy/pick/task_layout.json')))
    task.reset(); ex = PickExpert(task, seed=7); ex.reset()
    assert ex.ik_ok.all(), 'IK failure in the Part 1 region'
    for _ in range(task.max_steps - 1):
        task.step(ex.act())
        if ex.done.all(): break
    met = task.pick_metrics()
    ok = ex.done & ~task.failed & (met['lift_m'] > .06) & (met['tilt_deg'] < 3)
    print(f'GPU expert: {int(ok.sum())}/64 lifted flat')
    assert ok.float().mean() >= .9
print('check_pick_part1: PASS')
