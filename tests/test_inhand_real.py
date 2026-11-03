"""The real in-hand preset is opt in: with the flag off the scene and the distributions stay as they were."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def silver_extent(m):
    """(width along reader x, length along reader z) of the silver plate's mesh, m."""
    import mujoco
    g = m.geom('detail_reader_silver')
    i = g.dataid[0]
    v = m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, g.quat)
    b = v @ R.reshape(3, 3).T
    return np.ptp(b[:, 0]), np.ptp(b[:, 2])


def scene(**kw):
    from blind_insertion.scenes import pc_bench as S
    from deploy.bimanual.scene import CONTROLLER_PROFILE, PROFILE
    return S.build_scene(detailed=True, bimanual=True, measured_support=True, support_mode='free',
                         profile=json.loads(PROFILE.read_text()), controller_profile=CONTROLLER_PROFILE, **kw)


def test_defaults_off():
    from blind_insertion.envs.bimanual_runtime import BimanualCfg
    from blind_insertion.envs.insertion_task import InsertionCfg
    from blind_insertion.envs.left_hold import InsertCfg
    from blind_insertion.scenes import pc_bench as S
    assert BimanualCfg().inhand_real is None
    assert InsertionCfg().grasp == 'part1'
    assert InsertCfg().inhand == 'measured'
    assert inspect.signature(S.build_scene).parameters['reader_look'].default is None


def test_flag_off_scene_unchanged():
    """Silver plate at the old size and place, no shoulder geom: reader_look=None is the old scene."""
    m, lay = scene()
    assert 'detail_reader_shoulder' not in [m.geom(i).name for i in range(m.ngeom)]
    (c, half) = lay.variants[0].peg_boxes[0]
    assert abs(2 * half[2] - .04126) < 1e-5                  # the body stays 41.26 mm
    g = m.geom('detail_reader_silver')
    assert abs(g.pos[2] - (c[2] - .001)) < 1e-6
    assert np.allclose(silver_extent(m), (.0346, .0296), atol=1e-5)


def test_flag_on_only_look_differs():
    from blind_insertion.envs import inhand_real as IR
    m0, l0 = scene()
    m1, l1 = scene(reader_look=IR.load('v1')['look'])
    assert m1.ngeom == m0.ngeom + 1
    assert m1.nbody == m0.nbody and m1.nq == m0.nq and m1.nu == m0.nu
    assert np.allclose(m0.body_mass, m1.body_mass) and np.allclose(m0.body_ipos, m1.body_ipos)
    assert l0.variants[0].peg_boxes == l1.variants[0].peg_boxes  # collision shape unchanged
    assert np.allclose(silver_extent(m1), (.0296, .0346), atol=1e-5)


def test_part1_distribution_unchanged():
    from blind_insertion.envs.insertion_task import InsertionCfg, inhand_pose
    c0, c1 = InsertionCfg(), InsertionCfg()
    c1.grasp = 'real'
    R0, p0 = inhand_pose(1, c0, 'cpu', nominal=True)
    R1, p1 = inhand_pose(1, c1, 'cpu', nominal=True)       # same reference pose: the policy's in-hand input keeps its meaning
    assert torch.allclose(R0, R1) and torch.allclose(p0, p1)
    g = torch.Generator().manual_seed(0)
    a = inhand_pose(4, c0, 'cpu', g)
    g = torch.Generator().manual_seed(0)
    b = inhand_pose(4, c0, 'cpu', g)
    assert torch.allclose(a[1], b[1])


def test_real_distribution_matches_preset():
    from blind_insertion.envs import inhand_real as IR
    from blind_insertion.envs.inhand import nominal_reader_in_site
    c = IR.load('v1')['inhand']
    d = IR.sample_delta(1, 'v1', spread=0.)
    assert d[0, 1] == 0 and abs(d[0, 4] - c['mean_rotvec_deg'][1]) < 1e-5
    R, p = IR.reader_in_site(d)
    R0, p0 = nominal_reader_in_site()
    assert abs((p[0, 0] - p0[0]) * 1e3 - c['mean_mm'][0]) < 1e-3
    # the reader's mid-plane passes through the pad centroid
    pad = torch.tensor([0., 0., -IR.PAD_BEHIND_SITE])
    assert abs(float((R[0].T @ (pad - p[0]))[0])) < 1e-3
    d = IR.sample_delta(2000, 'v1')
    h = torch.tensor(c['half_range_mm'] + c['half_range_deg'])
    assert ((d - d.mean(0)).abs() <= h + .05).all()


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
