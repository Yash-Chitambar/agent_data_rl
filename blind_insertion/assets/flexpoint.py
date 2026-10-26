"""I2RT flexible_4310 geometry with rigid contact-pad approximation.

Upstream source/license in flexpoint/. No soft-body/material calibration is implied.
Keep the existing link_6/base inertia (it already includes the gripper housing);
use upstream moving-finger inertias. Payload still requires a physical weighing.
"""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial import ConvexHull

ASSET = Path(__file__).with_suffix('') / 'flexible_4310.xml'
# Black rubber pad on each soft tip's inner face (not in the upstream CAD). Thickness is a photo
# estimate until measured with calipers; the contact surface is the pad's inner face.
PAD_THICKNESS = 0.001
PAD_LENGTH = 0.016       # from the tip end to just past the first (small) hole; user-checked against the real tips
PAD_RGBA = [.03, .03, .03, 1]
TIP_LINK_BITS = (8, 1)            # builder.FINGER_LINK_BITS: tips collide with table and fixture only


def _tip_geometry(model, data, body, s):
    """Inner face of a soft tip in the source world frame (closing axis x, approach z).

    `s` = +1 if the tip sits on the +x side. Returns the face plane u = a + b*z + c*y with
    u = s*x (the tip lies at u >= plane), the face outline points (y, z) and the tip vertices.
    """
    g = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(body).id
             and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
             and model.mesh(model.geom_dataid[i]).name == 'soft_tips')
    mid = model.geom_dataid[g]
    a0, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    V = model.mesh_vert[a0:a0 + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    u, y, z = s * V[:, 0], V[:, 1], V[:, 2]
    # face points: within 0.6 mm of the innermost vertex of their 5 mm slice along z
    band = np.floor(z / .005)
    umin = {b: u[band == b].min() for b in np.unique(band)}
    lip = min(umin.values())
    face = np.array([u[i] - umin[band[i]] < 6e-4 and umin[band[i]] < lip + 2e-3 for i in range(n)])
    A = np.stack([np.ones(face.sum()), z[face], y[face]], 1)
    _, b, c = np.linalg.lstsq(A, u[face], rcond=None)[0]
    a = (u - b * z - c * y).min()               # support plane: no tip vertex inside it
    on = (u - (a + b * z + c * y)) < 1e-3
    pts = np.stack([y[on], z[on]], 1)
    return (a, b, c), _clip_below(pts[ConvexHull(pts).vertices], z.max() - PAD_LENGTH), V


def _clip_below(poly, z0):
    """Part of a convex polygon [(y, z)] with z >= z0 (the pad covers the face's tip end, also
    across the recess behind the lip)."""
    out = []
    for p, q in zip(poly, np.roll(poly, -1, 0)):
        if p[1] >= z0:
            out.append(p)
        if (p[1] - z0) * (q[1] - z0) < 0:
            out.append(p + (q - p) * (z0 - p[1]) / (q[1] - p[1]))
    return np.array(out)


def _hull_points(V, slices=8):
    """Few-vertex convex proxy of a tip: per-slice extremes (MuJoCo Warp needs small polygons)."""
    z = V[:, 2]
    out = []
    edges = np.linspace(z.min(), z.max(), slices + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        S = V[(z >= lo) & (z <= hi)]
        for x in (S[:, 0].min(), S[:, 0].max()):
            for y in (S[:, 1].min(), S[:, 1].max()):
                out += [[x, y, S[:, 2].min()], [x, y, S[:, 2].max()]]   # reaches the tip ends
    out = np.array(out)
    return out[ConvexHull(out).vertices]


def install(spec):
    for key in list(spec.keys): spec.delete(key)
    source = mujoco.MjSpec.from_file(str(ASSET))
    model = source.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # Remove the old fingers and all references before installing the upstream pair.
    for e in list(spec.equalities):
        if e.name1 in ('left_finger', 'right_finger'):
            spec.delete(e)
    for e in list(spec.excludes):
        if any(n in ('left_finger','right_finger','left_pad','right_pad') for n in (e.bodyname1,e.bodyname2)):
            spec.delete(e)
    spec.delete(spec.actuator('gripper'))
    for name in ('left_finger','right_finger'):
        spec.delete(spec.body(name))
    for g in list(spec.geoms):
        if g.name == 'gripper_housing' or g.meshname == 'gripper':
            spec.delete(g)
    for name in ('tcp_site','grasp_site'):
        spec.delete(spec.site(name))
    for side, old in [('left','linear_module'),('right','linear_module_2')]:
        body = source.body(old)
        bid = model.body(old).id
        R = data.xmat[bid].reshape(3,3)
        sign = 1 if side == 'left' else -1
        local = lambda p: R.T @ (np.asarray(p) - data.xpos[bid])   # noqa: E731  source world -> body
        (pa, pb, pc), outline, V = _tip_geometry(model, data, old, sign)
        nu = np.array([1.0, -pc, -pb]) / np.linalg.norm([1.0, -pc, -pb])     # face normal (u, y, z), outward
        n_w = np.array([sign * nu[0], nu[1], nu[2]])                         # ... in the source world
        # visual pad: the face outline, and the same outline PAD_THICKNESS toward the other tip
        back = np.array([[sign * (pa + pb * z + pc * y), y, z] for y, z in outline])
        verts = np.concatenate([back, back - PAD_THICKNESS * n_w])
        source.add_mesh(name=side + '_pad_rubber', uservert=np.array([local(v) for v in verts]).ravel().tolist())
        body.add_geom(name=side + '_pad_visual', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=side + '_pad_rubber',
                      rgba=PAD_RGBA, contype=0, conaffinity=0, mass=0)
        # contact pad (builder.sphere_pads puts the contact spheres on its inner face): its inner
        # face lies on the rubber's inner surface, near the tip where the reader is held
        h, y0, z0 = .0017, .0007, .134
        inner = np.array([sign * (pa + pb * z0 + pc * y0), y0, z0]) - PAD_THICKNESS * n_w
        ax = n_w
        ay = np.array([0., 1, 0]) - ax[1] * ax; ay /= np.linalg.norm(ay)
        Rb = np.stack([ax, ay, np.cross(ax, ay)], 1)
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, (R.T @ Rb).flatten())
        body.add_geom(name=side+'_pad', type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=local(inner + h * n_w), quat=quat,
                      size=[h,.0072,.0082], rgba=PAD_RGBA, group=3,
                      contype=4, conaffinity=3, condim=4, priority=1,
                      friction=[1,.005,.0005],solref=[.01,1],mass=0)
        # the yellow tip itself keeps out of the table and the PC (not the reader: pads only)
        source.add_mesh(name=side + '_tip_hull', uservert=np.array([local(v) for v in _hull_points(V)]).ravel().tolist())
        body.add_geom(name=side + '_tip_hull', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=side + '_tip_hull',
                      group=3, contype=TIP_LINK_BITS[0], conaffinity=TIP_LINK_BITS[1], mass=0)
        body.name=side+'_finger'
        joint=next(iter(body.joints)); joint.name=side+'_finger'
        joint.range=[-.005,.048]; joint.actfrcrange=[-100,100]
        joint.armature=.1; joint.damping=[200,0,0]
        for g in body.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and not g.name.endswith('_tip_hull'):
                g.contype=g.conaffinity=0
                if g.meshname == 'linear_module':g.rgba=[.12,.12,.12,1]
    # Body names changed but equality names refer to the original joints.
    for e in source.equalities:
        e.name1='left_finger';e.name2='right_finger'
    for e in list(source.excludes):source.delete(e)
    for site in source.sites:site.group=4
    root=source.body('gripper');root.mass=0;root.inertia=[0,0,0]
    root.add_geom(name='gripper_housing',type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0,0,-.04],size=[.03,.05,.02],group=3,contype=4,conaffinity=3,mass=0)
    # Keep the established TCP orientation; tip midpoint comes from upstream CAD.
    source.site('grasp_site').quat=[0,1,0,0]
    spec.body('link_6').add_frame().attach_body(root,'','')
    a=spec.add_actuator(name='gripper',target='left_finger',trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.gainprm[0]=1
