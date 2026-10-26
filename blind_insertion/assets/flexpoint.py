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
# Black rubber pad (not in the upstream CAD): it fills the pocket in each soft tip's inner face,
# ~3.5-19.5 mm from the tip end (its far edge is the notch on the real tip), flush with the face.
# The grip surface is that flush face: the yellow lip at the tip end and the rubber.
PAD_THICKNESS = 0.001   # visible rubber sheet (the CAD pocket is ~4.4 mm deep); photo estimate
PAD_RGBA = [.03, .03, .03, 1]
TIP_LINK_BITS = (8, 1)            # builder.FINGER_LINK_BITS: tips collide with table and fixture only


def _tip_geometry(model, data, body, s):
    """Inner face and rubber pocket of a soft tip in the source world frame (closing axis x,
    approach z; tip end at max z).

    `s` = +1 if the tip sits on the +x side, u = s*x. Returns the face plane u = a + b*z + c*y
    (no tip vertex inside it), the pocket's z range (lo, hi), depth below the face, outline
    [(y, z)], and the tip vertices.
    """
    g = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(body).id
             and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH
             and model.mesh(model.geom_dataid[i]).name == 'soft_tips')
    mid = model.geom_dataid[g]
    a0, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    V = model.mesh_vert[a0:a0 + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
    u, y, z = s * V[:, 0], V[:, 1], V[:, 2]
    dz = z.max() - z                             # distance from the tip end
    # face plane: fit to the innermost vertices of 0.5 mm slices 20-45 mm from the end (past the
    # pocket), then make it the support plane of the face there and of the lip at the end
    sl = np.floor(dz / 5e-4).astype(int)
    umin = np.full(sl.max() + 1, np.inf)
    np.minimum.at(umin, sl, u)
    face = (dz > .020) & (dz < .045) & (u - umin[sl] < 3e-4)
    A = np.stack([np.ones(face.sum()), z[face], y[face]], 1)
    _, b, c = np.linalg.lstsq(A, u[face], rcond=None)[0]
    r = u - b * z - c * y                        # offset from a plane with the face's slope
    a = r[(dz < .003) | ((dz > .020) & (dz < .045))].min()
    rec = r - a                                  # recess below the face plane
    # pocket: from the end of the lip to the first slice back on the face plane (innermost vertex
    # of each 0.5 mm slice; slices without vertices lie on large flat facets and are skipped)
    rmin = np.full(sl.max() + 1, np.inf)
    np.minimum.at(rmin, sl, rec)
    idx = np.arange(len(rmin))
    deep = idx[(rmin > 1.5e-3) & np.isfinite(rmin) & (idx * 5e-4 < .030)]
    flat = idx[rmin < 5e-4]
    for first in deep:                           # first recessed run >= 5 mm (not the rounded end)
        nxt = flat[flat > first]
        if len(nxt) and (nxt.min() - first) * 5e-4 >= .005:
            break
    lo_dz = (flat[flat < first].max() + 1) * 5e-4
    hi_dz = nxt.min() * 5e-4
    inside = (dz > lo_dz - 5e-4) & (dz < hi_dz)    # the floor's facet corners sit on the lip slice
    floor = inside & (rec > 1.5e-3) & (rec < 6e-3)
    depth = float(np.median(rec[floor]))
    pts = np.stack([y[floor], z[floor]], 1)
    return (a, b, c), (z.max() - hi_dz, z.max() - lo_dz), depth, pts[ConvexHull(pts).vertices], V


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
        (pa, pb, pc), (z_lo, z_hi), depth, outline, V = _tip_geometry(model, data, old, sign)
        nu = np.array([1.0, -pc, -pb]) / np.linalg.norm([1.0, -pc, -pb])     # face normal (u, y, z), outward
        n_w = np.array([sign * nu[0], nu[1], nu[2]])                         # ... in the source world
        # visual pad: a PAD_THICKNESS sheet in the pocket outline, flush with the face (1e-4 proud, no z-fighting)
        front = np.array([[sign * (pa + pb * z + pc * y), y, z] for y, z in outline]) - 1e-4 * n_w
        verts = np.concatenate([front, front + (PAD_THICKNESS + 1e-4) * n_w])
        source.add_mesh(name=side + '_pad_rubber', uservert=np.array([local(v) for v in verts]).ravel().tolist())
        body.add_geom(name=side + '_pad_visual', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=side + '_pad_rubber',
                      rgba=PAD_RGBA, contype=0, conaffinity=0, mass=0)
        # contact pad (builder.sphere_pads puts the contact spheres on its inner face): on the flush
        # face, from the tip end (yellow lip) over the rubber to the pocket's far edge
        z_end = V[:, 2].max()
        h, y0, hz = .0017, .0007, (z_end - z_lo) / 2
        z0 = z_end - hz
        inner = np.array([sign * (pa + pb * z0 + pc * y0), y0, z0])
        ax = n_w
        ay = np.array([0., 1, 0]) - ax[1] * ax; ay /= np.linalg.norm(ay)
        Rb = np.stack([ax, ay, np.cross(ax, ay)], 1)
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, (R.T @ Rb).flatten())
        body.add_geom(name=side+'_pad', type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=local(inner + h * n_w), quat=quat,
                      size=[h,.0072,hz], rgba=PAD_RGBA, group=3,
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
