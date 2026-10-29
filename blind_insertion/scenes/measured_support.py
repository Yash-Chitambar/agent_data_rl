"""Operator-measured table offsets; mounting-plate footprint remains estimated."""
import mujoco
import numpy as np
from blind_insertion.assets.builder import MENAGERIE, FIX_BITS

WALL_EDGE_GAP = .480
MOUNT_RISE = .023


def base_bounds():
    """Bounds of the stationary base mesh in the unchanged right model-base frame."""
    m=mujoco.MjModel.from_xml_path(str(MENAGERIE/'i2rt_yam/yam_linear_4310.xml'))
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    points=[]
    for g in range(m.ngeom):
        if m.geom_bodyid[g]!=m.body('arm').id or m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh=m.geom_dataid[g];a=m.mesh_vertadr[mesh];n=m.mesh_vertnum[mesh]
        points.append(m.mesh_vert[a:a+n]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
    vertices=np.concatenate(points)
    return vertices.min(0),vertices.max(0)


def dimensions():
    lo,hi=base_bounds()
    return {'wall_edge_x_m':float(hi[0]+WALL_EDGE_GAP),
            'mount_surface_z_m':float(lo[2]),
            'table_top_z_m':float(lo[2]-MOUNT_RISE),
            'base_min_m':lo.tolist(),'base_max_m':hi.tolist(),
            'reference_assumption':'stationary base mesh bottom is the mounting interface',
            'plate_footprint_status':'estimated envelope of both base meshes, not surveyed'}


def apply(spec):
    from blind_insertion.scenes.physical_rig import load
    dims=dimensions();lo=np.array(dims['base_min_m']);hi=np.array(dims['base_max_m'])
    table=next(g for g in spec.geoms if g.name=='table')
    # +x points toward the PC / wall. Keep other unsurveyed edges explicitly estimated.
    front=table.pos[0]-table.size[0];back=dims['wall_edge_x_m']
    table.pos[0]=(front+back)/2;table.size[0]=(back-front)/2
    table.pos[2]=dims['table_top_z_m']-table.size[2]
    # Procedural dressing was authored relative to a z=0 tabletop. Move world
    # decorations with the measured surface; task objects and robot frames have
    # independent measured placement and must not be translated here.
    for geom in spec.worldbody.geoms:
        if geom.name.startswith('detail_'):
            if np.isfinite(geom.fromto).all():
                geom.fromto[2] += dims['table_top_z_m']
                geom.fromto[5] += dims['table_top_z_m']
            else:
                geom.pos[2] += dims['table_top_z_m']
    # Shared support spans both bases; exact outer footprint awaits survey/overlay.
    pose=load()['arms']['left_base_in_right_base'];q=np.array(pose['quat']);q/=np.linalg.norm(q)
    rot=np.zeros(9);mujoco.mju_quat2Mat(rot,q)
    corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    left=corners@rot.reshape(3,3).T+pose['pos']
    xylo=np.minimum(lo[:2],left.min(0)[:2]);xyhi=np.maximum(hi[:2],left.max(0)[:2])
    spec.worldbody.add_geom(name='shared_mounting_plate',type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[*((xylo+xyhi)/2),dims['table_top_z_m']+MOUNT_RISE/2],
        size=[*((xyhi-xylo)/2),MOUNT_RISE/2],rgba=[.12,.12,.13,1],
        contype=FIX_BITS[0],conaffinity=FIX_BITS[1],mass=0)
    return dims
