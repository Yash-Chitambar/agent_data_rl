"""Restore authored YAM link proxies for the opt-in bimanual scene.

The upstream capsules approximate the arm envelope; they are not newly measured CAD.
Same-arm link contact remains disabled because adjacent authored capsules overlap.
Cross-arm, arm-object and arm-table contacts are enabled independently of visuals.
"""
import mujoco

RIGHT_ARM = 32
LEFT_ARM = 64


def configure_spec(spec):
    """Enable before compilation so dynamic collision BVHs contain these shapes."""
    for body in spec.bodies:
        side = 'left' if body.name.startswith(('left_link_', 'left_left_finger', 'left_right_finger')) or body.name == 'left_arm' else 'right'
        prefix = 'left_' if side == 'left' else ''
        local = body.name[len(prefix):]
        own, other = (LEFT_ARM, RIGHT_ARM) if side == 'left' else (RIGHT_ARM, LEFT_ARM)
        for i, g in enumerate(body.geoms):
            if local.startswith('link_') and g.group == 3 and g.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                if not g.name:g.name = f'{body.name}_collision_{i}'
                g.contype = own
                g.conaffinity = 1 | 2 | other
            if g.contype & 4 and local in ('left_finger','right_finger','link_6'):
                g.contype |= own
                g.conaffinity |= other


def apply(m):
    report = {'right': [], 'left': [], 'assumption': 'upstream link capsules; same-arm self-collision disabled'}
    for gid in range(m.ngeom):
        body = m.body(int(m.geom_bodyid[gid])).name
        side = 'left' if body.startswith(('left_link_', 'left_left_finger', 'left_right_finger')) or body == 'left_arm' else 'right'
        prefix = 'left_' if side == 'left' else ''
        local = body[len(prefix):]
        own, other = (LEFT_ARM, RIGHT_ARM) if side == 'left' else (RIGHT_ARM, LEFT_ARM)
        if local.startswith('link_') and m.geom_group[gid] == 3 and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            m.geom_contype[gid] = own
            m.geom_conaffinity[gid] = 1 | 2 | other
            report[side].append({'geom_id': gid, 'body': body})
        # Both finger pads and the housing keep existing environment/object contacts.
        # Separate arm bits enable opposing-arm contacts without introducing own-link contacts.
        if m.geom_contype[gid] & 4 and local in ('left_finger','right_finger','link_6'):
            m.geom_contype[gid] |= own
            m.geom_conaffinity[gid] |= other
    # Keep compiled broad-phase masks consistent with final per-geom flags.
    m.body_contype[:] = 0
    m.body_conaffinity[:] = 0
    for gid in range(m.ngeom):
        bid = m.geom_bodyid[gid]
        m.body_contype[bid] |= m.geom_contype[gid]
        m.body_conaffinity[bid] |= m.geom_conaffinity[gid]
    if not report['left'] or not report['right']:
        raise ValueError('missing authored collision capsules on one or both arms')
    return report
