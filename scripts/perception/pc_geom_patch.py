"""Opt-in sim PC geometry for ZED studies (docs/real/ZED_MAP_20261114.md). Nothing changes unless apply() is called.

The real case, measured on the real ZED frames (case-edge fit, validated against the arm, n = 29 runs) and the
wrist-camera panel model: top face 103.5 mm along the port panel x 103 mm deep (the sim: 102.5 x 97 mm, from
operator calipers whose orientation was never confirmed), port 2 19.8 mm from the case centre toward the left
(+y) edge, i.e. 31.95 mm from the left edge (sim: 30.4 mm). Height unchanged (44.5 mm with feet).

apply() patches pc_bench.build_scene (profile width/depth) and pc_rear.layout (target_from_left) for this process.
"""
from __future__ import annotations

import copy

REAL = dict(width=0.1035, depth=0.103, target_from_left=0.1035 / 2 - 0.0198)


def apply(geom=REAL):
    from blind_insertion.scenes import pc_bench as S, pc_rear as PR
    if getattr(S.build_scene, '_pc_geom', None):
        return
    orig_build, orig_layout = S.build_scene, PR.layout

    def layout(profile):
        out = orig_layout(profile)
        out['target_from_left'] = geom['target_from_left']
        return out

    def build_scene(*a, profile=None, **kw):
        if profile is not None and profile.get('pc'):
            profile = copy.deepcopy(profile)
            m = profile['pc']['measurements']
            m['width']['value'], m['depth']['value'] = geom['width'], geom['depth']
        return orig_build(*a, profile=profile, **kw)

    build_scene._pc_geom = True
    PR.layout = layout
    S.build_scene = build_scene
