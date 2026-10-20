"""Gear-meshing assets (paper §2.1, App. A.1, Table 1 gear column) for the batched env.

One compiled model holds every variant's meshes; each world selects its gear set through
per-world geom slots (as `builder.py` does for pegs). Three bodies carry the assembly:

  fixture (mocap): plate, collar, shaft.   Frame: origin on the shaft axis at the collar's
                   top face (the seat plane, z = 0), z along the shaft.
  gear (free):     involute spur gear with a bore, plus the printed 18 mm boss on top that
                   the gripper holds. Frame: origin at the bore centre of the bottom face,
                   z towards the boss. Seated <=> gear origin at the fixture origin.
  anchor_0/1:      hinged anchor gears, children of the fixture (hinge about fixture z),
                   faces from z = 0 to z = face width (flush with the seated gear).

Mesh count is kept small because build time is quadratic in it: all teeth of a gear are
one mesh used by T geom slots rotated about z (geom_quat), likewise the hub sectors, the
boss segments and the anchors' teeth. Per variant: gear tooth, gear hub sector, shaft,
and per anchor a tooth and a hub (<= 7 meshes; random-tooth OOD gears use one mesh per
tooth). Plate, collar and boss segment are shared by all variants.

Backlash: each anchor's centre distance is solved so that the circular backlash on the
operating pitch circles is 2 x (backlash per flank), using the exact involute relations.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np
from mjfit import _gear as G
from mjfit import _polygon as pg

from blind_insertion.assets import builder as B

# collision bits (same scheme as pegs): fixture + anchors = FIX, gear = PEG (part)
PART_BITS = B.PEG_BITS
PARK = {"gear": -50.0, "anchor": -80.0}     # parked slots, along the body z axis

BOSS_D = 0.018          # printed boss diameter (paper)
BOSS_H = 0.030          # boss height (ours; pads 37.5 mm tall cover its lower part)
BOSS_SEG = 16           # boss ring segments
BOSS_BORE_R = 0.0066    # boss bore inradius: larger than any gear bore, never touches the shaft
COLLAR_R = 0.010
COLLAR_H = 0.005
PLATE_HALF = 0.10
PLATE_T = 0.010
SHAFT_SIDES = 24       # MuJoCo Warp sizes multi-CCD buffers by the largest mesh polygon
ANCHOR_DENSITY = 1250.0  # printed PLA
MAX_TEETH = 24
N_ANCHOR = 2
SECTOR_ARC = 8           # root-arc samples per hub sector


# ----------------------------------------------------------------------------- variants
@dataclass
class GearVariant:
    module: float
    teeth: int
    pressure_deg: float
    tooth_width: float          # fraction of the circular pitch, at the pitch circle
    face_width: float
    bore_clearance: float       # diametral: bore diameter - shaft diameter
    shaft_r: float
    stick_out: float            # shaft top above the gear's top face when seated
    mass: float                 # gear + boss, kg
    anchors: list = field(default_factory=list)  # dicts: teeth, tooth_width, backlash, azimuth
    addendum: float = 1.0       # modules above the pitch circle (gear and anchors)
    random_tooth: bool = False
    tooth_seed: int = 0
    kind: str = "study"

    @property
    def dedendum(self):
        return self.addendum + 0.25

    @property
    def bore_r(self):
        return self.shaft_r + self.bore_clearance / 2

    def to_dict(self):
        return dict(self.__dict__)

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def _dims(teeth, module, pressure_deg, addendum, tooth_width):
    return G._dimensions(teeth, module, math.radians(pressure_deg), addendum, addendum + 0.25,
                         tooth_width)


def tooth_polygon(d: dict) -> np.ndarray:
    """mjfit's convex tooth piece centred on angle 0 (root point to root point)."""
    outline = G._outline(d, 0.0)
    pieces = G._pieces(outline, d["root_radius"])
    teeth = pieces[1:]
    ang = [abs(math.atan2(t.mean(0)[1], t.mean(0)[0])) for t in teeth]
    return pg.ensure_ccw(np.asarray(teeth[int(np.argmin(ang))]))


HUB_SIDES = 24


def hub_polygon(d: dict) -> np.ndarray:
    """Anchor hub: a HUB_SIDES-gon inscribed in the root circle (mjfit's hub through every
    root point has ~4T vertices, which makes MuJoCo Warp's multi-CCD buffers 3x larger)."""
    return _ngon(d["root_radius"], HUB_SIDES, math.pi / HUB_SIDES)


def sector_polygon(d: dict, bore_r: float) -> np.ndarray:
    """Hub piece between the bore and the root circle spanning one pitch around angle 0.
    The bore side is a chord whose midpoint is at bore_r (bore = T-gon of inradius bore_r)."""
    p = 2 * math.pi / d["teeth"]
    rb = bore_r / math.cos(p / 2)
    arc = np.linspace(-p / 2, p / 2, SECTOR_ARC + 1)
    pts = [(rb * math.cos(-p / 2), rb * math.sin(-p / 2))]
    pts += [(d["root_radius"] * math.cos(a), d["root_radius"] * math.sin(a)) for a in arc]
    pts += [(rb * math.cos(p / 2), rb * math.sin(p / 2))]
    return pg.ensure_ccw(np.asarray(pts))


def random_tooth_polygon(d: dict, rng) -> np.ndarray:
    """Table 6 'random tooth': a random convex quadrilateral inside the involute tooth."""
    root, outer, rp = d["root_radius"], d["outer_radius"], d["pitch_radius"]
    th_root = G._half_angle(max(root, d["base_radius"]), d)
    for _ in range(100):
        a1, a2 = -rng.uniform(0.3, 1.0) * th_root, rng.uniform(0.3, 1.0) * th_root
        r3, r4 = rng.uniform(rp, outer), rng.uniform(rp, outer)
        a3 = rng.uniform(0.0, 1.0) * G._half_angle(r3, d)
        a4 = -rng.uniform(0.0, 1.0) * G._half_angle(r4, d)
        q = np.array([[root * math.cos(a1), root * math.sin(a1)],
                      [root * math.cos(a2), root * math.sin(a2)],
                      [r3 * math.cos(a3), r3 * math.sin(a3)],
                      [r4 * math.cos(a4), r4 * math.sin(a4)]])
        q = pg.ensure_ccw(q)
        if pg.is_convex(q) and abs(pg.signed_area(q)) > 1e-7:
            return q
    raise RuntimeError("no convex random tooth")


def tip_points(d: dict, n: int = 64, polys=None) -> np.ndarray:
    """n points on the gear's tooth tips (gear xy), spread over all teeth.

    polys: optional per-tooth polygons (random teeth, in the tooth's local frame); the tip of
    a polygon is its outermost edge."""
    T = d["teeth"]
    p = 2 * math.pi / T
    out = []
    per = [sum(1 for j in range(n) if int(j * T / n) == t) for t in range(T)]
    for j in range(n):
        t = int(j * T / n)
        i = sum(1 for jj in range(j) if int(jj * T / n) == t)
        s = (i + 0.5) / per[t] * 2 - 1           # -1..1 across the tip
        if polys is None:
            ht = G._half_angle(d["outer_radius"], d) * 0.8
            a = t * p + s * ht
            out.append((d["outer_radius"] * math.cos(a), d["outer_radius"] * math.sin(a)))
        else:
            q = polys[t]
            r = np.linalg.norm(q, axis=1)
            k = np.argsort(-r)[:2]
            a_, b_ = q[k[0]], q[k[1]]
            loc = a_ + (b_ - a_) * (0.5 + 0.4 * s)
            c, sn = math.cos(t * p), math.sin(t * p)
            out.append((c * loc[0] - sn * loc[1], sn * loc[0] + c * loc[1]))
    return np.asarray(out)


def _thickness(r, d):
    return 2 * r * G._half_angle(r, d)


def center_distance(d1: dict, d2: dict, backlash_flank: float) -> float:
    """Centre distance giving 2*backlash_flank of circular backlash on the operating pitch
    circles (involute gears; bisection)."""
    rb1, rb2 = d1["base_radius"], d2["base_radius"]

    def backlash(a):
        aw = math.acos(min(1.0, (rb1 + rb2) / a))
        rw1, rw2 = rb1 / math.cos(aw), rb2 / math.cos(aw)
        return 2 * math.pi * rw1 / d1["teeth"] - _thickness(rw1, d1) - _thickness(rw2, d2)

    lo = (rb1 + rb2) * 1.0000001
    hi = d1["pitch_radius"] + d2["pitch_radius"] + 0.01
    target = 2 * backlash_flank
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if backlash(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _valid(v: GearVariant) -> bool:
    try:
        d = _dims(v.teeth, v.module, v.pressure_deg, v.addendum, v.tooth_width)
        G._outline(d, 0.0)
        if G._half_angle(d["outer_radius"], d) <= 0.05 * math.pi / v.teeth:
            return False                      # pointed tips
        if d["root_radius"] < v.bore_r / math.cos(math.pi / v.teeth) + 0.004:
            return False
        for a in v.anchors:
            da = _dims(a["teeth"], v.module, v.pressure_deg, v.addendum, a["tooth_width"])
            G._outline(da, 0.0)
            if G._half_angle(da["outer_radius"], da) <= 0.05 * math.pi / a["teeth"]:
                return False
            cd = center_distance(d, da, a["backlash"])
            # anchor tips must clear the gear root, and the gear tips the anchor root
            if cd - da["outer_radius"] <= d["root_radius"] + 1e-4:
                return False
            if cd - d["outer_radius"] <= da["root_radius"] + 1e-4:
                return False
            a["center_distance"] = cd
        if len(v.anchors) == 2:
            c = [a["center_distance"] * np.array([math.cos(a["azimuth"]), math.sin(a["azimuth"])])
                 for a in v.anchors]
            ro = [_dims(a["teeth"], v.module, v.pressure_deg, v.addendum, a["tooth_width"])["outer_radius"]
                  for a in v.anchors]
            if np.linalg.norm(c[0] - c[1]) < ro[0] + ro[1] + 0.002:
                return False
        return True
    except ValueError:
        return False


def gear_volume(module, teeth, face_width, bore_r):
    """Approximate solid volume of gear + boss (pitch-circle disc minus bore, plus boss)."""
    rp = module * teeth / 2
    boss = math.pi * ((BOSS_D / 2) ** 2 - BOSS_BORE_R ** 2) * BOSS_H
    return math.pi * (rp ** 2 - bore_r ** 2) * face_width + boss


# mass "set by geometry": linear in volume, mapped so the study range spans 128-202 g
_V_LO = gear_volume(2.375e-3, 16, 9.75e-3, 4.5e-3 + 0.5e-3)
_V_HI = gear_volume(3.125e-3, 24, 13.25e-3, 5.5e-3 + 0.375e-3)


def gear_mass(module, teeth, face_width, bore_r, mass=(0.128, 0.202)):
    u = (gear_volume(module, teeth, face_width, bore_r) - _V_LO) / (_V_HI - _V_LO)
    return max(0.08, mass[0] + (mass[1] - mass[0]) * u)


def _draw(rng, r):
    if isinstance(r[0], tuple):
        return B._two_range(rng, *r)
    return rng.uniform(*r)


STUDY = dict(module=(2.375e-3, 3.125e-3), teeth=(16, 24), pressure=(21.25, 23.75),
             tooth_width=(0.45, 0.47), face_width=(9.75e-3, 13.25e-3), backlash=(0.4e-3, 0.6e-3),
             bore=(0.75e-3, 1.0e-3), shaft_r=(4.5e-3, 5.5e-3), stick=(4.75e-3, 8.25e-3),
             mass=(0.128, 0.202), addendum=(1.0, 1.0), n_anchor=(1, 2), anchor_teeth=(16, 24))

# Table 6 (gear column): each bin widens one property beyond the study range.
GEAR_OOD_BINS = {
    "in_dist": {},
    "tighter": dict(backlash=(0.2e-3, 0.25e-3)),
    "taller": dict(addendum=(1.1, 1.2)),
    "much_taller": dict(addendum=(1.2, 1.3)),
    "mild_size": dict(module=((2.19e-3, 2.375e-3), (3.125e-3, 3.31e-3))),
    "extreme_size": dict(module=((2.0e-3, 2.19e-3), (3.31e-3, 3.5e-3))),
    "random_tooth": dict(random_tooth=True),
}


def sample_gears(n: int, seed: int, **over) -> list[GearVariant]:
    """Study gear sets (Table 1, gear column), with any range overridden (OOD bins)."""
    r = dict(STUDY)
    random_tooth = over.pop("random_tooth", False)
    r.update(over)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        na = int(rng.integers(r["n_anchor"][0], r["n_anchor"][1] + 1))
        az0 = rng.uniform(0, 2 * math.pi)
        anchors = []
        for i in range(na):
            az = az0 if i == 0 else az0 + rng.uniform(math.radians(70), math.radians(290))
            anchors.append(dict(teeth=int(rng.integers(r["anchor_teeth"][0], r["anchor_teeth"][1] + 1)),
                                tooth_width=float(rng.uniform(*r["tooth_width"])),
                                backlash=float(_draw(rng, r["backlash"])), azimuth=float(az % (2 * math.pi))))
        v = GearVariant(module=float(_draw(rng, r["module"])),
                        teeth=int(rng.integers(r["teeth"][0], r["teeth"][1] + 1)),
                        pressure_deg=float(rng.uniform(*r["pressure"])),
                        tooth_width=float(rng.uniform(*r["tooth_width"])),
                        face_width=float(rng.uniform(*r["face_width"])),
                        bore_clearance=float(rng.uniform(*r["bore"])),
                        shaft_r=float(rng.uniform(*r["shaft_r"])),
                        stick_out=float(rng.uniform(*r["stick"])),
                        mass=0.0, anchors=anchors,
                        addendum=float(_draw(rng, r["addendum"])),
                        random_tooth=random_tooth, tooth_seed=int(rng.integers(1 << 30)),
                        kind="random_tooth" if random_tooth else "study")
        v.mass = float(gear_mass(v.module, v.teeth, v.face_width, v.bore_r, r["mass"]))
        if _valid(v):
            out.append(v)
    return out


def sample_study_gears(n: int, seed: int) -> list[GearVariant]:
    return sample_gears(n, seed)


def sample_ood_gears(n: int, seed: int, bin: str) -> list[GearVariant]:
    return sample_gears(n, seed, **dict(GEAR_OOD_BINS[bin]))


# ----------------------------------------------------------------------------- pieces
def _prism(poly: np.ndarray, z0: float, z1: float) -> np.ndarray:
    poly = np.asarray(poly, float)
    return np.vstack([np.column_stack([poly, np.full(len(poly), z0)]),
                      np.column_stack([poly, np.full(len(poly), z1)])])


def _ngon(r, n, phase=0.0):
    a = phase + 2 * math.pi * np.arange(n) / n
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


def boss_segment() -> np.ndarray:
    """One of BOSS_SEG ring pieces around angle 0 (outer faces normal to k*2pi/BOSS_SEG, so
    two faces face the pads along +-y)."""
    h = math.pi / BOSS_SEG
    ro, ri = (BOSS_D / 2) / math.cos(h), BOSS_BORE_R / math.cos(h)
    poly = np.array([[ri * math.cos(-h), ri * math.sin(-h)], [ro * math.cos(-h), ro * math.sin(-h)],
                     [ro * math.cos(h), ro * math.sin(h)], [ri * math.cos(h), ri * math.sin(h)]])
    return poly


@dataclass
class GearPieces:
    d: dict
    teeth: list[np.ndarray]           # one polygon (normal) or T polygons (random teeth)
    sector: np.ndarray
    anchor_d: list[dict]
    anchor_tooth: list[np.ndarray]
    anchor_hub: list[np.ndarray]
    anchor_center: np.ndarray         # (n_anchor, 2) in fixture xy
    tips: np.ndarray                  # (64, 2) tooth-tip points (gear xy)


def make_pieces(v: GearVariant) -> GearPieces:
    d = _dims(v.teeth, v.module, v.pressure_deg, v.addendum, v.tooth_width)
    if v.random_tooth:
        rng = np.random.default_rng(v.tooth_seed)
        teeth = [random_tooth_polygon(d, rng) for _ in range(v.teeth)]
        tips = tip_points(d, polys=teeth)
    else:
        teeth = [tooth_polygon(d)]
        tips = tip_points(d)
    ad, at, ah, ac = [], [], [], []
    for a in v.anchors:
        da = _dims(a["teeth"], v.module, v.pressure_deg, v.addendum, a["tooth_width"])
        cd = a.get("center_distance") or center_distance(d, da, a["backlash"])
        a["center_distance"] = cd
        ad.append(da)
        at.append(tooth_polygon(da))
        ah.append(hub_polygon(da))
        ac.append([cd * math.cos(a["azimuth"]), cd * math.sin(a["azimuth"])])
    return GearPieces(d=d, teeth=teeth, sector=sector_polygon(d, v.bore_r), anchor_d=ad,
                      anchor_tooth=at, anchor_hub=ah, anchor_center=np.asarray(ac).reshape(-1, 2),
                      tips=tips)


def _quat_z(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def _gear_slots(v: GearVariant, p: GearPieces):
    """Gear-body slot list: (mesh key, angle) for MAX_TEETH tooth slots, MAX_TEETH sector
    slots and BOSS_SEG boss slots. Unused slots: None (parked)."""
    T = v.teeth
    pitch = 2 * math.pi / T
    s = []
    for i in range(MAX_TEETH):
        s.append((f"tooth{i if v.random_tooth else 0}", i * pitch) if i < T else None)
    for i in range(MAX_TEETH):
        s.append(("sector", i * pitch) if i < T else None)
    for i in range(BOSS_SEG):
        s.append(("boss", i * 2 * math.pi / BOSS_SEG))
    return s


def _anchor_slots(v: GearVariant, j: int):
    if j >= len(v.anchors):
        return [None] * (MAX_TEETH + 1)
    T = v.anchors[j]["teeth"]
    pitch = 2 * math.pi / T
    return [("tooth", i * pitch) if i < T else None for i in range(MAX_TEETH)] + [("hub", 0.0)]


def _piece_verts(v: GearVariant, p: GearPieces):
    """mesh key -> vertices, for the gear and each anchor."""
    fw = v.face_width
    gear = {f"tooth{i}": _prism(t, 0.0, fw) for i, t in enumerate(p.teeth)}
    gear["sector"] = _prism(p.sector, 0.0, fw)
    gear["boss"] = _prism(boss_segment(), fw, fw + BOSS_H)
    anchors = [{"tooth": _prism(p.anchor_tooth[j], 0.0, fw), "hub": _prism(p.anchor_hub[j], 0.0, fw)}
               for j in range(len(v.anchors))]
    shaft = _prism(_ngon(v.shaft_r, SHAFT_SIDES), -COLLAR_H, fw + v.stick_out)
    return gear, anchors, shaft


def shared_meshes():
    return {"plate": _prism(_ngon(PLATE_HALF * math.sqrt(2), 4, math.pi / 4), -COLLAR_H - PLATE_T, -COLLAR_H),
            "collar": _prism(_ngon(COLLAR_R, 24), -COLLAR_H, 0.0),
            "boss": None}


def _mini_model(v: GearVariant, p: GearPieces) -> tuple[mujoco.MjModel, dict]:
    """Single-variant model with the exact slot structure -> per-slot geom frames and the
    bodies' inertial fields."""
    gear_v, anch_v, shaft_v = _piece_verts(v, p)
    sh = shared_meshes()
    spec = mujoco.MjSpec()
    for k_, vv in gear_v.items():
        spec.add_mesh(name=f"g_{k_}", uservert=vv.ravel().tolist())
    for j, a in enumerate(anch_v):
        for k_, vv in a.items():
            spec.add_mesh(name=f"a{j}_{k_}", uservert=vv.ravel().tolist())
    spec.add_mesh(name="shaft", uservert=shaft_v.ravel().tolist())
    spec.add_mesh(name="plate", uservert=sh["plate"].ravel().tolist())
    spec.add_mesh(name="collar", uservert=sh["collar"].ravel().tolist())
    fb = spec.worldbody.add_body(name="fixture", mocap=True)
    for nm in ("plate", "collar", "shaft"):
        fb.add_geom(name=f"fix_{nm}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=nm)
    gb = spec.worldbody.add_body(name="gear", pos=[0, 0, 0.1])
    gb.add_freejoint()
    for i, s in enumerate(_gear_slots(v, p)):
        key, ang = s if s is not None else ("sector", 0.0)
        gb.add_geom(name=f"gear_{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"g_{key}",
                    quat=_quat_z(ang), density=1000 if s is not None else 1e-6)
    for j in range(N_ANCHOR):
        ab = fb.add_body(name=f"anchor_{j}",
                         pos=[*p.anchor_center[j], 0.0] if j < len(v.anchors) else [0, 0, -2.0])
        ab.add_joint(name=f"anchor_{j}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
        for i, s in enumerate(_anchor_slots(v, j)):
            if j < len(v.anchors):
                key, ang = s if s is not None else ("hub", 0.0)
                ab.add_geom(name=f"anchor_{j}_{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname=f"a{j}_{key}", quat=_quat_z(ang),
                            density=ANCHOR_DENSITY if s is not None else 1e-6)
            else:
                ab.add_geom(name=f"anchor_{j}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.01, 0.01, 0.005], density=ANCHOR_DENSITY)
    m = spec.compile()
    info = {"gear": _gear_slots(v, p), "anchor": [_anchor_slots(v, j) for j in range(N_ANCHOR)]}
    return m, info


# ----------------------------------------------------------------------------- layout
@dataclass
class GearLayout:
    robot: B.RobotInfo
    variants: list
    fixture_body: int
    gear_body: int
    anchor_body: np.ndarray       # (2,)
    gear_qadr: int
    gear_dofadr: int
    anchor_qadr: np.ndarray       # (2,)
    anchor_dofadr: np.ndarray     # (2,)
    arm_qadr: np.ndarray
    arm_dofadr: np.ndarray
    finger_qadr: np.ndarray
    arm_act: np.ndarray
    grip_act: np.ndarray
    tcp_site: int
    pad_geoms: np.ndarray
    fix_slots: np.ndarray         # (3,) plate, collar, shaft
    gear_slots: np.ndarray        # (2*MAX_TEETH + BOSS_SEG,)
    anchor_slots: np.ndarray      # (2, MAX_TEETH + 1)
    slot: dict                    # role -> {dataid,pos,quat,size,aabb,rbound}: (K, S, ...)
    body: dict                    # gear_/anchor_ inertial fields: (K, ...) / (K, 2, ...)
    # derived per-variant arrays
    n_anchor: np.ndarray          # (K,)
    anchor_center: np.ndarray     # (K, 2, 2)
    anchor_outer: np.ndarray      # (K, 2)
    anchor_teeth: np.ndarray      # (K, 2)
    teeth: np.ndarray             # (K,)
    outer_r: np.ndarray
    root_r: np.ndarray
    bore_r: np.ndarray
    shaft_r: np.ndarray
    face_width: np.ndarray
    shaft_top: np.ndarray         # z of the shaft's top face (fixture frame)
    tips: np.ndarray              # (K, 64, 2)


def _slot_fields(mm, names, mesh_ids, park):
    out = {k: [] for k in ("dataid", "pos", "quat", "size", "aabb", "rbound")}
    for nm, mid, parked in zip(names, mesh_ids, park):
        g = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, nm)
        pos = mm.geom_pos[g].copy()
        if parked is not None:
            pos[2] += parked
        out["dataid"].append(mid)
        out["pos"].append(pos)
        out["quat"].append(mm.geom_quat[g].copy())
        out["size"].append(mm.geom_size[g].copy())
        out["aabb"].append(mm.geom_aabb[g].copy())
        out["rbound"].append(mm.geom_rbound[g])
    return out


def build(robot: str, variants: list[GearVariant], *, gains=None, timestep=0.001,
          backlash_arcmin: float = 0.0, multiccd: bool = False):
    spec, info = B.load_robot(robot, gains=gains, backlash_arcmin=backlash_arcmin)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = 50
    spec.option.ls_iterations = 20
    if multiccd:
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_MULTICCD
    spec.stat.extent = 1.5
    spec.stat.center = [0.4, 0, 0.3]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.worldbody.add_light(name="key", pos=[0.6, -0.6, 1.5], dir=[-0.3, 0.3, -1],
                             diffuse=[0.8, 0.8, 0.8], castshadow=True)
    spec.worldbody.add_light(name="fill", pos=[-0.5, 0.5, 1.2], dir=[0.4, -0.4, -1],
                             diffuse=[0.35, 0.35, 0.35], castshadow=False)

    pieces = [make_pieces(v) for v in variants]
    sh = shared_meshes()
    spec.add_mesh(name="plate", uservert=sh["plate"].ravel().tolist())
    spec.add_mesh(name="collar", uservert=sh["collar"].ravel().tolist())
    spec.add_mesh(name="boss", uservert=_prism(boss_segment(), 0, BOSS_H).ravel().tolist())
    for k, (v, p) in enumerate(zip(variants, pieces)):
        gear_v, anch_v, shaft_v = _piece_verts(v, p)
        for key, vv in gear_v.items():
            if key != "boss":
                spec.add_mesh(name=f"k{k}_g_{key}", uservert=vv.ravel().tolist())
        for j, a in enumerate(anch_v):
            for key, vv in a.items():
                spec.add_mesh(name=f"k{k}_a{j}_{key}", uservert=vv.ravel().tolist())
        spec.add_mesh(name=f"k{k}_shaft", uservert=shaft_v.ravel().tolist())

    def mesh_geom(body, name, mesh, rgba, bits, part=False, **kw):
        g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, rgba=rgba,
                          friction=[1.0, 0.01, 0.001], condim=4, contype=bits[0],
                          conaffinity=bits[1], margin=0, gap=0,
                          solimp=[0.99, 0.999, 0.001, 0.5, 2], **kw)
        if part:
            g.solref = [0.002, 1]
        else:
            g.priority = 1
            g.solref = [0.005, 1]
        return g

    fix = spec.worldbody.add_body(name="fixture", pos=[0.65, 0.0, 0.1], mocap=True)
    mesh_geom(fix, "fix_plate", "plate", [0.70, 0.75, 0.82, 1], B.FIX_BITS)
    mesh_geom(fix, "fix_collar", "collar", [0.45, 0.47, 0.52, 1], B.FIX_BITS)
    mesh_geom(fix, "fix_shaft", "k0_shaft", [0.25, 0.27, 0.30, 1], B.FIX_BITS)
    gear = spec.worldbody.add_body(name="gear", pos=[0.65, 0.0, 0.3])
    gear.add_freejoint(name="gear_joint")
    nG = 2 * MAX_TEETH + BOSS_SEG
    for i in range(nG):
        boss = i >= 2 * MAX_TEETH
        mesh_geom(gear, f"gear_{i}", "boss" if boss else "k0_g_sector",
                  [0.93, 0.93, 0.90, 1] if boss else [0.90, 0.45, 0.20, 1], PART_BITS, part=True,
                  density=1000)
    for j in range(N_ANCHOR):
        ab = fix.add_body(name=f"anchor_{j}", pos=[0.05 * (j + 1), 0, 0])
        ab.add_joint(name=f"anchor_{j}_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                     damping=1e-3, frictionloss=1e-3, armature=1e-5)
        for i in range(MAX_TEETH + 1):
            mesh_geom(ab, f"anchor_{j}_{i}", "collar", [0.35, 0.55, 0.80, 1] if j == 0 else
                      [0.35, 0.70, 0.45, 1], B.FIX_BITS, density=ANCHOR_DENSITY)
    m = spec.compile()
    # CPU MuJoCo skips the geom transform for geoms compiled in their body's (inertial) frame
    # (geom_sameframe); per-variant slot frames differ, so disable it for every part geom and
    # body. MuJoCo Warp does not use these flags.
    part_bodies = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                   for n in ["fixture", "gear"] + [f"anchor_{j}" for j in range(N_ANCHOR)]]
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in part_bodies:
            m.geom_sameframe[g] = 0
    m.body_sameframe[part_bodies] = 0

    def gid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)

    def mid(n):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, n)
        assert i >= 0, n
        return i

    def bid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    def jq(n):
        return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]

    def jd(n):
        return m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]

    roles = {"fix": [], "gear": [], "anchor0": [], "anchor1": []}
    body = {k: [] for k in ("gear_mass", "gear_inertia", "gear_ipos", "gear_iquat", "gear_invweight",
                            "gear_dof_invweight", "anchor_mass", "anchor_inertia", "anchor_ipos",
                            "anchor_iquat", "anchor_invweight", "anchor_dof_invweight", "anchor_pos")}
    for k, (v, p) in enumerate(zip(variants, pieces)):
        mm, si = _mini_model(v, p)
        roles["fix"].append(_slot_fields(mm, ["fix_plate", "fix_collar", "fix_shaft"],
                                         [mid("plate"), mid("collar"), mid(f"k{k}_shaft")], [None] * 3))
        names, ids, park = [], [], []
        for i, s in enumerate(si["gear"]):
            names.append(f"gear_{i}")
            if s is None:
                ids.append(mid(f"k{k}_g_sector"))
                park.append(PARK["gear"])
            else:
                ids.append(mid("boss") if s[0] == "boss" else mid(f"k{k}_g_{s[0]}"))
                park.append(None)
        roles["gear"].append(_slot_fields(mm, names, ids, park))
        for j in range(N_ANCHOR):
            names, ids, park = [], [], []
            for i, s in enumerate(si["anchor"][j]):
                names.append(f"anchor_{j}_{i}")
                if j >= len(v.anchors):
                    ids.append(mid("collar"))
                    park.append(PARK["anchor"])
                elif s is None:
                    ids.append(mid(f"k{k}_a{j}_hub"))
                    park.append(PARK["anchor"])
                else:
                    ids.append(mid(f"k{k}_a{j}_{s[0]}"))
                    park.append(None)
            f_ = _slot_fields(mm, names, ids, park)
            if j >= len(v.anchors):   # absent anchor: the slots hold the collar mesh, parked
                cg = gid("fix_collar")
                f_["size"] = [m.geom_size[cg].copy() for _ in names]
                f_["aabb"] = [m.geom_aabb[cg].copy() for _ in names]
                f_["rbound"] = [m.geom_rbound[cg] for _ in names]
                f_["quat"] = [np.array([1.0, 0, 0, 0]) for _ in names]
                f_["pos"] = [np.array([0.0, 0, PARK["anchor"]]) for _ in names]
            roles[f"anchor{j}"].append(f_)
        gb = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "gear")
        sc = v.mass / mm.body_mass[gb]
        body["gear_mass"].append(v.mass)
        body["gear_inertia"].append(mm.body_inertia[gb] * sc)
        body["gear_ipos"].append(mm.body_ipos[gb].copy())
        body["gear_iquat"].append(mm.body_iquat[gb].copy())
        body["gear_invweight"].append(mm.body_invweight0[gb] / sc)
        da = mm.jnt_dofadr[mm.body_jntadr[gb]]
        body["gear_dof_invweight"].append(mm.dof_invweight0[da:da + 6] / sc)
        am, ai, ap, aq, aw, adw, apos = [], [], [], [], [], [], []
        for j in range(N_ANCHOR):
            ab = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, f"anchor_{j}")
            am.append(mm.body_mass[ab])
            ai.append(mm.body_inertia[ab].copy())
            ap.append(mm.body_ipos[ab].copy())
            aq.append(mm.body_iquat[ab].copy())
            aw.append(mm.body_invweight0[ab].copy())
            adw.append(mm.dof_invweight0[mm.jnt_dofadr[mm.body_jntadr[ab]]])
            apos.append(mm.body_pos[ab].copy())
        for key, val in zip(("anchor_mass", "anchor_inertia", "anchor_ipos", "anchor_iquat",
                             "anchor_invweight", "anchor_dof_invweight", "anchor_pos"),
                            (am, ai, ap, aq, aw, adw, apos)):
            body[key].append(val)
    slot = {r: {f: np.asarray([x[f] for x in lst]) for f in lst[0]} for r, lst in roles.items()}
    body = {k_: np.asarray(v_) for k_, v_ in body.items()}

    K = len(variants)
    n_anchor = np.array([len(v.anchors) for v in variants])
    ac = np.zeros((K, N_ANCHOR, 2))
    ao = np.zeros((K, N_ANCHOR))
    at = np.ones((K, N_ANCHOR), int)
    for k, p in enumerate(pieces):
        for j in range(len(variants[k].anchors)):
            ac[k, j] = p.anchor_center[j]
            ao[k, j] = p.anchor_d[j]["outer_radius"]
            at[k, j] = p.anchor_d[j]["teeth"]
    acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in info.arm_joints]
    grips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"grip_{s}") for s in ("left", "right")]
    lay = GearLayout(
        robot=info, variants=variants, fixture_body=bid("fixture"), gear_body=bid("gear"),
        anchor_body=np.array([bid(f"anchor_{j}") for j in range(N_ANCHOR)]),
        gear_qadr=int(jq("gear_joint")), gear_dofadr=int(jd("gear_joint")),
        anchor_qadr=np.array([jq(f"anchor_{j}_hinge") for j in range(N_ANCHOR)]),
        anchor_dofadr=np.array([jd(f"anchor_{j}_hinge") for j in range(N_ANCHOR)]),
        arm_qadr=np.array([jq(j) for j in info.arm_joints]),
        arm_dofadr=np.array([jd(j) for j in info.arm_joints]),
        finger_qadr=np.array([jq(j) for j in info.finger_joints]),
        arm_act=np.array(acts), grip_act=np.array(grips),
        tcp_site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, info.tcp_site),
        pad_geoms=np.array([gid(g) for g in info.pad_geoms]),
        fix_slots=np.array([gid(n) for n in ("fix_plate", "fix_collar", "fix_shaft")]),
        gear_slots=np.array([gid(f"gear_{i}") for i in range(nG)]),
        anchor_slots=np.array([[gid(f"anchor_{j}_{i}") for i in range(MAX_TEETH + 1)]
                               for j in range(N_ANCHOR)]),
        slot=slot, body=body, n_anchor=n_anchor, anchor_center=ac, anchor_outer=ao, anchor_teeth=at,
        teeth=np.array([v.teeth for v in variants]),
        outer_r=np.array([p.d["outer_radius"] for p in pieces]),
        root_r=np.array([p.d["root_radius"] for p in pieces]),
        bore_r=np.array([v.bore_r for v in variants]),
        shaft_r=np.array([v.shaft_r for v in variants]),
        face_width=np.array([v.face_width for v in variants]),
        shaft_top=np.array([v.face_width + v.stick_out for v in variants]),
        tips=np.asarray([p.tips for p in pieces]))
    return m, lay


def slot_groups(lay: GearLayout):
    """(role key, geom ids) for every slot group."""
    return [("fix", lay.fix_slots), ("gear", lay.gear_slots),
            ("anchor0", lay.anchor_slots[0]), ("anchor1", lay.anchor_slots[1])]


def apply_variant_cpu(m: mujoco.MjModel, lay: GearLayout, k: int):
    """Write variant k into a CPU MjModel (single-world checks and rendering)."""
    for role, ids in slot_groups(lay):
        s = lay.slot[role]
        m.geom_dataid[ids] = s["dataid"][k]
        m.geom_pos[ids] = s["pos"][k]
        m.geom_quat[ids] = s["quat"][k]
        m.geom_size[ids] = s["size"][k]
        m.geom_aabb[ids] = s["aabb"][k].reshape(len(ids), 6)
        m.geom_rbound[ids] = s["rbound"][k]
    b, bd = lay.gear_body, lay.body
    m.body_mass[b] = bd["gear_mass"][k]
    m.body_inertia[b] = bd["gear_inertia"][k]
    m.body_ipos[b] = bd["gear_ipos"][k]
    m.body_iquat[b] = bd["gear_iquat"][k]
    m.body_invweight0[b] = bd["gear_invweight"][k]
    a = lay.gear_dofadr
    m.dof_invweight0[a:a + 6] = bd["gear_dof_invweight"][k]
    for j in range(N_ANCHOR):
        ab = lay.anchor_body[j]
        m.body_mass[ab] = bd["anchor_mass"][k, j]
        m.body_inertia[ab] = bd["anchor_inertia"][k, j]
        m.body_ipos[ab] = bd["anchor_ipos"][k, j]
        m.body_iquat[ab] = bd["anchor_iquat"][k, j]
        m.body_invweight0[ab] = bd["anchor_invweight"][k, j]
        m.dof_invweight0[lay.anchor_dofadr[j]] = bd["anchor_dof_invweight"][k, j]
        m.body_pos[ab] = bd["anchor_pos"][k, j]
