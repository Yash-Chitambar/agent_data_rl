"""Compose one MJCF model: robot + gripper + mjfit assembly with per-world geometry slots.

The model holds the meshes of every assembly variant. Each world uses one variant by
setting its batched `geom_dataid` (and the geom frame fields that depend on the mesh)
for the fixture and peg geom slots. `Layout` records everything the batched env needs.

Frames (our convention, not mjfit's):
  fixture body: origin at the hole's top centre, z out of the plate (top face at z=0,
                hole floor at z=-depth).
  peg body:     origin at the centre of the peg's bottom face, z along the peg.
"""
from __future__ import annotations

import hashlib
import json
import itertools
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mjfit
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"

# Collision bits: fixture=1, peg=2, gripper=4, finger links=8 (collide with the fixture only).
FIX_BITS = (1, 2 | 4 | 8)
FINGER_LINK_BITS = (8, 1)
PEG_BITS = (2, 1 | 4)
GRIP_BITS = (4, 1 | 2)
PARK_Z = -50.0  # unused geom slots are parked this far below their body

# The lab's YAM workcell, as RoboSimStudio's `yam_i2rt` embodiment and its 0.75 m bench define
# it, restated in the task ("right") arm's base frame: the base is level with the table top,
# 0.3475 m behind the table centre and 0.31 m to its right (the bimanual mount, also used with
# one arm). Base orientation is the same in both models (arm reaches along +x, z up).
YAM_TABLE = dict(center=(0.3475, 0.31), half=(0.2975, 0.65), height=0.75)


# ----------------------------------------------------------------------------- variants
@dataclass
class Variant:
    """One assembly: outline of the hole, clearance, dimensions, mass."""
    hole_w: float              # long side of the opening (m)
    hole_h: float              # short side of the opening (m)
    clearance: float           # per side (m)
    depth: float               # hole depth = plate thickness (m)
    plate_size: float          # square plate side (m)
    peg_length: float          # m
    peg_mass: float            # kg
    islands: list = field(default_factory=list)   # mjfit island specs (USB-C tongue)
    n_sides: int = 24          # stadium tessellation (fixed => fixed topology)
    bottom: float = 0.005      # closed hole floor thickness
    kind: str = "stadium"
    hole_spec: dict | None = None  # explicit mjfit hole spec overrides the stadium
    handle: dict | None = None     # mjfit handle outline (e.g. the flash-drive body)
    handle_height: float = 0.0
    grasp_z: float | None = None   # pad centre height above the peg bottom (default 0.75 L)
    grip_half: float | None = None # half width across the pads (default: peg width at x=0)
    chamfer: float = 0.0           # 45 deg lead-in on the hole's top edge (m), plan §11.1 / E4
    peg_boxes: list = field(default_factory=list)  # extra convex boxes on the peg [(centre, half)], peg frame
    plug_widen: float = 0.0        # move the insert's walls out along x by this (m): x clearance =
                                   # clearance - plug_widen (anisotropic fit, e.g. USB-C spec, task H)

    def hole(self):
        if self.hole_spec is not None:
            return self.hole_spec
        return {"type": "stadium", "width": self.hole_w, "height": self.hole_h, "n": self.n_sides}

    def mjfit_xml(self) -> str:
        kw = {}
        if self.handle_height > 0:
            kw = dict(handle=self.handle, handle_height=self.handle_height)
        return mjfit.peg(hole=self.hole(), islands=self.islands, clearance=self.clearance,
                         peg_length=self.peg_length, plate_size=self.plate_size,
                         plate_thickness=self.depth, bottom_thickness=self.bottom,
                         peg_mass=self.peg_mass, standalone=False, spawn_height=0.0, **kw)

    def to_dict(self):
        return dict(self.__dict__)

    def digest(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def stadium_dims(scale: float, aspect: float, area0: float = 515e-6):
    """Area-preserving stadium (long w, short h) with w/h = aspect and area0*scale^2."""
    # area = (w-h)h + pi h^2/4 = h^2 (aspect - 1 + pi/4)
    h = math.sqrt(area0 / (aspect - 1 + math.pi / 4)) * scale
    return aspect * h, h


def sample_study_pegs(n: int, seed: int, *, scale=(1.35, 2.45), aspect=(1.25, 2.5),
                      clearance=(1.0e-3, 1.5e-3), mass=(0.35, 0.675), length=(0.050, 0.060),
                      n_sides: int = 24) -> list[Variant]:
    """Table 1 'study peg' ranges of the paper (aspect log-uniform)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.uniform(*scale)
        ar = math.exp(rng.uniform(math.log(aspect[0]), math.log(aspect[1])))
        w, h = stadium_dims(s, ar)
        out.append(Variant(hole_w=w, hole_h=h, clearance=float(rng.uniform(*clearance)),
                           depth=0.010 * s, plate_size=0.080 * s,
                           peg_length=float(rng.uniform(*length)) * s,
                           peg_mass=float(rng.uniform(*mass)), n_sides=n_sides))
    return out


def _two_range(rng, lo_rng, hi_rng):
    """Draw uniformly from the union of two intervals (weighted by length)."""
    la, lb = lo_rng[1] - lo_rng[0], hi_rng[1] - hi_rng[0]
    r = lo_rng if rng.uniform() < la / (la + lb) else hi_rng
    return rng.uniform(*r)


def _random_convex_in_box(rng, w, h):
    """Paper Fig. 13: a convex quadrilateral with one vertex on each side of the w x h box."""
    x_top, x_bot = rng.uniform(-w / 2, w / 2), rng.uniform(-w / 2, w / 2)
    y_r, y_l = rng.uniform(-h / 2, h / 2), rng.uniform(-h / 2, h / 2)
    return np.array([[x_bot, -h / 2], [w / 2, y_r], [x_top, h / 2], [-w / 2, y_l]])


# Table 6 (peg column): each bin widens one property beyond the study range.
OOD_BINS = {
    "in_dist": {},
    "tighter": dict(clearance=(0.5e-3, 0.625e-3)),
    "slender": dict(aspect=(3.54, 4.20)),
    "very_slender": dict(aspect=(4.20, 5.0)),
    "mild_size": dict(scale=((1.075, 1.35), (2.45, 2.725))),
    "extreme_size": dict(scale=((0.8, 1.075), (2.725, 3.0))),
    "random_convex": dict(shape="random_convex"),
}


def sample_ood_pegs(n: int, seed: int, bin: str, n_sides: int = 24) -> list[Variant]:
    """Study-peg ranges with one property widened per OOD_BINS[bin]."""
    spec = OOD_BINS[bin]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        sc = spec.get("scale", (1.35, 2.45))
        s = _two_range(rng, *sc) if isinstance(sc[0], tuple) else rng.uniform(*sc)
        ar_lo, ar_hi = spec.get("aspect", (1.25, 2.5))
        ar = math.exp(rng.uniform(math.log(ar_lo), math.log(ar_hi)))
        w, h = stadium_dims(s, ar)
        c = float(rng.uniform(*spec.get("clearance", (1.0e-3, 1.5e-3))))
        v = Variant(hole_w=w, hole_h=h, clearance=c, depth=0.010 * s, plate_size=0.080 * s,
                    peg_length=float(rng.uniform(0.050, 0.060)) * s,
                    peg_mass=float(rng.uniform(0.35, 0.675)), n_sides=n_sides)
        if spec.get("shape") == "random_convex":
            # the peg is the random convex outline inside the matching stadium's box
            # (minus the clearance), so the hole is its outward offset
            poly = _random_convex_in_box(rng, w - 2 * c, h - 2 * c)
            from mjfit import _polygon as pg  # noqa: PLC0415
            v.hole_spec = {"type": "polygon", "vertices": pg.offset_outward(poly, c).tolist()}
            v.kind = "random_convex"
        out.append(v)
    return out


def usbc_variant(clearance=5e-5, housing=0.025, recess=0.0, body=(0.020, 0.0065, 0.030),
                 mass=0.008, n_sides=16, chamfer=0.0) -> Variant:
    """USB-C flash drive in a blind receptacle (plan §11.1, Type-C R2.0 dims).

    Receptacle opening 8.34 x 2.56 mm stadium with a 6.5 x 0.7 mm tongue, 6.6 mm deep,
    blind (1 mm floor). Plug shell 8.5 mm long; the drive body is the handle the gripper
    holds across its thickness.
    """
    bw, bt, bl = body
    return Variant(hole_w=0.00834, hole_h=0.00256, clearance=clearance, depth=0.0066 + recess,
                   plate_size=housing, peg_length=0.0085, peg_mass=mass, n_sides=n_sides,
                   bottom=0.001, kind="usbc",
                   islands=[{"type": "stadium", "width": 0.0065, "height": 0.0007, "n": 8}],
                   hole_spec={"type": "stadium", "width": 0.00834, "height": 0.00256, "n": n_sides},
                   handle={"type": "rect", "width": bw, "height": bt}, handle_height=bl,
                   grasp_z=0.0085 + bl - 0.009, grip_half=bt / 2,   # pads near the body's top
                   chamfer=chamfer)


def sample_usbc(n: int, seed: int, *, clearance=(2e-5, 1e-4), housing=(0.015, 0.040),
                recess=(0.0, 0.0015), body_w=(0.012, 0.025), body_t=(0.0045, 0.009),
                body_l=(0.025, 0.040), mass=(0.003, 0.015), chamfer=(0.0, 0.0)) -> list[Variant]:
    """Plan §11.1 pool; clearance log-uniform. Pass fixed tuples (a, a) to pin a value.
    chamfer (E4): uniform lead-in size, drawn after the other parameters (so (0, 0) keeps the
    original pool and a chamfer range gives the same parts with a lead-in)."""
    rng = np.random.default_rng(seed)
    lu = lambda r: math.exp(rng.uniform(math.log(r[0]), math.log(r[1]))) if r[0] != r[1] else r[0]  # noqa: E731
    out = [usbc_variant(clearance=lu(clearance), housing=rng.uniform(*housing),
                        recess=rng.uniform(*recess),
                        body=(rng.uniform(*body_w), rng.uniform(*body_t), rng.uniform(*body_l)),
                        mass=rng.uniform(*mass)) for _ in range(n)]
    if chamfer[1] > 0:
        for v in out:
            v.chamfer = float(rng.uniform(*chamfer))
    return out


# USB Type-C R2.x (USB-IF): receptacle opening 8.34 (+0.06/-0.02) x 2.56 mm, plug shell
# 8.25 x 2.40 (+-0.03) mm and 6.65 mm long, receptacle tongue 6.69 x 0.70 mm.
USBC_REC = (0.00834, 0.00256)
USBC_PLUG = (0.00825, 0.00240)
USBC_PLUG_LEN = 0.00665
USBC_TONGUE = (0.00669, 0.00070)


def usbc_spec_variant(rec=USBC_REC, plug=USBC_PLUG, housing=0.030, neck=(0.014, 0.007, 0.028),
                      mass=0.027, n_sides=24, chamfer=0.0) -> Variant:
    """USB-C plug in a USB-C receptacle at true spec dimensions (task H).

    Clearance differs per axis (nominal 45 um per side across the width, 80 um across the
    thickness), so the plug is the opening offset by the thickness clearance and then widened
    (`plug_widen`). The cavity is 6.50 mm deep: mjfit needs the hollow plug longer than the
    cavity plus clearance, so the plug stops on the floor with the overmold (neck) 0.15 mm
    short of the panel instead of on the panel as on the real part. Stadium ends: 24-gon (<= 11 um chord
    error at the 1.28 mm radius).
    """
    c = (rec[1] - plug[1]) / 2
    cx = (rec[0] - plug[0]) / 2
    nw, nt, nl = neck
    return Variant(hole_w=rec[0], hole_h=rec[1], clearance=c, depth=min(0.0065, USBC_PLUG_LEN - c - 2e-5),   # shallower only for loose curriculum parts
                   
                   plate_size=housing, peg_length=USBC_PLUG_LEN, peg_mass=mass, n_sides=n_sides,
                   bottom=0.001, kind="usbc_spec",
                   islands=[{"type": "stadium", "width": USBC_TONGUE[0], "height": USBC_TONGUE[1], "n": 8}],
                   hole_spec={"type": "stadium", "width": rec[0], "height": rec[1], "n": n_sides},
                   handle={"type": "rect", "width": nw, "height": nt}, handle_height=nl,
                   chamfer=chamfer, plug_widen=c - cx)


# Caliper photos IMG_8745–8748: body width/thickness, neck width, total tip-to-tail.
READER_TOTAL_LENGTH = 0.09046
READER_BODY = (0.03473, 0.00883, 0.042)  # body length estimated from photo ruler
READER_NECK = (0.01287, 0.0045, READER_TOTAL_LENGTH - USBC_PLUG_LEN - READER_BODY[2])
# Neck thickness (4.5 mm) and the length split are photo estimates, not caliper readings.


def reader_variant(rec=USBC_REC, plug=USBC_PLUG, housing=0.030, body=READER_BODY,
                   neck=READER_NECK, mass=0.027, chamfer=0.0) -> Variant:
    """SanDisk USB-C SD reader (task H) at USB-C spec dimensions (`usbc_spec_variant`).

    The plug, then a neck (width, thickness, length) and the wide paddle body (width,
    thickness, length) the gripper holds top-down across its width. Widths lie along peg x
    (the opening's long side), thicknesses along peg y. Measured envelope; length split and neck thickness remain photo estimates.
    """
    v = usbc_spec_variant(rec=rec, plug=plug, housing=housing, neck=neck, mass=mass, chamfer=chamfer)
    bw, bt, bl = body
    z0 = USBC_PLUG_LEN + neck[2]
    v.kind = "reader"
    v.peg_boxes = [((0.0, 0.0, z0 + bl / 2), (bw / 2, bt / 2, bl / 2))]
    v.grasp_z = z0 + bl / 2          # pads at the body's centre
    v.grip_half = bw / 2             # pads close across the body's width
    return v


def sample_reader(n: int, seed: int, *, clearance="spec", dim_jitter=0.0005,
                  mass=(0.022, 0.032), chamfer=(0.0, 0.0)) -> list[Variant]:
    """Reader pool. clearance="spec": receptacle and plug drawn uniformly within the USB-C
    tolerances (opening 8.32-8.40 x 2.53-2.59 mm, plug 8.22-8.28 x 2.37-2.43 mm, i.e. 20-90 um
    per side across the width and 50-110 um across the thickness); a number c: a looser
    curriculum part, plug = opening - 2c on both axes. Neck/body +-dim_jitter, mass uniform."""
    rng = np.random.default_rng(seed)
    j = lambda x: float(x + rng.uniform(-dim_jitter, dim_jitter))  # noqa: E731
    out = []
    for _ in range(n):
        if clearance == "spec":
            rec = (rng.uniform(0.00832, 0.00840), rng.uniform(0.00253, 0.00259))
            plug = (rng.uniform(0.00822, 0.00828), rng.uniform(0.00237, 0.00243))
        else:
            rec = USBC_REC
            plug = (rec[0] - 2 * clearance, rec[1] - 2 * clearance)
        out.append(reader_variant(rec=rec, plug=plug, body=tuple(j(x) for x in READER_BODY),
                                  neck=tuple(j(x) for x in READER_NECK), mass=rng.uniform(*mass),
                                  chamfer=float(rng.uniform(*chamfer))))
    return out


def scale_variant(opening: float, clearance: float, rng=None) -> Variant:
    """Phase D grid cell (plan §10.1): stadium, aspect 2.0, long side `opening`.

    The insertion feature scales with the opening; it sits under a carrier body (handle) at
    least 20 x 8 mm so the same grasp works at every size (USB-C-like construction).
    Depth = clip(0.33 * opening, 3, 20) mm; mass 20-60 g (randomized if rng given).
    """
    w, h = opening, opening / 2
    depth = float(np.clip(0.33 * w, 0.003, 0.020))
    ins = depth + 0.002
    hw, ht = max(w + 0.004, 0.020), max(h + 0.003, 0.008)
    hl = 0.030
    mass = float(rng.uniform(0.02, 0.06)) if rng is not None else 0.04
    return Variant(hole_w=w, hole_h=h, clearance=clearance, depth=depth,
                   plate_size=max(2.5 * w, 0.025), peg_length=ins, peg_mass=mass, n_sides=24,
                   bottom=0.002, kind="scale",
                   handle={"type": "rect", "width": hw, "height": ht}, handle_height=hl,
                   grasp_z=ins + hl - 0.010, grip_half=ht / 2)


def sample_scale_cell(n: int, seed: int, opening: float, clearance: float) -> list[Variant]:
    rng = np.random.default_rng(seed)
    return [scale_variant(opening, clearance, rng) for _ in range(n)]


# ------------------------------------------------------------------- mjfit -> pieces
@dataclass
class Pieces:
    fixture: list[np.ndarray]   # vertex arrays in our fixture frame
    peg: list[np.ndarray]       # vertex arrays in our peg frame
    hole_poly: np.ndarray       # (n, 2) opening outline in fixture xy
    peg_poly: np.ndarray        # (n, 2) peg outline in peg xy
    islands: list[np.ndarray]


def extract_pieces(v: Variant) -> Pieces:
    root = ET.fromstring(v.mjfit_xml())
    meshes = {m.get("name"): np.array(m.get("vertex").split(), float).reshape(-1, 3)
              for m in root.iter("mesh")}
    plate_z = v.bottom + v.depth / 2          # mjfit plate body origin height
    fixture, peg = [], []
    for body in root.iter("body"):
        name = body.get("name")
        for g in body.findall("geom"):
            verts = meshes[g.get("mesh")].copy()
            if name == "plate":
                verts[:, 2] -= v.depth / 2      # top face to z=0
                fixture.append(verts)
            elif name == "peg":
                verts[:, 2] += v.peg_length / 2  # bottom face to z=0
                peg.append(verts)
    # mjfit centres all outlines on the hole centroid (center=True); reproduce that shift.
    from mjfit import _polygon as pg  # noqa: PLC0415
    from mjfit._peg import _centroid  # noqa: PLC0415 (private but stable)
    hole_poly = mjfit.shapes.from_spec(v.hole(), name="hole")
    shift = _centroid(hole_poly)
    hole_poly = hole_poly - shift
    peg_poly = pg.offset_inward(hole_poly, v.clearance)
    isl = [mjfit.shapes.from_spec(s, name="island") - shift for s in v.islands]
    del plate_z
    if v.chamfer > 0:
        fixture = _chamfer(fixture, hole_poly, v.chamfer)
    if v.plug_widen:              # only the insert (z <= peg_length), not the handle above it
        for pc in peg:
            if pc[:, 2].max() <= v.peg_length + 1e-9:
                pc[:, 0] += np.sign(pc[:, 0]) * v.plug_widen
        peg_poly = peg_poly + np.c_[np.sign(peg_poly[:, 0]) * v.plug_widen, np.zeros(len(peg_poly))]
    for c, h in v.peg_boxes:      # e.g. the SD reader's wide body above its neck
        peg.append(np.array(c) + np.array(list(itertools.product(*[(-x, x) for x in h]))))
    return Pieces(fixture, peg, hole_poly, peg_poly, isl)


def _chamfer(fixture: list[np.ndarray], hole_poly: np.ndarray, c: float) -> list[np.ndarray]:
    """45 deg lead-in of size c: in every plate piece, each top vertex on the hole outline
    becomes the outline point c lower plus its outward offset (miter) point on the top face.
    Pieces are convex hulls, so this cuts the top inner edge of each wedge; neighbouring
    wedges share the new vertices, so the chamfer surface has no gaps."""
    from mjfit import _polygon as pg  # noqa: PLC0415
    off = pg.offset_outward(hole_poly, c)
    assert len(off) == len(hole_poly), "chamfer: offset changed the vertex count"
    out = []
    for f in fixture:
        d = np.linalg.norm(f[:, None, :2] - hole_poly[None], axis=2)
        j, dmin = d.argmin(1), d.min(1)
        top_rim = (np.abs(f[:, 2]) < 1e-9) & (dmin < 1e-9)
        if not top_rim.any():
            out.append(f)
            continue
        new = [f[~top_rim]]
        for jj in j[top_rim]:
            new.append([[*hole_poly[jj], -c], [*off[jj], 0.0]])
        out.append(np.vstack(new))
    return out


# ----------------------------------------------------------------------------- robots
@dataclass
class RobotInfo:
    name: str
    arm_joints: list[str]
    finger_joints: list[str]
    tcp_site: str
    base_body: str
    kp: np.ndarray
    kv: np.ndarray
    action_scale: float
    home: np.ndarray
    grip_force: float           # nominal force per pad (N)
    max_opening: float          # per finger travel (m)
    pad_geoms: list[str] = field(default_factory=list)
    no_collide: list[str] = field(default_factory=list)   # geoms to switch off after compiling
    gripper_geoms: list[str] = field(default_factory=list)
    nom_yaw_deg: float = 0.0    # gripper yaw about the fixture normal at the nominal grasp


def _set_position_gains(act, kp, kv):
    act.gainprm[0] = kp
    act.biasprm[0] = 0.0
    act.biasprm[1] = -kp
    act.biasprm[2] = -kv


def _strip_collisions(spec: mujoco.MjSpec, keep: set[str]):
    for g in spec.geoms:
        if g.name not in keep:
            g.contype = 0
            g.conaffinity = 0


def _add_parallel_gripper(spec: mujoco.MjSpec, parent: mujoco.MjsBody, pos, quat):
    """Simplified Robotiq 2F-85: housing + two coupled slide fingers with 22x37.5 mm pads.

    Finger joint q in [0, 0.0425] is the closing displacement of each finger from fully
    open (85 mm). Pads close along the gripper y axis; the TCP is the pad centre.
    """
    g = parent.add_body(name="gripper", pos=pos, quat=quat, gravcomp=1)
    g.mass = 0.925 - 2 * 0.05
    g.ipos = [0, 0, 0.05]
    g.inertia = [1.2e-3, 1.2e-3, 6e-4]
    g.explicitinertial = True
    g.add_geom(name="gripper_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.035, 0.047, 0.04],
               pos=[0, 0, 0.04], rgba=[0.15, 0.15, 0.17, 1], contype=GRIP_BITS[0],
               conaffinity=GRIP_BITS[1])
    pad_z = 0.135
    g.add_site(name="tcp", pos=[0, 0, pad_z], size=[0.004, 0, 0], rgba=[1, 0, 0, 1])
    # wrist camera on the housing side, looking along the fingers at the part tip (§2.3)
    c = np.array([0.09, 0.0, 0.08])
    tgt = np.array([0.0, 0.0, 0.25])
    zc = -(tgt - c) / np.linalg.norm(tgt - c)          # MuJoCo cameras look along -z
    yc = np.cross(zc, [0, 1, 0])
    yc /= np.linalg.norm(yc)
    xc = np.cross(yc, zc)
    g.add_camera(name="wrist", pos=c.tolist(), xyaxes=np.r_[xc, yc].tolist(), fovy=75)
    g.add_geom(name="wrist_cam_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.02, 0.012],
               pos=(c + [0.012, 0, -0.012]).tolist(), rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0, group=3)
    half_open = 0.0425
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        f = g.add_body(name=f"finger_{side}", pos=[0, sgn * (half_open + 0.004), pad_z], gravcomp=1)
        f.mass = 0.05
        f.inertia = [2e-5, 2e-5, 1e-5]
        f.explicitinertial = True
        f.add_joint(name=f"finger_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, -sgn, 0],
                    range=[0, half_open], damping=5.0, armature=0.01)
        f.add_geom(name=f"finger_{side}_link", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.011, 0.006, 0.045], pos=[0, sgn * 0.006, -0.06],
                   rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)
        f.add_geom(name=f"pad_{side}", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.011, 0.004, 0.01875], rgba=[0.1, 0.1, 0.1, 1], priority=1,
                   friction=[1.0, 0.005, 0.0001], solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                   contype=GRIP_BITS[0], conaffinity=GRIP_BITS[1], condim=4)
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="finger_left", name2="finger_right")
    eq.data[:5] = [0, 1, 0, 0, 0]
    eq.solref = [0.005, 1]
    for side in ("left", "right"):
        a = spec.add_actuator(name=f"grip_{side}", target=f"finger_{side}",
                              trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.ctrlrange = [0, 400]
        a.ctrllimited = 1
        a.gainprm[0] = 1.0  # motor: force = ctrl


# Backlash limit constraint. MuJoCo's default (solref 0.02 s, solimp 0.9/0.95) is a soft spring:
# under a 5 N wrist load every hinge then turns ~0.011 rad (~38 arcmin) whatever the nominal zone,
# i.e. it modelled ~0.6 deg of series compliance, not backlash (found 2026-10-21; all runs before
# this fix). Stiff limits keep the hinge within ~1.2x its zone (CPU check in docs/FINDINGS.md).
BACKLASH_LIMIT_SOLREF = (0.002, 1.0)
BACKLASH_LIMIT_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)


def add_backlash(spec: mujoco.MjSpec, joints: list[str], arcmin: float, stiffness=0.05, damping=0.01,
                 legacy_soft_limit: bool = False):
    """Plan §9.3: a passive dead-zone hinge in series with each motor joint (same body, same
    axis), free within +-arcmin/2 and weakly sprung. Encoders read only the motor joint.
    legacy_soft_limit=True reproduces the pre-fix model (soft default limit, see above)."""
    half = math.radians(arcmin / 60.0) / 2
    for j in joints:
        jt = spec.joint(j)
        body = jt.parent
        h = body.add_joint(name=f"{j}_backlash", type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(jt.axis),
                           pos=list(jt.pos), range=[-half, half], limited=1, stiffness=[stiffness, 0, 0],
                           damping=[damping, 0, 0], armature=1e-5)
        if not legacy_soft_limit:
            h.solref_limit = list(BACKLASH_LIMIT_SOLREF)
            h.solimp_limit = list(BACKLASH_LIMIT_SOLIMP)


def sphere_pads(spec: mujoco.MjSpec, info: "RobotInfo", n: int = 3, radius: float = 0.0012):
    """Replace each box pad's contact by an n x n grid of spheres on its inner face.

    MuJoCo's box-mesh multi-contact (CPU and MuJoCo Warp) can return only an edge (2 points) or no
    contact when a narrow part's face is rotated within the pad plane and the two rectangles
    overlap only partly (Phase E finding, docs/FINDINGS.md). Sphere-mesh contacts give one robust
    point each, independent of face clipping. The sphere surfaces lie exactly on the pad's inner
    face; the box stays as a visual and stops colliding. The spheres take the pad's contact
    parameters and become the robot's pad geoms (so the env's per-reset pad friction and time
    constant randomization applies to them).
    """
    m0 = spec.compile()
    d0 = mujoco.MjData(m0)
    for j in info.finger_joints:   # half open: closed FlexPoint pads overlap past the tcp, which
        d0.qpos[m0.jnt_qposadr[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, j)]] = info.max_opening / 2
    mujoco.mj_kinematics(m0, d0)   # would put the spheres on the pads' outer faces
    tcp = d0.site_xpos[mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_SITE, info.tcp_site)]
    new = []
    # look geoms up by iterating: after add_geom, spec.geom(name) can return a stale element (a
    # sphere just added to the other pad), which switched off one contact sphere per gripper
    find = lambda name: next(x for x in spec.geoms if x.name == name)   # noqa: E731
    for pname in info.pad_geoms:
        g = find(pname)
        gid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, pname)
        # use the COMPILED pose and parent body: the compiler may fuse a joint-less pad body
        # into its parent, and then the geom's spec-local pose is in a different frame
        body = spec.body(mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, m0.geom_bodyid[gid]))
        gpos = m0.geom_pos[gid].copy()
        Rl = np.zeros(9)
        mujoco.mju_quat2Mat(Rl, m0.geom_quat[gid])
        Rl = Rl.reshape(3, 3)
        size = m0.geom_size[gid].copy()
        thin = int(np.argmin(size))
        Rw = d0.geom_xmat[gid].reshape(3, 3)
        s = float(np.sign((Rw.T @ (tcp - d0.geom_xpos[gid]))[thin]))     # inner face side
        a1, a2 = [i for i in range(3) if i != thin]
        for i, j in itertools.product(range(n), range(n)):
            loc = np.zeros(3)
            loc[a1] = (2 * (i + 0.5) / n - 1) * (size[a1] - radius)
            loc[a2] = (2 * (j + 0.5) / n - 1) * (size[a2] - radius)
            loc[thin] = s * (size[thin] - radius)
            sg = body.add_geom(name=f"{pname}_s{i}{j}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                               size=[radius, 0, 0], pos=(gpos + Rl @ loc).tolist())
            sg.contype, sg.conaffinity = g.contype, g.conaffinity
            sg.condim, sg.priority = g.condim, g.priority
            sg.friction, sg.solref, sg.solimp = list(g.friction), list(g.solref), list(g.solimp)
            sg.margin, sg.gap, sg.group = g.margin, g.gap, 3
            sg.rgba = [0.9, 0.2, 0.2, 1.0]
            new.append(sg.name)
        g = find(pname)
        g.contype, g.conaffinity = 0, 0
        info.no_collide.append(pname)       # also enforced on the compiled model (see build)
    info.pad_geoms = new
    return info


def tip_points_tcp(m: mujoco.MjModel, lay) -> np.ndarray:
    """Fingertip collision points in the tcp frame at qpos0 (FlexPoint tip hulls; else the pad
    boxes' corners). Used to keep a tilted top grasp's lowest tip point off the table."""
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    R_s, p_s = d0.site_xmat[lay.tcp_site].reshape(3, 3), d0.site_xpos[lay.tcp_site]
    hulls = [g for g in range(m.ngeom) if m.geom(g).name.endswith("_tip_hull")]
    pts = []
    pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_pad", "right_pad")]
    for g in hulls or [g for g in pads if g >= 0]:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            a = m.mesh_vertadr[m.geom_dataid[g]]
            V = m.mesh_vert[a:a + m.mesh_vertnum[m.geom_dataid[g]]]
        else:
            V = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * m.geom_size[g]
        pts.append((V @ d0.geom_xmat[g].reshape(3, 3).T + d0.geom_xpos[g] - p_s) @ R_s)
    return np.concatenate(pts)


def load_robot(name: str, *, gains=None, backlash_arcmin: float = 0.0,
               pad_spheres: int = 0, gripper="linear") -> tuple[mujoco.MjSpec, RobotInfo]:
    if gripper not in ("linear", "flexpoint") or (gripper == "flexpoint" and name != "yam"):
        raise ValueError("gripper must be linear, or flexpoint on a YAM")
    spec, info = _load_robot(name, gains=gains, gripper=gripper)
    if backlash_arcmin > 0:   # before sphere_pads: joints added after its spec.compile() land on link_1
        add_backlash(spec, info.arm_joints, backlash_arcmin)
    if pad_spheres > 0:
        info = sphere_pads(spec, info, pad_spheres)
    return spec, info


def _load_robot(name: str, *, gains=None, gripper="linear") -> tuple[mujoco.MjSpec, RobotInfo]:
    if name == "flexiv":
        spec = mujoco.MjSpec.from_file(str(MENAGERIE / "flexiv_rizon4s" / "flexiv_rizon4s.xml"))
        for s in list(spec.sensors):
            spec.delete(s)
        for k in list(spec.keys):
            spec.delete(k)
        for lt in list(spec.lights):
            spec.delete(lt)
        for b in spec.bodies:
            if b.name:
                b.gravcomp = 1
        _strip_collisions(spec, keep=set())
        link7 = spec.body("link7")
        _add_parallel_gripper(spec, link7, pos=[0, 0, 0.124], quat=[1, 0, 0, 0])
        joints = [f"joint{i}" for i in range(1, 8)]
        kp = np.full(7, 80.0) if gains is None else np.asarray(gains[0], float)
        kv = np.full(7, 20.0) if gains is None else np.asarray(gains[1], float)
        for j, act in zip(joints, [spec.actuator(f"joint{i}") for i in range(1, 8)]):
            _set_position_gains(act, *(kp[joints.index(j)], kv[joints.index(j)]))
            act.forcerange = [-1e3, 1e3]
            act.forcelimited = 0
        info = RobotInfo(name="flexiv", arm_joints=joints, finger_joints=["finger_left", "finger_right"],
                         tcp_site="tcp", base_body="base", kp=kp, kv=kv, action_scale=0.02,
                         home=np.array([0.0, -0.35, 0.0, 1.75, 0.0, 0.53, 0.0]),
                         grip_force=20.0, max_opening=0.0425,
                         pad_geoms=["pad_left", "pad_right"], gripper_geoms=["gripper_housing"])
        return spec, info
    if name == "yam":
        spec = mujoco.MjSpec.from_file(str(MENAGERIE / "i2rt_yam" / "yam_linear_4310.xml"))
        if gripper == "flexpoint":
            from blind_insertion.assets.flexpoint import install
            install(spec)
        for k in list(spec.keys):
            spec.delete(k)
        for lt in list(spec.lights):
            spec.delete(lt)
        for b in spec.bodies:
            if b.name:
                b.gravcomp = 1
        joints = [f"joint{i}" for i in range(1, 7)]
        if gains is None:
            kp = np.array([80, 80, 80, 30, 30, 30], float)
            kv = np.array([6, 6, 6, 2, 2, 2], float)
        else:
            kp, kv = (np.asarray(x, float) for x in gains)
        for i, j in enumerate(joints):
            jt = spec.joint(j)
            jt.armature = 0.032 if i < 3 else 0.0018
            jt.damping = [0.5, 0.0, 0.0]
            jt.frictionloss = 0.01
            act = spec.actuator(j)
            _set_position_gains(act, kp[i], kv[i])
        pads = ["left_pad", "right_pad"]
        finger_links = [g.name for g in spec.geoms if g.name and ("left_" in g.name or "right_" in g.name)
                        and g.name not in pads and g.classname.name == "finger_body_collision"
                        and any(k in g.name for k in ("tip", "strut", "pad_edge"))]
        finger_links += [g.name for g in spec.geoms if g.name.endswith("_tip_hull")]   # FlexPoint tips
        _strip_collisions(spec, keep=set(pads) | {"gripper_housing"} | set(finger_links))
        by_name = {g.name: g for g in spec.geoms if g.name}   # not spec.geom(): see sphere_pads
        for gname in finger_links:   # fingertips must not pass through the fixture
            by_name[gname].contype, by_name[gname].conaffinity = FINGER_LINK_BITS
        for gname in pads + ["gripper_housing"]:
            by_name[gname].contype, by_name[gname].conaffinity = GRIP_BITS
        # finger: replace the position servo with a constant-force motor
        grip = spec.actuator("gripper")
        spec.delete(grip)
        for side in ("left", "right"):
            a = spec.add_actuator(name=f"grip_{side}", target=f"{side}_finger",
                                  trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.ctrlrange = [-400, 400]
            a.ctrllimited = 1
            a.gainprm[0] = 1.0
        info = RobotInfo(name="yam", arm_joints=joints, finger_joints=["left_finger", "right_finger"],
                         tcp_site="grasp_site", base_body="arm", kp=kp, kv=kv, action_scale=0.04,
                         home=np.array([0.0, 1.96, 1.29, -0.9, 0.0, 0.0]),
                         grip_force=-20.0, max_opening=0.048 if gripper == "flexpoint" else 0.0475, pad_geoms=pads,
                         gripper_geoms=["gripper_housing"], nom_yaw_deg=180.0)
        return spec, info
    raise ValueError(name)


# ----------------------------------------------------------------------------- layout
@dataclass
class Layout:
    """Indices and per-variant data the env writes into batched model fields."""
    robot: RobotInfo
    variants: list[Variant]
    fixture_body: int
    peg_body: int
    peg_qadr: int
    peg_dofadr: int
    arm_qadr: np.ndarray
    arm_dofadr: np.ndarray
    finger_qadr: np.ndarray
    arm_act: np.ndarray
    grip_act: np.ndarray
    tcp_site: int
    fixture_slots: np.ndarray        # geom ids
    peg_slots: np.ndarray
    pad_geoms: np.ndarray
    # per-variant (K, S, ...) arrays
    fix_dataid: np.ndarray
    fix_pos: np.ndarray
    fix_quat: np.ndarray
    fix_size: np.ndarray
    fix_aabb: np.ndarray
    fix_rbound: np.ndarray
    peg_dataid: np.ndarray
    peg_pos: np.ndarray
    peg_quat: np.ndarray
    peg_size: np.ndarray
    peg_aabb: np.ndarray
    peg_rbound: np.ndarray
    peg_mass: np.ndarray             # (K,)
    peg_inertia: np.ndarray          # (K, 3)
    peg_ipos: np.ndarray
    peg_iquat: np.ndarray
    peg_invweight: np.ndarray        # (K, 2)
    peg_dof_invweight: np.ndarray    # (K, 6)
    hole_polys: list[np.ndarray]
    peg_polys: list[np.ndarray]
    # derived per-variant scalars
    depth: np.ndarray
    peg_length: np.ndarray
    peg_half_short: np.ndarray       # half the peg width across the pads
    grasp_z: np.ndarray              # pad centre height above the peg bottom
    edge_points: np.ndarray          # (K, 64, 2) points on the peg bottom edge (peg xy)
    fix_corners: np.ndarray          # (K, 8, 3) fixture bounding-box corners (fixture frame)


def _half_width_at_x0(poly: np.ndarray) -> float:
    """Half the peg's extent along y on the grasp line x = 0 (pads close along y)."""
    ys = []
    closed = np.vstack([poly, poly[:1]])
    for (x1, y1), (x2, y2) in zip(closed[:-1], closed[1:]):
        if (x1 - 0) * (x2 - 0) <= 0 and x1 != x2:
            ys.append(y1 + (0 - x1) * (y2 - y1) / (x2 - x1))
        elif x1 == 0:
            ys.append(y1)
    return (max(ys) - min(ys)) / 2


def _resample_outline(poly: np.ndarray, n: int) -> np.ndarray:
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(t, s, closed[:, 0])
    y = np.interp(t, s, closed[:, 1])
    return np.column_stack([x, y])


def _mini_model(pieces: Pieces, v: Variant, n_fix: int, n_peg: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    fb = spec.worldbody.add_body(name="fixture")
    pb = spec.worldbody.add_body(name="peg")
    pb.add_freejoint()
    for j, verts in enumerate(pieces.fixture):
        spec.add_mesh(name=f"f{j}", uservert=verts.ravel().tolist())
        fb.add_geom(name=f"fix_{j}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"f{j}")
    vol = []
    for j, verts in enumerate(pieces.peg):
        spec.add_mesh(name=f"g{j}", uservert=verts.ravel().tolist())
        pb.add_geom(name=f"peg_{j}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"g{j}")
    m = spec.compile()
    # peg mass: scale densities so the total equals v.peg_mass (inertia scales linearly)
    del vol
    return m


def build(robot: str, variants: list[Variant], *, gains=None, timestep=0.001,
          multiccd: bool = False, extra=None, backlash_arcmin: float = 0.0,
          pad_spheres: int = 0, table_top: float | None = None,
          fixture_boxes=None, gripper="linear") -> tuple[mujoco.MjModel, Layout]:
    """`table_top`: add the YAM workcell table (`YAM_TABLE`) with its top at this height.
    `fixture_boxes`: extra colliding boxes on the fixture body [(centre, half)] in the fixture
    frame, e.g. the mini PC's case around the port plate (task H)."""
    spec, info = load_robot(robot, gains=gains, backlash_arcmin=backlash_arcmin, pad_spheres=pad_spheres, gripper=gripper)
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

    pieces = [extract_pieces(v) for v in variants]
    n_fix = max(len(p.fixture) for p in pieces)
    n_peg = max(len(p.peg) for p in pieces)

    # meshes of every variant
    for k, p in enumerate(pieces):
        for j, verts in enumerate(p.fixture):
            spec.add_mesh(name=f"v{k}_f{j}", uservert=verts.ravel().tolist())
        for j, verts in enumerate(p.peg):
            spec.add_mesh(name=f"v{k}_g{j}", uservert=verts.ravel().tolist())

    # mocap: MuJoCo Warp computes static (world-welded) geom poses once in make_data, so a
    # per-world fixture pose and per-world geom frames need a mocap body.
    fix = spec.worldbody.add_body(name="fixture", pos=[0.65, 0.0, 0.1], mocap=True)
    fix_geoms = []
    for j in range(n_fix):
        mesh = f"v0_f{min(j, len(pieces[0].fixture) - 1)}"
        fix_geoms.append(fix.add_geom(
            name=f"fix_{j}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
            rgba=[0.70, 0.75, 0.82, 1], priority=1, friction=[1.0, 0.01, 0.001],
            solref=[0.005, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2], condim=4,
            contype=FIX_BITS[0], conaffinity=FIX_BITS[1], margin=0, gap=0))
    for j, (c, h) in enumerate(fixture_boxes or []):
        fix.add_geom(name=f"fixbox_{j}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(h), pos=list(c),
                     rgba=[0.78, 0.78, 0.80, 1], friction=[1.0, 0.01, 0.001], solref=[0.005, 1],
                     solimp=[0.99, 0.999, 0.001, 0.5, 2], condim=4,
                     contype=FIX_BITS[0], conaffinity=FIX_BITS[1])
    if table_top is not None:
        t = YAM_TABLE
        spec.worldbody.add_geom(
            name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[*t["half"], t["height"] / 2],
            pos=[*t["center"], table_top - t["height"] / 2], rgba=[0.55, 0.45, 0.35, 1],
            friction=[1.0, 0.01, 0.001], condim=3, contype=FIX_BITS[0], conaffinity=FIX_BITS[1])
    peg = spec.worldbody.add_body(name="peg", pos=[0.65, 0.0, 0.3])
    peg.add_freejoint(name="peg_joint")
    for j in range(n_peg):
        mesh = f"v0_g{min(j, len(pieces[0].peg) - 1)}"
        peg.add_geom(name=f"peg_{j}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                     rgba=[0.90, 0.30, 0.30, 1], friction=[1.0, 0.01, 0.001],
                     solref=[0.002, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2], condim=4,
                     contype=PEG_BITS[0], conaffinity=PEG_BITS[1], margin=0, gap=0, density=1000)
    if extra is not None:
        extra(spec)
    m = spec.compile()
    for name in info.no_collide:   # the compiler can keep a fused body's old geom flags
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        m.geom_contype[i] = m.geom_conaffinity[i] = 0

    def gid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)

    def mid(name):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, name)

    K = len(variants)
    fix_slots = np.array([gid(f"fix_{j}") for j in range(n_fix)])
    peg_slots = np.array([gid(f"peg_{j}") for j in range(n_peg)])
    out = {k: [] for k in ("fix_dataid", "fix_pos", "fix_quat", "fix_size", "fix_aabb", "fix_rbound",
                           "peg_dataid", "peg_pos", "peg_quat", "peg_size", "peg_aabb", "peg_rbound",
                           "peg_mass", "peg_inertia", "peg_ipos", "peg_iquat", "peg_invweight",
                           "peg_dof_invweight")}
    edge_pts, half_short, grasp_z, corners = [], [], [], []
    for k, (p, v) in enumerate(zip(pieces, variants)):
        mm = _mini_model(p, v, n_fix, n_peg)
        for role, slots, src in (("fix", n_fix, p.fixture), ("peg", n_peg, p.peg)):
            prefix = "f" if role == "fix" else "g"
            ids, pos, quat, size, aabb, rb = [], [], [], [], [], []
            for j in range(slots):
                jj = min(j, len(src) - 1)
                g_m = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, f"{role}_{jj}")
                ids.append(mid(f"v{k}_{prefix}{jj}"))
                ps = mm.geom_pos[g_m].copy()
                if j >= len(src):
                    ps[2] += PARK_Z
                pos.append(ps)
                quat.append(mm.geom_quat[g_m].copy())
                size.append(mm.geom_size[g_m].copy())
                aabb.append(mm.geom_aabb[g_m].copy())
                rb.append(mm.geom_rbound[g_m])
            out[f"{role}_dataid"].append(ids)
            out[f"{role}_pos"].append(pos)
            out[f"{role}_quat"].append(quat)
            out[f"{role}_size"].append(size)
            out[f"{role}_aabb"].append(aabb)
            out[f"{role}_rbound"].append(rb)
        pb = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, "peg")
        scale = v.peg_mass / mm.body_mass[pb]
        out["peg_mass"].append(v.peg_mass)
        out["peg_inertia"].append(mm.body_inertia[pb] * scale)
        out["peg_ipos"].append(mm.body_ipos[pb].copy())
        out["peg_iquat"].append(mm.body_iquat[pb].copy())
        out["peg_invweight"].append(mm.body_invweight0[pb] / scale)
        dadr = mm.jnt_dofadr[mm.body_jntadr[pb]]
        out["peg_dof_invweight"].append(mm.dof_invweight0[dadr:dadr + 6] / scale)
        edge_pts.append(_resample_outline(p.peg_poly, 64))
        fv = np.concatenate([f.reshape(-1, 3) for f in p.fixture])
        lo, hi = fv.min(0), fv.max(0)
        corners.append([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        half_short.append(v.grip_half if v.grip_half is not None else _half_width_at_x0(p.peg_poly))
        grasp_z.append(v.grasp_z if v.grasp_z is not None else 0.75 * v.peg_length)
    arr = {k: np.asarray(v_) for k, v_ in out.items()}

    def jq(name):
        return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]

    def jd(name):
        return m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)]

    acts = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in info.arm_joints]
    grips = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"grip_{s}") for s in ("left", "right")]
    pegb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "peg")
    layout = Layout(
        robot=info, variants=variants,
        fixture_body=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "fixture"),
        peg_body=pegb, peg_qadr=int(jq("peg_joint")), peg_dofadr=int(jd("peg_joint")),
        arm_qadr=np.array([jq(j) for j in info.arm_joints]),
        arm_dofadr=np.array([jd(j) for j in info.arm_joints]),
        finger_qadr=np.array([jq(j) for j in info.finger_joints]),
        arm_act=np.array(acts), grip_act=np.array(grips),
        tcp_site=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, info.tcp_site),
        fixture_slots=fix_slots, peg_slots=peg_slots,
        pad_geoms=np.array([gid(g) for g in info.pad_geoms]),
        hole_polys=[p.hole_poly for p in pieces], peg_polys=[p.peg_poly for p in pieces],
        depth=np.array([v.depth for v in variants]),
        peg_length=np.array([v.peg_length for v in variants]),
        peg_half_short=np.array(half_short), edge_points=np.asarray(edge_pts),
        grasp_z=np.array(grasp_z), fix_corners=np.asarray(corners),
        **arr)
    return m, layout


def apply_variant_cpu(m: mujoco.MjModel, lay: Layout, k: int):
    """Write variant k into a CPU MjModel (for single-world checks and rendering)."""
    m.geom_dataid[lay.fixture_slots] = lay.fix_dataid[k]
    m.geom_pos[lay.fixture_slots] = lay.fix_pos[k]
    m.geom_quat[lay.fixture_slots] = lay.fix_quat[k]
    m.geom_size[lay.fixture_slots] = lay.fix_size[k]
    m.geom_aabb[lay.fixture_slots] = lay.fix_aabb[k].reshape(len(lay.fixture_slots), 6)
    m.geom_rbound[lay.fixture_slots] = lay.fix_rbound[k]
    m.geom_dataid[lay.peg_slots] = lay.peg_dataid[k]
    m.geom_pos[lay.peg_slots] = lay.peg_pos[k]
    m.geom_quat[lay.peg_slots] = lay.peg_quat[k]
    m.geom_size[lay.peg_slots] = lay.peg_size[k]
    m.geom_aabb[lay.peg_slots] = lay.peg_aabb[k].reshape(len(lay.peg_slots), 6)
    m.geom_rbound[lay.peg_slots] = lay.peg_rbound[k]
    b = lay.peg_body
    m.body_mass[b] = lay.peg_mass[k]
    m.body_inertia[b] = lay.peg_inertia[k]
    m.body_ipos[b] = lay.peg_ipos[k]
    m.body_iquat[b] = lay.peg_iquat[k]
    m.body_invweight0[b] = lay.peg_invweight[k]
    m.dof_invweight0[lay.peg_dofadr:lay.peg_dofadr + 6] = lay.peg_dof_invweight[k]
