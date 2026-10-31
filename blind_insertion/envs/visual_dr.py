"""Per-world visual domain randomization and the nominal (real) look of the bimanual Warp runtime.

Used by `BimanualRuntime` (`BimanualCfg.visual_look`, `BimanualCfg.visual_dr`). Every reset draws,
per world: colours and gloss of each appearance group (PC case, reader, table, wall, backdrop,
arms, gripper, ...), patterns on the table and wall (procedural or real-photo patches), the key and
fill lights (elevation, intensity, colour temperature, shadows) and the camera extrinsic and
intrinsic error. Every rendered frame then goes through a camera model on the GPU: per-camera
nominal gain, radial distortion, blur, exposure, per-channel gain, brightness, sensor noise.

Seed 0 = real look: with `visual_dr=None` (or in a world drawn as nominal, `nominal_prob`) the
images are the nominal look: `configs/warp_look_duchamp3.json`, fitted to the recorded real frames
in the Warp renderer by `scripts/real2sim/fit_warp_look.py`.

What MuJoCo Warp 3.12 batches per world (Model fields with a leading `*` dimension, read as
`field[worldid % shape[0]]` in the render kernel) and what it does not:
  per world : geom_matid, geom_rgba, mat_rgba / specular / shininess / emission / texid /
              texrepeat, light_type / pos / dir / diffuse / ambient / specular / castshadow / active,
              cam_pos / cam_quat / cam_fovy / cam_intrinsic.
  global    : the texture images (one array for all worlds: per world we pick among the model's
              textures through mat_texid), the headlight (ambient / diffuse / specular are compile-time
              constants of the render kernel), use_shadows (a render-context switch; per-world shadows
              come from light_castshadow), cam_sensorsize, mesh geometry, alpha (ignored: every
              enabled geom renders opaque).
  textures  : sampled on MESH and PLANE geoms only (boxes give one texel). Patterns therefore go on
              two visual-only textured quads (`vdr_table_decal`, `vdr_wall_decal`) laid on the table
              top and the wall face; their nominal material equals the surface below.
Workarounds: headlight intensity is fixed per model (the key/fill lights and the image exposure carry
the per-world intensity range); alpha-0 geoms (hidden in the CPU renderer) and the wrist lenses are
left out of the Warp render context (`render_hidden`).
"""
from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / 'blind_insertion/assets/visual_dr'
TEX_SIZE = 256
N_PROCEDURAL = 8
DECAL_LIFT = 2e-4

# Appearance groups: geoms that look like one colour on the real rig. Matched by geom name prefix
# ('n:'), body name ('b:'), material name ('m:') or nominal colour; first match wins.
GROUPS = {
    'pc_case': ['n:detail_pc_body', 'n:detail_pc_lid'],
    'pc_dark': ['n:fix_', 'n:detail_pc_seam', 'n:pc_pad_', 'n:pc_socket_dark_interior'],
    'reader_trim': ['n:detail_reader_silver', 'n:detail_sd_rim'],
    'reader_body': ['n:detail_reader_shell', 'n:detail_neck_seam', 'n:detail_sd_slot', 'peg_dark'],
    'reader_plug': ['peg_light'],
    'table': ['n:table', 'n:vdr_table_decal'],
    'wall': ['n:bd_wall', 'n:vdr_wall_decal', 'n:detail_room_wall', 'n:detail_room_ceiling'],
    'room': ['n:detail_room_floor', 'n:detail_room_bench', 'n:detail_room_mat', 'n:detail_room_box', 'n:bd_gap'],
    'raceway': ['n:bd_raceway', 'n:bd_plate', 'n:bd_ryobi_label'],
    'accents': ['n:bd_ryobi', 'n:bd_strip_switch'],
    'backdrop_black': ['n:bd_plug', 'n:bd_strip', 'n:bd_device', 'n:bd_cable'],
    'mount': ['n:shared_mounting_plate'],
    'arm_dark': ['m:black', 'm:left_black'],
    'arm_light': ['m:white', 'm:left_white'],
    'tips': ['yellow'],
    'gripper': ['b:gripper', 'b:left_gripper', 'b:left_finger', 'b:right_finger', 'b:left_left_finger',
                'b:left_right_finger', 'b:calibrated_webcam_visual', 'b:left_calibrated_webcam_visual'],
}
TEXTURED = ('table', 'wall')            # groups with a textured decal
DECAL = {'table': 'vdr_table_decal', 'wall': 'vdr_wall_decal'}
# Real calibration error (key-box deck slide 16, robot-fit calibration): mm and deg per camera.
REAL_CAM_ERR_MM = {'top': 4.3, 'wrist_cam': 5.1, 'left_wrist_cam': 3.4}
REAL_CAM_ERR_DEG = {'top': .6, 'wrist_cam': 1.2, 'left_wrist_cam': 1.2}


@dataclass
class VisualDRCfg:
    """Visual DR ranges (`BimanualCfg.visual_dr`). U(lo, hi) unless noted; resampled on reset."""
    seed: int = 0
    # Fraction of episodes drawn with the nominal look (scene, lights, camera pose; the camera-image
    # noise below still applies unless `nominal_clean`).
    nominal_prob: float = .1
    nominal_clean: bool = False
    # --- appearance, per group (GROUPS) ---
    random_color_prob: float = .75          # new random colour; else nominal colour jittered
    tips_random_prob: float = .25           # yellow fingertips: rarer (they stay mostly yellow on the rig)
    hue_range: tuple = (0., 1.)
    saturation_range: tuple = (0., .8)
    value_range: tuple = (.03, .95)         # HSV value of the random colour (dark to light)
    nominal_jitter: float = .2              # jittered nominal: brightness x U(1 -/+ j), per channel x U(1 -/+ j/4)
    specular_range: tuple = (0., 1.)        # flat (0) ... glossy (1), MuJoCo material specular
    shininess_range: tuple = (0., 1.)       # x 128 = Phong exponent
    emission_range: tuple = (0., .05)
    # --- patterns on the table and wall decals ---
    texture_prob: float = .5
    real_texture_prob: float = .4           # of the textured episodes: a real-photo patch (assets/visual_dr)
    texrepeat_range: tuple = (2., 12.)      # pattern repeats across the surface
    # --- lighting: key light (point light) placed on a sphere around the workspace ---
    light_elevation_deg: tuple = (35., 90.)
    light_azimuth_deg: tuple = (0., 360.)
    light_distance_m: tuple = (.9, 1.6)
    light_intensity: tuple = (.3, 1.)       # key diffuse = I / 0.65 x nominal key diffuse
    light_kelvin: tuple = (3200., 6500.)    # colour temperature of key, fill and ambient
    shadow_prob: float = .5                 # key light casts shadows
    fill_scale: tuple = (.2, 1.2)           # fill diffuse x nominal
    ambient_scale: tuple = (.5, 1.5)        # per-light ambient x nominal
    # --- camera image, on the rendered frame (per frame if image_per_frame, else per episode) ---
    gain: tuple = (.93, 1.07)               # per channel
    brightness: float = 12.                 # +/- offset, 0-255 units
    exposure: float = .06                   # +/- relative
    noise_std: tuple = (1., 4.)             # per-episode Gaussian sensor-noise level, 0-255 units
    blur_sigma_px: tuple = (0., 1.)         # per-episode Gaussian blur at the render resolution
    radial_k1: float = .02                  # +/- radial distortion residual (normalized radius 1 = half width)
    image_per_frame: bool = True
    # --- camera pose and intrinsics (per episode) ---
    # Extrinsic error: random direction, magnitude U(0, cam_err_scale x real) (mean = real error when 2).
    cam_err_scale: float = 2.
    cam_pos_mm: dict = field(default_factory=lambda: dict(REAL_CAM_ERR_MM))
    cam_rot_deg: dict = field(default_factory=lambda: dict(REAL_CAM_ERR_DEG))
    focal_jitter: float = .015              # +/- relative fx, fy
    principal_jitter: float = .005          # +/- principal point, fraction of the sensor size

    @staticmethod
    def make(v):
        if v is None or isinstance(v, VisualDRCfg):
            return v
        if v is True:
            return VisualDRCfg()
        if isinstance(v, dict):
            return VisualDRCfg(**v)
        raise TypeError('visual_dr must be None, True, a dict or a VisualDRCfg')


# ----------------------------------------------------------------------------- textures and decals
def procedural_textures(seed=0):
    """N_PROCEDURAL RGB patterns (uint8, TEX_SIZE^2): checker, stripes, noise, wood, dots, grid,
    blotches, speckle; colours drawn once from `seed`."""
    rng = np.random.default_rng(seed)
    n = TEX_SIZE
    y, x = np.mgrid[0:n, 0:n] / n

    def col():
        return rng.uniform(.15, 1., 3)

    def smooth(scale):
        g = rng.random((scale + 1, scale + 1))
        g[-1], g[:, -1] = g[0], g[:, 0]                      # tileable
        xi, yi = x * scale, y * scale
        x0, y0 = xi.astype(int), yi.astype(int)
        fx, fy = xi - x0, yi - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        a, b = g[y0, x0] * (1 - fx) + g[y0, x0 + 1] * fx, g[y0 + 1, x0] * (1 - fx) + g[y0 + 1, x0 + 1] * fx
        return a * (1 - fy) + b * fy

    def fbm():
        return sum(smooth(s) / k for k, s in zip((1, 2, 4, 8), (4, 8, 16, 32))) / 1.875

    def mix(t, c0, c1):
        return t[..., None] * c1 + (1 - t[..., None]) * c0

    out = []
    k = rng.integers(2, 6)
    out.append(mix((((x * 2 * k).astype(int) + (y * 2 * k).astype(int)) % 2).astype(float), col(), col()))
    out.append(mix(((np.sin(2 * np.pi * (x * rng.integers(3, 9))) > 0)).astype(float), col(), col()))
    out.append(mix(fbm(), col(), col()))
    w = fbm()
    out.append(mix(.5 + .5 * np.sin(2 * np.pi * (y * 6 + 2 * w)), col() * .6, col()))
    k = rng.integers(4, 9)
    dx, dy = (x * k) % 1 - .5, (y * k) % 1 - .5
    out.append(mix((dx * dx + dy * dy < .08).astype(float), col(), col()))
    k = rng.integers(4, 10)
    out.append(mix((((x * k) % 1 < .08) | ((y * k) % 1 < .08)).astype(float), col(), col() * .4))
    b = fbm()
    out.append(np.where((b > .55)[..., None], col(), np.where((b > .4)[..., None], col(), col())))
    out.append(mix(np.clip(rng.normal(.5, .2, (n, n)), 0, 1), col() * .5, col()))
    return [(np.clip(t, 0, 1) * 255).astype(np.uint8) for t in out]


def real_textures():
    """[(name, RGB uint8)] real-photo patches (scripts/real2sim/extract_real_textures.py)."""
    import cv2
    out = []
    for p in sorted(ASSETS.glob('real_*.png')):
        img = cv2.imread(str(p))
        out.append((p.stem, cv2.cvtColor(cv2.resize(img, (TEX_SIZE, TEX_SIZE)), cv2.COLOR_BGR2RGB)))
    return out


def _quad(spec, name, corners, material, inertia_back):
    """Visual-only textured quad (two triangles, UV 0..1) on the world body; corners CCW seen from
    the visible side."""
    c = np.asarray(corners, float)
    back = c + inertia_back
    verts = np.vstack([c, back])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    spec.add_mesh(name=name + '_mesh', inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,
                  uservert=verts.ravel(), userface=faces.ravel(), usertexcoord=uv.ravel(),
                  userfacetexcoord=faces.ravel())
    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name + '_mesh',
                            material=material, contype=0, conaffinity=0, mass=0, group=0)


def add_assets(spec):
    """Scene hook (pc_bench.build_scene(extra_spec=...)): DR textures, one material per appearance group,
    and the table/wall decals. Adds no collision geometry."""
    bank = [(f'vdr_tex_{i}', img) for i, img in enumerate(procedural_textures())] + \
           [(f'vdr_{name}', img) for name, img in real_textures()]
    for name, img in bank:
        tex = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=TEX_SIZE, height=TEX_SIZE, nchannel=3)
        tex.data = np.ascontiguousarray(img).tobytes()
    for g in GROUPS:
        spec.add_material(name='vdr_' + g, rgba=[.5, .5, .5, 1], specular=.5, shininess=.5)
    for g in TEXTURED:
        spec.add_material(name=f'vdr_{g}_tex', rgba=[.5, .5, .5, 1], specular=.5, shininess=.5)
    table = next(g for g in spec.geoms if g.name == 'table')
    tx, ty, tz = table.pos
    sx, sy, sz = table.size
    z = tz + sz + DECAL_LIFT
    _quad(spec, 'vdr_table_decal', [[tx - sx, ty - sy, z], [tx + sx, ty - sy, z], [tx + sx, ty + sy, z], [tx - sx, ty + sy, z]],
          'vdr_table_tex', [0, 0, -1e-4])
    wall = next((g for g in spec.geoms if g.name == 'bd_wall'), None)
    if wall is not None:
        x = wall.pos[0] - wall.size[0] - DECAL_LIFT
        (y0, y1), (z0, z1) = (wall.pos[1] - wall.size[1], wall.pos[1] + wall.size[1]), (wall.pos[2] - wall.size[2], wall.pos[2] + wall.size[2])
        _quad(spec, 'vdr_wall_decal', [[x, y0, z0], [x, y0, z1], [x, y1, z1], [x, y1, z0]], 'vdr_wall_tex', [1e-4, 0, 0])


# ----------------------------------------------------------------------------- groups and nominal look
def render_hidden(m):
    """Geoms the Warp renderer must skip: alpha 0 (hidden in the CPU renderer, opaque in Warp) and the
    wrist-camera lenses (the left camera sits inside its lens cylinder)."""
    return [g for g in range(m.ngeom) if m.geom_rgba[g, 3] == 0 or m.geom(g).name.endswith('webcam_lens')]


@contextlib.contextmanager
def render_groups(m):
    """Temporarily move render_hidden geoms to group 5 (not rendered) while a render context is built."""
    hidden = render_hidden(m)
    old = m.geom_group[hidden].copy()
    m.geom_group[hidden] = 5
    try:
        yield hidden
    finally:
        m.geom_group[hidden] = old


def group_geoms(m):
    """{group: np.array(geom ids)} for the compiled model; hidden geoms excluded."""
    hidden = set(render_hidden(m))
    mats = {m.material(i).name: i for i in range(m.nmat)}
    out = {g: [] for g in GROUPS}
    for gid in range(m.ngeom):
        if gid in hidden or m.geom_group[gid] > 2:
            continue
        name, body = m.geom(gid).name, m.body(m.geom_bodyid[gid]).name
        rgba = m.geom_rgba[gid]
        for grp, rules in GROUPS.items():
            hit = False
            for r in rules:
                if r.startswith('n:'):
                    hit = name.startswith(r[2:])
                elif r.startswith('b:'):
                    hit = body == r[2:]
                elif r.startswith('m:'):
                    hit = m.geom_matid[gid] == mats.get(r[2:], -2)
                elif r == 'yellow':
                    hit = rgba[0] > .9 and rgba[1] > .7 and rgba[2] < .4
                elif r in ('peg_dark', 'peg_light'):
                    hit = body == 'peg' and name.startswith('peg_') and (rgba[:3].mean() < .3) == (r == 'peg_dark')
                if hit:
                    break
            if hit:
                if grp == 'gripper' and rgba[0] > .9 and rgba[1] > .7 and rgba[2] < .4:
                    continue                                    # tips matched below by colour
                out[grp].append(gid)
                break
    return {g: np.array(v, int) for g, v in out.items()}


def load_look(path):
    if path is None:
        return None
    p = Path(path)
    return json.loads((p if p.is_absolute() else REPO / p).read_text())


def apply_look(m, look, groups=None):
    """Write the fitted nominal colours (look['colors'][group], look['emission'][group]) into the compiled
    model: the group's own materials, or geom_rgba for geoms without one."""
    if not look:
        return
    groups = groups if groups is not None else group_geoms(m)
    for grp, rgb in look.get('colors', {}).items():
        ids = groups.get(grp, [])
        for gid in ids:
            mid = m.geom_matid[gid]
            if mid >= 0 and not m.material(mid).name.startswith('vdr_'):
                m.mat_rgba[mid, :3] = rgb
                m.mat_emission[mid] = look.get('emission', {}).get(grp, m.mat_emission[mid])
            elif mid < 0:
                m.geom_rgba[gid, :3] = rgb
    if 'headlight' in look:
        h = look['headlight']
        m.vis.headlight.ambient[:] = h['ambient']
        m.vis.headlight.diffuse[:] = h['diffuse']


def sync_decals(m):
    """Nominal decal materials = the effective appearance of the surface below (no texture)."""
    for grp, base_name in (('table', 'table'), ('wall', 'bd_wall')):
        try:
            mat, base = m.material(f'vdr_{grp}_tex').id, m.geom(base_name).id
        except KeyError:
            continue
        bm = m.geom_matid[base]
        if bm >= 0:
            m.mat_rgba[mat], m.mat_specular[mat] = m.mat_rgba[bm], m.mat_specular[bm]
            m.mat_shininess[mat], m.mat_emission[mat] = m.mat_shininess[bm], m.mat_emission[bm]
        else:   # geom without material: Warp's defaults (render.py DEFAULT_MAT_*)
            m.mat_rgba[mat] = m.geom_rgba[base]
            m.mat_specular[mat], m.mat_shininess[mat], m.mat_emission[mat] = .5, .5, 0.
        m.mat_texid[mat] = -1


def kelvin_rgb(k):
    """Colour-temperature tint (Tanner Helland fit), torch tensor of kelvin -> (..., 3) with mean 1."""
    t = k / 100.
    r = torch.where(t <= 66, torch.ones_like(t), (329.698727446 * (t - 60).clamp(min=1e-3) ** -.1332047592 / 255).clamp(0, 1))
    g = torch.where(t <= 66, (99.4708025861 * torch.log(t) - 161.1195681661) / 255,
                    288.1221695283 * (t - 60).clamp(min=1e-3) ** -.0755148492 / 255).clamp(0, 1)
    b = torch.where(t >= 66, torch.ones_like(t),
                    torch.where(t <= 19, torch.zeros_like(t), (138.5177312231 * torch.log((t - 10).clamp(min=1e-3)) - 305.0447915307) / 255)).clamp(0, 1)
    c = torch.stack([r, g, b], -1)
    return c / c.mean(-1, keepdim=True)


def batch_fields(cfg: VisualDRCfg | None):
    """Model fields that need a per-world copy."""
    if cfg is None:
        return []
    return ['geom_matid', 'mat_rgba', 'mat_specular', 'mat_shininess', 'mat_emission', 'mat_texid', 'mat_texrepeat',
            'light_type', 'light_pos', 'light_dir', 'light_diffuse', 'light_ambient', 'light_specular',
            'light_castshadow', 'cam_pos', 'cam_quat', 'cam_intrinsic']


def render_context_kwargs(cfg: VisualDRCfg | None):
    if cfg is None:
        return dict(use_shadows=False)
    return dict(use_shadows=cfg.shadow_prob > 0, use_precomputed_rays=not (cfg.focal_jitter or cfg.principal_jitter))


# ----------------------------------------------------------------------------- the per-world sampler
class VisualDR:
    """Nominal look + per-world visual DR on a Warp model (`m`: mujoco_warp Model, `mjm`: CPU model).

    sample(ids): new look for worlds `ids` (call on reset). image(camera, rgb): camera model on a
    rendered float RGB frame (nworld, H, W, 3) in [0, 1] -> uint8.
    """

    def __init__(self, mjm, m, cfg: VisualDRCfg | None, look, camera_names, nworld, device, centre_xy=(.25, .3)):
        import warp as wp
        self.cfg, self.mjm, self.nworld, self.device = cfg, mjm, nworld, torch.device(device)
        self.camera_names = tuple(camera_names)
        gain = (look or {}).get('camera_gain', {})
        self.cam_gain = {c: torch.tensor(gain.get(c, [1., 1., 1.]), device=self.device, dtype=torch.float32)
                         for c in self.camera_names}
        self.nominal = torch.ones(nworld, dtype=torch.bool, device=self.device)
        if cfg is None:
            return
        dev = self.device
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(cfg.seed))
        self.f = {k: wp.to_torch(getattr(m, k)) for k in batch_fields(cfg)}
        self.f0 = {k: torch.as_tensor(getattr(mjm, k).copy(), device=dev).to(v.dtype).reshape(v.shape[1:])
                   for k, v in self.f.items()}
        self.groups = {g: torch.as_tensor(v, device=dev, dtype=torch.long) for g, v in group_geoms(mjm).items() if len(v)}
        self.group_mat = {g: mjm.material('vdr_' + g).id for g in self.groups}
        self.decal = {g: (mjm.geom(DECAL[g]).id, mjm.material(f'vdr_{g}_tex').id) for g in TEXTURED if g in self.groups}
        names = [mjm.texture(i).name for i in range(mjm.ntex)]
        self.tex_proc = torch.tensor([i for i, n in enumerate(names) if n.startswith('vdr_tex_')], device=dev)
        self.tex_real = torch.tensor([i for i, n in enumerate(names) if n.startswith('vdr_real_')], device=dev)
        # nominal colour per group: mean effective colour of its geoms (material or geom rgba)
        self.group_rgb0 = {}
        for g, ids in self.groups.items():
            cols = [mjm.mat_rgba[mjm.geom_matid[i], :3] if mjm.geom_matid[i] >= 0 else mjm.geom_rgba[i, :3] for i in ids.tolist()]
            self.group_rgb0[g] = torch.tensor(np.median(np.array(cols), 0), device=dev, dtype=torch.float32)
        self.key, self.fill = mjm.light('key').id, mjm.light('fill').id
        self.centre = torch.tensor([*centre_xy, float(mjm.geom('table').pos[2] + mjm.geom('table').size[2])], device=dev)
        self.wall_x = float(mjm.geom('bd_wall').pos[0] - mjm.geom('bd_wall').size[0]) if 'bd_wall' in names_geoms(mjm) else 9.
        self.cam_ids = [mjm.camera(c).id for c in self.camera_names]
        self.sensor = torch.tensor(mjm.cam_sensorsize[self.cam_ids], device=dev, dtype=torch.float32)
        # per-world camera-image state (per episode)
        nc = len(self.camera_names)
        self.k1 = torch.zeros(nworld, nc, device=dev)
        self.blur = torch.zeros(nworld, nc, device=dev)
        self.noise = torch.zeros(nworld, nc, device=dev)
        self.img = {k: torch.zeros(nworld, nc, *s, device=dev) for k, s in (('gain', (3,)), ('exposure', ()), ('brightness', ()))}
        self.img['gain'][:] = 1
        self.img['exposure'][:] = 1
        self.state = {}
        self.sample(torch.arange(nworld, device=dev))

    # -------------------------------------------------------------- sampling helpers
    def _u(self, lo, hi, shape):
        return torch.rand(shape, generator=self.gen, device=self.device) * (hi - lo) + lo

    def _pm(self, a, shape):
        return self._u(-a, a, shape)

    def _hsv(self, n):
        c = self.cfg
        h, s, v = self._u(*c.hue_range, n), self._u(*c.saturation_range, n), self._u(*c.value_range, n)
        k = lambda p: (p + h * 6) % 6                                      # noqa: E731
        f = lambda p: v - v * s * torch.clamp(torch.minimum(k(p), 4 - k(p)), 0, 1)   # noqa: E731
        return torch.stack([f(5), f(3), f(1)], -1)

    def sample(self, ids):
        """New look for worlds `ids`: nominal first, then (unless drawn nominal) the randomized fields."""
        if self.cfg is None:
            return
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        for k, v in self.f.items():
            v[ids] = self.f0[k]
        c, n = self.cfg, len(ids)
        nominal = torch.rand(n, generator=self.gen, device=self.device) < c.nominal_prob
        self.nominal[ids] = nominal
        self._sample_image(ids, nominal & c.nominal_clean)
        r = ids[~nominal]
        if len(r):
            self._sample_appearance(r)
            self._sample_lights(r)
            self._sample_cameras(r)

    def _sample_appearance(self, ids):
        c, f, n = self.cfg, self.f, len(ids)
        for g, geoms in self.groups.items():
            mat = self.group_mat[g]
            p = c.tips_random_prob if g == 'tips' else c.random_color_prob
            rnd = (torch.rand(n, generator=self.gen, device=self.device) < p)[:, None]
            jit = self.group_rgb0[g] * self._u(1 - c.nominal_jitter, 1 + c.nominal_jitter, (n, 1)) * \
                self._u(1 - c.nominal_jitter / 4, 1 + c.nominal_jitter / 4, (n, 3))
            rgb = torch.where(rnd, self._hsv(n), jit).clamp(0, 1)
            spec, shin = self._u(*c.specular_range, n), self._u(*c.shininess_range, n)
            emis = self._u(*c.emission_range, n)
            f['geom_matid'][ids[:, None], geoms] = mat
            f['mat_rgba'][ids, mat, :3] = rgb
            f['mat_specular'][ids, mat], f['mat_shininess'][ids, mat], f['mat_emission'][ids, mat] = spec, shin, emis
            if g in self.decal:
                gid, dmat = self.decal[g]
                f['geom_matid'][ids, gid] = dmat
                textured = torch.rand(n, generator=self.gen, device=self.device) < c.texture_prob
                real = torch.rand(n, generator=self.gen, device=self.device) < c.real_texture_prob
                pick = lambda bank: bank[torch.randint(len(bank), (n,), generator=self.gen, device=self.device)]  # noqa: E731
                tex = torch.where(real & (len(self.tex_real) > 0), pick(self.tex_real) if len(self.tex_real) else -1,
                                  pick(self.tex_proc))
                tex = torch.where(textured, tex, torch.full_like(tex, -1))
                # textured: the pattern carries the colour (bright neutral base); else the group colour
                base = torch.where(textured[:, None], self._u(.6, 1., (n, 1)).expand(n, 3) *
                                   self._u(.9, 1., (n, 3)), rgb)
                f['mat_rgba'][ids, dmat, :3] = base.clamp(0, 1)
                f['mat_specular'][ids, dmat], f['mat_shininess'][ids, dmat], f['mat_emission'][ids, dmat] = spec, shin, emis
                f['mat_texid'][ids, dmat, 1] = tex.to(f['mat_texid'].dtype)
                rep = self._u(*c.texrepeat_range, n)
                aspect = 1. if g == 'table' else 3.
                f['mat_texrepeat'][ids, dmat] = torch.stack([rep, rep * aspect], -1)

    def _sample_lights(self, ids):
        c, f, n, dev = self.cfg, self.f, len(ids), self.device
        kel = kelvin_rgb(self._u(*c.light_kelvin, n))
        for li, (el_r, scale) in ((self.key, (c.light_elevation_deg, None)), (self.fill, ((20., 90.), c.fill_scale))):
            el = torch.deg2rad(self._u(*el_r, n))
            az = torch.deg2rad(self._u(*c.light_azimuth_deg, n))
            dist = self._u(*c.light_distance_m, n)
            d = torch.stack([torch.cos(el) * torch.cos(az), torch.cos(el) * torch.sin(az), torch.sin(el)], -1)
            pos = self.centre + d * dist[:, None]
            pos[:, 0] = pos[:, 0].clamp(max=self.wall_x - .05)               # stay in front of the wall
            pos[:, 2] = pos[:, 2].clamp(max=self.centre[2] + 1.6)           # under the room ceiling
            dirn = self.centre - pos
            f['light_type'][ids, li] = int(mujoco.mjtLightType.mjLIGHT_POINT)
            f['light_pos'][ids, li] = pos
            f['light_dir'][ids, li] = dirn / dirn.norm(dim=-1, keepdim=True)
            if scale is None:
                inten = self._u(*c.light_intensity, n) / .65
                f['light_castshadow'][ids, li] = torch.rand(n, generator=self.gen, device=dev) < c.shadow_prob
            else:
                inten = self._u(*scale, n)
                f['light_castshadow'][ids, li] = False
            f['light_diffuse'][ids, li] = self.f0['light_diffuse'][li] * inten[:, None] * kel
            f['light_specular'][ids, li] = self.f0['light_specular'][li] * inten[:, None] * kel
            f['light_ambient'][ids, li] = self.f0['light_ambient'][li] * self._u(*c.ambient_scale, (n, 1)) * kel

    def _sample_cameras(self, ids):
        c, f, n = self.cfg, self.f, len(ids)
        for k, (cam, cid) in enumerate(zip(self.camera_names, self.cam_ids)):
            pos_err = c.cam_pos_mm.get(cam, 0.) * 1e-3 * c.cam_err_scale
            rot_err = math.radians(c.cam_rot_deg.get(cam, 0.)) * c.cam_err_scale
            u = torch.randn(n, 3, generator=self.gen, device=self.device)
            f['cam_pos'][ids, cid] = self.f0['cam_pos'][cid] + u / u.norm(dim=-1, keepdim=True) * self._u(0, pos_err, (n, 1))
            a = torch.randn(n, 3, generator=self.gen, device=self.device)
            a = a / a.norm(dim=-1, keepdim=True)
            ang = self._u(0, rot_err, (n, 1))
            dq = torch.cat([torch.cos(ang / 2), a * torch.sin(ang / 2)], -1)
            f['cam_quat'][ids, cid] = _qmul(self.f0['cam_quat'][cid].expand(n, 4), dq)
            intr = self.f0['cam_intrinsic'][cid].expand(n, 4).clone()
            intr[:, :2] *= 1 + self._pm(c.focal_jitter, (n, 2))
            intr[:, 2:] += self._pm(c.principal_jitter, (n, 2)) * self.sensor[k]
            f['cam_intrinsic'][ids, cid] = intr
            self.k1[ids, k] = self._pm(c.radial_k1, n)

    def _sample_image(self, ids, clean):
        c, n, nc = self.cfg, len(ids), len(self.camera_names)
        self.blur[ids] = torch.where(clean[:, None], 0., self._u(*c.blur_sigma_px, (n, nc)))
        self.noise[ids] = torch.where(clean[:, None], 0., self._u(*c.noise_std, (n, nc)))
        self.k1[ids] = 0
        self.clean = getattr(self, 'clean', torch.zeros(self.nworld, dtype=torch.bool, device=self.device))
        self.clean[ids] = clean
        self._draw_photometric(ids)

    def _draw_photometric(self, ids):
        c, n, nc = self.cfg, len(ids), len(self.camera_names)
        self.img['gain'][ids] = self._u(*c.gain, (n, nc, 3))
        self.img['exposure'][ids] = 1 + self._pm(c.exposure, (n, nc))
        self.img['brightness'][ids] = self._pm(c.brightness, (n, nc))
        cl = self.clean[ids]
        self.img['gain'][ids[cl]] = 1
        self.img['exposure'][ids[cl]] = 1
        self.img['brightness'][ids[cl]] = 0

    # -------------------------------------------------------------- camera model on rendered frames
    def image(self, camera, rgb, new_frame=True):
        """Float RGB (nworld, H, W, 3) in [0, 1] from the renderer -> uint8 camera image."""
        k = self.camera_names.index(camera)
        x = rgb * self.cam_gain[camera]
        if self.cfg is None:
            return (x.clamp(0, 1) * 255).to(torch.uint8)
        if new_frame and self.cfg.image_per_frame and k == 0:
            self._draw_photometric(torch.arange(self.nworld, device=self.device))
        out = torch.empty(x.shape, dtype=torch.uint8, device=x.device)
        for s in range(0, self.nworld, 512):                    # bounded temporaries at 2048 worlds
            sl = slice(s, s + 512)
            out[sl] = self._camera_model(x[sl], k, sl)
        return out

    def _camera_model(self, x, k, sl):
        n, H, W, _ = x.shape
        y = x.permute(0, 3, 1, 2)                                # (n, 3, H, W)
        k1 = self.k1[sl, k]
        if bool((k1 != 0).any()):
            v, u = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing='ij')
            r2 = (u * u + v * v)[None]
            s = 1 + k1[:, None, None] * r2                       # sample the ideal image at the distorted radius
            grid = torch.stack([u[None] * s, v[None] * s], -1)
            y = torch.nn.functional.grid_sample(y, grid, mode='bilinear', padding_mode='border', align_corners=True)
        sig = self.blur[sl, k]
        if bool((sig > .05).any()):
            t = torch.arange(-2, 3, device=x.device, dtype=y.dtype)
            ker = torch.exp(-t[None] ** 2 / (2 * sig.clamp(min=1e-3)[:, None] ** 2))
            ker = torch.where((sig > .05)[:, None], ker, (t == 0).to(y.dtype)[None])
            ker = (ker / ker.sum(1, keepdim=True)).repeat_interleave(3, 0)          # (n*3, 5)
            z = y.reshape(1, n * 3, H, W)
            z = torch.nn.functional.conv2d(torch.nn.functional.pad(z, (2, 2, 0, 0), mode='replicate'), ker[:, None, None, :], groups=n * 3)
            z = torch.nn.functional.conv2d(torch.nn.functional.pad(z, (0, 0, 2, 2), mode='replicate'), ker[:, None, :, None], groups=n * 3)
            y = z.reshape(n, 3, H, W)
        y = y.permute(0, 2, 3, 1) * 255
        y = y * (self.img['exposure'][sl, k, None, None, None] * self.img['gain'][sl, k, None, None, :]) + \
            self.img['brightness'][sl, k, None, None, None]
        noise = self.noise[sl, k]
        y = y + torch.randn(y.shape, generator=self.gen, device=y.device) * noise[:, None, None, None]
        return y.clamp(0, 255).to(torch.uint8)


def names_geoms(m):
    return {m.geom(i).name for i in range(m.ngeom)}


def _qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
