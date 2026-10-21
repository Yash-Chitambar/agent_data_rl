"""Labelled success / failure demonstration videos from `eval.py --demo_k K` rollouts.

The eval saves the rollouts of K successes (fastest, median, slowest, ...) and up to K episodes of
each failure type, plus its own arguments (`meta`), so this script rebuilds the same model on the
CPU, replays the joint states and renders a grid video: one tile per episode, captioned with the
outcome (green: success and time to seat; red: failure type and time). Qualitative only.

python -m blind_insertion.viz.demos --rollouts results/E/x.rollouts.npz --out results/demos/E_x \
    --title "USB-C final stage" [--k_success 3 --k_fail 2 --ncol 4]
-> <out>.mp4 (grid), <out>_sheet.png (last frames), <out>.json (episodes shown)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.font_manager as fm
import mujoco
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from blind_insertion.viz.render import save_video

# camera per family: (distance m or None = from part size, azimuth, elevation, look-at z offset)
CAMERAS = {"peg": (None, 150, -20, 0.0), "usbc": (0.09, 120, -12, 0.0), "gear": (0.22, 120, -25, 0.01),
           "usbc_side": (0.12, 90, -10, 0.0)}
GREEN, RED, GREY = (34, 136, 51), (204, 51, 17), (60, 60, 60)


def rebuild(meta: dict):
    """(model, layout, family, variant-per-world fn) for the eval arguments in `meta`."""
    import argparse as ap_  # noqa: PLC0415
    a = ap_.Namespace(**meta)
    if meta.get("task") == "gear":
        from blind_insertion.assets import gear_builder as GB  # noqa: PLC0415
        fam = a.family or (f"ood:{a.ood}" if a.ood else "study")
        vs = (GB.sample_study_gears(a.n_variants, a.variant_seed) if fam == "study"
              else GB.sample_ood_gears(a.n_variants, a.variant_seed, fam[4:]))
        cfg = json.load(open(a.env_file)) if a.env_file else {}
        m, lay = GB.build(cfg.get("robot", a.robot), vs, gains=cfg.get("gains"))
        return m, lay, "gear", GB.apply_variant_cpu, len(vs)
    from blind_insertion.assets import builder as B  # noqa: PLC0415
    from blind_insertion.envs.peg_env import EnvCfg  # noqa: PLC0415
    from blind_insertion.eval import env_kwargs  # noqa: PLC0415
    kw = env_kwargs(a, 1)
    cfg = EnvCfg(**{k: v for k, v in kw.items() if k in EnvCfg.__dataclass_fields__})
    vs = cfg.variants or (cfg.variant_sampler or B.sample_study_pegs)(cfg.n_variants, cfg.variant_seed)
    nr = cfg.noise_rand or {}
    bl = max([cfg.backlash_arcmin, *nr.get("backlash_arcmin", [])])
    m, lay = B.build(cfg.robot, vs, gains=cfg.gains, backlash_arcmin=bl, pad_spheres=cfg.pad_spheres,
                     table_top=cfg.table_top)
    fam = "usbc" if getattr(a, "family", "") == "usbc" else "peg"
    if fam == "usbc" and any(cfg.fix_base_rpy_deg):
        fam = "usbc_side"
    return m, lay, fam, B.apply_variant_cpu, len(vs)


def render_world(m, lay, apply_fn, k, roll, cam_spec, size, part_size):
    apply_fn(m, lay, k)
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, size[1], size[0])
    cam = mujoco.MjvCamera()
    dist, az, el, dz = cam_spec
    frames = []
    for t in range(len(roll["qpos"])):
        d.qpos[:] = roll["qpos"][t]
        d.mocap_pos[0] = roll["mocap_pos"][t]
        d.mocap_quat[0] = roll["mocap_quat"][t]
        mujoco.mj_forward(m, d)
        cam.lookat[:] = roll["mocap_pos"][t] + np.array([0, 0, dz])
        cam.distance = dist or max(0.25, 2.2 * part_size)
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam)
        frames.append(r.render().copy())
    r.close()
    return frames


def caption(img, text, color, font):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    w = im.width
    dr.rectangle([0, 0, w, 26], fill=color)
    dr.text((8, 4), text, fill=(255, 255, 255), font=font)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--parquet", default="", help="default: <rollouts stem>.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--k_success", type=int, default=3)
    ap.add_argument("--k_fail", type=int, default=2)
    ap.add_argument("--max_fail_types", type=int, default=5)
    ap.add_argument("--ncol", type=int, default=4)
    ap.add_argument("--size", default="360x270")
    ap.add_argument("--camera", default="", help="override: distance,azimuth,elevation")
    a = ap.parse_args()
    z = np.load(a.rollouts)
    meta = json.loads(str(z["meta"]))
    pq = Path(a.parquet or a.rollouts.replace(".rollouts.npz", ".parquet"))
    df = pd.read_parquet(pq)
    worlds = sorted({int(k.split("_")[0][1:]) for k in z.files if k.startswith("w")})
    have = df[df.world.isin(worlds) & df.world.isin([w for w in worlds if f"w{w}_qpos" in z.files])]
    ok = have[have.success > 0].sort_values("time").head(a.k_success)
    fails = have[have.success == 0]
    types = fails.failure.value_counts().index[:a.max_fail_types]
    bad = pd.concat([fails[fails.failure == t].head(a.k_fail) for t in types]) if len(types) else fails.head(0)
    show = pd.concat([ok, bad])
    m, lay, fam, apply_fn, nK = rebuild(meta)
    cam = CAMERAS[fam]
    if a.camera:
        dist, az, el = (float(x) for x in a.camera.split(","))
        cam = (dist, az, el, cam[3])
    size = tuple(int(x) for x in a.size.split("x"))
    font = ImageFont.truetype(fm.findfont("DejaVu Sans"), 15)
    tiles, shown = [], []
    for _, row in show.iterrows():
        w = int(row.world)
        roll = {k.split("_", 1)[1]: z[k] for k in z.files if k.startswith(f"w{w}_")}
        k = w % nK
        part = getattr(lay.variants[k], "plate_size", 0.1) if hasattr(lay, "variants") else 0.1
        fr = render_world(m, lay, apply_fn, k, roll, cam, size, part)
        if row.success > 0:
            txt, col = f"success  {row.time:.2f} s", GREEN
        else:
            txt, col = f"{row.failure.replace('_', ' ')}  {row.time:.1f} s", RED
        if "start_dist" in row:
            txt += f"  start {1000 * row.start_dist:.1f} mm"
        tiles.append([caption(f, txt, col, font) for f in fr])
        shown.append(dict(world=w, success=bool(row.success > 0), failure=row.failure, time=float(row.time)))
        print(f"rendered world {w}: {txt}", flush=True)
    if not tiles:
        raise SystemExit("no episodes to show")
    T = max(len(t) for t in tiles)
    tiles = [t + [t[-1]] * (T + 25 - len(t)) for t in tiles]     # hold the last frame
    ncol = min(a.ncol, len(tiles))
    nrow = int(np.ceil(len(tiles) / ncol))
    H, W = size[1], size[0]
    head = 40 if a.title else 0
    blank = np.full((H, W, 3), 255, np.uint8)
    tfont = ImageFont.truetype(fm.findfont("DejaVu Sans"), 20)
    frames = []
    for t in range(T + 25):
        grid = np.full((head + nrow * H, ncol * W, 3), 255, np.uint8)
        for i in range(nrow * ncol):
            r_, c_ = divmod(i, ncol)
            grid[head + r_ * H:head + (r_ + 1) * H, c_ * W:(c_ + 1) * W] = tiles[i][t] if i < len(tiles) else blank
        if head:
            im = Image.fromarray(grid)
            ImageDraw.Draw(im).text((10, 9), f"{a.title}   (t = {min(t, T - 1) * 0.02:.2f} s)", fill=GREY,
                                    font=tfont)
            grid = np.asarray(im)
        frames.append(grid)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_video(frames, str(out) + ".mp4", fps=50)
    Image.fromarray(frames[-1]).save(str(out) + "_sheet.png")
    json.dump(dict(title=a.title, source=str(pq), n=int(len(df)), success_rate=float(df.success.mean()),
                   failures=df[df.success == 0].failure.value_counts().to_dict(), shown=shown),
              open(str(out) + ".json", "w"), indent=1)
    print("wrote", str(out) + ".mp4")


if __name__ == "__main__":
    main()
