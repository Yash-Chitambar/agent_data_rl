"""Server verification: imports, versions, GPU visible to torch and Warp, EGL render, MuJoCo Warp step.

Run from the repo root:  .venv/bin/python scripts/server/verify.py [--no-egl] [--json out.json]
Exit code 0 only if every required check passes (EGL is required unless --no-egl).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from importlib.metadata import version

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

EXPECTED = {"mujoco": "3.12.0", "mujoco-warp": "3.12.0", "warp-lang": "1.18"}
results: dict[str, dict] = {}


def check(name, required=True):
    def deco(fn):
        t = time.time()
        try:
            info = fn() or {}
            results[name] = {"ok": True, "required": required, "s": round(time.time() - t, 1), **info}
            print(f"[ OK ] {name} {info if info else ''}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[name] = {"ok": False, "required": required, "error": f"{type(e).__name__}: {e}"}
            print(f"[{'FAIL' if required else 'WARN'}] {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=2)
        return fn
    return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-egl", action="store_true", help="EGL failure is a warning, not an error")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    @check("versions")
    def _():
        v = {p: version(p) for p in ("mujoco", "mujoco-warp", "warp-lang", "torch", "mjfit",
                                     "robosimstudio", "numpy", "pyarrow", "h5py")}
        bad = {p: v[p] for p, e in EXPECTED.items() if not v[p].startswith(e)}
        if bad:
            raise RuntimeError(f"unexpected versions {bad} (expected {EXPECTED})")
        return v

    @check("torch_cuda")
    def _():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False (driver too old for cu130? "
                               "CUDA_VISIBLE_DEVICES empty?)")
        n = torch.cuda.device_count()
        cap = torch.cuda.get_device_capability(0)
        x = torch.randn(1024, 1024, device="cuda")
        (x @ x).sum().item()
        return {"torch_cuda": torch.version.cuda, "n_gpu": n, "gpu0": torch.cuda.get_device_name(0),
                "capability": f"sm_{cap[0]}{cap[1]}",
                "mem_GB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}

    @check("warp_cuda")
    def _():
        import warp as wp
        wp.config.quiet = True
        wp.init()
        devs = [str(d) for d in wp.get_cuda_devices()]
        if not devs:
            raise RuntimeError("Warp sees no CUDA device")
        return {"warp_devices": devs, "kernel_cache": wp.config.kernel_cache_dir,
                "cuda_driver": wp.get_cuda_driver_version(), "warp_toolkit": wp.get_cuda_toolkit_version()}

    @check("blind_insertion_imports")
    def _():
        import blind_insertion.envs.peg_env  # noqa: F401
        import blind_insertion.eval  # noqa: F401
        import blind_insertion.train  # noqa: F401
        import blind_insertion.train_visual  # noqa: F401
        from blind_insertion.assets import builder as B
        if not B.MENAGERIE.joinpath("flexiv_rizon4s").is_dir() or not B.MENAGERIE.joinpath("i2rt_yam").is_dir():
            raise RuntimeError(f"Menagerie robots missing under {B.MENAGERIE}")
        return {"menagerie": str(B.MENAGERIE)}

    @check("egl_render", required=not args.no_egl)
    def _():
        import mujoco
        import numpy as np
        m = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><light pos='0 0 1'/><geom type='box' size='.1 .1 .1' rgba='1 0 0 1'/>"
            "<camera name='c' pos='0 -0.5 0.2' xyaxes='1 0 0 0 0.4 1'/></worldbody></mujoco>")
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, 64, 64)
        try:
            r.update_scene(d, camera="c")
            img = r.render()
        finally:
            r.close()
        if img.shape != (64, 64, 3) or int(img[..., 0].max()) == 0:
            raise RuntimeError("EGL render returned an empty image")
        vendor = None
        try:
            from OpenGL import GL
            vendor = GL.glGetString(GL.GL_VENDOR)
        except Exception:  # noqa: BLE001
            pass
        return {"MUJOCO_GL": os.environ.get("MUJOCO_GL"), "mean_px": float(np.mean(img)),
                "gl_vendor": vendor.decode() if isinstance(vendor, bytes) else vendor}

    @check("mujoco_warp_step")
    def _():
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        m = mujoco.MjModel.from_xml_string(
            "<mujoco><option timestep='0.001'/><worldbody><geom type='plane' size='1 1 .1'/>"
            "<body pos='0 0 .2'><freejoint/><geom type='box' size='.02 .02 .02'/></body>"
            "</worldbody></mujoco>")
        mw = mjw.put_model(m)
        dw = mjw.make_data(m, nworld=64)
        for _ in range(10):
            mjw.step(mw, dw)
        wp.synchronize()
        z = wp.to_torch(dw.qpos)[:, 2]
        if not bool(z.isfinite().all()):
            raise RuntimeError("non-finite qpos")
        return {"z_mean": round(float(z.mean()), 4)}

    failed = [k for k, v in results.items() if not v["ok"] and v["required"]]
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1, default=str)
    print("VERIFY", "FAILED: " + ", ".join(failed) if failed else "PASSED", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
