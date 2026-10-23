"""Headless check: generate an mjfit assembly and step it in RoboSimStudio."""

from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import mjfit
import mujoco
import numpy as np
import robosimstudio as rss


def main():
    for package in ("mjfit", "robosimstudio", "mujoco", "mujoco-warp", "newton", "numpy"):
        print(f"{package}: {version(package)}", flush=True)

    xml = mjfit.peg(hole=mjfit.shapes.circle(0.008), standalone=False)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()

    with TemporaryDirectory(prefix="peg-test-") as directory:
        path = Path(directory) / "assembly.xml"
        path.write_text(xml)
        scene = rss.Scene(cell="void", physics=rss.MUJOCO_500HZ)
        scene.add(rss.StaticSpec(name="assembly", asset=rss.AssetSpec(path=str(path))))
        sim = scene.build(device="cpu")
        try:
            sim.run(10).sync()
            positions = sim.state.body_q.numpy()
            assert positions.size and np.isfinite(positions).all()
        finally:
            sim.close()
    print("PASS: mjfit assembly compiled and stepped in MuJoCo and RoboSimStudio")


if __name__ == "__main__":
    main()
