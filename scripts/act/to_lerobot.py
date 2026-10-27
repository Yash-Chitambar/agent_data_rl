"""Convert task H pick demos (scripts/act/gen_demos.py) into a LeRobot dataset. LeRobot venv.

Features: observation.images.top / .wrist (240x320 RGB video), observation.state (7: joints,
finger opening), action (8: joint targets, gripper command, done). Episodes where the expert
failed are skipped.

    .venv-act/bin/python scripts/act/to_lerobot.py --raw data/act/raw_v1 [more raw dirs] --root data/act/reader_pick_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "pick up the SD card reader and line its USB-C plug up with the PC's port"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+", default=["data/act/raw_v1"])
    ap.add_argument("--root", default="data/act/reader_pick_v1")
    ap.add_argument("--repo_id", default="local/reader_pick_v1")
    ap.add_argument("--max_episodes", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.root)
    if root.exists():
        shutil.rmtree(root)
    eps = [f for r in a.raw for f in sorted(Path(r).glob("ep*.npz"))]
    if a.max_episodes:
        eps = eps[:a.max_episodes]
    z0 = np.load(eps[0])
    h, w = iio.imread(eps[0].parent / f"{eps[0].stem}_top.mp4", index=0).shape[:2]
    names_q = [f"joint{i}" for i in range(1, 7)]
    features = {
        "observation.images.top": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.images.wrist": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
        "observation.state": {"dtype": "float32", "shape": (z0["state"].shape[1],),
                              "names": (names_q + ["finger"] + [f"cmd_err{i}" for i in range(1, 7)])[:z0["state"].shape[1]]},
        "action": {"dtype": "float32", "shape": (z0["action"].shape[1],), "names": names_q + ["gripper", "done"]},
    }
    ds = LeRobotDataset.create(a.repo_id, fps=50, features=features, root=root, robot_type="yam",
                               use_videos=True, streaming_encoding=True, encoder_threads=4,
                               rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=20, g=2, preset="fast"))
    kept = 0
    for f in eps:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        if not meta["ok"]:
            continue
        top = iio.imread(f.parent / f"{f.stem}_top.mp4")
        wrist = iio.imread(f.parent / f"{f.stem}_wrist.mp4")
        for t in range(len(z["state"])):
            ds.add_frame({"observation.images.top": top[t], "observation.images.wrist": wrist[t],
                          "observation.state": z["state"][t], "action": z["action"][t], "task": TASK})
        ds.save_episode()
        kept += 1
        if kept % 50 == 0:
            print(f"{kept} episodes", flush=True)
    ds.finalize()
    print(f"done: {kept} of {len(eps)} episodes -> {root}")


if __name__ == "__main__":
    main()
