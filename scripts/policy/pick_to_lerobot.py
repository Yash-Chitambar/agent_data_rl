"""Convert Part 1 pick demos (scripts/policy/gen_pick_demos.py) into a LeRobot dataset. LeRobot venv.

Features: observation.images.top / .wrist (square RGB video, ZED top + right wrist), observation.state (7: joints,
finger opening), action (8: joint targets, gripper command, done). Episodes where the expert
failed are skipped.

    .venv-act/bin/python scripts/act/to_lerobot.py --raw data/act/pick_raw_v1 --root data/act/pick_v1
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

TASK = "pick up the SD card reader with the side grasp and lift it"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/act/pick_raw_v1")
    ap.add_argument("--root", default="data/act/pick_v1")
    ap.add_argument("--repo_id", default="local/pick_v1")
    ap.add_argument("--max_episodes", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1, help="convert every nshards-th episode (run shards in parallel, then pick_merge_shards.py)")
    ap.add_argument("--encoder_threads", type=int, default=4)
    ap.add_argument("--cams", nargs="+", default=["top", "wrist"], help="cameras to keep (e.g. wrist only)")
    ap.add_argument("--grip_from_phase", type=int, default=None,
                    help="relabel the gripper action: close (-1) from the first step in this expert phase on (2 = settle: "
                         "close as soon as the hand arrives at the grasp pose, not after the invisible settle wait)")
    a = ap.parse_args()
    raw, root = Path(a.raw), Path(a.root)
    if root.exists():
        shutil.rmtree(root)
    eps = sorted(raw.glob("ep*.npz"))
    if a.max_episodes:
        eps = eps[:a.max_episodes]
    eps = eps[a.shard::a.nshards]
    z0 = np.load(eps[0])
    h, w = iio.imread(raw / f"{eps[0].stem}_top.mp4", index=0).shape[:2]
    names_q = [f"joint{i}" for i in range(1, 7)]
    features = {
        **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channel"]} for c in a.cams},
        "observation.state": {"dtype": "float32", "shape": (z0["state"].shape[1],), "names": names_q + ["finger"]},
        "action": {"dtype": "float32", "shape": (z0["action"].shape[1],), "names": names_q + ["gripper", "done"]},
    }
    ds = LeRobotDataset.create(a.repo_id, fps=50, features=features, root=root, robot_type="yam",
                               use_videos=True, streaming_encoding=True, encoder_threads=a.encoder_threads,
                               rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=20, g=2, preset="fast"))
    kept = 0
    for f in eps:
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        if not meta["ok"]:
            continue
        vids = {c: iio.imread(raw / f"{f.stem}_{c}.mp4") for c in a.cams}
        act = z["action"].copy()
        if a.grip_from_phase is not None:
            hit = np.nonzero(z["phase"] >= a.grip_from_phase)[0]
            if len(hit):
                act[hit[0]:, 6] = -1.
        for t in range(len(z["state"])):
            ds.add_frame({**{f"observation.images.{c}": vids[c][t] for c in a.cams},
                          "observation.state": z["state"][t], "action": act[t], "task": TASK})
        ds.save_episode()
        kept += 1
        if kept % 50 == 0:
            print(f"{kept} episodes", flush=True)
    ds.finalize()
    print(f"done: {kept} of {len(eps)} episodes -> {root}")


if __name__ == "__main__":
    main()
