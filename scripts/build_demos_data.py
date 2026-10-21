"""Report data for the success / failure examples (figure `demos`): copies the grid videos
made by scripts/make_demos.sh into report/static/videos/demos/ and writes static/data/demos.json.

python scripts/build_demos_data.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "demos"
REP = ROOT / "report" / "static"

# group, id, button label, title, explanation (ASD-STE100: short sentences, one idea each)
CASES = [
    ("Peg, Flexiv", "B_blind_indist", "Blind", "Blind policy, unseen pegs of the training family",
     "The policy slides the peg on the plate until it feels the hole, then it drops the peg in. "
     "The rare failures do not find the hole in 5 s or push the peg off the plate."),
    ("Peg, Flexiv", "B_visual_indist", "Wrist RGB", "Wrist-camera policy, unseen pegs of the training family",
     "The visual policy moves almost directly to the hole. It is faster than the blind policy on familiar parts."),
    ("Peg, Flexiv", "B_visual_tighter", "RGB, tighter fit", "Wrist-camera policy, tighter clearance (held out)",
     "With 0.5 to 0.6 mm clearance, the visual policy often stops partly inside the hole. "
     "The blind policy seats 97% of the same parts."),
    ("Peg, Flexiv", "B_blind_extreme_size", "Blind, extreme size", "Blind policy, extreme part sizes (held out)",
     "Very small and very large parts. Most failures leave the plate or do not find the hole."),
    ("Peg, Flexiv", "B_blind_random_convex", "Blind, random convex", "Blind policy, random convex shapes (held out)",
     "These shapes have corners that the training pegs do not have. The peg often slips in the grasp or "
     "stops partly inside the hole."),
    ("Peg, Flexiv", "B_spiral", "Spiral search", "Scripted spiral search with the true hole position",
     "The spiral covers the area around the hole in circles. It moves much farther than the learned policies."),
    ("Gear", "G_gear_indist", "Unseen gear sets", "Blind gear policy, unseen gear sets",
     "The policy slides the gear down the shaft and turns it until the teeth mesh with the anchor gears."),
    ("Gear", "G_gear_random_tooth", "Random tooth shape", "Blind gear policy, random tooth shapes (held out)",
     "The tooth shapes are different from training. The policy meshes them as well."),
    ("YAM robot errors", "C_yam_latency1", "1-step latency", "YAM peg policy with 20 ms of latency it did not train with",
     "Old mount (fixture below the base); the lab-table runs show the same result. The policy overshoots "
     "and pushes the peg off the plate or past the hole."),
    ("YAM robot errors", "C_yam_backlash1", "1 arcmin backlash", "YAM peg policy with backlash it did not train with",
     "Old mount. With a small dead zone in each joint, the policy often does not find the hole."),
    ("USB-C", "E_usbc_final", "Final stage", "USB-C, final stage: 0.02 to 0.1 mm clearance, 3 mm median start",
     "Lab-table mount. The policy presses the plug on the housing and searches in small steps. "
     "Most failures start far from the port and do not find it in 8 s."),
    ("USB-C", "E_usbc_far", "Far starts", "USB-C, starts up to 9 mm from the port, 0.05 mm clearance",
     "The capture radius is approximately 6 mm. Starts farther away usually time out."),
    ("USB-C", "E_usbc_latency1", "1-step latency", "USB-C policy with 20 ms of latency it did not train with",
     "Success decreases from 78% to 5%. The search becomes unstable and the drive slips or leaves the housing."),
    ("USB-C", "E_usbc_noiserandft_latency1", "Latency, after fine-tune",
     "USB-C policy after fine-tuning with random latency, backlash and noise, tested with 1-step latency", ""),
    ("USB-C", "E_usbc_wide_far", "Far starts, after fine-tune",
     "USB-C policy after fine-tuning on wider starts, starts up to 9 mm", ""),
    ("USB-C variants", "E_usbc_chamfer", "Chamfered port", "USB-C, port with a 0.2 to 0.5 mm lead-in (not in training)",
     "The lead-in guides the plug. The policy that never saw a chamfer seats more drives than without it."),
    ("USB-C variants", "E_usbc_latch", "Latch", "USB-C, latch spring of 5 to 15 N over the last 1.5 mm (not in training)",
     "The policy does not push hard enough. The drive slips in the grasp or stays partly inside."),
    ("USB-C variants", "E_usbc_latchft", "Latch, after fine-tune", "USB-C policy after fine-tuning with the latch", ""),
    ("USB-C variants", "E_usbc_side", "Side port", "USB-C, port facing the robot (port-up policy, not trained for it)",
     "The arm pose is different, so the port-up policy moves the plug off the housing. This task needs its own training."),
]


def main():
    vdir, idir = REP / "videos" / "demos", REP / "images" / "demos"
    vdir.mkdir(parents=True, exist_ok=True)
    idir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = {}
    texts = json.load(open(SRC / "texts.json")) if (SRC / "texts.json").exists() else {}
    for g, cid, label, title, text in CASES:
        j = SRC / f"{cid}.json"
        if not (j.exists() and (SRC / f"{cid}.mp4").exists()):
            continue
        meta = json.load(open(j))
        shutil.copy(SRC / f"{cid}.mp4", vdir / f"{cid}.mp4")
        shutil.copy(SRC / f"{cid}_sheet.png", idir / f"{cid}.png")
        groups.setdefault(g, []).append(dict(
            id=cid, label=label, title=title, text=texts.get(cid, text), video=f"videos/demos/{cid}.mp4",
            poster=f"images/demos/{cid}.png", n=meta["n"], rate=meta["success_rate"], failures=meta["failures"]))
    out = dict(groups=[dict(name=g, items=v) for g, v in groups.items()])
    json.dump(out, open(REP / "data" / "demos.json", "w"), indent=1)
    print({g: len(v) for g, v in groups.items()})


if __name__ == "__main__":
    main()
