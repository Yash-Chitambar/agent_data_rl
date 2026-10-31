"""Stage the Task H models into deploy/models/ (gitignored) and record their hashes in
deploy/models/MANIFEST.json (tracked), so the real rig runs exactly the evaluated files.

    .venv/bin/python -m deploy.models stage --act runs/H-act-pick-v5/act_80k_traced.pt --insert runs/H-reader-tilt-uncomp-s1/last.pt
    .venv/bin/python -m deploy.models verify          # on the rig, after copying deploy/models/
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent / "models"
NAMES = {"act": "act_pick.pt", "insert": "insert.pt", "insert_config": "insert.config.json"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stage(a):
    src = {"act": Path(a.act), "insert": Path(a.insert), "insert_config": Path(a.insert).parent / "config.json"}
    man = {"staged": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "files": {}}
    for k, p in src.items():
        shutil.copy2(p, HERE / NAMES[k])
        man["files"][NAMES[k]] = {"source": str(p.resolve()), "sha256": sha256(HERE / NAMES[k])}
    man["note"] = a.note
    (HERE / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))


def verify(a):
    man = json.loads((HERE / "MANIFEST.json").read_text())
    bad = [n for n, f in man["files"].items() if not (HERE / n).exists() or sha256(HERE / n) != f["sha256"]]
    print("models OK" if not bad else f"models MISSING or CHANGED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--act", required=True)
    s.add_argument("--insert", required=True)
    s.add_argument("--note", default="")
    sub.add_parser("verify")
    a = ap.parse_args()
    raise SystemExit({"stage": stage, "verify": verify}[a.cmd](a) or 0)
