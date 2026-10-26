"""Share the arms between the rig console (rigctl.py) and the scripts that drive them.

A script that connects an arm (safety.connect) calls `claim(side)` first. If the console holds
that arm, it parks it at the folded rest, lets it go limp there and removes its <side>.held file;
`claim` waits for that. The claim lasts as long as the script's process: the console treats a
claim of a dead process as free. Without a console running, `claim` returns at once.

State in /dev/shm/yam_arms/: <side>.claim (pid and script), <side>.held (pid of the console).
"""
import os
import sys
import time
from pathlib import Path

from camshare import read_pid

DIR = Path("/dev/shm/yam_arms")
SIDES = ("right", "left")


def claim(side, who=None, timeout=60.0):
    """Take the arm from the console (it homes the arm first: up to ~30 s)."""
    assert side in SIDES, side
    DIR.mkdir(exist_ok=True)
    (DIR / f"{side}.claim").write_text(f"{os.getpid()} {who or Path(sys.argv[0]).name}")
    held = DIR / f"{side}.held"
    t0, told = time.time(), False
    while read_pid(held)[0] not in (None, os.getpid()):
        if not told:
            print(f"{side} arm: waiting for rigctl to park it at rest and let go", flush=True)
            told = True
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{side} arm: rigctl did not release it in {timeout:.0f} s")
        time.sleep(0.1)


def owner(side):
    """(pid, script) of a live process that claimed the arm, else (None, None)."""
    return read_pid(DIR / f"{side}.claim")
