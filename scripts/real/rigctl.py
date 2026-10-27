"""Rig console for duchamp3: software stop, arms on/off, camera viewer, in one terminal.

    .venv/bin/python scripts/real/rigctl.py

Keys (no Enter needed):
  space      STOP: every arm holds its pose, any motion stops (the same stop file as estop.py)
  c          clear the stop
  o / f      both arms on / off        r / l   right / left arm on <-> off
  h          home: both powered arms move slowly to the folded rest and stay on
  v          camera viewer on / off (tmux session "camview", scripts/real/camview.py --share)
  q          quit: parks and turns off the powered arms first (an arm that cannot park keeps the
             console open; Ctrl-C does the same)

On = connected through safety.connect: PD hold at the measured pose (the gripper calibrates, it
opens and closes once). Off = move slowly to the folded rest, then stop commanding: the motors go
limp there, where the arm rests on its stops. An arm that cannot reach rest (a stop, a safety
check, a tracking error) stays on and holds.

Scripts that drive an arm (deploy.py, calib.py, snapshot.py) take it over: safety.connect claims
the arm (armshare.py), the console parks it at rest and lets go, and shows the script as its user.
"""
import curses
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import armshare  # noqa: E402
import calib  # noqa: E402
import camshare  # noqa: E402
import estop  # noqa: E402
import safety  # noqa: E402

REST = np.zeros(6)          # folded: joints 2-3 on their stops (measured rest 0.001 rad)
AT_REST = 0.05              # rad: joints 2-3 within this of rest -> safe to go limp
CAMVIEW = f"cd {HERE.parents[1]} && while true; do .venv/bin/python scripts/real/camview.py --share; sleep 5; done"


class Arm(threading.Thread):
    """Owns one arm's connection. Commands run one at a time on this thread."""

    def __init__(self, side, kin):
        super().__init__(daemon=True)
        self.side, self.kin = side, kin
        self.mover, self.q = None, None
        self.state, self.msg = "off", ""
        self.jobs = queue.Queue()
        self.held = armshare.DIR / f"{side}.held"

    # ---- called from the UI thread
    def ask(self, job):
        self.jobs.put(job)

    # ---- worker
    def run(self):
        while True:
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                job = None
            try:
                if job:
                    getattr(self, job)()
                self.poll()
            except Exception as e:   # noqa: BLE001
                self.msg = f"{type(e).__name__}: {e}"
                if self.mover is not None:
                    self.hold()

    def poll(self):
        if self.mover is None:
            return
        self.q = np.asarray(self.mover.robot.get_joint_pos(), float)
        pid, who = armshare.owner(self.side)
        if pid is not None and pid != _PID:   # a script wants the arm
            self.msg = f"{who} (pid {pid}) wants the arm: parking it at rest"
            self.off()

    def on(self):
        if self.mover is not None:
            return
        pid, who = armshare.owner(self.side)
        if pid is not None and pid != _PID:
            self.msg = f"in use by {who} (pid {pid})"
            return
        self.state, self.msg = "connecting", "connecting (the gripper calibrates)"
        armshare.DIR.mkdir(exist_ok=True)
        self.held.write_text(f"{_PID} rigctl")
        robot = None
        try:
            robot = safety.connect(self.side)
            self.mover = make_mover(self.side, self.kin, robot)
        except Exception:
            if robot is not None:   # e.g. the measured pose fails the checks: it was resting there
                safety.close_robot(robot)
            self.held.unlink(missing_ok=True)
            self.state = "off"
            raise
        self.state, self.msg = "on", "holding"

    def hold(self):
        try:
            self.mover.hold()
        except Exception:   # noqa: BLE001
            pass
        self.state = "on"

    def home(self):
        if self.mover is None:
            return
        if estop.stopped():
            self.msg = "stop is set: clear it first"
            return
        self.state, self.msg = "moving", "moving to rest"
        try:
            self.mover.arm.check = self.mover.recover
            self.mover.move_to(REST, max_vel=0.2, home=True)
            self.msg = "at rest, holding"
        except safety.SafetyStop as e:
            self.hold()
            self.msg = f"stopped: {e}"
        finally:
            self.mover.arm.check = self.mover.monitor
            if self.mover is not None:
                self.state = "on"

    def off(self):
        if self.mover is None:
            return
        self.home()
        q = np.asarray(self.mover.robot.get_joint_pos(), float)
        if np.abs(q[1:3] - REST[1:3]).max() > AT_REST:
            self.msg = f"not at rest (joints 2-3 {q[1:3].round(3).tolist()}): staying on. " + self.msg
            return
        self.mover.close()
        self.mover, self.q = None, None
        self.held.unlink(missing_ok=True)
        self.state, self.msg = "off", "off at rest"


_PID = __import__("os").getpid()


def make_mover(side, kin, robot):
    """calib.Mover around an already connected robot (so a failed check can still close it)."""
    m = calib.Mover.__new__(calib.Mover)
    m.side, m.kin, m.check = side, kin, calib.calib_checker(calib.PLAN_CLEARANCE)
    m.monitor = calib.calib_checker(calib.MONITOR_CLEARANCE)
    m.path_check = calib.calib_checker(calib.PATH_CLEARANCE)
    m.recover = calib.calib_checker(calib.RECOVER_CLEARANCE)
    m.robot, m.max_lag = robot, np.zeros(6)
    m.arm = safety.SafeArm(robot, m.monitor)
    return m


def camview_running():
    return subprocess.run(["tmux", "has-session", "-t", "camview"], capture_output=True).returncode == 0


def toggle_camview():
    if camview_running():
        subprocess.run(["tmux", "kill-session", "-t", "camview"], capture_output=True)
        for p in (camshare.DIR / "share_url",):
            p.unlink(missing_ok=True)
    else:
        subprocess.run(["tmux", "new-session", "-d", "-s", "camview", CAMVIEW], capture_output=True)


def draw(scr, arms, note):
    scr.erase()
    h, w = scr.getmaxyx()
    put = lambda y, s, a=0: y < h and scr.addnstr(y, 0, s, w - 1, a)  # noqa: E731
    stop = estop.stopped()
    put(0, "duchamp3 rig console", curses.A_BOLD)
    put(1, time.strftime("%H:%M:%S"))
    put(2, " STOP SET: arms hold, nothing moves (c clears) " if stop else " armed (space = STOP) ",
        curses.color_pair(1) | curses.A_BOLD if stop else curses.color_pair(2) | curses.A_BOLD)
    y = 4
    for a in arms.values():
        pid, who = armshare.owner(a.side)
        user = "" if pid in (None, _PID) else f"   in use by {who} (pid {pid})"
        put(y, f"{a.side:5s} arm: {a.state.upper():10s}{user}", curses.A_BOLD)
        if a.q is not None:
            put(y + 1, "   joints (deg): " + " ".join(f"{v:7.1f}" for v in np.degrees(a.q[:6]))
                + (f"   gripper {a.q[6]:.2f}" if len(a.q) > 6 else ""))
        put(y + 2, f"   {a.msg}")
        y += 4
    cams = "on" if camview_running() else "off"
    put(y, f"camera viewer: {cams}", curses.A_BOLD)
    if cams == "on":
        url = (camshare.DIR / "share_url")
        put(y + 1, "   public: " + (url.read_text().strip() if url.exists() else "(starting)"))
        put(y + 2, "   tailnet: http://duchamp3.elf-altair.ts.net:8090")
    users = [f"{c}: {camshare.read_pid(camshare.DIR / f'{c}.claim')[1]}" for c in camshare.CAMS
             if camshare.read_pid(camshare.DIR / f"{c}.claim")[0]]
    if users:
        put(y + 3, "   cameras in use by scripts: " + ", ".join(users))
    y += 5
    put(y - 1, f"log: {armshare.DIR / 'rigctl.log'}", curses.A_DIM)
    put(y, "space STOP   c clear   o/f both on/off   r/l toggle right/left   h home   v viewer   q quit",
        curses.A_DIM)
    put(y + 1, note)
    scr.refresh()


def main(scr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_RED)
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_GREEN)
    scr.timeout(200)
    kin = calib.Kin()
    arms = {s: Arm(s, kin) for s in armshare.SIDES}
    for a in arms.values():
        a.start()
    note, quitting = "", False
    while True:
        draw(scr, arms, note)
        if quitting:
            if all(a.mover is None for a in arms.values()):
                return
            if all(a.jobs.empty() and a.state in ("on", "off") for a in arms.values()) and \
                    any(a.mover is not None for a in arms.values()) and time.time() - quitting > 1.0:
                note, quitting = "could not park every arm: still on (see its message); q to retry", False
        try:
            k = scr.getch()
        except KeyboardInterrupt:
            k = ord("q")
        if k == -1:
            continue
        ch = chr(k).lower() if 0 <= k < 256 else ""
        if quitting and ch != " ":
            continue
        note = ""
        if ch in (" ", "\n", "\r"):
            estop.STOP_FILE.write_text("stop")
            for a in arms.values():
                if a.mover is not None:
                    a.ask("hold")
        elif ch == "c":
            estop.STOP_FILE.unlink(missing_ok=True)
        elif ch == "o":
            for a in arms.values():
                a.ask("on")
        elif ch == "f":
            for a in arms.values():
                a.ask("off")
        elif ch in ("r", "l"):
            a = arms["right" if ch == "r" else "left"]
            a.ask("off" if a.mover is not None else "on")
        elif ch == "h":
            for a in arms.values():
                a.ask("home")
        elif ch == "v":
            toggle_camview()
        elif ch == "q":
            note, quitting = "parking and turning off the arms, then quitting (space = STOP)", time.time()
            for a in arms.values():
                a.ask("off")


if __name__ == "__main__":
    # i2rt prints and logs (connect, close, motor thread errors) go to a file, not over the screen:
    # Python-level stdout and the stderr fd; curses keeps the terminal on fd 1
    armshare.DIR.mkdir(exist_ok=True)
    log = open(armshare.DIR / "rigctl.log", "a", buffering=1)
    log.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} rigctl started\n")
    import os
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    curses.wrapper(main)
