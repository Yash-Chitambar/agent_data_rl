"""Real rig: calibrate the gripper map (config gripper.*) on the right arm. The arm holds its pose
throughout; only the gripper moves.

  1. open fully -> reading g_open
  2. close empty -> reading g_empty (the sim fingers read ~-6 mm closed on nothing; the handoff gate
     needs the empty reading below handoff.min_finger = 5 mm after the map)
  3. you hold the SD reader between the fingers (top-down, across its 36 mm body width) -> close ->
     reading g_reader; the sim fingers read ~19 mm on the reader (m_reader)

Prints the config values. The squeeze is not measured here: the sim squeezes 9-15 N per finger;
measure the real one at the chosen close target (e.g. a luggage scale on a strap between the pads)
and lower or raise gripper.close until it falls in that range.

    .venv/bin/python -m deploy.checks.gripper_cal          (scripts/real/estop.py running)
"""
import time
import select

import numpy as np

from deploy import conventions as C  # noqa: F401  (sys.path: scripts/real)
from deploy.runner import load_config


def wait_input(prompt):
    import estop
    print(prompt, end="", flush=True)
    while True:
        if estop.stopped():
            raise RuntimeError("software stop set; holding current command until operator retrieves reader")
        ready, _, _ = select.select([__import__("sys").stdin], [], [], 0.05)
        if ready:
            return input()


def settle(arm, g, s=1.5):
    import estop
    if estop.stopped():
        raise RuntimeError("software stop set before gripper command")
    q = arm.read()
    q[6] = g
    arm.command(q)
    end = time.monotonic() + s
    while time.monotonic() < end:
        if estop.stopped():
            q = arm.read()
            arm._hold(q)
            raise RuntimeError("software stop set; holding current command until operator retrieves reader")
        time.sleep(0.02)
    return float(arm.read()[6])


def main():
    import safety
    cfg = load_config()
    gc = cfg["gripper"]
    import estop
    if estop.stopped():
        raise SystemExit("software stop is set")
    robot = safety.connect(cfg["side"])
    arm = safety.SafeArm(robot)
    reader_may_be_held = False
    try:
        g_open = settle(arm, gc["open"])
        g_empty = settle(arm, gc["close"])
        settle(arm, gc["open"])
        reader_may_be_held = True  # reader may be placed at any point while waiting
        wait_input("Hold the SD reader between the open fingers, then press Enter: ")
        g_reader = settle(arm, gc["close"], 2.0)
        wait_input("Hold the reader securely; press Enter to reopen the jaws: ")
        settle(arm, gc["open"])
        reader_may_be_held = False
    except Exception:
        if reader_may_be_held:
            print("Reader may be held: maintaining power. Secure the reader and contact the operator before disconnecting.", flush=True)
            while True:
                time.sleep(0.1)
        raise
    finally:
        safety.close_robot(robot)
    gm = C.GripperMap(gc["open"], gc["close"], gc["m_open"], g_reader, gc["m_reader"])
    print(f"readings: open {g_open:.3f}, closed empty {g_empty:.3f}, closed on the reader {g_reader:.3f}")
    print(f"map -> empty {gm.finger_m(g_empty) * 1e3:.1f} mm (gate threshold {cfg['handoff']['min_finger'] * 1e3:.0f} mm), "
          f"reader {gm.finger_m(g_reader) * 1e3:.1f} mm, open {gm.finger_m(g_open) * 1e3:.1f} mm")
    if gm.finger_m(g_empty) >= cfg["handoff"]["min_finger"]:
        print("WARNING: an empty closed gripper maps above the gate threshold: the grasp gate cannot tell a miss")
    print(f'config: "gripper": {{"open": {gc["open"]}, "close": {gc["close"]}, "m_open": {gc["m_open"]}, '
          f'"g_reader": {g_reader:.4f}, "m_reader": {gc["m_reader"]}}}')


if __name__ == "__main__":
    main()
