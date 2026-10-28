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

import numpy as np

from deploy import conventions as C  # noqa: F401  (sys.path: scripts/real)
from deploy.runner import load_config


def settle(robot, g, s=1.5):
    q = np.asarray(robot.get_joint_pos(), float)
    q[6] = g
    robot.command_joint_pos(q)
    time.sleep(s)
    return float(robot.get_joint_pos()[6])


def main():
    import safety
    cfg = load_config()
    gc = cfg["gripper"]
    robot = safety.connect(cfg["side"])
    try:
        safety.SafeArm(robot)                      # refuses to start from an unsafe pose
        g_open = settle(robot, gc["open"])
        g_empty = settle(robot, gc["close"])
        settle(robot, gc["open"])
        input("Hold the SD reader between the open fingers (top-down, across the body), then press Enter ")
        g_reader = settle(robot, gc["close"], 2.0)
        settle(robot, gc["open"])
    finally:
        robot.close()
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
