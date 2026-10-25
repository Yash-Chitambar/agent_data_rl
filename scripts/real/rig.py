"""Rig constants for the YAM cell on duchamp3 (yam-control's rig_config.py is for athena3).

CAN interface names (can0/can1) are given in USB probe order and can swap at boot, so each arm
is found by its USB-CAN adapter serial instead. Use `can_channel("left")` in place of a fixed name.

    .venv/bin/python scripts/real/rig.py     # print the arm -> interface map; exit 1 if not OK
"""
import sys
from pathlib import Path

# Arm side -> USB-CAN adapter serial (both CANable2). Confirmed 2026-10-25 by watching which
# gripper calibrated on connect. If a CAN cable moves to the other adapter, update this.
# Adapter 0045006F5230501920333159 was taken out: motors dropped out on any arm cabled to it.
CAN_SERIAL = {
    "left": "0051005C594E501820313332",
    "right": "00410031594E501820313332",
}
CAN_BITRATE = 1_000_000

# Wrist cameras, by USB port: both webcams report the same serial ("YHTek"). Sides confirmed
# 2026-10-25 (a hand in front of the right one). Moving a cable to another port breaks this map.
# Specs: blind_insertion/configs/real_cameras_duchamp3.json.
WRIST_CAM = {
    "right": "/dev/v4l/by-path/pci-0000:16:00.4-usb-0:2:1.0-video-index0",  # USB 7-2
    "left": "/dev/v4l/by-path/pci-0000:16:00.3-usb-0:2:1.0-video-index0",   # USB 5-2
}
WRIST_CAM_RESOLUTION = (640, 480)  # generic UVC (Realtek 0bda:5883); MJPG 60 fps, YUYV 30 fps
ZED_SERIAL = 20120598  # top camera, ZED 2; needs a USB 3 link for the SDK (HD720 at 30 fps)
ZED_UVC_DEVICE = "/dev/v4l/by-id/usb-Technologies__Inc._ZED_2-video-index0"


def _adapter_serial(iface: Path) -> str | None:
    try:
        return (iface / "device" / ".." / "serial").resolve().read_text().strip()
    except OSError:
        return None


def can_interfaces() -> dict[str, dict]:
    """{interface name: {"serial", "state"}} for every CAN interface."""
    out = {}
    for iface in sorted(Path("/sys/class/net").glob("can*")):
        state = (iface / "operstate").read_text().strip()
        out[iface.name] = {"serial": _adapter_serial(iface), "state": state}
    return out


def can_channel(side: str) -> str:
    """Interface name (e.g. "can1") of the arm on `side`; raises if its adapter is missing or down."""
    want = CAN_SERIAL[side]
    for name, info in can_interfaces().items():
        if info["serial"] == want:
            if info["state"] != "up":
                raise RuntimeError(f"{side} arm adapter {want} is {name} but {info['state']}: "
                                   f"sudo ip link set {name} up type can bitrate {CAN_BITRATE}")
            return name
    raise RuntimeError(f"{side} arm adapter {want} not found (USB unplugged?)")


if __name__ == "__main__":
    ok = True
    for side in CAN_SERIAL:
        try:
            print(f"{side:5s} arm: {can_channel(side)} (adapter {CAN_SERIAL[side]})")
        except RuntimeError as e:
            print(f"{side:5s} arm: NOT OK: {e}")
            ok = False
    sys.exit(0 if ok else 1)
