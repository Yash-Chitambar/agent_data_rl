#!/bin/bash
# Bring the YAM CAN adapters up by themselves: at boot and whenever an adapter is plugged in.
# One-time setup, needs sudo:
#
#     bash scripts/real/setup_can_autostart.sh
#
# Installs a udev rule for every gs_usb CAN interface (both YAM adapters): bitrate 1 Mbit/s,
# txqueuelen 1000, up. The interface names (can0/can1) can swap between boots; rig.py finds each
# arm by its adapter's USB serial, so that does not matter. Undo: sudo rm the rule file.
set -e
RULE=/etc/udev/rules.d/80-yam-can.rules
sudo tee "$RULE" > /dev/null <<'EOF'
# YAM arms (agent_data_rl scripts/real/setup_can_autostart.sh): gs_usb CAN adapters up at 1 Mbit/s
ACTION=="add", SUBSYSTEM=="net", KERNEL=="can*", DRIVERS=="gs_usb", RUN+="/usr/bin/ip link set %k type can bitrate 1000000", RUN+="/usr/bin/ip link set %k txqueuelen 1000", RUN+="/usr/bin/ip link set %k up"
EOF
sudo udevadm control --reload-rules
echo "installed $RULE:"
cat "$RULE"
# set the queue length on the interfaces that are up now (the rest of the rule acts from the next plug-in or boot)
for i in $(ls /sys/class/net | grep '^can'); do
  sudo ip link set "$i" txqueuelen 1000
  ip -br link show "$i"
done
echo "done. It takes effect at the next boot or adapter plug-in."
