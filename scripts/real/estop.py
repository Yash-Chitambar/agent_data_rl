"""Software stop for scripts that drive the real arms (calib.py). Run it in your own terminal:

    .venv/bin/python scripts/real/estop.py

  Enter      -> STOP: the arm holds where it is (PD hold, never limp) and the script waits
  c + Enter  -> clear: the script then moves the arm slowly back to rest and exits
  q + Enter  -> quit this window (a stop stays set)

It works through a flag file (STOP_FILE) that the motion loop checks every 20 ms. It does not cut
power: a killed process or the hardware e-stop drops a raised arm.
"""
import os
import sys
from pathlib import Path

# YAM_ESTOP_FILE: offline tests must point this elsewhere, never at the live stop
STOP_FILE = Path(os.environ.get("YAM_ESTOP_FILE", "/tmp/yam_estop"))


def stopped() -> bool:
    return STOP_FILE.exists()


def main():
    print(__doc__.split("\n\n")[1])
    while True:
        print(f"[{'STOPPED' if stopped() else 'armed'}] Enter = STOP, c = clear, q = quit > ", end="", flush=True)
        try:
            s = sys.stdin.readline().strip().lower()
        except KeyboardInterrupt:
            s = ""
        if s == "q":
            break
        if s == "c":
            STOP_FILE.unlink(missing_ok=True)
            print("cleared: the arm returns slowly to rest")
        else:
            STOP_FILE.write_text("stop")
            print("STOP set: the arm holds its pose")


if __name__ == "__main__":
    main()
