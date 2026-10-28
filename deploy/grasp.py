"""Real grasp evidence in normalized encoder units, separate from ACT state mapping."""
import math


def reader_present(readback, empty, reader):
    values = (readback, empty, reader)
    if not all(math.isfinite(float(v)) for v in values):
        return False
    gap = reader - empty
    if gap <= 0.1:
        return False
    # Reject empty closure and substantially open/partial grasps.
    return empty + 0.5 * gap < readback < reader + 0.5 * gap
