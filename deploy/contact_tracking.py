"""Pre-command attainment gate; never accumulate contact jogs through a deadband."""
import numpy as np


def require_attained(actual_position, target_position, max_error_m=.003):
    actual=np.asarray(actual_position,dtype=float)
    target=np.asarray(target_position,dtype=float)
    if actual.shape!=(3,) or target.shape!=(3,) or not np.all(np.isfinite([actual,target])):
        raise ValueError('contact position unavailable or nonfinite')
    error=float(np.linalg.norm(actual-target))
    if not np.isfinite(max_error_m) or max_error_m<=0:
        raise ValueError('invalid contact attainment bound')
    if error>max_error_m:
        raise ValueError(f'previous target missed by {error*1000:.1f} mm (limit {max_error_m*1000:.1f}); no further contact jog')
    return error
