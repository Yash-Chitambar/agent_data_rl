"""Coulomb friction feed-forward for the real YAM's joint servo (and the same law in the sims).

The real joints 2-3 have ~1-1.5 N m of static friction (FINDINGS 2026-10-25): a position command
closer than friction / kp (12-19 mrad) does not move them, and the sim action scheme
(q_target = q_meas + scale * a, PD) has no integral. i2rt's own compensation is
coulomb_friction * sign(qdot), which does nothing at rest.

Law, per joint, computed when the command is sent and held until the next one (i2rt holds the
`torques` field of the last command; the sims hold it for the control step):

    tau_ff = tau * clip((q_target - q_meas) / ramp, -1, 1)

It pushes toward the target with the estimated friction and fades out within `ramp` of it, so it
does not chatter at the target. Both q are in the encoder frame (the offset error cancels).
Configured as {"tau": [6 x N m], "ramp": rad}: EnvCfg.fric_comp, PickCfg.fric_comp,
scripts/real/deploy.py --fric_comp.
"""
from __future__ import annotations

import numpy as np
import torch

# measured effective friction (N m, on top of i2rt's 0.3 sign(qdot) on joints 1-3):
# results/real/step_fit.json
MEASURED_TAU = (0.01, 1.0, 1.5, 0.2, 0.2, 0.1)


def feedforward(err, tau, ramp: float):
    """err (..., 6) = q_target - q_meas (rad; torch or numpy) -> (..., 6) joint torques (N m)."""
    if isinstance(err, np.ndarray):
        return np.asarray(tau, float) * np.clip(err / ramp, -1.0, 1.0)
    tau = torch.as_tensor(tau, dtype=err.dtype, device=err.device)
    return tau * (err / ramp).clamp(-1.0, 1.0)


def parse(cfg, *, side="right") -> tuple[list, float] | None:
    """{"tau": [...], "ramp": r} or {"scale": c, "ramp": r} (c x MEASURED_TAU) -> (tau, ramp)."""
    if not cfg:
        return None
    if side not in ("right", "left"):
        raise ValueError("side must be right or left")
    if side == "left" and "tau" not in cfg:
        raise ValueError("right-arm friction prior cannot be implicitly applied to the left arm")
    tau = np.asarray(cfg["tau"] if "tau" in cfg else [cfg.get("scale", 1.0) * t for t in MEASURED_TAU], float)
    ramp = float(cfg.get("ramp", .01))
    if tau.shape != (6,) or not np.isfinite(tau).all() or (tau < 0).any():
        raise ValueError("tau must contain six finite nonnegative torque magnitudes")
    if not np.isfinite(ramp) or ramp <= 0:
        raise ValueError("ramp must be finite and positive")
    return tau.tolist(), ramp
