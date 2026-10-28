"""The shared handoff module (blind_insertion/pipeline/handoff.py) against the logic it replaced in
pipeline/pick_insert.ActPick (copied below from commit 2e5c599), on random input sequences.

    .venv/bin/python deploy/checks/handoff_equiv.py
"""
import torch

from blind_insertion.pipeline import handoff as H


def old_ensemble(buf, valid, c, k):
    N, C, D = c.shape
    if buf is None:
        buf = torch.zeros(N, C, C, D)
        valid = torch.zeros(C, dtype=torch.bool)
    buf = torch.roll(buf, 1, dims=1)
    buf[:, 0] = c
    valid = torch.roll(valid, 1)
    valid[0] = True
    ages = torch.arange(C)
    preds = buf[:, ages, ages]
    w = torch.exp(-k * ages.float()) * valid
    return buf, valid, (preds * w[None, :, None]).sum(1) / w.sum()


class OldGate:
    def __init__(self, n, p_nom, dt, rest_qd=0.03, rest_s=0.3, near=0.06, near_z=0.010, min_finger=0.005, grasp_s=0.5):
        self.p_nom, self.dt = torch.tensor(p_nom), dt
        self.rest_qd, self.rest_s, self.near, self.near_z, self.min_finger, self.grasp_s = rest_qd, rest_s, near, near_z, min_finger, grasp_s
        self.done_run = torch.zeros(n, dtype=torch.long)
        self.rest_run = torch.zeros(n, dtype=torch.long)
        self.grasp_run = torch.zeros(n, dtype=torch.long)

    def _settled(self, qd, closed, finger, tcp):
        rest = ((qd.abs().max(1).values < self.rest_qd) & closed & (finger > self.min_finger)
                & ((tcp - self.p_nom).norm(dim=1) < self.near)
                & ((tcp[:, 2] - self.p_nom[2]).abs() < self.near_z))
        self.rest_run = torch.where(rest, self.rest_run + 1, torch.zeros_like(self.rest_run))
        return self.rest_run >= int(self.rest_s / self.dt)

    def __call__(self, done_out, grip_cmd, finger, qd, tcp):
        self.done_run = torch.where(done_out > 0.5, self.done_run + 1, torch.zeros_like(self.done_run))
        grasped = (grip_cmd < 0) & (finger > self.min_finger)
        self.grasp_run = torch.where(grasped, self.grasp_run + 1, torch.zeros_like(self.grasp_run))
        held = self.grasp_run >= int(self.grasp_s / self.dt)
        done = (self.done_run >= 5) & held
        done |= self._settled(qd, grip_cmd < 0, finger, tcp) & held
        return done


def main():
    g = torch.Generator().manual_seed(0)
    n, T, dt = 256, 600, 0.02
    p_nom = (0.27, 0.237, 0.0228)
    old, new = OldGate(n, p_nom, dt), H.HandoffGate(p_nom, dt)
    new.reset(n, "cpu")
    fired = settle_only = 0
    # piecewise-constant random signals (each world redraws one with probability 3% per step), so the
    # runs of 5 / 15 / 25 steps the gate counts both complete and break
    s = lambda *shape: torch.rand(*shape, generator=g)  # noqa: E731
    draw = lambda: dict(done=s(n), grip=torch.where(s(n) < 0.7, -1.0, 1.0), finger=0.002 + 0.006 * s(n),  # noqa: E731
                        qd=0.04 * s(n, 6) ** 2, tcp=torch.tensor(p_nom) + 0.08 * (s(n, 3) - 0.5) * torch.tensor([1.0, 1.0, 0.4]))
    x = draw()
    for t in range(T):
        y, m = draw(), s(n) < 0.03
        x = {k: torch.where(m.view(-1, *[1] * (v.dim() - 1)), y[k], v) for k, v in x.items()}
        a, b = old(x["done"], x["grip"], x["finger"], x["qd"], x["tcp"]), new(x["done"], x["grip"] < 0, x["finger"], x["qd"], x["tcp"])
        assert torch.equal(a, b), f"gate differs at step {t}"
        fired += int(a.sum())
        settle_only += int((a & (old.done_run < 5)).sum())
    print(f"handoff gate: identical over {n} x {T} steps ({fired} handoff signals, {settle_only} by the settle path only)")
    assert fired > 100 and settle_only > 10, "the test did not exercise both paths"
    buf = valid = None
    ens = H.TemporalEnsemble(0.01)
    for t in range(120):
        c = torch.randn(4, 50, 8, generator=g)
        buf, valid, a = old_ensemble(buf, valid, c, 0.01)
        assert torch.equal(a, ens(c)), f"ensemble differs at step {t}"
    print("temporal ensemble: identical over 120 steps")
    print("CHECK PASSED")


if __name__ == "__main__":
    main()
