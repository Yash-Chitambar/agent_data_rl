import numpy as np
from deploy.approach_trim import ApproachTrim, settled


def test_capped_rate_and_stop_freeze():
    c = ApproachTrim()
    previous = c.bias.copy()
    for _ in range(1000):
        now = c.step(np.ones(6), .02)
        assert np.max(abs(now - previous)) <= c.rate * .02 + 1e-12
        previous = now
    assert np.allclose(now, c.limit)
    assert np.array_equal(c.step(-np.ones(6), .02, stopped=True), now)
    assert np.array_equal(c.step(np.full(6, np.nan), .02), now)
    assert np.array_equal(c.step(np.ones(6), .5), now)        # implausible dt: frozen


def test_deadband_freezes_offset():
    c = ApproachTrim()
    for _ in range(100):
        c.step(np.full(6, .01), .02)
    b = c.bias.copy()
    for _ in range(1000):                                      # joint now within the deadband
        c.step(np.full(6, .0015), .02)
    assert np.array_equal(c.bias, b)


def test_stick_slip_joint_converges():
    """A joint with Coulomb static friction under PD: it moves only while |kp (cmd - q)| > f_s."""
    kp, fs, goal, dt = 80.0, 1.8, .3, .02
    q, c = goal - .025, ApproachTrim()
    for _ in range(750):
        cmd = goal + c.step(np.full(6, goal - q), dt)[2]
        tau = kp * (cmd - q)
        if abs(tau) > fs:                                      # slips to where friction holds it again
            q = cmd - np.sign(tau) * fs / kp * .5
    assert abs(goal - q) < .005
    assert abs(c.bias[2]) < c.limit


def test_settled():
    assert not settled([], .005)
    assert settled([np.full(6, .004)] * 3, .005)
    assert not settled([np.full(6, .004), np.full(6, .006)], .005)
