"""Recorded PD gains, distinct from unmeasured plant friction/driver compensation.

The rest identification captures temporarily changed the native wrist gains.
Neither profile implies that ideal scene gravity compensation is hardware accurate.
"""
PROFILES = {
    'identification_20261027': {
        'kp': (80., 80., 80., 30., 30., 30.),
        'kd': (5., 5., 5., 2., 2., 2.),
        'source': 'motion/left_rest_j3_fit/run.json and reference/right_arm_20261027/motion/right_rest_empty_j3_triangle_001/run.json',
    },
    'native_free_settle_20261028': {
        'kp': (80., 80., 80., 40., 10., 10.),
        'kd': (5., 5., 5., 1.5, 1.5, 1.5),
        'source': 'contact/free_settle_001/trial.json controllers; confirmed in 002 and 003',
    },
}


def apply(model, name):
    """Apply one recorded gain profile to both arms, retaining force limits."""
    if name not in PROFILES:
        raise ValueError(f'unknown controller profile {name!r}; choose {tuple(PROFILES)}')
    profile = PROFILES[name]
    for prefix in ('', 'left_'):
        for i, (kp, kd) in enumerate(zip(profile['kp'], profile['kd']), 1):
            a = model.actuator(prefix + f'joint{i}').id
            model.actuator_gainprm[a, 0] = kp
            model.actuator_biasprm[a, 1:3] = (-kp, -kd)
