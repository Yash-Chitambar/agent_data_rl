import numpy as np
import pytest
from deploy.contact_tracking import require_attained


def test_measured_deadband_prevents_further_accumulated_contact_targets():
    actual=np.array([.345,.309,.042])
    assert require_attained(actual,actual)==0
    with pytest.raises(ValueError,match='no further contact jog'):
        require_attained(actual,[.345,.309,.034])


@pytest.mark.parametrize('actual,target', [([float('nan'),0,0],[0,0,0]),
    ([0,0],[0,0,0]),([0,0,0],[0,float('inf'),0])])
def test_invalid_feedback_cannot_pass_attainment(actual,target):
    with pytest.raises(ValueError):require_attained(actual,target)
