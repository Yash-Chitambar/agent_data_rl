import pytest
from deploy.contact_steps import ContactSteps


def test_contact_slide_is_one_way_bounded_and_requires_disengagement():
    c=ContactSteps((0,0,0),(0,-1,0))
    assert c.propose(operator_next=True) is None
    with pytest.raises(ValueError):c.arm(False,True)
    c.arm(True,True)
    for _ in range(10):
        target=c.propose(operator_next=True)
        assert target[1]<0
        c.executed(target)
    assert c.propose(operator_next=True) is None
    assert not c.may_reset
    with pytest.raises(ValueError):c.disengaged(False)
    c.disengaged(True)
    assert c.may_reset


def test_stale_observations_or_stop_cannot_execute_pending_target():
    for kwargs in ({'stopped':True},{'observations_valid':False}):
        c=ContactSteps((0,0,0),(1,0,0));c.arm(True,True)
        pending=c.propose(operator_next=True)
        assert c.propose(operator_next=True,**kwargs) is None
        with pytest.raises(ValueError):c.executed(pending)


def test_execution_requires_proposal_and_finite_exact_target():
    c=ContactSteps((0,0,0),(1,0,0));c.arm(True,True)
    with pytest.raises(ValueError):c.executed((.0005,0,0))
    c.propose(operator_next=True)
    with pytest.raises(ValueError):c.executed((float('nan'),0,0))
