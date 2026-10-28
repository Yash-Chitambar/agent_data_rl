import importlib.util
from pathlib import Path
import pytest
s=importlib.util.spec_from_file_location('steps',Path(__file__).parents[1]/'scripts/real/contact_steps.py');m=importlib.util.module_from_spec(s)
import sys
sys.modules[s.name]=m;s.loader.exec_module(m)


def test_no_motion_until_contact_and_operator_step():
    c=m.ContactSteps((0,0,0),(0,-1,0))
    assert c.propose(operator_next=True) is None
    with pytest.raises(ValueError):c.arm(False,True)
    c.arm(True,True)
    assert c.propose(operator_next=False) is None
    for _ in range(10):
        target=c.propose(operator_next=True);assert target[1]<0;c.executed(target)
    assert c.propose(operator_next=True) is None
    assert not c.may_reset
    c.disengaged(True);assert c.may_reset


def test_stop_and_stale_observations_hold():
    for kwargs in ({'stopped':True},{'observations_valid':False}):
        c=m.ContactSteps((0,0,0),(1,0,0));c.arm(True,True)
        assert c.propose(operator_next=True,**kwargs) is None
        assert c.propose(operator_next=True) is None


def test_execution_cannot_bypass_proposal():
    c=m.ContactSteps((0,0,0),(1,0,0));c.arm(True,True)
    with pytest.raises(ValueError):c.executed((.0005,0,0))
    c.propose(operator_next=True)
    with pytest.raises(ValueError):c.executed((float('nan'),0,0))
    c.executed((.0005,0,0))
    with pytest.raises(ValueError):c.executed((.0005,0,0))
