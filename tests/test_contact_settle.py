import sys
from types import SimpleNamespace
import numpy as np
import pytest
from deploy import contact_settle as S


@pytest.mark.parametrize('stop_first',[True,False])
def test_stop_or_geometry_veto_prevents_any_settle_command(monkeypatch,stop_first):
    class Stop(Exception):pass
    monkeypatch.setitem(sys.modules,'estop',SimpleNamespace(stopped=lambda:stop_first))
    monkeypatch.setitem(sys.modules,'safety',SimpleNamespace(SafetyStop=Stop))
    class Robot:
        def fk(self,q):return SimpleNamespace(numpy=lambda:np.zeros(3)),None
    monkeypatch.setattr(S.C,'Robot',Robot)
    monkeypatch.setattr(S.C,'joint_offsets',lambda side:np.zeros(6))
    def forbidden(*args):pytest.fail('command sent after stop/geometry veto')
    arms={side:SimpleNamespace(send=forbidden) for side in ['left','right']}
    def guard(poses):raise Stop('geometry veto')
    obs=lambda:{s:(np.zeros(7),np.zeros(7),0,True) for s in arms}
    rows=[]
    with pytest.raises(Stop):S.characterize(arms,obs,guard,lambda:None,rows,{},np.zeros(6))
    assert not rows


def test_pair_hold_attempts_both_arms_even_if_one_hold_fails():
    from deploy.contact_approach import Pair
    called=[]
    def first():
        called.append('right');raise RuntimeError('right hold failed')
    def second():called.append('left')
    pair=Pair({'right':SimpleNamespace(hold=first),'left':SimpleNamespace(hold=second)})
    with pytest.raises(RuntimeError,match='hold errors'):pair.hold()
    assert called==['right','left']
