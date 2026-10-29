"""Hardware-independent bounds for operator-paced PC-contact trials.

This module only proposes Cartesian targets. It never connects an arm. Every
proposal still needs IK, swept-geometry checks and controller health checks.
"""
from dataclasses import dataclass, field
import math


@dataclass
class ContactSteps:
    origin: tuple
    direction: tuple
    step_m: float = .0005
    max_travel_m: float = .005
    phase: str = 'unarmed'
    travelled_m: float = 0.
    events: list = field(default_factory=list)
    pending: tuple | None = None

    def __post_init__(self):
        if len(self.origin)!=3 or len(self.direction)!=3:
            raise ValueError('three-dimensional position/direction required')
        if not all(math.isfinite(v) for v in (*self.origin,*self.direction,self.step_m,self.max_travel_m)):
            raise ValueError('non-finite geometry')
        norm=math.sqrt(sum(v*v for v in self.direction))
        if abs(norm-1)>1e-6:raise ValueError('direction must be unit length')
        if not 0 < self.step_m <= .0005 or not self.step_m <= self.max_travel_m <= .005:
            raise ValueError('step <= 0.5 mm and trial travel <= 5 mm required')

    def arm(self, operator_contact_confirmed, observations_valid):
        if self.phase!='unarmed' or not operator_contact_confirmed or not observations_valid:
            raise ValueError('actual contact and valid observations required before sliding')
        self.phase='sliding'
        self.events.append('operator_contact_confirmed')

    def propose(self, *, operator_next, stopped=False, observations_valid=True):
        if stopped or not observations_valid:
            self.phase='hold';self.pending=None;self.events.append('stop_or_invalid_observations');return None
        if self.phase!='sliding' or not operator_next:return None
        distance=self.travelled_m+self.step_m
        if distance>self.max_travel_m+1e-12:
            self.phase='hold';self.events.append('travel_limit');return None
        self.pending=tuple(p+distance*d for p,d in zip(self.origin,self.direction))
        return self.pending

    def executed(self, target):
        if (self.phase!='sliding' or self.pending is None or len(target)!=3
            or not all(math.isfinite(v) for v in target)
            or self.travelled_m+self.step_m>self.max_travel_m+1e-12
            or any(abs(a-b)>1e-9 for a,b in zip(target,self.pending))):
            raise ValueError('executed target differs from bounded proposal')
        self.travelled_m+=self.step_m
        self.pending=None

    def disengaged(self, operator_confirmed):
        if not operator_confirmed:raise ValueError('confirm disengagement before reset')
        self.phase='disengaged';self.events.append('operator_disengaged')

    @property
    def may_reset(self):
        return self.phase=='disengaged'
