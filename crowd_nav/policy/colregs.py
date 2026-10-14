"""COLREGs encounter classification and compliance checks (Rules 13, 14, 15).

Plain geometry, with no simulator, policy or torch imports, so it can be tested with
hand-written numbers. It answers two questions:

    classify_encounter(own, target)    what kind of encounter is this, and do I give way?
    violates(action, encounter, ...)   would this action break the rule for that encounter?

Bearings are in degrees in the observing ship's frame, positive to starboard. A ship's
heading is taken from its velocity; a ship that is stopped has none and is not classified.

EncounterTracker adds the memory of Rule 13(d), where an overtaking ship stays the
overtaking ship until it is past and clear, and scores the actions actually taken.

Simplifications:
* risk of collision is judged by closest point of approach (CPA) alone
* the in-extremis clause of Rule 17 is not modelled
* Rule 13 names no side, so overtaking is broken when the predicted CPA is under safe_cpa
* "ample time" and "readily apparent" (Rule 8) are not modelled; an encounter counts as
  complied with when no step in it broke its rule
"""
import math
from dataclasses import dataclass

# --- COLREGs sector constants, degrees ---------------------------------------------------
HEAD_ON_HALF_ANGLE = 15.0     # target within this of dead ahead is a candidate head-on
RECIPROCAL_TOLERANCE = 30.0   # heading difference within 180 +/- this counts as reciprocal
OVERTAKING_ABAFT_BEAM = 112.5  # 22.5 deg abaft the beam, measured from the bow

STOPPED_SPEED = 1e-3          # below this a vessel has no heading and is ignored
MIN_SPEED_DROP = 0.2          # fraction of speed that counts as slackening speed, Rule 8(e)
STAND_ON_SPEED_TOLERANCE = 0.2  # fraction a stand-on vessel may change speed by, Rule 17
CLEAR_RANGE = 3.0             # metres; beyond this, and opening, an encounter is past and clear

# encounter kinds
HEAD_ON = 'head_on'
CROSSING = 'crossing'
OVERTAKING = 'overtaking'
OVERTAKEN = 'overtaken'       # the other vessel is overtaking us; we are stand-on

# roles
GIVE_WAY = 'give_way'
STAND_ON = 'stand_on'


@dataclass
class Encounter:
    """One classified encounter between own ship and one target."""
    kind: str
    role: str
    bearing: float             # target's bearing from own ship, +starboard
    bearing_from_target: float  # own ship's bearing from the target
    distance: float
    dcpa: float                # distance at closest point of approach
    tcpa: float                # time to CPA, seconds; negative means opening

    @property
    def must_give_way(self):
        return self.role == GIVE_WAY


def wrap180(degrees):
    """Wrap an angle into (-180, 180]."""
    wrapped = (degrees + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def heading_of(state):
    """Course over ground in degrees, or None when the vessel is effectively stopped."""
    speed = math.hypot(state.vx, state.vy)
    if speed < STOPPED_SPEED:
        return None
    return math.degrees(math.atan2(state.vy, state.vx))


def relative_bearing(observer, other, observer_heading):
    """Bearing of `other` from `observer`, positive to starboard.

    x is right and y is up, so turning right lowers the angle; the sign is flipped.
    """
    line_of_sight = math.degrees(math.atan2(other.py - observer.py, other.px - observer.px))
    return wrap180(-(line_of_sight - observer_heading))


def cpa(own, target):
    """Closest point of approach: (distance at CPA, time to CPA in seconds).

    tcpa is negative when the vessels are already opening.
    """
    rx, ry = target.px - own.px, target.py - own.py
    rvx, rvy = target.vx - own.vx, target.vy - own.vy
    closing_speed_sq = rvx * rvx + rvy * rvy
    if closing_speed_sq < 1e-12:            # parallel courses, range never changes
        return math.hypot(rx, ry), 0.0
    tcpa = -(rx * rvx + ry * rvy) / closing_speed_sq
    dcpa = math.hypot(rx + rvx * tcpa, ry + rvy * tcpa)
    return dcpa, tcpa


def classify_encounter(own, target, max_range=15.0, max_tcpa=30.0, safe_cpa=1.0):
    """Classify one encounter, or return None when no rule applies.

    None means the target is out of range (max_range, m), no risk (opening, CPA further
    off than max_tcpa s, or passing wider than safe_cpa m), or one ship has no heading.
    """
    own_heading = heading_of(own)
    target_heading = heading_of(target)
    if own_heading is None or target_heading is None:
        return None                          # static cells and stopped vessels are not vessels

    distance = math.hypot(target.px - own.px, target.py - own.py)
    if distance > max_range:
        return None

    dcpa, tcpa = cpa(own, target)
    if tcpa < 0 or tcpa > max_tcpa or dcpa > safe_cpa:
        return None                          # opening, too far off, or passing clear

    bearing = relative_bearing(own, target, own_heading)
    bearing_from_target = relative_bearing(target, own, target_heading)
    heading_difference = abs(wrap180(target_heading - own_heading))

    common = dict(bearing=bearing, bearing_from_target=bearing_from_target,
                  distance=distance, dcpa=dcpa, tcpa=tcpa)

    # Rule 13 takes precedence: an overtaking vessel stays an overtaking vessel.
    if abs(bearing_from_target) > OVERTAKING_ABAFT_BEAM:
        return Encounter(kind=OVERTAKING, role=GIVE_WAY, **common)
    if abs(bearing) > OVERTAKING_ABAFT_BEAM:
        return Encounter(kind=OVERTAKEN, role=STAND_ON, **common)

    # Rule 14: nearly reciprocal courses, target nearly dead ahead. Both give way.
    if (abs(bearing) <= HEAD_ON_HALF_ANGLE and
            abs(heading_difference - 180.0) <= RECIPROCAL_TOLERANCE):
        return Encounter(kind=HEAD_ON, role=GIVE_WAY, **common)

    # Rule 15: whoever has the other on her starboard side keeps out of the way.
    role = GIVE_WAY if bearing > 0 else STAND_ON
    return Encounter(kind=CROSSING, role=role, **common)


def turn_direction(own_heading, action_heading):
    """Signed course change in degrees, positive to starboard."""
    return wrap180(-(action_heading - own_heading))


def crosses_ahead(own, target, vx, vy, horizon=None):
    """True when this velocity takes own ship across the target's bow, judged at CPA."""
    target_heading = heading_of(target)
    if target_heading is None:
        return False
    rx, ry = target.px - own.px, target.py - own.py
    rvx, rvy = target.vx - vx, target.vy - vy
    closing_speed_sq = rvx * rvx + rvy * rvy
    if closing_speed_sq < 1e-12:
        return False
    tcpa = -(rx * rvx + ry * rvy) / closing_speed_sq
    if tcpa < 0:
        return False
    if horizon is not None:
        tcpa = min(tcpa, horizon)
    # positions at CPA
    own_x, own_y = own.px + vx * tcpa, own.py + vy * tcpa
    tgt_x, tgt_y = target.px + target.vx * tcpa, target.py + target.vy * tcpa
    bearing_of_own = math.degrees(math.atan2(own_y - tgt_y, own_x - tgt_x))
    return abs(wrap180(bearing_of_own - target_heading)) < 90.0


def keeps_clear(own, target, vx, vy, safe_cpa=1.0):
    """True when the closest approach predicted after taking this velocity is safe."""
    dcpa, tcpa = cpa(_moved(own, vx, vy), target)
    return tcpa < 0 or dcpa >= safe_cpa


def violates(action, encounter, own, target, min_turn=5.0, safe_cpa=1.0,
             stand_on_tolerance=15.0, min_speed_drop=MIN_SPEED_DROP,
             stand_on_speed_tolerance=STAND_ON_SPEED_TOLERANCE):
    """True when `action` (anything with vx, vy) breaks the rule of `encounter`.

    A give-way action that opens the predicted CPA to safe_cpa counts as keeping clear.
    Slowing down is allowed except in a head-on, where Rule 14 asks for a turn to
    starboard. Stopping only complies where it keeps the ship clear, and it breaks a
    stand-on ship's duty to hold course and speed. min_turn is the smallest turn that
    counts as a turn, in degrees.
    """
    own_heading = heading_of(own)
    if own_heading is None or encounter is None:
        return False
    own_speed = math.hypot(own.vx, own.vy)
    speed = math.hypot(action.vx, action.vy)
    moving = speed >= STOPPED_SPEED
    turn = turn_direction(own_heading, math.degrees(math.atan2(action.vy, action.vx))) if moving else 0.0
    slackened = own_speed > 0 and speed <= own_speed * (1 - min_speed_drop)
    clear = keeps_clear(own, target, action.vx, action.vy, safe_cpa)

    if encounter.role == STAND_ON:
        # Rule 17: hold course and speed. Stopping is a change of speed.
        if not moving:
            return True
        return (abs(turn) > stand_on_tolerance
                or abs(speed - own_speed) > own_speed * stand_on_speed_tolerance)

    if encounter.kind == HEAD_ON:
        # Rule 14: alter to starboard and pass port to port.
        if moving and turn < -min_turn:
            return True                      # a port alteration is always wrong here
        if clear:
            return False                     # the alteration has already opened the CPA
        return not (moving and turn >= min_turn)

    if encounter.kind == CROSSING:           # give-way, Rules 15 and 16
        if moving and crosses_ahead(own, target, action.vx, action.vy):
            return True
        if clear:
            return False
        return not (moving and (turn >= min_turn or slackened))

    if encounter.kind == OVERTAKING:
        # Rule 13: keep clear; no side is mandated, so judge the resulting clearance.
        return not clear

    return False


class _State:
    """Minimal state carrier so cpa() can be reused with a substituted velocity."""
    __slots__ = ('px', 'py', 'vx', 'vy')

    def __init__(self, px, py, vx, vy):
        self.px, self.py, self.vx, self.vy = px, py, vx, vy


def _moved(state, vx, vy):
    return _State(state.px, state.py, vx, vy)


def classify_all(own, targets, **kwargs):
    """Classify every target, returning the encounters that carry a rule (None dropped)."""
    found = []
    for target in targets:
        encounter = classify_encounter(own, target, **kwargs)
        if encounter is not None:
            found.append((target, encounter))
    return found


def _geometry(own, target, previous=None):
    """The measured part of an Encounter; bearings fall back to `previous` when stopped."""
    dcpa, tcpa = cpa(own, target)
    own_heading, target_heading = heading_of(own), heading_of(target)
    bearing = (relative_bearing(own, target, own_heading) if own_heading is not None
               else (previous.bearing if previous else 0.0))
    from_target = (relative_bearing(target, own, target_heading) if target_heading is not None
                   else (previous.bearing_from_target if previous else 0.0))
    return dict(bearing=bearing, bearing_from_target=from_target,
                distance=math.hypot(target.px - own.px, target.py - own.py),
                dcpa=dcpa, tcpa=tcpa)


@dataclass
class EncounterRecord:
    """What happened during one encounter with one vessel."""
    key: object
    kind: str
    role: str
    steps: int = 0
    violations: int = 0
    stopped_steps: int = 0
    min_distance: float = math.inf
    closed: bool = False

    @property
    def compliant(self):
        """An encounter is complied with when no action taken in it broke its rule."""
        return self.violations == 0


class EncounterTracker:
    """Classify encounters with memory, and score the actions taken during them.

    An encounter lives until the ships are opening and more than ``clear_range`` apart, so
    an overtaking ship stays overtaking (Rule 13(d)) and give-way and stand-on roles do
    not swap while an encounter is being resolved.

    Pass the executed action to ``update`` to score it; leave it out to classify only.
    Targets are told apart by their position in ``targets`` unless ``keys`` is given.
    """

    def __init__(self, clear_range=CLEAR_RANGE, safe_cpa=1.0, min_turn=5.0, **classify_kwargs):
        self.clear_range = clear_range
        self.safe_cpa = safe_cpa
        self.min_turn = min_turn
        self.classify_kwargs = dict(classify_kwargs, safe_cpa=safe_cpa)
        self.active = {}        # key -> Encounter currently in force
        self.open_records = {}  # key -> EncounterRecord being filled
        self.closed_records = []

    # -- classification with memory ------------------------------------------

    def update(self, own, targets, action=None, keys=None):
        """Advance one step; returns {key: Encounter} for the encounters in force."""
        keys = list(range(len(targets))) if keys is None else list(keys)
        current = {}
        for key, target in zip(keys, targets):
            encounter = self._encounter(own, target, key)
            if encounter is None:
                continue
            current[key] = encounter
            if action is not None:
                self._score(key, encounter, own, target, action)
        for key in [k for k in self.active if k not in current]:
            self._close(key)
        return current

    def _encounter(self, own, target, key):
        fresh = classify_encounter(own, target, **self.classify_kwargs)
        previous = self.active.get(key)
        if previous is not None and not self._past_and_clear(own, target):
            # Rule 13(d): the encounter keeps its kind and role until past and clear.
            kept = Encounter(kind=previous.kind, role=previous.role,
                             **_geometry(own, target, previous))
            self.active[key] = kept
            return kept
        if previous is not None:
            self._close(key)
        if fresh is None:
            return None
        self.active[key] = fresh
        self.open_records[key] = EncounterRecord(key, fresh.kind, fresh.role)
        return fresh

    def _past_and_clear(self, own, target):
        distance = math.hypot(target.px - own.px, target.py - own.py)
        _, tcpa = cpa(own, target)
        return distance > self.clear_range and tcpa <= 0

    # -- compliance ----------------------------------------------------------

    def _score(self, key, encounter, own, target, action):
        record = self.open_records.get(key)
        if record is None:
            return
        record.steps += 1
        record.min_distance = min(record.min_distance, encounter.distance)
        if math.hypot(action.vx, action.vy) < STOPPED_SPEED:
            record.stopped_steps += 1
        if self.breaks(action, encounter, own, target):
            record.violations += 1

    def breaks(self, action, encounter, own, target):
        """`violates` with this tracker's thresholds, so every caller scores the same way."""
        return violates(action, encounter, own, target, min_turn=self.min_turn,
                        safe_cpa=self.safe_cpa)

    def _close(self, key):
        self.active.pop(key, None)
        record = self.open_records.pop(key, None)
        if record is not None:
            record.closed = True
            self.closed_records.append(record)

    def records(self):
        """Every encounter seen so far, finished or still in progress."""
        return self.closed_records + list(self.open_records.values())

    def summary(self):
        """Compliance over all encounters. ``stopped_steps`` exposes waiting it out."""
        records = self.records()
        compliant = sum(r.compliant for r in records)
        by_kind, by_role = {}, {}
        for r in records:
            for table, key in ((by_kind, r.kind), (by_role, r.role)):
                entry = table.setdefault(key, {'encounters': 0, 'compliant': 0})
                entry['encounters'] += 1
                entry['compliant'] += int(r.compliant)
        return {
            'encounters': len(records),
            'compliant': compliant,
            'compliance_rate': compliant / len(records) if records else None,
            'steps': sum(r.steps for r in records),
            'violating_steps': sum(r.violations for r in records),
            'stopped_steps': sum(r.stopped_steps for r in records),
            'by_kind': by_kind,
            'by_role': by_role,
        }
