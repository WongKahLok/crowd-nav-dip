"""Unit tests for COLREGs classification. No simulator, no torch, no trained model.

Run either way:
    python -m crowd_nav.policy.test_colregs      # plain, no dependencies
    pytest crowd_nav/policy/test_colregs.py
"""
import math

from crowd_nav.policy.colregs import (CROSSING, GIVE_WAY, HEAD_ON, OVERTAKEN, OVERTAKING,
                                      STAND_ON, EncounterTracker, classify_encounter, cpa,
                                      relative_bearing, violates, wrap180)


class S:
    """A state stub with the fields the classifier reads."""
    def __init__(self, px, py, vx, vy, radius=0.3):
        self.px, self.py, self.vx, self.vy, self.radius = px, py, vx, vy, radius


class A:
    """An action stub (ActionXY has the same vx/vy fields)."""
    def __init__(self, vx, vy):
        self.vx, self.vy = vx, vy


def test_wrap180():
    assert wrap180(0) == 0
    assert wrap180(190) == -170
    assert wrap180(-190) == 170
    assert wrap180(540) == 180


def test_relative_bearing_sign_is_starboard_positive():
    # Heading east (0 deg). A target due south is to starboard -> positive.
    own = S(0, 0, 1, 0)
    assert math.isclose(relative_bearing(own, S(0, -5, 0, 0), 0.0), 90.0)
    # A target due north is to port -> negative.
    assert math.isclose(relative_bearing(own, S(0, 5, 0, 0), 0.0), -90.0)
    # Dead ahead is zero.
    assert math.isclose(relative_bearing(own, S(5, 0, 0, 0), 0.0), 0.0)


def test_cpa_head_on_closing():
    own = S(0, 0, 1, 0)
    target = S(10, 0, -1, 0)          # closing at 2 m/s along the x axis
    dcpa, tcpa = cpa(own, target)
    assert math.isclose(tcpa, 5.0)     # 10 m gap / 2 m per s
    assert math.isclose(dcpa, 0.0, abs_tol=1e-9)


def test_cpa_negative_when_opening():
    own = S(0, 0, -1, 0)
    target = S(10, 0, 1, 0)           # already separating
    _, tcpa = cpa(own, target)
    assert tcpa < 0


def test_head_on():
    """Rule 14: reciprocal courses, dead ahead. Both vessels give way."""
    own = S(0, 0, 1, 0)               # heading east
    target = S(10, 0, -1, 0)          # heading west, dead ahead
    enc = classify_encounter(own, target)
    assert enc is not None
    assert enc.kind == HEAD_ON
    assert enc.role == GIVE_WAY


def test_crossing_from_starboard_gives_way():
    """Rule 15: target on our starboard side -> we keep out of the way."""
    own = S(0, 0, 1, 0)               # heading east
    target = S(10, -10, 0, 1)         # to the south-east, heading north: crosses our bow
    enc = classify_encounter(own, target)
    assert enc is not None
    assert enc.kind == CROSSING
    assert enc.role == GIVE_WAY
    assert enc.bearing > 0            # starboard side


def test_crossing_from_port_stands_on():
    """Rule 15/17: target on our port side -> we hold course and speed."""
    own = S(0, 0, 1, 0)               # heading east
    target = S(10, 10, 0, -1)         # to the north-east, heading south
    enc = classify_encounter(own, target)
    assert enc is not None
    assert enc.kind == CROSSING
    assert enc.role == STAND_ON
    assert enc.bearing < 0            # port side


def test_overtaking_we_are_give_way():
    """Rule 13: we come up on a slower vessel from astern -> we keep clear."""
    own = S(0, 0, 2, 0)               # heading east, fast
    target = S(6, 0, 1, 0)            # same course ahead of us, slower
    enc = classify_encounter(own, target)
    assert enc is not None
    assert enc.kind == OVERTAKING
    assert enc.role == GIVE_WAY


def test_being_overtaken_we_stand_on():
    """The mirror image: a faster vessel comes up astern of us -> we hold course."""
    own = S(6, 0, 1, 0)
    target = S(0, 0, 2, 0)            # behind us, faster, same course
    enc = classify_encounter(own, target)
    assert enc is not None
    assert enc.kind == OVERTAKEN
    assert enc.role == STAND_ON


def test_no_encounter_when_passing_clear():
    """Wide CPA is not a risk of collision, so no rule applies."""
    own = S(0, 0, 1, 0)
    target = S(10, 40, -1, 0)         # passes 40 m away
    assert classify_encounter(own, target) is None


def test_no_encounter_when_opening():
    own = S(0, 0, -1, 0)
    target = S(10, 0, 1, 0)
    assert classify_encounter(own, target) is None


def test_static_obstacle_is_not_a_vessel():
    """Land cells have zero velocity and must never generate an encounter."""
    own = S(0, 0, 1, 0)
    assert classify_encounter(own, S(5, 0, 0, 0)) is None


def test_head_on_violation_requires_starboard_turn():
    own = S(0, 0, 1, 0)               # heading east
    target = S(10, 0, -1, 0)
    enc = classify_encounter(own, target)
    # Heading east, starboard is -y (south). Turning south complies.
    assert not violates(A(0.7, -0.7), enc, own, target)
    # Turning north is to port: a Rule 14 violation.
    assert violates(A(0.7, 0.7), enc, own, target)
    # Holding course is also a violation: Rule 14 demands an alteration.
    assert violates(A(1.0, 0.0), enc, own, target)


def test_crossing_give_way_must_not_cross_ahead():
    own = S(0, 0, 1, 0)               # heading east
    target = S(10, -10, 0, 1)         # crossing from starboard, heading north
    enc = classify_encounter(own, target)
    assert enc.role == GIVE_WAY
    # Turning to starboard (south) passes astern of the target: compliant.
    assert not violates(A(0.7, -0.7), enc, own, target)
    # Sprinting straight ahead crosses its bow: violation.
    assert violates(A(1.0, 0.0), enc, own, target)


def test_stand_on_must_hold_course():
    own = S(0, 0, 1, 0)
    target = S(10, 10, 0, -1)         # crossing from port -> we are stand-on
    enc = classify_encounter(own, target)
    assert enc.role == STAND_ON
    assert not violates(A(1.0, 0.0), enc, own, target)      # holding course
    assert violates(A(0.0, -1.0), enc, own, target)         # a 90 deg swerve


# --- Rule 13(d): an overtaking vessel stays the overtaking vessel ------------------------

def test_tracker_keeps_overtaking_through_a_bearing_change():
    """Rule 13(d): a changing bearing must not turn an overtaker into a crossing vessel."""
    tracker = EncounterTracker(safe_cpa=2.0)
    astern, ahead = S(0, 0, 2, 0), S(6, 0, 1, 0)
    assert tracker.update(astern, [ahead])[0].kind == OVERTAKING

    # We have drawn level on the target's port side. Taken alone, the geometry now reads
    # as a crossing in which we would be the stand-on vessel.
    abeam, target = S(7, -1.5, 2, 0), S(7, 0, 1, 0)
    fresh = classify_encounter(abeam, target, safe_cpa=2.0)
    assert fresh.kind == CROSSING and fresh.role == STAND_ON

    kept = tracker.update(abeam, [target])[0]
    assert kept.kind == OVERTAKING
    assert kept.role == GIVE_WAY


def test_tracker_closes_the_encounter_once_past_and_clear():
    tracker = EncounterTracker(safe_cpa=2.0)
    tracker.update(S(0, 0, 2, 0), [S(6, 0, 1, 0)])
    assert tracker.update(S(12, 0, 2, 0), [S(7, 0, 1, 0)]) == {}
    records = tracker.records()
    assert len(records) == 1 and records[0].closed


# --- Rule 8(e): keeping clear by slackening speed ----------------------------------------

def test_crossing_give_way_may_slacken_speed():
    """Rule 8(e): slowing down is a legitimate way to keep out of the way."""
    own, target = S(0, 0, 1, 0), S(10, -10, 0, 1)
    enc = classify_encounter(own, target)
    assert enc.role == GIVE_WAY
    assert not violates(A(0.3, 0.0), enc, own, target)   # slowed to a third, passes astern
    assert violates(A(1.0, 0.0), enc, own, target)       # full speed still crosses its bow


def test_give_way_need_not_keep_turning_once_the_cpa_is_open():
    """Rule 16 asks for a manoeuvre that works, not one alteration per step."""
    tracker = EncounterTracker()
    own, target = S(0, 0, 1, 0), S(10, -10, 0, 1)
    turn = A(0.7, -0.7)
    assert tracker.update(own, [target], action=turn)[0].role == GIVE_WAY
    moved_own, moved_target = S(0.7, -0.7, 0.7, -0.7), S(10, -9, 0, 1)
    kept = tracker.update(moved_own, [moved_target], action=turn)[0]
    assert kept.role == GIVE_WAY                          # the encounter is still live
    assert not violates(turn, kept, moved_own, moved_target)   # holding the new course is fine
    assert tracker.records()[0].compliant


def test_head_on_slowing_alone_is_not_enough():
    """Rule 14 names an alteration to starboard, so speed alone does not satisfy it."""
    own, target = S(0, 0, 1, 0), S(10, 0, -1, 0)
    enc = classify_encounter(own, target)
    assert violates(A(0.3, 0.0), enc, own, target)
    assert not violates(A(0.7, -0.7), enc, own, target)


# --- the stop loophole -------------------------------------------------------------------

def test_stopping_is_judged_by_what_it_achieves():
    """Stopping is not an exemption: it counts only where it keeps the vessel clear."""
    own, target = S(0, 0, 1, 0), S(10, 0, -1, 0)          # head-on
    assert violates(A(0.0, 0.0), classify_encounter(own, target), own, target)

    faster, slower = S(0, 0, 2, 0), S(6, 0, 1, 0)         # overtaking
    enc = classify_encounter(faster, slower)
    assert enc.kind == OVERTAKING
    assert not violates(A(0.0, 0.0), enc, faster, slower)  # dropping back does keep clear


def test_stand_on_may_not_stop_or_slow():
    """Rule 17: hold course *and speed*."""
    own, target = S(0, 0, 1, 0), S(10, 10, 0, -1)
    enc = classify_encounter(own, target)
    assert enc.role == STAND_ON
    assert violates(A(0.0, 0.0), enc, own, target)        # stopping is a change of speed
    assert violates(A(0.5, 0.0), enc, own, target)        # so is halving speed
    assert not violates(A(1.0, 0.0), enc, own, target)


# --- the compliance metric ---------------------------------------------------------------

def test_tracker_counts_compliance_and_waiting():
    tracker = EncounterTracker()
    own, target = S(0, 0, 1, 0), S(10, -10, 0, 1)         # crossing, we give way
    for _ in range(3):
        tracker.update(own, [target], action=A(1.0, 0.0))  # crosses ahead every step
    tracker.update(own, [target], action=A(0.0, 0.0))      # then waits it out
    summary = tracker.summary()
    assert summary['encounters'] == 1
    assert summary['compliant'] == 0
    assert summary['compliance_rate'] == 0.0
    assert summary['violating_steps'] == 3
    assert summary['steps'] == 4
    assert summary['stopped_steps'] == 1
    assert summary['by_kind'][CROSSING]['encounters'] == 1


def _run():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print('  PASS  %s' % name)
        except AssertionError as exc:
            failures += 1
            print('  FAIL  %s  %s' % (name, exc))
    print('\n%d/%d passed' % (len(tests) - failures, len(tests)))
    return failures


if __name__ == '__main__':
    raise SystemExit(_run())
