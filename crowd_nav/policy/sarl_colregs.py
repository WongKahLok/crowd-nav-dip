"""SARL with the COLREGs steering rules applied at action-selection time.

The value network is untouched: this policy only narrows the set of actions it is
allowed to choose from. SARL already builds kinematically reachable candidates and
drops the ones that fail the swept safety check; this subclass drops, in addition,
the ones that would break the rule in force for any vessel currently in an encounter
(``crowd_nav.policy.colregs``). The value network then picks the best of what is left,
so nothing has to be retrained.

Order of preference, highest first:

1. safe and compliant with every give-way duty in force (see ``breaks_rules``),
2. safe and compliant with the most urgent duties: when nothing keeps them all, the
   duty furthest from its closest approach is dropped, then the next, and so on,
3. safe (every rule is dropped when obeying even the most urgent one would leave
   nothing safe to do, which is how Rule 2 and Rule 17(b) are approximated: the rules
   never require a collision),
4. the evasive action SARL falls back on when nothing is safe, steered by the rules:
   among the options within ``emergency_tolerance`` of the most room, the one the
   rules favour for the most urgent encounter wins (``colregs.emergency_preferred``).

``rule_relaxed`` counts the steps that fell to level 2, ``rule_yielded`` those that
fell to level 3, ``rule_filtered`` the candidates removed by the rules on the last
pass and ``rule_steered`` the emergency steps where the rules changed the evasive
action, so the cost of the rules can be reported. ``multi_duty = false`` enforces
only the most urgent duty, the behaviour before levels 1 and 2 were split.

Thresholds come from an optional ``[colregs]`` section of policy.config; the defaults
match crowd_nav.policy.colregs, and a config without that section works unchanged.
"""
import logging

from crowd_nav.policy import colregs
from crowd_nav.policy.sarl import SARL
from crowd_sim.envs.utils.info import Collision


class SARLColregs(SARL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL-COLREGs'
        # rule thresholds, all interpretations of the convention's wording
        self.safe_cpa = 1.0
        self.max_range = 10.0
        self.max_tcpa = 20.0
        self.clear_range = colregs.CLEAR_RANGE
        self.min_turn = 5.0
        self.enforce_stand_on = False
        # enforce every duty in force at once, relaxing the least urgent first
        self.multi_duty = True
        # metres of evasive room an emergency action may give up to follow the rules
        self.emergency_tolerance = 0.1
        # per-episode state and diagnostics
        self.tracker = None
        self.targets = []
        self.encounters = {}
        self.rule_yielded = 0
        self.rule_relaxed = 0
        self.rule_filtered = 0
        self.rule_steered = 0
        self._apply_rules = True
        self._duties = []         # (key, Encounter) enforced on the current pass
        self._rule_free_safe = 0  # candidates of the current pass safe before the rules
        self._last_time = -1.0

    def configure(self, config):
        super().configure(config)
        self.safe_cpa = config.getfloat('colregs', 'safe_cpa', fallback=self.safe_cpa)
        self.max_range = config.getfloat('colregs', 'max_range', fallback=self.max_range)
        self.max_tcpa = config.getfloat('colregs', 'max_tcpa', fallback=self.max_tcpa)
        self.clear_range = config.getfloat('colregs', 'clear_range', fallback=self.clear_range)
        self.min_turn = config.getfloat('colregs', 'min_turn', fallback=self.min_turn)
        self.enforce_stand_on = config.getboolean('colregs', 'enforce_stand_on',
                                                  fallback=self.enforce_stand_on)
        self.multi_duty = config.getboolean('colregs', 'multi_duty', fallback=self.multi_duty)
        self.emergency_tolerance = config.getfloat('colregs', 'emergency_tolerance',
                                                   fallback=self.emergency_tolerance)
        if self.emergency_tolerance < 0:
            raise ValueError('emergency_tolerance must not be negative')
        self.name = 'SARL-COLREGs'
        logging.info('COLREGs filter: safe CPA %.2f m, range %.1f m, horizon %.0f s, '
                     'clear at %.1f m, alteration %.0f deg',
                     self.safe_cpa, self.max_range, self.max_tcpa, self.clear_range, self.min_turn)
        logging.info('COLREGs enforced for: %s, %s; emergencies follow the rules within %.2f m',
                     'all roles' if self.enforce_stand_on else 'give-way duties only',
                     'every duty at once' if self.multi_duty else 'the most urgent duty only',
                     self.emergency_tolerance)

    # -- encounters -----------------------------------------------------------

    def new_tracker(self):
        return colregs.EncounterTracker(clear_range=self.clear_range, safe_cpa=self.safe_cpa,
                                        max_range=self.max_range, max_tcpa=self.max_tcpa)

    def update_encounters(self, state):
        """Classify this step's encounters, restarting the memory on a new episode."""
        now = getattr(self.env, 'global_time', None)
        if self.tracker is None or now is None or now <= self._last_time:
            self.tracker = self.new_tracker()
        self._last_time = -1.0 if now is None else now
        # Land cells are not vessels: they carry no heading and no right of way.
        self.targets = [h for h in state.human_states if not getattr(h, 'is_static', False)]
        return self.tracker.update(state.self_state, self.targets)

    def breaks_rules(self, nav, action):
        """True when this action would break a rule that is being enforced.

        Only the give-way duties (Rules 13, 14, 15 and 16) are enforced by default.
        Rule 17 tells a stand-on vessel to hold course and speed, but this geometry
        cannot tell a turn made to follow the planned route from one made to avoid
        the other vessel, and in a channel the two are constantly confused: enforcing
        it stopped the ship from following its own route. Stand-on compliance is still
        measured, and ``enforce_stand_on`` turns enforcement on for a comparison run.
        """
        return any(colregs.violates(action, encounter, nav, self.targets[key],
                                    safe_cpa=self.safe_cpa, min_turn=self.min_turn)
                   for key, encounter in self._duties)

    def enforced_duties(self):
        """The encounters whose rules are enforced this step, most urgent first.

        Urgency is time to closest approach, then range: seamanship deals with the
        closest quarters first. In a channel several vessels are in sight at once and
        their duties can conflict, so ``predict`` drops them from the end of this list
        until something safe complies with the rest. Stand-on duties are included
        only with ``enforce_stand_on``; every encounter is still measured.
        """
        duties = [(key, e) for key, e in self.encounters.items()
                  if self.enforce_stand_on or e.role == colregs.GIVE_WAY]
        duties.sort(key=lambda item: (max(item[1].tcpa, 0.0), item[1].distance))
        return duties if self.multi_duty else duties[:1]

    def urgent_encounter(self):
        """(target, encounter) closest in time to its CPA, whatever the role, or None."""
        if not self.encounters:
            return None
        key, encounter = min(self.encounters.items(),
                             key=lambda item: (max(item[1].tcpa, 0.0), item[1].distance))
        return self.targets[key], encounter

    # -- selection ------------------------------------------------------------

    def emergency_action(self, nav, candidates, others=()):
        """SARL's evasive action, turned the way the rules point when that costs little.

        The ranking is SARL's: no contact within the step first, then the most room
        with the velocity held. Among the options within ``emergency_tolerance`` of
        that room, the one ``colregs.emergency_preferred`` accepts for the most urgent
        encounter wins. Stand-on encounters count as well: in extremis Rule 17(b) lets
        the stand-on vessel act, and 17(c) still says which way not to.
        """
        urgent = self.urgent_encounter()
        if urgent is None:
            return super().emergency_action(nav, candidates, others)
        target, encounter = urgent
        held = {c.index: self.held_clearance(nav, c, others) for c in candidates}
        clear_now = [c for c in candidates if not isinstance(c.info, Collision)] or candidates

        def rank(c):
            return held[c.index], -c.velocity_change

        best = max(clear_now, key=rank)
        near = [c for c in clear_now if held[c.index] >= held[best.index] - self.emergency_tolerance]
        ruled = [c for c in near if colregs.emergency_preferred(c.action, encounter, nav, target,
                                                                 min_turn=self.min_turn)]
        choice = max(ruled, key=rank) if ruled else best
        if choice is not best and not self._apply_rules:
            self.rule_steered += 1  # counted on the pass whose action is executed
        return choice.action

    def check_action(self, index, nav, action, humans, next_humans):
        """SARL's swept safety check, plus the rules while ``_apply_rules`` is on."""
        candidate = super().check_action(index, nav, action, humans, next_humans)
        self._rule_free_safe += candidate.safe
        if not candidate.safe or not self._apply_rules or not self._duties:
            return candidate
        if self.breaks_rules(nav, action):
            self.rule_filtered += 1
            return candidate._replace(safe=False)
        return candidate

    def predict(self, state):
        self.encounters = self.update_encounters(state)
        duties = self.enforced_duties()
        try:
            # Every duty first; while that leaves nothing safe, drop the least urgent.
            for count in range(len(duties), 0, -1):
                action = self._select(state, duties[:count])
                if not self.emergency:
                    self.rule_relaxed += count < len(duties)
                    return action
                if not self._rule_free_safe:
                    break  # nothing is safe even without rules: relaxing cannot help
            # Obeying even the most urgent rule leaves nothing safe to do, so the rules
            # give way to safety: select from the candidates that only have to be safe.
            self.rule_yielded += bool(duties) and self._rule_free_safe > 0
            return self._select(state, [], final=True)
        finally:
            self._duties = []
            self._apply_rules = True

    def _select(self, state, duties, final=False):
        """One pass of SARL's selection with ``duties`` enforced.

        ``final`` marks the pass without rules, whose action is executed whatever it
        is; ``emergency_action`` counts ``rule_steered`` on that pass only.
        """
        self._duties = duties
        self._apply_rules = not final
        self._rule_free_safe = 0
        self.rule_filtered = 0
        return super().predict(state)
