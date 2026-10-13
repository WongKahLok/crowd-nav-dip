"""SARL with the COLREGs steering rules applied at action-selection time.

The value network is untouched: this policy only narrows the set of actions it is
allowed to choose from. SARL already builds kinematically reachable candidates and
drops the ones that fail the swept safety check; this subclass drops, in addition,
the ones that would break the rule in force for any vessel currently in an encounter
(``crowd_nav.policy.colregs``). The value network then picks the best of what is left,
so nothing has to be retrained.

Order of preference, highest first:

1. safe and rule-compliant (give-way duties; see ``breaks_rules``),
2. safe (a rule is dropped when obeying it would leave nothing safe to do, which is
   how Rule 2 and Rule 17(b) are approximated: the rules never require a collision),
3. the evasive action SARL falls back on when nothing is safe.

``rule_yielded`` counts the steps that fell to level 2 and ``rule_filtered`` the
candidates removed at level 1, so the cost of the rules can be reported.

Thresholds come from an optional ``[colregs]`` section of policy.config; the defaults
match crowd_nav.policy.colregs, and a config without that section works unchanged.
"""
import logging

from crowd_nav.policy import colregs
from crowd_nav.policy.sarl import SARL


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
        # per-episode state and diagnostics
        self.tracker = None
        self.targets = []
        self.encounters = {}
        self.rule_yielded = 0
        self.rule_filtered = 0
        self._apply_rules = True
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
        self.name = 'SARL-COLREGs'
        logging.info('COLREGs filter: safe CPA %.2f m, range %.1f m, horizon %.0f s, '
                     'clear at %.1f m, alteration %.0f deg',
                     self.safe_cpa, self.max_range, self.max_tcpa, self.clear_range, self.min_turn)
        logging.info('COLREGs enforced for: %s',
                     'all roles' if self.enforce_stand_on else 'give-way duties only')

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
        duty = self.enforced_duty()
        if duty is None:
            return False
        key, encounter = duty
        return colregs.violates(action, encounter, nav, self.targets[key],
                                safe_cpa=self.safe_cpa, min_turn=self.min_turn)

    def enforced_duty(self):
        """The single encounter whose rule is enforced this step, or None.

        Enforcing every encounter at once over-constrains the ship: in a channel
        several vessels are in sight at the same time and their duties conflict, so
        nothing is left that is both safe and compliant. Seamanship deals with the
        closest quarters first, so the duty enforced is the one with the least time
        to closest approach. The others are still measured.
        """
        duties = [(key, e) for key, e in self.encounters.items()
                  if self.enforce_stand_on or e.role == colregs.GIVE_WAY]
        if not duties:
            return None
        return min(duties, key=lambda item: (max(item[1].tcpa, 0.0), item[1].distance))

    # -- selection ------------------------------------------------------------

    def check_action(self, index, nav, action, humans, next_humans):
        """SARL's swept safety check, plus the rules while ``_apply_rules`` is on."""
        candidate = super().check_action(index, nav, action, humans, next_humans)
        if not candidate.safe or not self._apply_rules or not self.encounters:
            return candidate
        if self.breaks_rules(nav, action):
            self.rule_filtered += 1
            return candidate._replace(safe=False)
        return candidate

    def predict(self, state):
        self.encounters = self.update_encounters(state)
        self.rule_filtered = 0
        self._apply_rules = True
        action = super().predict(state)
        if self.encounters and self.emergency:
            # Obeying the rules would leave nothing safe to do, so they give way to
            # safety: re-select from the candidates that only have to be safe.
            self._apply_rules = False
            self.rule_yielded += 1
            action = super().predict(state)
        self._apply_rules = True
        return action
