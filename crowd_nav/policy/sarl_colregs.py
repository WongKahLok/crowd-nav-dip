"""SARL with the COLREGs rules applied when an action is chosen.

The value network is untouched. SARL already drops the candidate actions that fail its
safety check; this policy also drops the ones that break the rule in force, and the
network ranks what is left. If nothing is both safe and rule-following, the rule gives
way and SARL chooses among the safe ones. Nothing is retrained.

Thresholds can be set in a [colregs] section of policy.config; without it the defaults
below are used.
"""
import logging

from crowd_nav.policy import colregs
from crowd_nav.policy.sarl import SARL

THRESHOLDS = {'safe_cpa': 1.0,                    # closest approach that counts as clear, m
              'max_range': 10.0,                  # ships further away are not a risk yet, m
              'max_tcpa': 20.0,                   # nor are approaches further off than this, s
              'clear_range': colregs.CLEAR_RANGE,  # an encounter ends beyond this while opening, m
              'min_turn': 5.0}                    # a smaller turn does not count as a turn, deg


def rule_thresholds(config=None):
    """The thresholds of a policy config: its [colregs] section over the defaults.

    The compliance check uses the same values, so what is measured is what was enforced.
    """
    if config is None:
        return dict(THRESHOLDS)
    return {key: config.getfloat('colregs', key, fallback=value)
            for key, value in THRESHOLDS.items()}


class SARLColregs(SARL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL-COLREGs'
        self.safe_cpa = THRESHOLDS['safe_cpa']
        self.max_range = THRESHOLDS['max_range']
        self.max_tcpa = THRESHOLDS['max_tcpa']
        self.clear_range = THRESHOLDS['clear_range']
        self.min_turn = THRESHOLDS['min_turn']
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
        for key, value in rule_thresholds(config).items():
            setattr(self, key, value)
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
        # land cells are not ships
        self.targets = [h for h in state.human_states if not getattr(h, 'is_static', False)]
        return self.tracker.update(state.self_state, self.targets)

    def breaks_rules(self, nav, action):
        """True when this action breaks the rule being enforced.

        Only give-way duties (Rules 13 to 16) are enforced. Stand-on (Rule 17) is measured
        but not enforced: a turn to follow the route looks like a turn to avoid a ship,
        and enforcing it stopped the ship following its route. ``enforce_stand_on`` turns
        it on.
        """
        duty = self.enforced_duty()
        if duty is None:
            return False
        key, encounter = duty
        return colregs.violates(action, encounter, nav, self.targets[key],
                                safe_cpa=self.safe_cpa, min_turn=self.min_turn)

    def enforced_duty(self):
        """The one encounter whose rule is enforced this step, or None.

        Several duties at once can leave no action, so only the most urgent one (least
        time to closest approach) is enforced. The others are still measured.
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
