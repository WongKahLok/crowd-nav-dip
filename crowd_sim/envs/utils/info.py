class StepInfo:
    def __init__(self, min_sep_dynamic=float('inf'), min_sep_static=float('inf'),
                 collision_kind=None):
        self.min_sep_dynamic = min_sep_dynamic
        self.min_sep_static = min_sep_static
        self.collision_kind = collision_kind


class Timeout(StepInfo):
    def __str__(self):
        return 'Timeout'


class ReachGoal(StepInfo):
    def __str__(self):
        return 'Reaching goal'


class Danger(StepInfo):
    def __init__(self, min_dist, **kwargs):
        super().__init__(**kwargs)
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close'


class Collision(StepInfo):
    def __str__(self):
        return 'Collision'


class Nothing(StepInfo):
    def __str__(self):
        return ''
