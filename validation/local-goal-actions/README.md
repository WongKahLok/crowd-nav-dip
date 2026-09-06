# Local goals and constrained SARL actions — 2026-09-06

Follow-up to [collision-v2](../collision-v2/README.md). This change adds the
progress-tracking local goal (`crowd_nav/utils/local_goal.py`), the kinematically
constrained candidate actions, the swept safety filter with braking check and
safety horizon, the heading-continuity cost and the emergency fallback in
`SARL.predict`. The trained checkpoints were not modified. The `[motion]` and
`[safety]` sections in `policy.config` hold the parameters; `README.md` describes
the pipeline.

## Checks completed

- 82 pytest cases passed (`tests/test_local_goal.py`, `tests/test_sarl_actions.py`
  and the earlier suites). They cover progress that never moves backwards, the
  monitor/exit hysteresis and rate-limited look-ahead, corridor checks that walk
  the goal back before a blocked corner, the destination that only has to fit
  the ship, candidate sets inside the acceleration and turn limits, the safety
  margins, the braking check against land and moving ships, the stop action
  being unsafe in the path of an oncoming ship, the fallback that decelerates
  and dodges rather than parks, the heading-only continuity cost, and that the
  chosen action is executed unchanged.
- Runtime: macOS arm64, Python 3.12.7, NumPy 1.26.4, PyTorch 2.14.0 CPU,
  Gym 0.15.7, pytest in a throwaway venv; RVO2 built from the bundled source.

## Problems found and fixed during validation

1. The first version charged the continuity cost on the velocity change. With
   the empty-observation values of the checkpoint the gain from setting off is
   only about 0.003, so the penalty kept the ship at rest for the whole episode
   (all four smoke cases timed out). The cost is now charged per radian of
   heading change only; a ship at rest and the stop action pay nothing.
2. With the cost off the ship sailed the route but parked 0.4 m short of the
   destination: a land corner inside the 0.8 m shore corridor made the tracker
   hand out the point before the destination, and the policy's reach-goal
   shortcut stopped there unconditionally (in the crowded case a ship then ran
   into it). The destination now only has to fit the ship, and reaching a
   local goal no longer short-circuits to a stop.

A third problem showed up in the `test.py` demo on the SF map: a ship bearing
straight at the robot made every candidate except braking fail the braking
check, the value network slowed the robot to a halt in the ship's path and it
was hit while at rest (collision at step 73). The braking check now also
requires the stopped ship to stay clear of the extrapolated ships for
`safety_horizon` seconds (default 3 s), which rejects "slow down and wait" while
there is still speed to dodge with. The same scenario now ends with
`Reaching goal` at step 186 with a 0.31 m minimum gap and no fallback step.

## Smoke results (seed 0, one episode per map and density, 500 steps)

Same protocol as collision-v2. Rows: [sarl_smoke.csv](sarl_smoke.csv),
[orca_smoke.csv](orca_smoke.csv).

| Policy | Map | Ships | Outcome | Steps | Min dyn. sep (m) | Min static sep (m) | Danger frac | Fallback frac |
|---|---|---:|---|---:|---:|---:|---:|---:|
| SARL | SF | 0 | success (was timeout) | 104 (500) | - | 0.40 | 0.00 | 0.00 |
| SARL | SF | 4 | success | 102 (204) | 0.21 | 0.44 | 0.00 (0.12) | 0.00 |
| SARL | HT | 0 | success (was timeout) | 157 (500) | - | 0.41 | 0.00 | 0.00 |
| SARL | HT | 4 | success | 129 (300) | 6.35 | 0.41 | 0.00 | 0.00 |
| ORCA | SF | 0 | success | 100 | - | 0.60 | 0.00 | - |
| ORCA | SF | 4 | success | 101 (99) | 0.19 | 0.50 (0.01) | 0.02 (0.24) | - |
| ORCA | HT | 0 | success | 134 | - | 0.41 | 0.00 | - |
| ORCA | HT | 4 | success | 134 | 7.03 | 0.41 | 0.00 | - |

Values in parentheses are the collision-v2 record where it differs. ORCA only
uses the new local goal; its SF/4 case no longer grazes the coast because the
corridor check keeps the goal off the shore.

## Continuity weight sweep (before the safety horizon was added)

Heading reversals are sign changes of the per-step heading change while moving.

| Scenario | weight 0 | 0.005 | 0.01 (default) | 0.02 | 0.05 |
|---|---|---|---|---|---|
| SF, 0 ships: reversals / steps | 20 / 103 | 10 / 103 | 6 / 104 | 4 / 105 | 3 / 130 |
| SF, 4 ships: reversals / steps | 40 / 104 | 20 / 103 | 8 / 102 | 2 / 107 | 3 / 160 |
| HT, 4 ships: reversals / steps | 52 / 129 | 21 / 129 | 13 / 129 | 5 / 130 | 7 / 288 |

All runs reached the goal. 0.01 removes most of the flapping without costing
time; 0.05 makes the ship reluctant to turn.

## Demo CLI (`crowd_nav/test.py`, test case 0, 6 ships, 2000 step budget)

| Map | Outcome | Steps | Fallback steps |
|---|---|---:|---:|
| SF | Reaching goal (was collision at 73 before the safety horizon) | 186 | 0 |
| HT | Reaching goal | 253 | 5 |

## Batch (5 seeds, densities 4 and 8, 500 steps)

Rows: [sarl_batch.csv](sarl_batch.csv), [orca_batch.csv](orca_batch.csv).

| Policy | Map | Ships | Runs | Success | Collision | Timeout | Mean steps (successes) | Mean fallback frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SARL | HT | 4 | 5 | 5 | 0 | 0 | 142 | 0.004 |
| SARL | HT | 8 | 5 | 5 | 0 | 0 | 155 | 0.007 |
| SARL | SF | 4 | 5 | 5 | 0 | 0 | 147 | 0.001 |
| SARL | SF | 8 | 5 | 5 | 0 | 0 | 170 | 0.010 |
| ORCA | HT | 4 | 5 | 4 | 0 | 1 | 134 | - |
| ORCA | HT | 8 | 5 | 4 | 0 | 1 | 135 | - |
| ORCA | SF | 4 | 5 | 3 | 1 | 1 | 120 | - |
| ORCA | SF | 8 | 5 | 2 | 2 | 1 | 102 | - |

SARL kept every minimum dynamic gap at or above the 0.2 m safety margin
(`min_sep_dynamic` 0.200 to 6.35 m) with `danger_frac` 0 in all 20 runs; the
fallback acted in at most 3.4 % of the steps of an episode. ORCA's failures are
the invisible-robot cases (the other ships do not avoid it) and two scenarios
where it never reaches the goal, unchanged from its behaviour before this work.

## Limitations

Single seeds and a five-seed batch are a functional check, not a benchmark.
The value network was trained in the circle-crossing environment; its
preferences on the maps are weak (value spreads of 1e-3 between candidates on
open water), which is why the continuity cost has to be small. Other ships do
not see the robot (`robot.visible = false`), so a ship that turns towards the
robot after the constant-velocity extrapolation can still produce a collision;
`emergency_frac` in the CSV shows how often the fallback had to act. No
retraining, GPU runs or video QA were performed.

## Reproduce

```sh
MPLBACKEND=Agg python -m pytest tests -q

for policy in sarl orca; do
  MPLBACKEND=Agg OMP_NUM_THREADS=1 python -m crowd_nav.evaluate \
    --policy "$policy" --model_dir crowd_nav/data/output-trained \
    --static_map static_maps/land_shapes_sf_crop.png static_maps/land_shapes_ht_crop.png \
    --episodes 1 --densities 0,4 --seed 0 --max_steps 500 --out "${policy}_smoke.csv"
done

cd crowd_nav && MPLBACKEND=Agg python test.py --policy sarl --model_dir data/output-trained \
  --phase test --test_case 0 --static_map ../static_maps/land_shapes_sf_crop.png
```
