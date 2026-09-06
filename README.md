# CrowdNav

## Setup
1. Install necessary library for the system
```
sudo apt update && sudo apt install cmake build-essential ffmpeg
```

2. Create Anconda environment
```terminal
conda create -n crowdnav python=3.10
```

3. Install crowd_sim and crowd_nav into pip
```
pip install -r requirements.txt
```

4. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library 
```
unzip Python-RVO2-main.zip
cd Python-RVO2-main
python setup.py build && python setup.py install
cd ..
pip install -e .
```

## Getting Started
1. Train a policy.
```
python train.py --policy sarl --gpu
```
2. Visualize a test case.
```
python test.py \
               --policy sarl \
               --model_dir data/output-trained \
               --phase test \
               --visualize \
               --test_case 0 \
               --video_file data/output-trained/test_case_0.mp4 \
               --static_map ../static_maps/land_shapes_sf_crop.png
```
```
python test.py \
               --policy sarl \
               --model_dir data/output-trained \
               --phase test \
               --visualize \
               --test_case 0 \
               --video_file data/output-trained/test_case_1.mp4 \
               --static_map ../static_maps/land_shapes_ht_crop.png
```
## Collision checks and evaluation (metrics v2)

Prediction and execution use the same swept-motion geometry: circles for moving
agents, filled rectangles for land, and the map extent for boundaries. Contact
at zero clearance is allowed; negative clearance is a collision. Static circle
features remain in the SARL input for checkpoint compatibility. The optional
`ObservableState.is_static` flag is metadata and adds no network features.

`query_env=false` retains constant-velocity prediction from observations. It does
not know another agent's next ORCA action. `query_env=true` previews that action
without changing the episode clock, waypoints, policy simulator or history.
Both paths use configured rewards and omit future value at terminal states.
Local goals shape reward; only the final route goal terminates navigation.

## Local goals and the SARL action pipeline

`crowd_nav/utils/local_goal.py` replaces the per-frame nearest-point search.
The tracker keeps a progress index along the dense route and only searches
forward of it (never backwards, and only to points the ship can see), so the
goal cannot jump to a later leg of a winding route. The look-ahead is
`--local_goal_dist` on open water and `--monitor_dist` while a moving ship is
closer than that; monitoring ends only beyond `--monitor_exit_dist` (default
1.3 x monitor distance), and the look-ahead moves between the two values at
`--lookahead_rate` m/s instead of switching. A candidate goal is accepted only
if the straight segment from the ship keeps `--shore_clearance` (plus the
radius) from land and the map edge; otherwise the goal walks back towards the
ship, and if nothing is visible the previous goal is kept. The destination
itself only has to fit the ship.

`SARL.predict` builds its candidate actions from the current velocity using the
`[motion]` section of `policy.config`: speeds within `max_accel` / `max_decel`
per step, headings within `max_turn_rate` per step (`heading_samples` around
the current heading, the current velocity always included, all directions when
at rest below `stop_speed`). Every candidate is swept over the next step with
the shared collision geometry and must keep `safety_margin_static` from land
and `safety_margin_dynamic` from the predicted ships; with `braking_check` it
must also be able to stop at `max_decel` afterwards without violating those
margins, and the stopped ship must stay clear of the extrapolated ships for
`safety_horizon` seconds, so being stopped is never assumed safe. Only safe
candidates are scored by the value network, and the score is
reduced by `continuity_weight` per radian of heading change so close scores
keep the current course; a ship at rest and the stop action pay no penalty.
Reaching a local goal does not force a stop: the stop action is a candidate
like any other and must pass the same check. When no candidate is safe the
policy returns a constrained deceleration: among the non-accelerating
candidates (or all of them when at rest) the one with the largest swept
clearance, so stopping into an oncoming ship is not chosen. The action is
executed exactly as scored. `constrained = false` restores the fixed 81-action
set; older configs without these sections use the defaults. `evaluate.py`
records the fraction of steps that used the fallback in the `emergency_frac`
column. Recorded smoke, sweep and batch results are in
[validation/local-goal-actions/README.md](validation/local-goal-actions/README.md).

Run from the repository root after installing the project dependencies:

```sh
MPLBACKEND=Agg python -m pytest tests -q

MPLBACKEND=Agg python -m crowd_nav.evaluate \
  --policy sarl --model_dir crowd_nav/data/output-trained \
  --static_map static_maps/land_shapes_sf_crop.png static_maps/land_shapes_ht_crop.png \
  --episodes 1 --densities 0,4 --seed 0 --max_steps 500 --out sarl_results.csv
```

Use `--policy orca` with the same arguments for the baseline. Install `pytest` in
the project environment if needed. `evaluate.py` defaults to 500 steps;
`test.py` accepts `--max_steps` and defaults to 2000. These map entry points set
the episode budget to `max_steps * time_step`; training keeps its configured
budget. Observation refreshes and previews do not advance time.

CSV version 2 keeps existing columns and adds `requested_n_dynamic`,
`collision_kind` and `metrics_version`. `n_dynamic` is the actual placed count.
Generation failures are recorded as `scenario_failed` with empty motion metrics
and excluded from executed-episode rates. Underfilled scenes are reported and
grouped by actual count. Collision outcomes distinguish `collision_dynamic`,
`collision_static` and `collision_both`; both means each type occurred somewhere
within the same executed step, not necessarily at the same instant.

Minimum gaps cover the initial state and every executed segment. Static gap is
the distance to filled land minus the agent radius; it is not a penetration-depth
estimate once the center enters land. `danger_frac` includes collision steps.
`planned_dist` includes the full global route, while traveled distance stops
within the goal radius, so successful detour ratios can be slightly below one.
Version 1 and version 2 metrics must not be pooled as equivalent measurements.

Recorded smoke results and limitations are in
[validation/collision-v2/README.md](validation/collision-v2/README.md).
