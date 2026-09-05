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
