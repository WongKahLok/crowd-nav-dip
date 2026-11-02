# Workflow

The jobs you will do with this project, in the order you will need them. Run everything
from the repository root with the environment active (`conda activate crowdnav`). Setup is
in the [README](README.md).

## 1. See it run

```bash
python -m crowd_nav.exp demo --compare
```

Films one trip twice, side by side: SARL without the rules and with them. It prints the
path of the mp4 and takes about a minute. Nothing has to be run first. Another trip:
`--map sf --ships 12 --seed 5`. The maps are `sf`, `ht` and `sg`.

## 2. Get the results table

```bash
python -m crowd_nav.exp colregs
```

Runs the experiments that have no results yet, then prints the table and saves it as
`crowd_nav/results-v2/colregs_report.md`. With results on file it takes seconds. From
nothing it takes about an hour and a half on 16 cores, so keep the laptop awake and plugged
in: when it sleeps, the run sleeps with it.

## 3. Run one experiment

```bash
python -m crowd_nav.exp run colregs
```

The experiments are the rows of [crowd_nav/experiments.txt](crowd_nav/experiments.txt).
About 10 minutes for the 8-ship ones, 30 to 40 for the dense ones and 50 for
`visible_sarl`. It writes one CSV per map, the rule check of every trip, and a `NAME.json`
with the settings, the commit and the weights.

- It will not replace results that exist. Add `--again` when you mean to.
- `--missing` instead of a name runs every row that has no results yet. All 16 rows take
  about 4 hours from nothing.
- `--trips 10` is a quick trial; give it `--results some_folder` too, or the short result
  is written under the real name.
- Ctrl-C stops it. Nothing is written until an experiment has finished.

## 4. Look at the failures

```bash
python -m crowd_nav.exp video colregs --types collision_dynamic collision_static timeout
python -m crowd_nav.exp video colregs ht:8:60
```

The first films one trip of each kind that exists in the results, on each map, the second
exactly that trip. A trip is `map:ships:seed`. The kinds are `success_typical`,
`success_closest`, `longest_detour`, `most_fallback`, `collision_dynamic`,
`collision_static`, `collision_both` and `timeout`. `--dry_run` lists the picks and films
nothing.

## 5. Try an idea

1. Add a row to [crowd_nav/experiments.txt](crowd_nav/experiments.txt). For example, to
   enforce the stand-on rule as well:

   ```text
   colregs_standon  sarl_colregs  output-sarl  8  8  8  --env_set robot.visible=true --policy_set colregs.enforce_stand_on=true
   ```

   The three numbers are the ships on SF, HT and SG. `-` leaves a map out.

2. Run it: `python -m crowd_nav.exp run colregs_standon`
3. To put it in the table, add a pair to `COLREGS` at the top of
   [crowd_nav/exp.py](crowd_nav/exp.py): `('colregs', 'colregs_standon')`. Then run
   `python -m crowd_nav.exp colregs`. Any two experiments can be a pair, for example
   `('sarl_base', 'sarl_reward_dsgn')`.

Every experiment sails seeds 0 to 99, so any two face the same trips.

## 6. Set up a trip by hand

```bash
python -m crowd_nav.gui             # SARL
python -m crowd_nav.gui --colregs   # SARL + COLREGs
```

Opens a window where you click the start and goal, add ships, run the trip and export a
video. It uses `output-sarl` unless you give `--model_dir`. The steps are under Scenario
GUI in the [README](README.md).

## 7. Train a new model

```bash
OMP_NUM_THREADS=1 python crowd_nav/train.py --policy sarl \
  --env_config crowd_nav/configs/sarl/env.config \
  --policy_config crowd_nav/configs/sarl/policy.config \
  --train_config crowd_nav/configs/train.config \
  --output_dir crowd_nav/data/output-mine
```

10,000 episodes, about 13 hours on one core. The weights are saved every 1,000 episodes.

- Use `configs/sarl`, not the plain `configs` files: it has `query_env = false`, which is
  how the experiments test. `configs/reward_design` and `configs/domain_rdm` made the other
  two models.
- Give a new folder name. If the folder exists it asks, and `y` deletes the old model.
- Leave out `--gpu`. It was about twice as slow as the CPU here.

To test the model, put its folder name in the model column of a new row (job 5).

## 8. After you change the code

```bash
python -m crowd_nav.policy.test_colregs     # the rules, a second
python -m pytest tests -q                   # the rest, 10 seconds (pip install pytest first)
```

Results on file belong to the commit in their `NAME.json`. `local_changes: true` there means
the code also had changes that were not committed. If your change can alter the results,
run those experiments again with `--again`.

## Where things are

```text
crowd_nav/experiments.txt    the experiments
crowd_nav/configs/           the settings a training starts from
crowd_nav/data/output-*/     the trained models, output-sarl is the main one
crowd_nav/results-v2/
  NAME_sf.csv, _ht, _sg      one row per trip: outcome, time, distance, clearances
  NAME.json                  settings, commit and weights of that run
  compliance/NAME_*.csv      one row per trip: encounters and whether the rules were kept
  colregs_report.md          the table
  videos/NAME/               the videos of an experiment
  videos/demo/               the demo videos
```
