# Workflow

The jobs you will do with this project, in the order you will need them. Run everything
from the repository root with the environment active (`conda activate crowdnav`). Setup is
in the [README](README.md).

## 1. See it run

```bash
python -m crowd_nav.exp demo --compare
```

Films one trip twice, side by side: SARL without the rules and with them. It prints the
path of the mp4 and takes a few minutes. Nothing has to be run first. Another trip:
`--map sf --ships 12 --seed 5`.

## 2. Get the results table

```bash
python -m crowd_nav.exp colregs
```

Runs the experiments that have no results yet, then prints the table and saves it as
`crowd_nav/results-v2/colregs_report.md`. With results on file it takes seconds. From
nothing it takes about two hours on 16 cores, so keep the laptop awake and plugged in: when
it sleeps, the run sleeps with it.

## 3. Run one experiment

```bash
python -m crowd_nav.exp run colregs
```

About 20 minutes for the 8-ship experiments and 35 to 40 for the dense ones. It writes one
CSV per map, the rule check of every trip, and a `NAME.json` with the settings, the commit
and the weights.

- It will not replace results that exist. Add `--again` when you mean to.
- `--trips 10` is a quick trial; give it `--results some_folder` too, or the short result
  is written under the real name.
- Ctrl-C stops it. Nothing is written until an experiment has finished.

## 4. Look at the failures

```bash
python -m crowd_nav.exp video colregs --types collision_dynamic collision_static timeout
python -m crowd_nav.exp video colregs ht:8:60
```

The first films one trip of each kind that exists in the results, the second exactly that
trip. A trip is `map:ships:seed`. The kinds are `success_typical`, `success_closest`,
`longest_detour`, `most_fallback`, `collision_dynamic`, `collision_static`,
`collision_both` and `timeout`. `--dry_run` lists the picks and films nothing.

## 5. Try an idea

1. Add a row to [crowd_nav/experiments.txt](crowd_nav/experiments.txt). For example, to
   enforce the stand-on rule as well:

   ```text
   colregs_standon  sarl_colregs  output-trained  8  8  --env_set robot.visible=true --policy_set colregs.enforce_stand_on=true
   ```

2. Run it: `python -m crowd_nav.exp run colregs_standon`
3. To put it in the table, add a pair to `COLREGS` at the top of
   [crowd_nav/exp.py](crowd_nav/exp.py): `('colregs', 'colregs_standon')`.

Every experiment sails seeds 0 to 99, so any two face the same trips.

## 6. After you change the code

```bash
python -m crowd_nav.policy.test_colregs     # the rules, a second
python -m pytest tests -q                   # the rest (pip install pytest first)
```

Results on file belong to the commit in their `NAME.json`. If your change can alter them,
run those experiments again with `--again`. If PyQt5 is installed, add
`--ignore=tests/test_gui.py` to the second command.

## Where things are

```text
crowd_nav/results-v2/
  NAME_sf.csv, NAME_ht.csv   one row per trip: outcome, time, distance, clearances
  NAME.json                  settings, commit and weights of that run
  compliance/NAME_*.csv      one row per trip: encounters and whether the rules were kept
  colregs_report.md          the table
  videos/                    the videos
```
