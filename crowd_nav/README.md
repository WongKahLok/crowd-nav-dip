# Evaluation and COLREGs

How to run every experiment in the project, and what each file is for.

Everything runs through one script, `exp.sh`

## Quick start

```bash
./crowd_nav/exp.sh list                                      # what experiments exist
./crowd_nav/exp.sh run visible_colregs                       # run one (writes CSVs) (refer to experiment.txt to choose which you want to run)
./crowd_nav/exp.sh compare visible_sarl_dense visible_colregs_dense      # compare two model, success rates + p-values
./crowd_nav/exp.sh rules visible_colregs                     # COLREGs compliance
./crowd_nav/exp.sh video visible_colregs ht:8:60             # output mp4 of map:ships:trip
```


## Model folders

| folder | difference with data/output-trained | used by |
| --- | --- | --- |
| `output-visible` | `visible = true` in `env.config` | the COLREGs experiments |

They are derived, so they are git-ignored. Delete them freely; they rebuild in seconds.

## Where results go

```
results-v2/
  NAME_sf.csv, NAME_ht.csv     one row per trip: outcome, time, distance, clearances
  compliance/NAME_sf.csv, ...  one row per trip: encounters, which rule, obeyed or not
  videos/NAME/*.mp4
  parts/                       per-shard run logs (git-ignored); the receipts for a run
  extra/                       measured but not reported
```

## COLREGs

| file | what it is |
| --- | --- |
| `policy/colregs.py` | the rules: who gives way, who stands on, when a meeting ends |
| `policy/sarl_colregs.py` | `SARLColregs` - drops rule-breaking actions before SARL ranks what is left. No retraining |
| `policy/test_colregs.py` | 23 unit tests for the classifier. No simulator, no torch |

Run the tests any time:

```bash
PYTHONPATH=. python -m crowd_nav.policy.test_colregs
```

The policy is registered as `sarl_colregs`, so it works anywhere a policy name is taken,
including `test.py`. Note `test.py` runs a single trip with a hardcoded 6 ships and
writes no metrics; all reported numbers come from `evaluate.py` via `exp.sh run`.
