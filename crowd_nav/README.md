# Evaluation and COLREGs

What each file is for. How to run things is in [WORKFLOW.md](../WORKFLOW.md); everything
runs through `python -m crowd_nav.exp`, and the experiments are the rows of `experiments.txt`.

## Model folders

| folder | what it is | used by |
| --- | --- | --- |
| `data/output-sarl` | the retrained SARL | the main experiments and the GUI |
| `data/output-reward_design` | trained with `discomfort_dist = 0.4` | `sarl_reward_dsgn` |
| `data/output-domain_rdm` | trained with random ship sizes and speeds | the `sarl_domain_rdm` rows |
| `data/output-trained` | the first model | the tests |

The settings each training started from are in `configs/sarl`, `configs/reward_design` and
`configs/domain_rdm`.

## Where results go

```
results-v2/
  NAME_sf.csv, _ht, _sg        one row per trip: outcome, time, distance, clearances
  NAME.json                    settings, commit and weights of that run
  compliance/NAME_sf.csv, ...  one row per trip: encounters, which rule, obeyed or not
  colregs_report.md            the table
  videos/NAME/*.mp4
```

## COLREGs

| file | what it is |
| --- | --- |
| `policy/colregs.py` | the rules: who gives way, who stands on, when a meeting ends |
| `policy/sarl_colregs.py` | `SARLColregs` - drops rule-breaking actions before SARL ranks what is left. No retraining |
| `policy/test_colregs.py` | 27 unit tests for the classifier. No simulator, no torch |

Run the tests any time:

```bash
python -m crowd_nav.policy.test_colregs
```

The policy is registered as `sarl_colregs`, so it works anywhere a policy name is taken,
including `test.py`. Note `test.py` runs a single trip with a hardcoded list of ships and
writes no metrics; all reported numbers come from `evaluate.py` via `crowd_nav.exp run`.
