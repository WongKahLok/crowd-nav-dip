# COLREGs result: SARL without and with the rule filter

Made on 2026-11-02 by `python -m crowd_nav.exp colregs`. The two policies of a pair sail the same
trips (same map, number of ships and seed). The other ships can see ours.

## SF

| ships | experiment    | trips | collision rate  | success rate        | give-way compliance |
|-------|---------------|-------|-----------------|---------------------|---------------------|
| 8     | sarl          | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 2.3% (4 of 176)     |
| 8     | colregs       | 100   | 1.0% (1 of 100) | 99.0% (99 of 100)   | 63.1% (125 of 198)  |
| 12    | sarl_dense    | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 1.7% (5 of 291)     |
| 12    | colregs_dense | 100   | 1.0% (1 of 100) | 98.0% (98 of 100)   | 57.0% (174 of 305)  |

## HT

| ships | experiment    | trips | collision rate  | success rate        | give-way compliance |
|-------|---------------|-------|-----------------|---------------------|---------------------|
| 8     | sarl          | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 1.2% (2 of 165)     |
| 8     | colregs       | 100   | 1.0% (1 of 100) | 99.0% (99 of 100)   | 54.9% (89 of 162)   |
| 16    | sarl_dense    | 100   | 1.0% (1 of 100) | 99.0% (99 of 100)   | 1.9% (6 of 315)     |
| 16    | colregs_dense | 100   | 2.0% (2 of 100) | 98.0% (98 of 100)   | 52.1% (176 of 338)  |
| 24    | sarl_dense    | 100   | 1.0% (1 of 100) | 99.0% (99 of 100)   | 4.5% (22 of 494)    |
| 24    | colregs_dense | 100   | 2.0% (2 of 100) | 98.0% (98 of 100)   | 50.8% (281 of 553)  |

## SG

| ships | experiment    | trips | collision rate  | success rate        | give-way compliance |
|-------|---------------|-------|-----------------|---------------------|---------------------|
| 8     | sarl          | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 5.6% (8 of 144)     |
| 8     | colregs       | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 71.1% (106 of 149)  |
| 12    | sarl_dense    | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 2.4% (4 of 169)     |
| 12    | colregs_dense | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 72.7% (136 of 187)  |

## Reading the table

- Collision rate: the share of trips that ended in a collision, with a ship or with land. Success
  rate: the share that reached the goal. The rest ran out of time.
- Give-way compliance: of the encounters in which our ship had to give way, the share sailed by the
  rules, meaning no step of the encounter broke its rule. It counts encounters, not trips.
- With only 100 trips per row, a difference of a few points between two rows is within what other
  trips would change.
- A ship counts as a risk of collision within 10 m and 20 s; passing 1 m clear is safe.

## Where the numbers come from

- `sarl`: run 2026-11-01 17:45:04, commit a459388 plus uncommitted changes, settings `--env_set
  robot.visible=true`
- `colregs`: run 2026-11-01 17:57:27, commit a459388 plus uncommitted changes, settings `--env_set
  robot.visible=true`
- `sarl_dense`: run 2026-11-01 18:09:46, commit a459388 plus uncommitted changes, settings
  `--env_set robot.visible=true`
- `colregs_dense`: run 2026-11-01 18:39:58, commit a459388 plus uncommitted changes, settings
  `--env_set robot.visible=true`
