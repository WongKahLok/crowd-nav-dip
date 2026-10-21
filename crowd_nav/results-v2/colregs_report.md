# COLREGs result: SARL without and with the rule filter

Made on 2026-10-22 by `python -m crowd_nav.exp colregs`. The two policies of a pair sail the same
trips (same map, number of ships and seed). The other ships can see ours.

## SF

| ships | experiment    | trips | collision rate  | success rate        | give-way compliance |
|-------|---------------|-------|-----------------|---------------------|---------------------|
| 8     | sarl          | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 2.1% (4 of 190)     |
| 8     | colregs       | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 57.7% (120 of 208)  |
| 12    | sarl_dense    | 100   | 0.0% (0 of 100) | 99.0% (99 of 100)   | 3.0% (9 of 300)     |
| 12    | colregs_dense | 100   | 1.0% (1 of 100) | 99.0% (99 of 100)   | 55.4% (184 of 332)  |

## HT

| ships | experiment    | trips | collision rate  | success rate      | give-way compliance |
|-------|---------------|-------|-----------------|-------------------|---------------------|
| 8     | sarl          | 100   | 0.0% (0 of 100) | 99.0% (99 of 100) | 3.7% (6 of 161)     |
| 8     | colregs       | 100   | 0.0% (0 of 100) | 99.0% (99 of 100) | 59.2% (87 of 147)   |
| 16    | sarl_dense    | 100   | 0.0% (0 of 100) | 99.0% (99 of 100) | 3.9% (12 of 304)    |
| 16    | colregs_dense | 100   | 1.0% (1 of 100) | 99.0% (99 of 100) | 50.3% (178 of 354)  |
| 24    | sarl_dense    | 100   | 2.0% (2 of 100) | 98.0% (98 of 100) | 3.2% (17 of 525)    |
| 24    | colregs_dense | 100   | 2.0% (2 of 100) | 96.0% (96 of 100) | 48.9% (278 of 569)  |

## SG

| ships | experiment    | trips | collision rate  | success rate        | give-way compliance |
|-------|---------------|-------|-----------------|---------------------|---------------------|
| 8     | sarl          | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 2.8% (4 of 141)     |
| 8     | colregs       | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 68.4% (106 of 155)  |
| 12    | sarl_dense    | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 3.6% (6 of 166)     |
| 12    | colregs_dense | 100   | 0.0% (0 of 100) | 100.0% (100 of 100) | 64.1% (109 of 170)  |

## Reading the table

- Collision rate: the share of trips that ended in a collision, with a ship or with land. Success
  rate: the share that reached the goal. The rest ran out of time.
- Give-way compliance: of the encounters in which our ship had to give way, the share sailed by the
  rules, meaning no step of the encounter broke its rule. It counts encounters, not trips.
- With only 100 trips per row, a difference of a few points between two rows is within what other
  trips would change.
- A ship counts as a risk of collision within 10 m and 20 s; passing 1 m clear is safe.

## Where the numbers come from

- `sarl`: run 2026-10-21 20:42:07, commit 51093d0 plus uncommitted changes, settings `--env_set
  robot.visible=true`
- `colregs`: run 2026-10-21 20:53:07, commit 51093d0 plus uncommitted changes, settings `--env_set
  robot.visible=true`
- `sarl_dense`: run 2026-10-21 21:04:41, commit 51093d0 plus uncommitted changes, settings
  `--env_set robot.visible=true`
- `colregs_dense`: run 2026-10-21 21:26:09, commit 51093d0 plus uncommitted changes, settings
  `--env_set robot.visible=true`
