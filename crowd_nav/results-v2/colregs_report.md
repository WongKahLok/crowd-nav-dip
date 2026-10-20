# COLREGs result: SARL without and with the rule filter

Made on 2026-10-14 by `python -m crowd_nav.exp colregs`. The two policies of a pair sail the same
trips (same map, number of ships and seed). The other ships can see ours.

## SF

| ships | experiment    | trips | collision rate  | success rate      | give-way compliance |
|-------|---------------|-------|-----------------|-------------------|---------------------|
| 8     | sarl          | 100   | 2.0% (2 of 100) | 95.0% (95 of 100) | 1.1% (2 of 183)     |
| 8     | colregs       | 100   | 2.0% (2 of 100) | 89.0% (89 of 100) | 39.2% (82 of 209)   |
| 12    | sarl_dense    | 100   | 7.0% (7 of 100) | 87.0% (87 of 100) | 3.6% (10 of 280)    |
| 12    | colregs_dense | 100   | 5.0% (5 of 100) | 83.0% (83 of 100) | 32.6% (98 of 301)   |

## HT

| ships | experiment    | trips | collision rate    | success rate      | give-way compliance |
|-------|---------------|-------|-------------------|-------------------|---------------------|
| 8     | sarl          | 100   | 0.0% (0 of 100)   | 92.0% (92 of 100) | 2.6% (4 of 155)     |
| 8     | colregs       | 100   | 1.0% (1 of 100)   | 93.0% (93 of 100) | 46.4% (70 of 151)   |
| 16    | sarl_dense    | 100   | 10.0% (10 of 100) | 83.0% (83 of 100) | 4.7% (15 of 318)    |
| 16    | colregs_dense | 100   | 11.0% (11 of 100) | 79.0% (79 of 100) | 29.4% (89 of 303)   |
| 24    | sarl_dense    | 100   | 19.0% (19 of 100) | 71.0% (71 of 100) | 2.4% (12 of 492)    |
| 24    | colregs_dense | 100   | 19.0% (19 of 100) | 72.0% (72 of 100) | 24.8% (121 of 488)  |

## Reading the table

- Collision rate: the share of trips that ended in a collision, with a ship or with land. Success
  rate: the share that reached the goal. The rest ran out of time.
- Give-way compliance: of the encounters in which our ship had to give way, the share sailed by the
  rules, meaning no step of the encounter broke its rule. It counts encounters, not trips.
- With only 100 trips per row, a difference of a few points between two rows is within what other
  trips would change.
- A ship counts as a risk of collision within 10 m and 20 s; passing 1 m clear is safe.

## Where the numbers come from

- `sarl`: run 2026-10-06 21:50:17, commit 76fe40a plus uncommitted changes, settings `--env_set
  robot.visible=true`
- `colregs`: run 2026-10-06 22:09:22, commit 76fe40a plus uncommitted changes, settings `--env_set
  robot.visible=true`
- `sarl_dense`: run 2026-10-06 22:29:34, commit 76fe40a plus uncommitted changes, settings
  `--env_set robot.visible=true`
- `colregs_dense`: run 2026-10-06 23:02:25, commit 76fe40a plus uncommitted changes, settings
  `--env_set robot.visible=true`
