# Collision and evaluation validation — 2026-09-05

The implementation shares swept collision geometry and transition outcomes across
training, map simulation and SARL prediction. Existing weights were retained.
These are metrics version 2 results; they are not comparable to version 1 as if
the geometry and measurement definitions were unchanged.

## Checks completed

- 59 pytest cases passed, covering circle/rectangle sweeps, tangency, initial
  overlap, corner collision, bounds, collisions outside observation range,
  configured rewards, terminal priority, episode time, observation refresh,
  waypoint and ORCA preview purity, repeated preview/execution agreement,
  executed/guarded human velocities, static observations in both prediction
  branches, empty inputs, terminal value masking, CSV outcomes and denominators.
- Both prediction modes ran the original five-human training environment with
  the existing SARL checkpoint. No optimizer or checkpoint write was performed.
- The demonstration CLI ran with `--max_steps 2` and exited with `Timeout`.
- Modified Python files passed syntax parsing; whitespace checks allow the
  repository's existing CRLF line endings (`git -c core.whitespace=cr-at-eol diff --check`).
- Both checkpoint files were byte-for-byte equal to their Git HEAD versions.

Runtime: macOS arm64, Python 3.12.7, NumPy 1.26.4, PyTorch 2.8.0 CPU,
Gym 0.15.7, SciPy 1.15.3, Matplotlib 3.10.7, pytest 9.1.1. RVO2 was built
from the bundled source in a temporary directory. The isolated environment was
`/private/tmp/crowdnav-collision-venv`; no global Python packages were changed.

## Map smoke results

One episode per map and density, seed 0, `time_step=0.25`, `max_steps=500`
(125 simulated seconds). SARL used the saved `query_env=false` configuration.
After finalizing ORCA state commits, all dynamic cases were rerun and produced
the same records. The no-human cases do not execute that code path.

| Policy | Map | Dynamic count | Outcome | Steps | Simulated seconds |
|---|---|---:|---|---:|---:|
| SARL | SF | 0 | timeout | 500 | 125.00 |
| SARL | SF | 4 | success | 204 | 51.00 |
| SARL | HT | 0 | timeout | 500 | 125.00 |
| SARL | HT | 4 | success | 300 | 75.00 |
| ORCA | SF | 0 | success | 100 | 25.00 |
| ORCA | SF | 4 | success | 99 | 24.75 |
| ORCA | HT | 0 | success | 134 | 33.50 |
| ORCA | HT | 4 | success | 134 | 33.50 |

Full records: [sarl.csv](sarl.csv), [orca.csv](orca.csv).
No robot collisions or scenario-generation failures occurred in these eight
cases. SARL's two timeouts are retained as navigation failures. Their causes
were not isolated in this change; local-goal continuity, empty-interaction value
estimation and map-domain training remain separate follow-up work. The HT
four-human cases had minimum dynamic separation above 6 m, so those cases do
not establish close-interaction avoidance performance.

This is a functional smoke test, not a statistical success-rate benchmark.
Large density sweeps, retraining, GPU execution, physical vessel dynamics and
video-render QA were not performed.

## Reproduce

From the repository root with the project dependencies and pytest installed:

```sh
MPLBACKEND=Agg python -m pytest tests -q

for policy in sarl orca; do
  MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m crowd_nav.evaluate \
    --policy "$policy" --model_dir crowd_nav/data/output-trained \
    --static_map static_maps/land_shapes_sf_crop.png static_maps/land_shapes_ht_crop.png \
    --episodes 1 --densities 0,4 --seed 0 --max_steps 500 \
    --out "/tmp/collision-v2-$policy.csv"
done
```

Checkpoint SHA-256:

- `rl_model.pth`: `f8240c0c73fe08597fbf53332efa1fb6877f7bc34b2ba512eb3eeb39e0b0529c`
- `il_model.pth`: `f058df9cc01064d89efab1af171290d435dcec467443d4e9b550a2e2ec69e0ac`
