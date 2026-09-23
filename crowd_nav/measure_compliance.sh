#!/usr/bin/env bash
# Measure the COLREGs compliance of an arm that has already been evaluated.
#
# Use the same settings as the run_arm.sh call that produced the arm:
#
#   ./crowd_nav/measure_compliance.sh b4_theta_sarl
#   ./crowd_nav/measure_compliance.sh b4_theta_sarl --sample 50 --videos 3
#   POLICY=orca ./crowd_nav/measure_compliance.sh b3_theta_orca
#
# Options: --sample N (episodes per map, 0 for all), --videos N, --maps sf, --ships 8,
#          --safe_cpa / --max_range / --max_tcpa / --clear_range (the rule thresholds).
# Run from the repository root with the crowdnav environment active.
set -uo pipefail

NAME=${1:?usage: measure_compliance.sh NAME [same evaluate.py flags as the run] [options]}
shift

POLICY=${POLICY:-sarl}
MODEL_DIR=${MODEL_DIR:-crowd_nav/data/output-trained}
MAX_STEPS=${MAX_STEPS:-500}
OUT=${OUT:-crowd_nav/results-v2}

MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m crowd_nav.measure_compliance \
    --arm "$NAME" --results "$OUT" --policy "$POLICY" --model_dir "$MODEL_DIR" \
    --max_steps "$MAX_STEPS" "$@"
