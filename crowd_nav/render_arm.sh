#!/usr/bin/env bash
# Render example episodes of a finished arm as mp4 videos, one per kind of outcome.
#
# Use the same settings as the run_arm.sh call that produced the arm, with
# render_arm.sh in its place (density settings are not needed):
#
#   ./crowd_nav/render_arm.sh b4_theta_sarl
#   POLICY=orca ./crowd_nav/render_arm.sh b1_orca_noglobal --no_global_plan
#   MODEL_DIR=crowd_nav/data/output-nearpaper ./crowd_nav/render_arm.sh filter_off
#
# Render options can follow the flags:
#   --dry_run                     list the chosen episodes, render nothing
#   --types timeout collision_static
#   --per_type 3                  up to three episodes of each kind
#   --maps ht   --ships 16
#   --episodes ht:16:2 sf:8:36    exactly these episodes (map:ships:seed)
#
# Videos go to crowd_nav/results-v2/videos/<name>/. Run from the repository root with
# the crowdnav environment active. See crowd_nav/render_samples.py for the kinds.
set -uo pipefail

NAME=${1:?usage: render_arm.sh NAME [same evaluate.py flags as the run] [render options]}
shift

POLICY=${POLICY:-sarl}
MODEL_DIR=${MODEL_DIR:-crowd_nav/data/output-trained}
MAX_STEPS=${MAX_STEPS:-500}
OUT=${OUT:-crowd_nav/results-v2}

MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python -m crowd_nav.render_samples \
    --arm "$NAME" --results "$OUT" --policy "$POLICY" --model_dir "$MODEL_DIR" \
    --max_steps "$MAX_STEPS" "$@"
