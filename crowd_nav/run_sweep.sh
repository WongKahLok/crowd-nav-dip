#!/usr/bin/env bash
# Phase 3 baseline sweep: the four baselines of TODO.md over a density sweep,
# every configuration on identical seeds so the results are directly comparable.
#
#   ./run_sweep.sh [out_dir] [episodes] [densities] [map]
#
# Run from crowd_nav/ with the crowdnav env active (config paths are relative).
set -e

OUT=${1:-results}
EPISODES=${2:-20}
DENSITIES=${3:-4,8,16,32}
MAP=${4:-../static_maps/land_shapes_sf_crop.png}
PYTHON=${PYTHON:-python}

mkdir -p "$OUT"
COMMON="--model_dir data/output-trained --static_map $MAP --episodes $EPISODES --densities $DENSITIES"

echo "=== B1: ORCA alone, no global planner ==="
$PYTHON evaluate.py --policy orca --no_global_plan $COMMON --out "$OUT/b1_orca_noglobal.csv"

echo "=== B2: Theta* + pure pursuit (no avoidance) ==="
$PYTHON evaluate.py --policy linear $COMMON --out "$OUT/b2_theta_linear.csv"

echo "=== B3: Theta* + ORCA (the classical hierarchy to beat) ==="
$PYTHON evaluate.py --policy orca $COMMON --out "$OUT/b3_theta_orca.csv"

echo "=== B4: Theta* + SARL (this system) ==="
$PYTHON evaluate.py --policy sarl $COMMON --out "$OUT/b4_theta_sarl.csv"

echo "=== sweep complete: CSVs in $OUT/ ==="
