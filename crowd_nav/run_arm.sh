#!/usr/bin/env bash
# Run one evaluation configuration (an "arm") over both maps and write one CSV per map.
#
# Usage, from the repository root with the crowdnav environment active:
#
#   ./crowd_nav/run_arm.sh NAME [extra evaluate.py flags...]
#
#   ./crowd_nav/run_arm.sh b4_theta_sarl
#   ./crowd_nav/run_arm.sh lookahead_fixed1 --local_goal_dist 1 --monitor_dist 1
#   POLICY=orca ./crowd_nav/run_arm.sh b3_theta_orca
#   MODEL_DIR=crowd_nav/data/output-nearpaper ./crowd_nav/run_arm.sh filter_off
#
# Every arm uses seeds 0..EPISODES-1 on both maps, so any two arms can be compared
# seed by seed with compare_arms.py. Episodes are split over SHARDS processes.
#
# Settings (environment variables):
#   EPISODES=100    episodes per map per density
#   SHARDS=10       parallel processes per map
#   POLICY=sarl     sarl | orca | linear | cadrl | lstm_rl
#   MODEL_DIR=crowd_nav/data/output-trained
#   SF_DENSITIES=8  comma separated; SF water fits about 12 ships
#   HT_DENSITIES=8  comma separated; HT fits about 25
#   MAX_STEPS=500   episode budget (500 x 0.25 s = 125 s)
#   OUT=crowd_nav/results-v2
#   KEEP_PARTS=1    keep the per-shard CSVs; by default they are deleted once the
#                   merged CSV is verified to hold every row (their logs are kept)
set -uo pipefail

NAME=${1:?usage: run_arm.sh NAME [evaluate.py flags...]}
shift

EPISODES=${EPISODES:-100}
SHARDS=${SHARDS:-10}
POLICY=${POLICY:-sarl}
MODEL_DIR=${MODEL_DIR:-crowd_nav/data/output-trained}
SF_DENSITIES=${SF_DENSITIES:-8}
HT_DENSITIES=${HT_DENSITIES:-8}
MAX_STEPS=${MAX_STEPS:-500}
OUT=${OUT:-crowd_nav/results-v2}

if [ ! -f crowd_nav/evaluate.py ]; then
    echo "run this from the repository root (the folder holding crowd_nav/ and static_maps/)" >&2
    exit 1
fi

mkdir -p "$OUT/parts"
per=$(( (EPISODES + SHARDS - 1) / SHARDS ))
started=$(date +%s)
echo "arm '$NAME': policy=$POLICY episodes=$EPISODES model=$MODEL_DIR sf=$SF_DENSITIES ht=$HT_DENSITIES extra: $*"

for map in sf ht; do
    case $map in
        sf) densities=$SF_DENSITIES ;;
        ht) densities=$HT_DENSITIES ;;
    esac
    seed=0
    while [ "$seed" -lt "$EPISODES" ]; do
        n=$per
        [ $((seed + n)) -gt "$EPISODES" ] && n=$((EPISODES - seed))
        part="$OUT/parts/${NAME}_${map}_s${seed}"
        MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONPATH=. python -m crowd_nav.evaluate \
            --policy "$POLICY" --model_dir "$MODEL_DIR" \
            --static_map "static_maps/land_shapes_${map}_crop.png" \
            --episodes "$n" --seed "$seed" --densities "$densities" \
            --max_steps "$MAX_STEPS" --out "${part}.csv" "$@" > "${part}.log" 2>&1 &
        seed=$((seed + n))
    done
done
wait

failed=$(grep -l "Traceback" "$OUT/parts/${NAME}"_*.log 2>/dev/null | wc -l)
[ "$failed" -gt 0 ] && echo "WARNING: $failed shard log(s) contain a traceback; check $OUT/parts/${NAME}_*.log" >&2

for map in sf ht; do
    merged="$OUT/${NAME}_${map}.csv"
    first=1
    shard_rows=0
    : > "$merged"
    for part in "$OUT/parts/${NAME}_${map}_s"*.csv; do
        [ -f "$part" ] || continue
        if [ $first -eq 1 ]; then head -n 1 "$part" >> "$merged"; first=0; fi
        tail -n +2 "$part" >> "$merged"
        shard_rows=$(( shard_rows + $(tail -n +2 "$part" | wc -l) ))
    done
    rows=$(( $(wc -l < "$merged") - 1 ))
    echo "wrote $merged ($rows episodes)"
    # the merged file now holds every shard row, so the shard CSVs are duplicates;
    # their logs stay for debugging. KEEP_PARTS=1 keeps the CSVs too.
    if [ "${KEEP_PARTS:-0}" = "0" ] && [ "$rows" -gt 0 ] && [ "$rows" -eq "$shard_rows" ]; then
        rm -f "$OUT/parts/${NAME}_${map}_s"*.csv
    fi
done

echo "arm '$NAME' finished in $(( ($(date +%s) - started) / 60 )) min"
python3 - "$OUT" "$NAME" <<'PY'
import csv, glob, sys
out, name = sys.argv[1], sys.argv[2]
rows = [r for p in sorted(glob.glob(f'{out}/{name}_*.csv')) for r in csv.DictReader(open(p))]
if not rows:
    sys.exit('no rows written')
cells = {}
for r in rows:
    cells.setdefault((r['map'][12:14], r['requested_n_dynamic']), []).append(r)
print(f"{'map':5s} {'ships':>5s} {'n':>4s} {'success':>8s} {'collision':>9s} {'timeout':>8s} {'failed gen':>10s}")
for (m, d), rs in sorted(cells.items()):
    n = len(rs)
    print(f'{m:5s} {d:>5s} {n:4d} '
          f"{sum(r['outcome'] == 'success' for r in rs):8d} "
          f"{sum(r['outcome'].startswith('collision') for r in rs):9d} "
          f"{sum(r['outcome'] == 'timeout' for r in rs):8d} "
          f"{sum(r['outcome'] == 'scenario_failed' for r in rs):10d}")
PY
