#!/usr/bin/env bash
# One command for every experiment. The settings live in crowd_nav/experiments.txt.
#
#   ./crowd_nav/exp.sh list                               what experiments exist
#   ./crowd_nav/exp.sh status                             what has finished or is running
#   ./crowd_nav/exp.sh run visible_colregs                run an experiment
#   ./crowd_nav/exp.sh compare visible_sarl visible_colregs
#   ./crowd_nav/exp.sh video visible_colregs sf:8:80      film trips (map:ships:trip)
#   ./crowd_nav/exp.sh video visible_colregs              film one trip of each outcome
#   ./crowd_nav/exp.sh rules visible_colregs              how often the sea rules were followed
#
# Needs the crowdnav conda environment. Works from any folder.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TABLE=crowd_nav/experiments.txt
RESULTS=crowd_nav/results-v2

# shared settings, the same for every experiment
export EPISODES=100 SHARDS=10 MAX_STEPS=500
RULE_SETTINGS=(--sample 25 --max_range 10 --max_tcpa 20)

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

lookup() {
    local line model
    line=$(awk -v n="$1" '!/^#/ && $1 == n' "$TABLE")
    if [ -z "$line" ]; then
        echo "No experiment called '$1'. See: ./crowd_nav/exp.sh list" >&2
        exit 1
    fi
    read -r _ POLICY model SF_DENSITIES HT_DENSITIES EXTRA <<< "$line"
    MODEL_DIR="crowd_nav/data/$model"
    export POLICY MODEL_DIR SF_DENSITIES HT_DENSITIES
    read -r -a EXTRA_ARGS <<< "${EXTRA:-}"
}

ensure_model() {
    [ -d "$MODEL_DIR" ] && return
    case "$(basename "$MODEL_DIR")" in
        output-visible)
            cp -r crowd_nav/data/output-trained "$MODEL_DIR" &&
                sed -i 's/^visible = false/visible = true/' "$MODEL_DIR/env.config" ;;
        output-slowturn)
            cp -r crowd_nav/data/output-trained "$MODEL_DIR" &&
                sed -i 's/^max_turn_rate = .*/max_turn_rate = 20/' "$MODEL_DIR/policy.config" ;;
        *)
            echo "Model folder $MODEL_DIR is missing." >&2
            exit 1 ;;
    esac
    echo "Created $MODEL_DIR from output-trained."
}

command=${1:-}
[ -n "$command" ] || usage
shift

case "$command" in
    list)
        grep -v '^#' "$TABLE" | awk 'NF' |
            awk 'BEGIN { printf "%-18s %-13s %-16s %-9s %-9s %s\n", "EXPERIMENT", "POLICY", "MODEL FOLDER", "SF SHIPS", "HT SHIPS", "EXTRA" }
                 { extra = ""; for (i = 6; i <= NF; i++) extra = extra " " $i
                   printf "%-18s %-13s %-16s %-9s %-9s%s\n", $1, $2, $3, $4, $5, extra }' ;;
    status)
        python3 crowd_nav/status.py ;;
    run)
        name=${1:?usage: exp.sh run NAME [--again]}
        lookup "$name"
        if ls "$RESULTS/${name}_sf.csv" "$RESULTS/${name}_ht.csv" >/dev/null 2>&1 && [ "${2:-}" != "--again" ]; then
            echo "Results for '$name' already exist in $RESULTS."
            echo "Running it again would replace them. Add --again if that is what you want."
            exit 1
        fi
        ensure_model
        ./crowd_nav/run_arm.sh "$name" "${EXTRA_ARGS[@]}" ;;
    compare)
        a=${1:?usage: exp.sh compare NAME1 NAME2}
        b=${2:?usage: exp.sh compare NAME1 NAME2}
        python3 crowd_nav/compare_arms.py --base "$RESULTS/${a}_sf.csv" "$RESULTS/${a}_ht.csv" \
            --variant "$RESULTS/${b}_sf.csv" "$RESULTS/${b}_ht.csv" --labels "$a" "$b" ;;
    video)
        name=${1:?usage: exp.sh video NAME [map:ships:trip ...]}
        shift
        lookup "$name"
        ensure_model
        if [ $# -gt 0 ]; then
            ./crowd_nav/render_arm.sh "$name" "${EXTRA_ARGS[@]}" --episodes "$@"
        else
            ./crowd_nav/render_arm.sh "$name" "${EXTRA_ARGS[@]}"
        fi ;;
    rules)
        name=${1:?usage: exp.sh rules NAME}
        shift
        lookup "$name"
        ensure_model
        ./crowd_nav/measure_compliance.sh "$name" "${EXTRA_ARGS[@]}" "${RULE_SETTINGS[@]}" "$@" ;;
    *)
        usage ;;
esac
