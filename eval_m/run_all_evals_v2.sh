#!/bin/bash
# Batch enhanced evaluation for all Clio scenes
# Usage: bash eval_m/run_all_evals_v2.sh

DSG_DIR="/tmp/clio_output"
DATA_DIR="/data/YueChang/Clio"
EVAL_SCRIPT="$(dirname "$0")/eval_clio_v2.py"
OUTPUT_DIR="$(dirname "$0")"

scenes=("cubicle" "apartment" "office" "building")

for scene in "${scenes[@]}"; do
    echo "============================================"
    echo "  Evaluating: $scene"
    echo "============================================"

    DSG="$DSG_DIR/$scene/dsg.json"
    ROOMS="$DATA_DIR/$scene/rooms_${scene}.yaml"
    TASKS="$DATA_DIR/$scene/tasks_${scene}.yaml"
    OUTPUT="$OUTPUT_DIR/${scene}_results.json"

    if [ ! -f "$DSG" ]; then
        echo "SKIP: $DSG not found"
        continue
    fi

    ROOMS_ARG=""
    if [ -f "$ROOMS" ]; then
        ROOMS_ARG="--rooms $ROOMS"
    else
        echo "NOTE: No rooms annotation for $scene"
    fi

    TASKS_ARG=""
    if [ -f "$TASKS" ]; then
        TASKS_ARG="--tasks $TASKS"
    else
        echo "NOTE: No tasks annotation for $scene"
    fi

    python3 "$EVAL_SCRIPT" --dsg "$DSG" $ROOMS_ARG $TASKS_ARG --output "$OUTPUT"
    echo ""
done

echo "=== All enhanced evaluations complete ==="
