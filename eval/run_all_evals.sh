#!/bin/bash
# Batch evaluation for all Clio scenes
# Usage: bash eval/run_all_evals.sh

DSG_DIR="/tmp/clio_output"
DATA_DIR="/data/YueChang/Clio"
EVAL_SCRIPT="$(dirname "$0")/eval_clio.py"

scenes=("cubicle" "apartment" "office" "building")

for scene in "${scenes[@]}"; do
    echo "============================================"
    echo "  Evaluating: $scene"
    echo "============================================"

    DSG="$DSG_DIR/$scene/dsg.json"
    ROOMS="$DATA_DIR/$scene/rooms_${scene}.yaml"
    TASKS="$DATA_DIR/$scene/tasks_${scene}.yaml"
    OUTPUT="$DSG_DIR/$scene/eval_results.json"

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

echo "=== All evaluations complete ==="
