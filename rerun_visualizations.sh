#!/bin/bash

main() {
BASE_DIR="memory_profiling_results"
VISUALIZATION_SCRIPT="visualize_memory_breakdown.py"

if [ ! -f "$VISUALIZATION_SCRIPT" ]; then
    echo "Error: Visualization script not found at '$VISUALIZATION_SCRIPT'"
    return 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory '$BASE_DIR' not found."
    return 1
fi

echo "Starting to regenerate visualizations..."
echo "=================================================="

for run_dir in "$BASE_DIR"/run_2025*; do
    if [ -d "$run_dir" ]; then
        echo "Processing directory: $run_dir"
        if ls "$run_dir"/*_result.json 1> /dev/null 2>&1; then
            echo "Found JSON files, running visualization script..."
            python3 "$VISUALIZATION_SCRIPT" --input-dir "$run_dir" --output-dir "$run_dir/visualizations"
            echo "Finished processing $run_dir"
        else
            echo "No '*_result.json' files found in $run_dir, skipping."
        fi
        echo "--------------------------------------------------"
    fi
done

echo "=================================================="
echo "All visualizations have been regenerated."
}

main "$@"