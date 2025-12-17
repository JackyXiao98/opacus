#!/usr/bin/env bash

# Run FlashNorm ablation (baseline vs TMA vs Split-T) on fixed shapes.
# Environment variables to override defaults:
#   OUT_DIR   - output directory for CSV/plot
#   ITERS     - benchmark iterations (default 10)
#   WARMUP    - warmup iterations (default 5)
#   SPLIT_K   - split factor for Split-T (default 4)
#   SHAPE     - B,T,Din,Dout (default 1,8196,1024,1024)


OUT_DIR="./out"
ITERS=10
WARMUP=5
SPLIT_K=4
SHAPE="1,8192,512,512"

mkdir -p "${OUT_DIR}"

python "ablation_flashnorm.py" \
    --output-dir "${OUT_DIR}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    --split-k "${SPLIT_K}" \
    --shape "${SHAPE}"

