#!/bin/bash

# NVIDIA NCU Profiling Script for Opacus DP-SGD Performance Analysis
# This script runs NCU profiling for different trainer configurations
# and collects memory I/O metrics to analyze the cost of per-sample gradient clipping

PYTHON_SCRIPT="profiling_script_ncu.py"

# Experiment configurations
TRAINERS=("standard" "dpsgd" "dpsgd_ghost")
BATCH_SIZES=(8 16)
SEQ_LENGTHS=(256 1024)

# Key memory I/O metrics for analyzing gradient clipping overhead
# Focus on DRAM and L2 cache throughput to measure I/O costs
NCU_METRICS="dram_read_throughput,dram_write_throughput,l2_read_throughput,l2_write_throughput,gld_throughput,gst_throughput"

echo "=========================================="
echo "NVIDIA NCU Profiling for Opacus DP-SGD"
echo "=========================================="
echo "Trainers: ${TRAINERS[*]}"
echo "Batch sizes: ${BATCH_SIZES[*]}"
echo "Sequence lengths: ${SEQ_LENGTHS[*]}"
echo "Metrics: ${NCU_METRICS}"
echo "=========================================="

# Check if NCU is available
if ! command -v ncu &> /dev/null; then
    echo "ERROR: NCU (NVIDIA Nsight Compute) not found!"
    echo "Please install NVIDIA Nsight Compute and ensure 'ncu' is in your PATH."
    exit 1
fi

# Check if Python script exists
if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python script '${PYTHON_SCRIPT}' not found!"
    echo "Please ensure the script is in the current directory."
    exit 1
fi

# Create output directory for reports
mkdir -p ncu_reports
cd ncu_reports

# Run local test first to ensure everything works
echo "Running local test first..."
python "../${PYTHON_SCRIPT}" --mode=test
if [ $? -ne 0 ]; then
    echo "ERROR: Local test failed! Please fix issues before running NCU profiling."
    exit 1
fi
echo "Local test passed. Starting NCU profiling..."

# Main profiling loop
total_experiments=$((${#TRAINERS[@]} * ${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]}))
current_experiment=0

for trainer in "${TRAINERS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    for seq in "${SEQ_LENGTHS[@]}"; do
      
      current_experiment=$((current_experiment + 1))
      
      LOG_NAME="report_${trainer}_bs${bs}_seq${seq}"
      REPORT_FILE="${LOG_NAME}.ncu-rep"

      echo ""
      echo "-----------------------------------------------------"
      echo "Experiment ${current_experiment}/${total_experiments}"
      echo "Running: Trainer=${trainer}, BS=${bs}, SeqLen=${seq}"
      echo "Output file: ${REPORT_FILE}"
      echo "-----------------------------------------------------"

      # NCU command with memory I/O focus
      ncu -o "${REPORT_FILE}" \
          --metrics "${NCU_METRICS}" \
          --target-processes all \
          --kernel-regex ".*" \
          --launch-skip 0 \
          --launch-count 0 \
          python "../${PYTHON_SCRIPT}" \
              --mode=profile \
              --trainer="${trainer}" \
              --batch_size="${bs}" \
              --seq_len="${seq}"

      # Check NCU execution status
      if [ $? -ne 0 ]; then
          echo "!!! NCU FAILED for ${LOG_NAME}"
          echo "    This might be due to:"
          echo "    - Insufficient GPU memory"
          echo "    - CUDA/Opacus compatibility issues"
          echo "    - NCU permission problems"
          echo "    Continuing with next experiment..."
      else
          echo "+++ NCU SUCCESS for ${LOG_NAME}"
          
          # Check if report file was created and has reasonable size
          if [ -f "${REPORT_FILE}" ] && [ -s "${REPORT_FILE}" ]; then
              file_size=$(stat -f%z "${REPORT_FILE}" 2>/dev/null || stat -c%s "${REPORT_FILE}" 2>/dev/null)
              echo "    Report file size: ${file_size} bytes"
          else
              echo "    WARNING: Report file is empty or missing"
          fi
      fi

    done
  done
done

echo ""
echo "=========================================="
echo "All profiling runs finished!"
echo "=========================================="

# Summary of generated reports
echo "Generated reports:"
ls -la *.ncu-rep 2>/dev/null || echo "No .ncu-rep files found"

echo ""
echo "Next steps:"
echo "1. Open NCU GUI: ncu-ui"
echo "2. Load .ncu-rep files in the GUI for analysis"
echo "3. Compare memory throughput metrics between trainers:"
echo "   - Standard SGD (baseline)"
echo "   - DP-SGD (should show higher I/O cost)"
echo "   - DP-SGD Ghost Clipping (should show reduced I/O vs flat clipping)"
echo ""
echo "Key metrics to compare:"
echo "- dram_read_throughput: Higher in DP-SGD due to per-sample gradient reads"
echo "- dram_write_throughput: Higher in DP-SGD due to gradient storage"
echo "- l2_read/write_throughput: Cache pressure from gradient operations"
echo ""
echo "For batch analysis, you can also export to CSV:"
echo "ncu --import report_dpsgd_bs8_seq256.ncu-rep --csv > analysis.csv"