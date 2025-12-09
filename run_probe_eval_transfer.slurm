#!/bin/bash
#SBATCH --job-name=probe-transfer
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-14

# ==============================================================================
# Model Size Transfer Experiment (OPTIMIZED)
# 
# Tests probe transfer across model sizes:
# - Qwen2.5-1.5B-Instruct (same family)
# - Qwen3-4B-Instruct (new generation)
# - Qwen2.5-14B-Instruct (new generation, base model)
#
# Strategy: Load model ONCE, run ALL probes (MLP, LogReg, Tree, etc.)
# Array job: 3 models × 5 datasets = 15 tasks
# ==============================================================================

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

# Define model IDs and datasets for array job
MODEL_IDS=(
    # Qwen2.5-1.5B (5 datasets)
    "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct"
    # Qwen3-4B (5 datasets)  
    "Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen3-4B-Instruct-2507" "Qwen/Qwen3-4B-Instruct-2507"
    # Qwen2.5-14b-Instruct (5 datasets)
    "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-14B-Instruct"
)

DATASETS=(
    # Qwen2.5-1.5B
    "triviaqa" "hotpotqa" "squad_v2" "gsm8k" "mmlu"
    # Qwen3-4B
    "triviaqa" "hotpotqa" "squad_v2" "gsm8k" "mmlu"
    # Qwen3-14B
    "triviaqa" "hotpotqa" "squad_v2" "gsm8k" "mmlu"
)

MODEL_NAMES=(
    # Qwen2.5-1.5B
    "qwen2.5_1.5b" "qwen2.5_1.5b" "qwen2.5_1.5b" "qwen2.5_1.5b" "qwen2.5_1.5b"
    # Qwen3-4B
    "qwen3_4b" "qwen3_4b" "qwen3_4b" "qwen3_4b" "qwen3_4b"
    # Qwen3-14B
    "qwen2.5_14b" "qwen2.5_14b" "qwen2.5_14b" "qwen2.5_14b" "qwen2.5_14b"
)

# Get this task's configuration
MODEL_ID=${MODEL_IDS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "PROBE TRANSFER EVALUATION (OPTIMIZED)"
echo "  Model ID: $MODEL_ID"
echo "  Model Name: $MODEL_NAME"
echo "  Dataset: $DATASET"
echo "  Array Task: $SLURM_ARRAY_TASK_ID / 14"
echo "  Strategy: Load model ONCE, test ALL probes"
echo "=========================================="

# Create output directory for transfer results
mkdir -p results/transfer

# Run optimized evaluation
# Uses probe trained on Qwen2.5-7B, tests on different model sizes
python -m scripts.eval_probe_optimized \
    --model qwen \
    --model_id "$MODEL_ID" \
    --model_name_prefix "$MODEL_NAME" \
    --probe_dirs backend_qwen/probe_mlp,backend_qwen/probe_logreg,backend_qwen/probe_logreg_cal,backend_qwen/probe_tree,backend_qwen/probe_xform \
    --use_hidden \
    --datasets $DATASET \
    --output_dir results/transfer

echo ""
echo "=========================================="
echo "Completed: $MODEL_NAME on $DATASET"
echo "=========================================="
echo "Generated files in results/transfer/:"
echo "  - ${MODEL_NAME}_${DATASET}_mlp_eval.json"
echo "  - ${MODEL_NAME}_${DATASET}_logreg_eval.json"
echo "  - ${MODEL_NAME}_${DATASET}_logreg_cal_eval.json"
echo "  - ${MODEL_NAME}_${DATASET}_tree_eval.json"
echo "  - ${MODEL_NAME}_${DATASET}_xform_eval.json"
echo ""

# ==============================================================================
# NOTES:
#
# After all jobs complete, analyze with:
#   python analyze_transfer_results.py
#
# Expected output: 75 JSON files (3 models × 5 datasets × 5 probes)
#
# Results structure:
#   results/transfer/
#   ├── qwen2.5_1.5b_triviaqa_mlp_eval.json
#   ├── qwen2.5_1.5b_triviaqa_logreg_eval.json
#   ├── ... (25 files for Qwen2.5-1.5B)
#   ├── qwen3_4b_triviaqa_mlp_eval.json
#   ├── ... (25 files for Qwen3-4B)
#   ├── qwen3_8b_triviaqa_mlp_eval.json
#   └── ... (25 files for Qwen3-8B)
#
# Timeline: ~3-5 hours for all 15 tasks (each task runs all 5 probes on 1 dataset)
# ==============================================================================
