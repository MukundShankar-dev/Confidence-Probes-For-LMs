#!/bin/bash
#SBATCH --job-name=train-cross-dataset
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-9

# ==============================================================================
# CROSS-DATASET PROBE TRAINING
# 
# Trains probes on INDIVIDUAL datasets to test cross-task generalization
# 
# Research Question: Do probes learn task-specific patterns?
# - Train on TriviaQA → Test on HotpotQA, SQuAD
# - Train on GSM8K → Test on MMLU
# 
# Array indices:
#   0-4: Qwen2.5-7B (triviaqa, hotpotqa, squadv2, gsm8k, mmlu)
#   5-9: Llama-3.1-8B (triviaqa, hotpotqa, squadv2, gsm8k, mmlu)
# ==============================================================================

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

mkdir -p logs

echo "========================================================================"
echo "CROSS-DATASET PROBE TRAINING - Array Job $SLURM_ARRAY_TASK_ID"
echo "========================================================================"
echo ""

# Model and dataset mapping
MODELS=("qwen2.5_7b" "qwen2.5_7b" "qwen2.5_7b" "qwen2.5_7b" "qwen2.5_7b" \
        "llama31_8b" "llama31_8b" "llama31_8b" "llama31_8b" "llama31_8b")

DATASETS=("triviaqa" "hotpotqa" "squadv2" "gsm8k" "mmlu" \
          "triviaqa" "hotpotqa" "squadv2" "gsm8k" "mmlu")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Training probe:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo ""

python -m scripts.train_probe \
    --data_dir data/probe_splits_per_dataset/${MODEL}/${DATASET}/backend_${MODEL} \
    --output_dir models/probes_cross_dataset/${MODEL}_${DATASET}_only \
    --probe_type mlp \
    --use_hidden \
    --only_generalizable

echo ""
echo "✓ Cross-dataset probe complete: ${MODEL}_${DATASET}"
echo ""
echo "========================================================================"
echo "JOB $SLURM_ARRAY_TASK_ID COMPLETE"
echo "========================================================================"
echo ""
echo "Next: Evaluate this probe on OTHER datasets to measure transfer"
echo ""
