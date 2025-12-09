#!/bin/bash
#SBATCH --job-name=probe-eval-opt
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

# OPTIMIZED: Load model once, run all 5 probes on same features
# Array job handles 2 models × 5 datasets = 10 combinations

# Use "llama" to match your training output directories (backend_llama)
MODELS=("llama31" "llama31" "llama31" "llama31" "llama31" "qwen" "qwen" "qwen" "qwen" "qwen")
DATASETS=("triviaqa" "hotpotqa" "squad_v2" "gsm8k" "mmlu" "triviaqa" "hotpotqa" "squad_v2" "gsm8k" "mmlu")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "OPTIMIZED PROBE EVALUATION"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "  Strategy: Load model ONCE, run ALL probes"
echo "  Datasets: All 5 (triviaqa, hotpotqa, squad_v2, gsm8k, mmlu)"
echo "=========================================="

# Run optimized evaluation: Single call evaluates all 5 probes
# Dataset-specific prompts are handled automatically (SQuAD v2 gets special prompt)
python -m scripts.eval_probe_optimized \
    --model $MODEL \
    --probe_dirs backend_${MODEL}/probe_mlp,backend_${MODEL}/probe_logreg,backend_${MODEL}/probe_logreg_cal,backend_${MODEL}/probe_tree,backend_${MODEL}/probe_xform \
    --use_hidden \
    --datasets $DATASET \
    --output_dir results

echo "Completed evaluation for $MODEL on $DATASET (all 5 probes)"
echo "Generated files:"
echo "  - results/${MODEL}_${DATASET}_mlp_eval.json"
echo "  - results/${MODEL}_${DATASET}_logreg_eval.json"
echo "  - results/${MODEL}_${DATASET}_logreg_cal_eval.json"
echo "  - results/${MODEL}_${DATASET}_tree_eval.json"
echo "  - results/${MODEL}_${DATASET}_xform_eval.json"
