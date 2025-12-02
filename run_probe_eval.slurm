#!/bin/bash
#SBATCH --job-name=probe-eval
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-9

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

MODELS=("llama31" "llama31" "llama31" "llama31" "llama31" "qwen" "qwen" "qwen" "qwen" "qwen")
DATASETS=("triviaqa" "hotpotqa" "squad_v2" "gsm8k" "mmlu" "triviaqa" "hotpotqa" "squad_v2" "gsm8k" "mmlu")
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Running evaluation for:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="

python -m scripts.eval_probe --model $MODEL --probe_dir backend_${MODEL}/probe_mlp --use_hidden --datasets $DATASET --output results/${MODEL}_${DATASET}_mlp_eval.json
python -m scripts.eval_probe --model $MODEL --probe_dir backend_${MODEL}/probe_logreg --use_hidden --datasets $DATASET --output results/${MODEL}_${DATASET}_logreg_eval.json
python -m scripts.eval_probe --model $MODEL --probe_dir backend_${MODEL}/probe_logreg_cal --use_hidden --datasets $DATASET --output results/${MODEL}_${DATASET}_logreg_cal_eval.json
python -m scripts.eval_probe --model $MODEL --probe_dir backend_${MODEL}/probe_tree --use_hidden --datasets $DATASET --output results/${MODEL}_${DATASET}_tree_eval.json
python -m scripts.eval_probe --model $MODEL --probe_dir backend_${MODEL}/probe_xform --use_hidden --datasets $DATASET --output results/${MODEL}_${DATASET}_xform_eval.json

# python -m scripts.eval_probe --model llama31 --probe_dir backend_llama31/probe_mlp --use_hidden --datasets triviaqa
# python -m scripts.eval_probe --model llama31 --probe_dir backend_llama31/probe_mlp --use_hidden --datasets hotpotqa
# python -m scripts.eval_probe --model llama31 --probe_dir backend_llama31/probe_mlp --use_hidden --datasets squad_v2
# python -m scripts.eval_probe --model llama31 --probe_dir backend_llama31/probe_mlp --use_hidden --datasets gsm8k
# python -m scripts.eval_probe --model llama31 --probe_dir backend_llama31/probe_mlp --use_hidden --datasets mmlu
