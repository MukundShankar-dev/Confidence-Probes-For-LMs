#!/bin/bash
#SBATCH --job-name=eval-cross-dataset
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

# ====== USER SETTINGS ======
DATASETS="triviaqa,hotpotqa,squadv2,gsm8k,mmlu"

# Point these at your *existing* probe maps (one per model).
# Each JSON must be: {"triviaqa": "/path/to/probe", ...}
QWEN_PROBE_MAP_JSON="data/qwen_cross_dataset_probes.json"
LLAMA_PROBE_MAP_JSON="data/llama_cross_dataset_probes.json"

# Output roots (script writes into these)
QWEN_OUT_DIR="results/cross_datasets/qwen"
LLAMA_OUT_DIR="results/cross_datasets/llama"

source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1
# ===========================

mkdir -p logs

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    mkdir -p "${QWEN_OUT_DIR}"
    python -m scripts.eval_probe_cross_datasets \
      --model "qwen" \
      --model_id "Qwen/Qwen2.5-7B-Instruct" \
      --datasets "mmlu" \
      --probe_map_json "${QWEN_PROBE_MAP_JSON}" \
      --output_dir "${QWEN_OUT_DIR}" \
    ;;

  1)
    mkdir -p "${LLAMA_OUT_DIR}"
    python -m scripts.eval_probe_cross_datasets \
      --model "llama" \
      --model_id "meta-llama/Llama-3.1-8B-Instruct" \
      --datasets "mmlu,gsm8k" \
      --probe_map_json "${LLAMA_PROBE_MAP_JSON}" \
      --output_dir "${LLAMA_OUT_DIR}" \
    ;;

  *)
    echo "ERROR: invalid SLURM_ARRAY_TASK_ID='${SLURM_ARRAY_TASK_ID}'. Expected 0 or 1."
    exit 2
    ;;
esac
