#!/bin/bash
#SBATCH --job-name=probe-transfer
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-99

# ==============================================================================
# Model Size Transfer Experiment
# 
# Tests whether probes trained on Qwen2.5-7B generalize to:
# - Qwen2.5-1.5B (same architecture, smaller scale)
# - Qwen3-4B-Instruct (new generation, smaller)
# - Qwen3-8B (new generation, similar scale, NOT instruct-tuned)
#
# Strategy: Collect internals on validation set (smaller, faster) to test transfer
# ==============================================================================

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

num_shards=$SLURM_ARRAY_TASK_COUNT
shard_id=$SLURM_ARRAY_TASK_ID

# ==============================================================================
# QWEN2.5-1.5B (Same Family Transfer)
# ==============================================================================
# Expected: Good transfer (F1 drop < 10%)
# This is the most important test - same architecture, just smaller

echo "=================================================="
echo "Starting Qwen2.5-1.5B collections (same family)"
echo "=================================================="

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset triviaqa --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen2.5_1.5b/triviaqa/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset hotpot_qa --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen2.5_1.5b/hotpotqa/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset squad_v2 --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen2.5_1.5b/squadv2/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset gsm8k --split test \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen2.5_1.5b/gsm8k/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset mmlu --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen2.5_1.5b/mmlu/shard${shard_id}.jsonl

# ==============================================================================
# QWEN3-4B-INSTRUCT (Cross-Generation Transfer, Smaller)
# ==============================================================================
# Expected: Moderate transfer degradation (F1 drop 15-30%)
# Different architecture but instruct-tuned

echo "=================================================="
echo "Starting Qwen3-4B-Instruct collections (new gen)"
echo "=================================================="

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset triviaqa --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_4b/triviaqa/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset hotpot_qa --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_4b/hotpotqa/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset squad_v2 --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_4b/squadv2/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset gsm8k --split test \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_4b/gsm8k/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset mmlu --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_4b/mmlu/shard${shard_id}.jsonl

# ==============================================================================
# QWEN3-8B (Cross-Generation Transfer, Similar Scale, NOT Instruct)
# ==============================================================================
# Expected: Significant degradation (F1 drop 30-50%?)
# Different architecture AND not instruct-tuned (but has instruction capability)
# Note: Model card says it has instruction-following but isn't explicitly instruct-tuned

echo "=================================================="
echo "Starting Qwen3-8B collections (new gen, base)"
echo "=================================================="

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-8B" \
  --dataset triviaqa --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_8b/triviaqa/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-8B" \
  --dataset hotpot_qa --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_8b/hotpotqa/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-8B" \
  --dataset squad_v2 --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_8b/squadv2/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-8B" \
  --dataset gsm8k --split test \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_8b/gsm8k/shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-8B" \
  --dataset mmlu --split validation \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer/qwen3_8b/mmlu/shard${shard_id}.jsonl

echo "=================================================="
echo "All collections complete for shard ${shard_id}"
echo "=================================================="

# ==============================================================================
# NOTES FOR ANALYSIS:
# 
# After collections complete, analyze with:
#
# 1. Merge shards for each model:
#    cat data/probe_transfer/qwen2.5_1.5b/triviaqa/shard*.jsonl > \
#        data/probe_transfer/qwen2.5_1.5b/triviaqa_val.jsonl
#
# 2. Build evaluation dataset (skip training since we're testing transfer):
#    python -m scripts.make_probe_split \
#      --root data/probe_transfer/qwen2.5_1.5b \
#      --out_dir data/probe_transfer/qwen2.5_1.5b/splits \
#      --train_frac 0.0 --val_frac 0.0 --test_frac 1.0
#
# 3. Evaluate with existing Qwen2.5-7B probe:
#    python -m scripts.eval_probe \
#      --model qwen \
#      --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
#      --probe_dir backend_qwen/probe_mlp \
#      --use_hidden \
#      --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu
#
# 4. Compare results across models:
#    - Qwen2.5-7B (baseline): F1 = 0.579
#    - Qwen2.5-1.5B (same family): F1 = ?
#    - Qwen3-4B (new gen): F1 = ?
#    - Qwen3-8B (new gen, base): F1 = ?
#
# ==============================================================================
