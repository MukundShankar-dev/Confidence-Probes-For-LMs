#!/bin/bash
#SBATCH --job-name=collect-standard
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-99

# ==============================================================================
# CONSOLIDATED DATA COLLECTION - STANDARD PROMPTS
# 
# Two experiments:
# 1. Train separate probes for each model (Qwen2.5-7B, Llama-3.1-8B)
# 2. Test transfer from Qwen2.5-7B to other Qwen sizes
#
# Uses standard dataset prompts (not JSON) for better generalization
# ==============================================================================

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

num_shards=$SLURM_ARRAY_TASK_COUNT
shard_id=$SLURM_ARRAY_TASK_ID

echo "========================================================================"
echo "DATA COLLECTION WITH STANDARD PROMPTS"
echo "Shard: ${shard_id}/${num_shards}"
echo "========================================================================"

# ==============================================================================
# EXPERIMENT 1: Train separate probes for each model
# Collect training data on TRAIN splits for probe training
# ==============================================================================

echo ""
echo "========================================================================"
echo "EXPERIMENT 1: Training Data Collection (for per-model probes)"
echo "========================================================================"

# --- Qwen2.5-7B Training Data ---
echo ""
echo "[1/2] Collecting Qwen2.5-7B training data..."

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset triviaqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/qwen2.5_7b/triviaqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset hotpotqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/qwen2.5_7b/hotpotqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset squad_v2 --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/qwen2.5_7b/squadv2/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset gsm8k --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/qwen2.5_7b/gsm8k/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset mmlu --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/qwen2.5_7b/mmlu/train_shard${shard_id}.jsonl

# --- Llama-3.1-8B Training Data ---
echo ""
echo "[2/2] Collecting Llama-3.1-8B training data..."

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset triviaqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/llama31_8b/triviaqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset hotpotqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/llama31_8b/hotpotqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset squad_v2 --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/llama31_8b/squadv2/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset gsm8k --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/llama31_8b/gsm8k/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --dataset mmlu --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_standard/llama31_8b/mmlu/train_shard${shard_id}.jsonl

# ==============================================================================
# EXPERIMENT 2: Test transfer from Qwen2.5-7B to other Qwen sizes
# Collect validation data on VALIDATION splits for transfer testing
# ==============================================================================

echo ""
echo "========================================================================"
echo "EXPERIMENT 2: Transfer Test Data Collection (validation splits)"
echo "========================================================================"

# --- Qwen2.5-1.5B (Same Family, Down-Scaling) ---
echo ""
echo "[1/3] Collecting Qwen2.5-1.5B validation data..."

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset triviaqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_1.5b/triviaqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset hotpotqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_1.5b/hotpotqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset squad_v2 --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_1.5b/squadv2/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset gsm8k --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_1.5b/gsm8k/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset mmlu --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_1.5b/mmlu/train_shard${shard_id}.jsonl

# --- Qwen3-4B (Cross-Generation, Smaller) ---
echo ""
echo "[2/3] Collecting Qwen3-4B validation data..."

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset triviaqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen3_4b/triviaqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset hotpotqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen3_4b/hotpotqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset squad_v2 --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen3_4b/squadv2/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset gsm8k --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen3_4b/gsm8k/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen3-4B-Instruct-2507" \
  --dataset mmlu --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen3_4b/mmlu/train_shard${shard_id}.jsonl

# --- Qwen2.5-14B (Same Family, Up-Scaling) ---
echo ""
echo "[3/3] Collecting Qwen2.5-14B validation data..."

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --dataset triviaqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_14b/triviaqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --dataset hotpotqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_14b/hotpotqa/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --dataset squad_v2 --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_14b/squadv2/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --dataset gsm8k --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_14b/gsm8k/train_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-14B-Instruct" \
  --dataset mmlu --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe_transfer_standard/qwen2.5_14b/mmlu/train_shard${shard_id}.jsonl

echo ""
echo "========================================================================"
echo "COMPLETED: All collections for shard ${shard_id}"
echo "========================================================================"

# ==============================================================================
# POST-PROCESSING INSTRUCTIONS
# ==============================================================================
# 
# After all shards complete, merge and process:
#
# 1. MERGE SHARDS FOR EACH MODEL+DATASET:
#    for model in qwen2.5_7b llama31_8b; do
#      for dataset in triviaqa hotpotqa squadv2 gsm8k mmlu; do
#        cat data/probe_standard/${model}/${dataset}/train_shard*.jsonl > \
#            data/probe_standard/${model}/${dataset}_train.jsonl
#      done
#    done
#
#    for model in qwen2.5_1.5b qwen3_4b qwen2.5_14b; do
#      for dataset in triviaqa hotpotqa squadv2 gsm8k mmlu; do
#        cat data/probe_transfer_standard/${model}/${dataset}/val_shard*.jsonl > \
#            data/probe_transfer_standard/${model}/${dataset}_val.jsonl
#      done
#    done
#
# 2. CREATE TRAIN/VAL/TEST SPLITS FOR PROBE TRAINING:
#    python -m scripts.make_probe_split \
#      --root data/probe_standard/qwen2.5_7b \
#      --out_dir data/probe_splits_standard/qwen2.5_7b \
#      --train_frac 0.70 --val_frac 0.15 --test_frac 0.15
#
#    python -m scripts.make_probe_split \
#      --root data/probe_standard/llama31_8b \
#      --out_dir data/probe_splits_standard/llama31_8b \
#      --train_frac 0.70 --val_frac 0.15 --test_frac 0.15
#
# 3. TRAIN PROBES:
#    python -m scripts.train_probe \
#      --data_train data/probe_splits_standard/qwen2.5_7b/train_data.jsonl \
#      --data_val data/probe_splits_standard/qwen2.5_7b/val_data.jsonl \
#      --output_dir backend_qwen_standard/probe_mlp
#
# 4. TEST TRANSFER (use probe trained on Qwen2.5-7B, test on other models):
#    # Will need to update eval script to use standard format
#
# ==============================================================================
