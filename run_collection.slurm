#!/bin/bash
#SBATCH --job-name=run-collection
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-99

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1
reasoning=medium

num_shards=$SLURM_ARRAY_TASK_COUNT
shard_id=$SLURM_ARRAY_TASK_ID

# === TRIVIAQA EXPERIMENTS ===
python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset triviaqa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/triviaqa/qwen/qwen_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend gemma \
#   --model_id "google/gemma-3-12b-it" \
#   --dataset triviaqa --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/triviaqa/gemma/gemma_final_only_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset triviaqa --split train \
  --max_new_tokens 64 \
  --num_shards $num_shards --shard_id $shard_id \
  --out data/probe/triviaqa/llama31/llama31_final_only_shard${shard_id}.jsonl

# === HOTPOTQA EXPERIMENTS ===

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset hotpot_qa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/hotpotqa/qwen/qwen_final_only_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset hotpot_qa --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/hotpotqa/llama31/llama31_final_only_shard${shard_id}.jsonl

# === SQuADv2 EXPERIMENTS ===

python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset squad_v2 --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/squadv2/qwen/qwen_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend gemma \
#   --model_id "google/gemma-3-12b-it" \
#   --dataset squad_v2 --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/squadv2/gemma/gemma_final_only_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset squad_v2 --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/squadv2/llama31/llama31_final_only_shard${shard_id}.jsonl

# === GSM8K EXPERIMENTS ===
python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset gsm8k --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/gsm8k/qwen/qwen_final_only_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset gsm8k --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/gsm8k/llama31/llama31_final_only_shard${shard_id}.jsonl

# === validation EXPERIMENTS ===
python -m scripts.collect_internals \
  --backend qwen \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --dataset mmlu --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/mmlu/qwen/qwen_final_only_shard${shard_id}.jsonl

python -m scripts.collect_internals \
  --backend llama31 \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dataset mmlu --split train \
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \
  --out data/probe/mmlu/llama31/llama31_final_only_shard${shard_id}.jsonl

# DATASET OPTIONS:
# # TriviaQA (unfiltered), validation split
# --dataset triviaqa --split validation

# # HotpotQA (distractor), validation split
# --dataset hotpot_qa --split validation

# # SQuAD v1.1, validation split
# --dataset squad --split validation

# # SQuAD v2.0, validation split
# --dataset squad_v2 --split validation

# # Natural Questions Open, validation split
# --dataset nq_open --split validation

# # GSM8K, test split (no validation split available)
# --dataset gsm8k --split test

# # MMLU, test or validation split
# --dataset mmlu --split test

# python -m scripts.make_probe_split \
#   --root data/probe \
#   --out_dir data/probe_splits_80_10_10 \
#   --train_frac 0.80 --val_frac 0.10 --test_frac 0.10