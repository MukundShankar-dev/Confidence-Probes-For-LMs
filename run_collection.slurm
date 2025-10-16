#!/bin/bash
#SBATCH --job-name=run-collection
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa4000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-0

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1
reasoning=medium

num_shards=$SLURM_ARRAY_TASK_COUNT
shard_id=$SLURM_ARRAY_TASK_ID

# === TRIVIAQA EXPERIMENTS ===
# python -m scripts.collect_internals \
#   --backend qwen \
#   --model_id "Qwen/Qwen2.5-7B-Instruct" \
#   --dataset triviaqa --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/triviaqa/qwen/qwen_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend gemma \
#   --model_id "google/gemma-3-12b-it" \
#   --dataset triviaqa --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/triviaqa/gemma/gemma_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend llama31 \
#   --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
#   --dataset triviaqa --split validation \
#   --max_new_tokens 64 \
#   --num_shards $num_shards --shard_id $shard_id \
#   --out data/probe/triviaqa/llama31/llama31_final_only_shard${shard_id}.jsonl

# === HOTPOTQA EXPERIMENTS ===

# python -m scripts.collect_internals \
#   --backend qwen \
#   --model_id "Qwen/Qwen2.5-7B-Instruct" \
#   --dataset hotpot_qa --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/hotpotqa/qwen/qwen_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend gemma \
#   --model_id "google/gemma-3-12b-it" \
#   --dataset hotpot_qa --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/hotpotqa/gemma/gemma_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend llama31 \
#   --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
#   --dataset hotpot_qa --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/hotpotqa/llama31/llama31_final_only_shard${shard_id}.jsonl

# === SQuADv2 EXPERIMENTS ===

# python -m scripts.collect_internals \
#   --backend qwen \
#   --model_id "Qwen/Qwen2.5-7B-Instruct" \
#   --dataset squad_v2 --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/squadv2/qwen/qwen_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend gemma \
#   --model_id "google/gemma-3-12b-it" \
#   --dataset squad_v2 --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/squadv2/gemma/gemma_final_only_shard${shard_id}.jsonl

# python -m scripts.collect_internals \
#   --backend llama31 \
#   --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
#   --dataset squad_v2 --split validation \
#   --max_new_tokens 64 \
#   --num_shards "$num_shards" --shard_id "$shard_id" \
#   --out data/probe/squadv2/llama31/llama31_final_only_shard${shard_id}.jsonl

# DATASET OPTIONS:
# # TriviaQA (unfiltered), validation split
# --dataset triviaqa --split validation

# # HotpotQA (distractor), train split
# --dataset hotpot_qa --split train

# # SQuAD v1.1
# --dataset squad --split validation

# # SQuAD v2.0
# --dataset squad_v2 --split validation

# # Natural Questions Open
# --dataset nq_open --split validation

python -m scripts.make_probe_split \
  --root data/probe \
  --out_dir data/probe_splits_80_10_10 \
  --train_frac 0.80 --val_frac 0.10 --test_frac 0.10