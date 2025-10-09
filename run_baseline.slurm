#!/bin/bash
#SBATCH --job-name=llm-inference
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-24

set -euo pipefail

source ~/.bashrc
conda activate llm

reasoning=medium

num_shards=$SLURM_ARRAY_TASK_COUNT
shard_id=$SLURM_ARRAY_TASK_ID

# python -m scripts.run_baseline \
#   --config configs/default.yaml \
#   --thinking_mode harmony \
#   --prompt_style fewshot_fixed \
#   --reasoning $reasoning \
#   --max_new_tokens 512 \
#   --analysis_cap 512 \
#   --final_allowance 32 \
#   --num_shards $num_shards \
#   --shard_id $shard_id \
#   --save_jsonl results/triviaqa/harmony_${reasoning}_shard${shard_id}.json

python -m scripts.run_baseline \
  --config configs/default.yaml \
  --backend qwen \
  --thinking_mode off \
  --prompt_style fewshot_fixed \
  --max_new_tokens 512 \
  --num_shards $num_shards \
  --shard_id $shard_id \
  --save_jsonl results/triviaqa/qwen_final_only_shard${shard_id}.json