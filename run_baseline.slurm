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
#SBATCH --array=0-50

set -euo pipefail

source ~/.bashrc

# cd /vulcanscratch/mukunds/downloads/
# rm -rf easi3r_project
# rm -rf /vulcanscratch/mukunds/anaconda3/envs/wham
# conda env remove --name T_CORE
# conda env remove --name cotracker
# conda env remove --name easi3r
# conda env remove --name sam2
# conda env remove --name smpl
# conda env remove --name temos
# conda env remove --name tmr
# conda env remove --name vidseg

conda activate llm
export PYTHONUNBUFFERED=1
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

# python -m scripts.run_baseline \
#   --config configs/default.yaml \
#   --backend qwen \
#   --model_id "Qwen/Qwen2.5-7B-Instruct" \
#   --thinking_mode off \
#   --prompt_style fewshot_fixed \
#   --max_new_tokens 256 \
#   --num_shards "$num_shards" \
#   --shard_id "$shard_id" \
#   --save_jsonl "results/triviaqa/qwen/qwen_final_only_shard${shard_id}.json"

python -m scripts.run_baseline \
  --config configs/default.yaml \
  --backend gemma \
  --model_id "google/gemma-3-12b-it" \
  --thinking_mode off \
  --prompt_style fewshot_fixed \
  --max_new_tokens 64 \
  --num_shards "$num_shards" \
  --shard_id "$shard_id" \
  --save_jsonl "results/triviaqa/gemma/gemma_final_only_shard${shard_id}.json"
