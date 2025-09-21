#!/bin/bash
#SBATCH --job-name=llm-inference
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/gpt-oss-medium%j.out
#SBATCH --error=logs/gpt-oss-medium%j.err

source ~/.bashrc
conda activate easi3r

# Low
# python -m scripts.run_baseline \
#   --config configs/default.yaml \
#   --thinking_mode harmony \
#   --prompt_style fewshot_fixed \
#   --reasoning low \
#   --max_new_tokens 256 \
#   --analysis_cap 512 \
#   --final_allowance 32 \
#   --save_jsonl results/harmony_low.json

# Medium
python -m scripts.run_baseline \
  --config configs/default.yaml \
  --thinking_mode harmony \
  --prompt_style fewshot_fixed \
  --reasoning medium \
  --max_new_tokens 512 \
  --analysis_cap 512 \
  --final_allowance 32 \
  --save_jsonl results/baseline_harmony_medium.json

# High
# python -m scripts.run_baseline \
#   --config configs/default.yaml \
#   --thinking_mode harmony \
#   --prompt_style fewshot_fixed \
#   --reasoning high \
#   --max_new_tokens 256 \
#   --analysis_cap 512 \
#   --final_allowance 32 \
#   --save_jsonl results/harmony_high.json
