#!/bin/bash
#SBATCH --job-name=run-eval
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

python -m scripts.eval_probe --model llama31 --probe_dir backend_llama/probe_mlp --use_hidden --datasets gsm8k
# python -m scripts.eval_probe --model llama31 --probe_dir backend_llama/probe_mlp --use_hidden --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu
# python -m scripts.eval_probe --model qwen --probe_dir backend_qwen/probe_mlp --use_hidden --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu
