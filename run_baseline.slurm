#!/bin/bash
#SBATCH --job-name=llm-baseline
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=ALL

set -euo pipefail

# --- Project paths ---
WORKDIR="/vulcanscratch/mukunds/downloads/723_project"   # change if needed
mkdir -p "$WORKDIR/logs"
mkdir -p "$WORKDIR/language_models"/{hf_home,transformers,datasets,xdg,torch,torch_extensions,sentence-transformers}

# --- Cache dirs: keep everything local to the repo ---
export HF_HOME="$WORKDIR/language_models/hf_home"
export HF_DATASETS_CACHE="$WORKDIR/language_models/datasets"
export XDG_CACHE_HOME="$WORKDIR/language_models/xdg"
export TORCH_HOME="$WORKDIR/language_models/torch"
export TORCH_EXTENSIONS_DIR="$WORKDIR/language_models/torch_extensions"
export SENTENCE_TRANSFORMERS_HOME="$WORKDIR/language_models/sentence-transformers"
export TOKENIZERS_PARALLELISM=false

export TRANSFORMERS_CACHE="$WORKDIR/language_models/transformers"

# Ensure conda shell is available
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
eval "$(conda shell.bash hook)"
conda activate llm

cd "$WORKDIR"
echo "[SLURM] PWD=$PWD"
echo "[SLURM] Job: $SLURM_JOB_NAME | ID: $SLURM_JOB_ID"
nvidia-smi || true

: "${PY_MODULE:=scripts.run_baseline}"
: "${PY_ARGS:=--config configs/default.yaml}"          # e.g. add "--limit 2000 --bs 2" for quicker runs

echo "[SLURM] Running: python -m ${PY_MODULE} ${PY_ARGS}"
PYTHONPATH="$WORKDIR" python -m ${PY_MODULE} ${PY_ARGS}

echo "[SLURM] Disk usage for language_models:"
du -h -d1 "$WORKDIR/language_models" || true
