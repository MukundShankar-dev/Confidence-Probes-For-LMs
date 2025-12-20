# Create indexed splits for all collected probe data

set -euo pipefail

echo "========================================================================"
echo "CREATE INDEXED PROBE SPLITS"
echo "========================================================================"
echo ""

# ==============================================================================
# EXPERIMENT 1: PER-MODEL TRAINING DATA (70/15/15 split)
# ==============================================================================
echo "[1/2] Creating splits for per-model training data..."
echo ""

# Qwen2.5-7B
echo "Processing: Qwen2.5-7B"
python -m scripts.make_probe_split \
  --root data/probe_standard/qwen2.5_7b \
  --out_dir data/probe_splits_standard/qwen2.5_7b \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# Llama-3.1-8B
echo "Processing: Llama-3.1-8B"
python -m scripts.make_probe_split \
  --root data/probe_standard/llama31_8b \
  --out_dir data/probe_splits_standard/llama31_8b \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# ==============================================================================
# EXPERIMENT 2: TRANSFER TESTING DATA (70/15/15 split for consistency)
# ==============================================================================
echo "[2/2] Creating splits for transfer testing data..."
echo ""

# Qwen2.5-1.5B
echo "Processing: Qwen2.5-1.5B"
python -m scripts.make_probe_split \
  --root data/probe_transfer_standard/qwen2.5_1.5b \
  --out_dir data/probe_splits_standard/qwen2.5_1.5b \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# Qwen3-4B
echo "Processing: Qwen3-4B"
python -m scripts.make_probe_split \
  --root data/probe_transfer_standard/qwen3_4b \
  --out_dir data/probe_splits_standard/qwen3_4b \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# Qwen2.5-7B (for Job 4 - uses SAME split as training data)
# This creates a separate split dir for super-generalizable probe
echo "Processing: Qwen2.5-7B (reusing standard split)"
# Just symlink to avoid duplication
if [ ! -e data/probe_splits_standard/qwen2.5_7b ]; then
    echo "  [ERROR] Need to run qwen2.5_7b standard split first!"
    exit 1
fi
echo "  Using existing split from data/probe_splits_standard/qwen2.5_7b"
echo ""

# Qwen2.5-14B
echo "Processing: Qwen2.5-14B"
python -m scripts.make_probe_split \
  --root data/probe_transfer_standard/qwen2.5_14b \
  --out_dir data/probe_splits_standard/qwen2.5_14b \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

echo "========================================================================"
echo "SPLIT CREATION COMPLETE"
echo "========================================================================"
echo ""
echo "Created indexed splits:"
echo "  - data/probe_splits_standard/qwen2.5_7b/"
echo "  - data/probe_splits_standard/llama31_8b/"
echo "  - data/probe_splits_standard/qwen2.5_1.5b/"
echo "  - data/probe_splits_standard/qwen3_4b/"
echo "  - data/probe_splits_standard/qwen2.5_14b/"
echo ""
echo "Each directory contains:"
echo "  - split_map.json (indices for train/val/test)"
echo "  - file_index.json (file metadata)"
echo "  - train_indices.txt, val_indices.txt, test_indices.txt"
echo ""
