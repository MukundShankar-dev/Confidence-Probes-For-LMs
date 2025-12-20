# Create splits for INDIVIDUAL datasets (for cross-task generalization studies)

set -euo pipefail

echo "========================================================================"
echo "CREATE PER-DATASET PROBE SPLITS"
echo "For cross-task generalization: train on one dataset, test on others"
echo "========================================================================"
echo ""

# ==============================================================================
# PER-DATASET SPLITS FOR QWEN2.5-7B
# ==============================================================================
echo "[1/2] Creating per-dataset splits for Qwen2.5-7B..."
echo ""

# TriviaQA only
echo "Processing: TriviaQA only"
python -m scripts.make_probe_split \
  --root data/probe_standard/qwen2.5_7b/triviaqa \
  --out_dir data/probe_splits_per_dataset/qwen2.5_7b/triviaqa \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# HotpotQA only
echo "Processing: HotpotQA only"
python -m scripts.make_probe_split \
  --root data/probe_standard/qwen2.5_7b/hotpotqa \
  --out_dir data/probe_splits_per_dataset/qwen2.5_7b/hotpotqa \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# SQuAD-v2 only
echo "Processing: SQuAD-v2 only"
python -m scripts.make_probe_split \
  --root data/probe_standard/qwen2.5_7b/squadv2 \
  --out_dir data/probe_splits_per_dataset/qwen2.5_7b/squadv2 \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# GSM8K only
echo "Processing: GSM8K only"
python -m scripts.make_probe_split \
  --root data/probe_standard/qwen2.5_7b/gsm8k \
  --out_dir data/probe_splits_per_dataset/qwen2.5_7b/gsm8k \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# MMLU only
echo "Processing: MMLU only"
python -m scripts.make_probe_split \
  --root data/probe_standard/qwen2.5_7b/mmlu \
  --out_dir data/probe_splits_per_dataset/qwen2.5_7b/mmlu \
  --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
  --seed 42 \
  --skip_global
echo ""

# ==============================================================================
# PER-DATASET SPLITS FOR LLAMA-3.1-8B
# ==============================================================================
echo "[2/2] Creating per-dataset splits for Llama-3.1-8B..."
echo ""

for dataset in triviaqa hotpotqa squadv2 gsm8k mmlu; do
  echo "Processing: $dataset"
  python -m scripts.make_probe_split \
    --root data/probe_standard/llama31_8b/$dataset \
    --out_dir data/probe_splits_per_dataset/llama31_8b/$dataset \
    --train_frac 0.70 --val_frac 0.15 --test_frac 0.15 \
    --seed 42 \
    --skip_global
  echo ""
done

echo "========================================================================"
echo "PER-DATASET SPLITS COMPLETE"
echo "========================================================================"
echo ""
echo "Created splits in: data/probe_splits_per_dataset/"
echo ""
