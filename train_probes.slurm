#!/bin/bash
#SBATCH --job-name=train-probes
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-5

# ==============================================================================
# COMPREHENSIVE PROBE TRAINING (INDEXED SPLITS)
# 
# Uses indexed splits (no data duplication, saves 100s of GB)
# 
# WORKFLOW:
# 1. Collect data with run_collection_consolidated.slurm
# 2. Create indexed splits with make_probe_split_efficient.py
# 3. Train probes with this script (reads indices, loads data on-demand)
#
# Trains multiple probe variants for within-model and cross-model experiments:
#
# 1. PER-MODEL GENERALIZABLE PROBES (for within-model performance)
#    - Qwen2.5-7B: generalizable features + hidden states
#    - Llama-3.1-8B: generalizable features + hidden states
#
# 2. QWEN SUPER-GENERALIZABLE PROBES (for cross-model transfer)
#    Tests which model's data generalizes best across Qwen family:
#    - Trained on Qwen2.5-1.5B (smallest)
#    - Trained on Qwen3-4B (mid-small, different generation)
#    - Trained on Qwen2.5-7B (mid-size, baseline)
#    - Trained on Qwen2.5-14B (largest)
#
# NOTE: With standard prompts (no JSON), "generalizable" and "super-generalizable"
# both exclude answer_len and is_unknown. The ONLY difference is hidden states:
#   - Generalizable: 6 scalars + 1024 hidden = 1030 features
#   - Super-generalizable: 6 scalars only = 6 features (for cross-model transfer)
#
# Research Questions:
# Q1: Does probe trained on 1.5B generalize UP to 7B/14B?
# Q2: Does probe trained on 7B generalize BOTH ways (down to 1.5B, up to 14B)?
# Q3: Does probe trained on 14B generalize DOWN to smaller models?
# Q4: Does Qwen3-4B (different arch) transfer to Qwen2.5 family?
#
# Array job indices:
#   0: Qwen2.5-7B generalizable (standard)
#   1: Llama-3.1-8B generalizable (standard)
#   2: Qwen2.5-1.5B super-generalizable (transfer: up-scaling test)
#   3: Qwen3-4B super-generalizable (transfer: cross-generation test)
#   4: Qwen2.5-7B super-generalizable (transfer: baseline)
#   5: Qwen2.5-14B super-generalizable (transfer: down-scaling test)
# ==============================================================================

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

# Create logs directory
mkdir -p logs

echo "========================================================================"
echo "PROBE TRAINING - Job Array Index: $SLURM_ARRAY_TASK_ID"
echo "========================================================================"

case $SLURM_ARRAY_TASK_ID in

# ==============================================================================
# JOB 0: Qwen2.5-7B Generalizable (Standard Within-Model Performance)
# ==============================================================================
0)
    echo ""
    echo "=========================================="
    echo "JOB 0: Qwen2.5-7B Generalizable Probe"
    echo "=========================================="
    echo "Purpose: Best performance on Qwen2.5-7B (excludes answer_len, is_unknown)"
    echo "Features: Generalizable scalars (6) + Hidden states (1024) = 1030 features"
    echo "Use case: Within-model evaluation, cross-dataset transfer"
    echo ""
    
    python -m scripts.train_probe \
        --data_dir data/probe_splits_standard/qwen2.5_7b/backend_qwen2.5_7b \
        --output_dir models/probes_standard/qwen2.5_7b_generalizable \
        --probe_type mlp \
        --use_hidden \
        --only_generalizable
    
    echo "✓ Qwen2.5-7B generalizable probe complete"
    ;;

# ==============================================================================
# JOB 1: Llama-3.1-8B Generalizable (Standard Within-Model Performance)
# ==============================================================================
1)
    echo ""
    echo "=========================================="
    echo "JOB 1: Llama-3.1-8B Generalizable Probe"
    echo "=========================================="
    echo "Purpose: Best performance on Llama-3.1-8B"
    echo "Features: Generalizable scalars (6) + Hidden states (1024) = 1030 features"
    echo "Use case: Cross-architecture comparison with Qwen"
    echo ""
    
    python -m scripts.train_probe \
        --data_dir data/probe_splits_standard/llama31_8b/backend_llama31_8b \
        --output_dir models/probes_standard/llama31_8b_generalizable \
        --probe_type mlp \
        --use_hidden \
        --only_generalizable
    
    echo "✓ Llama-3.1-8B generalizable probe complete"
    ;;

# ==============================================================================
# JOB 2: Qwen2.5-1.5B Super-Generalizable (UP-SCALING Transfer Test)
# ==============================================================================
2)
    echo ""
    echo "=========================================="
    echo "JOB 2: Qwen2.5-1.5B Super-Generalizable"
    echo "=========================================="
    echo "Purpose: Test if probe trained on SMALLEST model transfers UP"
    echo "Features: Super-generalizable scalars only (6 features, NO hidden states)"
    echo "Research Question: Can smallest model's data teach larger models?"
    echo "Transfer targets: Qwen2.5-7B, Qwen2.5-14B"
    echo ""
    
    python -m scripts.train_probe \
        --data_dir data/probe_splits_standard/qwen2.5_1.5b/backend_qwen2.5_1.5b \
        --output_dir models/probes_transfer/qwen2.5_1.5b_super_gen \
        --probe_type mlp \
        --super_generalizable
    
    echo "✓ Qwen2.5-1.5B super-generalizable probe complete"
    echo "  → Test this on 7B and 14B to measure up-scaling transfer"
    ;;

# ==============================================================================
# JOB 3: Qwen3-4B Super-Generalizable (CROSS-GENERATION Transfer Test)
# ==============================================================================
3)
    echo ""
    echo "=========================================="
    echo "JOB 3: Qwen3-4B Super-Generalizable"
    echo "=========================================="
    echo "Purpose: Test if probe trained on DIFFERENT GENERATION transfers"
    echo "Features: Super-generalizable scalars only (6 features, NO hidden states)"
    echo "Research Question: Does Qwen3 probe transfer to Qwen2.5 family?"
    echo "Transfer targets: Qwen2.5-1.5B, Qwen2.5-7B, Qwen2.5-14B"
    echo ""
    
    python -m scripts.train_probe \
        --data_dir data/probe_splits_standard/qwen3_4b/backend_qwen3_4b \
        --output_dir models/probes_transfer/qwen3_4b_super_gen \
        --probe_type mlp \
        --super_generalizable
    
    echo "✓ Qwen3-4B super-generalizable probe complete"
    echo "  → Test this on Qwen2.5 family to measure cross-generation transfer"
    ;;

# ==============================================================================
# JOB 4: Qwen2.5-7B Super-Generalizable (BASELINE Transfer Test)
# ==============================================================================
4)
    echo ""
    echo "=========================================="
    echo "JOB 4: Qwen2.5-7B Super-Generalizable"
    echo "=========================================="
    echo "Purpose: BASELINE for transfer - mid-size model"
    echo "Features: Super-generalizable scalars only (6 features, NO hidden states)"
    echo "Research Question: Does mid-size model transfer BOTH up and down?"
    echo "Transfer targets: Qwen2.5-1.5B (down), Qwen2.5-14B (up), Qwen3-4B (cross-gen)"
    echo ""
    
    python -m scripts.train_probe \
        --data_dir data/probe_splits_standard/qwen2.5_7b/backend_qwen2.5_7b \
        --output_dir models/probes_transfer/qwen2.5_7b_super_gen \
        --probe_type mlp \
        --super_generalizable
    
    echo "✓ Qwen2.5-7B super-generalizable probe complete"
    echo "  → Test this on all other Qwen models to measure bi-directional transfer"
    ;;

# ==============================================================================
# JOB 5: Qwen2.5-14B Super-Generalizable (DOWN-SCALING Transfer Test)
# ==============================================================================
5)
    echo ""
    echo "=========================================="
    echo "JOB 5: Qwen2.5-14B Super-Generalizable"
    echo "=========================================="
    echo "Purpose: Test if probe trained on LARGEST model transfers DOWN"
    echo "Features: Super-generalizable scalars only (6 features, NO hidden states)"
    echo "Research Question: Does largest model's rich data help smaller models?"
    echo "Transfer targets: Qwen2.5-1.5B, Qwen3-4B, Qwen2.5-7B"
    echo ""
    
    python -m scripts.train_probe \
        --data_dir data/probe_splits_standard/qwen2.5_14b/backend_qwen2.5_14b \
        --output_dir models/probes_transfer/qwen2.5_14b_super_gen \
        --probe_type mlp \
        --super_generalizable
    
    echo "✓ Qwen2.5-14B super-generalizable probe complete"
    echo "  → Test this on smaller models to measure down-scaling transfer"
    ;;

*)
    echo "ERROR: Unknown array index $SLURM_ARRAY_TASK_ID"
    exit 1
    ;;
esac

echo ""
echo "========================================================================"
echo "JOB $SLURM_ARRAY_TASK_ID COMPLETE"
echo "========================================================================"

# ==============================================================================
# POST-TRAINING ANALYSIS INSTRUCTIONS
# ==============================================================================
# 
# After all probes are trained, evaluate transfer performance:
#
# 1. WITHIN-MODEL PERFORMANCE (Jobs 0-1)
#    - Qwen2.5-7B generalizable: Eval on Qwen2.5-7B test set
#    - Llama-3.1-8B generalizable: Eval on Llama-3.1-8B test set
#
# 2. CROSS-MODEL TRANSFER MATRIX (Jobs 2-5)
#    Create transfer matrix: Train on X, Test on Y
#    
#    Test Set →  1.5B   4B(Qwen3)  7B    14B
#    Train ↓
#    1.5B        IN-DIST  CROSS     UP    UP
#    4B(Qwen3)   CROSS    IN-DIST   CROSS CROSS
#    7B          DOWN     CROSS     IN-DIST UP
#    14B         DOWN     CROSS     DOWN  IN-DIST
#
#    Example commands:
#    python -m scripts.eval_probe_transfer \
#      --probe_dir models/probes_transfer/qwen2.5_1.5b_super_gen \
#      --test_models qwen2.5_7b,qwen2.5_14b \
#      --test_data_dir data/probe_transfer_standard
#
# 3. FEATURE COMPARISON
#    Compare feature sets:
#    - Job 0 (generalizable + hidden): 1030 features (6 scalars + 1024 hidden)
#    - Job 4 (super-generalizable): 6 features (scalars only)
#    
#    Note: With standard prompts, Jobs 0 and 4 use SAME scalars
#    The only difference is hidden states (1024 dims)
#    
#    Expected: Job 0 > Job 4 on in-distribution (hidden states help)
#              Job 4 > Job 0 on transfer (hidden states hurt)
#
# 4. KEY HYPOTHESES TO TEST:
#    H1: Up-scaling helps (14B → 7B better than 1.5B → 7B)
#    H2: Mid-size is optimal for transfer (7B transfers best)
#    H3: Down-scaling hurts less than up-scaling
#    H4: Cross-generation (Qwen3 → Qwen2.5) works with scalars only
#    H5: Hidden states prevent transfer (Job 0 fails, Job 4 succeeds)
#
# ==============================================================================
