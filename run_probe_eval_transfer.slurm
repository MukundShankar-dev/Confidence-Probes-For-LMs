#!/bin/bash
#SBATCH --job-name=eval-baseline
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-1

# ==============================================================================
# BASELINE EVALUATION
# 
# Evaluates trained probes by running fresh inference on test/val datasets
# Uses standard prompts (no JSON)
# 
# Array indices:
#   0: Qwen2.5-7B
#   1: Llama-3.1-8B
# ==============================================================================

set -euo pipefail
source ~/.bashrc
conda activate llm
export PYTHONUNBUFFERED=1

mkdir -p logs results/baseline

echo "========================================================================"
echo "BASELINE PROBE EVALUATION - Array Job $SLURM_ARRAY_TASK_ID"
echo "========================================================================"
echo ""

case $SLURM_ARRAY_TASK_ID in

0)
    echo "Evaluating: Qwen2.5-7B"
    echo ""
    
    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-7B-Instruct" \
        --probe_dir models/probes_standard/qwen2.5_7b_generalizable/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/baseline/qwen2.5_7b \
        --debug_first_n 10
    
    echo "Qwen2.5-7B evaluation complete"
    ;;

1)
    echo "Evaluating: Llama-3.1-8B"
    echo ""
    
    python -m scripts.eval_probe_optimized \
        --model llama \
        --model_id "meta-llama/Llama-3.1-8B-Instruct" \
        --probe_dir models/probes_standard/llama31_8b_generalizable/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/baseline/llama31_8b \
        --debug_first_n 10
    
    echo "Llama-3.1-8B evaluation complete"
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
echo ""
echo "Results saved to: results/baseline/"
echo ""
echo "Each dataset directory contains:"
echo "  - predictions.csv (per-example predictions)"
echo "  - roc_*.png (ROC curve)"
echo "  - pr_*.png (Precision-Recall curve)"
echo "  - calibration_*.png (calibration plot)"
echo "  - confusion_*.png (confusion matrix)"
echo "  - confidence_dist_*.png (confidence histograms)"
echo ""
echo "Summary: results/baseline/{model}/summary.json"
echo ""
