#!/bin/bash
#SBATCH --job-name=eval-baseline
#SBATCH --partition=vulcan-scavenger
#SBATCH --qos=vulcan-scavenger
#SBATCH --account=vulcan-abhinav
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0-14

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
    echo "Evaluating: Qwen2.5-1.5B Probe on Qwen2.5-4B Model"
    echo ""
    
    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen3-4B-Instruct-2507" \
        --probe_dir models/probes_transfer/qwen2.5_1.5b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_1.5b_super_gen/qwen3_4b \
        --debug_first_n 10
    
    echo "Qwen2.5-1.5B Probe on Qwen2.5-4B Model evaluation complete"
    ;;

1)
    echo "Evaluating: Qwen2.5-1.5B Probe on Qwen2.5-7B Model"
    echo ""
    
    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-7B-Instruct" \
        --probe_dir models/probes_transfer/qwen2.5_1.5b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_1.5b_super_gen/qwen2.5_7b \
        --debug_first_n 10
    
    echo "Qwen2.5-1.5B Probe on Qwen2.5-7B Model evaluation complete"
    ;;

2)
    echo "Evaluating: Qwen2.5-1.5B Probe on Qwen2.5-14B Model"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-14B-Instruct" \
        --probe_dir models/probes_transfer/qwen2.5_1.5b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_1.5b_super_gen/qwen2.5_14b \
        --debug_first_n 10

    echo "Qwen2.5-1.5B Probe on Qwen2.5-14B Model evaluation complete"
    ;;

3)
    echo "Evaluating: Qwen3-4B Probe on Qwen2.5-1.5B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
        --probe_dir models/probes_transfer/qwen3_4b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen3_4b_super_gen/qwen2.5_1.5b \
        --debug_first_n 10

    echo "Qwen3-4B Probe on Qwen2.5-1.5B evaluation complete"
    ;;

4)
    echo "Evaluating: Qwen3-4B Probe on Qwen2.5-7B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-7B-Instruct" \
        --probe_dir models/probes_transfer/qwen3_4b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen3_4b_super_gen/qwen2.5_7b \
        --debug_first_n 10

    echo "Qwen3-4B Probe on Qwen2.5-7B evaluation complete"
    ;;

5)
    echo "Evaluating: Qwen3-4B Probe on Qwen2.5-14B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-14B-Instruct" \
        --probe_dir models/probes_transfer/qwen3_4b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen3_4b_super_gen/qwen2.5_14b \
        --debug_first_n 10
    
    echo "Qwen3-4B Probe on Qwen2.5-14B evaluation complete"
    ;;

6)
    echo "Evaluating: Qwen2.5-7B Probe on Qwen2.5-1.5B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
        --probe_dir models/probes_transfer/qwen2.5_7b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_7b_super_gen/qwen2.5_1.5b \
        --debug_first_n 10

    echo "Qwen2.5-7B Probe on Qwen2.5-1.5B evaluation complete"
    ;;

7)
    echo "Evaluating: Qwen2.5-7B Probe on Qwen3-4B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen3-4B-Instruct-2507" \
        --probe_dir models/probes_transfer/qwen2.5_7b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_7b_super_gen/qwen3_4b \
        --debug_first_n 10

    echo "Qwen2.5-7B Probe on Qwen3-4B evaluation complete"
    ;;

8)
    echo "Evaluating: Qwen2.5-7B Probe on Qwen2.5-14B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-14B-Instruct" \
        --probe_dir models/probes_transfer/qwen2.5_7b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_7b_super_gen/qwen2.5_14b \
        --debug_first_n 10

    echo "Qwen2.5-7B Probe on Qwen2.5-14B evaluation complete"
    ;;

9)
    echo "Evaluating: Qwen2.5-14B Probe on Qwen2.5-1.5B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
        --probe_dir models/probes_transfer/qwen2.5_14b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_14b_super_gen/qwen2.5_1.5b \
        --debug_first_n 10

    echo "Qwen2.5-14B Probe on Qwen2.5-1.5B evaluation complete"
    ;;

10)
    echo "Evaluating: Qwen2.5-14B Probe on Qwen3-4B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen3-4B-Instruct-2507" \
        --probe_dir models/probes_transfer/qwen2.5_14b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_14b_super_gen/qwen3_4b \
        --debug_first_n 10

    echo "Qwen2.5-14B Probe on Qwen3-4B evaluation complete"
    ;;

11)
    echo "Evaluating: Qwen2.5-14B Probe on Qwen2.5-7B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-7B-Instruct" \
        --probe_dir models/probes_transfer/qwen2.5_14b_super_gen/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_14b_super_gen/qwen2.5_7b \
        --debug_first_n 10

    echo "Qwen2.5-14B Probe on Qwen2.5-7B evaluation complete"
    ;;

12)
    echo "Evaluating: Qwen2.5-7B (Generalizable Probe) on Qwen2.5-1.5B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-1.5B-Instruct" \
        --probe_dir models/probes_standard/qwen2.5_7b_generalizable/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_7b_generalizable/qwen2.5_1.5b \
        --debug_first_n 10

    echo "Qwen2.5-7B (Generalizable Probe) on Qwen2.5-1.5B evaluation complete"
    ;;

13)
    echo "Evaluating: Qwen2.5-7B (Generalizable Probe) on Qwen3-4B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen3-4B-Instruct-2507" \
        --probe_dir models/probes_standard/qwen2.5_7b_generalizable/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_7b_generalizable/qwen3_4b \
        --debug_first_n 10
    
    echo "Qwen2.5-7B (Generalizable Probe) on Qwen3-4B evaluation complete"
    ;;

14)
    echo "Evaluating: Qwen2.5-7B (Generalizable Probe) on Qwen2.5-14B"
    echo ""

    python -m scripts.eval_probe_optimized \
        --model qwen \
        --model_id "Qwen/Qwen2.5-14B-Instruct" \
        --probe_dir models/probes_standard/qwen2.5_7b_generalizable/probe_mlp \
        --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
        --output_dir results/transfer/qwen2.5_7b_generalizable/qwen2.5_14b \
        --debug_first_n 10
    
    echo "Qwen2.5-7B (Generalizable Probe) on Qwen2.5-14B evaluation complete"
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
