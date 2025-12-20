# Confidence Probes for Correctness Estimation in Large Language Models

This repository is the codebase to collect data, train confidence probes, and then evaluate these probes on 5 datasets: TriviaQA, SQuAD2.0, TriviaQA, GSM8K, and MMLU. 
Note that we have used `Python 3.9.23` and a conda environment containing all the libraries defined in our `requirements.txt`.

## Step 1: Data Collection
Data can be collected using the following (example) command:

```
python -m scripts.collect_internals \
  --backend qwen \                                        # qwen | llama31
  --model_id "Qwen/Qwen2.5-7B-Instruct" \                 # model id from huggingface. (we use qwen2.5-1.5B | qwen2.5-7B | qwen2.5-14B | qwen3-4B | llama3.1-8B)
  --dataset triviaqa --split train \                      # we always use train splits for this section
  --max_new_tokens 64 \
  --num_shards "$num_shards" --shard_id "$shard_id" \     # use if needed. else, simply use --num shards 1 --shard_id 0
  --out data/probe_standard/qwen2.5_7b/triviaqa/train_shard${shard_id}.jsonl    # or desired output dir
```

We run this for all desired models and all datasets.

## Step 2: Create data splits
In this step, we create dataset splits to trin our probes on. Here, there are two scripts to run.

### Pooled Dataset Splits
Here, we create splits from ALL datasets using this command:
```
bash scripts/create_probe_splits.sh
```

### Splits-by-Dataset
For cross-dataset training, we must create new data splits:
```
bash scripts/create_probe_splits_per_dataset.sh
```

Note that output paths can be changed as desired by changing the paths in the bash scripts as desired (you will need to follow along with file destinations though).

## Step 3: Train Probes
Probes can be trained using commands like:

```
    python -m scripts.train_probe \
        --data_dir data/probe_splits_standard/qwen2.5_7b/backend_qwen2.5_7b \ # directory with probe split map
        --output_dir models/probes_standard/qwen2.5_7b_generalizable \        # where probe models will be saved
        --probe_type mlp \
        --use_hidden \
        --only_generalizable
```
for generalizable probes, or:

```
  python -m scripts.train_probe \
      --data_dir data/probe_splits_standard/qwen2.5_1.5b/backend_qwen2.5_1.5b \
      --output_dir models/probes_transfer/qwen2.5_1.5b_super_gen \
      --probe_type mlp \
      --super_generalizable
```
for super-generalizable probes, or using commands such as:
```
python -m scripts.train_probe \
    --data_dir data/probe_splits_per_dataset/${MODEL}/${DATASET}/backend_${MODEL} \
    --output_dir models/probes_cross_dataset/${MODEL}_${DATASET}_only \
    --probe_type mlp \
    --use_hidden \
    --only_generalizable
```
for cross-dataset probes (here, probes are trained using data from just one dataset).

## Step 4: Probe evaluation
Example eval script:
```
python -m scripts.eval_probe_optimized \
    --model qwen \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --probe_dir models/probes_standard/qwen2.5_7b_generalizable/probe_mlp \
    --datasets triviaqa,hotpotqa,squad_v2,gsm8k,mmlu \
    --output_dir results/baseline/qwen2.5_7b \
```
Note that this can take (many) hours to run depending on the datasets used. This script can be adapted for cross-dataset or cross-model experiments. Cross-dataset example:
```
python -m scripts.eval_probe_cross_datasets \
    --model llama \
    --model_id "meta-llama/Llama-3.1-8B-Instruct" \
    --datasets mmlu,gsm8k \
    --probe_map_json "data/llama_cross_dataset_probes.json" \
    --output_dir "results/cross_datasets/llama"
```

After eval is run, graphs and summaries (in JSON format) can be found in `output_dir`.
