# Astrolabe

**Astrolabe: Predictive Load Balancing for Large Language Model Serving**

Astrolabe is a research prototype that explores *predictive, performance-aware scheduling* for distributed large-language-model (LLM) inference. It builds on top of Microsoft's [Vidur](https://github.com/microsoft/vidur) simulator and adds:

* A side-car **Predictor** service that forecasts per-instance latency metrics using simulation at runtime
* A **Global Scheduler** that uses these predictions to route requests optimally
* Tooling for training a lightweight length-estimator model so the scheduler can reason about prompts it has never seen before

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Global Scheduler                             │
│    (Receives requests, queries Predictors, applies scheduling)       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
      │  Predictor 1 │      │  Predictor 2 │      │  Predictor N │
      │  + vLLM      │      │  + vLLM      │      │  + vLLM      │
      │  Instance    │      │  Instance    │      │  Instance    │
      └──────────────┘      └──────────────┘      └──────────────┘
           GPU 1                 GPU 2                 GPU N
```

### Components

- **Predictor** (`astrolabe/predictor/`): Co-locates with every inference node. Collects live stats or runs Vidur simulation on-demand to answer "What if I got one more request?"

- **Global Scheduler** (`astrolabe/global_scheduler/`): Receives requests, queries Predictors, and applies the scheduling policy (default: Astrolabe; alternatives: Llumnix-, round-robin, random, etc.)

- **Length Estimator** (`astrolabe/length_estimation/`): A RoBERTa-based regressor that predicts response token count for unseen prompts, enabling input-aware cost estimates.

Astrolabe is inference-engine agnostic. We provide an implementation for vLLM 0.7.2.

---

## 2. Repository Layout

```
astrolabe/
├── predictor/              # Side-car prediction service
│   ├── api_server.py       # FastAPI server for predictor
│   ├── simulate_predictor.py  # Vidur-based simulation predictor
│   └── predictor.py        # Core predictor logic
├── global_scheduler/       # Request router
│   ├── api_server.py       # FastAPI server for global scheduler
│   └── instance.py         # Instance state management
├── length_estimation/      # Training & inference of token-length regressor
│   ├── train_roberta.py    # Training script
│   └── eval_roberta.py     # Evaluation and data tagging
├── benchmark/              # Load generator (forked from vLLM)
│   └── benchmark_serving.py
├── config/
│   ├── host_configs.json   # Cluster description template
│   ├── llama_config.json   # Llama-7B model config
│   └── llama70b_config.json # Llama-70B model config
└── exp/
    ├── setup.sh            # Installs dependencies & deploys cluster
    ├── experiment.sh       # Main experiment runner
    ├── generate_config.py  # Config generator from cluster manifest
    └── end_to_end_exp_scripts/
        ├── a30_main/       # Main experiment scripts (12x A30)
        ├── a100_supplementary/  # A100 cluster experiments
        └── ablation/       # Ablation studies

vidur/                      # Modified Vidur simulator
├── scheduler/
│   ├── global_scheduler/   # Global scheduling policies
│   └── replica_scheduler/  # Per-instance scheduling
├── execution_time_predictor/  # Latency prediction models
└── types/                  # Enum definitions for schedulers

experiments_analysis/       # Plotting and analysis scripts
├── experiment_plot.py      # Main results plotting
├── burstiness_plot.py      # Burstiness study plots
├── po2_plot.py             # Power-of-2 ablation plots
└── cpu_overhead_plot.py    # CPU overhead analysis
```

---

## 3. Quick Start

### 3.1 Prerequisites

- Python 3.10+
- CUDA 12.6+
- PyTorch 2.5+
- vLLM 0.7.2 (modified version)

### 3.2 Cluster Setup

1. **Generate cluster configuration** from your cluster manifest:
   ```bash
   python astrolabe/exp/generate_config.py --username YOUR_SSH_USERNAME
   ```
   Or manually create configuration files following examples in `astrolabe/config/`.

2. **Deploy software stack**:
   ```bash
   bash astrolabe/exp/setup.sh
   ```

3. **Configure credentials**: Set your HuggingFace token in `exp/run_exp_vllm.sh`:
   ```bash
   export HF_TOKEN="YOUR_HF_TOKEN_HERE"
   ```

### 3.3 Running Experiments

Run the main end-to-end experiment:
```bash
bash astrolabe/exp/end_to_end_exp_scripts/a30_main/main_experiment.sh
# Results saved to experiment_output/data/
```

### 3.4 Plotting Results

Generate figures from experiment data:
```bash
export PYTHONPATH=.
python experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/main/sharegpt \
    --output-dir experiment_output/results/main \
    --system-name Astrolabe
```

---

## 4. Experiment Scripts

All experiment scripts are in `astrolabe/exp/end_to_end_exp_scripts/`:

| Script | Description | Paper Reference |
|--------|-------------|-----------------|
| `a30_main/main_experiment.sh` | Main results (12x A30 GPUs) | Figure 6, Section 5.3 |
| `ablation/po2_ablation_exp.sh` | Power-of-2 sampling ablation | Section 5.4 |
| `ablation/burstiness_exp.sh` | Burstiness sensitivity study | Section 5.4 |
| `ablation/error_heatmap_exp.sh` | Prediction error sensitivity | Section 5.4 |
| `ablation/cpu_tracking_experiment.sh` | CPU overhead analysis | Section 5.4 |
| `a100_supplementary/full_comparison.sh` | Llumnix migration comparison | Section 5.5 |
| `ablation/auto_provision_exp.sh` | Auto-provisioning experiment | Section 5.5 |
| `ablation/extension_experiment.sh` | Qwen model & BurstGPT dataset | Section 5.6 |

---

## 5. Training the Length Estimator

To train a new length estimation model:

1. **Download dataset**:
   ```bash
   wget https://huggingface.co/datasets/shibing624/sharegpt_gpt4/blob/main/sharegpt_gpt4.jsonl
   ```

2. **Sample and train**:
   ```bash
   python astrolabe/length_estimation/sample.py
   python astrolabe/length_estimation/train_roberta.py
   ```

3. **Tag dataset with predictions**:
   ```bash
   python astrolabe/length_estimation/eval_roberta.py --tag-data True
   ```

---

## 6. Extending Astrolabe

### Adding a New Scheduling Policy

1. Implement metrics export from the inference engine to the Predictor
2. Define load scoring in `simulate_predictor.py`
3. Add the scheduler name to `vidur/types/optimal_global_scheduler_target_metric.py`
4. Update experiment scripts to use the new scheduler

### Supporting a New Model

1. Profile the model following [Vidur instructions](https://github.com/microsoft/vidur/blob/main/docs/profiling.md)
2. Add model config to `vidur/config/model_config.py`
3. Create a config file in `astrolabe/config/`
4. (Optional) Train a length estimator for the new model

### Supporting a New Inference Engine

Expose required metrics via API, following the vLLM implementation in `vllm/entrypoints/api_server.py` as reference.

---

## 7. Scheduling Policies

Astrolabe supports multiple scheduling policies via the `--scheduler` flag:

| Policy | Description |
|--------|-------------|
| `min_new_request_latency` | **Astrolabe** (default) - Predictive scheduling minimizing new request latency |
| `min_total_unprocessed_tokens` | **Astrolabe-NoSim** - Heuristic without simulation |
| `min_lunmnix_load` | **Llumnix-** - Llumnix dispatcher heuristic |
| `round_robin` | Round-robin distribution |
| `random` | Random instance selection |

---

## 8. Configuration

### Model Configuration (`astrolabe/config/llama_config.json`)

```json
{
  "model_name": "meta-llama/Llama-2-7b-hf",
  "max_batch_size": 48,
  "chunk_size": 512,
  "block_size": 16
}
```

### Cluster Configuration (`astrolabe/config/host_configs.json`)

```json
{
  "nodes": [
    {"hostname": "node1", "gpu_type": "A30", "num_gpus": 1},
    {"hostname": "node2", "gpu_type": "A30", "num_gpus": 1}
  ],
  "predictor_port": 8100,
  "vllm_port": 8000
}
```

---

## 9. Generating Figures

All plotting scripts are in `experiments_analysis/`. Each script supports the `--system-name` argument (default: "Astrolabe").

### 9.1 Main Results (Figure 6)

```bash
export PYTHONPATH=.
python experiments_analysis/experiment_plot.py \
    --experiments-dir experiment_output/data/main/sharegpt \
    --output-dir experiment_output/results/main \
    --system-name Astrolabe
```

**Output files:**
- `qps.png` / `qps.pdf` - QPS vs latency/throughput
- `linear.png` / `linear.pdf` - Linear scaling analysis
- `cdf.png` / `cdf.pdf` - Latency CDF

### 9.2 Power-of-2 Ablation

```bash
python experiments_analysis/po2_plot.py \
    --data-dir experiment_output/data/extra_ablation/po2 \
    --output-dir experiment_output/results/extra_ablation \
    --system-name Astrolabe
```

### 9.3 Burstiness Study

```bash
python experiments_analysis/burstiness_plot.py \
    --data-dir experiment_output/data/extra_ablation/burstiness \
    --output-dir experiment_output/results/extra_ablation \
    --system-name Astrolabe
```

### 9.4 Prediction Error Heatmap

```bash
python experiments_analysis/prediction_error_plot.py \
    --data-dir experiment_output/data/extra_ablation/prediction_error \
    --output-dir experiment_output/results/extra_ablation \
    --system-name Astrolabe
```

### 9.5 CPU Overhead Analysis

```bash
python experiments_analysis/cpu_overhead_plot.py \
    --data-dir experiment_output/data/extra_ablation/cpu_tracker/sharegpt/min_new_request_latency \
    --output-dir experiment_output/results/extra_ablation \
    --system-name Astrolabe
```

### 9.6 Llumnix Migration Comparison

```bash
python experiments_analysis/llumnix_comparison_plot.py \
    --data-dir experiment_output/data/extra_ablation/llumnix_compare \
    --output-dir experiment_output/results/extra_ablation \
    --system-name Astrolabe
```

---

## 10. Requirements

See `requirements.txt` for full dependencies. Key packages:

- Python 3.10+
- PyTorch 2.5+
- CUDA 12.6+
- vLLM 0.7.2 (modified)
- flashinfer-python 0.2.5
- triton 3.2.0
- transformers
- fastapi
- uvicorn

---

## 11. License

This work is released under the MIT license. See `LICENSE` for details.

---

## 12. Artifact Evaluation

For artifact evaluation, please follow these steps:

1. **Setup**: Follow Section 3 to configure the cluster
2. **Main Results**: Run `main_experiment.sh` (~50 hours on 12x A30)
3. **Ablations**: Run scripts in `ablation/` directory
4. **Plotting**: Use scripts in `experiments_analysis/`

All experiment outputs are saved to `experiment_output/data/` and figures to `experiment_output/results/`.