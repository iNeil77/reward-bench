<div align="center">
  <h1>RewardBench: Evaluating Reward Models</h1>
  <p> V2 (<strong>NEW!</strong>):
  <a href="https://huggingface.co/spaces/allenai/reward-bench">Leaderboard</a> 📐 |
  <a href="https://huggingface.co/datasets/allenai/reward-bench-2">Eval. Dataset</a> |
  <a href="https://huggingface.co/datasets/allenai/reward-bench-2-results">Results</a> 📊 | 
  <a href="https://huggingface.co/collections/allenai/reward-bench-2-683d2612a4b3e38a3e53bb51">Trained Models</a> 🏆 | 
  <a href="https://arxiv.org/abs/2506.01937"> Paper📝 </a>
</p>

  <p> V1:
  <a href="https://huggingface.co/spaces/allenai/reward-bench">Leaderboard</a> 📐 |
  <a href="https://huggingface.co/datasets/allenai/reward-bench">Eval. Dataset</a> |
  <a href="https://huggingface.co/datasets/allenai/preference-test-sets">Existing Test Sets</a> |
  <a href="https://huggingface.co/datasets/allenai/reward-bench-results">Results</a> 📊 |
  <a href="https://arxiv.org/abs/2403.13787"> Paper📝</a>
</p>
  <img width="1280" alt="Github RewardBench Logo" src="https://github.com/allenai/reward-bench/assets/10695622/39b213ba-9971-4338-b5f9-8e042d22d8fc" style="margin-left:'auto' margin-right:'auto' display:'block' "/>
</div>
<p align="center">
  <a href="https://github.com/allenai/reward-bench/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/reward-bench">
  </a>
  <a href="https://pypi.org/project/rewardbench/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/rewardbench">
  </a>
</p>

---

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [RewardBench CLI](#rewardbench-cli)
  - [RewardBench 2 (Scripts)](#rewardbench-2-scripts)
  - [Generative Models (LLM-as-judge)](#generative-models-llm-as-judge)
  - [DPO Models](#dpo-models)
- [Configuration & Performance](#configuration--performance)
- [Advanced Usage](#advanced-usage)
  - [Custom Datasets](#custom-datasets)
  - [Result Uploading](#result-uploading)
  - [Ensembling Models](#ensembling-models)
  - [Best-of-N Rankings](#best-of-n-rankings)
- [Development](#development)
- [Docker & Maintenance](#docker--maintenance)
- [Citation](#citation)

---

## Overview

**RewardBench** is a benchmark designed to evaluate the capabilities and safety of reward models (including those trained with Direct Preference Optimization, DPO).

**Features:**
- Common inference code for reward models (Starling, PairRM, OpenAssistant, DPO, and more)
- Unified dataset formatting for fair evaluation
- Analysis and visualization tools
- Support for both traditional reward models and generative judges (LLM-as-judge)

**Key Components:**
- `rewardbench` CLI: Quick evaluation on core dataset
- `scripts/run_v2.py`: RewardBench 2 with best-of-4 and Ties data
- `scripts/run_rm.py`: Advanced reward model evaluation
- `scripts/run_dpo.py`: DPO model evaluation
- `scripts/run_generative_v2.py`: LLM-as-judge evaluation

---

## Installation

### Quick Install

**With UV (recommended):**
```bash
# Base install (uses SDPA attention, excellent performance)
uv pip install rewardbench

# With Flash Attention 2 for maximum speed (optional, requires CUDA toolkit)
pip install ninja  # speeds up compilation
uv pip install rewardbench[flash-attn]

# For LLM-as-judge (API providers + vLLM)
uv pip install rewardbench[generative]
```

**With pip:**
```bash
pip install rewardbench
pip install rewardbench[flash-attn]    # optional: Flash Attention 2
pip install rewardbench[generative]    # optional: generative models
```

**One-off usage (no install):**
```bash
uv run --with rewardbench rewardbench --model=your-model
```

### Development Install

```bash
# Clone repository
git clone https://github.com/allenai/reward-bench.git
cd reward-bench

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install in editable mode
uv sync                        # base
uv sync --extra generative     # with generative support
uv sync --extra dev            # with development tools

# Set HuggingFace token
export HF_TOKEN="your_token_here"
```

**Requirements:**
- Python 3.10+ (3.10, 3.11, 3.12 supported)
- PyTorch 2.0+ (auto-installed)
- CUDA toolkit (for GPU support)

**Key versions:**
- transformers: 5.6.2
- flash-attn: 2.8.3 (optional, in `[flash-attn]` extra)
- vLLM: 0.18+ (optional, in `[generative]` extra)

---

## Quick Start

**Evaluate a reward model:**
```bash
rewardbench --model=OpenAssistant/reward-model-deberta-v3-large-v2
```

**Evaluate on RewardBench 2:**
```bash
python scripts/run_v2.py --model=your-model --batch_size=64
```

**Evaluate DPO model:**
```bash
rewardbench --model=Qwen/Qwen1.5-0.5B-Chat --ref_model=Qwen/Qwen1.5-0.5B
```

**With uv (from editable install):**
```bash
uv run rewardbench --model=your-model
uv run python scripts/run_v2.py --model=your-model
```

---

## Usage

### RewardBench CLI

The `rewardbench` binary evaluates models on the RewardBench v1 core set.

**Basic usage:**
```bash
# Standard evaluation
rewardbench --model=your-model

# With options
rewardbench \
    --model=your-model \
    --batch_size=32 \
    --num_proc=16 \
    --attn_implementation=sdpa

# DPO model
rewardbench \
    --model=your-dpo-model \
    --ref_model=base-model

# Custom dataset
rewardbench \
    --model=your-model \
    --dataset=allenai/ultrafeedback_binarized_cleaned \
    --split=test_gen

# Local JSON dataset
rewardbench \
    --model=your-model \
    --dataset=/path/to/dataset.jsonl \
    --load_json
```

**With uv:**
```bash
# From editable install
uv run rewardbench --model=your-model

# One-off (no install)
uv run --with rewardbench rewardbench --model=your-model
```

**Help:**
```bash
rewardbench --help
```

### RewardBench 2 (Scripts)

RewardBench 2 includes best-of-N and Ties evaluation. Use `scripts/run_v2.py`:

```bash
# Basic usage
python scripts/run_v2.py --model=your-model

# With performance tuning
python scripts/run_v2.py \
    --model=your-model \
    --batch_size=128 \
    --num_proc=16 \
    --dataloader_num_workers=8

# With Flash Attention 2 (requires flash-attn extra)
python scripts/run_v2.py \
    --model=your-model \
    --attn_implementation=flash_attention_2

# Advanced reward model script (more control)
python scripts/run_rm.py \
    --model=your-model \
    --batch_size=16 \
    --num_proc=32 \
    --dataloader_num_workers=4
```

**Config reference:**
See [eval_configs.yaml](scripts/configs/eval_configs.yaml) for model-specific configurations.

### Generative Models (LLM-as-judge)

Evaluate LLM-based reward models. Requires `[generative]` extra.

**Rankings-based (default, compares 4 responses):**
```bash
python scripts/run_generative_v2.py --model=gpt-4
python scripts/run_generative_v2.py --model=meta-llama/Llama-3-70b-chat-hf
```

**Ratings-based (scores each response separately):**
```bash
python scripts/run_generative_v2.py --model=your-model --score_w_ratings
```

**Using the CLI:**
```bash
rewardbench-gen --model=gpt-3.5-turbo-0125
```

**Supported providers:**
- OpenAI (gpt-4, gpt-3.5-turbo)
- Anthropic (claude-3-*)
- Google (gemini-*)
- Together AI
- Local models via vLLM (Linux + CUDA only)

**Note:** Ties subset (20+ completions) automatically uses ratings mode.

### DPO Models

Evaluate Direct Preference Optimization models:

```bash
# Via CLI
rewardbench \
    --model=stabilityai/stablelm-zephyr-3b \
    --ref_model=stabilityai/stablelm-3b-4e1t \
    --batch_size=64

# Via script (more control)
python scripts/run_dpo.py \
    --model=your-dpo-model \
    --ref_model=base-model \
    --batch_size=8
```

---

## Configuration & Performance

### Default Settings

Optimized for modern GPUs (Ampere+):
- **dtype**: `bfloat16` (better stability than float16)
- **attn_implementation**: `sdpa` (PyTorch native, works everywhere)
- **num_proc**: `8` (dataset parallelism)
- **dataloader_num_workers**: `4` (DataLoader workers)

### Attention Implementations

**SDPA (default):**
- Excellent performance on modern GPUs
- Works out-of-box, no compilation
- PyTorch 2.0+ native

**Flash Attention 2 (optional):**
- 2-4x faster on Ampere+ GPUs (A100, H100, RTX 30/40)
- Requires installation: `pip install rewardbench[flash-attn]`
- Build time: 5-10 min with ninja, 30-45 min without
- Use: `--attn_implementation=flash_attention_2`

**Eager (fallback):**
- Use for CPU or debugging: `--attn_implementation=eager`

### Performance Tuning

```bash
# High-performance system (16+ cores, A100/H100)
python scripts/run_v2.py \
    --model=your-model \
    --batch_size=128 \
    --num_proc=16 \
    --dataloader_num_workers=8

# Maximum speed with Flash Attention 2
python scripts/run_v2.py \
    --model=your-model \
    --batch_size=128 \
    --attn_implementation=flash_attention_2

# Debugging (single-threaded)
python scripts/run_v2.py \
    --model=your-model \
    --batch_size=32 \
    --num_proc=1 \
    --dataloader_num_workers=0

# Older GPUs (use float16)
python scripts/run_v2.py \
    --model=your-model \
    --torch_dtype=float16
```

---

## Advanced Usage

### Custom Datasets

**From HuggingFace:**
```bash
rewardbench \
    --model=your-model \
    --dataset=allenai/ultrafeedback_binarized_cleaned \
    --split=test_gen
```

**From local JSON:**
```bash
rewardbench \
    --model=your-model \
    --dataset=/path/to/dataset.jsonl \
    --load_json
```

**Instruction datasets:**
RewardBench auto-detects instruction datasets (no `chosen`/`rejected`, has `messages`) and logs model outputs without accuracy.

### Result Uploading

Upload results to HuggingFace Hub:

```bash
# Upload results as dataset
rewardbench \
    --model=your-model \
    --push_results_to_hub

# Add results to model card metadata
rewardbench \
    --model=your-model \
    --upload_model_metadata_to_hf

# Both
rewardbench \
    --model=your-model \
    --push_results_to_hub \
    --upload_model_metadata_to_hf
```

**Examples:**
- Model with metadata: [vwxyzjn/rm_zephyr_new](https://huggingface.co/vwxyzjn/rm_zephyr_new)
- Preference dataset outputs: [natolambert/rewardbench_eval_2339270924_2339270924](https://huggingface.co/datasets/natolambert/rewardbench_eval_2339270924_2339270924)

### Ensembling Models

Run offline ensemble tests to approximate using multiple reward models:

```bash
python analysis/run_ensemble_offline.py \
    --models sfairXC/FsfairX-LLaMA3-RM-v0.1 \
             openbmb/Eurus-RM-7b \
             Nexusflow/Starling-RM-34B
```

**Generative ensembles (API only):**
```bash
python scripts/run_generative.py \
    --model gpt-3.5-turbo-0125 \
            claude-3-sonnet-20240229 \
            meta-llama/Llama-3-70b-chat-hf
```
Note: Must use odd number of models > 1.

### Best-of-N Rankings

Create rankings across datasets:

```bash
python scripts/run_bon.py \
    --model=OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5 \
    --best_of=16
```

### Leaderboard Section Scores

Compute prompt-weighted scores for Chat, Chat Hard, Safety, and Reasoning:

```python
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rewardbench.utils import calculate_scores_per_section

metrics = {
  "alpacaeval-easy": 0.5,
  "alpacaeval-hard": 0.7052631578947368,
  # ... (subset scores)
  "model": "your-model",
  "model_type": "Seq. Classifier",
}

scores_per_section = calculate_scores_per_section(
    EXAMPLE_COUNTS, SUBSET_MAPPING, metrics
)
print(scores_per_section)
```

---

## Development

### Contributing Models

To add your model to the leaderboard:
1. Open an issue with the model name on HuggingFace
2. If custom code is needed, open a PR ([see `rewardbench/models`](rewardbench/models))

Local models can be evaluated without submission.

### Training

For training reward models, use [`open-instruct`](https://github.com/allenai/open-instruct).

### Repository Structure

```
├── README.md                   <- This file
├── analysis/                   <- Analysis tools
├── rewardbench/                <- Core utils and models
│   ├── models/                 <- Model implementations
│   └── *.py                    <- Utilities
├── scripts/                    <- Evaluation scripts
│   ├── run_v2.py              <- RewardBench 2
│   ├── run_rm.py              <- Reward models
│   ├── run_dpo.py             <- DPO models
│   └── configs/               <- Model configs
├── tests/                      <- Unit tests
├── Dockerfile                  <- Docker build
├── pyproject.toml             <- Package config (uv)
└── CLAUDE.md                  <- Development guide
```

### Code Quality

```bash
# Format code
uv run black .
uv run isort .

# Lint
uv run flake8 --max-line-length 120 rewardbench/ scripts/

# Test
uv run pytest
```

---

## Docker & Maintenance

### Docker Images

Two images available:

| Image | Dockerfile | Use Case | Build Time |
|-------|------------|----------|------------|
| `rewardbench` | `Dockerfile` | Reward models, API judges | ~5-10 min |
| `rewardbench-vllm` | `Dockerfile.vllm` | Local LLM inference (vLLM) | ~45 min |

**Build locally:**
```bash
docker build -t rewardbench . --platform linux/amd64
docker build -f Dockerfile.vllm -t rewardbench-vllm . --platform linux/amd64
```

**Auto-built on main:**
- `nathanl/rewardbench_auto`: Base image
- `nathanl/rewardbench_vllm_auto`: vLLM image

### AI2 Infrastructure

For AI2 users only:
```bash
# Submit evaluation jobs
python scripts/submit_eval_jobs.py

# Best-of-N sweep
python scripts/submit_eval_jobs.py --eval_on_bon --image=nathanl/herm_bon

# Note: Set beaker secret: beaker secret write HF_TOKEN <token>
```

---

## Citation

If you use RewardBench in your research, please cite:

**RewardBench 2:**
```bibtex
@misc{malik2025rewardbench2advancingreward,
      title={RewardBench 2: Advancing Reward Model Evaluation}, 
      author={Saumya Malik and Valentina Pyatkin and Sander Land and Jacob Morrison and Noah A. Smith and Hannaneh Hajishirzi and Nathan Lambert},
      year={2025},
      eprint={2506.01937},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01937}, 
}
```

**RewardBench (v1):**
```bibtex
@misc{lambert2024rewardbench,
      title={RewardBench: Evaluating Reward Models for Language Modeling}, 
      author={Nathan Lambert and Valentina Pyatkin and Jacob Morrison and LJ Miranda and Bill Yuchen Lin and Khyathi Chandu and Nouha Dziri and Sachin Kumar and Tom Zick and Yejin Choi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2403.13787},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
