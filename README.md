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
  - [Quick Install](#quick-install)
  - [Development Install](#development-install)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [RewardBench CLI](#rewardbench-cli)
  - [RewardBench 2 (Scripts)](#rewardbench-2-scripts)
  - [RM-Bench (Scripts)](#rm-bench-scripts)
  - [JudgeBench (Scripts)](#judgebench-scripts)
  - [Code Reward Bench (Scripts)](#code-reward-bench-scripts)
  - [Generative Models (LLM-as-judge)](#generative-models-llm-as-judge)
  - [DPO Models](#dpo-models)
- [Configuration & Performance](#configuration--performance)
  - [Default Settings](#default-settings)
  - [Attention Implementations](#attention-implementations)
  - [Performance Tuning](#performance-tuning)
- [Advanced Usage](#advanced-usage)
  - [Custom Datasets](#custom-datasets)
  - [Saving Results](#saving-results)
  - [Ensembling Models](#ensembling-models)
  - [Best-of-N Rankings](#best-of-n-rankings)
  - [Leaderboard Section Scores](#leaderboard-section-scores)
- [Development](#development)
  - [Contributing Models](#contributing-models)
  - [Training](#training)
  - [Repository Structure](#repository-structure)
  - [Code Quality](#code-quality)
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
- `scripts/run_rm_bench.py`: RM-Bench (style-robustness) evaluation
- `scripts/run_judge_bench.py`: JudgeBench (hard objective prompts) evaluation
- `scripts/run_code_reward_bench.py`: Themis Code Reward Bench (code-preference evaluation with optional aspect-specific prompts for Themis models)
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
uv run --with rewardbench rewardbench --model=your-model --do_not_save --trust_remote_code
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

> **Flags shown in every example below:**
> - `--do_not_save`: skip writing results to disk entirely (accuracy still prints to stdout). The default behavior is to write results JSON locally under `./results/` — there is no HuggingFace Hub upload, leaderboard submission, or Weights & Biases logging. Available on every runner.
> - `--trust_remote_code`: required for models that ship custom modeling code via `auto_map` (e.g., Qwen/WorldPM, some Nemotron variants). Safe to leave on by default; a no-op for stock architectures.

**Evaluate a reward model:**
```bash
rewardbench --model=OpenAssistant/reward-model-deberta-v3-large-v2 --do_not_save --trust_remote_code
```

**Evaluate on RewardBench 2:**
```bash
python scripts/run_v2.py --model=your-model --batch_size=64 --do_not_save --trust_remote_code
```

**Evaluate DPO model:**
```bash
rewardbench --model=Qwen/Qwen1.5-0.5B-Chat --ref_model=Qwen/Qwen1.5-0.5B --do_not_save --trust_remote_code
```

**With uv (from editable install):**
```bash
uv run rewardbench --model=your-model --do_not_save --trust_remote_code
uv run python scripts/run_v2.py --model=your-model --do_not_save --trust_remote_code
```

---

## Usage

### RewardBench CLI

The `rewardbench` binary evaluates models on the RewardBench v1 core set.

**Basic usage:**
```bash
# Standard evaluation
rewardbench --model=your-model --do_not_save --trust_remote_code

# With options
rewardbench \
    --model=your-model \
    --batch_size=32 \
    --num_proc=16 \
    --attn_implementation=sdpa \
    --do_not_save \
    --trust_remote_code

# DPO model
rewardbench \
    --model=your-dpo-model \
    --ref_model=base-model \
    --do_not_save \
    --trust_remote_code

# Custom dataset
rewardbench \
    --model=your-model \
    --dataset=allenai/ultrafeedback_binarized_cleaned \
    --split=test_gen \
    --do_not_save \
    --trust_remote_code

# Local JSON dataset
rewardbench \
    --model=your-model \
    --dataset=/path/to/dataset.jsonl \
    --load_json \
    --do_not_save \
    --trust_remote_code
```

**With uv:**
```bash
# From editable install
uv run rewardbench --model=your-model --do_not_save --trust_remote_code

# One-off (no install)
uv run --with rewardbench rewardbench --model=your-model --do_not_save --trust_remote_code
```

**Help:**
```bash
rewardbench --help
```

### RewardBench 2 (Scripts)

RewardBench 2 includes best-of-N and Ties evaluation. Use `scripts/run_v2.py`:

```bash
# Basic usage
python scripts/run_v2.py --model=your-model --do_not_save --trust_remote_code

# With performance tuning
python scripts/run_v2.py \
    --model=your-model \
    --batch_size=128 \
    --num_proc=16 \
    --dataloader_num_workers=8 \
    --do_not_save \
    --trust_remote_code

# With Flash Attention 2 (requires flash-attn extra)
python scripts/run_v2.py \
    --model=your-model \
    --attn_implementation=flash_attention_2 \
    --do_not_save \
    --trust_remote_code

# Advanced reward model script (more control)
python scripts/run_rm.py \
    --model=your-model \
    --batch_size=16 \
    --num_proc=32 \
    --dataloader_num_workers=4 \
    --do_not_save \
    --trust_remote_code
```

**Config reference:**
See [eval_configs.yaml](scripts/configs/eval_configs.yaml) for model-specific configurations.

### RM-Bench (Scripts)

[RM-Bench](https://github.com/THU-KEG/RM-Bench) evaluates reward models for style robustness. Each prompt has 3 chosen and 3 rejected responses across stylistic variants (concise, detailed_plain, detailed_markdown); scoring produces a 3x3 chosen-vs-rejected matrix that yields hard/normal/easy accuracy per domain (chat, math, code, safety).

The dataset JSON files are bundled under [`data/rm-bench/`](data/rm-bench/), so evaluation works out of the box.

By default, `run_rm_bench.py` writes per-example scores and metrics under `results/rm-bench/`. Pass `--do_not_save` to skip disk writes. `--trust_remote_code` is recommended if your model ships custom code via `auto_map`.

```bash
# Full eval on all domains
uv run python scripts/run_rm_bench.py \
    --model=your-model \
    --datapath=data/rm-bench/total_dataset.json \
    --batch_size=8 \
    --trust_remote_code

# Single domain (chat / code / math / safety-refuse / safety-response)
uv run python scripts/run_rm_bench.py \
    --model=your-model \
    --datapath=data/rm-bench/code_filtered.json \
    --batch_size=16 \
    --trust_remote_code

# Quick smoke test (10 examples per style variant)
uv run python scripts/run_rm_bench.py \
    --model=your-model \
    --datapath=data/rm-bench/total_dataset.json \
    --debug \
    --trust_remote_code
```

**Outputs:**
- `results/rm-bench/<model>/<dataset>_<model>_<timestamp>.json`: per-example scores (3 chosen, 3 rejected per row)
- `results/rm-bench/<model>/<dataset>_<model>_<timestamp>_metrics.json`: aggregate hard/normal/easy accuracy per domain plus `total_avg_acc`

### JudgeBench (Scripts)

[JudgeBench](https://huggingface.co/datasets/ScalerLab/JudgeBench) evaluates reward/judge models on 620 hard, objectively-verifiable prompts drawn from LiveBench, LiveCodeBench, and MMLU-Pro. Each example has a single `chosen` (correct) and `rejected` (incorrect) response; scoring reports pairwise accuracy per category (knowledge / reasoning / math / coding) plus micro/macro averages.

The dataset JSON files are bundled under [`data/judge-bench/`](data/judge-bench/), so evaluation works out of the box.

By default, `run_judge_bench.py` writes per-example scores and metrics under `results/judge-bench/`. Pass `--do_not_save` to skip disk writes. `--trust_remote_code` is recommended if your model ships custom code via `auto_map`.

```bash
# Full eval across all 4 categories
uv run python scripts/run_judge_bench.py \
    --model=your-model \
    --datapath=data/judge-bench/total_dataset.json \
    --batch_size=8 \
    --trust_remote_code

# Single category (knowledge / reasoning / math / coding)
uv run python scripts/run_judge_bench.py \
    --model=your-model \
    --datapath=data/judge-bench/coding_filtered.json \
    --batch_size=16 \
    --trust_remote_code

# Quick smoke test (10 examples)
uv run python scripts/run_judge_bench.py \
    --model=your-model \
    --datapath=data/judge-bench/total_dataset.json \
    --debug \
    --trust_remote_code
```

**Outputs:**
- `results/judge-bench/<model>/<dataset>_<model>_<timestamp>.json`: per-example chosen/rejected scores
- `results/judge-bench/<model>/<dataset>_<model>_<timestamp>_metrics.json`: per-category accuracy, per-subset breakdown, plus `micro_avg_acc` and `macro_avg_acc`

### Code Reward Bench (Scripts)

[Themis Code Reward Bench](https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench) (CRB) evaluates reward models on 8,866 pairwise code-preference rows spanning 5 aspects (`Functional_Correctness`, `Memory_Efficiency`, `Readability_Maintainability`, `Runtime_Efficiency`, `Security_Hardness`), 8 programming languages, and 19 source benchmarks. Scoring reports pairwise accuracy overall, macro-averaged over aspects, and per-aspect / per-language / per-subset.

Data is streamed directly from HuggingFace (no local bundling). For Themis reward models (`project-themis/Themis-RM-*`), `--use_system_prompts` injects the Themis judge-persona system prompt, and adding `--use_aspect_prompts` selects a per-row aspect-specific variant instead of the combined `Full` prompt. For non-Themis models those flags are ignored (with a logged warning) and the tokenizer's chat template is applied with no system message.

```bash
# Vanilla: any reward model
uv run python scripts/run_code_reward_bench.py \
    --model=your-model \
    --batch_size=16 \
    --max_length=4096 \
    --trust_remote_code

# Themis with aspect-specific system prompts (recommended for Themis)
uv run python scripts/run_code_reward_bench.py \
    --model=project-themis/Themis-RM-4B \
    --use_system_prompts \
    --use_aspect_prompts \
    --batch_size=16 \
    --max_length=4096 \
    --trust_remote_code

# Quick smoke test (32 examples)
uv run python scripts/run_code_reward_bench.py \
    --model=project-themis/Themis-RM-0.6B \
    --use_system_prompts --use_aspect_prompts \
    --debug \
    --trust_remote_code
```

**Outputs:**
- `results/code-reward-bench/<model>/<model>_<timestamp>_scores.json`: per-example chosen/rejected scores with id/subset/aspect/language
- `results/code-reward-bench/<model>/<model>_<timestamp>_metrics.json`: overall and macro-aspect accuracy plus per-aspect / per-language / per-subset breakdowns

### Generative Models (LLM-as-judge)

Evaluate LLM-based reward models. Requires `[generative]` extra.

**Rankings-based (default, compares 4 responses):**
```bash
python scripts/run_generative_v2.py --model=gpt-4 --do_not_save --trust_remote_code
python scripts/run_generative_v2.py --model=meta-llama/Llama-3-70b-chat-hf --do_not_save --trust_remote_code
```

**Ratings-based (scores each response separately):**
```bash
python scripts/run_generative_v2.py --model=your-model --score_w_ratings --do_not_save --trust_remote_code
```

**Using the CLI:**
```bash
rewardbench-gen --model=gpt-3.5-turbo-0125 --do_not_save --trust_remote_code
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
    --batch_size=64 \
    --do_not_save \
    --trust_remote_code

# Via script (more control)
python scripts/run_dpo.py \
    --model=your-dpo-model \
    --ref_model=base-model \
    --batch_size=8 \
    --do_not_save \
    --trust_remote_code
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
    --dataloader_num_workers=8 \
    --do_not_save \
    --trust_remote_code

# Maximum speed with Flash Attention 2
python scripts/run_v2.py \
    --model=your-model \
    --batch_size=128 \
    --attn_implementation=flash_attention_2 \
    --do_not_save \
    --trust_remote_code

# Debugging (single-threaded)
python scripts/run_v2.py \
    --model=your-model \
    --batch_size=32 \
    --num_proc=1 \
    --dataloader_num_workers=0 \
    --do_not_save \
    --trust_remote_code

# Older GPUs (use float16)
python scripts/run_v2.py \
    --model=your-model \
    --torch_dtype=float16 \
    --do_not_save \
    --trust_remote_code
```

---

## Advanced Usage

### Custom Datasets

**From HuggingFace:**
```bash
rewardbench \
    --model=your-model \
    --dataset=allenai/ultrafeedback_binarized_cleaned \
    --split=test_gen \
    --do_not_save \
    --trust_remote_code
```

**From local JSON:**
```bash
rewardbench \
    --model=your-model \
    --dataset=/path/to/dataset.jsonl \
    --load_json \
    --do_not_save \
    --trust_remote_code
```

**Instruction datasets:**
RewardBench auto-detects instruction datasets (no `chosen`/`rejected`, has `messages`) and logs model outputs without accuracy.

### Saving Results

All runners write results to the local filesystem only. There is **no
HuggingFace Hub upload, no leaderboard submission, and no Weights & Biases
logging**. To suppress the local write entirely (e.g., smoke testing),
pass `--do_not_save` to any runner — accuracy still prints to stdout.

```bash
# Default: write results under ./results/
rewardbench --model=your-model

# Skip all writes
rewardbench --model=your-model --do_not_save
```

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
├── data/rm-bench/              <- Bundled RM-Bench JSON eval files
├── data/judge-bench/           <- Bundled JudgeBench JSON eval files
├── rewardbench/                <- Core utils and models
│   ├── models/                 <- Model implementations
│   └── *.py                    <- Utilities
├── scripts/                    <- Evaluation scripts
│   ├── run_v2.py              <- RewardBench 2
│   ├── run_rm.py              <- Reward models
│   ├── run_rm_bench.py        <- RM-Bench (style robustness)
│   ├── run_judge_bench.py     <- JudgeBench (hard objective prompts)
│   ├── run_code_reward_bench.py <- Themis Code Reward Bench
│   ├── run_dpo.py             <- DPO models
│   ├── run_generative.py      <- Generative judges (v1)
│   ├── run_generative_v2.py   <- Generative judges (RewardBench 2)
│   ├── run_bon.py             <- Best-of-N ranking
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
