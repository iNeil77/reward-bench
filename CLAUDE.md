# RewardBench Development Guide

## Package Manager

This project uses **uv** for dependency management. Always use `uv` commands:

```bash
# Install base dependencies
uv sync

# Install with API clients (OpenAI, Anthropic, etc.) for LLM-as-judge
uv sync --extra api

# Install with vLLM for local LLM inference (Linux + CUDA only)
uv sync --extra vllm

# Install everything (api + vllm)
uv sync --extra generative

# Run commands
uv run python scripts/run_rm.py
uv run rewardbench --help
```

## Optional Extras

- `api` - API-based LLM clients (openai, anthropic, google-genai, together) - works on any platform
- `vllm` - Local LLM inference via vLLM - Linux + CUDA only, pins torch to 2.9
- `generative` - Both api + vllm (backwards compatible alias)
- `v1` - Legacy dependencies (fschat, trl) for v1 scripts
- `dev` - Development tools (black, flake8, isort, pytest)

## Default Configuration

**Default PyTorch dtype: bfloat16**
- All scripts default to `bfloat16` for better numerical stability and modern GPU compatibility
- Override with `--torch_dtype` if needed: `--torch_dtype=float16` for older GPUs

**Default Attention Implementation: flash_attention_2**
- All scripts default to Flash Attention 2 for faster inference on modern GPUs
- Requires `flash-attn` package (installed automatically with base dependencies)
- Override with `--attn_implementation` if needed: `--attn_implementation=sdpa` or `--attn_implementation=eager`
- Falls back to SDPA or eager if flash-attn is not available

**Parallelism Configuration**
- `--num_proc=8`: Number of processes for dataset operations (map, filter)
- `--dataloader_num_workers=4`: Number of worker processes for PyTorch DataLoader
- Set to 0 to disable parallelism for debugging
- Recommended: Use 4-8 workers on multi-core systems for optimal performance

## Version Pinning Policy

**Always pin `transformers` and `vllm` versions** to avoid dependency headaches. These packages have frequent breaking changes and complex dependency trees.

Current pinned versions:
- `transformers==5.6.2`
- `flash-attn>=2.7.2` (currently resolves to 2.8.3)
- `vllm>=0.18.0` (in `[vllm]` extra, currently resolves to 0.20.0)

When updating these versions:
1. Update the pin in `pyproject.toml`
2. Run `uv lock` to update the lock file
3. Test the entry points: `uv run rewardbench --help` and `uv run rewardbench-gen --help`
4. Run tests: `uv run pytest`

### Transformers 5.x Compatibility

The codebase now supports both transformers 4.x and 5.x. Key changes for 5.x compatibility:

- **Quantization**: Uses `BitsAndBytesConfig(load_in_8bit=True)` instead of direct `load_in_8bit` parameter
- **LlamaTokenizer**: Falls back to `AutoTokenizer` if `LlamaTokenizer` is not available (removed in 5.x)
- **Performance Optimizations**: 
  - `use_safetensors=True` - Faster and safer weight loading
  - `low_cpu_mem_usage=True` - Reduces memory peaks during model loading
  - `HF_HUB_ENABLE_HF_TRANSFER=1` - Uses hf_transfer for faster downloads (requires `hf_transfer` package)

To test with transformers 5.x:
```bash
pip install "transformers>=5.0"
rewardbench --model=<model-path> --batch-size=32
```

**Note**: Weight loading in transformers 5.x is optimized with safetensors. Ensure your models use `.safetensors` format for best performance. PyTorch `.bin` files will still work but load slower.

## Docker Images

Two Docker images are available:

| Image | Dockerfile | Use Case | Build Time |
|-------|------------|----------|------------|
| `rewardbench` | `Dockerfile` | Reward models, API-based judges | ~5-10 min |
| `rewardbench-vllm` | `Dockerfile.vllm` | Local LLM inference via vLLM | ~45 min |

The base image uses prebuilt flash-attn wheels (torch ≤2.8). The vllm image builds flash-attn from source (torch 2.9 required by vllm).

## Entry Points

- `rewardbench` - Main evaluation CLI (works with base install)
- `rewardbench-gen` - Generative RM evaluation (requires `[api]` or `[generative]` extra)

## Code Quality Checks

Run these before committing:

```bash
# Format code (automatically fixes formatting)
uv run black .
uv run isort .

# Check formatting only (for CI)
uv run black --check .
uv run isort --check-only .

# Lint
uv run flake8 --max-line-length 120 rewardbench/ scripts/
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_data.py
```


## Common Issues

### Import errors for generative modules

If you see `ModuleNotFoundError: No module named 'anthropic'` or similar, you need to install with the generative extra:

```bash
uv sync --extra generative
```

### vLLM platform issues

vLLM has specific platform requirements. On unsupported platforms (like aarch64/ARM), you may need custom wheels. See the main CLAUDE.md in `~/dev/` for DGX Spark-specific instructions.

## AI2 Beaker Scripts

The following scripts are AI2-internal and require access to Beaker infrastructure:
- `scripts/submit_eval_jobs.py`
- `scripts/submit_eval_jobs_v2.py`
- `scripts/submit_generative_jobs.py`

External users should use the `run_*.py` scripts directly instead. All Beaker-specific functionality has been removed from the main evaluation scripts.
