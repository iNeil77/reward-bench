# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Runner for Themis Code Reward Bench (CRB):
https://huggingface.co/datasets/project-themis/Themis-CodeRewardBench

CRB is a pairwise code-reward preference benchmark. Each row has:
    prompt       - problem statement
    chosen       - preferred code response
    rejected     - dispreferred code response
    language     - programming language (Python, Cpp, Java, JavaScript, Ruby,
                   Go, C, CSharp)
    aspect       - one of Functional_Correctness, Memory_Efficiency,
                   Readability_Maintainability, Runtime_Efficiency,
                   Security_Hardness
    subset       - source benchmark (RUNBUGRUN, COMMITPREFS_*, ECCO, ...)
    id           - unique id

The runner scores chosen and rejected with a reward model and reports pairwise
accuracy broken down by aspect, language, and subset.

For standard reward models the tokenizer's chat template is applied with no
system message (vanilla behavior).

For Themis reward models (``project-themis/Themis-RM-*``), the runner
optionally swaps in an *aspect-specific* system prompt per row, matching the
recipe the Themis team uses in their reference evaluation script. The prompts
are defined in ``rewardbench.models.themis.CODE_REWARD_BENCH_ASPECT_PROMPTS``.

Example:
    uv run python scripts/run_code_reward_bench.py \\
        --model project-themis/Themis-RM-4B \\
        --batch_size 16 \\
        --max_length 4096
"""

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline

# fschat is optional - only needed if --chat_template is specified
try:
    from fastchat.conversation import get_conv_template
except ImportError:
    get_conv_template = None

# Enable faster downloads with hf_transfer (if available)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from rewardbench import (  # noqa: E402
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    torch_dtype_mapping,
)
from rewardbench.models.themis import (  # noqa: E402
    CODE_REWARD_BENCH_ASPECT_PROMPTS,
    ThemisPipeline,
)

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


############################
# Dataset loading & formatting
############################
def load_code_reward_bench(
    dataset_name: str = "project-themis/Themis-CodeRewardBench",
    config: str = "Full",
    split: str = "Full",
) -> Dataset:
    return load_dataset(dataset_name, config, split=split)


def _format_one(
    tokenizer,
    prompt: str,
    response: str,
    system_prompt: Optional[str],
) -> str:
    """Apply the tokenizer's chat template to (system?, user, assistant)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    )
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Fallback (tokenizer has no chat_template): return a simple string.
    sys_block = f"System: {system_prompt}\n" if system_prompt else ""
    return f"{sys_block}User: {prompt}\nAssistant: {response}"


def build_formatted_dataset(
    raw: Dataset,
    tokenizer,
    *,
    is_themis: bool,
    use_system_prompts: bool,
    use_aspect_prompts: bool,
    num_proc: int,
) -> Dataset:
    """Apply chat template to raw CRB rows, returning a Dataset with
    ``text_chosen`` / ``text_rejected`` columns ready for tokenization.

    For Themis models with system prompts enabled, the aspect-specific prompt
    from ``CODE_REWARD_BENCH_ASPECT_PROMPTS`` is inserted per-row. Otherwise
    no system message is added and the tokenizer's chat template runs with
    just user+assistant turns.
    """

    def _pick_system(aspect: str) -> Optional[str]:
        if not (is_themis and use_system_prompts):
            return None
        if use_aspect_prompts and aspect in CODE_REWARD_BENCH_ASPECT_PROMPTS:
            return CODE_REWARD_BENCH_ASPECT_PROMPTS[aspect]
        return CODE_REWARD_BENCH_ASPECT_PROMPTS["Full"]

    def _map(example):
        sys_prompt = _pick_system(example["aspect"])
        example["text_chosen"] = _format_one(tokenizer, example["prompt"], example["chosen"], sys_prompt)
        example["text_rejected"] = _format_one(tokenizer, example["prompt"], example["rejected"], sys_prompt)
        return example

    return raw.map(_map, num_proc=num_proc, load_from_cache_file=False)


############################
# Metrics
############################
def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pairwise accuracy overall and per (aspect, language, subset)."""
    if not results:
        return {}

    by_aspect: Dict[str, List[int]] = defaultdict(list)
    by_language: Dict[str, List[int]] = defaultdict(list)
    by_subset: Dict[str, List[int]] = defaultdict(list)
    correct_flags: List[int] = []

    for r in results:
        flag = 1 if r["score_chosen"] > r["score_rejected"] else 0
        correct_flags.append(flag)
        by_aspect[r["aspect"]].append(flag)
        by_language[r["language"]].append(flag)
        by_subset[r["subset"]].append(flag)

    def _agg(d):
        return {k: {"accuracy": float(np.mean(v)), "count": len(v)} for k, v in sorted(d.items())}

    overall_acc = float(np.mean(correct_flags))
    per_aspect_acc = _agg(by_aspect)
    macro_aspect_acc = float(np.mean([v["accuracy"] for v in per_aspect_acc.values()]))

    return {
        "overall_accuracy": overall_acc,
        "macro_aspect_accuracy": macro_aspect_acc,
        "num_total": len(correct_flags),
        "num_correct": int(np.sum(correct_flags)),
        "by_aspect": per_aspect_acc,
        "by_language": _agg(by_language),
        "by_subset": _agg(by_subset),
    }


############################
# CLI
############################
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path to reward model")
    parser.add_argument("--tokenizer", type=str, default=None, help="path to non-matching tokenizer to model")
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="fastchat chat template (optional; uses tokenizer template if not specified)",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="load model with trust_remote_code=True"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="project-themis/Themis-CodeRewardBench",
        help="HF dataset id (default: project-themis/Themis-CodeRewardBench)",
    )
    parser.add_argument("--config", type=str, default="Full", help="HF dataset config (default: Full)")
    parser.add_argument("--split", type=str, default="Full", help="HF dataset split (default: Full)")
    parser.add_argument(
        "--use_system_prompts",
        action="store_true",
        help="Themis-only: inject the Themis CRB judge-persona system prompt. Has no effect on non-Themis models.",
    )
    parser.add_argument(
        "--use_aspect_prompts",
        action="store_true",
        help="Themis-only: use the per-row aspect-specific prompt instead of the combined 'Full' prompt. "
             "Only honored when --use_system_prompts is also set.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2560, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument("--debug", action="store_true", help="run on only 32 examples for debugging")
    parser.add_argument(
        "--not_quantized", action="store_true", help="disable quantization for models that are quantized by default"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "float64"],
        help="PyTorch dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use (default: sdpa)",
    )
    parser.add_argument(
        "--num_proc", type=int, default=8, help="Number of processes for dataset operations (default: 8)"
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=4, help="Number of worker processes for DataLoader (default: 4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/code-reward-bench",
        help="Directory to save per-example scores JSON (default: results/code-reward-bench)",
    )
    parser.add_argument(
        "--do_not_save",
        action="store_true",
        help="Skip writing per-example scores and metrics JSON (accuracy still prints to stdout).",
    )
    args = parser.parse_args()
    args.torch_dtype = torch_dtype_mapping(args.torch_dtype)
    return args


def main():
    args = get_args()

    ###############
    # Setup logging
    ###############
    accelerator = Accelerator()
    current_device = accelerator.process_index

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity(logging.INFO)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Running Code Reward Bench on model={args.model}")
    if args.trust_remote_code:
        logger.info("Loading model with trust_remote_code=True")

    if args.chat_template is not None:
        if get_conv_template is None:
            raise ImportError(
                "--chat_template requires fschat, which is unmaintained. "
                "Consider using the model's built-in tokenizer chat template instead (omit --chat_template). "
                "If you need legacy templates, install with: pip install rewardbench[v1]"
            )
        _ = get_conv_template(args.chat_template)
        logger.info(f"Note: --chat_template={args.chat_template} is accepted for API compatibility but not "
                    "actively used — CRB always uses the tokenizer's chat template.")

    if args.model in REWARD_MODEL_CONFIG:
        config = REWARD_MODEL_CONFIG[args.model]
    else:
        config = REWARD_MODEL_CONFIG["default"]
    logger.info(f"Using reward model config: {config}")

    quantized = config["quantized"]
    if (
        ("llama-3" in args.model)
        or ("Llama3" in args.model)
        or ("Llama-3" in args.model)
        or ("LLaMA3" in args.model)
        or ("llama3" in args.model)
        or args.not_quantized
    ):
        quantized = False
        logger.info(f"Disabling quantization (llama-3 family or --not_quantized={args.not_quantized})")

    custom_dialogue = config["custom_dialogue"]
    if custom_dialogue:
        raise NotImplementedError(
            "Custom-dialogue models (e.g. PairRM, SteamSHP) are not supported by CRB — "
            "this benchmark requires a reward model that consumes formatted text."
        )
    model_type = config["model_type"]
    if model_type == "Custom Classifier":
        raise NotImplementedError(
            "Custom Classifier models are not supported by run_code_reward_bench.py. "
            "Please refer to the model's original code."
        )
    model_builder = config["model_builder"]
    pipeline_builder = config["pipeline_builder"]
    torch_dtype = config.get("torch_dtype", None)
    if torch_dtype is None:
        if args.torch_dtype == torch.bfloat16:
            quantized = False
            logger.info("Disabling quantization for bfloat16 datatype")
        torch_dtype = args.torch_dtype

    trust_remote_code = args.trust_remote_code
    is_themis = pipeline_builder is ThemisPipeline
    if is_themis:
        logger.info(
            f"Themis model detected. use_system_prompts={args.use_system_prompts}, "
            f"use_aspect_prompts={args.use_aspect_prompts}"
        )
    elif args.use_system_prompts or args.use_aspect_prompts:
        logger.warning(
            "--use_system_prompts / --use_aspect_prompts only take effect for Themis models. "
            "Ignoring for the current model."
        )

    ############################
    # Load tokenizer
    ############################
    logger.info("*** Load tokenizer ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    tokenizer.truncation_side = "left"

    ############################
    # Load reward model
    ############################
    BATCH_SIZE = args.batch_size
    logger.info("*** Load reward model ***")
    reward_pipeline_kwargs = {
        "batch_size": BATCH_SIZE,
        "truncation": True,
        "padding": True,
        "max_length": args.max_length,
        "function_to_apply": "none",
        "return_token_type_ids": False,
    }
    if quantized:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs = {
            "quantization_config": quantization_config,
            "device_map": {"": current_device},
            "torch_dtype": torch_dtype if torch.cuda.is_available() else None,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
    else:
        model_kwargs = {
            "device_map": "auto" if torch.cuda.is_available() else "cpu",
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
            "low_cpu_mem_usage": True,
        }
    model_kwargs["attn_implementation"] = args.attn_implementation

    model = model_builder(args.model, **model_kwargs, trust_remote_code=trust_remote_code)
    reward_pipe = pipeline_builder("text-classification", model=model, tokenizer=tokenizer)

    # pad token fallbacks
    if reward_pipe.tokenizer.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.eos_token_id
        reward_pipe.tokenizer.pad_token_id = reward_pipe.tokenizer.eos_token_id
    if reward_pipe.model.config.pad_token_id is None:
        reward_pipe.model.config.pad_token_id = reward_pipe.tokenizer.pad_token_id

    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Load & format dataset
    ############################
    logger.info(f"*** Loading {args.dataset} (config={args.config}, split={args.split}) ***")
    raw = load_code_reward_bench(args.dataset, args.config, args.split)
    logger.info(f"Loaded {len(raw)} rows")

    if args.debug:
        raw = raw.select(range(min(32, len(raw))))
        logger.info(f"--debug: truncated to {len(raw)} rows")

    dataset = build_formatted_dataset(
        raw,
        tokenizer,
        is_themis=is_themis,
        use_system_prompts=args.use_system_prompts,
        use_aspect_prompts=args.use_aspect_prompts,
        num_proc=args.num_proc,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")

    ############################
    # Run inference
    ############################
    logger.info("*** Running dataloader to collect results ***")
    from torch.utils.data.dataloader import default_collate

    def custom_collate_fn(batch):
        if isinstance(batch[0]["text_chosen"][0], dict):
            return batch
        return default_collate(batch)

    # Keep only the tokenized-text columns for the dataloader; metadata columns
    # stay on `raw` for later joining.
    infer_ds = dataset.remove_columns([c for c in dataset.column_names if c not in ("text_chosen", "text_rejected")])

    dataloader = torch.utils.data.DataLoader(
        infer_ds,
        batch_size=BATCH_SIZE,
        collate_fn=custom_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    dataloader, prepared_model = accelerator.prepare(dataloader, reward_pipe.model)
    reward_pipe.model = prepared_model

    score_chosen_list: List[float] = []
    score_rejected_list: List[float] = []

    if pipeline_builder == pipeline:
        # built-in HF pipeline: feed full lists at once
        reward_pipe = accelerator.prepare(reward_pipe)
        results_rej = reward_pipe(infer_ds["text_rejected"], **reward_pipeline_kwargs)
        results_cho = reward_pipe(infer_ds["text_chosen"], **reward_pipeline_kwargs)
        score_chosen_list = [r["score"] for r in results_cho]
        score_rejected_list = [r["score"] for r in results_rej]
    else:
        for step, batch in enumerate(tqdm(dataloader, desc="Code Reward Bench")):
            rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
            rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

            if isinstance(rewards_chosen[0], dict):
                sc = [r["score"] for r in rewards_chosen]
                sr = [r["score"] for r in rewards_rejected]
            else:
                if rewards_chosen.dim() > 1 and rewards_chosen.shape[-1] == 1:
                    rewards_chosen = rewards_chosen.squeeze(-1)
                if rewards_rejected.dim() > 1 and rewards_rejected.shape[-1] == 1:
                    rewards_rejected = rewards_rejected.squeeze(-1)
                sc = rewards_chosen.float().cpu().numpy().tolist()
                sr = rewards_rejected.float().cpu().numpy().tolist()
            score_chosen_list.extend(sc)
            score_rejected_list.extend(sr)

    ############################
    # Assemble results
    ############################
    per_example: List[Dict[str, Any]] = []
    for i, ex in enumerate(raw):
        sc = score_chosen_list[i]
        sr = score_rejected_list[i]
        if isinstance(sc, list) and len(sc) == 1:
            sc = sc[0]
        if isinstance(sr, list) and len(sr) == 1:
            sr = sr[0]
        per_example.append(
            {
                "id": ex["id"],
                "subset": ex["subset"],
                "aspect": ex["aspect"],
                "language": ex["language"],
                "score_chosen": sc,
                "score_rejected": sr,
                "correct": bool(sc > sr),
                "score_diff": float(sc - sr),
            }
        )

    acc = compute_accuracy(per_example)
    model_name = args.model.rstrip("/").split("/")[-1]

    print(f"\nCode Reward Bench results for {args.model}:")
    print(f"  num_total            : {acc['num_total']}")
    print(f"  overall_accuracy     : {acc['overall_accuracy']:.4f}")
    print(f"  macro_aspect_accuracy: {acc['macro_aspect_accuracy']:.4f}")
    print("  by_aspect:")
    for k, v in acc["by_aspect"].items():
        print(f"    {k:<28} {v['accuracy']:.4f}  (n={v['count']})")
    print("  by_language:")
    for k, v in acc["by_language"].items():
        print(f"    {k:<12} {v['accuracy']:.4f}  (n={v['count']})")
    # only print per-subset if small; otherwise just count
    print(f"  by_subset: {len(acc['by_subset'])} subsets (see metrics.json)")

    if args.do_not_save:
        logger.info("--do_not_save set: skipping local writes.")
        return

    output_dir = os.path.join(args.output_dir, args.model.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    scores_path = os.path.join(output_dir, f"{model_name}_{ts}_scores.json")
    with open(scores_path, "w") as f:
        json.dump(per_example, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved per-example scores to {scores_path}")

    metrics_path = os.path.join(output_dir, f"{model_name}_{ts}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "config": args.config,
                "split": args.split,
                "use_system_prompts": args.use_system_prompts and is_themis,
                "use_aspect_prompts": args.use_aspect_prompts and is_themis,
                **acc,
            },
            f,
            indent=4,
        )
    logger.info(f"Saved aggregate metrics to {metrics_path}")


if __name__ == "__main__":
    main()
