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
Runner for JudgeBench (https://huggingface.co/datasets/ScalerLab/JudgeBench).

JudgeBench evaluates reward/judge models on hard prompts drawn from
LiveBench, LiveCodeBench, and MMLU-Pro. Each example has a single
`chosen` response (objectively correct) and a single `rejected` response
(objectively incorrect). The leaderboard reports pairwise accuracy per
category (knowledge / reasoning / math / coding) and an overall average.

Data is bundled as JSON files under data/judge-bench/, produced by
serializing the Hugging Face dataset. JSON schema per example:
    {
        "id": "<subset>_<uuid>",
        "subset": "LIVEBENCH-REASONING" | "MMLU-PRO-MATH" | ...,
        "category": "reasoning" | "math" | "coding" | "knowledge",
        "prompt": "...",
        "chosen": "...",
        "rejected": "...",
        "language": "NL" | ...,
        "aspect": "Helpfulness" | ...,
        "system": "..."  # JudgeBench's LLM-as-judge system prompt (unused for RMs)
    }

Example:
    uv run python scripts/run_judge_bench.py \\
        --model allenai/tulu-v2.5-13b-hh-rlhf-60k-rm \\
        --datapath data/judge-bench/total_dataset.json \\
        --batch_size 8
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline

# fschat is optional - only needed if --chat_template is specified
try:
    from fastchat.conversation import get_conv_template
except ImportError:
    get_conv_template = None

# Enable faster downloads with hf_transfer (if available)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    prepare_dialogue,
    prepare_dialogue_from_tokenizer,
    torch_dtype_mapping,
)

# Enable TensorFloat32 (TF32) tensor cores on Ampere GPUs for matrix multiplications (faster than FP32)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


# JudgeBench top-level categories reported on the leaderboard
JUDGE_BENCH_CATEGORIES = ("knowledge", "reasoning", "math", "coding")


############################
# JudgeBench dataset helpers
############################
def load_judge_bench_json(datapath: str) -> Dataset:
    """Load a JudgeBench JSON file into a pairwise preference Dataset."""
    with open(datapath) as f:
        raw = json.load(f)

    return Dataset.from_dict(
        {
            "id": [r["id"] for r in raw],
            "subset": [r["subset"] for r in raw],
            "category": [r.get("category", _derive_category(r["subset"])) for r in raw],
            "prompt": [r["prompt"] for r in raw],
            "chosen": [r["chosen"] for r in raw],
            "chosen_model": ["chosen" for _ in raw],
            "rejected": [r["rejected"] for r in raw],
            "rejected_model": ["rejected" for _ in raw],
        }
    )


def _derive_category(subset: str) -> str:
    """Fallback category inference if the JSON lacks a `category` field."""
    mapping = {
        "LIVEBENCH-REASONING": "reasoning",
        "LIVEBENCH-MATH": "math",
        "LIVECODEBENCH": "coding",
    }
    if subset in mapping:
        return mapping[subset]
    if subset.startswith("MMLU-PRO-"):
        return "knowledge"
    return "other"


def apply_chat_template(
    raw_dataset: Dataset,
    tokenizer,
    conv,
    custom_dialogue_formatting: bool,
    num_proc: int,
    logger,
) -> Dataset:
    """Apply chat template to the dataset to produce text_chosen and text_rejected columns."""
    if custom_dialogue_formatting:
        logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations(example):
            example["text_chosen"] = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ]
            example["text_rejected"] = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ]
            return example

        return raw_dataset.map(map_conversations, num_proc=num_proc, load_from_cache_file=False)

    usable_tokenizer = check_tokenizer_chat_template(tokenizer)
    assert conv is not None or usable_tokenizer, (
        "Either a conv template (--chat_template) or a tokenizer with a chat template must be provided."
    )

    if usable_tokenizer:
        logger.info("*** Preparing dataset with HF Transformers chat template ***")
        return raw_dataset.map(
            prepare_dialogue_from_tokenizer,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=num_proc,
            load_from_cache_file=False,
        )

    logger.info("*** Preparing dataset with FastChat chat template ***")
    return raw_dataset.map(
        prepare_dialogue,
        fn_kwargs={"dialogue_template": conv},
        num_proc=num_proc,
        load_from_cache_file=False,
    )


############################
# JudgeBench accuracy metrics
############################
def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute pairwise accuracy per category, per subset, and overall."""
    num_total = len(results)
    if num_total == 0:
        return {}

    per_category: Dict[str, List[int]] = {}
    per_subset: Dict[str, List[int]] = {}
    for r in results:
        correct = 1 if r["score_chosen"] > r["score_rejected"] else 0
        per_category.setdefault(r.get("category", _derive_category(r["subset"])), []).append(correct)
        per_subset.setdefault(r["subset"], []).append(correct)

    out: Dict[str, float] = {}
    # per-category accuracy
    for cat in JUDGE_BENCH_CATEGORIES:
        if cat in per_category and per_category[cat]:
            out[f"{cat}_acc"] = float(np.mean(per_category[cat]))
    # per-subset accuracy (fine-grained)
    out["per_subset"] = {s: float(np.mean(v)) for s, v in sorted(per_subset.items())}
    # micro average across all examples
    all_correct = [c for v in per_category.values() for c in v]
    out["micro_avg_acc"] = float(np.mean(all_correct)) if all_correct else 0.0
    # macro average across reported categories
    cat_accs = [out[f"{c}_acc"] for c in JUDGE_BENCH_CATEGORIES if f"{c}_acc" in out]
    out["macro_avg_acc"] = float(np.mean(cat_accs)) if cat_accs else 0.0
    out["num_total"] = num_total
    return out


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
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help=(
            "path to a JudgeBench JSON file, e.g. data/judge-bench/total_dataset.json or "
            "data/judge-bench/{knowledge,reasoning,math,coding}_filtered.json"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2560, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument("--debug", action="store_true", help="run on only 10 examples for debugging")
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
        default="results/judge-bench",
        help="Directory to save per-example scores JSON (default: results/judge-bench)",
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

    logger.info(f"Running JudgeBench on model={args.model} with chat_template={args.chat_template}")
    if args.trust_remote_code:
        logger.info("Loading model with trust_remote_code=True")

    # Load chat template (fastchat) only if explicitly requested
    if args.chat_template is not None:
        if get_conv_template is None:
            raise ImportError(
                "--chat_template requires fschat, which is unmaintained. "
                "Consider using the model's built-in tokenizer chat template instead (omit --chat_template). "
                "If you need legacy templates, install with: pip install rewardbench[v1]"
            )
        conv = get_conv_template(args.chat_template)
        logger.info(f"Using conversation template {args.chat_template}")
    else:
        conv = None

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
    model_type = config["model_type"]
    if model_type == "Custom Classifier":
        raise NotImplementedError(
            "Custom Classifier models (e.g. NVIDIA SteerLM) are not supported by run_judge_bench.py. "
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

    ############################
    # Load tokenizer
    ############################
    logger.info("*** Load tokenizer ***")
    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if not custom_dialogue:
        tokenizer.truncation_side = "left"

    ############################
    # Load reward model pipeline
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

    # if using fastchat template (no template in tokenizer), make the RM tokenizer output an EOS token
    if not check_tokenizer_chat_template(tokenizer):
        reward_pipe.tokenizer.add_eos_token = True

    ############################
    # Load & prepare dataset
    ############################
    logger.info(f"*** Loading JudgeBench dataset from {args.datapath} ***")
    raw_dataset = load_judge_bench_json(args.datapath)
    logger.info(f"Loaded {len(raw_dataset)} examples")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")

    dataset = apply_chat_template(
        raw_dataset=raw_dataset,
        tokenizer=tokenizer,
        conv=conv,
        custom_dialogue_formatting=custom_dialogue,
        num_proc=args.num_proc,
        logger=logger,
    )

    keep_columns = ["text_chosen", "text_rejected", "id"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_columns])
    dataset = dataset.remove_columns("id")

    if args.debug:
        dataset = dataset.select(range(min(10, len(dataset))))

    ############################
    # Run inference
    ############################
    if pipeline_builder == pipeline:
        logger.info("*** Running forward pass via built-in pipeline abstraction ***")
        reward_pipe = accelerator.prepare(reward_pipe)

        results_rej = reward_pipe(dataset["text_rejected"], **reward_pipeline_kwargs)
        results_cho = reward_pipe(dataset["text_chosen"], **reward_pipeline_kwargs)

        score_chosen_list = [r["score"] for r in results_cho]
        score_rejected_list = [r["score"] for r in results_rej]
    else:
        logger.info("*** Running dataloader to collect results ***")
        from torch.utils.data.dataloader import default_collate

        def custom_collate_fn(batch):
            if isinstance(batch[0]["text_chosen"][0], dict):
                return batch
            return default_collate(batch)

        dataloader = torch.utils.data.DataLoader(
            dataset,
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
        for step, batch in enumerate(tqdm(dataloader, desc="JudgeBench")):
            rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
            rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

            if isinstance(rewards_chosen[0], dict):
                score_chosen_batch = [r["score"] for r in rewards_chosen]
                score_rejected_batch = [r["score"] for r in rewards_rejected]
            else:
                score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
                score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

            score_chosen_list.extend(score_chosen_batch)
            score_rejected_list.extend(score_rejected_batch)

    ############################
    # Assemble and save results
    ############################
    with open(args.datapath) as f:
        dataset_json = json.load(f)

    if args.debug:
        dataset_json = dataset_json[: len(score_chosen_list)]

    for idx, unit in enumerate(dataset_json):
        sc = score_chosen_list[idx]
        sr = score_rejected_list[idx]
        unit["score_chosen"] = sc[0] if isinstance(sc, list) and len(sc) == 1 else sc
        unit["score_rejected"] = sr[0] if isinstance(sr, list) and len(sr) == 1 else sr
        if "category" not in unit:
            unit["category"] = _derive_category(unit["subset"])

    filename = os.path.basename(args.datapath).replace(".json", "")
    model_name = args.model.rstrip("/").split("/")[-1]
    output_dir = os.path.join(args.output_dir, args.model.replace("/", "__"))
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{filename}_{model_name}_{ts}.json")
    with open(output_path, "w") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved per-example scores to {output_path}")

    acc_dict = compute_accuracy(dataset_json)
    print(f"\nAccuracy of {model_name} on {filename}:")
    for k, v in acc_dict.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for sk, sv in v.items():
                print(f"    {sk}: {sv:.4f}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    metrics_path = os.path.join(output_dir, f"{filename}_{model_name}_{ts}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": args.model, "datapath": args.datapath, **acc_dict}, f, indent=4)
    logger.info(f"Saved aggregate metrics to {metrics_path}")


if __name__ == "__main__":
    main()
