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
Runner for RM-Bench (https://github.com/THU-KEG/RM-Bench).

RM-Bench evaluates reward models on a JSON dataset where each example contains
3 chosen and 3 rejected responses corresponding to different stylistic
variants (concise, detailed_plain, detailed_markdown). The 3x3 score matrix
is aggregated into hard/normal/easy accuracy per domain.

Port of the original RM-Bench run_rm.py, adapted to the current rewardbench
APIs (transformers 5.x, optional fastchat, sdpa default, bfloat16 default).

Example:
    uv run python scripts/run_rm_bench.py \\
        --model allenai/tulu-v2.5-13b-hh-rlhf-60k-rm \\
        --datapath data/rm-bench/total_dataset.json \\
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


############################
# RM-Bench dataset helpers
############################
def convert_robust_dataset_to_preference_dataset_list(robust_dataset_path: str) -> List[Dataset]:
    """Split an RM-Bench JSON file into N preference datasets, one per style variant."""
    with open(robust_dataset_path) as f:
        robust_dataset = json.load(f)

    num_pairs = len(robust_dataset[0]["chosen"])
    assert num_pairs == len(
        robust_dataset[0]["rejected"]
    ), "The number of chosen and rejected pairs should be the same."

    para_corp_dataset_list = []
    for idx in range(num_pairs):
        para_corp_dataset = Dataset.from_dict(
            {
                "id": [unit["id"] for unit in robust_dataset],
                # total_dataset.json has "domain"; filtered JSONs already have "subset"
                "subset": [unit.get("domain", unit.get("subset", "default")) for unit in robust_dataset],
                "prompt": [unit["prompt"] for unit in robust_dataset],
                "chosen": [unit["chosen"][idx] for unit in robust_dataset],
                "chosen_model": ["chosen" for _ in robust_dataset],
                "rejected": [unit["rejected"][idx] for unit in robust_dataset],
                "rejected_model": ["rejected" for _ in robust_dataset],
            }
        )
        para_corp_dataset_list.append(para_corp_dataset)

    return para_corp_dataset_list


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
# RM-Bench accuracy metrics
############################
def split_dataset_by_domain(dataset: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    domains = ["chat", "math", "code", "safety"]
    domain_dataset_dict: Dict[str, List[Dict[str, Any]]] = {}
    for domain in domains:
        domain_dataset_dict[domain] = [ex for ex in dataset if ex["domain"].startswith(domain)]
    # drop the domain key from copies so downstream doesn't recurse
    for domain in domain_dataset_dict:
        for example in domain_dataset_dict[domain]:
            example.pop("domain", None)
    return domain_dataset_dict


def compute_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute hard/normal/easy accuracy from the 3x3 score matrix per domain.

    If `domain` is present in the entries, splits by domain and also reports
    per-domain averages plus aggregate hard/normal/easy/total accuracy.
    """
    if "domain" in results[0]:
        print("We are handling total_dataset.json")
        print("Splitting the dataset by domain...")
        split_results = split_dataset_by_domain(results)
        domain_results = {}
        for domain in split_results:
            if len(split_results[domain]) == 0:
                continue
            domain_results[domain] = compute_accuracy(split_results[domain])
        domain_avg_results = {d: float(np.mean(list(r.values()))) for d, r in domain_results.items()}
        domain_hard_normal_easy_acc = {
            "hard_acc": float(np.mean([domain_results[d]["hard_acc"] for d in domain_results])),
            "normal_acc": float(np.mean([domain_results[d]["normal_acc"] for d in domain_results])),
            "easy_acc": float(np.mean([domain_results[d]["easy_acc"] for d in domain_results])),
        }
        total_avg_acc = float(np.mean(list(domain_avg_results.values())))
        final_results: Dict[str, float] = {}
        final_results.update(domain_avg_results)
        final_results.update(domain_hard_normal_easy_acc)
        final_results["total_avg_acc"] = total_avg_acc
        return final_results

    # score_chosen/score_rejected are length-3 lists: [concise, detailed_plain, detailed_markdown]
    MATRIX_SIZE = 3
    acc_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    for result in results:
        for i in range(len(result["score_chosen"])):
            for j in range(len(result["score_rejected"])):
                if result["score_chosen"][i] > result["score_rejected"][j]:
                    acc_matrix[i][j] += 1
    acc_matrix /= len(results)

    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = float(np.sum(np.triu(acc_matrix, 1)) / upper_right_count)
    normal_acc = float(np.mean(np.diag(acc_matrix)))
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = float(np.sum(np.tril(acc_matrix, -1)) / lower_left_count)

    return {"hard_acc": hard_acc, "normal_acc": normal_acc, "easy_acc": easy_acc}


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
        help="path to an RM-Bench JSON file (e.g. data/rm-bench/total_dataset.json or data/rm-bench/chat_filtered.json)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size for inference")
    parser.add_argument("--max_length", type=int, default=2048, help="Max length of RM inputs (passed to pipeline)")
    parser.add_argument(
        "--debug", action="store_true", help="run on only 10 examples per style variant for debugging"
    )
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
        default="results/rm-bench",
        help="Directory to save per-example scores JSON (default: results/rm-bench)",
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

    logger.info(f"Running RM-Bench on model={args.model} with chat_template={args.chat_template}")
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

    # Pick reward model config (same override logic as run_rm.py)
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
            "Custom Classifier models (e.g. NVIDIA SteerLM) are not supported by run_rm_bench.py. "
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
    # Iterate over style variants
    ############################
    logger.info(f"*** Loading RM-Bench dataset from {args.datapath} ***")
    raw_dataset_list = convert_robust_dataset_to_preference_dataset_list(args.datapath)
    num_variants = len(raw_dataset_list)
    logger.info(f"Found {num_variants} style variants per example")

    score_chosen: List[List[float]] = []
    score_rejected: List[List[float]] = []

    for variant_idx, raw_dataset in enumerate(raw_dataset_list):
        logger.info(f"=== Scoring style variant {variant_idx + 1}/{num_variants} ===")

        # clear cuda memory cache between variants
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} GB")

        # apply chat template
        dataset = apply_chat_template(
            raw_dataset=raw_dataset,
            tokenizer=tokenizer,
            conv=conv,
            custom_dialogue_formatting=custom_dialogue,
            num_proc=args.num_proc,
            logger=logger,
        )

        # only keep the columns we need for inference
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

            unit_score_chosen_list = [r["score"] for r in results_cho]
            unit_score_rejected_list = [r["score"] for r in results_rej]
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

            unit_score_chosen_list = []
            unit_score_rejected_list = []
            for step, batch in enumerate(tqdm(dataloader, desc=f"Variant {variant_idx + 1}/{num_variants}")):
                rewards_chosen = reward_pipe(batch["text_chosen"], **reward_pipeline_kwargs)
                rewards_rejected = reward_pipe(batch["text_rejected"], **reward_pipeline_kwargs)

                if isinstance(rewards_chosen[0], dict):
                    score_chosen_batch = [r["score"] for r in rewards_chosen]
                    score_rejected_batch = [r["score"] for r in rewards_rejected]
                else:
                    score_chosen_batch = rewards_chosen.float().cpu().numpy().tolist()
                    score_rejected_batch = rewards_rejected.float().cpu().numpy().tolist()

                unit_score_chosen_list.extend(score_chosen_batch)
                unit_score_rejected_list.extend(score_rejected_batch)

        score_chosen.append(unit_score_chosen_list)
        score_rejected.append(unit_score_rejected_list)

        # free per-variant state before next iteration
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ############################
    # Assemble and save results
    ############################
    with open(args.datapath) as f:
        dataset_json = json.load(f)

    if args.debug:
        dataset_json = dataset_json[: len(score_chosen[0])]

    for idx, unit in enumerate(dataset_json):
        unit["score_chosen"] = [variant_scores[idx] for variant_scores in score_chosen]
        unit["score_rejected"] = [variant_scores[idx] for variant_scores in score_rejected]
        # flatten any [[x]] wrappers
        if all(isinstance(e, list) and len(e) == 1 for e in unit["score_chosen"]):
            unit["score_chosen"] = [e[0] for e in unit["score_chosen"]]
        if all(isinstance(e, list) and len(e) == 1 for e in unit["score_rejected"]):
            unit["score_rejected"] = [e[0] for e in unit["score_rejected"]]

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
        print(f"  {k}: {v:.4f}")

    # also write the aggregate metrics to a sibling JSON for easy machine reading
    metrics_path = os.path.join(output_dir, f"{filename}_{model_name}_{ts}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"model": args.model, "datapath": args.datapath, **acc_dict}, f, indent=4)
    logger.info(f"Saved aggregate metrics to {metrics_path}")


if __name__ == "__main__":
    main()
