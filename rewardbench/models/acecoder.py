"""TIGER-Lab/AceCodeRM-{7B,32B} reward model wrapper.

The AceCodeRM models (https://huggingface.co/TIGER-Lab/AceCodeRM-7B,
https://huggingface.co/TIGER-Lab/AceCodeRM-32B) ship as TRL-style value-head
reward models on top of ``Qwen2ForCausalLM``:

* Backbone: ``Qwen2ForCausalLM`` (``lm_head.weight`` is kept in the checkpoint,
  unused for scoring).
* Head: ``v_head.summary = nn.Linear(hidden_size, 1, bias=True)`` with a
  training-time dropout (p=0.1) that is a no-op at eval.
* Pooling: value at the **last non-pad token** of each sequence
  (``attention_mask.sum(-1) - 1`` — works for left- or right-padded batches).
* Config quirks: ``architectures=['Qwen2ForCausalRM']`` (not in HF registry)
  and ``pad_token_id`` is missing. The tokenizer already carries
  ``pad_token=<|endoftext|>`` (151643), which we copy onto the config.

This file reproduces the canonical ``AceCodeRM`` class from
https://github.com/TIGER-AI-Lab/AceCoder/blob/main/src/acecoder/rm_utils.py
locally, so we don't need to pip-install the ``acecoder`` extra.
"""

from typing import Iterable, List, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, Qwen2ForCausalLM


class _ValueHead(nn.Module):
    """TRL ValueHead: Linear(hidden, 1, bias=True) with eval-time no-op dropout."""

    def __init__(self, config, summary_dropout_prob: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        self.summary = nn.Linear(config.hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        return self.summary(output)


class _AceCodeRM(Qwen2ForCausalLM):
    """Port of ``acecoder.AceCodeRM`` — Qwen2ForCausalLM + ValueHead, pooled at last real token."""

    def __init__(self, config):
        super().__init__(config)
        self.v_head = _ValueHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        return_past_key_values: bool = False,
        **kwargs,
    ):
        kwargs["output_hidden_states"] = True
        kwargs["past_key_values"] = past_key_values

        base_output = super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        last_hidden = base_output.hidden_states[-1]
        if last_hidden.device != self.v_head.summary.weight.device:
            last_hidden = last_hidden.to(self.v_head.summary.weight.device)

        value = self.v_head(last_hidden).squeeze(-1)  # (B, T)
        # Pick the last real token in each row (works for any padding side).
        last_idx = attention_mask.sum(dim=-1, keepdim=True) - 1  # (B, 1)
        rm_scores = value.gather(dim=-1, index=last_idx).squeeze(-1)  # (B,)

        if return_past_key_values:
            return rm_scores, base_output.past_key_values
        return rm_scores


def build_acecoder_model(
    model_name_or_path: str,
    *,
    torch_dtype=torch.bfloat16,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Loader compatible with ``REWARD_MODEL_CONFIG['model_builder']``."""
    kwargs.pop("quantization_config", None)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    # Config ships architectures=['Qwen2ForCausalRM'] and lacks pad_token_id.
    # Reset to a known class so .from_pretrained doesn't try to resolve the custom name.
    config.architectures = ["Qwen2ForCausalLM"]
    if getattr(config, "pad_token_id", None) is None:
        try:
            tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            pad_token_id = tok.pad_token_id or getattr(config, "eos_token_id", None)
        except Exception:
            pad_token_id = getattr(config, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError(f"Cannot determine pad_token_id for {model_name_or_path}")
        config.pad_token_id = pad_token_id

    # from_pretrained will load all matching keys (model.* + lm_head.weight + v_head.summary.*).
    # Qwen2ForCausalLM knows about the backbone + lm_head; our _AceCodeRM adds v_head on top,
    # and checkpoint keys v_head.summary.* line up exactly with _ValueHead's submodule names.
    model = _AceCodeRM.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    model.eval()
    return model


class AceCoderPipeline:
    """Scoring pipeline for TIGER-Lab/AceCodeRM-{7B,32B}."""

    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model.eval()
        self.tokenizer = tokenizer
        # Qwen2 tokenizers default pad_token=<|endoftext|>; make sure it's set.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, samples: Union[str, Iterable[str]], **kwargs):
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        if "max_length" not in kwargs:
            raise ValueError("AceCoderPipeline requires `max_length` to be passed explicitly.")
        max_length = kwargs["max_length"]

        if isinstance(samples, str):
            prepped: Union[str, List[str]] = samples
        else:
            prepped = list(samples)

        inputs = self.tokenizer(
            prepped,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            scores = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        return scores
