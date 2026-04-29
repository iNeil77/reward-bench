"""Nexusflow/Athene-RM-70B reward model wrapper.

The model card (https://huggingface.co/Nexusflow/Athene-RM-70B) ships a
``CustomAutoModelForSequenceClassification`` architecture that is NOT
registered with transformers (no auto_map, no bundled modeling_*.py). Its
checkpoint is a ``LlamaModel`` backbone with a single scalar head
``v_head = nn.Linear(hidden, 1, bias=False)`` — loading via the default
``AutoModelForSequenceClassification`` path would either refuse to load or
drop ``v_head.weight`` and random-init a ``score.weight``.

This module reproduces the exact recipe from the model card:

* Backbone: ``LlamaModel`` via ``AutoModel.from_pretrained``.
* Head: ``nn.Linear(hidden_size, 1, bias=False)``, loaded from the
  ``v_head.weight`` tensor in the checkpoint's safetensors shards.
* Tokenization: ``apply_chat_template`` + appending ``<|reserved_special_token_1|>``
  (the tokenizer's ``cls_token``, id 128003) as the final position.
* Pooling: the hidden state at the last CLS_ID (128003) position.
"""

import json
import os
from typing import Iterable, List, Union

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


ATHENE_CLS_ID = 128003  # <|reserved_special_token_1|>
ATHENE_PAD_ID = 128002  # <|reserved_special_token_0|>


class _AtheneRewardModel(nn.Module):
    """LlamaModel backbone + Linear(hidden, 1, bias=False) v_head, CLS pooling."""

    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.model = backbone
        self.v_head = nn.Linear(hidden_size, 1, bias=False)
        self.config = backbone.config

    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask=None, **_unused):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        # Under device_map="auto" sharding, the last hidden state lands on the
        # last stage's device, which may differ from where the head ended up
        # during construction. Move the hidden state to the head's device.
        if hidden_states.device != self.v_head.weight.device:
            hidden_states = hidden_states.to(self.v_head.weight.device)
        rewards = self.v_head(hidden_states).squeeze(-1)  # (B, T)

        # Pool at the last CLS_ID position in each row (matches the model card's code exactly).
        scores = []
        for i in range(input_ids.shape[0]):
            c_inds = (input_ids[i] == ATHENE_CLS_ID).nonzero()
            if c_inds.numel() == 0:
                raise ValueError(
                    "Athene pooling requires a CLS token (id 128003) in every input. "
                    "Did AthenePipeline's preprocessing run?"
                )
            # c_inds is on input_ids' device; pull a python int so it's device-agnostic.
            scores.append(rewards[i, int(c_inds[-1].item())])
        return torch.stack(scores)


def build_athene_model(
    model_name_or_path: str,
    *,
    torch_dtype=torch.bfloat16,
    trust_remote_code: bool = False,
    **kwargs,
):
    """Loader compatible with ``REWARD_MODEL_CONFIG['model_builder']``."""
    kwargs.pop("quantization_config", None)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    # The config's architecture is 'CustomAutoModelForSequenceClassification' which HF can't
    # resolve; AutoModel uses model_type ('llama') and returns a LlamaModel backbone.
    backbone = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    model = _AtheneRewardModel(backbone=backbone, hidden_size=config.hidden_size)
    # Place v_head on the same device as the backbone's final norm layer, which is where
    # the last hidden state is produced under device_map="auto" sharding. Fall back to
    # the last parameter's device for backbones that don't expose a `.norm` attribute.
    final_norm = getattr(backbone, "norm", None)
    if final_norm is not None and hasattr(final_norm, "weight"):
        head_ref = final_norm.weight
    else:
        head_ref = list(backbone.parameters())[-1]
    model.v_head = model.v_head.to(device=head_ref.device, dtype=head_ref.dtype)

    _load_v_head(model, model_name_or_path)
    model.eval()
    return model


def _load_v_head(model: _AtheneRewardModel, model_name_or_path: str) -> None:
    """Copy ``v_head.weight`` from the checkpoint's safetensors shards into the local head."""
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    if os.path.isdir(model_name_or_path):
        index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
        resolve = lambda fn: os.path.join(model_name_or_path, fn)  # noqa: E731
    else:
        index_path = hf_hub_download(model_name_or_path, "model.safetensors.index.json")
        resolve = lambda fn: hf_hub_download(model_name_or_path, fn)  # noqa: E731

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    if "v_head.weight" not in weight_map:
        raise RuntimeError(f"v_head.weight not found in {model_name_or_path} safetensors index")

    shard = weight_map["v_head.weight"]
    with safe_open(resolve(shard), framework="pt") as f:
        src = f.get_tensor("v_head.weight")

    target = model.v_head.weight
    src = src.to(device=target.device).to(dtype=target.dtype)
    if src.shape != target.shape:
        raise RuntimeError(
            f"v_head.weight shape mismatch: checkpoint {tuple(src.shape)} vs local {tuple(target.shape)}"
        )
    with torch.no_grad():
        target.copy_(src)


class AthenePipeline:
    """Scoring pipeline for Nexusflow/Athene-RM-70B."""

    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model.eval()
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            # Athene's config has no pad_token_id; the checkpoint uses
            # <|reserved_special_token_0|> (128002) during training.
            self.tokenizer.pad_token_id = ATHENE_PAD_ID
            self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        if self.tokenizer.cls_token_id is None:
            self.tokenizer.cls_token_id = ATHENE_CLS_ID
            self.tokenizer.cls_token = "<|reserved_special_token_1|>"

    def __call__(self, samples: Union[str, Iterable[str]], **kwargs):
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        if "max_length" not in kwargs:
            raise ValueError("AthenePipeline requires `max_length` to be passed explicitly.")
        max_length = kwargs["max_length"]

        cls_tok = self.tokenizer.cls_token or "<|reserved_special_token_1|>"
        if isinstance(samples, str):
            prepped: Union[str, List[str]] = samples + cls_tok
        else:
            prepped = [s + cls_tok for s in samples]

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
            )
        if scores.dim() > 1 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        return scores
