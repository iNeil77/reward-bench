"""WorldPM reward model wrapper.

Qwen/WorldPM-* checkpoints ship a custom ``Qwen2ForRewardModel`` class (from
the repo's ``modeling_qwen2_rm.py``) whose scoring head is a 2-layer MLP:
``Linear(hidden, hidden) -> ReLU -> Linear(hidden, 1)``. The class is resolved
via the checkpoint's ``auto_map["AutoModel"]`` when ``trust_remote_code=True``.

Faithfulness notes vs the model card
(https://huggingface.co/Qwen/WorldPM-72B-RLHFLow):

* The card says "Reward computation uses the hidden state of ``<|endoftext|>``"
  but the shipped ``forward()`` actually pools the token *before* the first
  ``pad_token_id`` (= ``<|endoftext|>``), which is ``<|im_end|>``. Our pipeline
  preserves that code-level behavior.
* The card's ``Key Notes`` state "System prompt remains empty during training".
  The tokenizer's chat template injects ``"You are a helpful assistant."`` by
  default, which is a distribution mismatch. ``WorldPMPipeline`` strips that
  default system prompt (replacing it with an empty system block) right before
  tokenization, so inputs match the training distribution.

Two gotchas we work around here:

1. The checkpoint's ``Qwen2RMConfig`` does not set ``pad_token_id``, but
   ``Qwen2Model.__init__`` reads it at construction time. We pre-load the
   config, fill ``pad_token_id`` from the tokenizer's pad token (falling back
   to EOS/BOS), and pass the patched config into ``from_pretrained``.
2. Loading via the top-level ``AutoModel`` on ``model_type == "qwen2"`` in
   transformers 5.x resolves to the canonical ``Qwen2Model`` backbone and
   silently drops the ``score.{0,2}`` MLP weights. ``trust_remote_code=True``
   plus the patched config forces the custom class, which correctly restores
   the head.
"""

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def build_worldpm_model(
    model_name_or_path: str,
    *,
    torch_dtype=torch.bfloat16,
    trust_remote_code: bool = True,
    **kwargs,
):
    """Builder compatible with ``REWARD_MODEL_CONFIG['model_builder']``.

    Loads the custom ``Qwen2ForRewardModel`` from the checkpoint via
    ``AutoModel`` + ``trust_remote_code``, after patching ``pad_token_id``
    onto the config so ``Qwen2Model.__init__`` doesn't raise.
    """
    # drop kwargs that don't apply to AutoModel
    kwargs.pop("quantization_config", None)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)

    if getattr(config, "pad_token_id", None) is None:
        # Prefer the tokenizer's pad_token_id; fall back to EOS, then BOS.
        pad_token_id = None
        try:
            tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            pad_token_id = tok.pad_token_id
        except Exception:
            pass
        if pad_token_id is None:
            pad_token_id = getattr(config, "eos_token_id", None) or getattr(config, "bos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError(
                f"Cannot determine pad_token_id for {model_name_or_path}; "
                "tokenizer has no pad/eos/bos token set."
            )
        config.pad_token_id = pad_token_id

    # The bundled modeling_qwen2_rm.py calls DynamicCache.from_legacy_cache,
    # which no longer exists in transformers 5.x. We don't need KV caching
    # for scoring, so disable it to avoid that code path.
    config.use_cache = False

    model = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    model.eval()
    return model


_DEFAULT_SYSTEM_BLOCK = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
_EMPTY_SYSTEM_BLOCK = "<|im_start|>system\n<|im_end|>"


def _strip_default_system_prompt(text: str) -> str:
    """Normalize an empty system prompt to match WorldPM's training distribution.

    The Qwen2 chat template injects ``You are a helpful assistant.`` when no
    explicit system message is given. WorldPM was trained with an empty system
    prompt (per the model card's Key Notes), so we rewrite the default block
    to match the model card's own example code.
    """
    if text.startswith(_DEFAULT_SYSTEM_BLOCK):
        return _EMPTY_SYSTEM_BLOCK + text[len(_DEFAULT_SYSTEM_BLOCK):]
    return text


class WorldPMPipeline:
    """Pipeline for Qwen WorldPM models."""

    def __init__(self, task, model, tokenizer):
        self.task = task
        self.model = model.eval()
        self.tokenizer = tokenizer

    def __call__(self, samples, **kwargs):
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        if "max_length" not in kwargs:
            raise ValueError("WorldPMPipeline requires `max_length` to be passed explicitly.")
        max_length = kwargs["max_length"]

        if isinstance(samples, str):
            samples = _strip_default_system_prompt(samples)
        else:
            samples = [_strip_default_system_prompt(s) for s in samples]

        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
            )

        # Qwen2ForRewardModel returns SequenceClassifierOutputWithPast with .logits (B, 1) or (B,)
        if hasattr(outputs, "logits"):
            scores = outputs.logits
        elif isinstance(outputs, (list, tuple)):
            scores = outputs[0]
        elif isinstance(outputs, dict):
            scores = next(iter(outputs.values()))
        else:
            scores = outputs

        if scores.dim() > 1 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        return scores
