"""Model loading: target (bf16), middle verifier variants, DFlash drafter."""

from __future__ import annotations

import gc
import warnings

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# bitsandbytes 8-bit prints this on every matmul; it floods the logs
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")


def load_target(model_id: str, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    return model.to(device).eval()


def load_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id)


def load_mid(spec: str, device: str = "cuda"):
    """Load a middle verifier from a spec string "kind:model_id".

    kinds:
      bnb4  4-bit NF4 (bitsandbytes), bf16 compute
      bnb8  8-bit LLM.int8 (bitsandbytes)
      hf    plain bf16 checkpoint (e.g. a smaller model of the same family,
            or a pre-quantized AWQ/GPTQ checkpoint whose kernels are installed)
    """
    kind, model_id = spec.split(":", 1)
    kwargs = {"attn_implementation": "sdpa"}
    if kind == "bnb4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        kwargs["device_map"] = device
        kwargs["dtype"] = torch.bfloat16
    elif kind == "bnb8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = device
        kwargs["dtype"] = torch.bfloat16
    elif kind == "hf":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = device
    else:
        raise ValueError(f"unknown mid kind {kind!r} in spec {spec!r}")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model.eval()


def load_draft(draft_id: str, device: str = "cuda"):
    from dflash.model import DFlash2DraftModel, DFlashDraftModel

    config = AutoConfig.from_pretrained(draft_id)
    cls = DFlash2DraftModel if "DFlash2DraftModel" in (config.architectures or []) else DFlashDraftModel
    draft = cls.from_pretrained(draft_id, attn_implementation="sdpa", dtype=torch.bfloat16)
    return draft.to(device).eval()


def same_hidden_space(a, b) -> bool:
    """True if b's hidden states can stand in for a's (same width and depth)."""
    ca, cb = a.config, b.config
    return ca.hidden_size == cb.hidden_size and ca.num_hidden_layers == cb.num_hidden_layers


def free(*models):
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


def gpu_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
