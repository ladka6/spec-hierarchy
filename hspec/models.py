"""Model loading: target (bf16), middle verifier variants, DFlash drafter."""

from __future__ import annotations

import gc
import time
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


@torch.no_grad()
def fake_quantize_(model, bits: int, group: int = 64):
    """In-place round-to-nearest N-bit quantize-dequantize of every Linear in the decoder layers."""
    qmax = 2 ** bits - 1
    for layer in model.model.layers:
        for mod in layer.modules():
            if isinstance(mod, torch.nn.Linear):
                w = mod.weight
                out_f, in_f = w.shape
                g = w.float().reshape(out_f, in_f // group, group)
                lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
                scale = (hi - lo).clamp_min(1e-8) / qmax
                q = ((g - lo) / scale).round().clamp(0, qmax)
                w.copy_((q * scale + lo).reshape(out_f, in_f).to(w.dtype))


def kept_layers(n_layers: int, keep: int, last: int = 33) -> list[int]:
    """keep layer indices spread evenly over 0..last (incl. both ends)."""
    if keep >= last + 1:
        return list(range(last + 1))
    return sorted({round(i * last / (keep - 1)) for i in range(keep)})


@torch.no_grad()
def skip_layers_(model, keep: int, last: int = 33):
    """Turn every decoder layer not in kept_layers into identity by zeroing its output projections."""
    keep_ids = set(kept_layers(len(model.model.layers), keep, last))
    for i, layer in enumerate(model.model.layers):
        if i not in keep_ids:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
    model.kept_layers = sorted(keep_ids)


def load_mid(spec: str, device: str = "cuda"):
    """Load a middle verifier from a spec string "kind:model_id".

    kinds:
      bnb4  4-bit NF4 (bitsandbytes), bf16 compute. Accurate but slow at batch 1.
      bnb8  8-bit LLM.int8 (bitsandbytes). Very slow at batch 1.
      ao4   4-bit weight-only (torchao, group 128). Fast small-batch kernel.
      ao8   8-bit weight-only (torchao).
      tao4 / tao8  torchao int4 / int8 weight-only via quantize_ on the decoder layers (fast at batch 1)
      keepN layer-skipped target, N layers kept evenly over 0..33 (others identity): quality only
      rtnN  simulated N-bit round-to-nearest weights (group 64), bf16 storage: quality only
      hqqN  N-bit HQQ (N = 2, 3, 4; needs `pip install hqq`). Quality test only: default backend is slow.
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
    elif kind in ("tao4", "tao8"):
        # torchao weight-only int4 / int8 applied directly to the decoder layers of a bf16 model
        # (bypasses transformers' TorchAoConfig weight conversion, which fails under transformers 5)
        from torchao.quantization import quantize_

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device,
                                                     attn_implementation="sdpa")
        if kind == "tao4":
            from torchao.quantization import Int4WeightOnlyConfig

            try:
                cfg = Int4WeightOnlyConfig(group_size=128, int4_packing_format="tile_packed_to_4d")
            except TypeError:
                cfg = Int4WeightOnlyConfig(group_size=128)
        else:
            from torchao.quantization import Int8WeightOnlyConfig

            cfg = Int8WeightOnlyConfig()
        quantize_(model.model.layers, cfg)
        return model.eval()
    elif kind.startswith("keep"):
        # layer-skipped target: keep N decoder layers spread evenly over 0..33 (the deepest layer DFlash
        # reads); the others become identity (their o_proj / down_proj are zeroed, so the residual
        # passes through). bf16 storage: quality test of a pruned feature generator, no speed benefit
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device,
                                                     attn_implementation="sdpa")
        skip_layers_(model, int(kind[4:]))
        return model.eval()
    elif kind.startswith("rtn"):
        # simulated N-bit weight quantization (round-to-nearest, asymmetric, group 64) of the
        # decoder layers, stored back in bf16: quality test only, no speed benefit
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device,
                                                     attn_implementation="sdpa")
        fake_quantize_(model, int(kind[3:]))
        return model.eval()
    elif kind.startswith("hqq"):
        # HQQ calibration-free low-bit quantization (hqq2 / hqq3 / hqq4), group size 64
        from transformers import HqqConfig

        kwargs["quantization_config"] = HqqConfig(nbits=int(kind[3:]), group_size=64)
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = device
    elif kind == "hf":
        kwargs["dtype"] = torch.bfloat16
        kwargs["device_map"] = device
    elif kind in ("ao4", "ao8"):
        # torchao weight-only quantization: fast small-batch kernels (tinygemm int4 on A100)
        from transformers import TorchAoConfig

        if kind == "ao4":
            try:
                from torchao.quantization import Int4WeightOnlyConfig

                quant = Int4WeightOnlyConfig(group_size=128)
            except ImportError:
                quant = "int4_weight_only"
        else:
            try:
                from torchao.quantization import Int8WeightOnlyConfig

                quant = Int8WeightOnlyConfig()
            except ImportError:
                quant = "int8_weight_only"
        kwargs["quantization_config"] = TorchAoConfig(quant_type=quant)
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


class Latency:
    """Adds a fixed delay after every forward of a model (simulated network round trip).

    The delay starts after the GPU work finishes, so it adds serially like a real RTT.
    Change ``.ms`` at any time; 0 disables it."""

    def __init__(self, model, ms: float = 0.0):
        self.ms = ms
        self.handle = model.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.ms > 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(self.ms / 1000.0)

    def remove(self):
        self.handle.remove()


def gpu_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
