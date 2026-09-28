"""Prompt loading. Reuses DFlash's benchmark dataset formats so numbers are comparable."""

from __future__ import annotations

import random

import torch


def load_prompts(dataset: str, n: int, seed: int = 0) -> list[str]:
    from dflash.benchmark import load_and_process_dataset

    rows = load_and_process_dataset(dataset)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[i]["turns"][0] for i in order[:n]]


def format_prompt(tokenizer, prompt: str) -> str:
    """Chat template when the tokenizer has one (DFlash Qwen3 drafters are trained for
    non-thinking mode), else a plain completion prompt for base models (e.g. LayerSkip
    Llama3-8B). The text includes BOS; tokenize with add_special_tokens=False."""
    if tokenizer.chat_template:
        kw = {"enable_thinking": False} if "enable_thinking" in tokenizer.chat_template else {}
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, **kw)
    return (tokenizer.bos_token or "") + f"Question: {prompt}\nAnswer:"


def encode(tokenizer, prompt: str, device: str = "cuda") -> torch.Tensor:
    text = format_prompt(tokenizer, prompt)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def stop_ids(model, tokenizer) -> list[int]:
    ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return [ids] if isinstance(ids, int) else list(ids)
