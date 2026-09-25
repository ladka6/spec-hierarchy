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


def encode(tokenizer, prompt: str, device: str = "cuda") -> torch.Tensor:
    # DFlash Qwen3 drafters are trained for non-thinking mode.
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


def stop_ids(model, tokenizer) -> list[int]:
    ids = model.generation_config.eos_token_id or tokenizer.eos_token_id
    return [ids] if isinstance(ids, int) else list(ids)
