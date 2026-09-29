"""Make a fine-tuned target: LoRA SFT of the target on one domain, merged into a full checkpoint.

Used to study how a DFlash drafter trained for the base target degrades when the target is
fine-tuned (continual post-training / per-domain adapters). Domains use training splits only
(no overlap with the evaluation sets):

  math   GSM8K train, question -> worked solution
  code   MBPP (full) train, task -> code
  chat   Alpaca, instruction (+input) -> output

  python scripts/lora_target.py --domain math --steps 1000 --out /scratch-shared/$USER/hspec_drift/qwen3-8b-math
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.models import load_target, load_tokenizer  # noqa: E402


def domain_pairs(domain: str, seed: int = 0):
    from datasets import load_dataset

    if domain == "math":
        rows = load_dataset("openai/gsm8k", "main", split="train")
        pairs = [(r["question"], r["answer"].replace("####", "The answer is")) for r in rows]
    elif domain == "code":
        rows = load_dataset("google-research-datasets/mbpp", "full", split="train")
        pairs = [(r["text"], "```python\n" + r["code"] + "\n```") for r in rows]
    elif domain == "chat":
        rows = load_dataset("tatsu-lab/alpaca", split="train")
        pairs = [(r["instruction"] + (("\n\n" + r["input"]) if r["input"] else ""), r["output"]) for r in rows]
    else:
        raise ValueError(domain)
    random.Random(seed).shuffle(pairs)
    return pairs


def encode_pair(tok, q, a, max_len):
    kw = {"enable_thinking": False} if tok.chat_template and "enable_thinking" in tok.chat_template else {}
    if tok.chat_template:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                         add_generation_prompt=True, **kw)
    else:
        prompt = (tok.bos_token or "") + f"Question: {q}\nAnswer:"
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(a + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    return ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--domain", required=True, choices=["math", "code", "chat"])
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    tok = load_tokenizer(args.target)
    model = load_target(args.target, args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                             "gate_proj", "up_proj", "down_proj"]))
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"{args.domain}: LoRA r={args.rank}, {sum(p.numel() for p in params) / 1e6:.1f}M trainable, "
          f"{args.steps} steps x {args.batch} examples", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 30) * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)))
    pairs = domain_pairs(args.domain)
    it, t0 = 0, time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.batch // args.micro):
            batch = [encode_pair(tok, *pairs[(it + j) % len(pairs)], args.max_len) for j in range(args.micro)]
            it += args.micro
            L = max(len(b[0]) for b in batch)
            pad = tok.pad_token_id if tok.pad_token_id is not None else 0
            ids = torch.tensor([b[0] + [pad] * (L - len(b[0])) for b in batch], device=args.device)
            lab = torch.tensor([b[1] + [-100] * (L - len(b[1])) for b in batch], device=args.device)
            att = torch.tensor([[1] * len(b[0]) + [0] * (L - len(b[0])) for b in batch], device=args.device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss * args.micro / args.batch).backward()
            tot += float(loss) * args.micro / args.batch
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 50 == 0:
            print(f"[step {step + 1}/{args.steps}] loss {tot:.3f} {(time.time() - t0) / (step + 1):.2f}s/step", flush=True)
    model = model.merge_and_unload()
    model.config.use_cache = True
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved merged target to {out}")


if __name__ == "__main__":
    main()
