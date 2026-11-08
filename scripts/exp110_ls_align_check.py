#!/usr/bin/env python
"""
exp110: zero-shot check: z-lab Llama-3.1 DFlash drafter on LayerSkip Llama-3 8B.

What it measures
  1. Drafter config: tap layers, block size, mask token (whatever keys exist).
  2. Embedding / LM-head drift between Llama-3.1-8B-Instruct and LayerSkip
     (the drafter reuses the target's embed + head), incl. the mask-token row.
  3. Per-layer hidden-state shift on identical tokens: cosine, norm ratio,
     linear CKA. Tap layers are marked.
  4. Shift after the drafter's own fc (+ hidden_norm if found), i.e. what the
     drafter actually sees.
  5. Zero-shot tau on LayerSkip (Q/A format) and optionally the reference tau
     on Llama-3.1-Instruct (chat template), plus a greedy losslessness check.

tau is counted API-agnostically: forward calls into target.model during
spec_generate, minus the prefill. Raw counts are printed so you can sanity
check them against the drafter's own code.

Usage
  python scripts/exp110_ls_align_check.py --ref_tau --out results/exp110/exp110.json
  (also needs meta-llama/Llama-3.1-8B-Instruct downloaded; not in setup_layerskip.sh)
Needs ~20 GB GPU (one 8B target + drafter at a time) and access to both
gated Llama repos.
"""
import argparse
import gc
import inspect
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

REF = "meta-llama/Llama-3.1-8B-Instruct"
LS = "facebook/layerskip-llama3-8B"
DRAFT = "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"

PROMPTS = [
    "Write a Python function that checks whether a string is a palindrome, ignoring punctuation and case.",
    "A train travels 180 km in 2.5 hours. What is its average speed in km/h? Explain step by step.",
    "Explain the difference between a process and a thread in operating systems.",
    "Summarize the main causes of the French Revolution in a short paragraph.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Give three tips for writing clear technical documentation.",
    "Implement binary search in C++ and explain its time complexity.",
    "What is the derivative of x^3 * sin(x)? Show the steps.",
    "Describe how a hash table handles collisions.",
    "Write a short email asking a professor for a meeting next week.",
]

DIAG_TEXTS = [
    "The mitochondrion is an organelle found in most eukaryotic cells. It generates most of the cell's supply of adenosine triphosphate, which is used as a source of chemical energy. Mitochondria have their own DNA and are thought to have originated from bacteria.",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    out, i, j = [], 0, 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    return out + left[i:] + right[j:]\n",
    "Question: If 3x + 7 = 22, what is x?\nAnswer: Subtract 7 from both sides to get 3x = 15. Divide both sides by 3, so x = 5.",
    "The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War and is often cited as the origin of the modern system of sovereign states. It established the principle that each state has authority over its own territory and domestic affairs.",
    "To make a simple tomato sauce, heat olive oil in a pan, add chopped garlic and cook until fragrant. Add crushed tomatoes, salt and a pinch of sugar, then simmer for twenty minutes, stirring occasionally. Finish with fresh basil.",
    "In machine learning, overfitting occurs when a model learns the noise in the training data rather than the underlying pattern. Regularization, early stopping and more data are common ways to reduce it.",
    "Dear Professor Smith,\n\nI hope this message finds you well. I would like to ask whether you have time for a short meeting next week to discuss my project proposal. I am available on Tuesday and Thursday afternoons.\n\nBest regards,\nAlex",
    "A hash table maps keys to values using a hash function. When two keys hash to the same bucket, the table resolves the collision either by chaining, where each bucket holds a list, or by open addressing, where it probes for another free slot.",
]


# ---------------------------------------------------------------- helpers
def load(cls, path, **kw):
    try:
        return cls.from_pretrained(path, dtype=torch.bfloat16, **kw)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=torch.bfloat16, **kw)


def free():
    # callers must `del` their own references first; this only collects
    gc.collect()
    torch.cuda.empty_cache()


def find_keys(d, needles, prefix=""):
    """Recursively pull config entries whose key contains any needle."""
    hits = {}
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            if any(n in str(k).lower() for n in needles):
                hits[p] = v
            hits.update(find_keys(v, needles, p))
    return hits


@torch.no_grad()
def collect_hidden(model, tok, texts, max_len=512):
    """Returns list over hidden_states index (0 = embeddings) of (N_tok, d) fp16 CPU."""
    per_layer = None
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(model.device)
        hs = model(ids, output_hidden_states=True).hidden_states
        if per_layer is None:
            per_layer = [[] for _ in hs]
        for i, h in enumerate(hs):
            per_layer[i].append(h[0, 1:].half().cpu())  # drop BOS (huge-norm sink)
    return [torch.cat(x) for x in per_layer]


def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    num = (Y.T @ X).pow(2).sum()
    den = (X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()
    return (num / den).item()


def shift_stats(X, Y):
    X, Y = X.float().cuda(), Y.float().cuda()
    return {
        "cos": F.cosine_similarity(X, Y, dim=-1).mean().item(),
        "norm_ratio_ls_over_ref": (Y.norm(dim=-1) / X.norm(dim=-1)).mean().item(),
        "cka": linear_cka(X, Y),
    }


def call_spec_generate(drafter, **kw):
    params = inspect.signature(drafter.spec_generate).parameters
    if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kw = {k: v for k, v in kw.items() if k in params}
    return drafter.spec_generate(**kw)


@torch.no_grad()
def measure_tau(drafter, target, tok, prompts, fmt, eos_ids, max_new, lossless):
    calls = {"n": 0}
    hook = target.model.register_forward_hook(lambda m, i, o: calls.__setitem__("n", calls["n"] + 1))
    rows = []
    try:
        for p in prompts:
            ids = fmt(p).to(target.device)
            calls["n"] = 0
            torch.cuda.synchronize()
            t0 = time.time()
            out = call_spec_generate(
                drafter, input_ids=ids, max_new_tokens=max_new, temperature=0.0,
                target=target, stop_token_ids=eos_ids,
            )
            torch.cuda.synchronize()
            dt = time.time() - t0
            out = out if torch.is_tensor(out) else out[0]
            out = out.view(1, -1) if out.dim() == 1 else out
            new = out[0, ids.shape[1]:] if out.shape[1] > ids.shape[1] and torch.equal(out[0, :ids.shape[1]], ids[0]) else out[0]
            steps = max(calls["n"] - 1, 1)
            row = {"new_tokens": int(new.numel()), "target_calls": calls["n"],
                   "tau": new.numel() / steps, "tok_per_s": new.numel() / dt}
            if lossless:
                hook.remove()
                g = target.generate(ids, max_new_tokens=max_new, do_sample=False,
                                    eos_token_id=eos_ids, pad_token_id=eos_ids[0])[0, ids.shape[1]:]
                hook = target.model.register_forward_hook(lambda m, i, o: calls.__setitem__("n", calls["n"] + 1))
                n = min(len(g), len(new))
                mism = (g[:n].cpu() != new[:n].cpu()).nonzero()
                row["greedy_match_prefix"] = int(mism[0, 0]) if len(mism) else n
                row["greedy_len"] = int(len(g))
            rows.append(row)
            print(f"  tau={row['tau']:.2f}  new={row['new_tokens']}  calls={row['target_calls']}"
                  + (f"  match_prefix={row['greedy_match_prefix']}/{row['greedy_len']}" if lossless else ""))
    finally:
        hook.remove()
    tot_new = sum(r["new_tokens"] for r in rows)
    tot_steps = sum(max(r["target_calls"] - 1, 1) for r in rows)
    return {"tau_token_weighted": tot_new / tot_steps, "per_prompt": rows}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--ls", default=LS)
    ap.add_argument("--drafter", default=DRAFT)
    ap.add_argument("--ref_tau", action="store_true", help="also measure tau on Llama-3.1-Instruct")
    ap.add_argument("--no_lossless", action="store_true")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--hs_offset", type=int, default=1,
                    help="hidden_states index = layer_id + offset; match z-lab's extraction code")
    ap.add_argument("--out", default="exp110_ls_align_check.json")
    args = ap.parse_args()
    res = {}

    # ---- drafter + config
    drafter = load(AutoModel, args.drafter, trust_remote_code=True).cuda().eval()
    cfg = drafter.config.to_dict()
    res["drafter_cfg"] = find_keys(cfg, ["layer_id", "block", "mask", "target", "num_hidden_layers", "rope"])
    print("Drafter config hits:", json.dumps(res["drafter_cfg"], indent=1, default=str))
    taps = next((v for k, v in res["drafter_cfg"].items() if "layer_id" in k and isinstance(v, list)), None)
    mask_id = next((v for k, v in res["drafter_cfg"].items() if "mask" in k and isinstance(v, int)), None)
    print("taps:", taps, " mask_token_id:", mask_id)

    # ---- phase A: reference target
    print("\n[A] loading", args.ref)
    tok_ref = AutoTokenizer.from_pretrained(args.ref)
    ref = load(AutoModelForCausalLM, args.ref).cuda().eval()
    ref_hs = collect_hidden(ref, tok_ref, DIAG_TEXTS)
    ref_emb = ref.model.embed_tokens.weight.detach().half().cpu()
    ref_head = ref.lm_head.weight.detach().half().cpu()
    if args.ref_tau:
        print("[A] reference tau (chat template)")
        def fmt_ref(p):
            # newer transformers return a BatchEncoding here, older ones a tensor
            out = tok_ref.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt")
            return out if torch.is_tensor(out) else out["input_ids"]
        res["ref_tau"] = measure_tau(drafter, ref, tok_ref, PROMPTS, fmt_ref,
                                     [128001, 128009], args.max_new, lossless=False)
        print(f"[A] ref tau = {res['ref_tau']['tau_token_weighted']:.3f}")
    del ref
    free()

    # ---- phase B: LayerSkip
    print("\n[B] loading", args.ls)
    tok_ls = AutoTokenizer.from_pretrained(args.ls)
    ls = load(AutoModelForCausalLM, args.ls).cuda().eval()
    # identical tokenization check (diag compares token-by-token)
    same_tok = all(tok_ref(t).input_ids == tok_ls(t).input_ids for t in DIAG_TEXTS)
    res["same_tokenization"] = same_tok
    print("same tokenization on diag texts:", same_tok)
    ls_hs = collect_hidden(ls, tok_ls if not same_tok else tok_ref, DIAG_TEXTS)

    # embeddings / head drift
    ls_emb = ls.model.embed_tokens.weight.detach().half().cpu()
    ls_head = ls.lm_head.weight.detach().half().cpu()
    res["embed_cos_mean"] = F.cosine_similarity(ref_emb.float(), ls_emb.float(), dim=-1).mean().item()
    res["head_cos_mean"] = F.cosine_similarity(ref_head.float(), ls_head.float(), dim=-1).mean().item()
    if mask_id is not None:
        res["mask_row"] = {
            "ref_norm": ref_emb[mask_id].float().norm().item(),
            "ls_norm": ls_emb[mask_id].float().norm().item(),
            "cos": F.cosine_similarity(ref_emb[mask_id].float(), ls_emb[mask_id].float(), dim=0).item(),
            "median_row_norm_ls": ls_emb.float().norm(dim=-1).median().item(),
        }
    print("embed cos:", round(res["embed_cos_mean"], 4), " head cos:", round(res["head_cos_mean"], 4),
          " mask row:", res.get("mask_row"))
    del ref_emb, ref_head, ls_emb, ls_head
    free()

    # per-layer hidden-state shift
    print("\nlayer  cos    norm_ratio  cka   (* = drafter tap)")
    res["layers"] = {}
    tap_set = set(taps or [])
    for l in range(len(ref_hs) - 1):
        s = shift_stats(ref_hs[l + args.hs_offset], ls_hs[l + args.hs_offset]) if l + args.hs_offset < len(ref_hs) else None
        if s is None:
            continue
        res["layers"][l] = s
        print(f"{l:>4}{'*' if l in tap_set else ' '} {s['cos']:.3f}   {s['norm_ratio_ls_over_ref']:.3f}     {s['cka']:.3f}")

    # what the drafter actually sees: fc (+ hidden_norm)
    if taps:
        d = ref_hs[0].shape[-1]
        fc = next((m for n, m in drafter.named_modules()
                   if isinstance(m, torch.nn.Linear) and m.in_features == len(taps) * d), None)
        hn = next((m for n, m in drafter.named_modules() if "hidden_norm" in n), None)
        if fc is not None:
            def fused(hs):
                x = torch.cat([hs[l + args.hs_offset] for l in taps], -1).to("cuda", torch.bfloat16)
                with torch.no_grad():
                    y = fc(x)
                    y = hn(y) if hn is not None else y
                return y.float()
            res["fused_after_fc"] = shift_stats(fused(ref_hs), fused(ls_hs))
            print("\nafter fc" + (" + hidden_norm" if hn is not None else "") + ":", res["fused_after_fc"])
        else:
            print("\ncould not find fc (Linear with in_features = n_taps * d)")
    del ref_hs, ls_hs
    free()

    # zero-shot tau on LayerSkip
    print("\n[B] zero-shot tau on LayerSkip (Q/A format)")
    fmt_ls = lambda p: tok_ls(f"Question: {p}\nAnswer:", return_tensors="pt").input_ids
    eos_ls = [tok_ls.eos_token_id] if isinstance(tok_ls.eos_token_id, int) else list(tok_ls.eos_token_id)
    res["ls_tau"] = measure_tau(drafter, ls, tok_ls, PROMPTS, fmt_ls, eos_ls,
                                args.max_new, lossless=not args.no_lossless)
    print(f"[B] LayerSkip zero-shot tau = {res['ls_tau']['tau_token_weighted']:.3f}"
          + (f"  (ref {res['ref_tau']['tau_token_weighted']:.3f})" if "ref_tau" in res else ""))

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("saved", args.out)


if __name__ == "__main__":
    main()
