"""Experiment 54: does a DFlash preview of the target's next tokens help judge where a reasoning sample is going?

Idea: a block diffusion drafter reads the target's own hidden states and writes the next 15 tokens the target
would produce, at a small fraction of the target's cost. Search and best-of-N methods today score samples on the
text generated so far (Speculative Rejection, partial reward models). If a cheap preview of the continuation makes
those scores predict the final outcome better, the drafter becomes a foresight model for test-time search.

Setup: GSM8K test problems, Qwen3-8B (non-thinking), N samples per problem at temperature T. For each sample, at
response offsets t in --checkpoints, the partial solution is scored by a judge under five views:
  prefix     the text so far (what partial-reward pruning sees)
  preview    + the DFlash drafter's 15-token block from the target's features at t (cheap foresight)
  greedy     + the target's own 15-token greedy continuation (what a perfect drafter would show; costs 15 passes)
  future     + the sample's actual next 15 tokens (oracle for this sample)
  final      the full response (upper bound: what the judge can do at the end)
Judge: Qwen3-8B itself, asked whether the partial solution is correct so far; score = P(Yes) / (P(Yes) + P(No)).
Metric: within-problem AUC of the score for the sample's final correctness, averaged over problems where samples
disagree, per checkpoint and view; plus pruning: accuracy of the samples kept when keeping the top half by score.

  python scripts/exp54_foresight.py --problems 100 --samples 8
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402

VIEWS = ["prefix", "preview", "greedy", "future", "future48", "final"]


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--problems", type=int, default=100)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--checkpoints", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--scorers", nargs="+", default=["prm_last", "prm_min"], help="prm_last, prm_min, self")
    ap.add_argument("--prm", default="Qwen/Qwen2.5-Math-PRM-7B")
    ap.add_argument("--out", default="exp54_foresight.json")
    args = ap.parse_args()
    from datasets import load_dataset

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    yes_id = tok.encode("Yes", add_special_tokens=False)[0]
    no_id = tok.encode("No", add_special_tokens=False)[0]
    eos = target.generation_config.eos_token_id
    eos = set([eos] if isinstance(eos, int) else eos)

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    random.Random(0).shuffle(rows)
    rows = rows[: args.problems]

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    @torch.inference_mode()
    def judge_self(problem, partials):
        texts = [chat(f"Problem:\n{problem}\n\nPartial solution (may be cut off mid-sentence):\n{p}\n\n"
                      "Is the partial solution correct so far, with no mistakes in what it says? "
                      "Answer with Yes or No only.") for p in partials]
        out = []
        for i in range(0, len(texts), 8):
            enc = tok(texts[i : i + 8], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            logits = target(**enc).logits[:, -1].float()
            lp = torch.log_softmax(logits[:, [yes_id, no_id]], -1)
            out += lp[:, 0].exp().tolist()
        return out

    prm = prm_tok = None
    if any(sc.startswith("prm") for sc in args.scorers):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        prm_tok = AutoTokenizer.from_pretrained(args.prm, trust_remote_code=True)
        pcfg = AutoConfig.from_pretrained(args.prm, trust_remote_code=True)
        for k in ("pad_token_id", "bos_token_id", "eos_token_id"):       # remote code predates transformers 5
            if not hasattr(pcfg, k) or getattr(pcfg, k, None) is None:
                setattr(pcfg, k, getattr(prm_tok, k, None) if getattr(prm_tok, k, None) is not None
                        else prm_tok.eos_token_id)
        prm = AutoModel.from_pretrained(args.prm, config=pcfg, dtype=torch.bfloat16, device_map="cuda",
                                        trust_remote_code=True).eval()
        sep_id = prm_tok.encode("<extra_0>")[0]

    @torch.inference_mode()
    def judge_prm(problem, partials):
        """Qwen2.5-Math-PRM: steps split at blank lines, a step reward at every <extra_0>; (last, min)."""
        last, mn = [], []
        for p in partials:
            steps = [x for x in re.split(r"\n\s*\n", p.strip()) if x.strip()] or [p]
            msgs = [{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
            ids = prm_tok(prm_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False),
                          return_tensors="pt").input_ids.to("cuda")
            logits = prm(input_ids=ids)[0][0].float()                      # [T, 2]
            r = torch.softmax(logits[ids[0] == sep_id], -1)[:, 1]
            last.append(float(r[-1]))
            mn.append(float(r.min()))
        return last, mn

    def judge(problem, partials):
        out = {}
        if "self" in args.scorers:
            out["self"] = judge_self(problem, partials)
        if prm is not None:
            out["prm_last"], out["prm_min"] = judge_prm(problem, partials)
        return {k: v for k, v in out.items() if k in args.scorers}

    @torch.inference_mode()
    def preview(ids):
        out = target(ids[None], output_hidden_states=True)
        feats = extract_context_feature(out.hidden_states, lids)
        s = len(ids) - 1
        return draft_logits_lagged(draft, target, feats, ids, s, 0, bs).argmax(-1)

    @torch.inference_mode()
    def greedy(ids, n):
        g = target.generate(ids[None], max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id or 0)
        return g[0, len(ids):]

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    per_ck = {sc: {t: {v: [] for v in VIEWS} for t in args.checkpoints} for sc in args.scorers}
    prune = {sc: {t: {v: [0, 0] for v in VIEWS} for t in args.checkpoints} for sc in args.scorers}
    acc_all, used = [], 0
    for pi, r in enumerate(rows):
        gold = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        prompt = chat(r["question"])
        p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0].cuda()
        with torch.inference_mode():
            gen = target.generate(p_ids[None].repeat(args.samples, 1), max_new_tokens=args.max_new, do_sample=True,
                                  temperature=args.temperature, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
        resps = []
        for row in gen[:, len(p_ids):]:
            row = row.tolist()
            cut = next((i for i, t in enumerate(row) if t in eos), None)
            resps.append(row[: cut + 1] if cut is not None else row)
        correct = []
        for rr in resps:
            a = last_number(tok.decode(rr, skip_special_tokens=True))
            correct.append(a is not None and abs(a - gold) < 1e-6)
        acc_all += correct
        if all(correct) or not any(correct):
            print(f"{pi + 1}/{len(rows)}: {sum(correct)}/{len(correct)} correct, skipped", flush=True)
            continue
        used += 1
        for t in args.checkpoints:
            idx = [i for i, rr in enumerate(resps) if len(rr) > t + 48]
            lab = [correct[i] for i in idx]
            if len(set(lab)) < 2:
                continue
            texts = {v: [] for v in VIEWS}
            for i in idx:
                rr = resps[i]
                pre = torch.tensor(rr[:t], device=p_ids.device)
                full_prefix = torch.cat([p_ids, pre])
                dec = lambda x: tok.decode(x, skip_special_tokens=True)  # noqa: E731
                texts["prefix"].append(dec(rr[:t]))
                texts["preview"].append(dec(rr[:t] + preview(full_prefix).tolist()))
                texts["greedy"].append(dec(rr[:t] + greedy(full_prefix, bs - 1).tolist()))
                texts["future"].append(dec(rr[: t + bs - 1]))
                texts["future48"].append(dec(rr[: t + 47]))
                texts["final"].append(dec(rr))
            for v in VIEWS:
                for sname, sc in judge(r["question"], texts[v]).items():
                    a = auc(sc, lab)
                    if a is not None:
                        per_ck[sname][t][v].append(a)
                    keep = sorted(range(len(sc)), key=lambda j: -sc[j])[: max(1, len(sc) // 2)]
                    prune[sname][t][v][0] += sum(lab[j] for j in keep)
                    prune[sname][t][v][1] += len(keep)
        print(f"{pi + 1}/{len(rows)}: {sum(correct)}/{len(correct)} correct, scored", flush=True)

    print(f"\nsample accuracy {sum(acc_all) / len(acc_all):.3f}; {used} problems with mixed outcomes used")
    res = {}
    for sname in args.scorers:
        print(f"\n== scorer {sname}: within-problem AUC for final correctness (0.5 = chance)")
        print(f"{'checkpoint':>10s} " + " ".join(f"{v:>9s}" for v in VIEWS) + "   problems")
        res[sname] = {}
        for t in args.checkpoints:
            pc = per_ck[sname][t]
            m = {v: (sum(x) / len(x) if x else float("nan")) for v, x in pc.items()}
            res[sname][t] = {"auc": m, "n": len(pc["prefix"]),
                             "kept_acc": {v: c / max(n, 1) for v, (c, n) in prune[sname][t].items()}}
            print(f"{t:10d} " + " ".join(f"{m[v]:9.3f}" for v in VIEWS) + f"   {len(pc['prefix'])}")
        print("accuracy of the samples kept when keeping the top half by score")
        for t in args.checkpoints:
            print(f"{t:10d} " + " ".join(f"{res[sname][t]['kept_acc'][v]:9.3f}" for v in VIEWS))
    Path(args.out).write_text(json.dumps({"args": vars(args), "results": res}, indent=1))


if __name__ == "__main__":
    main()
