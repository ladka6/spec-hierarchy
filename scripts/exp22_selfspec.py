"""Experiment 22: does post-training destroy self-speculation (early-exit drafting)?

Self-speculative decoding (LayerSkip): draft K tokens greedily from layer E (the model's own
first E layers + final norm + LM head), verify them with one full pass. Under greedy decoding
the early layers only see tokens, so drafting from the exit on an accepted prefix gives exactly
the teacher-forced exit prediction on the model's own greedy trajectory. Acceptance can
therefore be computed offline, exactly, from per-position exit/final agreement:

  round starting at s: accept the longest run of agreeing positions, up to K; the verify pass
  adds one token (correction or bonus); the next round starts after that.

Reports per model, exit E and draft length K:
  tau       tokens per full pass (accepted + 1)
  est_x     speedup estimate over AR, tau / (1 + K*E/L) (layer-count cost model, no overheads)
  eps       per-position exit disagreement

Optionally GSM8K accuracy (--gsm8k N) so a preservation method can be judged on task quality.

  python scripts/exp22_selfspec.py --models facebook/layerskip-llama3-8B /scratch/.../ls-math200
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp12_exit_scan import encode_any  # noqa: E402

from hspec.data import load_prompts  # noqa: E402
from hspec.models import free, load_target, load_tokenizer  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def rounds(agree: list[bool], K: int):
    """Tokens per full pass when drafting K at a time over an agreement vector."""
    s, taus = 0, []
    n = len(agree)
    while s < n:
        a = 0
        while a < K and s + a < n and agree[s + a]:
            a += 1
        taus.append(a + 1)
        s += a + 1
    return taus


def last_number(text: str):
    m = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return m[-1].replace(",", "").rstrip(".") if m else None


@torch.inference_mode()
def gsm8k_acc(model, tok, n, max_new=320, batch=16):
    from datasets import load_dataset

    rows = list(load_dataset("openai/gsm8k", "main", split="test"))[:n]
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ok = 0
    for b in range(0, len(rows), batch):
        chunk = rows[b : b + batch]
        ids = [encode_any(tok, r["question"], "cpu")[0].tolist() for r in chunk]
        L = max(len(x) for x in ids)
        inp = torch.tensor([[pad] * (L - len(x)) + x for x in ids], device="cuda")
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids], device="cuda")
        out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=pad)
        for r, o in zip(chunk, out):
            text = tok.decode(o[L:], skip_special_tokens=True)
            text = text.split("\nQuestion:")[0]            # base models keep writing new questions
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            ok += last_number(text) == gold
    return ok / len(rows)


@torch.inference_mode()
def evaluate(model_id, name, args):
    tok = load_tokenizer(model_id)
    model = load_target(model_id)
    L = model.config.num_hidden_layers
    eos = model.generation_config.eos_token_id
    agree = {E: [] for E in args.exits}
    for d in args.datasets:
        for p in load_prompts(d, args.n):
            ids = encode_any(tok, p)
            n0 = ids.shape[1]
            seq = model.generate(ids, max_new_tokens=args.max_new, do_sample=False, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            if seq.shape[1] - n0 < 9:
                continue
            out = model(seq, output_hidden_states=True)
            final = out.logits[0, n0 - 1 : -1].argmax(-1)
            for E in args.exits:
                h = model.model.norm(out.hidden_states[E][0, n0 - 1 : -1])
                agree[E].append((model.lm_head(h).argmax(-1) == final).tolist())
            del out
        print(f"[{name} | {d}] done", flush=True)
    rows = []
    for E in args.exits:
        eps = 1 - mean(sum(agree[E], []))
        for K in args.ks:
            taus = sum((rounds(a, K) for a in agree[E]), [])
            tau = mean(taus)
            rows.append({"model": name, "exit": E, "of": L, "K": K, "eps": eps, "tau": tau,
                         "est_x": tau / (1 + K * E / L)})
    acc = gsm8k_acc(model, tok, args.gsm8k) if args.gsm8k else float("nan")
    del model
    free()
    return rows, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="path or name=path")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--exits", nargs="+", type=int, default=[4, 6, 8, 12, 16])
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--gsm8k", type=int, default=200, help="GSM8K test questions for accuracy (0 = skip)")
    ap.add_argument("--out", default="exp22_selfspec")
    args = ap.parse_args()
    rows, accs = [], []
    for m in args.models:
        name, path = m.split("=", 1) if "=" in m else (Path(m).name, m)
        r, acc = evaluate(path, name, args)
        rows += r
        accs.append({"model": name, "gsm8k_acc": acc})
        best = {}
        for x in rows:
            if x["model"] not in best or x["est_x"] > best[x["model"]]["est_x"]:
                best[x["model"]] = x
        print_table(rows, ["model", "exit", "of", "K", "eps", "tau", "est_x"],
                    "self-speculative decoding from the model's own early exit (exact greedy acceptance)")
        print_table([{**b, **a} for b, a in zip(best.values(), accs)],
                    ["model", "exit", "K", "tau", "est_x", "gsm8k_acc"], "best setting per model + GSM8K accuracy")
        save_json(args.out, {"args": vars(args), "rows": rows, "acc": accs})


if __name__ == "__main__":
    main()
