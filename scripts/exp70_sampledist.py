"""Experiment 70: under sampling, how close is the cheap copy to the target? (offline, Colab-ready)

Lossless speculative sampling accepts a proposed token from q with probability min(1, p/q), so the expected acceptance
at a position is alpha = sum_v min(p(v), q(v)) = 1 - TV(p, q). Earlier T=1 numbers compared a sampled copy token with
a sampled target token, which understates this. Here, per position of target-generated text:
  alpha_copy(T)      = 1 - TV(target, copy) at temperature T
  alpha_drafter(T,k) = 1 - TV(target, DFlash drafter at block position k), k = 1..4, for reference
  P(run >= L)        = expected share of start positions whose copy chain survives L tokens, with acceptance
                       independent across positions given the prefix (product of alpha along the text)
Text: by default sampled from the target at --gen-temp (so the prefix matches the sampling regime); --cache reuses a
file of (ids, n_prompt) sequences instead.

  python scripts/exp70_sampledist.py --gen-temp 1.0 --n 100
  python scripts/exp70_sampledist.py --cache /content/seqs.pt          # greedy text from exp69
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits  # noqa: E402

from hspec.data import format_prompt, load_prompts  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

LENS = (5, 10, 20, 33, 50, 100)


@torch.inference_mode()
def generate_sampled(target, tok, prompts, max_new, batch, temp, dev):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    seqs = []
    for i in range(0, len(prompts), batch):
        enc = tok([format_prompt(tok, p) for p in prompts[i : i + batch]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev)
        out = target.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=1.0, top_k=0,
                              pad_token_id=tok.pad_token_id)
        for j in range(out.shape[0]):
            prompt = enc["input_ids"][j][enc["attention_mask"][j].bool()]
            resp = out[j, enc["input_ids"].shape[1]:]
            stop = (resp == tok.pad_token_id).nonzero()
            resp = resp[: int(stop[0])] if len(stop) else resp
            seqs.append((torch.cat([prompt, resp]).cpu(), len(prompt)))
        print(f"[gen] {len(seqs)}/{len(prompts)}", flush=True)
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--copy", default="bnb4:Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--gen-batch", type=int, default=16)
    ap.add_argument("--gen-temp", type=float, default=1.0)
    ap.add_argument("--cache", default="", help="reuse (ids, n_prompt) sequences from this .pt instead of generating")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.6, 1.0])
    ap.add_argument("--draft-stride", type=int, default=4, help="drafter evaluated at every n-th position")
    ap.add_argument("--save", default="exp70_sampledist")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    tok = load_tokenizer(args.target)
    target = load_target(args.target, dev)
    if args.cache and Path(args.cache).exists():
        seqs = torch.load(args.cache)
        src = f"cache {args.cache}"
    else:
        prompts = []
        for ds in args.datasets:
            try:
                prompts += load_prompts(ds, args.n)
            except Exception as e:  # noqa: BLE001
                print(f"[data] {ds} skipped: {e}", flush=True)
        seqs = generate_sampled(target, tok, prompts, args.max_new, args.gen_batch, args.gen_temp, dev)
        random.Random(0).shuffle(seqs)
        torch.save(seqs, f"/content/seqs_T{args.gen_temp}.pt" if Path("/content").exists() else f"seqs_T{args.gen_temp}.pt")
        src = f"sampled at T={args.gen_temp}"
    print(f"[data] {len(seqs)} sequences ({src}), {time.time() - t0:.0f}s", flush=True)
    copy = load_mid(args.copy, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size

    alpha_c = {T: [] for T in args.temps}
    alpha_d = {T: {k: [] for k in range(1, 5)} for T in args.temps}
    per_seq = {T: [] for T in args.temps}
    with torch.inference_mode():
        for si, (ids, lo) in enumerate(seqs):
            x = ids.to(dev)
            T_len = len(x)
            if T_len - lo < 8:
                continue
            out = target(x[None], output_hidden_states=True, use_cache=False)
            lt = out.logits[0, lo - 1 : -1].float()                       # predicts lo..T-1
            feats = extract_context_feature(out.hidden_states, lids)
            lt_all = out.logits[0].float()
            del out
            lc = copy(x[None], use_cache=False, logits_to_keep=T_len - lo + 1).logits[0, :-1].float()
            anchors = list(range(lo, T_len - 5, args.draft_stride))
            dl = []
            for c in range(0, len(anchors), 64):
                ch = anchors[c : c + 64]
                dl.append(blocks_logits(draft, target, feats, x, [(s, 0) for s in ch], bs)[:, :4].float())
            dl = torch.cat(dl) if dl else None                             # [A, 4, V]: predicts s+1..s+4
            a_idx = torch.tensor(anchors, device=dev)
            for T in args.temps:
                p, q = torch.softmax(lt / T, -1), torch.softmax(lc / T, -1)
                a = torch.minimum(p, q).sum(-1)
                alpha_c[T].append(a.cpu())
                per_seq[T].append(a.cpu())
                if dl is not None:
                    for k in range(1, 5):
                        pk = torch.softmax(lt_all[a_idx + k - 1] / T, -1)  # target at s+k given the true prefix
                        qk = torch.softmax(dl[:, k - 1] / T, -1)
                        alpha_d[T][k].append(torch.minimum(pk, qk).sum(-1).cpu())
            if si % 50 == 0:
                print(f"[run] {si}/{len(seqs)}, {time.time() - t0:.0f}s", flush=True)

    rows = []
    for T in args.temps:
        A = torch.cat(alpha_c[T])
        reach, starts = {L: 0.0 for L in LENS}, {L: 0 for L in LENS}
        for a in per_seq[T]:
            la = torch.log(a.clamp(min=1e-9)).double()
            cs = torch.cat([torch.zeros(1, dtype=torch.float64), la.cumsum(0)])
            for L in LENS:
                if len(a) >= L:
                    reach[L] += float(torch.exp(cs[L:] - cs[: len(a) - L + 1]).sum())
                    starts[L] += len(a) - L + 1
        row = {"T": T, "alpha_copy": float(A.mean()), "alpha_copy_p10": float(A.quantile(0.1)),
               **{f"alpha_drafter_k{k}": float(torch.cat(alpha_d[T][k]).mean()) for k in range(1, 5)},
               **{f"P(run>={L})": reach[L] / max(starts[L], 1) for L in LENS}}
        rows.append(row)
    print_table(rows, list(rows[0]), "expected acceptance 1 - TV(target, proposer) per position; runs = expected "
                                     "survival of a copy chain (product of alpha)")
    save_json(args.save, {"args": vars(args), "source": src, "rows": rows})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
