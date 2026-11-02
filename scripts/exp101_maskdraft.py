"""Experiment 101: can the verifier-side model draft the continuation after a resolved fork? (offline, T=0, no training)

Question behind it: one copy level today is a copy pass (4 GB) plus a DFlash redraft (3.3 GB), because DFlash can only
read the copy's computation through five taps. If the copy could fill placeholder ("mask") slots after the corrected
prefix in the SAME pass, the redraft disappears. Before training anything, measure the untrained floor: what do the
bf16 target and the 4-bit copy predict at placeholder positions placed after a confirmed prefix, and how far is that
from a frozen DFlash redraft with real features from the same point (exp91 / exp26 regime)?

Per block (target greedy text = truth, plain DFlash draft y with real target features, first error e):
  corrected prefix  y[:e] + [true token at e]   (or the whole block when it has no error: the DLoop case)
  anchor a          the index of the last token of the corrected prefix in the sequence
  slots             K placeholder positions appended after a; slot j predicts x[a + 1 + j]; offset o = j + 1
  offset 1          predicted by the state at the real last token (trivial for the target: it is its own greedy text)
  inits (--inits)   what the placeholder input embeddings are, no training anywhere:
    mean            the mean input embedding (a fixed, token-free "mask")
    repeat          the last real token repeated (lookahead / Jacobi initialisation)
    zero            zero embeddings
    dflash          DFlash's own redraft tokens from anchor a with real features (the model rescoring the redraft:
                    this is what a second copy level does today; its argmax at slot j is a "healed" prediction)
    dflash_wrong    the original draft's tail y[e+1:], i.e. the copy's natural view today without any redraft
                    (only offsets that exist in the tail are counted)
  reference         DFlash redraft from a with REAL target features (frozen drafter): the row a trained copy head
                    would have to match for the redraft to be dropped

Models (--models): the bf16 target (ceiling for any copy) and bnb4 (what would be trained). Each block costs one
forward over the sequence prefix per (model, init), no KV reuse; ~1-2 h for 20 prompts x 4 datasets on an A100.

Reports, per model x init and for the DFlash reference:
  1. top-1 accuracy at offsets 1..K+1
  2. accuracy at offset o given all earlier offsets right
  3. mean leading run of right offsets (= accepted continuation after the fork, incl. the free offset 1)
Decision: if the best untrained init on bnb4 is >= ~0.6 at offsets 2-4, train the copy's block head; if it is near
the dflash_wrong row, placeholders need more than a head.

  python scripts/exp101_maskdraft.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import free, load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


class Stats:
    """Per-offset accuracy, conditional accuracy and leading-run length."""

    def __init__(self):
        self.right = defaultdict(lambda: [0, 0])   # offset -> [right, n]
        self.cond = defaultdict(lambda: [0, 0])    # offset -> [right, n] given all earlier offsets right
        self.runs = []

    def add(self, pred: list[int], truth: list[int]):
        alive = True
        run = 0
        for o, (p, t) in enumerate(zip(pred, truth), start=1):
            ok = p == t
            self.right[o][0] += ok
            self.right[o][1] += 1
            if alive:
                self.cond[o][0] += ok
                self.cond[o][1] += 1
                if ok:
                    run += 1
                else:
                    alive = False
        self.runs.append(run)

    def rows(self, label: str, K: int):
        acc = {f"o{o}": (self.right[o][0] / self.right[o][1] if self.right[o][1] else float("nan"))
               for o in range(1, K + 2)}
        cnd = {f"o{o}": (self.cond[o][0] / self.cond[o][1] if self.cond[o][1] else float("nan"))
               for o in range(1, K + 2)}
        n = {f"o{o}": self.right[o][1] for o in range(1, K + 2)}
        return ({"source": label, **acc}, {"source": label, **cnd},
                {"source": label, "leading run": mean(self.runs), "blocks": len(self.runs), **n})


@torch.inference_mode()
def slot_predictions(model, ids: torch.Tensor, slot_ids: list[int] | None, init: str, K: int, mean_emb):
    """Argmax at the last real position and at K appended placeholders. ids: [1, L] context + corrected prefix."""
    emb = model.get_input_embeddings()
    x = emb(ids)                                                       # [1, L, H]
    if init == "mean":
        ph = mean_emb.to(x.dtype).expand(1, K, -1)
    elif init == "zero":
        ph = torch.zeros(1, K, x.shape[-1], dtype=x.dtype, device=x.device)
    elif init == "repeat":
        ph = x[:, -1:].expand(1, K, -1)
    elif init in ("dflash", "dflash_wrong"):
        ph = emb(torch.tensor([slot_ids], device=ids.device, dtype=ids.dtype))
    else:
        raise ValueError(init)
    inp = torch.cat([x, ph], dim=1)
    out = model(inputs_embeds=inp, logits_to_keep=K + 1)
    return out.logits[0, -(K + 1):].argmax(-1).tolist()             # offsets 1..K+1


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--models", nargs="+", default=["target", "bnb4:Qwen/Qwen3-8B"],
                    help="'target' = the loaded bf16 target; otherwise a load_mid spec")
    ap.add_argument("--inits", nargs="+", default=["mean", "repeat", "dflash", "dflash_wrong"])
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--k", type=int, default=12, help="placeholder slots (offsets 2..k+1); <= block_size - 2")
    ap.add_argument("--out", default="exp101_maskdraft")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    K = min(args.k, D - 2)
    lids = list(draft.target_layer_ids)
    mean_emb = target.get_input_embeddings().weight.float().mean(0)

    # 1. collect blocks once: sequence, anchor a, corrected prefix, DFlash redraft from a, original tail
    print("[collect] generating greedy text and DFlash drafts", flush=True)
    seqs = []                       # (x, [(a, prefix_ids, redraft_tokens, wrong_tail, truth)])
    n_blocks = n_full = 0
    plain = []
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl = x.tolist()
            blocks = []
            # need truth for K+1 offsets after a, and a <= s + D
            for s in range(ids.shape[1], len(x) - bs - K - 3, args.stride):
                t = xl[s + 1 : s + 1 + D]
                y = blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].argmax(-1).tolist()
                e = matched(y, t)
                plain.append(e + 1)
                if e >= D:                      # whole block right: slots after the block (DLoop case)
                    a = s + D
                    prefix = y[:D]
                    wrong_tail = []
                    n_full += 1
                else:
                    a = s + 1 + e
                    prefix = y[:e] + [t[e]]
                    wrong_tail = y[e + 1 : e + 1 + K]
                truth = xl[a + 1 : a + 2 + K]                      # offsets 1..K+1
                z = blocks_logits(draft, target, feats, x, [(a, 0)], bs)[0].argmax(-1).tolist()   # z[o-1] -> x[a+o]
                blocks.append((a, prefix, z, wrong_tail, truth))
                n_blocks += 1
            seqs.append((x, ids.shape[1], blocks))
        print(f"[{dname}] {n_blocks} blocks ({n_full} without error)", flush=True)
    print(f"plain DFlash accepted: {mean(plain):.3f}", flush=True)

    # DFlash reference: redraft from a with real features (offset o = z[o-1])
    ref = Stats()
    for _, _, blocks in seqs:
        for a, prefix, z, wrong_tail, truth in blocks:
            ref.add(z[: K + 1], truth)
    stats = {"DFlash redraft, real features": ref}

    # 2. placeholder predictions per model and init
    for mspec in args.models:
        if mspec == "target":
            model, name = target, "target bf16"
        else:
            model, name = load_mid(mspec), mspec
        for init in args.inits:
            st = Stats()
            done = 0
            for x, n_prompt, blocks in seqs:
                xl = x.tolist()
                for a, prefix, z, wrong_tail, truth in blocks:
                    # corrected prefix == the true text up to a (y matched t before e, and t[e] is placed at e)
                    ids = x[None, : a + 1]
                    if init == "dflash":
                        slot_ids, k = z[:K], K
                    elif init == "dflash_wrong":
                        slot_ids, k = wrong_tail, len(wrong_tail)
                        if k == 0:
                            continue
                    else:
                        slot_ids, k = None, K
                    pred = slot_predictions(model, ids, slot_ids, init, k, mean_emb)
                    st.add(pred, truth[: k + 1])
                    done += 1
            print(f"[{name} / {init}] {done} blocks, leading run {mean(st.runs):.3f}", flush=True)
            stats[f"{name} / {init}"] = st
        if mspec != "target":
            free(model)

    cols = ["source"] + [f"o{o}" for o in range(1, K + 2)]
    acc_rows, cond_rows, run_rows = [], [], []
    for label, st in stats.items():
        r1, r2, r3 = st.rows(label, K)
        acc_rows.append(r1)
        cond_rows.append(r2)
        run_rows.append(r3)
    print_table(acc_rows, cols, "1. top-1 accuracy per offset after the corrected prefix (o1 = real last token)")
    print_table(cond_rows, cols, "2. accuracy at offset o given all earlier offsets right")
    print_table(run_rows, ["source", "leading run", "blocks"], "3. leading run of right offsets (accepted continuation)")
    save_json(args.out, {"args": vars(args), "K": K, "blocks": n_blocks, "blocks_without_error": n_full,
                         "plain": mean(plain), "accuracy": acc_rows, "conditional": cond_rows, "runs": run_rows})


if __name__ == "__main__":
    main()
