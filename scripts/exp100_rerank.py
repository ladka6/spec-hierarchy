"""Experiment 100: fork resolution as re-ranking the drafter's own candidates (offline, T=0).

exp99: open-vocabulary fork accuracy falls steeply with bits (bnb4 0.88, rtn3 0.67, 2-bit LoRA 0.59, rtn2 0.01),
while the true token at DFlash's fork is in DFlash's top-3 0.72 and top-16 0.80-0.99. If the fork is a choice among
the drafter's few candidates, a cheap copy may rank them well even when its open-vocabulary prediction is poor.
Does restricting the copy to DFlash's top-k close the gap between low-bit copies and 4-bit?

Same blocks as exp99 (target greedy text = truth, plain DFlash with real target features, anchor every --stride).
Per copy and mode (open = copy's argmax over the vocabulary; rK = copy's argmax among DFlash's top-K at that
position; hK@tau = open if the copy's top-1 probability >= tau, else rK):
  oracle fork acc      at DFlash's true first error e: the mode's token == truth (all forks)
  acc | in set         rK only: same, on forks whose true token is in DFlash's top-K (ranking quality alone)
  realistic resolved   the mode both FINDS the fork (its first disagreement with DFlash's draft is at e) and gives
                       the true token there; this is what a copy-resolved restart gets (exp27 pc)
  early alarm          the mode's first disagreement is before e (restart at a wrong position)
  miss                 the mode agrees with DFlash at e (fork not found)
  clean false alarm    blocks with no DFlash error where the mode disagrees somewhere
Also split by the target's own margin at e (near-tie < --tie vs clear).

Positions after e are not scored: there the copy would see DFlash's drafted (wrong) text, which the offline
replay of the true text cannot reproduce.

  python scripts/exp100_rerank.py --copies bnb4:Qwen/Qwen3-8B rtn3:Qwen/Qwen3-8B rtn2:Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import gc
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

KS = (2, 4, 8, 16)


def mode_preds(L: torch.Tensor, cand: torch.Tensor, ks=KS, taus=(0.5, 0.8), hks=(4, 16)) -> dict:
    """Per-position predictions of every mode for one block.

    L     [D, V] copy logits; L[k] predicts block position k
    cand  [D, Kmax] DFlash's top-Kmax token ids per block position (descending)
    returns {mode: LongTensor [D]}
    """
    out = {"open": L.argmax(-1)}
    Lc = L.gather(1, cand)                                             # [D, Kmax]
    ar = torch.arange(L.shape[0], device=L.device)
    for k in ks:
        out[f"r{k}"] = cand[ar, Lc[:, :k].argmax(-1)]
    p1 = torch.softmax(L.float(), -1).max(-1).values
    for k in hks:
        for tau in taus:
            out[f"h{k}@{tau}"] = torch.where(p1 >= tau, out["open"], out[f"r{k}"])
    return out


def score_block(pred: list, y: list, truth: list, e: int, D: int) -> str:
    """Outcome of one mode on one block when the mode both locates and resolves the fork.

    pred, y, truth: per-position token lists (mode, DFlash draft, target text); e = DFlash's first error (D: none).
    Positions before e have the true prefix, so the copy's prediction there is exact for the real pipeline.
    """
    for k in range(min(e + 1, D)):
        if pred[k] != y[k]:
            if k < e:
                return "early"
            return "resolved" if pred[k] == truth[k] else "wrong_token"
    return "miss" if e < D else "clean_ok"


@torch.inference_mode()
def main():
    from dflash.model import extract_context_feature
    from exp27_refresh import blocks_logits, matched

    from hspec.data import encode, load_prompts, stop_ids
    from hspec.models import load_draft, load_mid, load_target, load_tokenizer
    from hspec.pipeline import two_stage_generate
    from hspec.utils import print_table, save_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copies", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--out", default="exp100_rerank")
    args = ap.parse_args()
    dev = "cuda"
    kmax = max(KS)

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    # phase 1 (as exp99): target text, DFlash blocks with real features, plus DFlash's top-16 at every position
    seqs, blocks = [], []   # blocks: (prompt index, s, e, truth, y, cand [D, 16] cpu, target margin at e or None)
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            out = target(x[None], output_hidden_states=True, logits_to_keep=0)
            feats = extract_context_feature(out.hidden_states, lids)
            tv = torch.softmax(out.logits[0].float(), -1).topk(2, -1).values
            tmargin = (tv[:, 0] - tv[:, 1]).tolist()
            del out, tv
            xl = x.tolist()
            pi = len(seqs)
            seqs.append(x.cpu())
            for s in range(ids.shape[1], len(x) - bs - 1, args.stride):
                truth = xl[s + 1 : s + 1 + D]
                cand = blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float().topk(kmax, -1).indices
                y = cand[:, 0].tolist()
                e = matched(y, truth)
                blocks.append((pi, s, e, truth, y, cand.cpu(), tmargin[s + e] if e < D else None))
        print(f"[phase 1] {dname}: {len(blocks)} blocks", flush=True)
    del target, draft, feats
    gc.collect()
    torch.cuda.empty_cache()

    fails = [b for b in blocks if b[2] < D]
    clean = len(blocks) - len(fails)
    nf = len(fails)
    cov = {k: sum(b[3][b[2]] in b[5][b[2], :k].tolist() for b in fails) / nf for k in KS}
    print(f"[dflash] {len(blocks)} blocks, {nf} with an error; true token in DFlash top-k at e: "
          + ", ".join(f"k={k} {cov[k]:.3f}" for k in KS), flush=True)

    by_prompt = defaultdict(list)
    for b in blocks:
        by_prompt[b[0]].append(b)

    rows, split = [], []
    for spec in args.copies:
        name = spec.split(":", 1)[0] + ":" + Path(spec.split(":", 1)[1]).name
        try:
            m = load_mid(spec, dev)
        except Exception as ex:                                       # noqa: BLE001
            print(f"[{name}] failed to load: {type(ex).__name__}: {ex}", flush=True)
            continue
        oracle = defaultdict(int)                      # mode -> forks where the mode's token at e is right
        inset = defaultdict(lambda: [0, 0])            # rK -> [right, n] on forks with the truth in DFlash top-K
        outcome = defaultdict(lambda: defaultdict(int))  # mode -> outcome -> count
        tie = defaultdict(lambda: [0, 0])              # (mode, near-tie?) -> [right, n]
        for pi, bl in by_prompt.items():
            x = seqs[pi].to(dev)
            lg = m(x[None], logits_to_keep=0).logits[0]                # [T, V]; lg[t] predicts x[t + 1]
            for _, s, e, truth, y, cand, tm in bl:
                preds = mode_preds(lg[s : s + D], cand.to(dev))
                for mode, pv in preds.items():
                    pl = pv.tolist()
                    outcome[mode][score_block(pl, y, truth, e, D)] += 1
                    if e >= D:
                        continue
                    ok = pl[e] == truth[e]
                    oracle[mode] += ok
                    tie[(mode, tm < args.tie)][0] += ok
                    tie[(mode, tm < args.tie)][1] += 1
                    if mode.startswith("r"):
                        k = int(mode[1:])
                        if truth[e] in cand[e, :k].tolist():
                            inset[mode][0] += ok
                            inset[mode][1] += 1
            del lg
        for mode in outcome:
            oc = outcome[mode]
            rows.append({"copy": name, "mode": mode,
                         "oracle fork acc": oracle[mode] / nf,
                         "acc | in set": (inset[mode][0] / max(inset[mode][1], 1)) if mode in inset else None,
                         "realistic resolved": oc["resolved"] / nf,
                         "wrong token": oc["wrong_token"] / nf,
                         "early alarm": oc["early"] / nf,
                         "miss": oc["miss"] / nf,
                         "clean false alarm": 1 - oc["clean_ok"] / max(clean, 1)})
            for t in (True, False):
                v = tie[(mode, t)]
                split.append({"copy": name, "mode": mode, "target at e": "near-tie" if t else "clear",
                              "share of forks": v[1] / nf, "oracle fork acc": v[0] / max(v[1], 1)})
        print(f"[{name}] done", flush=True)
        m = None
        gc.collect()
        torch.cuda.empty_cache()

    print_table(rows, list(rows[0]), f"1. resolving DFlash's forks: open vocabulary vs DFlash's top-K ({nf} forks in "
                f"{len(blocks)} blocks; DFlash top-K coverage " + ", ".join(f"{k}:{cov[k]:.3f}" for k in KS) + ")")
    print_table(split, list(split[0]), f"2. oracle fork accuracy by the target's own margin at e (near-tie < {args.tie})")
    save_json(args.out, {"args": vars(args), "rows": rows, "split": split, "dflash_cov": cov, "forks": nf,
                         "blocks": len(blocks), "clean_blocks": clean})


if __name__ == "__main__":
    main()
