"""Experiment 99: is the cheap target copy right at DFlash's forks? (offline, T=0)

exp91 (holes): DFlash failures are single forks; with the right token at the first error e the block redrafts well
(10.2 accepted vs 8.0). exp97/98: drafters reading the target's features are wrong at e together (right at e
0.08-0.28); PARD (own computation, no features) is right there 0.45-0.49. The copy (quantized target) does real
target-like computation over the text. Is it right at the forks, or wrong together with DFlash (same network)?

Per block (target greedy text = truth, plain DFlash with real target features), first error e. The prefix before e
is the true text, so the copy's prediction at e is one forward pass of the copy over the true text.
Per copy:
  fork top-1 / top-2 / top-3   the true token at e among the copy's top-k
  same mistake                 the copy's top-1 at e is DFlash's wrong token
  fork resolved by set         true token in {copy top-1} + DFlash top-2 (what a 3-branch fork tree would hold)
  overall agreement            copy top-1 == target token over all positions (reference)
  AUROC locator                copy's probability of DFlash's token locating e (positions with the prefix right)
also split by the target's own margin at e (near-tie < --tie vs clear), and DFlash's own top-2/3 coverage at e.

  python scripts/exp99_copyforks.py --copies bnb4:Qwen/Qwen3-8B rtn3:Qwen/Qwen3-8B
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

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402
from exp89_hedge import auroc  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_mid, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--copies", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--tie", type=float, default=0.1)
    ap.add_argument("--out", default="exp99_copyforks")
    args = ap.parse_args()
    dev = "cuda"

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    seqs = []          # x (cpu) per prompt
    blocks = []        # (prompt index, s, e, truth, dflash draft, dflash top-3 at e, target margin at e)
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
                top = blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float().topk(3, -1).indices
                y = top[:, 0].tolist()
                e = matched(y, truth)
                blocks.append((pi, s, e, truth, y, top[e].tolist() if e < D else None, tmargin[s + e] if e < D
                               else None))
        print(f"[phase 1] {dname}: {len(blocks)} blocks", flush=True)
    del target, draft, feats
    gc.collect()
    torch.cuda.empty_cache()

    fails = [b for b in blocks if b[2] < D]
    nf = len(fails)
    d_cov = {k: sum(b[3][b[2]] in b[5][:k] for b in fails) / nf for k in (2, 3)}
    print(f"[dflash] {len(blocks)} blocks, {nf} with an error; true token in DFlash top-2 at e {d_cov[2]:.3f}, "
          f"top-3 {d_cov[3]:.3f}", flush=True)

    rows, split = [], []
    by_prompt = defaultdict(list)
    for b in blocks:
        by_prompt[b[0]].append(b)
    for spec in args.copies:
        name = spec.split(":", 1)[0] + ":" + Path(spec.split(":", 1)[1]).name
        try:
            m = load_mid(spec, dev)
        except Exception as ex:                                       # noqa: BLE001
            print(f"[{name}] failed to load: {type(ex).__name__}: {ex}", flush=True)
            continue
        hit = defaultdict(int)
        tie_hit = defaultdict(lambda: [0, 0])                          # (near-tie?, k) -> [hits, n]
        agree, agree_n = 0, 0
        lab, score = [], []
        for pi, bl in by_prompt.items():
            x = seqs[pi].to(dev)
            lg = m(x[None], logits_to_keep=0).logits[0]                # [T, V]; lg[t] predicts x[t + 1]
            am = lg.argmax(-1)
            agree += int((am[:-1] == x[1:]).sum())
            agree_n += len(x) - 1
            for _, s, e, truth, y, dtop, tm in bl:
                pr = torch.softmax(lg[s : s + min(e + 1, D)].float(), -1)        # positions 0..e of the block
                ys = torch.tensor(y[: pr.shape[0]], device=dev)
                p_y = pr.gather(-1, ys[:, None])[:, 0].tolist()
                for k in range(pr.shape[0]):
                    lab.append(k == e)
                    score.append(p_y[k])
                if e >= D:
                    continue
                ctop = pr[e].topk(3).indices.tolist()
                want = truth[e]
                for k in (1, 2, 3):
                    hit[f"top{k}"] += want in ctop[:k]
                    tie_hit[(tm < args.tie, k)][0] += want in ctop[:k]
                    tie_hit[(tm < args.tie, k)][1] += 1
                hit["same"] += ctop[0] == y[e]
                hit["set"] += want in set([ctop[0]] + dtop[:2])
            del lg, am
        rows.append({"copy": name, "fork top-1": hit["top1"] / nf, "fork top-2": hit["top2"] / nf,
                     "fork top-3": hit["top3"] / nf, "same mistake as DFlash": hit["same"] / nf,
                     "fork resolved by {copy top-1, DFlash top-2}": hit["set"] / nf,
                     "overall agreement": agree / agree_n, "AUROC locator": auroc(score, lab)})
        for tie in (True, False):
            v = tie_hit[(tie, 1)]
            split.append({"copy": name, "target at e": "near-tie" if tie else "clear", "share of forks": v[1] / nf,
                          "fork top-1": v[0] / max(v[1], 1),
                          "fork top-3": tie_hit[(tie, 3)][0] / max(tie_hit[(tie, 3)][1], 1)})
        print(f"[{name}] done", flush=True)
        m = None
        gc.collect()
        torch.cuda.empty_cache()

    print_table(rows, list(rows[0]), f"1. is the copy right at DFlash's first error? ({nf} forks in {len(blocks)} "
                f"blocks; DFlash's own top-2 {d_cov[2]:.3f}, top-3 {d_cov[3]:.3f}; PARD2 right at e 0.49)")
    print_table(split, list(split[0]), f"2. by the target's own margin at e (near-tie < {args.tie})")
    save_json(args.out, {"args": vars(args), "rows": rows, "split": split, "dflash_cov": d_cov, "forks": nf,
                         "blocks": len(blocks)})


if __name__ == "__main__":
    main()
