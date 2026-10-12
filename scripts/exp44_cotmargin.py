"""Experiment 44: why does DFlash halve on reasoning traces? Distribution shift vs intrinsic difficulty.

exp43: tau 3.65 on thinking-mode traces vs 7.83 on non-thinking answers (Qwen3-8B, official DFlash).
Two explanations:
  shift      the drafter was trained on non-thinking text; on reasoning text it is worse even where
             the target is equally decided
  intrinsic  reasoning text has more near-ties (the target itself is less decided), so any drafter
             accepts less
On greedy responses (thinking on: the <think> part and the answer after it; thinking off: the answer),
per drafted position: the target's top1-top2 probability margin and entropy at that position, and
whether DFlash's draft (anchored every --stride tokens) is correct there. Tables:
  1. how each region's positions are distributed over target-margin bins, mean entropy
  2. DFlash accuracy per margin bin, per region (offsets 1 and 1..8)
  3. thinking-region accuracy predicted from the answer regions' per-bin accuracy and the thinking
     region's own margin mix, vs the actual: intrinsic if they match, shift if actual is lower

  python scripts/exp44_cotmargin.py --n 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dflash.model import extract_context_feature  # noqa: E402

from hspec.data import load_prompts, stop_ids  # noqa: E402
from hspec.lagtrain import draft_logits_lagged  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

BINS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.01]
REGIONS = ["think", "answer_after_think", "answer_nothink"]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["math500", "gsm8k", "humaneval"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=3072)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--K", type=int, default=8, help="offsets scored per anchor")
    ap.add_argument("--out", default="exp44_cotmargin")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    lids = list(draft.target_layer_ids)
    bs = draft.block_size
    end_think = tok.convert_tokens_to_ids("</think>")
    # per drafted position: region id, offset k, target margin, entropy, correct
    R, Kk, M, E, C = [], [], [], [], []
    pos_stats = {r: {"margin": [], "entropy": []} for r in REGIONS}
    for mode in ("think", "nothink"):
        for d in args.datasets:
            for pr in load_prompts(d, args.n):
                text = tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=mode == "think")
                ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
                n0, T = ids.shape[1], len(x)
                out = target(x[None], output_hidden_states=True)
                feats = extract_context_feature(out.hidden_states, lids)
                margin = torch.empty(T, device=x.device)
                ent = torch.empty(T, device=x.device)
                for a in range(0, T, 512):                       # position p's distribution predicts x_{p+1}
                    lg = out.logits[0, a : a + 512].float()
                    lp = torch.log_softmax(lg, -1)
                    p = lp.exp()
                    top = p.topk(2, -1).values
                    margin[a : a + 512] = top[:, 0] - top[:, 1]
                    ent[a : a + 512] = -(p * lp).sum(-1)
                del out
                margin, ent = margin.cpu(), ent.cpu()
                region = torch.full((T,), 2 if mode == "nothink" else 0, dtype=torch.long)
                if mode == "think":
                    hit = (x[n0:] == end_think).nonzero()
                    if len(hit):
                        region[n0 + int(hit[0]) + 1 :] = 1
                for t in range(n0 + 1, T):                       # token t is predicted at position t-1
                    rn = REGIONS[int(region[t])]
                    pos_stats[rn]["margin"].append(float(margin[t - 1]))
                    pos_stats[rn]["entropy"].append(float(ent[t - 1]))
                for s in range(n0, T - bs, args.stride):
                    pred = draft_logits_lagged(draft, target, feats, x, s, 0, bs).argmax(-1)
                    ok = (pred[: args.K] == x[s + 1 : s + 1 + args.K]).cpu()
                    for k in range(args.K):
                        t = s + 1 + k
                        R.append(int(region[t]))
                        Kk.append(k + 1)
                        M.append(float(margin[t - 1]))
                        E.append(float(ent[t - 1]))
                        C.append(bool(ok[k]))
            print(f"[{mode} {d}] done, {len(C)} scored positions", flush=True)

    R, Kk, M, C = torch.tensor(R), torch.tensor(Kk), torch.tensor(M), torch.tensor(C, dtype=torch.float)
    rows = []
    for rn in REGIONS:
        m = torch.tensor(pos_stats[rn]["margin"])
        e = torch.tensor(pos_stats[rn]["entropy"])
        if not len(m):
            continue
        row = {"region": rn, "positions": len(m), "mean_entropy": float(e.mean()), "mean_margin": float(m.mean())}
        for lo, hi in zip(BINS, BINS[1:]):
            row[f"m{lo}-{min(hi, 1)}"] = float(((m >= lo) & (m < hi)).float().mean())
        rows.append(row)
    print_table(rows, list(rows[0]), "share of positions per target-margin bin (p_top1 - p_top2), per region")

    def acc_table(kmask, title):
        arows = []
        for ri, rn in enumerate(REGIONS):
            sel_r = (R == ri) & kmask
            if not sel_r.any():
                continue
            row = {"region": rn, "acc_all": float(C[sel_r].mean())}
            for lo, hi in zip(BINS, BINS[1:]):
                sel = sel_r & (M >= lo) & (M < hi)
                row[f"m{lo}-{min(hi, 1)}"] = float(C[sel].mean()) if sel.sum() >= 20 else None
            arows.append(row)
        print_table(arows, list(arows[0]), title)
        return arows

    a1 = acc_table(Kk == 1, "DFlash accuracy at offset 1, per target-margin bin (None: < 20 positions)")
    a8 = acc_table(Kk <= args.K, f"DFlash accuracy at offsets 1..{args.K}, per target-margin bin")

    # 3. counterfactual: thinking accuracy if per-bin accuracy were the answer regions' (pooled)
    prow = []
    for name, kmask in (("offset 1", Kk == 1), (f"offsets 1..{args.K}", Kk <= args.K)):
        ans = ((R == 1) | (R == 2)) & kmask
        th = (R == 0) & kmask
        pred = 0.0
        for lo, hi in zip(BINS, BINS[1:]):
            b_th = th & (M >= lo) & (M < hi)
            b_an = ans & (M >= lo) & (M < hi)
            if b_th.any():
                acc_an = C[b_an].mean() if b_an.sum() >= 20 else C[ans].mean()
                pred += float(b_th.float().sum() / th.float().sum()) * float(acc_an)
        prow.append({"scope": name, "think_actual": float(C[th].mean()), "think_pred_from_answer_bins": pred,
                     "answer_actual": float(C[ans].mean())})
    print_table(prow, ["scope", "think_actual", "think_pred_from_answer_bins", "answer_actual"],
                "thinking accuracy: actual vs predicted from its margin mix (intrinsic if close, shift if actual lower)")
    save_json(args.out, {"args": vars(args), "margin_mix": rows, "acc_k1": a1, "acc_all": a8, "counterfactual": prow})


if __name__ == "__main__":
    main()
