"""Experiment 98: do drafters WITHOUT target features fail where DFlash fails? (offline, T=0)

exp97: alternatives that share DFlash's information source (its next-best paths, samples, DFlash fine-tunes) are right
at DFlash's first error only 8-25% of the time (independent errors would give ~90%). Here: drafters that read only the
text, through their own weights:
  pard:<id>    PARD parallel drafter (Qwen3 family): context + mask tokens, one forward pass, 15 tokens
               (PARD2 checkpoints are run the same way, i.e. in their target-independent mode)
  dream:<id>   Dream diffusion LM: 15 tokens by its own diffusion_generate (--dream-steps denoising steps)
Phase 1 computes the target's greedy text, DFlash's drafts and the anchors; phase 2 loads each drafter in turn.

Per drafter: its own accepted length, and against DFlash (primary):
  rescue rate      share of DFlash failures where this drafter's draft gets further than DFlash
  right at e       when this drafter is still on the target's path at DFlash's first error e, how often it is right
                   there (independent errors -> its accuracy at a correct prefix; DFlash variants gave 0.08-0.25)
  best of both     accepted length when DFlash's and this drafter's drafts are verified together
and the union of all drafters.

  python scripts/exp98_dllm_decor.py --drafters pard:amd/PARD-Qwen3-0.6B dream:Dream-org/Dream-v0-Instruct-7B
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def own_accuracy(chains, truths):
    """Accuracy at a correct prefix: P(token right | everything before right), pooled over positions."""
    ok, tot = 0, 0
    for c, t in zip(chains, truths):
        m = matched(c, t)
        tot += min(m + 1, len(c))
        ok += m
    return ok / max(tot, 1)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--drafters", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--dream-steps", type=int, default=8)
    ap.add_argument("--out", default="exp98_dllm_decor")
    args = ap.parse_args()
    dev = "cuda"

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    blocks = []                                   # (x cpu, s, truth, dflash draft)
    for dname in args.datasets:
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl, xc = x.tolist(), x.cpu()
            for s in range(ids.shape[1], len(x) - bs - 1, args.stride):
                y = blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].argmax(-1).tolist()
                blocks.append((xc, s, xl[s + 1 : s + 1 + D], y))
        print(f"[phase 1] {dname}: {len(blocks)} blocks", flush=True)
    del target, draft
    gc.collect()
    torch.cuda.empty_cache()

    drafts = {}                                   # name -> list of chains aligned with blocks
    for spec in args.drafters:
        kind, mid = spec.split(":", 1)
        name = mid.split("/")[-1]
        try:
            if kind == "pard":
                from transformers import AutoConfig, AutoModelForCausalLM
                cfg = AutoConfig.from_pretrained(mid)
                m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map=dev).eval()
                pard = int(cfg.pard_token)
                out = []
                for xc, s, t, _ in blocks:
                    inp = torch.cat([xc[: s + 1], torch.full((D - 1,), pard, dtype=xc.dtype)]).to(dev)
                    lg = m(inp[None], logits_to_keep=D).logits[0]
                    out.append(lg.argmax(-1).tolist())
            elif kind == "dream":
                from transformers import AutoModel
                m = AutoModel.from_pretrained(mid, dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
                out = []
                for i, (xc, s, t, _) in enumerate(blocks):
                    inp = xc[: s + 1].to(dev)[None]
                    res = m.diffusion_generate(inp, attention_mask=torch.ones_like(inp), max_new_tokens=D,
                                               steps=args.dream_steps, temperature=0.0, alg="entropy",
                                               return_dict_in_generate=True, output_history=False)
                    out.append(res.sequences[0, s + 1 : s + 1 + D].tolist())
                    if i % 500 == 0:
                        print(f"[{name}] {i}/{len(blocks)}", flush=True)
            else:
                raise ValueError(f"unknown drafter kind {kind}")
            drafts[name] = out
            print(f"[{name}] done", flush=True)
        except Exception as e:
            print(f"[{name}] failed: {type(e).__name__}: {e}", flush=True)
        m = None
        gc.collect()
        torch.cuda.empty_cache()

    truths = [t for _, _, t, _ in blocks]
    prim = [y for _, _, _, y in blocks]
    a_prim = [matched(y, t) for y, t in zip(prim, truths)]
    rows = [{"drafter": "DFlash (primary)", "accepted alone": sum(a + 1 for a in a_prim) / len(a_prim),
             "accuracy at correct prefix": own_accuracy(prim, truths)}]
    for name, ch in drafts.items():
        a_alt = [matched(c, t) for c, t in zip(ch, truths)]
        fails = [i for i, a in enumerate(a_prim) if a < D]
        resc = sum(a_alt[i] > a_prim[i] for i in fails) / max(len(fails), 1)
        alive = [i for i in fails if a_alt[i] >= a_prim[i]]
        right = sum(ch[i][a_prim[i]] == truths[i][a_prim[i]] for i in alive) / max(len(alive), 1)
        rows.append({"drafter": name, "accepted alone": sum(a + 1 for a in a_alt) / len(a_alt),
                     "accuracy at correct prefix": own_accuracy(ch, truths),
                     "rescue rate": resc, "alive at e": len(alive) / max(len(fails), 1), "right at e": right,
                     "best of DFlash + this": sum(max(a, b) + 1 for a, b in zip(a_prim, a_alt)) / len(a_prim)})
    if len(drafts) > 1:
        best = [max([a_prim[i]] + [matched(ch[i], truths[i]) for ch in drafts.values()]) + 1
                for i in range(len(blocks))]
        rows.append({"drafter": "union of all", "best of DFlash + this": sum(best) / len(best)})
    cols = ["drafter", "accepted alone", "accuracy at correct prefix", "rescue rate", "alive at e", "right at e",
            "best of DFlash + this"]
    for r in rows:
        for c in cols:
            r.setdefault(c, None)
    print_table(rows, cols, f"do drafters without target features fail where DFlash fails? ({len(blocks)} blocks)")
    print("[reference] DFlash variants (exp97): right at e 0.08-0.25; independent errors would give about the "
          "drafter's accuracy at a correct prefix", flush=True)
    save_json(args.out, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
