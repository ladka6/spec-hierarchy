"""G1 data: Qwen3-8B greedy responses on the evaluation benchmarks, thinking ON and OFF (paired: same prompts).

Purpose: find where and why block drafters (DFlash) lose acceptance on reasoning traces. exp43 measured tau 3.65 with
thinking vs 7.83 without, but on only 18 prompts; this builds a proper set.

Two steps (the vLLM environment does not need `datasets` / dflash):
  prompts  (hspec venv, login node with internet): benchmark prompts in DFlash's formats (dflash.benchmark), chat
           template with enable_thinking True and False -> prompts.pt
  gen      (vllm venv, 1 GPU): greedy responses up to --max-new tokens -> one file per mode

Record format (as gen_traindata shards, plus metadata):
  {"ids": int32 prompt + response, "n_prompt": int, "mode": "think" | "nothink", "dataset": str, "idx": int,
   "capped": bool (hit --max-new; greedy thinking can loop), "think_end": index of </think> in ids or -1}

  python scripts/gen_thinking_vllm.py prompts --file /scratch-shared/$USER/g1/prompts.pt
  python scripts/gen_thinking_vllm.py gen --file /scratch-shared/$USER/g1/prompts.pt --out /scratch-shared/$USER/g1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_SETS = {"gsm8k": 200, "math500": 200, "humaneval": 164, "mbpp": 200, "mt-bench": 80}


def build_prompts(args):
    from transformers import AutoTokenizer

    from hspec.data import load_prompts

    tok = AutoTokenizer.from_pretrained(args.target)
    sets = dict(DEFAULT_SETS)
    if args.sets:
        sets = {s.split(":")[0]: int(s.split(":")[1]) for s in args.sets}
    recs = []
    for ds, n in sets.items():
        try:
            prompts = load_prompts(ds, n)
        except Exception as e:                                    # a benchmark name the loader does not know
            print(f"[prompts] skipping {ds}: {e}")
            continue
        for i, p in enumerate(prompts):
            for mode in ("think", "nothink"):
                text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=mode == "think")
                ids = tok(text, add_special_tokens=False)["input_ids"]
                recs.append({"dataset": ds, "idx": i, "mode": mode, "prompt_ids": ids})
        print(f"[prompts] {ds}: {len(prompts)} prompts x 2 modes")
    Path(args.file).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"target": args.target, "records": recs}, args.file)
    print(f"saved {len(recs)} prompt records to {args.file}")


def generate(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    blob = torch.load(args.file)
    recs = blob["records"]
    llm = LLM(model=blob["target"], max_model_len=max(len(r["prompt_ids"]) for r in recs) + args.max_new + 8,
              gpu_memory_utilization=0.90, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    end_think = tok.convert_tokens_to_ids("</think>")
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("think", "nothink"):
        sel = [r for r in recs if r["mode"] == mode]
        res = llm.generate([TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in sel], sp)
        rows = []
        for r, g in zip(sel, res):
            resp = list(g.outputs[0].token_ids)
            ids = r["prompt_ids"] + resp
            te = ids.index(end_think, len(r["prompt_ids"])) if end_think in resp else -1
            rows.append({"ids": torch.tensor(ids, dtype=torch.int32), "n_prompt": len(r["prompt_ids"]),
                         "mode": mode, "dataset": r["dataset"], "idx": r["idx"],
                         "capped": len(resp) >= args.max_new, "think_end": te})
        torch.save(rows, out / f"{mode}.pt")
        lens = [len(x["ids"]) - x["n_prompt"] for x in rows]
        print(f"[gen] {mode}: {len(rows)} responses, mean {sum(lens) / len(lens):.0f} tokens, "
              f"capped {sum(x['capped'] for x in rows) / len(rows):.1%}, "
              f"with </think> {sum(x['think_end'] >= 0 for x in rows) / len(rows):.1%}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("--target", default="Qwen/Qwen3-8B")
    a.add_argument("--sets", nargs="*", default=[], help="name:count overrides, e.g. gsm8k:200 math500:200")
    a.add_argument("--file", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--file", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--max-new", type=int, default=8192)
    args = ap.parse_args()
    build_prompts(args) if args.cmd == "prompts" else generate(args)


if __name__ == "__main__":
    main()
