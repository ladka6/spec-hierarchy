"""Experiment 12: early-exit quality per layer, target only (no drafter needed).

For each model, greedy trajectories on the evaluation prompts, then teacher-forced: the
prediction of the model's own final norm + LM head applied at layer k (LayerSkip models are
trained for exactly this head) against the model's final prediction. Reports per layer:

  eps      disagreement with the final layer
  conf90   share of positions where the exit's top-1 probability >= 0.9
  dis@90   disagreement rate on those positions
  run8     P(8 consecutive positions all agree): a rough proxy for passing a block

A usable middle verifier needs eps of a few percent at k around half depth (bnb4 Qwen3-8B:
eps 0.031, conf90 0.78, dis@90 0.001).

  python scripts/exp12_exit_scan.py --models facebook/layerskip-llama3-8B Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hspec.data import load_prompts  # noqa: E402
from hspec.models import free, load_target, load_tokenizer  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def encode_any(tok, prompt, device="cuda"):
    """Chat template when the tokenizer has one, else a plain completion prompt (base models)."""
    if tok.chat_template:
        kw = {"enable_thinking": False} if "enable_thinking" in (tok.chat_template or "") else {}
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, **kw)
        return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    return tok(f"Question: {prompt}\nAnswer:", return_tensors="pt")["input_ids"].to(device)


@torch.inference_mode()
def scan(model_id, datasets, n, max_new, every):
    tok = load_tokenizer(model_id)
    model = load_target(model_id)
    L = model.config.num_hidden_layers
    layers = sorted(set(list(range(every, L, every)) + [L // 2]))
    eos = model.generation_config.eos_token_id
    stats = {k: {"dis": [], "conf": [], "dis90": [], "run8": []} for k in layers}
    for d in datasets:
        for p in load_prompts(d, n):
            ids = encode_any(tok, p)
            n0 = ids.shape[1]
            seq = model.generate(ids, max_new_tokens=max_new, do_sample=False, eos_token_id=eos,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            if seq.shape[1] - n0 < 9:
                continue
            out = model(seq, output_hidden_states=True)
            final = out.logits[0, n0 - 1 : -1].argmax(-1)
            for k in layers:
                h = model.model.norm(out.hidden_states[k][0, n0 - 1 : -1])
                lg = model.lm_head(h).float()
                pr = torch.softmax(lg, -1)
                p1, pred = pr.max(-1)
                dis = (pred != final).float()
                c = p1 >= 0.9
                s = stats[k]
                s["dis"] += dis.tolist()
                s["conf"] += c.float().tolist()
                s["dis90"] += dis[c].tolist()
                ok = 1 - dis
                if len(ok) >= 8:
                    s["run8"] += ok.unfold(0, 8, 1).prod(-1).tolist()
                del lg, pr
            del out
        print(f"[{model_id} | {d}] done", flush=True)
    rows = [{"model": model_id.split("/")[-1], "layer": k, "of": L, "depth": k / L,
             "eps": mean(s["dis"]), "conf90": mean(s["conf"]),
             "dis@90": mean(s["dis90"]) if s["dis90"] else float("nan"), "run8": mean(s["run8"]),
             "positions": len(s["dis"])} for k, s in stats.items()]
    del model
    free()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--every", type=int, default=2, help="scan every k-th layer")
    args = ap.parse_args()
    rows = []
    for m in args.models:
        rows += scan(m, args.datasets, args.n, args.max_new, args.every)
        print_table(rows, ["model", "layer", "of", "depth", "eps", "conf90", "dis@90", "run8", "positions"],
                    "early-exit agreement with the final layer (teacher forced, own trajectories)")
        save_json("exp12_exit_scan", {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
