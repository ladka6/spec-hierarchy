"""Side by side: the target's L25 / L33 outputs, K / V and next-token outputs for the LAST token of several texts.

Use it to see how much a token's representation depends on its prefix (same last token, different prefixes) and how
different tokens look under the same prefix. For every text the script reports
  - the last token and the target's top-5 next tokens (the output)
  - pairwise cosine between texts for: L25 output, L33 output (DFlash's deep taps), and K / V of layers 25 and 33
    (pre-RoPE, so different positions are comparable; V has no RoPE anyway)
Cosines are printed raw and centred (the mean over every token of every text removed; raw cosines of these states
are dominated by a few huge constant dimensions).

  python scripts/inspect_states.py
  python scripts/inspect_states.py --texts "The capital of France is Paris" "I spent the summer in Paris" "Paris"
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_TEXTS = [
    "The capital of France is Paris",                                 # same last token, different prefixes
    "Last summer I spent two weeks walking around Paris",
    "def visit(city):\n    return city == 'Paris",
    "Paris",
    "The capital of France is London",                                # different last token, same prefix
    "The capital of France is Berlin",
    "The capital of France is",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--texts", nargs="+", default=DEFAULT_TEXTS)
    ap.add_argument("--layers", nargs="+", type=int, default=[25, 33])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    cfg = model.config
    n_kv, hd = cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", hd)

    reps = {}                                   # name -> [n_texts, dim] for the last token
    alltok = {}                                 # name -> [all tokens of all texts, dim] for centring
    labels = []
    with torch.inference_mode():
        for t in args.texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
            out = model(ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states                     # hs[i] = input of layer i = output of layer i-1
            probs = torch.softmax(out.logits[0, -1].float(), -1)
            top = probs.topk(5)
            last_tok = tok.decode(ids[0, -1:])
            preds = ", ".join(f"{tok.decode([int(i)])!r} {p:.2f}" for p, i in zip(top.values, top.indices))
            labels.append(f"{t[-28:]!r}")
            print(f"\n[{len(labels) - 1}] text {t!r}\n    last token {last_tok!r}   next-token top-5: {preds}")
            for L in args.layers:
                layer = model.model.layers[L]
                h_out = hs[L + 1][0].float()                               # output of layer L, all positions
                x = layer.input_layernorm(hs[L][0])                        # input of layer L, normalised
                k = layer.self_attn.k_norm(layer.self_attn.k_proj(x).view(-1, n_kv, hd)).reshape(x.shape[0], -1)
                v = layer.self_attn.v_proj(x)
                for name, val in ((f"L{L} output", h_out), (f"K{L} (pre-RoPE)", k.float()), (f"V{L}", v.float())):
                    reps.setdefault(name, []).append(val[-1])
                    alltok.setdefault(name, []).append(val)

    n = len(labels)
    print("\nlegend: " + "  ".join(f"[{i}] {l}" for i, l in enumerate(labels)))
    for name in reps:
        R = torch.stack(reps[name])
        mu = torch.cat(alltok[name]).mean(0)
        for kind, M in (("raw", R), ("centred", R - mu)):
            C = F.cosine_similarity(M[:, None], M[None], dim=-1)
            print(f"\n{name}, cosine ({kind})")
            print("      " + "".join(f"[{j}]".rjust(7) for j in range(n)))
            for i in range(n):
                print(f"[{i}]   " + "".join(f"{float(C[i, j]):7.2f}" for j in range(n)))


if __name__ == "__main__":
    main()
