"""Experiment 102: repair instead of restart. Can a bidirectional diffusion LM fill DFlash's error windows from both
sides, and can it locate them? (offline, T=0, no training)

exp96: when DFlash fails, the rest of its block is still partly right, and oracle repair of a short window at each
error (rest of the draft kept) reaches 11.5 accepted (1 repair, width 4) and 15.6 (3 repairs) vs 7.97 plain. A causal
corrector (target, copy) only sees the left context, so it has to restart. A masked diffusion LM sees both sides.

Repairer: Dream-v0-Instruct-7B (Qwen2.5-7B weights trained as a masked diffusion LM; full attention; shifted logits:
the token at position p is predicted from the logits at p - 1). Loaded as a plain Qwen2ForCausalLM with a full
(non-causal) attention mask, so Dream's remote code (which breaks under transformers 5: KeyError 'default') is not
used. The script checks that attention is really bidirectional before running. Dream shares the Qwen2.5 / Qwen3 BPE
vocabulary for ordinary tokens; its mask token is 151666.

Per block (target greedy text t = truth, plain DFlash draft y, real target features), oracle error positions:
  REPAIR    for width w and R successive repairs (each at the next true error e, window e .. e+w-1, rest of the draft
            kept): Dream fills the window, then accepted = longest match + 1 (max 16); also the tree max(plain, repaired)
            ctx = both   Dream sees the draft to the right of the window      (the idea)
            ctx = left   same, but positions right of the window are masked (control: what right context buys)
            fill = iter  w forwards, commit the most confident masked position each time;  par: one forward
            also: accuracy of the filled fork token itself (copy reference: 0.88 top-1, DFlash top-2: 0.56)
  LOCATE    at every block position i with a correct prefix (i <= first error), mask position i alone and score the
            draft token by Dream's probability given both sides (pseudo-likelihood); AUROC for "DFlash wrong here"
            vs DFlash's own confidence (exp98: 0.925), the product of both, and on the subset where DFlash was
            confident (>= 0.5; 42% of its errors, exp94)
Reference rows: plain DFlash, and exp96's oracle window repair (true tokens in the window) on the same blocks.

  python scripts/exp102_repair.py                       # n = 10 per dataset
  python scripts/exp102_repair.py --n 20 --fills iter par
Pre-download once on a login node (jobs run offline):  huggingface-cli download Dream-org/Dream-v0-Instruct-7B
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import free, load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import mean, print_table, save_json  # noqa: E402


def auroc(score, label):
    """P(score of a positive > score of a negative)."""
    score = torch.as_tensor(score, dtype=torch.float64)
    label = torch.as_tensor(label, dtype=torch.bool)
    order = torch.argsort(score)
    ranks = torch.empty(len(score), dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float64)
    npos, nneg = int(label.sum()), int((~label).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load_dream(model_id: str, device: str = "cuda"):
    """Dream weights in a stock Qwen2ForCausalLM (identical parameter names; q/k/v biases), attention made non-causal."""
    from huggingface_hub import snapshot_download
    from transformers import Qwen2Config, Qwen2ForCausalLM

    path = snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])
    c = json.loads((Path(path) / "config.json").read_text())
    cfg = Qwen2Config(
        vocab_size=c["vocab_size"], hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"],
        num_hidden_layers=c["num_hidden_layers"], num_attention_heads=c["num_attention_heads"],
        num_key_value_heads=c["num_key_value_heads"], hidden_act=c["hidden_act"],
        max_position_embeddings=c["max_position_embeddings"], rms_norm_eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"], rope_parameters={"rope_type": "default", "rope_theta": c["rope_theta"]},
        tie_word_embeddings=False, use_sliding_window=False, sliding_window=None,
    )
    model, info = Qwen2ForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                   device_map=device, output_loading_info=True)
    bad = [k for k in info.get("missing_keys", []) if "rotary" not in k]
    if bad:
        raise RuntimeError(f"Dream weights did not load into Qwen2: missing {bad[:5]} ...")
    for layer in model.model.layers:
        layer.self_attn.is_causal = False
    model.eval()
    return model, int(c["mask_token_id"])


@torch.inference_mode()
def dream_logits(model, seqs: list[list[int]], pos: list[int] | None = None) -> torch.Tensor:
    """Full-attention forward over equal-length id lists. Returns logits [B, len(pos), V] at sequence indices `pos`
    (unshifted; the LM head is applied only there, full-vocabulary logits for every position would not fit)."""
    ids = torch.tensor(seqs, device="cuda")
    B, L = ids.shape
    full = torch.ones(B, 1, L, L, dtype=torch.bool, device="cuda")
    h = model.model(input_ids=ids, attention_mask=full, use_cache=False).last_hidden_state
    pos = list(range(L)) if pos is None else pos
    return model.lm_head(h[:, pos])


@torch.inference_mode()
def check_bidirectional(model):
    a = list(range(1000, 1032))
    b = a[:-1] + [2000]
    la, lb = dream_logits(model, [a, b], [0])
    diff = (la[0, 0].float() - lb[0, 0].float()).abs().max().item()
    if diff < 1e-3:
        raise RuntimeError("attention is still causal (changing the last token did not change position 0); "
                           "the 4D full mask was ignored by this transformers version")
    print(f"[dream] bidirectional check ok (max logit change at position 0: {diff:.3f})", flush=True)


@torch.inference_mode()
def fill(model, mask_id: int, ctx: list[int], block: list[int], window: list[int], hide_right: bool, mode: str):
    """Fill block positions in `window` (block-relative). Block position k sits at sequence index len(ctx) + k and is
    predicted from the logits at len(ctx) + k - 1. Returns the filled block (tokens right of the window unchanged)."""
    blk = list(block)
    todo = set(window)
    for k in todo:
        blk[k] = mask_id
    right_lo = max(window) + 1
    while todo:
        view = list(blk)
        if hide_right:
            view[right_lo:] = [mask_id] * (len(view) - right_lo)
        order = sorted(todo)
        lg = dream_logits(model, [ctx + view], [len(ctx) + k - 1 for k in order])[0]
        p = torch.softmax(lg.float(), -1)
        conf, tok = p.max(-1)
        if mode == "par":
            for k, v in zip(order, tok.tolist()):
                blk[k] = v
            todo.clear()
        else:
            j = int(conf.argmax())
            blk[order[j]] = int(tok[j])
            todo.discard(order[j])
    if hide_right:
        blk[right_lo:] = block[right_lo:]
    return blk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--repairer", default="Dream-org/Dream-v0-Instruct-7B")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--widths", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--repairs", type=int, default=2, help="successive oracle-located repairs per block")
    ap.add_argument("--fills", nargs="+", default=["iter"], choices=["iter", "par"])
    ap.add_argument("--ctx", nargs="+", default=["both", "left"], choices=["both", "left"])
    ap.add_argument("--no-locate", action="store_true")
    ap.add_argument("--out", default="exp102_repair")
    args = ap.parse_args()

    # ---------------- phase 1: target text, DFlash drafts ----------------
    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)
    blocks = []                     # dicts: ctx, y, t, e, conf (DFlash prob of its own token per position)
    with torch.inference_mode():
        for dname in args.datasets:
            for p in load_prompts(dname, args.n):
                ids = encode(tok, p)
                x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
                feats = extract_context_feature(
                    target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states, lids)
                xl = x.tolist()
                for s in range(ids.shape[1], len(x) - bs - 1, args.stride):
                    lg = blocks_logits(draft, target, feats, x, [(s, 0)], bs)[0].float()
                    pr = torch.softmax(lg, -1)
                    y = pr.argmax(-1).tolist()
                    conf = pr.max(-1).values.tolist()
                    t = xl[s + 1 : s + 1 + D]
                    blocks.append({"ctx": xl[: s + 1], "y": y, "t": t, "e": matched(y, t), "conf": conf, "ds": dname})
            print(f"[{dname}] {len(blocks)} blocks", flush=True)
    free(target, draft)
    del target, draft
    torch.cuda.empty_cache()
    plain = [b["e"] + 1 for b in blocks]
    err_blocks = [b for b in blocks if b["e"] < D]
    print(f"plain DFlash accepted {mean(plain):.3f} over {len(blocks)} blocks; {len(err_blocks)} have an error",
          flush=True)

    # ---------------- phase 2: Dream ----------------
    model, mask_id = load_dream(args.repairer)
    check_bidirectional(model)

    def oracle(y, t, w, R):
        c = list(y)
        for _ in range(R):
            e = matched(c, t)
            if e >= len(c):
                break
            c[e : e + w] = t[e : e + w]
            c = c[: len(y)]
        return matched(c, t) + 1

    rows, fork_rows = [], []
    for w in args.widths:
        for R in range(1, args.repairs + 1):
            rows.append({"repairer": "oracle (true tokens)", "w": w, "repairs": R,
                         "accepted (error blocks)": mean(oracle(b["y"], b["t"], w, R) for b in err_blocks),
                         "tree with plain": mean(max(b["e"] + 1, oracle(b["y"], b["t"], w, R)) for b in err_blocks)})
        for ctx in args.ctx:
            for mode in args.fills:
                acc = defaultdict(list)
                tree = defaultdict(list)
                fork_right, win_right, win_n = 0, 0, 0
                for i, b in enumerate(err_blocks):
                    c = list(b["y"])
                    for r in range(1, args.repairs + 1):
                        e = matched(c, b["t"])
                        if e < D:
                            win = list(range(e, min(e + w, D)))
                            c = fill(model, mask_id, b["ctx"], c, win, hide_right=(ctx == "left"), mode=mode)
                            if r == 1:
                                fork_right += c[e] == b["t"][e]
                                win_right += sum(c[k] == b["t"][k] for k in win)
                                win_n += len(win)
                        a = matched(c, b["t"]) + 1
                        acc[r].append(a)
                        tree[r].append(max(b["e"] + 1, a))
                    if i % 500 == 0:
                        print(f"  [w={w} ctx={ctx} {mode}] {i}/{len(err_blocks)}", flush=True)
                for r in range(1, args.repairs + 1):
                    rows.append({"repairer": f"Dream, ctx={ctx}, {mode}", "w": w, "repairs": r,
                                 "accepted (error blocks)": mean(acc[r]), "tree with plain": mean(tree[r])})
                fork_rows.append({"repairer": f"Dream, ctx={ctx}, {mode}", "w": w,
                                  "fork token right": fork_right / len(err_blocks),
                                  "window tokens right": win_right / max(win_n, 1)})
                print(f"[repair w={w} ctx={ctx} {mode}] 1 repair {mean(acc[1]):.3f}, fork right "
                      f"{fork_right / len(err_blocks):.3f}", flush=True)
    base_err = mean(b["e"] + 1 for b in err_blocks)
    print(f"\nplain DFlash on the error blocks: {base_err:.3f} (all blocks: {mean(plain):.3f})")
    print_table(rows, ["repairer", "w", "repairs", "accepted (error blocks)", "tree with plain"],
                "1. repair at the true error position(s), rest of the draft kept")
    print_table(fork_rows, ["repairer", "w", "fork token right", "window tokens right"],
                "2. is the filled token right? (copy at forks: 0.88 top-1; DFlash 2nd choice: 0.56)")

    # ---------------- locate ----------------
    loc = {}
    if not args.no_locate:
        sc_dream, sc_df, lab = [], [], []
        for i, b in enumerate(blocks):
            pos = list(range(min(b["e"], D - 1) + 1))
            seqs = []
            for k in pos:
                v = list(b["y"])
                v[k] = mask_id
                seqs.append(b["ctx"] + v)
            L0 = len(b["ctx"])
            # sequence j has the mask at block position pos[j]; read every sequence at all positions, use its own
            lg = dream_logits(model, seqs, [L0 + k - 1 for k in pos])
            for j, k in enumerate(pos):
                p = torch.softmax(lg[j, j].float(), -1)[b["y"][k]].item()
                sc_dream.append(p)
                sc_df.append(b["conf"][k])
                lab.append(b["y"][k] != b["t"][k])
            if i % 500 == 0:
                print(f"  [locate] {i}/{len(blocks)}", flush=True)
        sd, sf = torch.tensor(sc_dream), torch.tensor(sc_df)
        lb = torch.tensor(lab)
        confident = sf >= 0.5
        loc = {
            "positions": len(lab), "DFlash error rate": float(lb.float().mean()),
            "AUROC DFlash conf": auroc(-sf, lb), "AUROC Dream pseudo-likelihood": auroc(-sd, lb),
            "AUROC product": auroc(-(sf.clamp_min(1e-9).log() + sd.clamp_min(1e-9).log()), lb),
            "confident errors (share of errors)": float((lb & confident).sum() / lb.sum()),
            "AUROC Dream, DFlash-confident positions": auroc(-sd[confident], lb[confident]),
            "AUROC DFlash, DFlash-confident positions": auroc(-sf[confident], lb[confident]),
            "Dream flags confident errors (p < 0.5)": float(((sd < 0.5) & lb & confident).sum() / (lb & confident).sum()),
            "Dream false alarms on confident right (p < 0.5)": float(((sd < 0.5) & ~lb & confident).sum()
                                                                     / (~lb & confident).sum()),
        }
        print("\n== 3. locating DFlash's errors (positions with a correct prefix) ==")
        for k, v in loc.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    save_json(args.out, {"args": vars(args), "plain all": mean(plain), "plain error blocks": base_err,
                         "blocks": len(blocks), "error blocks": len(err_blocks), "repair": rows,
                         "fork": fork_rows, "locate": loc})


if __name__ == "__main__":
    main()
