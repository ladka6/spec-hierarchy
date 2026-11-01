"""Experiment 91: infill from a later anchor (offline, T=0, frozen DFlash).

A block drafter conditions on any subset of its block positions. If the token at a later block position m is known,
positions before it get information from the future. Value of an anchor source at distance m: V(m) = p_m * G(m), with
  G(m)  oracle infill gain: accepted length with the TRUE token placed at block position m, minus plain DFlash
  p_m   how often the source's token at m is right
and with plain + infilled drafts both in the verification tree, a wrong anchor costs ~nothing.

Per anchor s (target greedy text = truth), plain DFlash block y, then infilled blocks (block input = [x_s, MASK..] with
position m set to a token v; the drafter is NOT trained for this, so a small effect may mean "untrained", see notes):
  oracle      v = true token at m, m in --ms                         -> G(m), and the tree max(plain, infill)
  wrong       v = the plain draft's own token at m when it is wrong  -> harm of a wrong anchor (infill alone)
  island      v = the plain draft's token at the farthest position m >= 3 with confidence >= --tau that comes after a
              position with confidence < 0.5                           -> p (island correct), tree gain
  retrieval   last earlier occurrence of the current 3-token suffix in the context; v = the token m steps after it
              (m = --ms)                                               -> p_m, tree gain; control: the looked-up chain itself
              as an extra tree branch (prompt-lookup style), to see whether infill adds anything beyond the lookup
Accepted = longest match with the truth + 1; "tree" rows take the best branch.

  python scripts/exp91_infill.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import _draft_value, _output_head, _raw_input_embeddings, extract_context_feature  # noqa: E402
from exp27_refresh import matched  # noqa: E402

from hspec.data import encode, load_prompts, stop_ids  # noqa: E402
from hspec.models import load_draft, load_target, load_tokenizer  # noqa: E402
from hspec.pipeline import two_stage_generate  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402


def infill_logits(draft, head_src, feats, x, s, injs, bs):
    """Logits [n, bs-1, V] for n blocks at anchor s; injs: list of {block index: token} (index 1..bs-1)."""
    n, T = len(injs), s
    toks = torch.full((n, bs), draft.mask_token_id, dtype=torch.long, device=x.device)
    toks[:, 0] = x[s]
    for b, inj in enumerate(injs):
        for m, v in inj.items():
            toks[b, m] = v
    scale = float(_draft_value(draft.config, "input_embedding_scale", 1.0))
    noise = _raw_input_embeddings(head_src, toks.reshape(1, -1), scale)
    pos = torch.cat([torch.arange(T, device=x.device)] + [torch.arange(s, s + bs, device=x.device)] * n)[None]
    mask = torch.zeros((n * bs, T + n * bs), dtype=torch.bool, device=x.device)
    for b in range(n):
        mask[b * bs : (b + 1) * bs, :T] = True
        mask[b * bs : (b + 1) * bs, T + b * bs : T + (b + 1) * bs] = True
    hid = draft(target_hidden=feats[:, :T], noise_embedding=noise, position_ids=pos, attention_mask=mask[None, None],
                past_key_values=None, use_cache=False)[0]
    h = torch.stack([hid[b * bs + 1 : (b + 1) * bs] for b in range(n)])
    return draft.compute_logits(h, _output_head(head_src))


def chain_with(logits_row, inj):
    y = logits_row.argmax(-1).tolist()
    for m, v in inj.items():
        y[m - 1] = v
    return y


def retrieve(ctx, n=3):
    """Index j of the last earlier occurrence of the final n tokens of ctx (end of the match), or None."""
    if len(ctx) <= n:
        return None
    key = ctx[-n:]
    for j in range(len(ctx) - 2, n - 2, -1):
        if ctx[j - n + 1 : j + 1] == key:
            return j
    return None


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math500", "humaneval", "mt-bench"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ms", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 15])
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--out", default="exp91_infill")
    args = ap.parse_args()

    tok = load_tokenizer(args.target)
    target = load_target(args.target)
    draft = load_draft(args.draft)
    stops = stop_ids(target, tok)
    bs, D = draft.block_size, draft.block_size - 1
    lids = list(draft.target_layer_ids)

    acc = defaultdict(list)          # key -> accepted lengths (aligned with plain on the same anchors)
    pr = defaultdict(list)           # source -> anchor correct?
    for dname in args.datasets:
        n_anc = 0
        for p in load_prompts(dname, args.n):
            ids = encode(tok, p)
            n0 = ids.shape[1]
            x = two_stage_generate(draft, target, target, target, ids, args.max_new, stops).output_ids[0]
            feats = extract_context_feature(target(x[None], output_hidden_states=True, logits_to_keep=1).hidden_states,
                                            lids)
            xl = x.tolist()
            for s in range(n0, len(x) - bs - 1, args.stride):
                n_anc += 1
                truth = xl[s + 1 : s + bs]
                injs, keys = [{}], [("plain", None)]
                for m in args.ms:
                    injs.append({m: truth[m - 1]})
                    keys.append(("oracle", m))
                base = infill_logits(draft, target, feats, x, s, [{}], bs)[0].float()
                prob = torch.softmax(base, -1)
                conf, y = prob.max(-1)
                y, conf = y.tolist(), conf.tolist()
                for m in args.ms:
                    if y[m - 1] != truth[m - 1]:
                        injs.append({m: y[m - 1]})
                        keys.append(("wrong", m))
                # confidence island: farthest confident position after an unsure one
                isl = None
                for m in range(D, 2, -1):
                    if conf[m - 1] >= args.tau and min(conf[: m - 1]) < 0.5:
                        isl = m
                        break
                if isl:
                    injs.append({isl: y[isl - 1]})
                    keys.append(("island", isl))
                # retrieval from the context
                j = retrieve(xl[: s + 1])
                look = None
                if j is not None:
                    look = xl[j + 1 : j + 1 + D]
                    for m in args.ms:
                        if j + m <= s:
                            injs.append({m: xl[j + m]})
                            keys.append(("retrieval", m))
                L = infill_logits(draft, target, feats, x, s, injs, bs).float()
                a_plain = matched(y, truth) + 1
                for (kind, m), inj, row in zip(keys, injs, L):
                    if kind == "plain":
                        acc[("plain", 0, "alone")].append(a_plain)
                        continue
                    ch = chain_with(row, inj)
                    a_inf = matched(ch, truth) + 1
                    acc[(kind, m, "alone")].append(a_inf)
                    acc[(kind, m, "plain")].append(a_plain)
                    acc[(kind, m, "tree")].append(max(a_plain, a_inf))
                    if kind in ("island", "retrieval"):
                        pr[(kind, m)].append(inj[m] == truth[m - 1])
                        if kind == "retrieval" and look:
                            a_look = matched(look, truth) + 1
                            acc[(kind, m, "tree+lookup")].append(max(a_plain, a_inf, a_look))
                            acc[(kind, m, "plain+lookup")].append(max(a_plain, a_look))
            print(f"[{dname}] {n_anc} anchors", flush=True)

    def mean(v):
        return sum(v) / len(v) if v else None

    rows = []
    for kind in ("oracle", "wrong", "island", "retrieval"):
        ms = sorted({m for (k, m, _) in acc if k == kind})
        for m in ms:
            row = {"source": kind, "m": m, "n": len(acc[(kind, m, "alone")]),
                   "p (anchor right)": mean(pr.get((kind, m), [])) if kind in ("island", "retrieval") else
                   (1.0 if kind == "oracle" else 0.0),
                   "plain": mean(acc[(kind, m, "plain")]), "infill alone": mean(acc[(kind, m, "alone")]),
                   "tree (plain+infill)": mean(acc[(kind, m, "tree")])}
            if kind == "retrieval":
                row["plain+lookup"] = mean(acc[(kind, m, "plain+lookup")])
                row["plain+lookup+infill"] = mean(acc[(kind, m, "tree+lookup")])
            rows.append(row)
    cols = sorted({c for r in rows for c in r}, key=lambda c: list(rows[-1]).index(c) if c in rows[-1] else 99)
    for r in rows:
        for c in cols:
            r.setdefault(c, None)
    print(f"plain DFlash accepted (all anchors): {mean(acc[('plain', 0, 'alone')]):.3f}")
    print_table(rows, cols, "infill from an anchor at block position m (accepted per verification; tree = best branch)")
    save_json(args.out, {"args": vars(args), "rows": rows})


if __name__ == "__main__":
    main()
