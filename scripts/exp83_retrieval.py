"""Experiment 83: in-flight states retrieved from earlier occurrences of the same tokens (no training).

Chained round as in exp80-82 (block 1 at anchor s matches n tokens, oracle cut; block 2 at anchor s+n needs features
for the in-flight positions s .. s+n-1). The verified context holds the target's real tap states for every earlier
position. For an in-flight token x_p, use the real states at the LATEST earlier verified position t < s whose last
k tokens match (x_{t-k+1..t} = x_{p-k+1..p}); tokens before p inside the span are known (assumed accepted).
Cross-context state reuse is CacheBlend's observation (same tokens computed in another context are close); here it is
applied within one sequence to estimate verifier states for not-yet-verified tokens.

Variants: uni (k=1), bi (k=2), tri (k=3), backoff (tri -> bi -> uni), each with the dataset-mean features where no
match exists; plus mean-only. All 5 taps are taken from the matched position. DFlash is the original, frozen.
Reported: match rate over in-flight positions, centred cosine at L25 / L33 (all positions and matched only), block 2
first token right, tokens per verification (chained n + m2 + 1 vs normal n + 1) over all rounds.

  python scripts/exp83_retrieval.py --cache seqs.pt --stops 600
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dflash.model import extract_context_feature  # noqa: E402
from exp27_refresh import blocks_logits, matched  # noqa: E402

from hspec.models import load_draft, load_target  # noqa: E402
from hspec.utils import print_table, save_json  # noqa: E402

# (k, fwd): match the last k tokens; fwd also requires the NEXT token to match (x_{t+1} = x_{p+1}). For in-flight
# positions the next token is a drafted token of block 1 (the last one's is block 2's anchor): no target run needed.
VARIANTS = {"uni": [(1, False)], "bi": [(2, False)], "tri": [(3, False)],
            "backoff": [(3, False), (2, False), (1, False)],
            "uni+next": [(1, True)], "bi+next": [(2, True)], "tri+next": [(3, True)],
            "backoff+next": [(3, True), (2, True), (1, True), (3, False), (2, False), (1, False)],
            "mean": []}


def find_match(xl, occ, p, s, k, fwd=False):
    """latest t < s with xl[t-k+1 .. t] == xl[p-k+1 .. p] (and xl[t+1] == xl[p+1] if fwd)
    (xl: list of ints, occ: token -> positions ascending)"""
    if p - k + 1 < 0:
        return None
    for t in reversed(occ.get(xl[p], ())):
        if t >= s:
            continue
        if t - k + 1 < 0:
            return None
        if fwd and xl[t + 1] != xl[p + 1]:
            continue
        if all(xl[t - j] == xl[p - j] for j in range(1, k)):
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="z-lab/Qwen3-8B-DFlash-b16")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stops", type=int, default=600)
    ap.add_argument("--save", default="exp83_retrieval")
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()

    seqs = torch.load(args.cache)
    random.Random(0).shuffle(seqs)
    val = seqs[: max(int(len(seqs) * 0.2), 1)]
    target = load_target(args.target, dev)
    draft = load_draft(args.draft, dev)
    lids, bs = list(draft.target_layer_ids), draft.block_size
    H = target.config.hidden_size
    i25, i33 = lids.index(25), lids.index(33)

    @torch.no_grad()
    def feats_of(x):
        out = target(x[None], output_hidden_states=True, use_cache=False, logits_to_keep=1)
        return extract_context_feature(out.hidden_states, lids)[0]

    vr = random.Random(7)
    cand = [(i, s) for i, (ids, lo) in enumerate(val) for s in range(lo + 8, len(ids) - 3 * bs)]
    vr.shuffle(cand)
    by = {}
    for i, s in cand[: args.stops]:
        by.setdefault(i, []).append(s)

    acc, cnt = None, 0
    with torch.inference_mode():
        for i in list(by)[:20]:
            f = feats_of(val[i][0].to(dev))[val[i][1] :].double()
            acc = f.sum(0) if acc is None else acc + f.sum(0)
            cnt += f.shape[0]
    mean = (acc / cnt).float()

    recs = []
    stats = {v: {"pos": 0, "hit": 0, "c25": 0.0, "c33": 0.0, "h25": 0.0, "h33": 0.0} for v in VARIANTS}
    with torch.inference_mode():
        for done, (i, anchors) in enumerate(by.items()):
            ids, lo = val[i]
            x = ids.to(dev)
            xl = ids.tolist()
            occ = {}
            for t, tok in enumerate(xl):
                occ.setdefault(tok, []).append(t)
            f = feats_of(x)
            for s in anchors:
                y1 = blocks_logits(draft, target, f[None, :s], x, [(s, 0)], bs)[0].argmax(-1).tolist()
                n = matched(y1, xl[s + 1 :])
                r = {"n": n}
                if n == 0:
                    r.update({k: 0 for k in ["real", "tokens"] + list(VARIANTS)})
                    recs.append(r)
                    continue
                a = s + n
                truth = xl[a + 1 :]
                lg = blocks_logits(draft, target, f[None, :a], x, [(a, 0)], bs)[0]
                r["real"] = matched(lg.argmax(-1).tolist(), truth)
                lg = blocks_logits(draft, target, f[None, :s], x, [(a, n)], bs)[0]
                r["tokens"] = matched(lg.argmax(-1).tolist(), truth)
                real = f[s:a].float()
                for name, ks in VARIANTS.items():
                    est = mean[None].expand(n, -1).clone()
                    hit = torch.zeros(n, dtype=torch.bool, device=dev)
                    for j, p in enumerate(range(s, a)):
                        for k, fwd in ks:
                            t = find_match(xl, occ, p, s, k, fwd)
                            if t is not None:
                                est[j] = f[t].float()
                                hit[j] = True
                                break
                    fz = torch.cat([f[:s], est.to(f.dtype)], 0)[None]
                    lg = blocks_logits(draft, target, fz, x, [(a, 0)], bs)[0]
                    r[name] = matched(lg.argmax(-1).tolist(), truth)
                    st = stats[name]
                    st["pos"] += n
                    st["hit"] += int(hit.sum())
                    for tag, ti in (("25", i25), ("33", i33)):
                        sl = slice(ti * H, (ti + 1) * H)
                        c = F.cosine_similarity(est[:, sl] - mean[sl], real[:, sl] - mean[sl], dim=-1)
                        c = torch.nan_to_num(c)
                        st["c" + tag] += float(c.sum())
                        st["h" + tag] += float(c[hit].sum())
                recs.append(r)
            if done % 10 == 0:
                print(f"[run] {done}/{len(by)} sequences, {len(recs)} rounds, {time.time() - t0:.0f}s", flush=True)

    k = len(recs)
    pos_r = [r for r in recs if r["n"] > 0]
    normal = sum(r["n"] + 1 for r in recs) / k

    def tv(c):
        return sum(r["n"] + r[c] + 1 for r in recs) / k

    rows = [{"features for in-flight positions": "none (normal DFlash)", "match rate": "-", "centred cos L25": "-",
             "centred cos L33": "-", "cos L25 (matched)": "-", "cos L33 (matched)": "-", "first token right": "-",
             "tokens / verification": normal, "vs normal": 1.0}]
    for c in ("real", "tokens"):
        rows.append({"features for in-flight positions": c, "match rate": "-",
                     "centred cos L25": 1.0 if c == "real" else "-", "centred cos L33": 1.0 if c == "real" else "-",
                     "cos L25 (matched)": "-", "cos L33 (matched)": "-",
                     "first token right": sum(r[c] >= 1 for r in pos_r) / len(pos_r),
                     "tokens / verification": tv(c), "vs normal": tv(c) / normal})
    for name in VARIANTS:
        st = stats[name]
        hits = max(st["hit"], 1)
        rows.append({"features for in-flight positions": f"retrieval: {name}",
                     "match rate": st["hit"] / max(st["pos"], 1),
                     "centred cos L25": st["c25"] / max(st["pos"], 1), "centred cos L33": st["c33"] / max(st["pos"], 1),
                     "cos L25 (matched)": st["h25"] / hits, "cos L33 (matched)": st["h33"] / hits,
                     "first token right": sum(r[name] >= 1 for r in pos_r) / len(pos_r),
                     "tokens / verification": tv(name), "vs normal": tv(name) / normal})
    print_table(rows, list(rows[0]), f"in-flight states retrieved from earlier occurrences, {k} rounds, oracle cut, "
                                     f"frozen DFlash")
    save_json(args.save, {"args": vars(args), "rows": rows, "records": recs})
    print(f"[done] {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
