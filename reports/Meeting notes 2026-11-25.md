# Meeting notes: what was done and measured (23 to 25 November 2026)

Setting for all numbers: target Qwen3-8B, DFlash block drafter (block 16), greedy, batch 1, one A100, offline on the
target's own text. tau = accepted drafted tokens + 1. Details and tables: `reports/Status 2026-11-24.md`
(sections 1 to 10) and `reports/Literature recheck 2026-11-25.md`.

## 1. The question

Tokens drafted in the current round (in-flight tokens) get no target-side information, so only one block is drafted
per verification. Working frame: draft a block, a cheap intermediate step gives a cut point and features for the
kept tokens, the drafter continues from the cut, repeat, one lossless verification at the end. Open item: what that
intermediate step is.

## 2. What was measured

**A. Real cost of the pieces (bench_band, CUDA graphs).**
Full verify 20.4 ms for 16 tokens, 21.7 ms for 48; drafter call 3.55 ms; one target layer 0.54 ms, linear in the
layer count; LM head 0.88 ms. Width is nearly free, depth is not. An extra stage pays only if it adds about one
token.

**B. What the drafter reads from the target (three families: Qwen3-8B, Llama-3.1-8B, Gemma-4-12B).**
- The last two taps carry 0.86 to 1.00 of the drafter's performance; the first taps almost nothing (exp111).
- Information about the coming tokens appears late in depth (Llama from about 50%, Qwen 70%, Gemma 75%), is best
  at or near the last layer, and reaches only the next two or three tokens; nothing is readable at distance 8
  (exp115, also with the next token given).
- With the interface refitted, one layer is enough and 3/4 depth is not special: a plateau from 72% to 94% depth,
  layer 33 alone gives 98% of all five taps for the last 8 positions; the last layer is worse (exp118).

**C. Cheap target computation over the in-flight tokens, frozen backbone.**
- Shallow verify for the cut decision: every shape and budget (first k, skip, sandwich, attention-only, 4 or 8
  layers) finds cut and token in 0.16 to 0.22 of the cases; backbone LoRA raises it to 0.21 (exp113, exp114).
- Band pass (2 shallow layers, bridge, layers 18 to 25): 0.87 of the value of the real tap, at 10 layers = 5.5 ms
  (exp116).

**D. The cut and the token at the cut.**
- With the cut placed at the first error: redraft right 0.54, with the known-wrong token excluded 0.69, true token
  in the top three 0.86 (exp117).
- A drafter trained for in-flight tokens (lag training) is the largest single lever: without any features the
  redraft goes from 0.31 to 0.57 right, and the two-block chain gets 70% of what real features give (was 30%).
- As a policy over all cut positions (exp123): the drafter's own confidence cuts at the first error in 47% of the
  chains. Two blocks, measured timings: oracle cut without features 1.19x, confidence cut 1.01x, oracle cut with
  real features (if free) 1.30x. Masking helps only when the cut is right.

**E. How deep the chain can go (exp124, measured timings).**
- Oracle cut, real features (if free): 1.65x at six stages; survival at each cut 0.64 rising to 0.84; a surviving
  chain adds 3.3 to 4.2 tokens per stage.
- Oracle cut, no features: 1.21x at three stages, 1.8 to 2.1 tokens per surviving stage.
- If the token chosen at the cut could also be checked, the chain never dies: 2.07x at eight stages and rising.
- Confidence cut: at most 1.08x.
- With a pass of 1.5 / 3.6 ms per stage the 1.65x becomes 1.42x / 1.23x: the budget for the intermediate step is
  about 2 ms per stage.

**F. Speculative features: the drafter trained to read approximate in-flight information (exp125).**
tau of one block when the last 4 / 8 / 16 positions have no real features:

| in-flight positions read | cost per stage | tau (lag 4 / 8 / 16) |
|---|---|---|
| tokens only, one training stage | 0 | 4.93 / 4.54 / 3.84 |
| learned code of the real feature, 16 numbers | (upper bound) | 5.01 / 4.70 / 4.03 |
| learned code, 64 numbers | (upper bound) | 5.12 / 5.05 / 4.45 |
| learned code, 256 numbers | (upper bound) | 5.38 / 5.44 / 4.89 |
| 3 target layers + bridge | 1.6 ms | 5.00 / 4.51 / 3.82 |
| 4 target layers + bridge, one stage | 2.2 ms | 4.98 / 4.86 / 4.17 |
| 10-layer band | 5.5 ms | 5.20 / 5.02 / 4.42 |
| real state of layer 29, one projection | | 5.45 / 5.63 / 5.15 |
| real features (five taps) | | 5.55 / 5.78 / 5.51 |

- The same 64 numbers are worse than nothing as a faithful slice of the target state (3.86 at lag 8) and close half
  of the gap when trained on the drafter's loss (5.05). The drafter does not need the target's state; it needs
  something much smaller, and fidelity is the wrong objective.
- One vector per token is enough (layer 29: 90% of five taps).
- One training stage from the original DFlash drafter is enough and better than lag pre-training plus a second
  stage (4.54 against 4.38).
- Producing that information from the frozen backbone follows a flat trade-off: a fifth of the gap at 2.2 ms,
  about 45% at 5.5 ms.

**G. Literature recheck (25 November).**
- Published: the chain itself (DLoop, with the drafter's own states for unverified tokens and no cut), the
  tokens-only drafter (DPara, which reports its "information deficit"; D3D), rejected-state reuse (DFlow,
  Carryover, ReTrace, ReAnchor), the fork observation and re-anchoring (DForkTree), partial-depth states of
  in-flight tokens for a one-token drafter on a GPU pipeline (Speculative Pipeline Decoding).
- Not found: target-derived computation over the current draft within a round for a block drafter on one GPU; the
  ceiling of real states at unverified positions (DLoop reports no such oracle); a cut inside the block for looped
  drafting; the objective result and the size of the needed information; the measured price curve.

## 3. Where this stands

Holds:
- the gap is real and large (5.78 against 4.54 at lag 8; 1.65x against 1.21x for the chain);
- the drafter side is solved in one training (dual mode: real features where they exist, tokens plus approximate
  information elsewhere);
- what the drafter needs per in-flight token is one small vector trained for the drafter.

Does not hold yet:
- a producer of that vector within about 2 ms per stage (best so far: a fifth of the gap);
- the cut: only the oracle so far.

Running or waiting (Snellius maintenance until 27 November): attention-only deep layers in the band; adapted
(LoRA) path layers trained with the drafter, including the plain early-exit shape (first 4 / 8 layers); what the
trained code contains and how the drafter uses it (exp126).

## 4. Points to discuss

1. Positioning against DLoop: it feeds the drafter its own states; we ask what the target knows beyond them, how
   small it is and what it costs. Is that a sufficient thesis question if the producer stays partial?
2. The objective finding (fidelity harmful, drafter loss useful, 64 to 256 numbers) as the core result: method or
   analysis?
3. Setup: everything so far is strictly sequential on one GPU. In overlapped setups (DPara-style) the
   partial-depth states of the block under verification are a free by-product, and layer 29 is enough for the
   drafter. Is concurrent execution acceptable as part of the setup?
4. The cut: learned detector, or let the verification choose among prepared continuations.
