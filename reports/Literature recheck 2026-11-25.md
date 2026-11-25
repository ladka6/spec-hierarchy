# Literature recheck for the in-flight feature line (25 November 2026)

Purpose: the gap report of 1 November ranks "fresh target computation over the current draft" (Gap 1) first and
names unscanned recent work as its main risk. This is a recheck three weeks later, with the sharper question the
24 November experiments gave us. Sources: web search, the arXiv listing for "DFlash" (67 entries, newest first), and
the papers below. Depth of reading is marked per paper: F = main text read through a fetch tool's summary,
A = abstract only. Numbers are as the papers report them and were not checked against the PDFs.

## 1. Papers that touch our line directly

| paper | what it does | what unverified tokens get | relation to us | read |
|---|---|---|---|---|
| DLoop (2610.07659, Oct 2026) | one verification covers several drafting stages; a gate on the block's summed log-probability decides "draft another stage or verify"; no cut inside a block, each stage is anchored at the last drafted token; loop-aware training (N = 3 stages, one epoch) | the drafter's OWN final-layer hidden states, concatenated to the target states of the verified prefix; no target computation | this is our chain, with the drafter's states in place of target features and without a cut. It states our problem in its own words (later stages need target states of unverified tokens). DFlash on Qwen3-4B: tau 6.10 to 7.21, speedup 4.19x to 4.67x; DSpark on Qwen3-8B: 5.58 to 9.08, 3.81x to 4.82x; gains shrink under concurrency. It reports no oracle with real target states | F |
| DPara (2609.27396) | M-DFlash: DFlash fine-tuned to draft from several anchors that have no target features, one branch per acceptance boundary, run while the target verifies | tokens only | this is the lag (tokens-only) drafter. tau 4.43 against 5.36 for DSpark; the paper names the "information deficit" of feature-less anchors as its limitation. Needs a second GPU in the main variant | F |
| D3D (2611.06425, Nov 2026) | standalone diffusion drafter that never reads target states; pre-drafts successor blocks for every rejection position during verification | nothing from the target at all | the opposite bet: remove the dependence instead of feeding it. tau 7.88 against 5.96 for DFlash on Qwen3-8B, 5.00x; drafter and target on separate GPUs | F |
| Speculative Pipeline Decoding (2605.30852) | target split into n depth stages on n GPUs; a one-token drafter reads, for every token, the states of the layers it has reached so far | partial-depth REAL target states of in-flight tokens | the only work found that feeds partial-depth target states of unverified tokens to a drafter. Autoregressive one-token drafter, n + 1 GPUs, streaming verification; not a block drafter and not a single-GPU method | F |
| ReAnchor (2610.26531) | several shifted blocks anchored on tokens decoded from the previous round's rejected-suffix states, merged into a trie, one tree verify; training-free | stale states of the previous round | +15.2% tau, +13.4% speedup over DFlash on four targets | F |
| DFlow (2609.06498), Carryover (2609.14717), ReTrace (2608.29748) | previous round's rejected-suffix verifier states conditioned into the next block | stale states | +10 to 13% tau (already in the gap report) | F / F / A |
| DForkTree (2611.12356, Nov 2026) | finds that after a path leaves the drafter's greedy path its top-1 accuracy falls from 89% to 18% and that re-running the drafter from the fork token restores it; training-free tree that re-anchors branches, scores adjusted by target n-gram PMI | nothing new from the target | publishes the fork observation of our 2 November notes (a fork is one decision; a redraft from the right token is easy again) and builds the tree on it. +41.5% over DFlash, +15.1% over DDTree on Qwen3.5-4B / 9B | A |

## 2. Papers on what the drafter reads at VERIFIED positions

| paper | claim | read |
|---|---|---|
| DNeo (2611.21925, Nov 2026) | standard feature encoding is lossy ("residual mixing"), memory-less and uses uniformly spaced layers that omit important states; mixture-of-features attention, a memory of rejected proposals, non-uniform layer selection; 1.3x over DFlash-2 | A |
| DynaDFlash (2611.19085) | per-layer gates on the target states before projection; no significant change in mean acceptance, +5% on math | A |
| NarrowSpec (2611.23835) | the drafter's residual width can be half or a quarter of the target's with little loss | A |
| DFlare (2606.02091), H-Spec (2609.24197) | learnable per-layer fusion; in-place target KV reuse | in the earlier reports |
| Beyond Parallel Blindness (2608.27339) | separates the information a parallel block cannot have from the model gap; one realised token removes 86 to 100% of the information floor; current drafters sit far above their floor | A |

## 3. Crowded neighbours (not our line)

Causal correction inside the block (Corrective Prefix Attention 2611.31112, Wavefront 2611.31638, DRelay
2610.01439, xPress, DSpark, Adjacent Causal Injection 2609.36173), trees (Multi-Future Drafting 2611.13017,
TreeSpark, Pando 2611.32269), training objectives (BV Loss, Verification-Aware Training, SpecOPD). Titles and
abstracts only; none computes anything with the target over the current draft.

## 4. What this changes for us

Already published, so not claimable:
- the chain itself (several drafting stages per verification, a trained loop): DLoop;
- a drafter that continues from unverified tokens without features: DPara's M-DFlash, D3D;
- rejected-state reuse in every form;
- the fork observation and re-anchoring at the fork: DForkTree, D2SD;
- which layers / how the verified features are encoded: DNeo, DFlare, DynaDFlash.

Not found in any paper read:
- target-derived computation over the CURRENT draft inside a round on one GPU, for a block drafter (DLoop uses the
  drafter's states, SPD needs a pipeline of GPUs, the reuse papers are one round stale);
- the ceiling of that computation: no paper reports what real target states at the unverified positions would give.
  We have it (13.81 against 8.50 tokens per verification over two blocks; 1.65x at six stages with the measured
  timings; tau 5.78 against 4.38 at lag 8 with a jointly trained drafter);
- a cut inside the block for looped drafting (DLoop continues from the end of the block; our oracle-cut numbers say
  the cut is where most of the chain's value is);
- the objective result of exp125: a code of 64 numbers per unverified token is harmful as a faithful slice of the
  target state and closes half of the gap when trained on the drafter's loss; one vector per token is enough;
- the measured price curve of producing that information from the backbone itself (exp116, exp125 path arms).

Consequences:
- DLoop is the baseline to beat and to cite first. The honest framing of our line is "DLoop feeds the drafter its
  own states; what does the target know that the drafter's states do not, how small is it, and what does it cost".
- The lag drafter and the chain are setup, not contribution.
- The fork-resolving block tree line is now largely covered by DForkTree.

## 5. Not checked

OpenReview ICLR 2027 submissions could not be searched directly (the search returned only ICLR 2026 pages).
Not read beyond the title: ChainFlow (2611.08669), CoPyro (2611.27217), Multi-Future Drafting, SpecOPD, FlashMap,
DStill, HyFlash (2611.07263), "Next Token or Vector?" (2610.22788). DynaDFlash's PDF was rate-limited; its abstract
was read in the browser. Summaries came through a fetch tool, so every number above needs a check against the PDF
before it is cited.

## Sources

- DLoop: https://arxiv.org/pdf/2610.07659
- DPara: https://arxiv.org/pdf/2609.27396
- D3D: https://arxiv.org/html/2611.06425v1
- Speculative Pipeline Decoding: https://arxiv.org/html/2605.30852
- ReAnchor: https://arxiv.org/pdf/2610.26531
- DFlow: https://arxiv.org/pdf/2609.06498
- Carryover: https://arxiv.org/pdf/2609.14717
- ReTrace: https://arxiv.org/pdf/2608.29748
- DForkTree: https://arxiv.org/abs/2611.12356
- DNeo: https://arxiv.org/abs/2611.21925
- DynaDFlash: https://arxiv.org/abs/2611.19085
- NarrowSpec: https://arxiv.org/abs/2611.23835
- Beyond Parallel Blindness: https://arxiv.org/abs/2608.27339
- Corrective Prefix Attention: https://arxiv.org/abs/2611.31112
- Wavefront: https://arxiv.org/abs/2611.31638
- DRelay: https://arxiv.org/abs/2610.01439
- arXiv listing: https://arxiv.org/search/?query=DFlash&searchtype=all&abstracts=hide&order=-announced_date_first&size=100
