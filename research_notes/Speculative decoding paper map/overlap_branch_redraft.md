# Overlap check: branch-point redrafting for block-diffusion drafters

Checked 2026-10-04. Sources read via arxiv HTML/PDF, blog, and search snippets. Paper details below come from fetched text summaries; verify exact numbers against the PDFs before citing.

## The idea under test

- DFlash drafts a 16-token block of per-position marginals in one pass, conditioned on target hidden features.
- Hypothesis H1: acceptance lost to in-block independence concentrates right after branch points (high-entropy positions with two plausible continuations), because later marginals become mixtures of incompatible futures (NAT multimodality).
- Method M: find the uncertain position j, redraft j+1..end once per top-m candidate at j, batched into one extra drafter pass, using a "lag" drafter that can condition on unverified tokens without target features for them. Build the tree from these branch-conditioned marginals.
- Diagnostic D: split acceptance failures into "target token not in drafter top-k" vs "in top-k but missed by tree / wrong parent".

## Paper-by-paper

Columns: (1) second drafting pass or re-conditioning on candidate tokens, (2) uncertainty-triggered or everywhere, (3) how conditioning is done, (4) analysis of where acceptance is lost, (5) closeness 0-3.

### D2SD, Dual Diffusion Draft Models (arXiv 2606.04446)
1. Yes, a real second drafting pass. A Variable-Prefix drafter (VP-Drafter, same DFlash-style architecture) re-anchors at a chosen prefix length i: anchor + first i DFlash tokens are visible, positions i+1..gamma-1 are re-masked and redrafted. K prefixes (default K=4, gamma=16) are stacked into one batched forward. K+1 chains (original + K redrafts) are verified jointly with cascade attention; longest accepted prefix wins.
2. Confidence-triggered. c_k = max_v p_k(v); rejection-boundary posterior r(i) = prod_{k<=i} c_k * (1 - c_{i+1}); top-K i by r(i) are chosen. So branches start where the drafter expects the first rejection, not uniformly.
3. Target hidden features of the verified context via KV injection (same as DFlash) plus the unverified prefix tokens 1..i as token inputs. No target features for the unverified prefix. Trained with prefix length l ~ truncated geometric, ground-truth prefix, exp-decayed CE on masked positions. This is functionally the "lag drafter".
4. Partial. Shows block-size scaling wall (gamma >= 24 gives no gain) and "error homogeneity": resampling K times from the same marginals gives +0.37 tau vs +2.25 for D2SD (GSM8K, Qwen3-8B: tau 6.96 -> 7.33 naive, 9.21 D2SD). No per-position or top-k-miss vs path-miss decomposition, no link to entropy of the target.
5. **3/3.** Same core move (confidence-located boundary, batched one-pass redraft of the suffix conditioned on unverified prefix tokens by a prefix-aware diffusion drafter). Differences: it does NOT fork over top-m alternatives at the uncertain position (the prefix is kept as the DFlash argmax and the boundary position itself is redrafted, not substituted), outputs chains not a tree, and gives no branch-point/multimodality analysis.

### xPress (arXiv 2608.02438; github.com/Supercomputing-System-AI-Lab/xPress)
1. No new draft pass. K parallel Jacobi iterations (4-7) of a low-rank (r=256) refiner add a logit bias conditioned on previously sampled tokens; re-conditions the whole block on its own sampled tokens.
2. Everywhere, fixed K.
3. Drafter hidden states (per position + block summary) + sampled token ids, strictly lower-triangular mixing, target LM head reads base logits.
4. No. Aggregate tau +30% avg (up to +56%), throughput ~1.3x. Repo has per-position acceptance bench code but no reported failure decomposition.
5. **1/3.** Re-conditions on sampled tokens but over a single chain, everywhere, with a small head, no branching.

### DBLAST, Dependent Block Drafting (arXiv 2608.05448)
1. No second pass. Low-rank latent mixture: K latent categories each give a hidden offset h_{i,z} = h_i + g_z(h_i); sample z then all positions in parallel.
2. Everywhere (block-level latent).
3. Category hidden expander + category-prior head on the DFlash block anchor; target LM head reused.
4. Partly relevant: uses "target-block early determinism" as a diagnostic, shows independent proposals degrade as determinism drops, largest gains in least deterministic regions (>12% macro gain on high-entropy Qwen3-8B). Not per-position, no top-k decomposition.
5. **2/3.** Directly targets the multimodality/mixture problem and shows it matters most in high-entropy regimes, but resolves it with a block-level latent mixture, not position-local branching at a detected branch point.

### TreeFlash (arXiv 2606.03819)
1. Two-stage, but no backbone redraft. Stage 1: top-M-ary tree from DFlash marginals. Stage 2: an AR-approximator (SwiGLU on h_{t+i} and embedding of preceding token) re-scores every node, then the final tree is built. M*gamma parallel evals.
2. Everywhere.
3. Drafter/verifier hidden states + previous token embedding (first-order only).
4. Distributional: TVD to target at depth 15 is 0.81 (DFlash) vs 0.62 (TreeFlash); at depth >= 10 TreeFlash top-1 coverage ~ DFlash top-5 coverage. No position-after-entropy or top-k vs tree-miss split.
5. **2/3.** Builds branch-conditioned tree marginals from candidate tokens, but with a one-token-lookback head applied everywhere, not a full redraft from a detected branch point.

### DARTree (arXiv 2608.13524)
1. No backbone redraft. Pretrained causal correction head (RNN-style state) propagated per branch while building a supertree depth-wise.
2. Everywhere (fixed nodes per depth).
3. DFlash logits + recurrent state over realized branch tokens.
4. No. tau 12.97 per round (+98.6% vs DFlash), 9.73x at T=0.
5. **2/3.** Full-path branch-conditioned marginals, everywhere, via a light head instead of a second drafter pass.

### GRAFT (arXiv 2608.20375)
1. No. Post-hoc tree assembly from frozen one-pass DFlash marginals.
2. Budget (SABA, 32-512) varies per round from confidence/degeneration/history; not per position.
3. Bi-tower scorer on (parent token, child token, depth) using frozen target token embeddings. No hidden states.
4. Names the right failure mode: "parent-child mismatch" (target-compatible token present in the tree but under the wrong parent), Fig 1(b) shows it across tasks; TDES reduces mismatch 54.71%. No top-k vs tree split, no entropy link.
5. **1/3.** Same diagnosis family (token present, wrong parent), fix is a cheap pairwise rescoring.

### DFlash 2 (inco.ai/blog/dflash2)
1. No. Path selection over top-16 candidates per position with an adjacent-pair bilinear score S_t(a,b) = U_t(b) + <A(a) * H(h_t), B(b)>.
2. Everywhere.
3. Candidate embeddings gated by drafter hidden state; no extra backbone pass.
4. Yes, partly: recall@1 drops 85.4% (pos 0) -> 72.9% (pos 6) ("suffix decay"); target in top-16 99.5% of the time. This is the closest thing to a "top-k miss vs selection miss" statement, at the per-position level, but not conditioned on entropy.
5. **1/3.** Shows headroom is in selection/coherence, not coverage; fix is pairwise, everywhere.

### Domino (arXiv 2605.29707)
1. No redraft. GRU over embeddings of preceding draft tokens + backbone H_i gives correction logits.
2. Everywhere.
3. Backbone hidden states + token embeddings (GRU).
4. No per-position failure analysis.
5. **1/3.** Single-chain causal correction head.

### DSpark (arXiv 2607.05147)
1. No. Parallel backbone + lightweight sequential Markov (or RNN) head adding a transition bias.
2. Everywhere; a separate confidence head schedules verification (not drafting).
3. Backbone hidden states + previous token (low-rank r=256).
4. Yes: Sec 4.3.1 "position-wise conditional acceptance". DFlash starts high (0.88 Math) then decays fast; AR stays flat; DSpark gets both. No entropy conditioning, no top-k split.
5. **1/3.** Useful per-position baseline curve; mechanism is everywhere and chain-level.

## Additional close work found by search

| Paper | What it does | Closeness |
|---|---|---|
| DominoTree (2607.08642) | DFlash backbone once, Domino GRU head re-run per tree node with path-specific state; best-first heap over cumulative log-prob, fixed budget 16. tau 8.09, 5.71x (Qwen3-8B). Notes marginal trees (DDTree) lose on code. | 2: branch-conditioned marginals everywhere via light head |
| PCTree, From Chains to Trees: Parent-Conditioned Drafting (2608.02123) | DSpark backbone once, Markov head applied per candidate parent: z_d(p) = L_d + Markov(p). Names "individually likely tokens combined into locally incoherent path". GSM8K Qwen3-4B B=16: tau 9.41 -> 11.16. | 2: parent-only conditioning everywhere |
| TreeSpark (2609.22098) | DSpark Markov-head trees, calibrated path-survival stopping, threshold adapts to load. Finds marginal trees fail on DSpark even at 8x budget. 20-34% of verified nodes committed. | 1-2 |
| DSpine / Adjacent Causal Injection (2609.36173) | Gated adjacent feature injection between positions at every backbone layer; single pass. | 1 |
| ReTrace (2608.29748) | DFlash; feeds rejected-suffix hidden states into next round. Reports at rejected positions the target token is in draft top-5 75.2%, top-1 30.4%. | 1 (but useful stat) |
| Carryover Drafting (2609.14717) | Rejected target hidden states carried as KV context into next round; DFlash and DSpark-like drafters. | 0-1 |
| TAPS (2606.00487) | Learned scorer for DFlash prefix-tree selection, uses positional entropy as a feature; tree size shrinks on uncertain rounds. Fig 3(a): target token in top-8 marginals 99% of the time; Fig 3(b): budgeted prefix-closed trees still lose 10-15% of correct tokens at later positions because they sit under prefixes that diverge from the accepted path. | 1, but closest existing D-style diagnostic |
| CaDDTree (2606.01813), BASTION (2605.29727) | Per-round confidence-dependent budget for DFlash trees; no re-conditioning. | 0-1 |
| JetSpec (2606.18394) | Causal parallel draft head producing branch-prefix-conditioned tree in one pass. | 1-2 |
| DDTree (2604.12989) | Marginal tree from one DFlash pass; explicitly no re-conditioning; no top-k vs budget decomposition. | baseline |
| DEdit (2609.38510) | Iterative draft editing. **Not read** (fetch rate-limited); check manually, title suggests possible overlap with refine/redraft. | unknown |
| AR-drafter adaptive trees: EAGLE-2 (2406.16858), OPT-Tree, TALON (2601.07353), SpecBlock (2605.07243) | Confidence-shaped trees (wide where uncertain, deep where confident) for AR drafters, which re-condition natively. | 1: prior art for "branch at uncertain tokens" in general |
| Entropy-Tree (2601.15296), Entropy-informed Decoding (2605.09745) | Entropy-guided branching for sampling/reasoning, not speculative drafting. | 0-1, cite for the branch-point framing |

## Top-k miss vs tree miss analyses

No block-drafter paper found that gives a clean per-position decomposition of rejections into "target not in top-k" vs "in top-k but missed by the tree", and none conditions that on the entropy of the preceding position. Partial versions:
- DFlash 2 blog: top-16 coverage 99.5%, recall@1 85.4% -> 72.9% by pos 6.
- TAPS Fig 3: top-8 coverage 99%, but budgeted trees lose 10-15% of correct tokens at later positions due to wrong-prefix placement (the closest to D).
- GRAFT Fig 1(b): parent-child mismatch rate across tasks.
- ReTrace: top-5 75.2% / top-1 30.4% at rejected positions.
- TreeFlash: TVD by depth, top-k coverage by depth.
- DSpark: per-position conditional acceptance curves.
- DBLAST: gains vs block determinism (block-level, not position-level).

## Verdict

- Closest prior work: D2SD. It already does confidence-located suffix redrafting with a variable-prefix diffusion drafter conditioned on unverified prefix tokens (the lag-drafter idea), batched in one extra pass. Anyone reviewing the thesis will cite it.
- Second tier: DominoTree, PCTree, TreeFlash, DARTree (branch-conditioned tree marginals, but via light heads applied everywhere, not via a second drafter pass), and DBLAST (attacks the multimodality mixture directly with a latent variable).
- Already done: (a) one extra batched drafter pass for suffix redrafting; (b) placing it via drafter confidence; (c) conditioning on unverified tokens with a prefix-trained diffusion drafter; (d) branch-conditioned tree marginals in general; (e) observation that top-8/16 coverage is ~99% and losses are in path selection (TAPS, DFlash 2, GRAFT).
- Seemingly open:
  1. Forking over top-m alternatives AT the branch point and redrafting the suffix once per alternative (D2SD keeps the argmax prefix and only redrafts from the boundary; it branches over WHERE, not WHICH token). This is the key mechanical difference and should be argued against D2SD with a direct ablation (D2SD-style prefix redraft vs per-candidate redraft at equal pass count and verify budget).
  2. Building a tree (not K+1 chains) from branch-conditioned marginals produced by a full drafter pass, and comparing it to light-head conditioning (DominoTree/PCTree/TreeFlash) at equal compute.
  3. The H1 analysis itself: per-position acceptance loss conditioned on entropy of the preceding position (target and drafter), plus the top-k-miss vs tree/path-miss split. Nobody reports this. It is a cheap, publishable diagnostic and the natural first chapter.
  4. Entropy of the target (or a calibrated drafter proxy) as the trigger, vs D2SD's product-of-max-prob boundary posterior; also triggering on bimodality (top-2 mass with low top-1) rather than low confidence, which is closer to the multimodality hypothesis.
- Risk: if H1 analysis shows losses are not concentrated after high-entropy positions, the method reduces to a D2SD variant. Run the diagnostic first.

## Sources
- https://arxiv.org/html/2606.04446 (D2SD)
- https://arxiv.org/html/2608.02438 , https://github.com/Supercomputing-System-AI-Lab/xPress (xPress)
- https://arxiv.org/html/2608.05448 (DBLAST)
- https://arxiv.org/pdf/2606.03819 (TreeFlash)
- https://arxiv.org/html/2608.13524 (DARTree)
- https://arxiv.org/html/2608.20375 (GRAFT)
- https://inco.ai/blog/dflash2/ (DFlash 2)
- https://arxiv.org/pdf/2605.29707 (Domino)
- https://arxiv.org/pdf/2607.05147 (DSpark)
- https://arxiv.org/html/2607.08642 (DominoTree)
- https://arxiv.org/html/2608.02123v1 (PCTree)
- https://arxiv.org/html/2609.22098 (TreeSpark)
- https://arxiv.org/abs/2609.36173 (DSpine)
- https://pith.science/paper/2608.29748 (ReTrace, review page)
- https://arxiv.org/html/2609.14717 (Carryover Drafting)
- https://arxiv.org/html/2606.00487 (TAPS)
- https://arxiv.org/html/2606.01813 (CaDDTree)
- https://arxiv.org/html/2605.29727v1 (BASTION)
- https://pith.science/paper/2606.18394 (JetSpec, review page)
- https://arxiv.org/pdf/2604.12989 (DDTree)
- https://arxiv.org/abs/2601.07353 (TALON)
- https://arxiv.org/abs/2609.38510 (DEdit, not read)
