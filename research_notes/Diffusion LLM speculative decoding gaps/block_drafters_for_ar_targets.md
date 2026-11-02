# Block-diffusion / dLLM / parallel drafters for AR targets (new papers, Jun-Nov 2026)

Scope note: only papers NOT on the user's known list. Verification level per paper is marked: [full text] = numbers read from arXiv HTML/PDF body; [abstract] = only abs page read, details missing; [aggregator] = only from Emergent Mind topic page. All numbers are as the paper states them. "Info to drafted tokens" uses the user's taxonomy: none / anchor features only / target computation over drafted tokens / verified (or rejected-verifier) states.

## Q1. Which new block-diffusion / dLLM drafters appeared Jun-Nov 2026, and what do they change vs DFlash?

### Takeaway
Roughly 15 new DFlash-derived papers not on the known list. They cluster into: (a) reuse of the previous round's verifier states over the rejected suffix (DFlow, Carryover; same family as the known ReAnchor), (b) joint-coherence correction of DFlash's per-position marginals without sequential passes (LiLiCorr, DCOCO, DominoTree), (c) training-objective changes for left-to-right acceptance (Teaching Diffusion L2R, DScout), (d) systems/serving (H-Spec, DScale, CaDDTree, BlockPilot, Packed-SS), (e) architecture ports (HyperDFlash for DeepSeek-V4, quantized-target DFlash). Typical gains over DFlash are +5 to +15% tau; none reports a step change.

### Cited Findings

**DFlow: Enabling Verifier Information Flow in Block Diffusion Speculative Decoding** (arXiv 2609.06498, 6 Sep 2026; Zhang, Zhang, Cui, Miao; no venue stated) [full text]
- Mechanism: hidden states from the selected target layers over the *rejected suffix* of the last verification are passed through DFlash's own projection W_c + RMSNorm, shifted by one so they refer to the same predicted token, then an MLP over (bonus token emb, first rejected proposal emb, h_i) produces FiLM-style scale/shift: x_i = E_mask + beta_i * LN(h_i) + gamma_i. No extra target pass. Info class: rejected verifier states (target computation over previously drafted tokens, reused next round). — [arXiv HTML](https://arxiv.org/html/2609.06498v1)
- Trained: yes, 5-layer drafter from scratch, block 16, multi-round rollout with k=2 reject-conditioned rounds, verifier states from a frozen target pass over actual draft proposals; 80K ShareGPT, 6 epochs. Chain, exact verification. — [arXiv HTML](https://arxiv.org/html/2609.06498v1)
- tau (avg 6 benchmarks, T=0): Qwen3-1.7B DFlash 3.56 -> DFlow 3.93; Qwen3-4B 3.71 -> 4.20; Qwen3-8B 3.66 -> 4.15 (EAGLE-3: 3.27/3.08/3.00). T=1: 2.99->3.26, 3.15->3.54, 3.10->3.45. Throughput ratio vs DFlash (derived from TPS by the extractor, SGLang, 8xA800): Qwen3-8B GSM8K 1.09x at conc. 1, 1.16x at conc. 32; HumanEval 1.06x/1.06x. Relay overhead ~0.51 ms/round (~1.2%). — [arXiv HTML](https://arxiv.org/html/2609.06498v1)
- Ablation (Qwen3-1.7B): DFlash 3.56; +self-conditioning 3.67; DFlow relay off 3.66; relay on 3.93. With Domino: Domino 4.02; Domino+DFlow without online rollout 2.22 (collapse); with online rollout 4.50. — [arXiv HTML](https://arxiv.org/html/2609.06498v1)
- Limitations: no dedicated limitations/future-work section. Evaluation only Qwen3 + ShareGPT. Conclusion: "verifier information beyond the rejection boundary can be effectively reused." — [arXiv HTML](https://arxiv.org/html/2609.06498v1)

**Carryover Drafting: Recycling Rejected States for Speculative Decoding** (arXiv 2609.14717, Sep 2026; Koo et al., SNU + NAVER Cloud) [full text]
- Mechanism: rejected target hidden states from the verification pass are inserted into the next draft as temporary read-only KV context through the drafter's existing projections plus one learned "rejected" embedding; replaced every round. Applied to DFlash and to a DSpark-derived semi-AR drafter ("Markov", rank-256 first-order correction head). Info class: rejected verifier states. — [arXiv HTML](https://arxiv.org/html/2609.14717v1)
- Ablation (Qwen3-4B Markov): carrying hidden states beat carrying draft tokens, target-emitted tokens, or target top-5 weighted hints. — [arXiv HTML](https://arxiv.org/html/2609.14717v1)
- Trained (3 extra epochs from a 10-epoch baseline, parallel draft-verify-draft training); 5 drafter layers, 15 tokens/round; chain; lossless by argument. — [arXiv HTML](https://arxiv.org/html/2609.14717v1)
- tau (T=0): Qwen3-4B DFlash 3.78 -> 4.14, Markov 4.33 -> 4.85; Gemma 4 12B DFlash 3.65 -> 4.03, Markov 4.13 -> 4.74. Speedup (vLLM, conc. 1, 1xH200): Qwen3-4B DFlash 3.34x -> 3.60x, Markov 3.44x -> 3.83x; Gemma 4 12B 2.58x -> 2.79x, 2.73x -> 3.13x. Gains persist but shrink up to concurrency 32. — [arXiv HTML](https://arxiv.org/html/2609.14717v1)
- Stated limitations/future work: confidence/adaptive scheduling not evaluated; tree verification + which rejected branch states to keep left open; only single-round retention trained (multi-round, on-policy rollouts future work); a dedicated projector was slightly better than one embedding; interface ablation done on a small probe (~40K ex.); benefit may grow with stronger drafters. — [arXiv HTML](https://arxiv.org/html/2609.14717v1)

**LiLiCorr: Lightweight Likelihood Correlation of Parallel Drafts** (arXiv 2608.20530, 25 Aug 2026; Rusanovsky ... Elad, NVIDIA) [project page]
- Mechanism: top-K lattice per position from DFlash; 2-layer Transformer scores lattice in one pass, out/in vectors, cosine links between neighbors, greedy left-to-right walk. Uses target's frozen embeddings of candidates, projected last-verified hidden state, and target scores in a hinge loss. Info class: anchor features only. Chain. B=16, K=8 in cost example. — [NVIDIA page](https://research.nvidia.com/labs/nemotron/lilicorr/)
- Results: tau 9-19% above vanilla DFlash; throughput over DFlash +4-13% (Qwen3-4B/8B, 1xH100, conc. 1 greedy); lead up to 7.2% across concurrency sweep; with hinge term leads all long-context points (ISL 1K-32K). Absolute speedups only in figures. — [NVIDIA page](https://research.nvidia.com/labs/nemotron/lilicorr/)
- Limitations noted: supervision only when ground truth is in top-K; path selection still sequential; gains narrow at higher concurrency; long-context robustness needed the hinge term. — [NVIDIA page](https://research.nvidia.com/labs/nemotron/lilicorr/)

**Contextual Correction of Parallel Drafts (DCOCO)** (arXiv 2610.12570, 8 Oct 2026; Eisenstadt ... Wolf ... Naiman) [abstract/alphaxiv summary]
- Mechanism: shortlist (top-16) per DFlash position, shared 2-layer MLP scores adjacent candidate pairs in parallel, adds learned correction to successor logit; greedy L2R walk. One DFlash pass only. Chain. — [alphaXiv](https://alphaxiv.org/abs/2610.12570)
- Results: Qwen3-8B, thinking off, greedy, 1xB200, conc. 1: 959.7 tok/s (3.62x vanilla) vs Domino 839.3 (3.16x); up to 6.6x over vanilla across configs; best tau in 11/12 verifier/temp/thinking configs; serving at conc. 4-32 beats strongest competitor by 6-23%; at 32 greedy streams 11,512.4 vs DFlash 9,655.0 tok/s. Evaluated with thinking on and off (4 verifier backbones, 10 benchmarks). — [alphaXiv](https://alphaxiv.org/abs/2610.12570)
- Stated caveats: one B200/vLLM setup, "not a claim of universal speedup"; greedy path selection nearly as good as geometric. — [alphaXiv](https://alphaxiv.org/abs/2610.12570)

**DominoTree: Conditional Tree-Structured Drafting with Domino** (arXiv 2607.08642, 9 Jul 2026, v4 16 Oct 2026; Lin, Jang; submitted to IEEE Access) [abstract]
- Mechanism: best-first tree scored by Domino's path-dependent causal correction (DDTree scores positions independently), top-M candidates; training-free on top of Domino. — [arXiv abs](https://arxiv.org/abs/2607.08642)
- Results: harness T=0 avg vs AR: Qwen3-4B 5.33x (peak 8.77x), Qwen3-8B 6.89x; SGLang 1xRTX 5090 batch 1: 4.11x / 4.83x / Qwen3.6-27B 4.58x (chain 4.54x); 20-36% faster than Domino in harness, 11-20% faster than best of DDTree/CaDDTree; HELMET 8K summarization up to 2.11x. Lead over chain reverses at concurrency >= 8. Hybrid Gated-DeltaNet target (Qwen3.6-27B) needs tree-aware recurrent verification. — [arXiv abs](https://arxiv.org/abs/2607.08642)

**Teaching Diffusion to Speculate Left-to-Right** (arXiv 2606.11552, 10 Jun 2026, v2 22 Jun; Whalen, Ito, Sakamoto) [abstract]
- Mechanism: training-only: positional weighting, first-error focal loss on the position that breaks the accepted prefix, differentiable chain loss surrogate for expected accepted length. 21-76% longer accepted length per benchmark vs position-uniform baseline across 6 reasoning/code/dialogue benchmarks and 4 targets (not named in abstract). Exactness preserved. Notes orthogonality to multi-draft self-selection. — [arXiv abs](https://arxiv.org/abs/2606.11552)

**DScout: Factorization-Guided Training and Inference for Diffusion-Drafted SD** (arXiv 2610.23935, 7 Oct 2026; Fan ... Zhihao Jia ...) [abstract]
- Mechanism: decomposes position-wise loss into inclusion/weighting/prediction -> mixed-horizon training, intra-block unmasking, acceptance-weighted loss; inference factorizes throughput = MAT / round latency to pick target-conditioning depth, drafter depth and block size per workload. 3.84x (Qwen3-4B), 4.14x (Qwen3-8B) avg over 6 benchmarks; +9.1%/+7.1% over unnamed SOTA baseline; config policy up to +9.75%. — [arXiv abs](https://arxiv.org/abs/2610.23935)

**H-Spec: Parallel SD Without a Drafter-Side KV Cache** (arXiv 2609.24197, 21 Sep 2026; Jiang ... Kurtic, Marques; Red Hat-affiliated authors) [abstract]
- Mechanism: hybrid Mamba-attention parallel drafter; attention modules reuse target KVs in place, Mamba initialized with projected last-token target hidden state ("hybrid target-context injection"); removes drafter-side KV cache that grows with concurrency. Info class: anchor features + target KV. +5.0-13.3% tau and +5.3-12.6% BS1 ITL speedup over best baseline across 3 targets; higher throughput, lower KV utilization under concurrency. Authors state that pure in-place target-KV reuse fails to keep draft quality across the block. — [arXiv abs](https://arxiv.org/abs/2609.24197)

**DScale: Scaling Block-Diffusion SD with Adaptive Verification** (arXiv 2609.37532, 29 Sep 2026; Chen, Xu et al.) [abstract]
- Systems: unchanged drafter; path-aware tiles, dynamic verify-length allocation into half the native verification capacity using a frozen 112K-param per-target predictor, fixed-address CUDA-graph workspaces. A100-40GB, Qwen3-8B/4B, conc. 8-32: geo-mean throughput +43.9%/+48.8% vs DFlash, +22.2%/+37.7% vs DSpark, +24.4%/+32.0% vs Domino. No tau reported in abstract. — [arXiv abs](https://arxiv.org/abs/2609.37532)

**BlockPilot: Instance-Adaptive Policy Learning for Diffusion-based SD** (arXiv 2606.31315, 30 Jun 2026; Zhang ... Chu) [abstract]
- Lightweight policy predicts per-input block size once from the prefilling representation; finds optimal sizes cluster around the training block size. Qwen3-4B T=1: tau 5.92, 4.20x. — [arXiv abs](https://arxiv.org/abs/2606.31315)

**CaDDTree: Cost-Aware Diffusion Draft Trees** (arXiv 2606.01813, 1 Jun 2026) [aggregator]
- Maximizes a throughput objective over tree node count with a greedy stopping rule instead of a fixed budget (extends DDTree). — [Emergent Mind](https://www.emergentmind.com/topics/block-diffusion-speculative-decoding)

**HyperDFlash** (arXiv 2606.26744, 25 Jun 2026) [abstract]
- DFlash port to DeepSeek-V4 Hyper-Connections: conditions on pre-collapse residual states, gated residual reducer initialized from target's hc_head (3 orders of magnitude fewer params than linear compressor), KL distillation on LM head. Beats native MTP and vanilla DFlash; no numbers in abstract. — [arXiv abs](https://arxiv.org/abs/2606.26744)

**One Call Ahead: Packed Self-Speculative Decoding with Diffusion Drafting (Packed-SS)** (arXiv 2610.33379, 24 Oct 2026; Chen, Cai) [abstract]
- Self-speculative: drafter runs one call ahead from an older verified prefix while current draft is verified in the same model call; input linear (not quadratic) in speculative width. Nemotron 3B/8B and SetDLM 1.7B, greedy only; "mild monotonicity assumption on the drafter". No numbers in abstract. — [arXiv abs](https://arxiv.org/abs/2610.33379)

**Quantize the Target, Quantize the Drafter (Qwen3.5-4B)** (arXiv 2607.04244, 5 Jul 2026; competition report) [abstract]
- Block-diffusion drafter (repo name suggests DFlash) trained in two stages: on the FP target, then adapted to the quantized target; drafter also quantized, with sliding-window attention for long context. 6.978x avg on A10G; 3rd in Efficient Qwen Competition. — [arXiv abs](https://arxiv.org/abs/2607.04244)

Earlier but not on the known list (essential context):
- **DEER: Draft with Diffusion, Verify with Autoregressive Models** (arXiv 2512.15176, Dec 2025) [full text]: separate 0.5B discrete dLLM drafter, one denoising step, two-stage training (AR-style distillation, then weighted suffix masking), block 4-32, chain, lossless proof. Qwen3-30B-A3B T=0: HumanEval tau 6.58 / 5.54x vs EAGLE-3 3.21 / 2.41x; code mean tau 5.03 / 4.04x; math mean tau 2.47 / 2.12x. — [arXiv PDF](https://www.arxiv.org/pdf/2512.15176)
- **SpecDiff-2** (MLSys 2026) [full text]: see Q4.
- **DART: Diffusion-Inspired Speculative Decoding** (arXiv 2601.19278, Jan 2026): found in search, not read. — [arXiv HTML](https://arxiv.org/html/2601.19278v1)

### Inferences
- The dominant 2026 H2 direction for giving drafted positions target-side information is recycling the *previous round's* verifier states over the rejected suffix (DFlow, Carryover, and the user-known ReAnchor). This is target computation over previously drafted tokens, obtained for free; none of these spends extra target (or cheap-copy) computation on the *current* draft before verification.
- Gains of all coherence-correction methods (LiLiCorr, DCOCO, DominoTree) are in the 5-20% range, consistent with the user's finding that drafters reading the same features fail at the same positions: reranking within the same marginals has bounded headroom (LiLiCorr explicitly limited by top-K coverage).

### Gaps
- Full texts not read for DCOCO, DominoTree, Teaching-L2R, DScout, H-Spec, DScale, BlockPilot, HyperDFlash, Packed-SS, CaDDTree; their limitations sections and absolute tau values are unverified.
- OpenReview ICLR 2027 submissions were not searched directly; some of the above may be ICLR 2027 submissions.

## Q2. Does any paper give drafted positions target-side information (target computation over drafted tokens, cheap/quantized copy, intermediate verifiers) for a block drafter?

### Takeaway
Yes, but only via *reuse of the last verification's rejected-suffix states* (DFlow, Carryover). No paper found uses a cheap or quantized copy of the target to compute features over the current draft for a block drafter, and no paper found inserts an intermediate verifier between a block drafter and the target. The quantized-target paper quantizes the verifier itself, not an auxiliary copy.

### Cited Findings
- DFlow feeds rejected-suffix verifier hidden states into mask embeddings (FiLM); relay alone adds +0.27 tau over self-conditioning on Qwen3-1.7B (3.66 -> 3.93). — [arXiv HTML](https://arxiv.org/html/2609.06498v1)
- Carryover inserts rejected states as read-only KV; hidden states beat tokens and target top-5 hints as the carried signal. — [arXiv HTML](https://arxiv.org/html/2609.14717v1)
- Carryover authors: "stronger drafters appear to leave more useful rejected states"; multi-round retention and tree integration left open. — [arXiv HTML](https://arxiv.org/html/2609.14717v1)
- H-Spec reuses target KVs in place for the drafter's attention but reports in-place KV reuse alone does not keep draft quality across the block; adds last-token hidden state to Mamba. — [arXiv abs](https://arxiv.org/abs/2609.24197)
- HyperDFlash conditions on richer target states (pre-collapse residuals) at anchors only. — [arXiv abs](https://arxiv.org/abs/2606.26744)
- Quantized-target competition paper: drafter trained first on FP target, then adapted to the quantized target (no copy-as-feature-source). — [arXiv abs](https://arxiv.org/abs/2607.04244)
- Pre-2026 hierarchical/quantized-draft work exists for AR drafters (HiSpec 2510.01336, ML-SpecQD 2503.13565, 2505.22179) but not with block-diffusion drafters. — [ML-SpecQD](https://arxiv.org/html/2503.13565v1); [2505.22179](https://arxiv.org/html/2505.22179v1)

### Inferences
- The user's measured result (redraft with a 4-bit copy's features over the drafted block: ~14 accepted vs ~10.4 tokens-only) is a different and stronger information source than DFlow/Carryover, which only see target states for tokens already rejected in the previous round, i.e. states at most one fork deep and stale by one round. DFlow/Carryover gains (+10-15% tau) are a plausible lower bound for what fresh target-like states over the current draft could give. Not tested in any found paper.

### Gaps
- No paper found evaluating a quantized/pruned/early-exit copy of the target as a feature provider for drafted positions of a block drafter; absence not proven (OpenReview not checked).

## Q3. Does any paper study error correlation between drafters, drafter diversity/decorrelation, or ensembles with different information sources?

### Takeaway
No paper found measures error correlation between drafters or designs for decorrelation. Closest: SpecDiff-2 lists "K specialized drafters and selecting among them at test time" as future work; Carryover's ablation compares information sources for the carried signal (not as separate drafters); a Findings-ACL 2026 multi-drafter paper exists but was not read.

### Cited Findings
- SpecDiff-2 future work: using K specialized drafters and selecting among them at test time; test-time self-selection samples K drafts from one drafter's marginals and lets the verifier rank them. — [MLSys 2026 PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/041dad5ed2191b44ba3ed0e00cdc3187-Paper-Conference.pdf)
- Carryover: hidden states > draft tokens > target tokens / top-5 hints as carried context (Qwen3-4B Markov). — [arXiv HTML](https://arxiv.org/html/2609.14717v1)
- "Multi-Drafter Speculative Decoding with Alignment Feedback" (Findings of ACL 2026) appeared in search; content not verified. — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1629.pdf)
- D2SD (known to user) is the only dual-diffusion-drafter design found; DominoTree/DCOCO/LiLiCorr all derive alternatives from the same DFlash pass. — [DominoTree](https://arxiv.org/abs/2607.08642); [LiLiCorr](https://research.nvidia.com/labs/nemotron/lilicorr/)

### Inferences
- Every alternative-generation method found (trees, lattices, self-selection, DCOCO) draws alternatives from a single drafter conditioned on the same target features. The user's result (alternatives rescue 8-25% of forks; PARD right at ~45%) is not contradicted or replicated anywhere found; this looks like an open gap.

### Gaps
- Findings-ACL multi-drafter paper not read; whether it measures error correlation unknown.

## Q4. Does any paper use a separate full dLLM (LLaDA, Dream, SDAR, Mercury etc.) as drafter or as repairer/editor for an AR target?

### Takeaway
Yes as drafter: SpecDiff-2 (DiffuCoder-7B / DiffuLLaMA-7B for Qwen2.5-72B / LLaMA-2-70B) and DEER (0.5B dLLM for Qwen3). Both are single-denoising-step drafters. No paper found uses a dLLM as a repairer/editor of an AR or DFlash draft for an AR target; the "draft-then-refine" papers found target dLLMs, not AR models.

### Cited Findings
- SpecDiff-2: LoRA fine-tuned DiffuCoder-7B and DiffuLLaMA-7B, gamma 32/16, T=1 denoising step; streak-distillation + self-selection over K sampled drafts; lossless; verifier probabilities only, enabling cross-tokenizer. Qwen2.5-72B T=0: 4.29x, 5.98 tokens/draft; T=1 4.39x, 6.14. LLaMA-2-70B T=0 3.93x, 5.49; T=1 4.27x, 5.98. Math500 Qwen2.5-72B 4.62x vs SpecDiff 3.19x. Batch 1 implied; batched only qualitative. — [MLSys 2026 PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/041dad5ed2191b44ba3ed0e00cdc3187-Paper-Conference.pdf)
- SpecDiff-2 stated limitations/future work: formal analysis of greedy-acceptance proxy; cross-family/tokenizer study; K specialized drafters; optimal drafter size and scaling laws (smaller verifiers get lower speedups because 7B drafters are oversized); kernels / KV caching for diffusion drafts; only one diffusion step tested; EAGLE-3 not reproducible in their environment. — [MLSys 2026 PDF](https://proceedings.mlsys.org/paper_files/paper/2026/file/041dad5ed2191b44ba3ed0e00cdc3187-Paper-Conference.pdf)
- DEER (0.5B dLLM, Qwen3-30B-A3B): HumanEval tau 6.58 / 5.54x; math tau 2.47 / 2.12x. Limitations: no mature dLLM KV cache, batched DEER not integrated into vLLM/SGLang, KV-cache usage described inconsistently between sections, math drafter partially converged, Stage II unstable (alpha = 1.05 diverged). — [arXiv PDF](https://www.arxiv.org/pdf/2512.15176)
- Draft-then-refine / correction papers found are for dLLM targets: Speculative Correction (2608.02625), SPOT (2610.22543), Spiffy (2509.18085), SimSD (2606.02544), S2D2 (2603.25702). — [2608.02625](https://arxiv.org/abs/2608.02625); [SPOT](https://arxiv.org/html/2610.22543); [Emergent Mind](https://www.emergentmind.com/topics/block-diffusion-speculative-decoding)

### Inferences
- Full dLLM drafters show a large code/math gap (DEER code tau 5.03 vs math 2.47), suggesting text-only dLLM drafters are domain-sensitive; but they are independent of target features, which by the user's PARD result is the property that decorrelates errors. Using a full dLLM as a fork repairer for DFlash is untested in the found literature.

### Gaps
- No results found for LLaDA, Dream, SDAR or Mercury as drafters for AR targets in 2026.

## Q5. Which papers address reasoning/thinking-mode traces, long context, or adaptive block length?

### Takeaway
Adaptive block length: BlockPilot (per-input, at prefill), DScout (per-workload config), DScale (adaptive verify length). Long context: LiLiCorr (ISL 1K-32K, needed hinge loss), DominoTree (HELMET 8K), H-Spec (concurrency KV memory), quantized-target paper (sliding-window drafter). Thinking mode: DCOCO explicitly evaluates thinking on/off. No found paper is dedicated to reasoning traces.

### Cited Findings
- BlockPilot: per-input block size from prefilling representation; optima cluster around training block size; Qwen3-4B T=1 tau 5.92, 4.20x. — [arXiv abs](https://arxiv.org/abs/2606.31315)
- DScout: per-workload choice of block size, drafter depth and target-conditioning depth; up to +9.75%. — [arXiv abs](https://arxiv.org/abs/2610.23935)
- DScale: dynamic verify-length into half native capacity; +43.9%/+48.8% throughput vs DFlash at conc. 8-32. — [arXiv abs](https://arxiv.org/abs/2609.37532)
- LiLiCorr leads all 90 long-context points (ISL 1K-32K) only with the hinge term. — [NVIDIA page](https://research.nvidia.com/labs/nemotron/lilicorr/)
- DominoTree up to 2.11x on HELMET 8K summarization (vs 4-7x short context). — [arXiv abs](https://arxiv.org/abs/2607.08642)
- DCOCO evaluates thinking on/off; tau lead in 11/12 configurations. — [alphaXiv](https://alphaxiv.org/abs/2610.12570)
- Teaching-L2R reports 21-76% accepted-length gains including reasoning benchmarks (training-side only). — [arXiv abs](https://arxiv.org/abs/2606.11552)

### Gaps
- Per-paper thinking-mode tau numbers not extracted (DCOCO full tables not read). No found study of how per-token hazard evolves along long reasoning traces.

## Q6. What limitations and future-work items recur?

### Takeaway
Recurring self-reported items: (1) gains shrink or reverse as concurrency grows (DFlow, Carryover, LiLiCorr, DominoTree); (2) tree verification and adaptive scheduling not combined with the new conditioning (Carryover); (3) multi-round / on-policy training needed to avoid train-inference mismatch (Carryover future work; DFlow-on-Domino collapse to 2.22 without online rollout); (4) evaluation limited to Qwen3 family and single hardware setups (DFlow, DCOCO); (5) top-K coverage caps reranking (LiLiCorr); (6) separate dLLM drafters lack KV-cache/batched serving support and scaling laws (SpecDiff-2, DEER).

### Cited Findings
- Concurrency: DominoTree lead reverses at >= 8 requests; LiLiCorr gains narrow at higher concurrency; Carryover gains shrink up to 32; DFlow HumanEval 1.06x at 32. — [DominoTree](https://arxiv.org/abs/2607.08642); [LiLiCorr](https://research.nvidia.com/labs/nemotron/lilicorr/); [Carryover](https://arxiv.org/html/2609.14717v1); [DFlow](https://arxiv.org/html/2609.06498v1)
- On-policy/multi-round training: Carryover future work; DFlow Domino combination needs online rollout. — [Carryover](https://arxiv.org/html/2609.14717v1); [DFlow](https://arxiv.org/html/2609.06498v1)
- Scope: DCOCO "not a claim of universal speedup" (one B200/vLLM setup). — [alphaXiv](https://alphaxiv.org/abs/2610.12570)
- Top-K cap: LiLiCorr. — [NVIDIA page](https://research.nvidia.com/labs/nemotron/lilicorr/)
- dLLM drafter infra: SpecDiff-2, DEER. — [SpecDiff-2](https://proceedings.mlsys.org/paper_files/paper/2026/file/041dad5ed2191b44ba3ed0e00cdc3187-Paper-Conference.pdf); [DEER](https://www.arxiv.org/pdf/2512.15176)

### Inferences
- Open gaps relative to the user's measurements: (a) fresh target-like computation over the *current* draft (cheap copy / intermediate verifier) for block drafters; (b) drafter ensembles chosen for error decorrelation (different information sources, e.g. feature-conditioned + text-only); (c) dLLM as fork repairer for AR targets; (d) position-wise hazard analysis (none of DFlow, Carryover analyze where errors occur within blocks beyond loss weighting).

### Gaps
- Limitations sections of abstract-only papers unknown; Google Scholar "cited by" and OpenReview not checked within budget.
