# Early Exit / Layer Skipping x Speculative Decoding (2024 to Sep 2026): Map and Open Gaps

Verification legend: [F] = fetched and read this session (paper page or HTML). [T] = title/ID/date/venue confirmed only via the hemingkx/SpeculativeDecodingPapers README or a search listing. [R] = numbers recalled from the abstract and NOT re-fetched this session, so verify before citing.
Master list source: [hemingkx/SpeculativeDecodingPapers README](https://github.com/hemingkx/SpeculativeDecodingPapers) (fetched raw; its 2026 coverage is thin, so 2026 items come from direct searches).

## Q0. Dense map of methods (reference table for all questions)

### Takeaway
There are three families. (1) Self-drafting via a shallow sub-network: layer skipping or early exit. (2) Multi-level / hierarchical pipelines that add an intermediate verifier or drafter. (3) Methods that make the target cheaper at verification time. Nearly all self-drafting work reports 1.3 to 2.5x over autoregressive decoding. External trained drafters (EAGLE-3, DFlash) report 3 to 6x. Only a few 2026 papers with training (SEED, ECHO) claim parity with or wins over EAGLE-3.

### Cited Findings

**A. Self-drafting (layer skip / early exit as the draft model)**
- Draft & Verify / Self-SD (arXiv 2309.08168, ACL 2024) skips intermediate layers chosen by Bayesian optimization, is training-free, and is lossless. [T] Its reported "up to 1.99x" on LLaMA-2 is [R]. — [README](https://github.com/hemingkx/SpeculativeDecodingPapers), [arXiv](https://arxiv.org/pdf/2309.08168.pdf)
- LayerSkip (2404.16710, ACL 2024) uses layer-dropout plus an early-exit loss during training so that early layers can draft and later layers verify, reusing the shared KV cache. [T] "Up to 2.16x CNN/DM, 1.82x coding" is [R]. It needs pre-training or continual pre-training with the recipe. — [arXiv](https://arxiv.org/abs/2404.16710)
- Kangaroo (2404.18911, NeurIPS 2024) drafts with a shallow sub-network plus a trained adapter, and uses a second early exit to stop drafting when confidence is low. [T] "1.68x on Spec-Bench, 88.7% fewer extra params than Medusa" is [R]. — [arXiv](https://arxiv.org/pdf/2404.18911)
- EESD, "Speculative Decoding via Early-exiting ... with Thompson Sampling Control" (2406.03853, ACL 2024 Findings), adds an early-exit head with self-distillation and uses Thompson sampling to set draft length. [T] — [arXiv](https://arxiv.org/html/2406.03853v1)
- S3D (2405.20314) combines layer skipping with mask-predict and targets low-memory GPUs. [T] — [arXiv](https://arxiv.org/pdf/2405.20314)
- Draft on the Fly / ASD (2410.01028, EMNLP 2024 Findings) is training-free and picks skipped layers by cosine similarity of hidden states. [T] — [arXiv](https://arxiv.org/pdf/2410.01028)
- SWIFT (2410.06916, ICLR 2025) optimizes the skipped-layer set on the fly with no training. [T] "1.3 to 1.6x" is [R]. — [arXiv](https://arxiv.org/pdf/2410.06916)
- DEL (2504.05598, COLM 2025) selects the exit layer and speculation length dynamically from context during LayerSkip-style SD. [T] Its speedup numbers were not re-verified. — [arXiv](https://arxiv.org/abs/2504.05598)
- SpecEE (2504.08850, ISCA 2025) uses a lightweight predictor that takes speculative (draft) tokens as features to decide early exit. It is an early-exit engine (lossy vs. full-depth), not lossless SD. [T] "2.25x cloud / 2.43x PC on Llama2-7B" is [R]. — [ACM](https://dl.acm.org/doi/10.1145/3695053.3730996)
- KNN-SSD (2505.16162) picks a skip-layer set per input domain via nearest neighbour. [T] "1.3 to 1.6x" is [R]. — [arXiv](https://arxiv.org/pdf/2505.16162)
- CLaSp (2505.24196, ACL 2025) re-optimizes in-context layer skipping every step using the last verification's hidden states (dynamic programming). [T] "1.3 to 1.7x on LLaMA3" is [R]. — [arXiv](https://arxiv.org/pdf/2505.24196)
- CAS-Spec (NeurIPS 2025) is cascade adaptive self-speculation: it builds a hierarchy of self-drafts (layer sparsity or quantization) with dynamic tree cascade routing. [T] — [OpenReview PDF](https://openreview.net/pdf/7be7febdbc687ff1d863bbeaf1f37fb1b683f4bc.pdf)
- ConfLayers (2604.14612, Apr 2026) does entropy-confidence layer skipping with adaptive windowing. It was tested on LLaMA-2/3 (8B to 70B), CodeLLaMA-34B and Qwen2.5-Math-72B on an AMD MI300X. It reports 1.06 to 1.35x vs SWIFT, a max speedup of 1.4x, and "DEL often below 1x" in its setup. [F] — [arXiv](https://arxiv.org/html/2604.14612)
- SEED (2609.36590, 29 Sep 2026, accepted NeurIPS 2026) treats the decoder-only model as an implicit encoder-decoder. Verification and encoding are merged, and cached deep representations are reused for drafting. It requires fine-tuning. It claims "up to 2.7x average on 4B-scale models; 28% faster than EAGLE-3". [F, abstract only; hardware not stated] — [arXiv](https://arxiv.org/abs/2609.36590)

**B. Hierarchical / multi-level (intermediate verifiers, cascades, pipelines)**
- TriForce (2404.11912, COLM 2024) is a hierarchy for long context: a tiny model drafts for the target running on a retrieval-based sparse KV cache, which in turn is verified by the full-KV target. [T] "2.31x Llama2-7B-128K on A100" is [R]. — [arXiv](https://arxiv.org/pdf/2404.11912.pdf)
- QuantSpec (2502.10424, ICML 2025) self-drafts with the same architecture using a hierarchical 4-bit quantized KV cache and 4-bit weights. [T] ">90% acceptance, ~2.5x" is [R]. — [arXiv](https://arxiv.org/pdf/2502.10424)
- ML-SpecQD (2503.13565) is multi-level SD with MXFP4 quantized drafts. [T] — [arXiv](https://arxiv.org/pdf/2503.13565)
- "SD Meets Quantization" / HierSpec (2505.22179) finds that 4-bit weight quantization erodes SD gains and proposes a hierarchical small-model to quantized-target design. [T, finding R] — [arXiv](https://arxiv.org/pdf/2505.22179)
- PipeSpec (2505.01572, ACL 2025 Findings) runs k-level model hierarchies asynchronously across devices. [T] "up to 2.54x" is [R]. — [arXiv](https://arxiv.org/pdf/2505.01572)
- SpecRouter (2505.07680) routes adaptively over multi-level model chains. [T] — [arXiv](https://arxiv.org/pdf/2505.07680)
- Polybasic SD (2510.26527, ICML 2025) gives theory for more than two models in a chain. [T] Numbers not verified. — [arXiv](https://arxiv.org/pdf/2510.26527)
- READER, "Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding" (2509.19368), pipelines the early-exit draft and the full verification across stages. [T] The fetch was blocked, so details were not verified. — [arXiv](https://arxiv.org/pdf/2509.19368)
- Edge-cloud SD with early exits (2505.21594) [T] — [arXiv](https://arxiv.org/pdf/2505.21594)
- HiSpec (2510.01336) uses an early-exit (EE) model: drafts at about 1/8 depth, an intermediate verification at about 1/4 depth, and periodic full verification after accumulating 4 tentatively-accepted tokens (default). Hidden-state and KV reuse means no auxiliary model is needed. It ran on 4x H100 with Llama2/3 and CodeLlama (7B to 70B); batch size is not specified. Reported: average 1.70x vs vanilla, 1.28x vs single-layer SD, peak 2.08x (Llama3-8B CNN/DM). The comparison set is LayerSkip, AdaDecode and Lookahead, not EAGLE-3 or DFlash. "About one-fourth of the layers produce up to 69% of the response correctly." [F] — [arXiv HTML](https://arxiv.org/html/2510.01336v1)
- ECHO, "Early-layer Collaborative Hierarchical Orchestration with Bonus Logits" (2609.17241, EMNLP 2026 per the GitHub repo), has an inner loop where early layers 1..L_e do tree verification and an outer loop where layers L_e+1..L run full causal verification with state reuse. It reuses early-layer and final-layer logits as drafts, with hybrid n-gram automaton plus top-k trees. It claims losslessness via M(S_acc) = V_s(V_e(S_acc)). It needs a one-shot fine-tune. Reported: 2.4 to 2.9x (peak 3.01x) on Llama-2-7B; hardware not given. [F] The fetch tool reported the date as "Sep 15, 2024", which conflicts with arXiv ID 2609 (Sep 2026); treat as Sep 2026. — [arXiv](https://arxiv.org/html/2609.17241), [GitHub](https://github.com/whucs21Mzy/ECHO)
- A different ECHO, "Elastic Speculative Decoding with Sparse Gating for High-Concurrency Scenarios" (2604.09603), is unrelated to early exit. [T] — [listing](https://awesomepapers.io/llm-papers/papers/2604.09603)
- HSD, "Overcoming Joint Intractability with Lossless Hierarchical Speculative Decoding" (2601.05724, ICLR 2026, Qwen team), is a verification *acceptance rule*: hierarchical resampling distributions with capped branch resampling. It is not a multi-model hierarchy. It is provably lossless. Reported: +6.2% block efficiency and +6.7% tokens/s on average, and +12.4% speed when plugged into EAGLE-3, on H20 GPUs. It is orthogonal to early exit and composable. [F] — [arXiv](https://arxiv.org/html/2601.05724v2)

**C. Making verification cheaper (layer-skipping or sparse verifiers)**
- TSS (2609.26100) does domain-specific layer skipping in the *target during verification*. An acceptance- and metric-aware BFS keeps configurations within 5% tolerance on acceptance and task metric. It is **not lossless w.r.t. the dense model**: outputs change, e.g. translation BLEU 0.131 to 0.237 and accept length 2.70 to 4.53. Tested with Vicuna-7B+EAGLE (1.07 to 1.68x) and Llama-2-13B+SAMD (1.16 to 1.30x) on 8x RTX 4090. It needs offline domain calibration, and unseen domains fall back to dense. [F] — [arXiv](https://arxiv.org/html/2609.26100)
- SpecPV (2512.02337, Dec 2025) is self-SD for long context where partial verification means a *subset of KV* (sink, retrieval and local tokens), not a subset of layers, plus periodic full verification. It is lossy ("minor degradation"). It reports up to 6x vs AR and about 2x vs full verification at 60K context, on LLaMA-3.1-8B and Qwen3-4/8/14B, A100-80GB and RTX 4090. [F] — [arXiv](https://arxiv.org/html/2512.02337v1)
- "Accelerate Speculative Decoding with Sparse Computation in Verification" (2512.21911) [T, title only; not read] — [arXiv](https://arxiv.org/abs/2512.21911)
- SPRINTER (2502.04557) does approximate verification with a trained auxiliary verifier and is lossy. [T] — [arXiv](https://arxiv.org/pdf/2502.04557)
- "Margins, Not Windows: Training-Free Per-Step Lossy SD" (2609.02897) [T] — [arXiv](https://arxiv.org/html/2609.02897)

**D. Budget control relevant to external drafters**
- CaDDTree (2606.01813, Jun 2026) is a cost-aware DDTree for diffusion (DFlash-style) drafters. It picks tree structure and node budget per round from the draft's per-position distributions plus a one-time cost profile, and shows throughput is unimodal in tree size under convex verification cost. On Qwen3-4B/8B with A100/A800 it matches DDTree-oracle (MATH-500 Qwen3-4B: 4.53 ms/tok for both; tau 10.37 vs 10.35). It uses **no target-internal or early-exit signal**. [F] — [arXiv](https://arxiv.org/html/2606.01813)
- DSpark (2607.05147) does confidence-scheduled SD with semi-AR generation, using drafter-side confidence. [T] — [arXiv](https://arxiv.org/html/2607.05147v1)
- Confidence-Modulated SD (2508.15371) [T] — [arXiv](https://arxiv.org/pdf/2508.15371)
- SVIP (2411.18462) is a draft-length policy from the draft's own entropy. [T] — [arXiv](https://arxiv.org/pdf/2411.18462)
- DFlash (2602.06036) is a block-diffusion drafter. [T] Its ">6x lossless, ~2.5x over EAGLE-3" is [R]. — [arXiv](https://arxiv.org/pdf/2602.06036)

### Inferences
- Across all self-drafting papers found, the drafter costs a large fraction of target depth (1/8 at best in HiSpec, typically 40 to 60% of layers skipped in Self-SD/SWIFT). EAGLE-3 and DFlash cost about 1 layer. That is the structural reason for the 2x vs 5x gap.
- None of the hierarchical EE papers (HiSpec, ECHO, READER) compares against EAGLE-3, DFlash or DDTree under the same hardware and batch. So "hierarchical verification beats a strong external drafter" is unestablished in the literature. This matches the user's 0.85 to 0.9x of DDTree result.

### Gaps
- READER (2509.19368), SpecRouter, Polybasic, CAS-Spec and DEL numbers were not fetched. Their speedups and hardware need a direct read.
- 2512.21911 (sparse computation in verification) is the closest possible competitor to layer-sparse verification and was not read.

## Q1. Where does self-speculation still lose to external drafters, and is there a regime where it wins?

### Takeaway
Self-speculation loses on batch-1 latency for short to medium context, because a sub-network of the target costs far more per draft token than a 1-layer EAGLE-3 or DFlash head, and its acceptance is not higher. It has published wins or plausible wins in (a) long context, where KV reads dominate and self-drafting with sparse or quantized KV shines, (b) memory-constrained and on-device settings where no extra drafter parameters or KV are wanted, and (c) with training-time co-design (SEED claims +28% over EAGLE-3 at 4B).

### Cited Findings
- Training-free self-drafting ceilings reported in 2025 to 2026 are about 1.3 to 1.7x. ConfLayers reports a max of 1.4x on MI300X [F]. SWIFT and CLaSp report roughly 1.3 to 1.7x [R]. — [ConfLayers](https://arxiv.org/html/2604.14612), [SWIFT](https://arxiv.org/pdf/2410.06916), [CLaSp](https://arxiv.org/pdf/2505.24196)
- HiSpec's best (EE-trained Llama3-8B) is 2.08x vs vanilla. It was compared only against other self-drafting baselines. — [HiSpec](https://arxiv.org/html/2510.01336v1)
- SEED, a trained self-speculative design, claims 28% faster than EAGLE-3 at 4B. This is the first claim found of self-speculation beating EAGLE-3. Model, hardware and batch are not in the abstract. — [SEED](https://arxiv.org/abs/2609.36590)
- ECHO reports 2.4 to 2.9x on Llama-2-7B with a one-shot fine-tune plus n-gram retrieval, i.e. it is not purely self-drafting. — [ECHO](https://arxiv.org/html/2609.17241)
- Long context: SpecPV (self-SD with partial KV) reaches up to 6x vs AR at 60K on A100 and scales with context beyond 20K [F]. TriForce and QuantSpec also target this regime [T]. — [SpecPV](https://arxiv.org/html/2512.02337v1), [TriForce](https://arxiv.org/pdf/2404.11912.pdf), [QuantSpec](https://arxiv.org/pdf/2502.10424)
- On-device / PC: SpecEE reports a PC-setting speedup (2.43x [R]). S3D targets low-memory GPUs [T]. — [SpecEE](https://dl.acm.org/doi/10.1145/3695053.3730996), [S3D](https://arxiv.org/pdf/2405.20314)
- Multi-LoRA: serving stacks are adding LoRA support for external drafters: a vLLM RFC for LoRA on DFlash drafters, and an SGLang PR for single-adapter LoRA with EAGLE/NEXTN. This shows per-adapter drafter mismatch is a live engineering problem. A vLLM PR adds "Uno shared-model drafting with a draft-only LoRA adapter", which is a self-drafting variant. — [vLLM #52038](https://github.com/vllm-project/vllm/issues/52038), [SGLang #28395](https://github.com/sgl-project/sglang/pull/28395), [vLLM #55947](https://github.com/vllm-project/vllm/pull/55947)

### Inferences
- At batch 1 on an A100 with an 8B model, verification is memory-bound (the user's measurement: flat cost up to ~128 tokens). The draft's cost per token matters more than any verifier savings, and a sub-network drafter pays a fraction of the full weight read per draft step. External 1-layer drafters win there almost by construction.
- Self-speculation's natural wins are where the external drafter's own cost or mismatch grows: very long context (drafter KV), many adapters (drafter misaligned with each LoRA), and tight memory (no room for drafter weights or KV). Large-batch compute-bound serving is a candidate, since skipped layers save FLOPs rather than bytes, but no paper was found that measures self-spec vs EAGLE-3 at batch 32 or more.

### Gaps
- No head-to-head benchmark was found of self-spec (SWIFT, CLaSp, HiSpec) vs EAGLE-3 or DFlash across batch sizes on the same hardware.
- No paper was found evaluating self-speculation specifically for multi-LoRA serving. Only engineering PRs exist, found by search, not exhaustive.

## Q2. Is there work on making models self-speculable by design (pre/post-training) beyond LayerSkip?

### Takeaway
Yes, and it is growing in 2026. Examples are SEED (NeurIPS 2026, implicit encoder-decoder fine-tuning), ECHO (one-shot fine-tune of early-layer logits), Kangaroo and EESD (trained exit adapters or heads), HiSpec (relies on EE-trained models), and the MTP line (in-model extra heads). No work was found that trains the exit specifically to be a good *verifier* (predicting final-layer acceptance of an external drafter's tokens) rather than a good token predictor.

### Cited Findings
- LayerSkip: layer dropout plus shared early-exit loss during pre-training or continual pre-training. — [arXiv](https://arxiv.org/abs/2404.16710)
- SEED: fine-tuning so that the early layers act as an encoder and the late layers as a decoder, with verification merged into encoding. — [arXiv](https://arxiv.org/abs/2609.36590)
- ECHO: requires a "one-shot fine-tuning" for best acceleration. — [arXiv](https://arxiv.org/html/2609.17241)
- HiSpec: "EE-models are explicitly trained so that hidden states at these exit layers can be interpreted". Post-training EE models have fewer exits, which limits configurations. — [arXiv](https://arxiv.org/html/2510.01336v1)
- Kangaroo (adapter on a shallow sub-network) and EESD (early-exit head with self-distillation). — [Kangaroo](https://arxiv.org/pdf/2404.18911), [EESD](https://arxiv.org/html/2406.03853v1)
- MTP-in-model: "Multi-Token Prediction via Self-Distillation" (2602.06019), FastMTP (2509.18362), and "Your LLM Knows the Future" (2507.11851). [T] — [README](https://github.com/hemingkx/SpeculativeDecodingPapers)

### Inferences
- The design space "post-train an exit so that it predicts the *target's accept/reject decision* on externally drafted tokens" sits between SpecEE's predictor (exit decision for the model's own output) and HiSpec (token-level exit). It appears open.

### Gaps
- Details of SEED's training cost, and whether it generalizes to 8B, were not available (abstract only).

## Q3. Using early-exit signals to control an external drafter's tree budget or draft length, or to skip verification layers safely

### Takeaway
Budget control for external drafters today uses *drafter-side* signals: CaDDTree (draft distributions plus cost model), DSpark, SVIP, and confidence-modulated SD. Layer-skipping verifiers exist (TSS, SpecEE-style exits) but are lossy. No paper was found that feeds the target's intermediate-layer states from the last verification pass back into the external drafter's tree budget or depth. No paper was found that skips verification layers while staying provably lossless.

### Cited Findings
- CaDDTree picks budget from the drafter's per-position distributions plus profiled verification cost, with no target-internal signal. — [arXiv](https://arxiv.org/html/2606.01813)
- TSS skips target layers at verification. It is lossy relative to the dense target and domain-calibrated. — [arXiv](https://arxiv.org/html/2609.26100)
- SpecEE uses draft tokens to decide target early exit (the opposite direction). It is lossy. — [ACM](https://dl.acm.org/doi/10.1145/3695053.3730996)
- CLaSp uses the last verification's hidden states to re-pick *which layers the self-draft skips*. This is the closest example of "reuse the previous verify pass's internal states to configure the next draft", but it is for self-drafts, not external trees. — [arXiv](https://arxiv.org/pdf/2505.24196)
- Kangaroo and EESD use an exit-head confidence to stop *self*-drafting. — [Kangaroo](https://arxiv.org/pdf/2404.18911), [EESD](https://arxiv.org/html/2406.03853v1)

### Inferences
- The previous full verification pass already computes every layer's hidden state at every tree node, so a per-layer "settledness" feature (e.g. the layer at which the target's argmax stabilized) comes for free. Whether it predicts next-round acceptance better than the drafter's own confidence is untested.

### Gaps
- The novelty checks were web-search based, not exhaustive over OpenReview.

## Q4. What is open about lossless early-exit verification (exact acceptance with partial-depth verification)?

### Takeaway
Every "lossless" hierarchical method found (HiSpec, ECHO, TriForce, PipeSpec) is lossless only because a full-depth pass eventually verifies every token. The intermediate level only filters. Methods that truly skip depth (TSS, SpecEE, SPRINTER, SpecPV on the KV axis) are lossy. No work was found that proves, per token, that the remaining layers cannot change the acceptance decision. Such a certificate would make partial-depth acceptance exact, for greedy decoding or with sampling-correct fallbacks.

### Cited Findings
- HiSpec keeps accuracy via periodic full verification (every 4 tentatively-accepted tokens by default). — [arXiv](https://arxiv.org/html/2510.01336v1)
- ECHO's losslessness identity composes the early verifier V_e with the subsequent verifier V_s, so full depth is always executed on accepted tokens. — [arXiv](https://arxiv.org/html/2609.17241)
- SpecPV is explicitly lossy and relies on periodic full verification to "rectify deviations". — [arXiv](https://arxiv.org/html/2512.02337v1)
- TSS changes outputs (BLEU shifts), so it is lossy. — [arXiv](https://arxiv.org/html/2609.26100)
- HSD shows acceptance-rule changes alone give about +6 to 12% while staying lossless, which suggests verification-rule headroom is small but real. — [arXiv](https://arxiv.org/html/2601.05724v2)

### Inferences
- Tie to the user's experiments: at batch 1 on an A100 with an 8B model, the full upper pass is mandatory and its cost is flat in token count up to ~128. So an intermediate verifier can only (a) shrink the tree, which is free there anyway, or (b) defer the full pass HiSpec-style, which risks rollback. That explains 0.85 to 0.9x of DDTree: the extra lower-layer pass is pure overhead. The async variant (0.53x) adds staleness on top.
- Hierarchical verification should only pay off where full-pass cost grows with verified tokens: large batch (compute-bound), large trees with DFlash (tau around 10), or bigger targets (e.g. 70B across 4 GPUs, where upper layers live on other devices).
- Two concrete lossless mechanisms are still unexplored:
  - **In-pass tree pruning.** Drop low-probability branches at layer L_e *inside the same forward*, so upper layers process fewer tree nodes. The output stays exact for the tokens kept, because acceptance still uses final logits; only tau can drop.
  - **Certified early accept.** For greedy decoding, if the layer-L_e margin exceeds a bound on how much the remaining layers can move the logits, the argmax is provably unchanged. Bounds may be very loose; this is a feasibility question.

### Gaps
- It is unknown whether any 2026 paper has certified-margin early exit for LLM verification. The searches ("lossless early exit verification", "skip verification layers lossless exact acceptance") returned no such work, but this is not exhaustive.

## Candidate gaps with novelty checks (testable on 1 to 4 A100s, ~8B models)

Each gap below lists the idea, the closest existing work found, a novelty verdict, and a test plan.

1. **Early-exit-guided tree budget for an external drafter (DFlash/DDTree).** Use per-layer settledness of the target from the previous verification pass as extra features in a CaDDTree-style budget and depth controller.
   - Closest work: CaDDTree 2606.01813 (drafter-only signals), DSpark 2607.05147, SVIP 2411.18462, CLaSp 2505.24196 (reuses verify hidden states, but for a self-draft layer set).
   - Novelty: no match found; likely novel. Medium confidence, search-based.
   - Test: Qwen3-8B plus the DFlash drafter on 1 A100, comparing tokens/s vs DDTree and CaDDTree.

2. **In-pass mid-depth tree pruning (lossless).** Prune tree nodes after layer L_e within one verification forward, then run upper layers on the survivors.
   - Closest work: ECHO 2609.17241 (separate inner and outer loops), HiSpec, 2512.21911 (unread, possibly overlapping), SpecVLM (prunes visual tokens, not tree nodes).
   - Novelty: possibly novel, but 2512.21911 must be read first.
   - Expected result from the user's data: no gain at batch 1 below 128 tokens, so the test must target batch ≥16 or trees ≥256 nodes.

3. **Hierarchical verification in the compute-bound regime.** Evaluate HiSpec/ECHO-style two-level verification with a strong external drafter at batch 16 to 64.
   - Closest work: HiSpec (batch unspecified), MineDraft 2603.18016, "Batch SD Done Right" 2510.22876.
   - Novelty: no paper was found reporting this crossover; this is an empirical gap and could be a negative-result or characterization paper.
   - This directly extends the user's finding.

4. **Verifier-aware exit training.** LoRA post-training of a layer-L_e head to predict the final layer's accept/reject on DFlash drafts (a binary or calibrated target), instead of next-token prediction.
   - Closest work: SpecEE predictor, Kangaroo adapter, HiSpec EE models, SelfJudge 2510.02329 (judge verification, lossy).
   - Novelty: likely novel as framed.

5. **Certified greedy-lossless early accept.** Use a margin at L_e versus an empirical or Lipschitz bound on the logit change over the remaining layers.
   - Closest work: 2609.02897 (lossy margins), SpecEE and TSS (lossy), HiSpec (deferred full verification).
   - Novelty: no match found. High risk that the bounds are too loose, so frame it as a feasibility study.

6. **Self-speculation for multi-LoRA serving.** Compare a shared-base self-draft against per-adapter external drafters across N adapters.
   - Closest work: vLLM PR #55947 "Uno" draft-only LoRA and the vLLM DFlash LoRA RFC #52038 (engineering, not papers).
   - Novelty: no academic paper found; likely open.

7. **DFlash drafter plus partial-KV intermediate verifier for long context.** Two-level verification: SpecPV-style partial-KV verification, then full-KV verification.
   - Closest work: SpecPV 2512.02337 (self-draft), TriForce, QuantSpec, LongSpec.
   - Novelty: the combination with a diffusion external drafter was not found.
   - This is the regime where hierarchy is most likely to beat DDTree, because upper-pass cost grows with context.
